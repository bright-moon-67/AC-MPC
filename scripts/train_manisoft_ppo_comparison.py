#!/usr/bin/env python
"""Train ManiSoft lifted-MLP PPO or PPO-KMPC without behavior cloning.

The two routes share the same history observation, environment, reward,
rollouts, PPO implementation, evaluation metrics, and frozen history-Koopman
lift. PPO-MLP maps the lift plus waypoint context directly to actions and
values with MLP heads. PPO-KMPC instead uses a randomly initialized cost-map
actor and differentiable finite-horizon Koopman MPC layer. Neither uses BC.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_tracking_env import (
    MANISOFT_WAYPOINT_SUCCESS_STREAK,
    MANISOFT_WAYPOINT_SUCCESS_THRESHOLD,
    ManiSoftThreeWaypointTrackingEnv,
    load_manisoft_waypoint_reference_bank,
)
from antmaze_ac.envs.process_vector_env import ProcessVectorEnv
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.manisoft_ppo_policies import (
    PPO_ACTOR_NAMES,
    make_manisoft_ppo_policy,
)
from antmaze_ac.rl.ppo import collect_rollout, collect_vector_rollout, ppo_update


TIP_INDICES = (30, 31, 32)
METHOD = "manisoft_ppo_from_scratch"
FORMAT_VERSION = 1
TRAINING_SPEC_VERSION = (
    "manisoft_three_waypoint_reward_5mm_normalized_delta_kmpc_v5"
)


def _device(specification: str) -> torch.device:
    return torch.device(
        "cuda"
        if specification == "auto" and torch.cuda.is_available()
        else ("cpu" if specification == "auto" else specification)
    )


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _module_sha256(module: torch.nn.Module) -> str:
    """Hash initialized module tensors for paired-ablation provenance."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _make_env(
    *,
    scenario: Path,
    waypoint_tips: np.ndarray,
    episode_steps: int,
    absolute_action_limit: float,
    progress_reward_scale: float,
    history_steps: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    max_delta: float | None,
) -> HistoryContextTrackingWrapper:
    base = ManiSoftThreeWaypointTrackingEnv(
        scenario,
        waypoint_tips=waypoint_tips,
        episode_steps=episode_steps,
        absolute_action_limit=absolute_action_limit,
        progress_reward_scale=progress_reward_scale,
    )
    return HistoryContextTrackingWrapper(
        base,
        history_steps=history_steps,
        state_mean=state_mean,
        state_std=state_std,
        tip_indices=TIP_INDICES,
        max_delta=max_delta,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", choices=PPO_ACTOR_NAMES, required=True)
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--waypoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.001,
        help=(
            "PPO-KMPC per-component physical action-rate limit. The policy "
            "and PPO distribution operate on normalized increments in [-1,1]."
        ),
    )
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)

    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--kmpc-hidden-dims", type=int, nargs="+", default=[128])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--solver-diagnostic-iterations", type=int, default=320)
    parser.add_argument("--quadratic-log-scale", type=float, default=1.5)
    parser.add_argument("--linear-scale", type=float, default=10.0)
    parser.add_argument("--action-quadratic-scale", type=float, default=1.0)
    parser.add_argument("--tip-weight", type=float, default=1.0)
    parser.add_argument(
        "--kmpc-cost-parameterization",
        choices=("full", "structured"),
        default="full",
        help=(
            "PPO-KMPC cost map: 'full' learns every horizon cost term; "
            "'structured' learns only five bounded positive reference-cost "
            "multipliers."
        ),
    )
    parser.add_argument(
        "--structured-log-scale",
        type=float,
        default=math.log(2.0),
        help=(
            "Log-range of each structured cost multiplier. The default "
            "constrains individual multipliers to [0.5, 2.0]."
        ),
    )
    parser.add_argument(
        "--kmpc-reference-mode",
        choices=("explicit", "implicit"),
        default="explicit",
        help=(
            "Reference source for structured PPO-KMPC. 'explicit' is the "
            "v15e -Q*x_ref/-R*u_ref cost; 'implicit' keeps the same "
            "structured q and learns upstream-style free stage linear terms."
        ),
    )
    parser.add_argument(
        "--kmpc-decision-space",
        choices=("normalized_delta", "absolute"),
        default="normalized_delta",
        help=(
            "PPO-KMPC decision/action semantics. 'absolute' reproduces the "
            "upstream direct-U box-only formulation and removes the rate "
            "limit while preserving matched physical exploration noise."
        ),
    )
    parser.add_argument(
        "--structured-terminal-multiplier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the learned final-tip multiplier. Disabling it keeps the "
            "fifth output head but fixes its multiplier to one."
        ),
    )
    parser.add_argument(
        "--normalized-delta-curvature",
        type=float,
        default=0.0,
        help=(
            "Fixed rho*I curvature added directly to the normalized-delta "
            "QP Hessian; zero preserves the original objective."
        ),
    )

    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--parallel-env-processes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run each vector environment in a spawned CPU process while "
            "keeping policy inference GPU-batched."
        ),
    )
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--std-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--freeze-log-std",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze exploration std while validating actor-mean learning.",
    )
    parser.add_argument(
        "--anneal-learning-rate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--clip-value-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=1e-4)
    parser.add_argument("--initial-action-std", type=float, default=0.10)
    parser.add_argument("--minimum-action-std", type=float, default=0.001)
    parser.add_argument("--maximum-action-std", type=float, default=0.20)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--kl-soft-stop-multiplier", type=float, default=1.5)
    parser.add_argument("--kl-hard-rollback-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--normalize-advantages-globally",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-interval-updates", type=int, default=10)
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    counts = (
        args.episode_steps,
        args.horizon,
        args.solver_iterations,
        args.solver_diagnostic_iterations,
        args.total_timesteps,
        args.rollout_steps,
        args.num_envs,
        args.minibatch_size,
        args.update_epochs,
        args.checkpoint_interval_updates,
        *args.mlp_hidden_dims,
        *args.kmpc_hidden_dims,
    )
    if min(counts) < 1:
        raise ValueError("All dimensions and PPO counts must be positive")
    if args.rollout_steps % args.num_envs:
        raise ValueError("--rollout-steps must be divisible by --num-envs")
    if args.total_timesteps % args.num_envs:
        raise ValueError("--total-timesteps must be divisible by --num-envs")
    if args.parallel_env_processes and args.num_envs < 2:
        raise ValueError("--parallel-env-processes requires --num-envs >= 2")
    if not 2 <= args.minibatch_size <= args.rollout_steps:
        raise ValueError("--minibatch-size must lie in [2, rollout-steps]")
    positive = (
        args.absolute_action_limit,
        args.progress_reward_scale,
        args.quadratic_log_scale,
        args.linear_scale,
        args.action_quadratic_scale,
        args.tip_weight,
        args.structured_log_scale,
        args.learning_rate,
        args.std_learning_rate,
        args.gamma,
        args.gae_lambda,
        args.clip_range,
        args.value_coefficient,
        args.initial_action_std,
        args.minimum_action_std,
        args.maximum_action_std,
        args.max_grad_norm,
        args.target_kl,
        args.kl_soft_stop_multiplier,
        args.kl_hard_rollback_multiplier,
    )
    if min(positive) <= 0:
        raise ValueError("PPO scales, rates, and standard deviations must be positive")
    if not (
        args.minimum_action_std
        <= args.initial_action_std
        <= args.maximum_action_std
    ):
        raise ValueError("Action standard-deviation bounds are inconsistent")
    if args.entropy_coefficient < 0:
        raise ValueError("--entropy-coefficient must be non-negative")
    if args.normalized_delta_curvature < 0:
        raise ValueError("--normalized-delta-curvature must be non-negative")
    if args.actor != "ppo_kmpc" and args.normalized_delta_curvature:
        raise ValueError(
            "--normalized-delta-curvature is only valid for PPO-KMPC"
        )
    if args.actor != "ppo_kmpc" and args.kmpc_cost_parameterization != "full":
        raise ValueError(
            "--kmpc-cost-parameterization is only valid for PPO-KMPC"
        )
    if args.actor != "ppo_kmpc" and args.kmpc_reference_mode != "explicit":
        raise ValueError("--kmpc-reference-mode is only valid for PPO-KMPC")
    if (
        args.actor != "ppo_kmpc"
        and args.kmpc_decision_space != "normalized_delta"
    ):
        raise ValueError("--kmpc-decision-space is only valid for PPO-KMPC")
    if args.actor != "ppo_kmpc" and not args.structured_terminal_multiplier:
        raise ValueError(
            "--no-structured-terminal-multiplier is only valid for PPO-KMPC"
        )
    if (
        args.kmpc_reference_mode == "implicit"
        and args.kmpc_cost_parameterization != "structured"
    ):
        raise ValueError(
            "The implicit-reference ablation keeps structured q; use "
            "--kmpc-cost-parameterization structured"
        )
    if (
        not args.structured_terminal_multiplier
        and args.kmpc_cost_parameterization != "structured"
    ):
        raise ValueError(
            "The terminal-multiplier ablation requires structured cost"
        )
    if args.actor_learning_rate is not None and args.actor_learning_rate <= 0:
        raise ValueError("--actor-learning-rate must be positive")
    if args.kl_hard_rollback_multiplier < args.kl_soft_stop_multiplier:
        raise ValueError(
            "--kl-hard-rollback-multiplier must be >= "
            "--kl-soft-stop-multiplier"
        )
    if args.max_wall_time_hours is not None and args.max_wall_time_hours <= 0:
        raise ValueError("--max-wall-time-hours must be positive")
    if args.actor == "ppo_kmpc" and args.max_delta <= 0:
        raise ValueError(
            "PPO-KMPC requires --max-delta > 0 as the v15e rate/noise scale"
        )
    if (
        args.kmpc_decision_space == "absolute"
        and args.normalized_delta_curvature
    ):
        raise ValueError(
            "--normalized-delta-curvature is invalid in absolute decision space"
        )
    if (
        args.actor == "ppo_kmpc"
        and args.kmpc_decision_space == "normalized_delta"
        and args.solver_iterations < 80
    ):
        raise ValueError(
            "Normalized-delta PPO-KMPC requires --solver-iterations >= 80"
        )
    if (
        args.actor == "ppo_kmpc"
        and args.solver_diagnostic_iterations < args.solver_iterations
    ):
        raise ValueError(
            "--solver-diagnostic-iterations must be >= --solver-iterations"
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)

    koopman_path = Path(args.koopman_checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    waypoint_root = Path(args.waypoint_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for name, path in (
        ("Koopman checkpoint", koopman_path),
        ("scenario", scenario),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    waypoint_bank = load_manisoft_waypoint_reference_bank(waypoint_root)
    if waypoint_bank.scenario_sha256 != sha256(scenario):
        raise ValueError("Waypoint bank was certified with another scenario")
    waypoint_tips = waypoint_bank.states[:, :, np.asarray(TIP_INDICES)]

    if args.actor_learning_rate is not None:
        actor_learning_rate = float(args.actor_learning_rate)
    elif args.actor == "ppo_mlp":
        actor_learning_rate = 3e-4
    elif args.kmpc_cost_parameterization == "structured":
        # A real-environment one-update smoke test kept KL at 8.2e-4 with
        # this rate.  The legacy full map remains far more sensitive.
        actor_learning_rate = 1e-5
    else:
        actor_learning_rate = 3e-8

    normalized_delta_policy = (
        args.actor == "ppo_kmpc"
        and args.kmpc_decision_space == "normalized_delta"
    )
    actor_max_delta = args.max_delta if normalized_delta_policy else None
    # v15e's 0.18 is dimensionless D-space noise.  For the direct-U
    # ablation, convert it to physical action units so Q3 changes only the
    # decision/constraint formulation instead of increasing exploration by
    # 1/max_delta (=80 for v15e).
    action_std_unit_scale = (
        args.max_delta
        if args.actor == "ppo_kmpc" and not normalized_delta_policy
        else 1.0
    )
    effective_initial_action_std = (
        args.initial_action_std * action_std_unit_scale
    )
    effective_minimum_action_std = (
        args.minimum_action_std * action_std_unit_scale
    )
    effective_maximum_action_std = (
        args.maximum_action_std * action_std_unit_scale
    )
    policy, koopman_payload = make_manisoft_ppo_policy(
        args.actor,
        koopman_path,
        device,
        absolute_action_limit=args.absolute_action_limit,
        initial_action_std=effective_initial_action_std,
        waypoint_count=3,
        mlp_hidden_dims=args.mlp_hidden_dims,
        kmpc_hidden_dims=args.kmpc_hidden_dims,
        horizon=args.horizon,
        solver_iterations=args.solver_iterations,
        quadratic_log_scale=args.quadratic_log_scale,
        linear_scale=args.linear_scale,
        action_quadratic_scale=args.action_quadratic_scale,
        tip_weight=args.tip_weight,
        max_delta=actor_max_delta,
        normalized_delta_curvature=args.normalized_delta_curvature,
        kmpc_cost_parameterization=args.kmpc_cost_parameterization,
        structured_log_scale=args.structured_log_scale,
        kmpc_reference_mode=args.kmpc_reference_mode,
        structured_terminal_multiplier=args.structured_terminal_multiplier,
    )
    actor_parameters = [p for p in policy.actor.parameters() if p.requires_grad]
    critic_parameters = [p for p in policy.critic.parameters() if p.requires_grad]
    policy.log_std.requires_grad_(not args.freeze_log_std)
    actor_optimizer = torch.optim.Adam(
        actor_parameters,
        lr=actor_learning_rate,
        eps=1e-5,
    )
    critic_optimizer = torch.optim.Adam(
        critic_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )
    std_optimizer = (
        None
        if args.freeze_log_std
        else torch.optim.Adam(
            [policy.log_std],
            lr=args.std_learning_rate,
            eps=1e-5,
        )
    )

    runtime = {
        "actor_name": args.actor,
        "absolute_action_limit": args.absolute_action_limit,
        "progress_reward_scale": args.progress_reward_scale,
        "success_threshold": MANISOFT_WAYPOINT_SUCCESS_THRESHOLD,
        "required_success_streak": MANISOFT_WAYPOINT_SUCCESS_STREAK,
        "initial_action_std": effective_initial_action_std,
        "minimum_action_std": effective_minimum_action_std,
        "maximum_action_std": effective_maximum_action_std,
        "observation_dim": policy.observation_dim,
        "history_steps": policy.history_steps,
        "waypoint_count": policy.waypoint_count,
        "mlp_hidden_dims": list(args.mlp_hidden_dims),
        "kmpc_hidden_dims": list(args.kmpc_hidden_dims),
        "horizon": args.horizon,
        "solver_iterations": args.solver_iterations,
        "solver_diagnostic_iterations": args.solver_diagnostic_iterations,
        "quadratic_log_scale": args.quadratic_log_scale,
        "linear_scale": args.linear_scale,
        "action_quadratic_scale": args.action_quadratic_scale,
        "tip_weight": args.tip_weight,
        "normalized_delta_curvature": args.normalized_delta_curvature,
        "kmpc_cost_parameterization": (
            None
            if args.actor == "ppo_mlp"
            else args.kmpc_cost_parameterization
        ),
        "structured_log_scale": (
            None
            if args.actor == "ppo_mlp"
            or args.kmpc_cost_parameterization != "structured"
            else args.structured_log_scale
        ),
        "solver": (
            None
            if args.actor == "ppo_mlp"
            else (
                "normalized_delta_box_fista_v1"
                if normalized_delta_policy
                else "absolute_action_box_fista_v1"
            )
        ),
        "fixed_smoothness": False,
        "max_delta": actor_max_delta,
        "policy_action_semantics": (
            "absolute_action"
            if args.actor == "ppo_mlp" or not normalized_delta_policy
            else "normalized_delta_action"
        ),
        "action_std_units": (
            "physical_action"
            if args.actor == "ppo_mlp" or not normalized_delta_policy
            else "normalized_delta"
        ),
        "action_distribution": policy.ACTION_DISTRIBUTION,
        "freeze_log_std": args.freeze_log_std,
        "std_learning_rate": args.std_learning_rate,
        "cost_initialization": (
            None
            if args.actor == "ppo_mlp"
            else policy.cost_initialization
        ),
    }
    if args.actor == "ppo_kmpc" and args.kmpc_reference_mode != "explicit":
        runtime["kmpc_reference_mode"] = args.kmpc_reference_mode
    if args.actor == "ppo_kmpc" and not args.structured_terminal_multiplier:
        runtime["structured_terminal_multiplier"] = False
    if args.actor == "ppo_kmpc" and not normalized_delta_policy:
        runtime.update(
            {
                "kmpc_decision_space": "absolute",
                "configured_max_delta": args.max_delta,
                "configured_initial_action_std": args.initial_action_std,
                "configured_minimum_action_std": args.minimum_action_std,
                "configured_maximum_action_std": args.maximum_action_std,
                "physical_exploration_std_initial": (
                    effective_initial_action_std
                ),
                "exploration_std_conversion": (
                    "v15e_normalized_delta_to_physical_v1"
                ),
            }
        )
    if args.actor == "ppo_mlp":
        runtime["mlp_feature_extractor"] = (
            "frozen_history_koopman_lift_plus_waypoints_v1"
        )
    training_signature = {
        "rollout_steps": args.rollout_steps,
        "num_envs": args.num_envs,
        "minibatch_size": args.minibatch_size,
        "update_epochs": args.update_epochs,
        "learning_rate": args.learning_rate,
        "actor_learning_rate": actor_learning_rate,
        "std_learning_rate": args.std_learning_rate,
        "freeze_log_std": args.freeze_log_std,
        "anneal_learning_rate": args.anneal_learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "clip_value_loss": args.clip_value_loss,
        "value_coefficient": args.value_coefficient,
        "entropy_coefficient": args.entropy_coefficient,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "kl_soft_stop_multiplier": args.kl_soft_stop_multiplier,
        "kl_hard_rollback_multiplier": args.kl_hard_rollback_multiplier,
        "normalize_advantages_globally": args.normalize_advantages_globally,
    }
    metadata = {
        "method": METHOD,
        "format_version": FORMAT_VERSION,
        "training_spec_version": TRAINING_SPEC_VERSION,
        "actor_name": args.actor,
        "initialization": (
            "frozen pretrained history Koopman lift; random MLP actor and "
            "critic; no behavior cloning"
            if args.actor == "ppo_mlp"
            else (
                f"random {args.kmpc_cost_parameterization} cost-map actor "
                "and MLP critic; frozen pretrained history Koopman encoder "
                "and A/B/C; no behavior cloning"
            )
        ),
        "bc_checkpoint": None,
        "expert_dataset": None,
        "koopman_checkpoint": str(koopman_path),
        "koopman_checkpoint_sha256": sha256(koopman_path),
        "koopman_usage": (
            "frozen_history_encoder_lift"
            if args.actor == "ppo_mlp"
            else "frozen_encoder_and_dynamics"
        ),
        "scenario": str(scenario),
        "waypoint_root": str(waypoint_root),
        "waypoint_bank_manifest": str(waypoint_bank.manifest_path),
        "waypoint_bank_sha256": waypoint_bank.manifest_sha256,
        "waypoint_triplet_count": waypoint_bank.triplet_count,
        "seed": args.seed,
        "runtime": runtime,
        "training_signature": training_signature,
    }

    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    status_path = output / "training_status.json"
    if args.resume is None and history_path.exists():
        raise FileExistsError(
            f"{history_path} already exists; use a new output or --resume"
        )

    timesteps = 0
    update = 0
    elapsed_before = 0.0
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser().resolve()
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if payload.get("method") != METHOD:
            raise ValueError("Resume checkpoint is not a from-scratch PPO run")
        if payload.get("actor_name") != args.actor:
            raise ValueError("Resume actor does not match --actor")
        if payload.get("training_spec_version") != TRAINING_SPEC_VERSION:
            raise ValueError("Resume training specification is incompatible")
        if payload.get("koopman_checkpoint_sha256") != metadata[
            "koopman_checkpoint_sha256"
        ]:
            raise ValueError("Resume checkpoint references another Koopman model")
        if payload.get("waypoint_bank_sha256") != waypoint_bank.manifest_sha256:
            raise ValueError("Resume checkpoint references another waypoint bank")
        if payload.get("runtime") != runtime:
            raise ValueError("Resume runtime configuration is incompatible")
        if payload.get("training_signature") != training_signature:
            raise ValueError("Resume PPO hyperparameters are incompatible")
        if int(payload.get("seed", -1)) != args.seed:
            raise ValueError("Resume seed does not match --seed")
        policy.load_state_dict(payload["policy"])
        actor_optimizer.load_state_dict(payload["actor_optimizer"])
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if std_optimizer is not None:
            std_optimizer.load_state_dict(payload["std_optimizer"])
        timesteps = int(payload["timesteps"])
        update = int(payload["update"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        stored_score = payload.get("best_score")
        if stored_score is not None:
            best_score = tuple(map(float, stored_score))

    _write_json(
        output / "run_config.json",
        {
            **metadata,
            "arguments": vars(args),
            "resolved_actor_learning_rate": actor_learning_rate,
            "device": str(device),
            "trainable_actor_parameters": sum(p.numel() for p in actor_parameters),
            "initial_actor_sha256": _module_sha256(policy.actor),
            "trainable_critic_parameters": sum(
                p.numel() for p in policy.critic.parameters()
            ),
            "initial_critic_sha256": _module_sha256(policy.critic),
            "trainable_log_std_parameters": (
                policy.log_std.numel() if policy.log_std.requires_grad else 0
            ),
            "environment_execution": (
                "spawn_process_vector_v1"
                if args.parallel_env_processes
                else "in_process_sequential_vector_v1"
            ),
        },
    )

    state_stats = koopman_payload["normalizers"]["state"]
    environment_kwargs = {
        "scenario": scenario,
        "waypoint_tips": waypoint_tips,
        "episode_steps": args.episode_steps,
        "absolute_action_limit": args.absolute_action_limit,
        "progress_reward_scale": args.progress_reward_scale,
        "history_steps": policy.history_steps,
        "state_mean": np.asarray(state_stats["mean"], dtype=np.float32),
        "state_std": np.asarray(state_stats["std"], dtype=np.float32),
        "max_delta": (
            actor_max_delta
        ),
    }
    if args.parallel_env_processes:
        envs = ProcessVectorEnv(
            [
                partial(_make_env, **environment_kwargs)
                for _ in range(args.num_envs)
            ]
        )
        envs.reset([args.seed + index for index in range(args.num_envs)])
    else:
        envs = [
            _make_env(**environment_kwargs)
            for _ in range(args.num_envs)
        ]
        for index, env in enumerate(envs):
            observation, _ = env.reset(seed=args.seed + index)
            env._ppo_observation = observation
            env.action_space.seed(args.seed + index)

    def checkpoint_payload(elapsed_seconds: float) -> dict:
        return {
            **metadata,
            "policy": policy.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "std_optimizer": (
                None if std_optimizer is None else std_optimizer.state_dict()
            ),
            "timesteps": timesteps,
            "update": update,
            "elapsed_seconds": elapsed_seconds,
            "best_score": list(best_score),
        }

    minimum_log_std = math.log(effective_minimum_action_std)
    maximum_log_std = math.log(effective_maximum_action_std)
    total_updates = math.ceil(args.total_timesteps / args.rollout_steps)
    started = time.monotonic()
    wall_time_reached = False
    try:
        while timesteps < args.total_timesteps:
            current_steps = min(
                args.rollout_steps,
                args.total_timesteps - timesteps,
            )
            if current_steps % args.num_envs:
                raise ValueError("Final rollout is not divisible by --num-envs")
            if args.anneal_learning_rate:
                fraction = max(0.0, 1.0 - update / max(total_updates, 1))
                actor_optimizer.param_groups[0]["lr"] = (
                    fraction * actor_learning_rate
                )
                critic_optimizer.param_groups[0]["lr"] = (
                    fraction * args.learning_rate
                )
                if std_optimizer is not None:
                    std_optimizer.param_groups[0]["lr"] = (
                        fraction * args.std_learning_rate
                    )

            synchronize = (
                (lambda: torch.cuda.synchronize(device))
                if device.type == "cuda"
                else (lambda: None)
            )
            synchronize()
            rollout_started = time.perf_counter()
            rollout = (
                collect_rollout(
                    envs[0],
                    policy,
                    current_steps,
                    args.gamma,
                    args.gae_lambda,
                    device,
                )
                if args.num_envs == 1
                else collect_vector_rollout(
                    envs,
                    policy,
                    current_steps,
                    args.gamma,
                    args.gae_lambda,
                    device,
                )
            )
            synchronize()
            rollout_seconds = time.perf_counter() - rollout_started

            update_started = time.perf_counter()
            metrics = ppo_update(
                policy,
                actor_optimizer,
                rollout,
                update_epochs=args.update_epochs,
                minibatch_size=args.minibatch_size,
                clip_range=args.clip_range,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
                clip_value_loss=args.clip_value_loss,
                minimum_log_std=minimum_log_std,
                maximum_log_std=maximum_log_std,
                critic_optimizer=critic_optimizer,
                std_optimizer=std_optimizer,
                kl_soft_stop_multiplier=args.kl_soft_stop_multiplier,
                kl_hard_rollback_multiplier=(
                    args.kl_hard_rollback_multiplier
                ),
                normalize_advantages_globally=(
                    args.normalize_advantages_globally
                ),
            )
            synchronize()
            update_seconds = time.perf_counter() - update_started
            timesteps += current_steps
            update += 1
            elapsed = elapsed_before + time.monotonic() - started

            with torch.no_grad():
                diagnostic = policy(
                    rollout.observations[: min(16, current_steps)]
                )
            completed_return = (
                float(rollout.episode_returns.mean())
                if len(rollout.episode_returns)
                else None
            )
            success_rate = (
                float(rollout.episode_successes.mean())
                if len(rollout.episode_successes)
                else None
            )
            waypoint_mean = (
                float(rollout.episode_waypoints_completed.mean())
                if len(rollout.episode_waypoints_completed)
                else None
            )
            finite_distances = rollout.distances[np.isfinite(rollout.distances)]
            row = {
                "method": METHOD,
                "actor_name": args.actor,
                "seed": args.seed,
                "update": update,
                "timesteps": timesteps,
                "elapsed_seconds": elapsed,
                "actor_learning_rate": actor_optimizer.param_groups[0]["lr"],
                "learning_rate": critic_optimizer.param_groups[0]["lr"],
                "std_learning_rate": (
                    None
                    if std_optimizer is None
                    else std_optimizer.param_groups[0]["lr"]
                ),
                "rollout_seconds": rollout_seconds,
                "ppo_update_seconds": update_seconds,
                "transitions_per_second": current_steps
                / max(rollout_seconds + update_seconds, 1e-12),
                **metrics,
                "reward_mean": float(rollout.rewards.mean()),
                "completed_episodes": len(rollout.episode_returns),
                "completed_episode_return_mean": completed_return,
                "episode_length_mean": (
                    float(rollout.episode_lengths.mean())
                    if len(rollout.episode_lengths)
                    else None
                ),
                "completed_successes": int(rollout.episode_successes.sum()),
                "completed_success_rate": success_rate,
                "waypoints_completed_mean": waypoint_mean,
                "distance_mean": (
                    float(finite_distances.mean())
                    if len(finite_distances)
                    else None
                ),
                "distance_minimum": (
                    float(finite_distances.min())
                    if len(finite_distances)
                    else None
                ),
                "action_clip_saturation_rate": float(rollout.saturation.mean()),
                "action_bound_rate": float(rollout.action_bound.mean()),
                "applied_action_abs_mean": float(
                    rollout.applied_action_abs_mean.mean()
                ),
                "applied_delta_action_l2_mean": float(
                    rollout.applied_delta_action_l2.mean()
                ),
                "applied_delta_action_abs_max": float(
                    rollout.applied_delta_action_abs_max.max()
                ),
                "policy_mean_abs_mean": float(diagnostic.mean.abs().mean()),
                "log_std": policy.log_std.detach().cpu().tolist(),
                "action_std_mean": float(policy.log_std.exp().mean()),
                "physical_exploration_std_mean": float(
                    policy.log_std.exp().mean()
                    * (
                        args.max_delta
                        if normalized_delta_policy
                        else 1.0
                    )
                ),
            }
            if args.actor == "ppo_kmpc":
                high_solution, high_residual = policy.actor.solve_condensed_qp(
                    diagnostic.mpc.qp_hessian,
                    diagnostic.mpc.qp_linear,
                    iterations=args.solver_diagnostic_iterations,
                    previous_action=diagnostic.mpc.previous_action,
                )
                high_first = high_solution.reshape(
                    -1,
                    policy.actor.horizon,
                    policy.actor.action_dim,
                )[:, 0]
                if normalized_delta_policy:
                    if diagnostic.mpc.normalized_delta is None:
                        raise RuntimeError(
                            "PPO-KMPC diagnostic is missing normalized delta"
                        )
                    deployed_first = diagnostic.mpc.normalized_delta.reshape(
                        -1,
                        policy.actor.action_dim,
                    )
                else:
                    deployed_first = diagnostic.mpc.action.reshape(
                        -1,
                        policy.actor.action_dim,
                    )
                difference = high_first - deployed_first
                kmpc_metrics = {
                    "quadratic_weight_mean": float(
                        diagnostic.mpc.quadratic_diagonal.mean()
                    ),
                    "linear_weight_abs_mean": float(
                        diagnostic.mpc.linear_term.abs().mean()
                    ),
                    "projected_gradient_residual_mean": float(
                        diagnostic.mpc.projected_gradient_residual.mean()
                    ),
                    "diagnostic_solver_residual_mean": float(
                        high_residual.mean()
                    ),
                    "diagnostic_first_action_l2_difference_mean": float(
                        torch.linalg.vector_norm(difference, dim=-1).mean()
                    ),
                    "diagnostic_first_action_abs_difference_max": float(
                        difference.abs().max()
                    ),
                }
                if normalized_delta_policy:
                    kmpc_metrics.update(
                        {
                            "normalized_delta_abs_mean": float(
                                rollout.actions.abs().mean()
                            ),
                            "normalized_delta_bound_rate": float(
                                (rollout.actions.abs() >= 1.0 - 1e-6)
                                .float()
                                .mean()
                            ),
                            "physical_delta_limit": args.max_delta,
                        }
                    )
                else:
                    kmpc_metrics.update(
                        {
                            "normalized_delta_abs_mean": None,
                            "normalized_delta_bound_rate": None,
                            "physical_delta_limit": None,
                        }
                    )
                row.update(kmpc_metrics)

            is_best = False
            if (
                success_rate is not None
                and waypoint_mean is not None
                and completed_return is not None
            ):
                score = (success_rate, waypoint_mean, completed_return)
                if score > best_score:
                    best_score = score
                    is_best = True
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            current_payload = checkpoint_payload(elapsed)
            _save(output / "last.pt", current_payload)
            if is_best:
                _save(output / "best.pt", current_payload)
            if update % args.checkpoint_interval_updates == 0:
                _save(
                    output / f"recovery_update_{update:06d}.pt",
                    current_payload,
                )
            _write_json(status_path, {"state": "running", **row})
            print(json.dumps(row, sort_keys=True), flush=True)
            if (
                args.max_wall_time_hours is not None
                and elapsed >= args.max_wall_time_hours * 3600.0
            ):
                wall_time_reached = True
                break

        elapsed = elapsed_before + time.monotonic() - started
        _write_json(
            status_path,
            {
                "state": "wall_time_reached" if wall_time_reached else "complete",
                "method": METHOD,
                "actor_name": args.actor,
                "timesteps": timesteps,
                "updates": update,
                "elapsed_seconds": elapsed,
                "best_score": list(best_score),
                "last_checkpoint_sha256": sha256(output / "last.pt"),
            },
        )
    except BaseException as error:
        elapsed = elapsed_before + time.monotonic() - started
        emergency = checkpoint_payload(elapsed)
        emergency["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _save(output / "emergency.pt", emergency)
        _write_json(
            status_path,
            {
                "state": "failed",
                "method": METHOD,
                "actor_name": args.actor,
                "timesteps": timesteps,
                "updates": update,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        if isinstance(envs, ProcessVectorEnv):
            envs.close()
        else:
            for env in envs:
                env.close()


if __name__ == "__main__":
    main()
