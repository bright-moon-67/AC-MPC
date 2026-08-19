#!/usr/bin/env python
"""Offline IQL training for the ManiSoft history Koopman-MPC policy."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from antmaze_ac.data.offline_transition_dataset import OfflineTransitionDataset
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.iql import (
    build_iql_candidate_policy,
    distill_behavior_means,
    fixed_critic_selection_metrics,
    IQLActionValue,
    IQLTrainer,
    IQLValue,
    offline_validation_metrics,
)


METHOD = "manisoft_kmpc_iql"
FORMAT_VERSION = 2
DEFAULT_DATASET = (
    "data/processed/manisoft_kmpc_offline/"
    "combined_zmixed904_v4_1498/dataset.npz"
)


def _device(specification: str) -> torch.device:
    return torch.device(
        "cuda"
        if specification == "auto" and torch.cuda.is_available()
        else ("cpu" if specification == "auto" else specification)
    )


def _save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _dataset_provenance(dataset_path: Path) -> dict[str, Any]:
    summary_path = dataset_path.with_name("summary.json")
    if not summary_path.is_file():
        return {"summary": None, "collection_configs": []}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    configs: list[dict[str, Any]] = []
    for source in summary.get("roots", []):
        root = Path(source["root"]).expanduser()
        if not root.is_absolute():
            root = (dataset_path.parent / root).resolve()
        paths = sorted(root.glob("part_*/collection_config.json"))
        if not paths:
            direct = root / "collection_config.json"
            paths = [direct] if direct.is_file() else []
        for path in paths:
            config = json.loads(path.read_text(encoding="utf-8"))
            configs.append(
                {
                    "path": str(path.resolve()),
                    "checkpoint": config.get("checkpoint"),
                    "checkpoint_sha256": config.get("checkpoint_sha256"),
                    "scenario_sha256": config.get("scenario_sha256"),
                    "waypoint_bank_sha256": config.get("waypoint_bank_sha256"),
                    "action_semantics": config.get("action_semantics"),
                    "max_delta": config.get("max_delta"),
                    "observation_dim": config.get("runtime", {}).get(
                        "observation_dim"
                    ),
                }
            )
    unique_configs = {
        json.dumps(config, sort_keys=True): config for config in configs
    }
    return {
        "summary": {
            key: value
            for key, value in summary.items()
            if key != "episode_summaries"
        },
        "collection_configs": list(unique_configs.values()),
    }


def _provenance_values(
    provenance: dict[str, Any], key: str
) -> list[Any]:
    return sorted(
        {
            config.get(key)
            for config in provenance["collection_configs"]
            if config.get(key) is not None
        },
        key=str,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--initial-policy-checkpoint",
        default=None,
        help="PPO-KMPC behavior checkpoint used to initialize the IQL actor.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument(
        "--activation",
        choices=("relu", "gelu", "silu", "tanh"),
        default="relu",
    )
    parser.add_argument("--actor-learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--candidate-cost-parameterization",
        choices=("checkpoint", "structured_v2"),
        default="structured_v2",
    )
    parser.add_argument("--candidate-solver-iterations", type=int, default=60)
    parser.add_argument("--distillation-steps", type=int, default=10_000)
    parser.add_argument("--distillation-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-warmup-steps", type=int, default=20_000)
    parser.add_argument(
        "--selection-behavior-mse-penalty", type=float, default=10.0
    )
    parser.add_argument("--structured-shape-weight", type=float, default=1e-3)
    parser.add_argument(
        "--structured-linear-velocity-weight", type=float, default=1e-2
    )
    parser.add_argument(
        "--structured-angular-velocity-weight", type=float, default=1e-2
    )
    parser.add_argument(
        "--structured-normalized-delta-weight", type=float, default=1e-4
    )
    parser.add_argument("--q-learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-advantage-weight", type=float, default=100.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--reward-bias",
        type=float,
        default=0.0,
        help="The local dense reward defaults to no shift; rlkit AntMaze used -1.",
    )
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--learn-log-std", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--minimum-action-std", type=float, default=0.001)
    parser.add_argument("--maximum-action-std", type=float, default=0.2)
    parser.add_argument(
        "--treat-timeouts-as-terminal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default bootstraps transition-complete time-limit truncations.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--validation-interval", type=int, default=5_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    args = parser.parse_args()
    if args.resume is None and args.initial_policy_checkpoint is None:
        parser.error("--initial-policy-checkpoint is required for a new run")
    positive = (
        args.gradient_steps,
        args.batch_size,
        args.validation_batch_size,
        args.validation_interval,
        args.log_interval,
        args.checkpoint_interval,
        args.candidate_solver_iterations,
        *args.hidden_dims,
    )
    if min(positive) < 1:
        parser.error("Counts, intervals, and hidden dimensions must be positive")
    for name in (
        "actor_learning_rate",
        "q_learning_rate",
        "value_learning_rate",
        "temperature",
        "max_advantage_weight",
        "target_tau",
        "max_grad_norm",
        "minimum_action_std",
        "maximum_action_std",
        "distillation_learning_rate",
        "selection_behavior_mse_penalty",
        "structured_shape_weight",
        "structured_linear_velocity_weight",
        "structured_angular_velocity_weight",
        "structured_normalized_delta_weight",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.discount <= 1.0:
        parser.error("--discount must be in [0,1]")
    if not 0.0 < args.expectile < 1.0:
        parser.error("--expectile must be in (0,1)")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0,1)")
    if args.minimum_action_std > args.maximum_action_std:
        parser.error("minimum action std must not exceed maximum action std")
    if args.max_wall_time_hours is not None and args.max_wall_time_hours <= 0:
        parser.error("--max-wall-time-hours must be positive")
    if args.distillation_steps < 0 or args.critic_warmup_steps < 0:
        parser.error("Distillation and critic warm-up steps must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    dataset_path = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    resume_payload = None
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser().resolve()
        resume_payload = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if resume_payload.get("method") != METHOD:
            raise ValueError("Resume checkpoint is not a ManiSoft KMPC-IQL run")
        if int(resume_payload.get("format_version", 0)) != FORMAT_VERSION:
            raise ValueError(
                "Resume checkpoint predates explicit-u_prev IQL format v2"
            )
        initial_policy_path = Path(
            resume_payload["initial_policy_checkpoint"]
        ).expanduser().resolve()
        if (
            args.initial_policy_checkpoint is not None
            and Path(args.initial_policy_checkpoint).expanduser().resolve()
            != initial_policy_path
        ):
            raise ValueError("Resume initialization checkpoint does not match")
    else:
        initial_policy_path = Path(
            args.initial_policy_checkpoint
        ).expanduser().resolve()
    if not dataset_path.is_file() or not initial_policy_path.is_file():
        raise FileNotFoundError("Dataset and initial PPO-KMPC checkpoint must exist")

    policy, initial_payload, koopman_payload = build_iql_candidate_policy(
        initial_policy_path,
        device,
        cost_parameterization=args.candidate_cost_parameterization,
        solver_iterations=args.candidate_solver_iterations,
        structured_shape_weight=args.structured_shape_weight,
        structured_linear_velocity_weight=(
            args.structured_linear_velocity_weight
        ),
        structured_angular_velocity_weight=(
            args.structured_angular_velocity_weight
        ),
        structured_normalized_delta_weight=(
            args.structured_normalized_delta_weight
        ),
    )
    if initial_payload.get("actor_name") != "ppo_kmpc":
        raise ValueError("IQL requires a PPO-KMPC initialization checkpoint")
    policy.requires_grad_(False)
    policy.actor.requires_grad_(True)
    policy.log_std.requires_grad_(args.learn_log_std)

    provenance = _dataset_provenance(dataset_path)
    source_max_deltas = _provenance_values(provenance, "max_delta")
    source_semantics = _provenance_values(provenance, "action_semantics")
    source_policy_hashes = _provenance_values(
        provenance, "checkpoint_sha256"
    )
    initial_policy_hash = sha256(initial_policy_path)
    if source_policy_hashes and source_policy_hashes != [initial_policy_hash]:
        raise ValueError(
            "The initialization checkpoint does not match the behavior "
            f"policy recorded by the dataset: {source_policy_hashes}"
        )
    if source_semantics and source_semantics != ["normalized_delta_action"]:
        raise ValueError(f"Unsupported dataset action semantics: {source_semantics}")
    if len(source_max_deltas) > 1:
        print(
            "WARNING: the merged dataset mixes physical max_delta values "
            f"{source_max_deltas}; normalized actions therefore come from "
            "slightly different transition dynamics. Training will continue.",
            flush=True,
        )

    dataset = OfflineTransitionDataset(
        dataset_path,
        cache_dir=args.cache_dir,
        treat_timeouts_as_terminal=args.treat_timeouts_as_terminal,
    )
    if not dataset.metadata()["has_behavior_action_means"]:
        raise ValueError(
            "IQL distillation/checkpoint selection requires "
            "behavior_action_means in the dataset"
        )
    if dataset.observation_dim != policy.observation_dim:
        raise ValueError(
            f"Dataset observation dim {dataset.observation_dim} != policy "
            f"dim {policy.observation_dim}"
        )
    if dataset.action_dim != policy.action_dim:
        raise ValueError(
            f"Dataset action dim {dataset.action_dim} != policy dim "
            f"{policy.action_dim}"
        )
    train_indices, validation_indices = dataset.split_by_episode(
        args.validation_fraction, args.seed
    )
    feature_dim = int(
        policy.koopman.lifted_dim
        + policy.task_context_dim
        + policy.action_dim
    )
    network_kwargs = {
        "hidden_dims": tuple(args.hidden_dims),
        "activation": args.activation,
    }
    qf1 = IQLActionValue(
        feature_dim, policy.action_dim, action_scale=1.0, **network_kwargs
    ).to(device)
    qf2 = IQLActionValue(
        feature_dim, policy.action_dim, action_scale=1.0, **network_kwargs
    ).to(device)
    target_qf1 = copy.deepcopy(qf1).requires_grad_(False)
    target_qf2 = copy.deepcopy(qf2).requires_grad_(False)
    selection_qf1 = copy.deepcopy(qf1).requires_grad_(False)
    selection_qf2 = copy.deepcopy(qf2).requires_grad_(False)
    vf = IQLValue(feature_dim, **network_kwargs).to(device)
    policy_parameters = list(policy.actor.parameters())
    if args.learn_log_std:
        policy_parameters.append(policy.log_std)
    policy_optimizer = torch.optim.Adam(
        policy_parameters, lr=args.actor_learning_rate
    )
    distillation_optimizer = torch.optim.Adam(
        list(policy.actor.parameters()), lr=args.distillation_learning_rate
    )
    qf_optimizer = torch.optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_learning_rate,
    )
    vf_optimizer = torch.optim.Adam(vf.parameters(), lr=args.value_learning_rate)
    trainer = IQLTrainer(
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        vf,
        policy_optimizer,
        qf_optimizer,
        vf_optimizer,
        discount=args.discount,
        expectile=args.expectile,
        temperature=args.temperature,
        max_advantage_weight=args.max_advantage_weight,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
        target_tau=args.target_tau,
        max_grad_norm=args.max_grad_norm,
        minimum_log_std=math.log(args.minimum_action_std),
        maximum_log_std=math.log(args.maximum_action_std),
    )

    runtime = {
        **initial_payload["runtime"],
        "algorithm": "iql",
        "feature_dim": feature_dim,
        "hidden_dims": list(args.hidden_dims),
        "activation": args.activation,
        "discount": args.discount,
        "expectile": args.expectile,
        "temperature": args.temperature,
        "max_advantage_weight": args.max_advantage_weight,
        "reward_scale": args.reward_scale,
        "reward_bias": args.reward_bias,
        "target_tau": args.target_tau,
        "learn_log_std": args.learn_log_std,
        "minimum_action_std": args.minimum_action_std,
        "maximum_action_std": args.maximum_action_std,
        "treat_timeouts_as_terminal": args.treat_timeouts_as_terminal,
        "source_max_deltas": source_max_deltas,
        "source_behavior_policy_sha256": source_policy_hashes,
        "deployment_max_delta": initial_payload["runtime"].get("max_delta"),
        "critic_state_includes_previous_action": True,
        "critic_warmup_steps": args.critic_warmup_steps,
        "checkpoint_selection": "frozen_warmup_q_minus_behavior_mean_mse",
        "kmpc_cost_parameterization": (
            initial_payload["runtime"].get("kmpc_cost_parameterization")
            if args.candidate_cost_parameterization == "checkpoint"
            else args.candidate_cost_parameterization
        ),
        "solver_iterations": args.candidate_solver_iterations,
        "structured_shape_weight": args.structured_shape_weight,
        "structured_linear_velocity_weight": args.structured_linear_velocity_weight,
        "structured_angular_velocity_weight": args.structured_angular_velocity_weight,
        "structured_normalized_delta_weight": args.structured_normalized_delta_weight,
    }
    candidate = {
        "cost_parameterization": args.candidate_cost_parameterization,
        "solver_iterations": args.candidate_solver_iterations,
        "structured_shape_weight": args.structured_shape_weight,
        "structured_linear_velocity_weight": (
            args.structured_linear_velocity_weight
        ),
        "structured_angular_velocity_weight": (
            args.structured_angular_velocity_weight
        ),
        "structured_normalized_delta_weight": (
            args.structured_normalized_delta_weight
        ),
    }
    metadata = {
        "method": METHOD,
        "format_version": FORMAT_VERSION,
        "actor_name": "ppo_kmpc",
        "initial_policy_checkpoint": str(initial_policy_path),
        "initial_policy_checkpoint_sha256": sha256(initial_policy_path),
        "koopman_checkpoint": initial_payload["koopman_checkpoint"],
        "koopman_checkpoint_sha256": initial_payload[
            "koopman_checkpoint_sha256"
        ],
        "scenario": initial_payload.get("scenario"),
        "waypoint_root": initial_payload.get("waypoint_root"),
        "waypoint_bank_sha256": initial_payload.get("waypoint_bank_sha256"),
        "dataset": str(dataset_path),
        "dataset_size": dataset_path.stat().st_size,
        "dataset_metadata": dataset.metadata(),
        "dataset_provenance": provenance,
        "seed": args.seed,
        "runtime": runtime,
        "candidate": candidate,
        "training_signature": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "resume",
                "max_wall_time_hours",
                "output",
                "initial_policy_checkpoint",
            }
        },
    }

    if resume_payload is not None:
        if resume_payload.get("training_signature") != metadata[
            "training_signature"
        ]:
            raise ValueError("Resume IQL hyperparameters or dataset do not match")
        if resume_payload.get("initial_policy_checkpoint_sha256") != metadata[
            "initial_policy_checkpoint_sha256"
        ]:
            raise ValueError("Resume initialization policy hash does not match")

    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    if resume_payload is None and history_path.exists():
        raise FileExistsError(
            f"{history_path} exists; use a new --output or pass --resume"
        )
    distillation_step = 0
    gradient_step = 0
    elapsed_before = 0.0
    best_offline_score = -float("inf")
    selection_critic_ready = False
    rng = np.random.default_rng(args.seed)
    if resume_payload is not None:
        for key, module in (
            ("policy", policy),
            ("qf1", qf1),
            ("qf2", qf2),
            ("target_qf1", target_qf1),
            ("target_qf2", target_qf2),
            ("vf", vf),
        ):
            module.load_state_dict(resume_payload[key])
        policy_optimizer.load_state_dict(resume_payload["policy_optimizer"])
        qf_optimizer.load_state_dict(resume_payload["qf_optimizer"])
        vf_optimizer.load_state_dict(resume_payload["vf_optimizer"])
        if "distillation_optimizer" in resume_payload:
            distillation_optimizer.load_state_dict(
                resume_payload["distillation_optimizer"]
            )
        distillation_step = int(resume_payload.get("distillation_step", 0))
        gradient_step = int(resume_payload["gradient_step"])
        elapsed_before = float(resume_payload.get("elapsed_seconds", 0.0))
        best_offline_score = float(
            resume_payload.get("best_offline_score", -float("inf"))
        )
        selection_critic_ready = bool(
            resume_payload.get("selection_critic_ready", False)
        )
        if selection_critic_ready:
            selection_qf1.load_state_dict(resume_payload["selection_qf1"])
            selection_qf2.load_state_dict(resume_payload["selection_qf2"])
        rng.bit_generator.state = resume_payload["numpy_rng_state"]
        torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and resume_payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_payload["cuda_rng_state"]]
            )

    _write_json(
        output / "run_config.json",
        {
            **metadata,
            "arguments": vars(args),
            "device": str(device),
            "train_transitions": int(len(train_indices)),
            "validation_transitions": int(len(validation_indices)),
            "trainable_policy_parameters": int(
                sum(parameter.numel() for parameter in policy_parameters)
            ),
            "q_parameters": int(
                sum(p.numel() for p in qf1.parameters())
                + sum(p.numel() for p in qf2.parameters())
            ),
            "value_parameters": int(sum(p.numel() for p in vf.parameters())),
            "frozen_koopman_parameters": int(
                sum(p.numel() for p in policy.koopman.parameters())
            ),
        },
    )

    def checkpoint_payload(elapsed: float) -> dict[str, Any]:
        return {
            **metadata,
            "policy": policy.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "target_qf1": target_qf1.state_dict(),
            "target_qf2": target_qf2.state_dict(),
            "vf": vf.state_dict(),
            "selection_qf1": selection_qf1.state_dict(),
            "selection_qf2": selection_qf2.state_dict(),
            "selection_critic_ready": selection_critic_ready,
            "policy_optimizer": policy_optimizer.state_dict(),
            "distillation_optimizer": distillation_optimizer.state_dict(),
            "qf_optimizer": qf_optimizer.state_dict(),
            "vf_optimizer": vf_optimizer.state_dict(),
            "distillation_step": distillation_step,
            "gradient_step": gradient_step,
            "elapsed_seconds": elapsed,
            "best_offline_score": best_offline_score,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    validation_rng_seed = args.seed + 1_000_003
    accumulated: dict[str, torch.Tensor] = {}
    accumulated_count = 0
    starting_gradient_step = gradient_step
    started = time.monotonic()
    status = "complete"
    distillation_target_steps = (
        args.distillation_steps
        if args.candidate_cost_parameterization == "structured_v2"
        else 0
    )
    if args.critic_warmup_steps == 0 and not selection_critic_ready:
        selection_qf1.load_state_dict(target_qf1.state_dict())
        selection_qf2.load_state_dict(target_qf2.state_dict())
        selection_critic_ready = True
    try:
        while distillation_step < distillation_target_steps:
            batch = dataset.sample_batch(
                args.batch_size, rng, device, indices=train_indices
            )
            metrics = distill_behavior_means(
                policy,
                batch,
                distillation_optimizer,
                max_grad_norm=args.max_grad_norm,
            )
            distillation_step += 1
            elapsed = elapsed_before + time.monotonic() - started
            if (
                distillation_step == 1
                or distillation_step % args.log_interval == 0
            ):
                row = {
                    "method": METHOD,
                    "phase": "behavior_mean_distillation",
                    "distillation_step": distillation_step,
                    "gradient_step": gradient_step,
                    "elapsed_seconds": elapsed,
                    **{key: float(value) for key, value in metrics.items()},
                }
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
            if (
                distillation_step % args.checkpoint_interval == 0
                or distillation_step == distillation_target_steps
            ):
                _save(output / "last.pt", checkpoint_payload(elapsed))
            if (
                args.max_wall_time_hours is not None
                and elapsed >= args.max_wall_time_hours * 3600.0
            ):
                status = "wall_time_reached"
                _save(output / "last.pt", checkpoint_payload(elapsed))
                break

        while gradient_step < args.gradient_steps:
            if status == "wall_time_reached":
                break
            batch = dataset.sample_batch(
                args.batch_size, rng, device, indices=train_indices
            )
            metrics = trainer.update(
                batch,
                update_policy=gradient_step >= args.critic_warmup_steps,
            )
            gradient_step += 1
            if (
                not selection_critic_ready
                and gradient_step >= args.critic_warmup_steps
            ):
                selection_qf1.load_state_dict(target_qf1.state_dict())
                selection_qf2.load_state_dict(target_qf2.state_dict())
                selection_critic_ready = True
            accumulated_count += 1
            for key, value in metrics.items():
                accumulated[key] = accumulated.get(
                    key, torch.zeros((), device=value.device)
                ) + value
            elapsed = elapsed_before + time.monotonic() - started
            validation = None
            if gradient_step == 1 or gradient_step % args.validation_interval == 0:
                validation_batch = dataset.sample_batch(
                    args.validation_batch_size,
                    np.random.default_rng(validation_rng_seed),
                    device,
                    indices=validation_indices,
                )
                validation = offline_validation_metrics(
                    policy,
                    qf1,
                    qf2,
                    vf,
                    validation_batch,
                    discount=args.discount,
                    expectile=args.expectile,
                    temperature=args.temperature,
                    max_advantage_weight=args.max_advantage_weight,
                    reward_scale=args.reward_scale,
                    reward_bias=args.reward_bias,
                )
                if selection_critic_ready:
                    selection = fixed_critic_selection_metrics(
                        policy,
                        selection_qf1,
                        selection_qf2,
                        validation_batch,
                        behavior_mse_penalty=(
                            args.selection_behavior_mse_penalty
                        ),
                    )
                    validation["checkpoint_selection"] = selection
                    if selection["score"] > best_offline_score:
                        best_offline_score = selection["score"]
                        _save(
                            output / "best_offline.pt",
                            checkpoint_payload(elapsed),
                        )

            if gradient_step == 1 or gradient_step % args.log_interval == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row = {
                    "method": METHOD,
                    "phase": (
                        "critic_warmup"
                        if gradient_step <= args.critic_warmup_steps
                        else "iql"
                    ),
                    "distillation_step": distillation_step,
                    "gradient_step": gradient_step,
                    "elapsed_seconds": elapsed,
                    "updates_per_second": (gradient_step - starting_gradient_step)
                    / max(time.monotonic() - started, 1e-12),
                    **{
                        key: float(value / accumulated_count)
                        for key, value in accumulated.items()
                    },
                    "action_std_mean": float(policy.log_std.exp().mean()),
                    "validation": validation,
                }
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                accumulated.clear()
                accumulated_count = 0

            if (
                gradient_step == 1
                or gradient_step % args.checkpoint_interval == 0
                or gradient_step == args.gradient_steps
            ):
                _save(output / "last.pt", checkpoint_payload(elapsed))
            if (
                gradient_step % args.checkpoint_interval == 0
                and gradient_step < args.gradient_steps
            ):
                _save(
                    output / f"recovery_step_{gradient_step:08d}.pt",
                    checkpoint_payload(elapsed),
                )
            if (
                args.max_wall_time_hours is not None
                and elapsed >= args.max_wall_time_hours * 3600.0
            ):
                status = "wall_time_reached"
                _save(output / "last.pt", checkpoint_payload(elapsed))
                break

        elapsed = elapsed_before + time.monotonic() - started
        _write_json(
            output / "training_status.json",
            {
                "state": status,
                "method": METHOD,
                "distillation_step": distillation_step,
                "gradient_step": gradient_step,
                "elapsed_seconds": elapsed,
                "best_offline_score": best_offline_score,
                "best_checkpoint": (
                    str(output / "best_offline.pt")
                    if (output / "best_offline.pt").is_file()
                    else None
                ),
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
            output / "training_status.json",
            {
                "state": "failed",
                "method": METHOD,
                "gradient_step": gradient_step,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


if __name__ == "__main__":
    main()
