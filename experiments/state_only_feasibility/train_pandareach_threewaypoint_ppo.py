"""Train a PandaReach3 actor with vectorized PPO.

Two initialization paths are supported:

* ``--bc-checkpoint`` (recommended): the policy is initialized from a
  BC-pretrained actor checkpoint and fine-tuned with a small actor learning
  rate. The BC dataset normalizers are reused and the PPO random init is
  skipped. This replaces the failed from-scratch H1-min actor-LR sweep, whose
  ``2e-6`` final-layer ``g_u`` gain pinned the policy mean near zero.
* from-scratch: every policy route is randomly initialized while the frozen
  Koopman lift is loaded from its dynamics checkpoint. The former ``2e-6``
  H1-min final-layer gain has been removed; H1-min from-scratch is no longer
  the intended path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor
from antmaze_ac.rl.quadratic_actors import (
    KoopmanLQRActor,
    LowRankValueActor,
)
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
    PandaReachThreeWaypointEnv,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_bc import (
    BCConfig,
    StandardPPOActor,
    TASK_CONTEXT_DIM,
    WAYPOINT_COUNT,
    _orthogonal_linear,
    load_koopman,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_ppo_smoke import (
    ValueNetwork,
)


ACTOR_NAMES = ("PPO", "KLQR", "AB-PQ", "BC-KMPC")
TRAINING_SPEC_VERSION = "dense_current_distance_waypoint_v2_scale005"


class StandardPPOValue(nn.Module):
    """Conventional raw-observation 256x256 value MLP for B0."""

    def __init__(self, state_dim: int, context_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + context_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        _orthogonal_linear(self.network[0], math.sqrt(2.0))
        _orthogonal_linear(self.network[2], math.sqrt(2.0))
        _orthogonal_linear(self.network[4], 1.0)

    def forward(
        self, state: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((state, context), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 3_000_000
    num_envs: int = 32
    rollout_steps: int = 256
    minibatch_size: int = 1024
    update_epochs: int = 8
    learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    anneal_learning_rate: bool = True
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    clip_value_loss: bool = True
    value_coefficient: float = 0.5
    entropy_coefficient: float = 1e-3
    initial_std_rad: float = 0.05
    minimum_std_rad: float = 1e-3
    maximum_std_rad: float = 0.2
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    checkpoint_interval_updates: int = 10
    max_wall_time_seconds: float | None = None
    max_episode_steps: int = 220
    goal_threshold: float = 0.01
    # Final-success radius (defaults to goal_threshold) and whether success
    # also requires a static robot. Relaxing these loosens only the final
    # success criterion, not waypoint passing.
    success_goal_threshold: float | None = None
    require_robot_static: bool = True
    reward_mode: str = "dense"
    dense_distance_penalty_scale: float = 0.05
    dense_waypoint_completion_reward: float = 1.0
    seed: int = 20_280_804

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.num_envs <= 0 or self.rollout_steps <= 0:
            raise ValueError("num_envs and rollout_steps must be positive")
        if not 0 < self.minibatch_size <= self.batch_size:
            raise ValueError("minibatch_size must be in [1, batch_size]")
        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        if self.learning_rate <= 0 or self.actor_learning_rate <= 0:
            raise ValueError("PPO learning rates must be positive")
        if not 0 < self.minimum_std_rad <= self.initial_std_rad <= self.maximum_std_rad:
            raise ValueError("PPO action std bounds are inconsistent")
        if self.checkpoint_interval_updates <= 0:
            raise ValueError("checkpoint_interval_updates must be positive")
        if self.reward_mode not in {"dense", "sparse"}:
            raise ValueError("reward_mode must be 'dense' or 'sparse'")
        if self.dense_distance_penalty_scale < 0:
            raise ValueError(
                "dense_distance_penalty_scale must be non-negative"
            )
        if self.dense_waypoint_completion_reward <= 0:
            raise ValueError(
                "dense_waypoint_completion_reward must be positive"
            )
        if (
            self.max_wall_time_seconds is not None
            and self.max_wall_time_seconds <= 0
        ):
            raise ValueError("max_wall_time_seconds must be positive when set")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _make_env(config: PPOConfig):
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    base = PandaArmOnlyActionWrapper(
        gym.make(
            "ACMPC-PandaReach3-v0",
            num_envs=config.num_envs,
            sim_backend="gpu" if torch.cuda.is_available() else "cpu",
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            reward_mode=config.reward_mode,
            render_mode=None,
            render_backend="none",
            max_episode_steps=config.max_episode_steps,
            goal_threshold=config.goal_threshold,
            success_goal_threshold=config.success_goal_threshold,
            require_robot_static=config.require_robot_static,
            dense_distance_penalty_scale=(
                config.dense_distance_penalty_scale
            ),
            dense_waypoint_completion_reward=(
                config.dense_waypoint_completion_reward
            ),
        )
    )
    return ManiSkillVectorEnv(
        base,
        config.num_envs,
        auto_reset=True,
        ignore_terminations=False,
        record_metrics=True,
    )


def _initial_context_normalizer(
    observation: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    waypoints = _as_tensor(observation["extra"]["waypoints"], device).float()
    flat = waypoints.reshape(waypoints.shape[0], -1)
    waypoint_center = flat.mean(0)
    waypoint_scale = flat.std(0, unbiased=False).clamp_min(1e-3)
    progress_center = torch.full((WAYPOINT_COUNT,), 1.0 / WAYPOINT_COUNT, device=device)
    progress_scale = torch.full(
        (WAYPOINT_COUNT,), math.sqrt(2.0) / 3.0, device=device
    )
    return (
        torch.cat((waypoint_center, progress_center)),
        torch.cat((waypoint_scale, progress_scale)),
    )


def _features(
    observation: dict[str, Any],
    koopman: nn.Module,
    state_center: torch.Tensor,
    state_scale: torch.Tensor,
    context_center: torch.Tensor,
    context_scale: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qpos = _as_tensor(observation["agent"]["qpos"], device).float()[..., :7]
    qvel = _as_tensor(observation["agent"]["qvel"], device).float()[..., :7]
    tcp = _as_tensor(observation["extra"]["tcp_pos"], device).float()
    state = torch.cat((qpos, qvel, tcp), dim=-1)
    normalized_state = (state - state_center) / state_scale
    waypoints = _as_tensor(observation["extra"]["waypoints"], device).float()
    active = _as_tensor(
        observation["extra"]["active_waypoint_index"], device
    ).long().reshape(-1)
    context_raw = torch.cat(
        (
            waypoints.reshape(waypoints.shape[0], -1),
            F.one_hot(active, WAYPOINT_COUNT).float(),
        ),
        dim=-1,
    )
    context = (context_raw - context_center) / context_scale
    with torch.no_grad():
        lifted = koopman.lift(normalized_state)
    return normalized_state, lifted, context


def _build_actor(
    name: str,
    koopman: Any,
    actor_config: BCConfig,
    device: torch.device,
) -> nn.Module:
    if name == "PPO":
        # Standard raw-state PPO actor (256x256 Tanh MLP, linear mean output,
        # no Koopman). Identical architecture in BC and PPO, so the
        # BC-pretrained weights transfer directly into fine-tuning.
        actor: nn.Module = StandardPPOActor(
            koopman.state_dim + TASK_CONTEXT_DIM,
            actor_config.ppo_hidden_dim,
            actor_config.action_limit_rad,
        )
    elif name == "KLQR":
        # KLQR: cost-map (lift + context) -> physical Q diag + p, mapped
        # through C and solved with the differentiable DARE into a
        # time-varying closed-loop gain u = -K z - d.
        actor = KoopmanLQRActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            context_dim=TASK_CONTEXT_DIM,
            hidden_dims=(actor_config.hidden_dim,),
            max_action=actor_config.action_limit_rad,
        )
    elif name == "AB-PQ":
        actor = LowRankValueActor(
            observation_dim=koopman.lifted_dim + TASK_CONTEXT_DIM,
            A=koopman.A,
            B=koopman.B,
            R=torch.eye(7, device=device, dtype=koopman.A.dtype),
            base_hessian=torch.eye(
                koopman.lifted_dim, device=device, dtype=koopman.A.dtype
            ),
            rank=actor_config.ab_rank,
            hidden_dims=(actor_config.hidden_dim,),
            value_linear_scale=1.0,
            max_action=actor_config.action_limit_rad,
        )
    elif name == "BC-KMPC":
        actor = KoopmanMPCActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            horizon=actor_config.kmpc_horizon,
            context_dim=TASK_CONTEXT_DIM,
            hidden_dims=(actor_config.hidden_dim,),
            action_low=-actor_config.action_limit_rad,
            action_high=actor_config.action_limit_rad,
            solver_iterations=actor_config.kmpc_solver_iterations,
        )
    else:
        raise ValueError(f"Unsupported actor {name!r}")
    return actor.to(device)


def _build_value(name: str, koopman: Any, device: torch.device) -> nn.Module:
    if name == "PPO":
        value: nn.Module = StandardPPOValue(
            koopman.state_dim, TASK_CONTEXT_DIM
        )
    else:
        value = ValueNetwork(koopman.lifted_dim + TASK_CONTEXT_DIM)
    return value.to(device)


def _initialize_ppo_modules(
    name: str,
    actor: nn.Module,
    value: nn.Module,
) -> None:
    """Apply PPO-style initialization to the value critic.

    The actor final layers are intentionally left at their constructor
    defaults (zero) instead of the former ``2e-6`` ``g_u`` gain: from-scratch
    H1-min PPO with that gain never escaped its near-zero policy. The
    recommended path is BC pretraining + fine-tuning via ``--bc-checkpoint``,
    which loads real policy weights and must not be overwritten here.

    All remaining routes (PPO, KLQR, AB-PQ, BC-KMPC) keep their constructor
    initialization; only the shared value critic is orthogonally initialized.
    """

    if name == "PPO":
        return
    _orthogonal_linear(value.network[0], math.sqrt(2.0))
    _orthogonal_linear(value.network[2], math.sqrt(2.0))
    _orthogonal_linear(value.network[4], 1.0)


def _value_estimate(
    name: str,
    value: nn.Module,
    normalized_state: torch.Tensor,
    lifted: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    if name == "PPO":
        return value(normalized_state, context)
    return value(lifted, context)


def _actor_mean(
    name: str,
    actor: nn.Module,
    normalized_state: torch.Tensor,
    lifted: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    if name == "PPO":
        return actor(normalized_state, context)
    if name == "KLQR":
        return actor(lifted, context).action
    if name == "AB-PQ":
        return actor(torch.cat((lifted, context), dim=-1), lifted).action
    return actor(lifted, context).action


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _optional_mean(values: deque[float]) -> float | None:
    return float(np.mean(values)) if values else None


def train(
    actor_name: str,
    koopman_path: Path,
    output_dir: Path,
    config: PPOConfig,
    device_name: str = "auto",
    resume: bool = True,
    bc_checkpoint: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    if actor_name not in ACTOR_NAMES:
        raise ValueError(f"Unsupported actor {actor_name!r}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    actor_config = BCConfig(seed=config.seed)
    koopman, koopman_payload = load_koopman(koopman_path, device)
    bc_payload: dict[str, Any] | None = None
    if bc_checkpoint is not None:
        bc_payload = torch.load(
            bc_checkpoint, map_location=device, weights_only=False
        )
        bc_name = str(bc_payload.get("name"))
        if bc_name != actor_name:
            raise ValueError(
                f"BC checkpoint actor {bc_name!r} does not match "
                f"requested actor {actor_name!r}"
            )
        if bc_payload.get("kind") != "pandareach_threewaypoint_bc_actor":
            raise ValueError(f"{bc_checkpoint} is not a PandaReach3 BC actor")
        actor = _build_actor(actor_name, koopman, actor_config, device)
        # The pretrained policy weights define the initialization; the
        # PPO-style random init must not overwrite them.
        actor.load_state_dict(bc_payload["actor_state"])
    else:
        actor = _build_actor(actor_name, koopman, actor_config, device)
    value = _build_value(actor_name, koopman, device)
    if bc_payload is None:
        _initialize_ppo_modules(actor_name, actor, value)
    log_std = nn.Parameter(
        torch.full((7,), math.log(config.initial_std_rad), device=device)
    )
    actor_parameters = list(actor.parameters())
    auxiliary_parameters = [*value.parameters(), log_std]
    parameters = [*actor_parameters, *auxiliary_parameters]
    optimizer = torch.optim.Adam(
        [
            {"params": actor_parameters, "lr": config.actor_learning_rate},
            {"params": auxiliary_parameters, "lr": config.learning_rate},
        ],
        eps=1e-5,
    )
    if bc_payload is not None:
        # BC trained with dataset-fitted state/context normalizers; reuse them
        # exactly so the fine-tuned policy sees the same input distribution.
        bc_normalizer = bc_payload["normalizer"]
        state_center = torch.as_tensor(
            bc_normalizer["state_center"], device=device, dtype=torch.float32
        )
        state_scale = torch.as_tensor(
            bc_normalizer["state_scale"], device=device, dtype=torch.float32
        )
        context_center = torch.as_tensor(
            bc_normalizer["context_center"], device=device, dtype=torch.float32
        )
        context_scale = torch.as_tensor(
            bc_normalizer["context_scale"], device=device, dtype=torch.float32
        )
    else:
        state_center = torch.as_tensor(
            koopman_payload["normalizer"]["center"],
            device=device,
            dtype=torch.float32,
        )
        state_scale = torch.as_tensor(
            koopman_payload["normalizer"]["scale"],
            device=device,
            dtype=torch.float32,
        )
    env = _make_env(config)
    observation, _ = env.reset(seed=config.seed)
    if bc_payload is None:
        context_center, context_scale = _initial_context_normalizer(
            observation, device
        )
    latest_path = output_dir / "latest.pt"
    if actor_name == "PPO":
        # The PPO route uses the identical StandardPPOActor in BC and PPO, so
        # the architecture version is the same for both initialization paths.
        architecture_version = "standard_raw_mlp_256x256_v1"
    else:
        architecture_version = (
            "structured_ppo_bc_v2"
            if bc_payload is not None
            else "structured_ppo_v1"
        )
    start_update = 0
    global_step = 0
    if resume and latest_path.exists():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        if payload["actor_name"] != actor_name:
            raise ValueError("Resume checkpoint actor does not match requested actor")
        if payload.get("architecture_version") != architecture_version:
            raise ValueError(
                "Resume checkpoint architecture is incompatible with the "
                f"current {architecture_version!r} implementation; use a new "
                "output directory or --no-resume"
            )
        if payload.get("training_spec_version") != TRAINING_SPEC_VERSION:
            raise ValueError(
                "Resume checkpoint uses an incompatible reward/training "
                "specification; use a new output directory or --no-resume"
            )
        actor.load_state_dict(payload["actor_state"])
        value.load_state_dict(payload["value_state"])
        log_std.data.copy_(payload["log_std"].to(device))
        optimizer.load_state_dict(payload["optimizer_state"])
        context_center = payload["context_center"].to(device)
        context_scale = payload["context_scale"].to(device)
        start_update = int(payload["update"])
        global_step = int(payload["global_step"])

    number_updates = math.ceil(config.total_timesteps / config.batch_size)
    metrics_path = output_dir / "metrics.jsonl"
    episode_returns: deque[float] = deque(maxlen=100)
    episode_lengths: deque[float] = deque(maxlen=100)
    episode_successes: deque[float] = deque(maxlen=100)
    episode_waypoints: deque[float] = deque(maxlen=100)
    started = time.perf_counter()
    wall_time_reached = False
    metadata = {
        "kind": (
            "pandareach_threewaypoint_ppo_from_scratch"
            if bc_payload is None
            else "pandareach_threewaypoint_ppo_bc_finetune"
        ),
        "actor_name": actor_name,
        "architecture_version": architecture_version,
        "training_spec_version": TRAINING_SPEC_VERSION,
        "bc_checkpoint": (
            str(bc_checkpoint.resolve()) if bc_checkpoint is not None else None
        ),
        "bc_sha256": (
            _sha256(bc_checkpoint) if bc_checkpoint is not None else None
        ),
        "bc_validation_mse": (
            float(bc_payload["report"]["best_validation_mse"])
            if bc_payload is not None
            else None
        ),
        "bc_closed_loop_success": (
            float(bc_payload["report"]["closed_loop"]["full_success_rate"])
            if bc_payload is not None
            else None
        ),
        "initialization": (
            (
                "BC-pretrained actor weights loaded from checkpoint; random "
                "value critic and log_std; frozen pretrained Koopman lift"
            )
            if bc_payload is not None
            else (
                "orthogonal random raw-state MLP actor and critic"
                if actor_name == "PPO"
                else "random actor; frozen pretrained Koopman lift"
            )
        ),
        "actor_input": (
            "normalize(x) + normalize(task_context)"
            if actor_name == "PPO"
            else "method-specific raw/lifted state + normalized task context"
        ),
        "critic_input": (
            "normalize(x) + normalize(task_context)"
            if actor_name == "PPO"
            else "frozen Koopman lift + normalized task context"
        ),
        "config": asdict(config),
        "actor_config": asdict(actor_config),
        "koopman_path": str(koopman_path.resolve()),
        "koopman_sha256": _sha256(koopman_path),
        "device": str(device),
        "torch_version": torch.__version__,
        "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
    }
    _atomic_json(output_dir / "run_config.json", metadata)

    try:
        for update in range(start_update + 1, number_updates + 1):
            if config.anneal_learning_rate:
                fraction = 1.0 - (update - 1.0) / number_updates
                optimizer.param_groups[0]["lr"] = (
                    fraction * config.actor_learning_rate
                )
                optimizer.param_groups[1]["lr"] = fraction * config.learning_rate
            normalized_states: list[torch.Tensor] = []
            lifted_states: list[torch.Tensor] = []
            contexts: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []
            log_probabilities: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            dones: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            rollout_started = time.perf_counter()
            for _ in range(config.rollout_steps):
                normalized_state, lifted, context = _features(
                    observation,
                    koopman,
                    state_center,
                    state_scale,
                    context_center,
                    context_scale,
                    device,
                )
                with torch.no_grad():
                    mean = _actor_mean(
                        actor_name, actor, normalized_state, lifted, context
                    )
                    distribution = Normal(mean, log_std.exp().expand_as(mean))
                    action = distribution.sample()
                    log_probability = distribution.log_prob(action).sum(-1)
                    state_value = _value_estimate(
                        actor_name,
                        value,
                        normalized_state,
                        lifted,
                        context,
                    )
                next_observation, reward, terminated, truncated, info = env.step(
                    torch.clamp(
                        action / actor_config.action_limit_rad, -1.0, 1.0
                    )
                )
                reward_tensor = _as_tensor(reward, device).float().reshape(-1)
                done = torch.logical_or(
                    _as_tensor(terminated, device).bool().reshape(-1),
                    _as_tensor(truncated, device).bool().reshape(-1),
                )
                if bool(done.any()):
                    final_info = info.get("final_info", info)
                    episode = final_info.get("episode", {})
                    for item in _as_tensor(
                        episode.get("return", reward_tensor), device
                    ).reshape(-1)[done].tolist():
                        episode_returns.append(float(item))
                    for item in _as_tensor(
                        episode.get(
                            "episode_len",
                            torch.ones(config.num_envs, device=device),
                        ),
                        device,
                    ).reshape(-1)[done].tolist():
                        episode_lengths.append(float(item))
                    success = final_info.get(
                        "success", torch.zeros(config.num_envs, device=device)
                    )
                    for item in _as_tensor(success, device).float().reshape(-1)[done].tolist():
                        episode_successes.append(float(item))
                    completed = final_info.get(
                        "waypoints_completed",
                        torch.zeros(config.num_envs, device=device),
                    )
                    for item in _as_tensor(completed, device).float().reshape(-1)[done].tolist():
                        episode_waypoints.append(float(item))
                normalized_states.append(normalized_state.detach())
                lifted_states.append(lifted.detach())
                contexts.append(context.detach())
                actions.append(action.detach())
                log_probabilities.append(log_probability.detach())
                rewards.append(reward_tensor)
                dones.append(done.float())
                values.append(state_value.detach())
                observation = next_observation
            rollout_seconds = time.perf_counter() - rollout_started

            with torch.no_grad():
                next_normalized_state, next_lifted, next_context = _features(
                    observation,
                    koopman,
                    state_center,
                    state_scale,
                    context_center,
                    context_scale,
                    device,
                )
                next_value = _value_estimate(
                    actor_name,
                    value,
                    next_normalized_state,
                    next_lifted,
                    next_context,
                )
            state_batch = torch.stack(normalized_states)
            lifted_batch = torch.stack(lifted_states)
            context_batch = torch.stack(contexts)
            action_batch = torch.stack(actions)
            old_log_prob = torch.stack(log_probabilities)
            reward_batch = torch.stack(rewards)
            done_batch = torch.stack(dones)
            old_value = torch.stack(values)
            advantages = torch.zeros_like(reward_batch)
            last_advantage = torch.zeros(config.num_envs, device=device)
            for step in range(config.rollout_steps - 1, -1, -1):
                following_value = next_value if step == config.rollout_steps - 1 else old_value[step + 1]
                nonterminal = 1.0 - done_batch[step]
                delta = (
                    reward_batch[step]
                    + config.discount * following_value * nonterminal
                    - old_value[step]
                )
                last_advantage = delta + (
                    config.discount * config.gae_lambda * nonterminal * last_advantage
                )
                advantages[step] = last_advantage
            returns = advantages + old_value

            flat_state = state_batch.flatten(0, 1)
            flat_lifted = lifted_batch.flatten(0, 1)
            flat_context = context_batch.flatten(0, 1)
            flat_action = action_batch.flatten(0, 1)
            flat_old_log_prob = old_log_prob.flatten()
            flat_old_value = old_value.flatten()
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()
            update_started = time.perf_counter()
            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            approximate_kls: list[float] = []
            observed_pre_update_kls: list[float] = []
            clip_fractions: list[float] = []
            stopped_early = False
            stop_kl: float | None = None
            for _ in range(config.update_epochs):
                order = torch.randperm(config.batch_size, device=device)
                for start in range(0, config.batch_size, config.minibatch_size):
                    index = order[start : start + config.minibatch_size]
                    advantage = flat_advantages[index]
                    advantage = (advantage - advantage.mean()) / (
                        advantage.std(unbiased=False) + 1e-8
                    )
                    mean = _actor_mean(
                        actor_name,
                        actor,
                        flat_state[index],
                        flat_lifted[index],
                        flat_context[index],
                    )
                    distribution = Normal(mean, log_std.exp().expand_as(mean))
                    new_log_probability = distribution.log_prob(
                        flat_action[index]
                    ).sum(-1)
                    log_ratio = new_log_probability - flat_old_log_prob[index]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        pre_update_kl = ((ratio - 1.0) - log_ratio).mean()
                    observed_pre_update_kls.append(float(pre_update_kl))
                    if pre_update_kl > config.target_kl:
                        stopped_early = True
                        stop_kl = float(pre_update_kl)
                        break
                    policy_loss = -torch.minimum(
                        ratio * advantage,
                        torch.clamp(
                            ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                        )
                        * advantage,
                    ).mean()
                    predicted_value = _value_estimate(
                        actor_name,
                        value,
                        flat_state[index],
                        flat_lifted[index],
                        flat_context[index],
                    )
                    if config.clip_value_loss:
                        unclipped_value_loss = (predicted_value - flat_returns[index]).square()
                        clipped_value = flat_old_value[index] + torch.clamp(
                            predicted_value - flat_old_value[index],
                            -config.clip_ratio,
                            config.clip_ratio,
                        )
                        clipped_value_loss = (clipped_value - flat_returns[index]).square()
                        value_loss = 0.5 * torch.maximum(
                            unclipped_value_loss, clipped_value_loss
                        ).mean()
                    else:
                        value_loss = 0.5 * (
                            predicted_value - flat_returns[index]
                        ).square().mean()
                    entropy = distribution.entropy().sum(-1).mean()
                    loss = (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, config.max_grad_norm
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("PPO produced a non-finite gradient")
                    optimizer.step()
                    with torch.no_grad():
                        log_std.clamp_(
                            math.log(config.minimum_std_rad),
                            math.log(config.maximum_std_rad),
                        )
                        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fraction = (
                            (ratio - 1.0).abs() > config.clip_ratio
                        ).float().mean()
                    policy_losses.append(float(policy_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                    entropies.append(float(entropy.detach()))
                    approximate_kls.append(float(approximate_kl.detach()))
                    clip_fractions.append(float(clip_fraction.detach()))
                if stopped_early:
                    break
                if approximate_kls[-1] > config.target_kl:
                    stopped_early = True
                    break
            update_seconds = time.perf_counter() - update_started
            global_step = min(update * config.batch_size, config.total_timesteps)
            value_variance = float(flat_returns.var(unbiased=False))
            explained_variance = (
                None
                if value_variance <= 1e-12
                else 1.0
                - float((flat_returns - flat_old_value).var(unbiased=False))
                / value_variance
            )
            report = {
                "update": update,
                "global_step": global_step,
                "actor_learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rate": optimizer.param_groups[1]["lr"],
                "rollout_reward_mean": float(reward_batch.mean()),
                "recent_episode_return": _optional_mean(episode_returns),
                "recent_episode_length": _optional_mean(episode_lengths),
                "recent_full_success_rate": _optional_mean(episode_successes),
                "recent_waypoints_completed": _optional_mean(episode_waypoints),
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "approximate_kl": float(np.mean(approximate_kls)),
                "maximum_observed_kl": float(np.max(observed_pre_update_kls)),
                "early_stop_kl": stop_kl,
                "clip_fraction": float(np.mean(clip_fractions)),
                "explained_variance": explained_variance,
                "log_std": log_std.detach().cpu().tolist(),
                "rollout_seconds": rollout_seconds,
                "ppo_update_seconds": update_seconds,
                "transitions_per_second": config.batch_size
                / (rollout_seconds + update_seconds),
                "target_kl_early_stop": stopped_early,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, sort_keys=True) + "\n")
            checkpoint = {
                **metadata,
                "actor_state": actor.state_dict(),
                "value_state": value.state_dict(),
                "log_std": log_std.detach(),
                "optimizer_state": optimizer.state_dict(),
                "context_center": context_center.detach(),
                "context_scale": context_scale.detach(),
                "update": update,
                "global_step": global_step,
                "last_report": report,
            }
            wall_time_reached = (
                config.max_wall_time_seconds is not None
                and report["elapsed_seconds"] >= config.max_wall_time_seconds
            )
            if (
                update % config.checkpoint_interval_updates == 0
                or update == number_updates
                or wall_time_reached
            ):
                _atomic_torch_save(latest_path, checkpoint)
                _atomic_torch_save(output_dir / f"checkpoint_{global_step:09d}.pt", checkpoint)
            _atomic_json(
                output_dir / "status.json",
                {
                    "state": (
                        "wall_time_reached" if wall_time_reached else "running"
                    ),
                    **report,
                    "pid": os.getpid(),
                },
            )
            print(json.dumps(report, sort_keys=True), flush=True)
            if wall_time_reached:
                break
        final = {
            "state": "wall_time_reached" if wall_time_reached else "complete",
            **report,
            "pid": os.getpid(),
        }
        _atomic_json(output_dir / "status.json", final)
        return final
    except BaseException as error:
        _atomic_json(
            output_dir / "status.json",
            {
                "state": "failed",
                "actor_name": actor_name,
                "global_step": global_step,
                "error": f"{type(error).__name__}: {error}",
                "pid": os.getpid(),
            },
        )
        raise
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", choices=ACTOR_NAMES, required=True)
    parser.add_argument(
        "--koopman",
        type=Path,
        default=Path("runs/pandareach_threewaypoint/koopman/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=None,
        help=(
            "BC-pretrained actor checkpoint to initialize from and "
            "fine-tune (recommended path; skips random PPO init and uses "
            "the BC dataset normalizers)."
        ),
    )
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coefficient", type=float, default=1e-3)
    parser.add_argument("--initial-std-rad", type=float, default=0.05)
    parser.add_argument(
        "--reward-mode", choices=("dense", "sparse"), default="dense"
    )
    parser.add_argument(
        "--dense-distance-penalty-scale", type=float, default=0.05
    )
    parser.add_argument(
        "--dense-waypoint-completion-reward", type=float, default=1.0
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=0.01,
        help="Waypoint-passing and default success radius in meters.",
    )
    parser.add_argument(
        "--success-goal-threshold",
        type=float,
        default=None,
        help=(
            "Final-success radius in meters; defaults to --goal-threshold. "
            "Only affects the final success criterion, not waypoint passing."
        ),
    )
    parser.add_argument(
        "--no-require-robot-static",
        action="store_true",
        help=(
            "Do not require the robot to be static for final success; "
            "makes the success criterion more lenient."
        ),
    )
    parser.add_argument("--checkpoint-interval-updates", type=int, default=10)
    parser.add_argument("--max-wall-time-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20_280_804)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The tiny per-method actor learning rates were calibrated for
    # from-scratch initialization. When fine-tuning from a BC-pretrained
    # policy, use a normal PPO actor LR instead (still overridable).
    if args.bc_checkpoint is not None:
        default_actor_learning_rates = {
            name: 1e-4 for name in ACTOR_NAMES
        }
    else:
        default_actor_learning_rates = {
            "PPO": 3e-4,
            "KLQR": 3e-5,
            "AB-PQ": 3e-5,
            "BC-KMPC": 3e-6,
        }
    result = train(
        args.actor,
        args.koopman,
        args.output_dir,
        PPOConfig(
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            minibatch_size=args.minibatch_size,
            update_epochs=args.update_epochs,
            learning_rate=args.learning_rate,
            actor_learning_rate=(
                args.actor_learning_rate
                if args.actor_learning_rate is not None
                else default_actor_learning_rates[args.actor]
            ),
            entropy_coefficient=args.entropy_coefficient,
            initial_std_rad=args.initial_std_rad,
            reward_mode=args.reward_mode,
            dense_distance_penalty_scale=(
                args.dense_distance_penalty_scale
            ),
            dense_waypoint_completion_reward=(
                args.dense_waypoint_completion_reward
            ),
            goal_threshold=args.goal_threshold,
            success_goal_threshold=args.success_goal_threshold,
            require_robot_static=not args.no_require_robot_static,
            checkpoint_interval_updates=args.checkpoint_interval_updates,
            max_wall_time_seconds=(
                None
                if args.max_wall_time_minutes is None
                else 60.0 * args.max_wall_time_minutes
            ),
            seed=args.seed,
        ),
        args.device,
        not args.no_resume,
        args.bc_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
