"""Train and evaluate five BC actors on the ordered PandaReach3 task."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from antmaze_ac.koopman.model import DeepKoopman
from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor
from antmaze_ac.rl.quadratic_actors import (
    KoopmanLQRActor,
    LowRankValueActor,
)
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
    PandaReachThreeWaypointEnv,
)


@dataclass(frozen=True)
class BCConfig:
    epochs: int = 250
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    hidden_dim: int = 128
    ppo_hidden_dim: int = 256
    ab_rank: int = 4
    kmpc_horizon: int = 10
    kmpc_solver_iterations: int = 20
    kmpc_sequence_weight: float = 0.25
    action_limit_rad: float = 0.1
    seed: int = 47
    validation_interval: int = 5
    early_stopping_patience: int = 50
    # Persist the best actor checkpoint (name.pt) at least every N epochs so a
    # killed/interrupted run still keeps a usable model.
    checkpoint_interval: int = 10
    evaluation_episodes: int = 100
    # Parallel vectorized closed-loop evaluation: N envs are stepped together
    # so per-sample actor cost (e.g. the KLQR DARE) is batched, not batch-1.
    evaluation_num_envs: int = 16
    evaluation_seed_start: int = 20_270_804
    max_episode_steps: int = 220
    goal_threshold: float = 0.01


def _orthogonal_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class StandardPPOActor(nn.Module):
    """Standard continuous-PPO Gaussian-mean MLP (no Koopman lift).

    The input is the concatenation of the normalized raw robot state and the
    normalized task context. The 256x256 Tanh MLP outputs a LINEAR mean scaled
    by ``action_limit`` (no tanh on the output); the environment clips the
    sampled action to bounds. This is the standard PPO flow, used identically
    by BC pretraining and PPO fine-tuning for the ``PPO`` route so that the
    pretrained weights transfer directly.
    """

    def __init__(self, input_dim: int, hidden_dim: int, action_limit: float):
        super().__init__()
        self.action_limit = float(action_limit)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 7),
        )
        _orthogonal_linear(self.network[0], math.sqrt(2.0))
        _orthogonal_linear(self.network[2], math.sqrt(2.0))
        _orthogonal_linear(self.network[4], 0.01)

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Standard PPO: linear mean, no output tanh. The environment clips the
        # sampled action to [-action_limit, action_limit].
        return self.action_limit * self.network(
            torch.cat((state, context), dim=-1)
        )


TASK_CONTEXT_DIM = 12
WAYPOINT_COUNT = 3
# Canonical training order. The seed offset for each actor is its index here,
# so parallel per-actor runs reproduce the sequential "all actors" seeding.
BC_ACTOR_ORDER = ("PPO", "KLQR", "AB-PQ", "BC-KMPC")


def _task_context(data: dict[str, np.ndarray]) -> np.ndarray:
    """Build [G1,G2,G3, one-hot(active waypoint)] for every transition."""

    episode_waypoints = data["episode_waypoints"][data["episode_id"]]
    active_index = data["active_waypoint_index"].astype(np.int64)
    if episode_waypoints.shape[1:] != (WAYPOINT_COUNT, 3):
        raise ValueError("Expected three XYZ waypoints per episode")
    if np.any((active_index < 0) | (active_index >= WAYPOINT_COUNT)):
        raise ValueError("active waypoint index is outside the task range")
    progress = np.eye(WAYPOINT_COUNT, dtype=np.float32)[active_index]
    return np.concatenate(
        (episode_waypoints.reshape(len(episode_waypoints), -1), progress),
        axis=-1,
    ).astype(np.float32)


def load_koopman(path: Path, device: torch.device) -> tuple[DeepKoopman, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint["architecture"]
    model = DeepKoopman(
        state_dim=int(architecture["state_dim"]),
        action_dim=int(architecture["action_dim"]),
        lift_dim=int(architecture["lift_dim"]),
        hidden_dims=tuple(architecture["hidden_dims"]),
        activation=str(architecture.get("activation", "silu")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.freeze_dynamics()
    if checkpoint.get("state_kind") != "q_qdot_tcp":
        raise ValueError("BC requires a q_qdot_tcp Koopman checkpoint")
    return model, checkpoint


def _future_targets(
    data: dict[str, np.ndarray], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    count = len(data["action"])
    future = np.zeros((count, horizon, 7), dtype=np.float32)
    mask = np.zeros((count, horizon), dtype=np.float32)
    episode = data["episode_id"]
    stage = data["active_waypoint_index"]
    for index in range(count):
        for offset in range(horizon):
            following = index + offset
            if (
                following >= count
                or episode[following] != episode[index]
                or stage[following] != stage[index]
            ):
                break
            future[index, offset] = data["action"][following]
            mask[index, offset] = 1.0
    return future, mask


def _actor_action(
    name: str,
    actor: nn.Module,
    normalized_state: torch.Tensor,
    lifted: torch.Tensor,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if name == "PPO":
        # PPO is the standard raw-state baseline: it deliberately does not
        # use the Koopman lift.  The other routes receive ``lifted`` below.
        return actor(normalized_state, context), None
    if name == "KLQR":
        # KLQR: cost-map (lift + context) -> Q_diag, p; differentiable DARE
        # gives the time-varying closed-loop gain; no action sequence.
        return actor(lifted, context).action, None
    if name == "AB-PQ":
        output = actor(torch.cat((lifted, context), dim=-1), lifted)
        return output.action, None
    output = actor(lifted, context)
    return output.action, output.action_sequence


def _evaluate_mse(
    name: str,
    actor: nn.Module,
    koopman: DeepKoopman,
    loader: DataLoader,
    device: torch.device,
) -> float:
    total = 0.0
    elements = 0
    actor.eval()
    with torch.no_grad():
        for state, goal, action, _, _ in loader:
            state, goal, action = state.to(device), goal.to(device), action.to(device)
            lifted = koopman.lift(state)
            prediction, _ = _actor_action(name, actor, state, lifted, goal)
            total += float((prediction - action).square().sum())
            elements += action.numel()
    return total / elements


def _make_loader(
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    future: np.ndarray,
    future_mask: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    tensors = tuple(
        torch.from_numpy(value[mask].astype(np.float32))
        for value in (state, goal, action, future, future_mask)
    )
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=min(batch_size, int(mask.sum())),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Write a torch checkpoint atomically (temp file + rename)."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _train_actor(
    name: str,
    actor: nn.Module,
    koopman: DeepKoopman,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: BCConfig,
    device: torch.device,
    checkpoint_payload_base: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    last_improvement_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        actor.train()
        for state, goal, action, future, future_mask in train_loader:
            state, goal, action = state.to(device), goal.to(device), action.to(device)
            future, future_mask = future.to(device), future_mask.to(device)
            with torch.no_grad():
                lifted = koopman.lift(state)
            prediction, sequence = _actor_action(
                name, actor, state, lifted, goal
            )
            loss = (prediction - action).square().mean()
            if sequence is not None and sequence.shape[-2] > 1:
                errors = (sequence[:, 1:] - future[:, 1:]).square().mean(-1)
                valid = future_mask[:, 1:]
                future_loss = (errors * valid).sum() / valid.sum().clamp_min(1.0)
                loss = loss + config.kmpc_sequence_weight * future_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"{name} produced a non-finite gradient")
            optimizer.step()
        should_validate = (
            epoch == 1
            or epoch % config.validation_interval == 0
            or epoch == config.epochs
        )
        validation = float("nan")
        if should_validate:
            validation = _evaluate_mse(
                name, actor, koopman, validation_loader, device
            )
            if validation < best_validation:
                best_validation = validation
                best_epoch = epoch
                last_improvement_epoch = epoch
                best_state = copy.deepcopy(actor.state_dict())
        if epoch == 1 or epoch % 50 == 0 or epoch == config.epochs:
            print(
                f"actor={name} epoch={epoch:04d} val_mse={validation:.6g}",
                flush=True,
            )
        # Periodic best-checkpoint persistence so an interrupted run keeps a
        # usable model (overwritten by the full checkpoint on completion).
        if (
            output_dir is not None
            and checkpoint_payload_base is not None
            and best_state is not None
            and (epoch % config.checkpoint_interval == 0 or epoch == config.epochs)
        ):
            _atomic_torch_save(
                output_dir / f"{name}.pt",
                {
                    **checkpoint_payload_base,
                    "actor_state": best_state,
                    "report": {
                        "best_epoch": best_epoch,
                        "completed_epochs": epoch,
                        "best_validation_mse": best_validation,
                        "checkpoint_partial": epoch < config.epochs,
                    },
                },
            )
            (output_dir / f"{name}.status.json").write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "completed_epochs": epoch,
                        "best_validation_mse": best_validation,
                        "pid": os.getpid(),
                    }
                ),
                encoding="utf-8",
            )
        if (
            should_validate
            and epoch - last_improvement_epoch
            >= config.early_stopping_patience
        ):
            print(
                f"actor={name} early_stop={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break
    if best_state is None:
        raise RuntimeError(f"{name} training did not produce a checkpoint")
    actor.load_state_dict(best_state)
    return actor, {
        "best_epoch": best_epoch,
        "completed_epochs": epoch,
        "best_validation_mse": best_validation,
        "training_seconds": time.perf_counter() - started,
    }


def _normalizers(
    data: dict[str, np.ndarray], train_mask: np.ndarray, checkpoint: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _cpu_array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    center = _cpu_array(checkpoint["normalizer"]["center"])
    scale = _cpu_array(checkpoint["normalizer"]["scale"])
    contexts = _task_context(data)[train_mask].astype(np.float64)
    context_center = contexts.mean(0).astype(np.float32)
    context_scale = np.maximum(contexts.std(0), 1e-4).astype(np.float32)
    return center, scale, context_center, context_scale


def _batch_features(
    observation: dict[str, Any],
    koopman: DeepKoopman,
    center: np.ndarray,
    scale: np.ndarray,
    context_center: np.ndarray,
    context_scale: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched [q, qdot, tcp] features + task context from a vector obs."""

    qpos = torch.as_tensor(
        observation["agent"]["qpos"], device=device, dtype=torch.float32
    )[..., :7]
    qvel = torch.as_tensor(
        observation["agent"]["qvel"], device=device, dtype=torch.float32
    )[..., :7]
    tcp = torch.as_tensor(
        observation["extra"]["tcp_pos"], device=device, dtype=torch.float32
    )
    state = torch.cat((qpos, qvel, tcp), dim=-1)
    normalized_state = (
        state - torch.as_tensor(center, device=device)
    ) / torch.as_tensor(scale, device=device)
    waypoints = torch.as_tensor(
        observation["extra"]["waypoints"], device=device, dtype=torch.float32
    )
    active = torch.as_tensor(
        observation["extra"]["active_waypoint_index"], device=device
    ).long().reshape(-1)
    context_raw = torch.cat(
        (
            waypoints.reshape(waypoints.shape[0], -1),
            F.one_hot(active, WAYPOINT_COUNT).float(),
        ),
        dim=-1,
    )
    context = (
        context_raw - torch.as_tensor(context_center, device=device)
    ) / torch.as_tensor(context_scale, device=device)
    with torch.no_grad():
        lifted = koopman.lift(normalized_state)
    return normalized_state, lifted, context


def closed_loop_evaluation(
    name: str,
    actor: nn.Module,
    koopman: DeepKoopman,
    center: np.ndarray,
    scale: np.ndarray,
    context_center: np.ndarray,
    context_scale: np.ndarray,
    config: BCConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Vectorized closed-loop evaluation over parallel GPU envs.

    ``evaluation_num_envs`` envs are stepped together so per-sample actor
    costs (e.g. the KLQR differentiable DARE) are batched instead of batch-1;
    this is what makes structured actors (KLQR in particular) evaluate in
    seconds-to-a-minute rather than ~16 minutes.
    """

    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    num_envs = min(config.evaluation_episodes, config.evaluation_num_envs)
    base = PandaArmOnlyActionWrapper(
        gym.make(
            "ACMPC-PandaReach3-v0",
            num_envs=num_envs,
            sim_backend="gpu" if device.type == "cuda" else "cpu",
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode=None,
            render_backend="none",
            max_episode_steps=config.max_episode_steps,
            goal_threshold=config.goal_threshold,
        )
    )
    env = ManiSkillVectorEnv(
        base,
        num_envs,
        auto_reset=True,
        ignore_terminations=False,
        record_metrics=True,
    )
    successes = 0
    completed: list[int] = []
    final_distances: list[float] = []
    lengths: list[int] = []
    bound_fractions: list[float] = []
    episodes_done = 0
    actor.eval()
    try:
        observation, _ = env.reset(seed=config.evaluation_seed_start)
        bound_counts = torch.zeros(num_envs, device=device)
        action_counts = torch.zeros(num_envs, device=device)
        while episodes_done < config.evaluation_episodes:
            normalized_state, lifted, context = _batch_features(
                observation,
                koopman,
                center,
                scale,
                context_center,
                context_scale,
                device,
            )
            with torch.no_grad():
                action, _ = _actor_action(
                    name, actor, normalized_state, lifted, context
                )
            action_rad = action
            bound_counts += (
                action_rad.abs() >= 0.99 * config.action_limit_rad
            ).sum(-1).float()
            action_counts += 7
            observation, _, terminated, truncated, info = env.step(
                torch.clamp(
                    action_rad / config.action_limit_rad, -1.0, 1.0
                )
            )
            done = torch.logical_or(
                torch.as_tensor(terminated, device=device).bool().reshape(-1),
                torch.as_tensor(truncated, device=device).bool().reshape(-1),
            )
            if bool(done.any()):
                final_info = info.get("final_info", info)
                episode = final_info.get("episode", {})
                lengths_t = torch.as_tensor(
                    episode.get(
                        "episode_len", torch.ones(num_envs, device=device)
                    ),
                    device=device,
                ).reshape(-1)
                success_t = torch.as_tensor(
                    final_info.get(
                        "success", torch.zeros(num_envs, device=device)
                    ),
                    device=device,
                ).float().reshape(-1)
                completed_t = torch.as_tensor(
                    final_info.get(
                        "waypoints_completed",
                        torch.zeros(num_envs, device=device),
                    ),
                    device=device,
                ).reshape(-1)
                distance_t = torch.as_tensor(
                    final_info.get(
                        "active_waypoint_distance",
                        torch.zeros(num_envs, device=device),
                    ),
                    device=device,
                ).float().reshape(-1)
                for index in done.nonzero(as_tuple=False).flatten().tolist():
                    if episodes_done >= config.evaluation_episodes:
                        break
                    episodes_done += 1
                    successes += int(success_t[index])
                    completed.append(int(completed_t[index]))
                    final_distances.append(float(distance_t[index]))
                    lengths.append(int(lengths_t[index]))
                    bound_fractions.append(
                        float(
                            bound_counts[index]
                            / max(int(action_counts[index]), 1)
                        )
                    )
                    bound_counts[index] = 0
                    action_counts[index] = 0
                if episodes_done >= config.evaluation_episodes:
                    break
    finally:
        env.close()
    return {
        "episodes": config.evaluation_episodes,
        "full_success_rate": successes / config.evaluation_episodes,
        "mean_waypoints_completed": float(np.mean(completed)),
        "waypoints_completed_histogram": {
            str(index): int(np.count_nonzero(np.asarray(completed) == index))
            for index in range(4)
        },
        "mean_final_active_distance_m": float(np.mean(final_distances)),
        "mean_episode_length": float(np.mean(lengths)),
        "action_bound_fraction": float(np.mean(bound_fractions)),
    }


def _make_builders(
    koopman: DeepKoopman,
    config: BCConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Actor constructors shared by BC training and checkpoint evaluation."""

    return {
        "PPO": lambda: StandardPPOActor(
            koopman.state_dim + TASK_CONTEXT_DIM,
            config.ppo_hidden_dim,
            config.action_limit_rad,
        ),
        "KLQR": lambda: KoopmanLQRActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            context_dim=TASK_CONTEXT_DIM,
            hidden_dims=(config.hidden_dim,),
            max_action=config.action_limit_rad,
        ),
        "AB-PQ": lambda: LowRankValueActor(
            observation_dim=koopman.lifted_dim + TASK_CONTEXT_DIM,
            A=koopman.A,
            B=koopman.B,
            R=torch.eye(7, device=device, dtype=koopman.A.dtype),
            base_hessian=torch.eye(
                koopman.lifted_dim, device=device, dtype=koopman.A.dtype
            ),
            rank=config.ab_rank,
            hidden_dims=(config.hidden_dim,),
            value_linear_scale=1.0,
            max_action=config.action_limit_rad,
        ),
        "BC-KMPC": lambda: KoopmanMPCActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            horizon=config.kmpc_horizon,
            context_dim=TASK_CONTEXT_DIM,
            hidden_dims=(config.hidden_dim,),
            action_low=-config.action_limit_rad,
            action_high=config.action_limit_rad,
            solver_iterations=config.kmpc_solver_iterations,
        ),
    }


def evaluate_checkpoint(
    output_dir: Path,
    actor_name: str,
    koopman_path: Path,
    config: BCConfig,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the vectorized closed-loop evaluation on a saved BC actor only.

    Used for manual re-evaluation after training (e.g. with a faster batched
    evaluation) without retraining.
    """

    if actor_name not in BC_ACTOR_ORDER:
        raise ValueError(f"Unknown actor {actor_name!r}")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    payload = torch.load(
        output_dir / f"{actor_name}.pt", map_location=device, weights_only=False
    )
    if payload.get("kind") != "pandareach_threewaypoint_bc_actor":
        raise ValueError(f"{output_dir / (actor_name + '.pt')} is not a BC actor")
    if payload.get("name") != actor_name:
        raise ValueError(f"Checkpoint actor name mismatch: {payload.get('name')!r}")
    koopman, _ = load_koopman(koopman_path, device)
    actor = _make_builders(koopman, config, device)[actor_name]().to(device)
    actor.load_state_dict(payload["actor_state"])
    normalizer = payload["normalizer"]
    result = closed_loop_evaluation(
        actor_name,
        actor,
        koopman,
        np.asarray(normalizer["state_center"], dtype=np.float32),
        np.asarray(normalizer["state_scale"], dtype=np.float32),
        np.asarray(normalizer["context_center"], dtype=np.float32),
        np.asarray(normalizer["context_scale"], dtype=np.float32),
        config,
        device,
    )
    print(json.dumps({actor_name: result}, sort_keys=True), flush=True)
    return result


def train(
    dataset_path: Path,
    koopman_path: Path,
    output_dir: Path,
    config: BCConfig,
    device_name: str = "auto",
    actor_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    koopman, checkpoint = load_koopman(koopman_path, device)
    episode = data["episode_id"]
    masks = {
        split: np.isin(episode, data[f"{split}_episode_ids"])
        for split in ("train", "validation", "test")
    }
    center, scale, context_center, context_scale = _normalizers(
        data, masks["train"], checkpoint
    )
    normalized_state = ((data["state"] - center) / scale).astype(np.float32)
    normalized_context = (
        (_task_context(data) - context_center) / context_scale
    ).astype(np.float32)
    future, future_mask = _future_targets(data, config.kmpc_horizon)
    loaders = {
        split: _make_loader(
            normalized_state,
            normalized_context,
            data["action"],
            future,
            future_mask,
            masks[split],
            config.batch_size,
            split == "train",
            config.seed,
        )
        for split in ("train", "validation", "test")
    }
    builders = _make_builders(koopman, config, device)
    if actor_names is not None:
        unknown = sorted(set(actor_names) - set(builders))
        if unknown:
            raise ValueError(f"Unknown actor names: {unknown}")
        builders = {name: builders[name] for name in actor_names}
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for name, builder in builders.items():
        # Deterministic per-actor seed: same whether run sequentially or as
        # separate parallel processes.
        torch.manual_seed(config.seed + BC_ACTOR_ORDER.index(name))
        actor = builder().to(device)
        # Shared checkpoint fields known before training; the actor_state and
        # full report are attached at save time (and periodically during
        # training so an interrupted run keeps a usable model).
        checkpoint_payload_base = {
            "kind": "pandareach_threewaypoint_bc_actor",
            "name": name,
            "config": asdict(config),
            "normalizer": {
                "state_center": center,
                "state_scale": scale,
                "context_center": context_center,
                "context_scale": context_scale,
            },
            "koopman_path": str(koopman_path),
        }
        actor, actor_report = _train_actor(
            name,
            actor,
            koopman,
            loaders["train"],
            loaders["validation"],
            config,
            device,
            checkpoint_payload_base,
            output_dir,
        )
        actor_report.update(
            {
                "test_mse": _evaluate_mse(
                    name, actor, koopman, loaders["test"], device
                ),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in actor.parameters()
                    if parameter.requires_grad
                ),
                "closed_loop": closed_loop_evaluation(
                    name,
                    actor,
                    koopman,
                    center,
                    scale,
                    context_center,
                    context_scale,
                    config,
                    device,
                ),
            }
        )
        checkpoint_payload = {
            **checkpoint_payload_base,
            "actor_state": actor.state_dict(),
            "report": actor_report,
        }
        _atomic_torch_save(output_dir / f"{name}.pt", checkpoint_payload)
        # Per-actor report, written independently so parallel processes can
        # share one output directory without racing on report.json.
        (output_dir / f"{name}.report.json").write_text(
            json.dumps(actor_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        reports[name] = actor_report
        print(json.dumps({name: actor_report}, sort_keys=True), flush=True)
    meta = {
        "kind": "pandareach_threewaypoint_bc_comparison",
        "dataset_path": str(dataset_path),
        "koopman_path": str(koopman_path),
        "config": asdict(config),
        "actor_input": (
            "PPO: [normalize(x), normalize(c)]; "
            "KLQR/AB-PQ/BC-KMPC: [psi(normalize(x)), normalize(c)]"
        ),
        "context": {
            "definition": "[G1,G2,G3,onehot(active_waypoint_index)]",
            "dimension": TASK_CONTEXT_DIM,
        },
        "data_split": {
            split: {
                "episodes": int(len(data[f"{split}_episode_ids"])),
                "transitions": int(masks[split].sum()),
            }
            for split in ("train", "validation", "test")
        },
        "test_metric": {
            "name": "one_step_action_mse",
            "units": "rad^2",
            "definition": (
                "mean over 7 action dimensions and held-out test transitions; "
                "no contiguous trajectory rollout"
            ),
            "test_episodes_are_unseen": True,
        },
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = {**meta, "actors": reports}
    if actor_names is None:
        # Sequential all-actor mode writes the aggregate report directly.
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def merge_reports(output_dir: Path) -> dict[str, Any]:
    """Rebuild ``report.json`` from per-actor ``*.report.json`` files.

    Used after parallel per-actor training so the aggregate comparison report
    is available in the shared output directory.
    """

    meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    actors: dict[str, Any] = {}
    for name in BC_ACTOR_ORDER:
        path = output_dir / f"{name}.report.json"
        if path.exists():
            actors[name] = json.loads(path.read_text(encoding="utf-8"))
    report = {**meta, "actors": actors}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "runs/pandareach_threewaypoint/data/pandareach_dls_500.npz"
        ),
    )
    parser.add_argument(
        "--koopman",
        type=Path,
        default=Path("runs/pandareach_threewaypoint/koopman/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/pandareach_threewaypoint/bc"),
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-num-envs", type=int, default=16)
    parser.add_argument("--kmpc-horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--actors",
        nargs="+",
        choices=("PPO", "KLQR", "AB-PQ", "BC-KMPC"),
        default=None,
        help="Train only the selected actor routes (default: all routes).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help=(
            "Rebuild report.json from per-actor *.report.json files after "
            "parallel per-actor training; skips training entirely."
        ),
    )
    parser.add_argument(
        "--eval-only",
        nargs="+",
        choices=("PPO", "KLQR", "AB-PQ", "BC-KMPC"),
        default=None,
        help=(
            "Do not train: run the (batched) closed-loop evaluation on the "
            "saved {output-dir}/{actor}.pt checkpoint(s) and exit."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        print(json.dumps(merge_reports(args.output_dir), indent=2))
        return
    config = BCConfig(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_num_envs=args.evaluation_num_envs,
        kmpc_horizon=args.kmpc_horizon,
        seed=args.seed,
    )
    if args.eval_only is not None:
        for actor_name in args.eval_only:
            evaluate_checkpoint(
                args.output_dir,
                actor_name,
                args.koopman,
                config,
                args.device,
            )
        return
    report = train(
        args.dataset,
        args.koopman,
        args.output_dir,
        config,
        args.device,
        tuple(args.actors) if args.actors is not None else None,
    )
    print(json.dumps(report["actors"], indent=2))


if __name__ == "__main__":
    main()
