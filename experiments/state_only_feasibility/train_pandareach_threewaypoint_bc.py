"""Train and evaluate B0, H1-min and BC-KMPC on PandaReach3."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from antmaze_ac.koopman.model import DeepKoopman
from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor
from antmaze_ac.rl.quadratic_actors import MinimalDirectQuadraticActor
from experiments.state_only_feasibility.collect_pandareach_threewaypoint import (
    _state_from_observation,
)
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
    PandaReachThreeWaypointEnv,
)


@dataclass(frozen=True)
class BCConfig:
    epochs: int = 500
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    hidden_dim: int = 128
    b0_hidden_dim: int = 188
    kmpc_horizon: int = 10
    kmpc_solver_iterations: int = 20
    kmpc_sequence_weight: float = 0.25
    action_limit_rad: float = 0.1
    seed: int = 47
    validation_interval: int = 5
    early_stopping_patience: int = 75
    evaluation_episodes: int = 100
    evaluation_seed_start: int = 20_270_804
    max_episode_steps: int = 220
    goal_threshold: float = 0.01


class B0Actor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_limit: float):
        super().__init__()
        self.action_limit = float(action_limit)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.action_limit * torch.tanh(
            self.network(torch.cat((state, context), dim=-1))
        )


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
    if name == "B0":
        # B0 is the raw-state baseline: it deliberately does not use the
        # Koopman lift.  The other routes receive ``lifted`` below.
        return actor(normalized_state, context), None
    if name == "H1-min":
        return actor(normalized_state, lifted, context).action, None
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


def _train_actor(
    name: str,
    actor: nn.Module,
    koopman: DeepKoopman,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: BCConfig,
    device: torch.device,
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
    center = np.asarray(checkpoint["normalizer"]["center"], dtype=np.float32)
    scale = np.asarray(checkpoint["normalizer"]["scale"], dtype=np.float32)
    goals = data["active_goal_position"][train_mask].astype(np.float64)
    goal_center = goals.mean(0).astype(np.float32)
    goal_scale = np.maximum(goals.std(0), 1e-4).astype(np.float32)
    return center, scale, goal_center, goal_scale


def closed_loop_evaluation(
    name: str,
    actor: nn.Module,
    koopman: DeepKoopman,
    center: np.ndarray,
    scale: np.ndarray,
    goal_center: np.ndarray,
    goal_scale: np.ndarray,
    config: BCConfig,
    device: torch.device,
) -> dict[str, Any]:
    env = PandaArmOnlyActionWrapper(
        gym.make(
            "ACMPC-PandaReach3-v0",
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode=None,
            render_backend="none",
            max_episode_steps=config.max_episode_steps,
            goal_threshold=config.goal_threshold,
        )
    )
    successes = 0
    completed: list[int] = []
    final_distances: list[float] = []
    lengths: list[int] = []
    bound_fractions: list[float] = []
    actor.eval()
    try:
        for episode in range(config.evaluation_episodes):
            observation, _ = env.reset(seed=config.evaluation_seed_start + episode)
            bound_count = 0
            action_count = 0
            for step in range(config.max_episode_steps):
                physical = _state_from_observation(observation)
                goal = np.asarray(
                    observation["extra"]["active_goal"], dtype=np.float32
                ).reshape(3)
                state_tensor = torch.as_tensor(
                    (physical - center) / scale,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                goal_tensor = torch.as_tensor(
                    (goal - goal_center) / goal_scale,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    lifted = koopman.lift(state_tensor)
                    action, _ = _actor_action(
                        name, actor, state_tensor, lifted, goal_tensor
                    )
                action_rad = action.squeeze(0).cpu().numpy()
                bound_count += int(
                    np.count_nonzero(
                        np.abs(action_rad) >= 0.99 * config.action_limit_rad
                    )
                )
                action_count += 7
                observation, _, terminated, truncated, info = env.step(
                    np.clip(
                        action_rad / config.action_limit_rad, -1.0, 1.0
                    ).astype(np.float32)
                )
                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
                    np.asarray(truncated).reshape(-1)[0]
                ):
                    break
            success = bool(np.asarray(info["success"]).reshape(-1)[0])
            successes += int(success)
            completed.append(
                int(np.asarray(info["waypoints_completed"]).reshape(-1)[0])
            )
            final_distances.append(
                float(np.asarray(info["active_waypoint_distance"]).reshape(-1)[0])
            )
            lengths.append(step + 1)
            bound_fractions.append(bound_count / max(action_count, 1))
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


def train(
    dataset_path: Path,
    koopman_path: Path,
    output_dir: Path,
    config: BCConfig,
    device_name: str = "auto",
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
    center, scale, goal_center, goal_scale = _normalizers(
        data, masks["train"], checkpoint
    )
    normalized_state = ((data["state"] - center) / scale).astype(np.float32)
    normalized_goal = (
        (data["active_goal_position"] - goal_center) / goal_scale
    ).astype(np.float32)
    future, future_mask = _future_targets(data, config.kmpc_horizon)
    loaders = {
        split: _make_loader(
            normalized_state,
            normalized_goal,
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
    builders = {
        "B0": lambda: B0Actor(
            koopman.state_dim + 3,
            config.b0_hidden_dim,
            config.action_limit_rad,
        ),
        "H1-min": lambda: MinimalDirectQuadraticActor(
            observation_dim=17,
            lifted_dim=koopman.lifted_dim,
            action_dim=7,
            hidden_dims=(config.hidden_dim,),
            context_dim=3,
            conditioning="lifted",
            max_action=config.action_limit_rad,
        ),
        "BC-KMPC": lambda: KoopmanMPCActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            horizon=config.kmpc_horizon,
            context_dim=3,
            hidden_dims=(config.hidden_dim,),
            action_low=-config.action_limit_rad,
            action_high=config.action_limit_rad,
            solver_iterations=config.kmpc_solver_iterations,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for offset, (name, builder) in enumerate(builders.items()):
        torch.manual_seed(config.seed + offset)
        actor = builder().to(device)
        actor, actor_report = _train_actor(
            name,
            actor,
            koopman,
            loaders["train"],
            loaders["validation"],
            config,
            device,
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
                    goal_center,
                    goal_scale,
                    config,
                    device,
                ),
            }
        )
        checkpoint_payload = {
            "kind": "pandareach_threewaypoint_bc_actor",
            "name": name,
            "actor_state": actor.state_dict(),
            "config": asdict(config),
            "normalizer": {
                "state_center": center,
                "state_scale": scale,
                "goal_center": goal_center,
                "goal_scale": goal_scale,
            },
            "koopman_path": str(koopman_path.resolve()),
            "report": actor_report,
        }
        torch.save(checkpoint_payload, output_dir / f"{name}.pt")
        reports[name] = actor_report
        print(json.dumps({name: actor_report}, sort_keys=True), flush=True)
    report = {
        "kind": "pandareach_threewaypoint_bc_comparison",
        "dataset_path": str(dataset_path.resolve()),
        "koopman_path": str(koopman_path.resolve()),
        "config": asdict(config),
        "actor_input": "[psi(normalize([q,qdot,tcp])), normalize(G_j)]",
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
        "actors": reports,
    }
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
            "runs/pandareach_threewaypoint/data/pandareach_dls_100.npz"
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--kmpc-horizon", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train(
        args.dataset,
        args.koopman,
        args.output_dir,
        BCConfig(
            epochs=args.epochs,
            evaluation_episodes=args.evaluation_episodes,
            kmpc_horizon=args.kmpc_horizon,
        ),
        args.device,
    )
    print(json.dumps(report["actors"], indent=2))


if __name__ == "__main__":
    main()
