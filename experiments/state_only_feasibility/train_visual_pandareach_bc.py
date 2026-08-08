"""Single-stage end-to-end visual BC on PandaReach3 with a KMPC actor.

Pipeline (final design, no [r,v] Koopman, no rollout pretraining):

    r(17) --frozen Koopman lift--> z_r(49)
    RGBD+DCT(8ch) --trainable VisualEncoder--> v(v_dim) + pos_branch(3)
    cost-map net([z_r, v]) --KoopmanMPCActor--> per-stage (Q_k, p_k) -> KMPC -> U*
    L = ‖u*_0 − a_expert‖²
      + λ · Σ_k mask_k·‖u*_k − a_expert[t+k]‖²          (multi-step sequence)
      + w_pos · ‖pos_branch(v) − goal_norm‖²            (privileged, training-only)

The whole gradient chain (encoder -> cost-map -> FISTA solver -> action) is
trained by the BC action loss; ``pos_branch`` anchors the visual latent
against collapse.  The robot Koopman lift is frozen.
"""

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

from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor
from experiments.state_only_feasibility.collect_pandareach import _first, _numpy
from experiments.state_only_feasibility.collect_pandareach_threewaypoint import (
    _state_from_observation,
)
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_bc import (
    _future_targets,
    load_koopman,
)
from experiments.state_only_feasibility.visual_encoder import VisualEncoder
from experiments.state_only_feasibility.visual_pandareach_env import (
    VisualPandaReachThreeWaypointEnv,
)
from experiments.state_only_feasibility.visual_pandareach_single_goal import (
    VisualPandaReachSingleGoalEnv,
)


@dataclass(frozen=True)
class VisualBCConfig:
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 3e-4
    encoder_lr_scale: float = 0.5
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    v_dim: int = 16
    encoder_hidden_dim: int = 128
    costmap_hidden_dim: int = 128
    use_dct: bool = True
    # ManiSkill minimal-shader depth is in millimeters (uint16).
    depth_scale: float = 2500.0
    pos_weight: float = 0.5
    # Fix A: use the pos_branch goal estimate as the costmap context (explicit
    # visual target) instead of the raw latent v.
    use_goal_context: bool = True
    env_id: str = "ACMPC-VisualPandaReach3-v0"
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
    # Single-goal eval must match the collection env (enlarged workspace goal
    # sampling region + larger visible marker) or the closed-loop metric is
    # measured on a different task distribution than the data.
    goal_region_radius: tuple[float, float, float] | None = (0.06, 0.06, 0.03)
    goal_marker_scale: float = 5.0

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs/batch_size must be positive")
        if self.learning_rate <= 0 or self.encoder_lr_scale <= 0:
            raise ValueError("learning rates must be positive")
        if self.v_dim < 1:
            raise ValueError("v_dim must be positive")
        if self.kmpc_horizon < 1:
            raise ValueError("kmpc_horizon must be positive")
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
        if self.pos_weight < 0:
            raise ValueError("pos_weight must be non-negative")
        if self.action_limit_rad <= 0:
            raise ValueError("action_limit_rad must be positive")


def _resolve_device(name: str) -> torch.device:
    return torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else ("cpu" if name == "auto" else name)
    )


def _normalizers(
    data: dict[str, np.ndarray],
    checkpoint: dict[str, Any],
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return robot (center, scale) from the Koopman checkpoint and goal stats."""

    def _cpu_array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    center = _cpu_array(checkpoint["normalizer"]["center"])
    scale = _cpu_array(checkpoint["normalizer"]["scale"])
    if center.shape != (17,) or scale.shape != (17,):
        raise ValueError("Koopman normalizer must be 17-dimensional")
    goals = data["active_goal_position"].astype(np.float64)
    goal_center = goals[train_mask].mean(0).astype(np.float32)
    goal_scale = np.maximum(goals[train_mask].std(0), 1e-4).astype(np.float32)
    return center, scale, goal_center, goal_scale


def _make_loader(
    state: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    action: np.ndarray,
    goal: np.ndarray,
    future: np.ndarray,
    future_mask: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    # Keep raw dtypes: state/action/goal/future are float32 already, rgb is
    # uint8 and depth is uint16 (millimeters); the encoder handles conversion.
    tensors = tuple(
        torch.from_numpy(value[mask]) for value in (state, rgb, depth, action, goal, future, future_mask)
    )
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=min(batch_size, int(mask.sum())),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _evaluate_mse(
    encoder: VisualEncoder,
    actor: KoopmanMPCActor,
    koopman: nn.Module,
    center: np.ndarray,
    scale: np.ndarray,
    goal_center: np.ndarray,
    goal_scale: np.ndarray,
    loader: DataLoader,
    device: torch.device,
    use_goal_context: bool,
) -> float:
    total = 0.0
    elements = 0
    encoder.eval()
    actor.eval()
    with torch.no_grad():
        for state, rgb, depth, action, _, _, _ in loader:
            state, rgb, depth, action = (
                state.to(device),
                rgb.to(device),
                depth.to(device),
                action.to(device),
            )
            lifted = koopman.lift(state)
            v, pos = encoder(rgb, depth)
            context = pos if use_goal_context else v
            prediction = actor(lifted, context).action
            total += float((prediction - action).square().sum())
            elements += action.numel()
    encoder.train()
    actor.train()
    return total / elements


def _visual_observation(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = _state_from_observation(observation).astype(np.float32)
    camera = observation["sensor_data"]["base_camera"]
    rgb = _numpy(camera["rgb"])
    depth = _numpy(camera["depth"])
    if rgb.ndim > 0 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if depth.ndim > 0 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 2:
        depth = depth[..., None]
    return robot, rgb.astype(np.uint8), depth.astype(np.uint16)


def closed_loop_evaluation(
    encoder: VisualEncoder,
    actor: KoopmanMPCActor,
    koopman: nn.Module,
    center: np.ndarray,
    scale: np.ndarray,
    config: VisualBCConfig,
    device: torch.device,
) -> dict[str, Any]:
    env = PandaArmOnlyActionWrapper(
        gym.make(
            config.env_id,
            num_envs=1,
            obs_mode="rgb+depth",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode=None,
            max_episode_steps=config.max_episode_steps,
            goal_threshold=config.goal_threshold,
            **(
                {
                    "goal_region_radius": config.goal_region_radius,
                    "goal_marker_scale": config.goal_marker_scale,
                }
                if config.env_id == "ACMPC-VisualPandaReach1-v0"
                else {}
            ),
        )
    )
    successes = 0
    completed: list[int] = []
    final_distances: list[float] = []
    lengths: list[int] = []
    bound_fractions: list[float] = []
    encoder.eval()
    actor.eval()
    try:
        for episode in range(config.evaluation_episodes):
            observation, _ = env.reset(seed=config.evaluation_seed_start + episode)
            bound_count = 0
            action_count = 0
            for step in range(config.max_episode_steps):
                robot, rgb, depth = _visual_observation(observation)
                state_tensor = torch.as_tensor(
                    (robot - center) / scale,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                rgb_tensor = torch.as_tensor(
                    rgb, dtype=torch.uint8, device=device
                ).unsqueeze(0)
                depth_tensor = torch.as_tensor(
                    depth, dtype=torch.uint16, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    lifted = koopman.lift(state_tensor)
                    v, pos = encoder(rgb_tensor, depth_tensor)
                    context = pos if config.use_goal_context else v
                    action = actor(lifted, context).action
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
                if bool(_first(terminated)) or bool(_first(truncated)):
                    break
            success = bool(_first(info["success"]))
            successes += int(success)
            completed.append(
                int(_first(info["waypoints_completed"]))
            )
            final_distances.append(
                float(_first(info["active_waypoint_distance"]))
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
    config: VisualBCConfig,
    device_name: str = "auto",
) -> dict[str, Any]:
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    koopman, checkpoint = load_koopman(koopman_path, device)
    episode = data["episode_id"]
    masks = {
        split: np.isin(episode, data[f"{split}_episode_ids"])
        for split in ("train", "validation", "test")
    }
    center, scale, goal_center, goal_scale = _normalizers(
        data, checkpoint, masks["train"]
    )
    normalized_state = ((data["state"] - center) / scale).astype(np.float32)
    normalized_goal = (
        (data["active_goal_position"] - goal_center) / goal_scale
    ).astype(np.float32)
    future, future_mask = _future_targets(data, config.kmpc_horizon)
    loaders = {
        split: _make_loader(
            normalized_state,
            data["rgb"],
            data["depth"],
            data["action"],
            normalized_goal,
            future,
            future_mask,
            masks[split],
            config.batch_size,
            split == "train",
            config.seed,
        )
        for split in ("train", "validation", "test")
    }

    encoder = VisualEncoder(
        config.v_dim,
        hidden_dims=(config.encoder_hidden_dim,),
        use_dct=config.use_dct,
        depth_scale=config.depth_scale,
        pos_dim=3,
    ).to(device)
    actor = KoopmanMPCActor(
        A=koopman.A,
        B=koopman.B,
        C=koopman.C,
        horizon=config.kmpc_horizon,
        context_dim=3 if config.use_goal_context else config.v_dim,
        hidden_dims=(config.costmap_hidden_dim,),
        action_low=-config.action_limit_rad,
        action_high=config.action_limit_rad,
        solver_iterations=config.kmpc_solver_iterations,
    ).to(device)

    encoder_parameters = list(encoder.parameters())
    actor_parameters = list(actor.parameters())
    optimizer = torch.optim.Adam(
        [
            {
                "params": actor_parameters,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": encoder_parameters,
                "lr": config.learning_rate * config.encoder_lr_scale,
                "weight_decay": config.weight_decay,
            },
        ]
    )

    best_state: dict[str, Any] | None = None
    best_validation = float("inf")
    best_epoch = 0
    last_improvement_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        encoder.train()
        actor.train()
        for state, rgb, depth, action, goal, future, future_mask in loaders["train"]:
            state, rgb, depth, action, goal = (
                state.to(device),
                rgb.to(device),
                depth.to(device),
                action.to(device),
                goal.to(device),
            )
            future, future_mask = future.to(device), future_mask.to(device)
            with torch.no_grad():
                lifted = koopman.lift(state)
            v, pos = encoder(rgb, depth)
            context = pos if config.use_goal_context else v
            output = actor(lifted, context)
            loss = (output.action - action).square().mean()
            if output.action_sequence.shape[-2] > 1:
                errors = (
                    output.action_sequence[:, 1:] - future[:, 1:]
                ).square().mean(-1)
                valid = future_mask[:, 1:]
                future_loss = (errors * valid).sum() / valid.sum().clamp_min(1.0)
                loss = loss + config.kmpc_sequence_weight * future_loss
            if config.pos_weight > 0:
                pos_loss = (pos - goal).square().mean()
                loss = loss + config.pos_weight * pos_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in encoder.parameters()]
                + [parameter for parameter in actor.parameters()],
                config.gradient_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Visual BC produced a non-finite gradient")
            optimizer.step()

        should_validate = (
            epoch == 1
            or epoch % config.validation_interval == 0
            or epoch == config.epochs
        )
        validation = float("nan")
        if should_validate:
            validation = _evaluate_mse(
                encoder,
                actor,
                koopman,
                center,
                scale,
                goal_center,
                goal_scale,
                loaders["validation"],
                device,
                config.use_goal_context,
            )
            if validation < best_validation:
                best_validation = validation
                best_epoch = epoch
                last_improvement_epoch = epoch
                best_state = {
                    "encoder": copy.deepcopy(encoder.state_dict()),
                    "actor": copy.deepcopy(actor.state_dict()),
                }
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "validation_mse": validation,
                            "best_epoch": best_epoch,
                            "best_validation_mse": best_validation,
                        }
                    )
                    + "\n"
                )
                handle.flush()
        if epoch % 50 == 0:
            torch.save(
                {
                    "method": "visual_bc_kmpc",
                    "config": asdict(config),
                    "koopman_checkpoint": str(koopman_path.resolve()),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_mse": best_validation,
                    "encoder": copy.deepcopy(encoder.state_dict()),
                    "actor": copy.deepcopy(actor.state_dict()),
                },
                output_dir / f"recovery_epoch_{epoch:04d}.pt",
            )
        if epoch == 1 or epoch % 50 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:04d} val_mse={validation:.6g}",
                flush=True,
            )
        if (
            should_validate
            and epoch - last_improvement_epoch >= config.early_stopping_patience
        ):
            print(
                f"visual_bc early_stop={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("Visual BC training did not produce a checkpoint")
    encoder.load_state_dict(best_state["encoder"])
    actor.load_state_dict(best_state["actor"])

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "visual_bc_kmpc",
            "config": asdict(config),
            "koopman_checkpoint": str(koopman_path.resolve()),
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation,
            "encoder": best_state["encoder"],
            "actor": best_state["actor"],
        },
        output_dir / "best.pt",
    )
    test_mse = _evaluate_mse(
        encoder,
        actor,
        koopman,
        center,
        scale,
        goal_center,
        goal_scale,
        loaders["test"],
        device,
        config.use_goal_context,
    )
    evaluation = closed_loop_evaluation(
        encoder,
        actor,
        koopman,
        center,
        scale,
        config,
        device,
    )
    report = {
        "method": "visual_bc_kmpc",
        "dataset_path": str(dataset_path.resolve()),
        "koopman_checkpoint": str(koopman_path.resolve()),
        "best_epoch": best_epoch,
        "completed_epochs": epoch,
        "best_validation_mse": best_validation,
        "test_mse": test_mse,
        "evaluation": evaluation,
        "training_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(
            parameter.numel() for parameter in encoder.parameters()
        )
        + sum(parameter.numel() for parameter in actor.parameters()),
        "config": asdict(config),
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"test_mse": test_mse, "evaluation": evaluation}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Visual PandaReach3 npz from collect_visual_pandareach_threewaypoint",
    )
    parser.add_argument(
        "--koopman-checkpoint",
        default="runs/pandareach_threewaypoint/koopman_coverage/best.pt",
        type=Path,
    )
    parser.add_argument("--output", default="runs/visual_pandareach_bc", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--v-dim", type=int, default=None)
    parser.add_argument("--use-dct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--kmpc-horizon", type=int, default=None)
    parser.add_argument("--env-id", default=VisualBCConfig.env_id)
    parser.add_argument(
        "--use-goal-context", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    defaults = VisualBCConfig()
    config = VisualBCConfig(
        epochs=args.epochs if args.epochs is not None else defaults.epochs,
        batch_size=args.batch_size if args.batch_size is not None else defaults.batch_size,
        learning_rate=(
            args.learning_rate if args.learning_rate is not None else defaults.learning_rate
        ),
        v_dim=args.v_dim if args.v_dim is not None else defaults.v_dim,
        use_dct=args.use_dct,
        pos_weight=args.pos_weight if args.pos_weight is not None else defaults.pos_weight,
        use_goal_context=args.use_goal_context,
        env_id=args.env_id,
        kmpc_horizon=(
            args.kmpc_horizon if args.kmpc_horizon is not None else defaults.kmpc_horizon
        ),
        evaluation_episodes=(
            args.evaluation_episodes
            if args.evaluation_episodes is not None
            else defaults.evaluation_episodes
        ),
        seed=args.seed if args.seed is not None else defaults.seed,
    )
    train(args.dataset, args.koopman_checkpoint, args.output, config, args.device)


if __name__ == "__main__":
    main()
