"""Train the controlled visual Koopman model on causal PickCube trajectories.

The frozen ResNet-18 is intentionally not part of this loop.  This script
consumes its cached 512-dimensional features, learns ``v=E(b)``, constructs
``s=[r,v]``, and fits ``s_next=A s+B u`` with multi-step rollout losses.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader, Dataset

from antmaze_ac.koopman.checkpoint import load_checkpoint, save_checkpoint, sha256
from antmaze_ac.koopman.visual_losses import VisualKoopmanLoss, visual_koopman_loss
from antmaze_ac.koopman.visual_model import VisualLinearKoopman
from experiments.maniskill_pick_visual.dataset import (
    VisualWindowDataset,
    fit_normalizers,
    list_episode_ids,
    split_episode_ids,
)


@dataclass(frozen=True)
class TrainConfig:
    trajectory_h5: Path
    feature_h5: Path
    output_dir: Path
    epochs: int = 250
    patience: int = 50
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    horizon: int = 20
    visual_latent_dim: int = 16
    encoder_hidden_dims: tuple[int, ...] = (256, 64)
    transform_mode: str = "identity"
    feature_key: str = "resnet18"
    seed: int = 43
    workers: int = 0
    rollout_discount: float = 0.99
    linear_weight: float = 10.0
    robot_rollout_weight: float = 2.0
    feature_reconstruction_weight: float = 0.2
    future_feature_reconstruction_weight: float = 0.2
    latent_variance_weight: float = 0.1
    stability_weight: float = 0.02
    identity_weight: float = 1e-4
    transform_reconstruction_weight: float = 1.0
    transform_condition_weight: float = 1e-3
    transform_singular_value_weight: float = 1e-2
    transform_minimum_singular_value: float = 0.25
    target_latent_std: float = 1.0
    spectral_radius_limit: float = 1.02
    require_provenance: bool = True
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-6
    early_stopping_min_delta: float = 0.0
    checkpoint_interval: int = 25
    preload: bool = False
    wandb_mode: str = "offline"
    wandb_project: str = "acmpc-visual-koopman"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_group: str | None = None
    max_episodes: int | None = None


class PreloadedVisualWindowDataset(Dataset[dict[str, Any]]):
    """In-memory equivalent of :class:`VisualWindowDataset`.

    The public dimensions, ``horizon`` attribute, and item schema are kept so
    the existing loaders, losses, and rollout metrics do not need a separate
    code path. The source HDF5 handles are closed as soon as materialization
    finishes.
    """

    def __init__(self, source: VisualWindowDataset) -> None:
        super().__init__()
        for name in (
            "trajectory_h5",
            "feature_h5",
            "episode_ids",
            "horizon",
            "feature_key",
            "normalize",
            "normalizers",
            "robot_dim",
            "feature_dim",
            "action_dim",
        ):
            setattr(self, name, getattr(source, name))
        robot: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        episode_ids: list[str] = []
        starts: list[int] = []
        try:
            for index in range(len(source)):
                item = source[index]
                robot.append(item["robot"])
                features.append(item["features"])
                actions.append(item["actions"])
                episode_ids.append(str(item["episode_id"]))
                starts.append(int(item["start"]))
        finally:
            source.close()
        self._robot = torch.stack(robot)
        self._features = torch.stack(features)
        self._actions = torch.stack(actions)
        self._episode_ids = tuple(episode_ids)
        self._starts = tuple(starts)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        robot = self._robot[index]
        features = self._features[index]
        return {
            "robot": robot,
            "features": features,
            "state": torch.cat((robot, features), dim=-1),
            "actions": self._actions[index],
            "episode_id": self._episode_ids[index],
            "start": self._starts[index],
        }

    @property
    def window_metadata(self) -> tuple[tuple[str, int], ...]:
        return tuple(zip(self._episode_ids, self._starts, strict=True))

    def action_windows(self, indices: Sequence[int]) -> torch.Tensor:
        resolved = torch.as_tensor(tuple(int(index) for index in indices))
        if resolved.numel() == 0:
            return self._actions[:0].clone()
        if bool((resolved < 0).any()) or bool((resolved >= len(self)).any()):
            raise IndexError("action-window index is out of range")
        return self._actions.index_select(0, resolved).clone()

    def close(self) -> None:
        """Match the lazy dataset interface; no file handles remain open."""


VisualDataset = VisualWindowDataset | PreloadedVisualWindowDataset


def preload_dataset(dataset: VisualWindowDataset) -> PreloadedVisualWindowDataset:
    return PreloadedVisualWindowDataset(dataset)


def validate_train_config(config: TrainConfig) -> None:
    """Fail before creating a run directory on malformed hyperparameters."""

    positive_integers = {
        "epochs": config.epochs,
        "patience": config.patience,
        "batch_size": config.batch_size,
        "horizon": config.horizon,
        "visual_latent_dim": config.visual_latent_dim,
    }
    for name, value in positive_integers.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(config.workers, bool)
        or not isinstance(config.workers, (int, np.integer))
        or config.workers < 0
    ):
        raise ValueError("workers must be a non-negative integer")
    if (
        isinstance(config.lr_patience, bool)
        or not isinstance(config.lr_patience, (int, np.integer))
        or config.lr_patience < 0
    ):
        raise ValueError("lr_patience must be a non-negative integer")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < 1
        for value in config.encoder_hidden_dims
    ):
        raise ValueError("encoder_hidden_dims must contain positive integers")
    if isinstance(config.seed, bool) or not isinstance(config.seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not isinstance(config.preload, bool) or not isinstance(
        config.require_provenance, bool
    ):
        raise ValueError("preload and require_provenance must be booleans")
    if config.max_episodes is not None and config.max_episodes < 3:
        raise ValueError("max_episodes must be at least 3 when provided")
    if config.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")

    positive_floats = {
        "learning_rate": config.learning_rate,
        "gradient_clip": config.gradient_clip,
        "spectral_radius_limit": config.spectral_radius_limit,
    }
    nonnegative_floats = {
        "weight_decay": config.weight_decay,
        "linear_weight": config.linear_weight,
        "robot_rollout_weight": config.robot_rollout_weight,
        "feature_reconstruction_weight": config.feature_reconstruction_weight,
        "future_feature_reconstruction_weight": (
            config.future_feature_reconstruction_weight
        ),
        "latent_variance_weight": config.latent_variance_weight,
        "stability_weight": config.stability_weight,
        "identity_weight": config.identity_weight,
        "transform_reconstruction_weight": config.transform_reconstruction_weight,
        "transform_condition_weight": config.transform_condition_weight,
        "transform_singular_value_weight": config.transform_singular_value_weight,
        "target_latent_std": config.target_latent_std,
        "min_lr": config.min_lr,
        "early_stopping_min_delta": config.early_stopping_min_delta,
    }
    for name, value in positive_floats.items():
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    for name, value in nonnegative_floats.items():
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if isinstance(config.rollout_discount, bool) or not math.isfinite(
        float(config.rollout_discount)
    ) or not (
        0.0 < config.rollout_discount <= 1.0
    ):
        raise ValueError("rollout_discount must lie in (0, 1]")
    if (
        isinstance(config.lr_factor, bool)
        or not math.isfinite(float(config.lr_factor))
        or not 0.0 < config.lr_factor < 1.0
    ):
        raise ValueError("lr_factor must lie in (0, 1)")
    if config.min_lr > config.learning_rate:
        raise ValueError("min_lr must not exceed learning_rate")
    if (
        isinstance(config.transform_minimum_singular_value, bool)
        or not math.isfinite(float(config.transform_minimum_singular_value))
        or float(config.transform_minimum_singular_value) <= 0.0
    ):
        raise ValueError("transform_minimum_singular_value must be finite and positive")
    if config.transform_mode not in {
        "identity",
        "learned",
        "learned_inverse",
        "learned_orthogonal",
    }:
        raise ValueError(
            "transform_mode must be identity, learned, learned_inverse, or "
            "learned_orthogonal"
        )
    if not str(config.feature_key).strip():
        raise ValueError("feature_key must not be empty")
    if config.wandb_mode not in {"offline", "disabled", "online"}:
        raise ValueError("wandb_mode must be offline, disabled, or online")
    if not str(config.wandb_project).strip():
        raise ValueError("wandb_project must not be empty")
    if config.wandb_entity is not None and not str(config.wandb_entity).strip():
        raise ValueError("wandb_entity must be None or non-empty")
    if config.wandb_name is not None and not str(config.wandb_name).strip():
        raise ValueError("wandb_name must be None or non-empty")
    if config.wandb_group is not None and not str(config.wandb_group).strip():
        raise ValueError("wandb_group must be None or non-empty")


def _assert_finite_parameters(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"Model parameter {name} contains NaN or Inf")


def _assert_finite_scalars(values: dict[str, float], label: str) -> None:
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"{label}.{name} is NaN or Inf")


def _start_wandb(config: TrainConfig, serialized_config: dict[str, Any]) -> Any:
    if config.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required for offline/online tracking; install the tracking extra "
            "or pass --wandb-mode disabled"
        ) from exc
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name,
        group=config.wandb_group,
        mode=config.wandb_mode,
        dir=str(config.output_dir),
        config=serialized_config,
    )


def _assert_finite_tree(value: Any, label: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"{label} is NaN or Inf")


def _flatten_numeric(prefix: str, value: Any) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    if isinstance(value, dict):
        for name, child in value.items():
            child_prefix = f"{prefix}/{name}" if prefix else str(name)
            flattened.update(_flatten_numeric(child_prefix, child))
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        flattened[prefix] = float(value) if isinstance(value, (float, np.floating)) else int(value)
    return flattened


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed metrics row {line_number}") from exc
            if not isinstance(row, dict) or "epoch" not in row:
                raise ValueError(f"Malformed metrics row {line_number}")
            rows.append(row)
    return rows


def _reconcile_history(path: Path, checkpoint_epoch: int) -> list[dict[str, Any]]:
    """Drop metric rows written after a checkpoint during an interrupted epoch."""

    rows = _read_history(path)
    retained = [row for row in rows if int(row["epoch"]) <= checkpoint_epoch]
    if len(retained) != len(rows):
        temporary = path.with_suffix(".jsonl.reconcile")
        temporary.write_text(
            "".join(json.dumps(row) + "\n" for row in retained),
            encoding="utf-8",
        )
        temporary.replace(path)
    return retained


def validate_data_provenance(
    config: TrainConfig,
    *,
    trajectory_digest: str | None = None,
) -> tuple[str, str]:
    """Reject non-causal or mismatched trajectory/feature pairs by default."""

    trajectory_digest = trajectory_digest or sha256(config.trajectory_h5)
    feature_digest = sha256(config.feature_h5)
    if not config.require_provenance:
        return trajectory_digest, feature_digest
    with h5py.File(config.trajectory_h5, "r") as trajectories:
        required = {
            "causal_replay": True,
            "goal_visible": True,
            "control_mode": "pd_joint_delta_pos",
            "actions_are_applied": True,
        }
        for name, expected in required.items():
            actual = trajectories.attrs.get(name)
            if actual != expected:
                raise ValueError(
                    f"Trajectory provenance {name}={actual!r}; expected {expected!r}"
                )
        if "action_low" not in trajectories.attrs or "action_high" not in trajectories.attrs:
            raise ValueError("Trajectory is missing applied-action bounds")
        action_low = np.asarray(trajectories.attrs["action_low"], dtype=np.float32)
        action_high = np.asarray(trajectories.attrs["action_high"], dtype=np.float32)
        if (
            action_low.ndim != 1
            or action_low.shape != action_high.shape
            or not np.isfinite(action_low).all()
            or not np.isfinite(action_high).all()
            or np.any(action_low >= action_high)
        ):
            raise ValueError("Trajectory action bounds are malformed")
        for episode_name in list_episode_ids(config.trajectory_h5):
            actions = np.asarray(trajectories[episode_name]["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1:] != action_low.shape:
                raise ValueError(f"Invalid action shape in {episode_name}")
            if np.any(actions < action_low - 1e-6) or np.any(
                actions > action_high + 1e-6
            ):
                raise ValueError(
                    f"{episode_name}/actions exceed the recorded applied bounds"
                )
    with h5py.File(config.feature_h5, "r") as features:
        if not bool(features.attrs.get("complete", False)):
            raise ValueError("Feature sidecar is not marked complete")
        source_digest = features.attrs.get("source_sha256")
        if isinstance(source_digest, bytes):
            source_digest = source_digest.decode("utf-8")
        if source_digest != trajectory_digest:
            raise ValueError(
                "Feature sidecar source_sha256 does not match the trajectory file"
            )
        feature_dim = int(features.attrs.get("feature_dim", -1))
        if feature_dim < 1:
            raise ValueError("Feature sidecar is missing a valid feature_dim")
        if config.feature_key == "resnet18" and feature_dim != 512:
            raise ValueError("The resnet18 cache must contain 512-D features")
        if config.feature_key == "rgbd_resnet18" and feature_dim != 1024:
            raise ValueError("The rgbd_resnet18 cache must contain 1024-D features")
    return trajectory_digest, feature_digest


def _loader(
    dataset: VisualDataset,
    *,
    config: TrainConfig,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.workers > 0,
        drop_last=False,
    )


STANDARD_EVALUATION_HORIZONS = (1, 5, 10, 20)


def evaluation_horizons(training_horizon: int) -> tuple[int, ...]:
    """Horizons reported for a checkpoint, including a non-standard train H."""

    return tuple(sorted({*STANDARD_EVALUATION_HORIZONS, int(training_horizon)}))


def make_evaluation_datasets(
    config: TrainConfig,
    episode_ids: Sequence[str],
    normalizers: Mapping[str, np.ndarray],
    *,
    reuse: Mapping[int, VisualDataset] | None = None,
) -> dict[int, VisualDataset]:
    """Build one test dataset per horizon so every legal start is evaluated.

    Evaluation is a single pass, so additional horizons remain lazy even when
    the repeated-epoch training datasets were preloaded. Horizons longer than
    every selected episode are omitted and later reported with zero windows.
    """

    datasets = dict(reuse or {})
    for horizon in evaluation_horizons(config.horizon):
        if horizon in datasets:
            continue
        try:
            datasets[horizon] = VisualWindowDataset(
                config.trajectory_h5,
                config.feature_h5,
                episode_ids,
                horizon,
                normalizers,
                feature_key=config.feature_key,
            )
        except ValueError as exc:
            if "No complete windows exist" not in str(exc):
                raise
    return datasets


def _loss(
    model: VisualLinearKoopman,
    batch: dict[str, Any],
    device: torch.device,
    config: TrainConfig,
) -> VisualKoopmanLoss:
    robot = batch["robot"].to(device, non_blocking=True)
    features = batch["features"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    return visual_koopman_loss(
        model,
        robot,
        features,
        actions,
        rollout_discount=config.rollout_discount,
        linear_weight=config.linear_weight,
        robot_rollout_weight=config.robot_rollout_weight,
        feature_reconstruction_weight=config.feature_reconstruction_weight,
        future_feature_reconstruction_weight=(
            config.future_feature_reconstruction_weight
        ),
        latent_variance_weight=config.latent_variance_weight,
        stability_weight=config.stability_weight,
        identity_weight=config.identity_weight,
        transform_reconstruction_weight=config.transform_reconstruction_weight,
        transform_condition_weight=config.transform_condition_weight,
        transform_singular_value_weight=config.transform_singular_value_weight,
        transform_minimum_singular_value=(
            config.transform_minimum_singular_value
        ),
        target_latent_std=config.target_latent_std,
        spectral_radius_limit=config.spectral_radius_limit,
    )


def _aggregate_loss(
    totals: dict[str, float],
    loss: VisualKoopmanLoss,
    batch_size: int,
) -> None:
    for name, value in loss.scalars().items():
        totals[name] = totals.get(name, 0.0) + value * batch_size
    totals["samples"] = totals.get("samples", 0.0) + batch_size


def _finish_loss(totals: dict[str, float]) -> dict[str, float]:
    totals = dict(totals)
    samples = totals.pop("samples", 0.0)
    if not math.isfinite(samples) or samples <= 0:
        raise RuntimeError("A loss epoch contained no samples")
    result = {name: value / samples for name, value in totals.items()}
    _assert_finite_scalars(result, "loss")
    return result


def observable_validation(
    values: dict[str, float],
    config: TrainConfig,
) -> float:
    """Return the coordinate-invariant validation objective.

    The full training loss intentionally contains latent-coordinate and
    transform regularizers.  Those terms are useful for optimization but are
    not comparable across transform parameterizations, so checkpoint selection
    uses only errors expressed in robot and frozen-feature coordinates.
    """

    metric = (
        config.robot_rollout_weight * float(values["robot_rollout"])
        + config.feature_reconstruction_weight
        * float(values["feature_reconstruction"])
        + config.future_feature_reconstruction_weight
        * float(values["future_feature_reconstruction"])
    )
    if not math.isfinite(metric):
        raise FloatingPointError("observable_validation is NaN or Inf")
    return float(metric)


@torch.no_grad()
def evaluate_loss(
    model: VisualLinearKoopman,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    for batch in loader:
        loss = _loss(model, batch, device, config)
        _aggregate_loss(totals, loss, int(batch["robot"].shape[0]))
    return _finish_loss(totals)


def _new_error_accumulator() -> dict[str, float]:
    return {
        "robot_square": 0.0,
        "robot_absolute": 0.0,
        "robot_elements": 0.0,
        "robot_hold_square": 0.0,
        "arm_qpos_square": 0.0,
        "gripper_qpos_square": 0.0,
        "arm_qvel_square": 0.0,
        "gripper_qvel_square": 0.0,
        "tcp_square": 0.0,
        "arm_qpos_elements": 0.0,
        "gripper_qpos_elements": 0.0,
        "arm_qvel_elements": 0.0,
        "gripper_qvel_elements": 0.0,
        "tcp_elements": 0.0,
        "feature_square": 0.0,
        "feature_elements": 0.0,
        "feature_hold_square": 0.0,
        "feature_oracle_reconstruction_square": 0.0,
    }


def _finish_errors(values: dict[str, float]) -> dict[str, float]:
    return {
        "robot_rmse": float(
            np.sqrt(values["robot_square"] / values["robot_elements"])
        ),
        "robot_mae": values["robot_absolute"] / values["robot_elements"],
        "robot_hold_rmse": float(
            np.sqrt(values["robot_hold_square"] / values["robot_elements"])
        ),
        "arm_qpos_rmse_rad": float(
            np.sqrt(values["arm_qpos_square"] / values["arm_qpos_elements"])
        ),
        "gripper_qpos_rmse_m": float(
            np.sqrt(
                values["gripper_qpos_square"] / values["gripper_qpos_elements"]
            )
        ),
        "gripper_qpos_rmse_mm": float(
            1000.0
            * np.sqrt(
                values["gripper_qpos_square"] / values["gripper_qpos_elements"]
            )
        ),
        "arm_qvel_rmse_rad_s": float(
            np.sqrt(values["arm_qvel_square"] / values["arm_qvel_elements"])
        ),
        "gripper_qvel_rmse_m_s": float(
            np.sqrt(
                values["gripper_qvel_square"] / values["gripper_qvel_elements"]
            )
        ),
        "gripper_qvel_rmse_mm_s": float(
            1000.0
            * np.sqrt(
                values["gripper_qvel_square"] / values["gripper_qvel_elements"]
            )
        ),
        "tcp_rmse_m": float(
            np.sqrt(values["tcp_square"] / values["tcp_elements"])
        ),
        "tcp_rmse_mm": float(
            1000.0 * np.sqrt(values["tcp_square"] / values["tcp_elements"])
        ),
        "resnet_feature_rmse": float(
            np.sqrt(values["feature_square"] / values["feature_elements"])
        ),
        "resnet_feature_hold_rmse": float(
            np.sqrt(values["feature_hold_square"] / values["feature_elements"])
        ),
        "resnet_feature_oracle_reconstruction_rmse": float(
            np.sqrt(
                values["feature_oracle_reconstruction_square"]
                / values["feature_elements"]
            )
        ),
    }


def _deterministic_action_derangement(dataset: VisualDataset) -> np.ndarray:
    """Map every window to a distinct non-overlapping action sequence.

    The permutation is deterministic. A source is valid when it comes from a
    different episode, or when its start is at least one full horizon away in
    the same episode. Thus the randomized baseline never reuses the target's
    own action sequence and same-episode pairs share no action transitions.
    """

    metadata = dataset.window_metadata
    window_count = len(metadata)
    if window_count < 2:
        raise ValueError("At least two windows are required for action derangement")
    episode_to_code = {
        episode_id: code
        for code, episode_id in enumerate(dict.fromkeys(item[0] for item in metadata))
    }
    episodes = np.fromiter(
        (episode_to_code[episode_id] for episode_id, _ in metadata),
        dtype=np.int64,
        count=window_count,
    )
    starts = np.fromiter(
        (start for _, start in metadata), dtype=np.int64, count=window_count
    )
    indices = np.arange(window_count, dtype=np.int64)
    largest_episode = int(np.bincount(episodes).max())
    preferred = (largest_episode, window_count - largest_episode)
    offsets = tuple(
        dict.fromkeys(
            offset
            for offset in (*preferred, *range(1, window_count))
            if 0 < offset < window_count
        )
    )
    for offset in offsets:
        permutation = (indices + offset) % window_count
        different_episode = episodes != episodes[permutation]
        far_same_episode = np.abs(starts - starts[permutation]) >= dataset.horizon
        if bool(np.all(different_episode | far_same_episode)):
            return permutation
    raise ValueError(
        "No whole-sequence action derangement satisfies the cross-episode/"
        "non-overlap rule"
    )


def _action_ablation_metadata(
    dataset: VisualDataset,
    permutation: np.ndarray,
) -> dict[str, Any]:
    metadata = dataset.window_metadata
    cross_episode = 0
    far_same_episode = 0
    for target, source_index in zip(metadata, permutation.tolist(), strict=True):
        source = metadata[source_index]
        if target[0] != source[0]:
            cross_episode += 1
        else:
            if abs(target[1] - source[1]) < dataset.horizon:
                raise RuntimeError("Invalid overlapping action derangement")
            far_same_episode += 1
    count = len(metadata)
    return {
        "deterministic": True,
        "selection_rule": "different_episode_or_nonoverlapping_same_episode",
        "whole_action_sequence": True,
        "sequence_steps": int(dataset.horizon),
        "pairs": count,
        "cross_episode_pairs": cross_episode,
        "far_same_episode_pairs": far_same_episode,
        "cross_episode_fraction": cross_episode / count,
        "far_same_episode_fraction": far_same_episode / count,
    }


@torch.no_grad()
def _rollout_metrics_at_horizon(
    model: VisualLinearKoopman,
    loader: DataLoader,
    normalizers: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = loader.dataset
    horizon = int(dataset.horizon)
    if model.robot_dim < 21:
        raise ValueError("PickCube rollout metrics require the 21-D robot state")
    robot_mean = torch.as_tensor(normalizers["robot_mean"], device=device)
    robot_std = torch.as_tensor(normalizers["robot_std"], device=device)
    feature_mean = torch.as_tensor(normalizers["feature_mean"], device=device)
    feature_std = torch.as_tensor(normalizers["feature_std"], device=device)
    accumulator = _new_error_accumulator()
    metadata = dataset.window_metadata
    try:
        permutation = _deterministic_action_derangement(dataset)
        randomized_actions = dataset.action_windows(permutation.tolist())
        if randomized_actions.shape != (
            len(dataset),
            horizon,
            model.action_dim,
        ):
            raise RuntimeError("Deranged action windows have an unexpected shape")
        derangement = _action_ablation_metadata(dataset, permutation)
    except ValueError as exc:
        randomized_actions = None
        derangement = {
            "deterministic": True,
            "available": False,
            "reason": str(exc),
            "whole_action_sequence": True,
            "sequence_steps": horizon,
        }

    actual_robot_square = 0.0
    randomized_robot_square = 0.0
    zero_robot_square = 0.0
    robot_elements = 0
    position = 0
    for batch in loader:
        batch_size = int(batch["robot"].shape[0])
        batch_episodes = tuple(str(value) for value in batch["episode_id"])
        batch_starts = tuple(int(value) for value in batch["start"])
        expected_metadata = metadata[position : position + batch_size]
        if tuple(zip(batch_episodes, batch_starts, strict=True)) != expected_metadata:
            raise RuntimeError("Rollout evaluation DataLoader must not shuffle windows")

        robot = batch["robot"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        initial_state = model.make_state(robot[:, 0], features[:, 0])
        predicted_states, _ = model.rollout(initial_state, actions)
        predicted_robot = predicted_states[:, -1, : model.robot_dim]
        predicted_feature = model.decode_visual(
            predicted_states[:, -1, model.robot_dim :]
        )
        initial_reconstructed_feature = model.decode_visual(
            model.encode_visual(features[:, 0])
        )

        prediction_r = predicted_robot * robot_std + robot_mean
        target_r = robot[:, horizon] * robot_std + robot_mean
        hold_r = robot[:, 0] * robot_std + robot_mean
        prediction_f = predicted_feature * feature_std + feature_mean
        target_f = features[:, horizon] * feature_std + feature_mean
        hold_f = initial_reconstructed_feature * feature_std + feature_mean
        oracle_f = model.decode_visual(
            model.encode_visual(features[:, horizon])
        ) * feature_std + feature_mean
        residual_r = prediction_r - target_r
        residual_f = prediction_f - target_f
        accumulator["robot_square"] += float(residual_r.square().sum())
        accumulator["robot_absolute"] += float(residual_r.abs().sum())
        accumulator["robot_elements"] += residual_r.numel()
        accumulator["robot_hold_square"] += float((hold_r - target_r).square().sum())

        robot_slices = {
            "arm_qpos": residual_r[:, 0:7],
            "gripper_qpos": residual_r[:, 7:9],
            "arm_qvel": residual_r[:, 9:16],
            "gripper_qvel": residual_r[:, 16:18],
            "tcp": residual_r[:, 18:21],
        }
        for name, values in robot_slices.items():
            accumulator[f"{name}_square"] += float(values.square().sum())
            accumulator[f"{name}_elements"] += values.numel()
        accumulator["feature_square"] += float(residual_f.square().sum())
        accumulator["feature_elements"] += residual_f.numel()
        accumulator["feature_hold_square"] += float((hold_f - target_f).square().sum())
        accumulator["feature_oracle_reconstruction_square"] += float(
            (oracle_f - target_f).square().sum()
        )

        zero_states, _ = model.rollout(initial_state, torch.zeros_like(actions))
        zero_robot = zero_states[:, -1, : model.robot_dim]
        actual_residual = predicted_robot - robot[:, horizon]
        zero_residual = zero_robot - robot[:, horizon]
        actual_robot_square += float(actual_residual.square().sum())
        zero_robot_square += float(zero_residual.square().sum())
        robot_elements += actual_residual.numel()
        if randomized_actions is not None:
            randomized_batch = randomized_actions[
                position : position + batch_size
            ].to(device, non_blocking=True)
            randomized_states, _ = model.rollout(initial_state, randomized_batch)
            randomized_robot = randomized_states[:, -1, : model.robot_dim]
            randomized_robot_square += float(
                (randomized_robot - robot[:, horizon]).square().sum()
            )
        position += batch_size

    if position != len(dataset) or robot_elements <= 0:
        raise RuntimeError("Rollout evaluation did not consume every window")
    actual_rmse = float(np.sqrt(actual_robot_square / robot_elements))
    zero_rmse = float(np.sqrt(zero_robot_square / robot_elements))
    ablation: dict[str, Any] = {
        "windows": len(dataset),
        "horizon": horizon,
        "actual_robot_normalized_rmse": actual_rmse,
        "zero_robot_normalized_rmse": zero_rmse,
        "zero_over_actual": zero_rmse / max(actual_rmse, 1e-12),
        "derangement": derangement,
    }
    if randomized_actions is not None:
        randomized_rmse = float(np.sqrt(randomized_robot_square / robot_elements))
        ablation.update(
            {
                "random_robot_normalized_rmse": randomized_rmse,
                "random_over_actual": randomized_rmse / max(actual_rmse, 1e-12),
                # Compatibility aliases for existing ablation summaries. The
                # implementation is a dataset-level whole-sequence derangement,
                # never the former batch-local roll(1).
                "shuffled_robot_normalized_rmse": randomized_rmse,
                "shuffled_over_actual": randomized_rmse / max(actual_rmse, 1e-12),
            }
        )
    errors: dict[str, Any] = _finish_errors(accumulator)
    errors["windows"] = len(dataset)
    return errors, ablation


@torch.no_grad()
def rollout_metrics(
    model: VisualLinearKoopman,
    loaders: DataLoader | Mapping[int, DataLoader],
    normalizers: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    primary_horizon: int | None = None,
    requested_horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Report each horizon from its own all-legal-start dataset."""

    model.eval()
    if isinstance(loaders, Mapping):
        loader_by_horizon = {int(key): value for key, value in loaders.items()}
    else:
        loader_by_horizon = {int(loaders.dataset.horizon): loaders}
    if not loader_by_horizon:
        raise ValueError("At least one rollout evaluation loader is required")
    for horizon, loader in loader_by_horizon.items():
        if int(loader.dataset.horizon) != horizon:
            raise ValueError("Evaluation loader key and dataset horizon differ")
    requested = tuple(
        sorted(
            set(
                int(value)
                for value in (
                    requested_horizons
                    if requested_horizons is not None
                    else loader_by_horizon
                )
            )
        )
    )
    if any(horizon < 1 for horizon in requested):
        raise ValueError("Evaluation horizons must be positive")

    horizon_metrics: dict[str, dict[str, Any]] = {}
    action_ablations: dict[str, dict[str, Any]] = {}
    for horizon in requested:
        loader = loader_by_horizon.get(horizon)
        if loader is None:
            horizon_metrics[str(horizon)] = {"windows": 0, "available": False}
            continue
        errors, ablation = _rollout_metrics_at_horizon(
            model, loader, normalizers, device
        )
        horizon_metrics[str(horizon)] = errors
        action_ablations[str(horizon)] = ablation

    primary_horizon = (
        max(loader_by_horizon) if primary_horizon is None else int(primary_horizon)
    )
    if primary_horizon not in loader_by_horizon:
        raise ValueError("primary_horizon has no evaluation loader")
    one_step_ablation = action_ablations.get("1")
    if one_step_ablation is None:
        raise ValueError("A one-step loader is required for action ablation")

    b_matrix = model.B.detach().to(dtype=torch.float64, device="cpu")
    b_column_norms = torch.linalg.vector_norm(b_matrix, dim=0)
    b_singular_values = torch.linalg.svdvals(b_matrix)
    largest_singular_value = float(b_singular_values.max())
    rank_tolerance = (
        max(b_matrix.shape)
        * torch.finfo(b_matrix.dtype).eps
        * largest_singular_value
    )
    numerical_rank = int((b_singular_values > rank_tolerance).sum())
    if model.action_dim == 8:
        action_labels = [*(f"arm_joint_{index + 1}" for index in range(7)), "gripper"]
    else:
        action_labels = [f"action_{index}" for index in range(model.action_dim)]
    readout_matrix = model.readout_matrix()
    column_norm_values = [float(value) for value in b_column_norms]
    return {
        "windows": len(loader_by_horizon[primary_horizon].dataset),
        "primary_horizon": primary_horizon,
        "horizons": horizon_metrics,
        "one_step_action_ablation": one_step_ablation,
        "action_ablation_by_horizon": action_ablations,
        "B_frobenius_norm": float(torch.linalg.matrix_norm(b_matrix)),
        "B_column_norms": column_norm_values,
        "B_column_norms_by_action": dict(
            zip(action_labels, column_norm_values, strict=True)
        ),
        "B_numerical_rank": numerical_rank,
        "B_rank_tolerance": rank_tolerance,
        "B_singular_values": [float(value) for value in b_singular_values],
        "readout_frobenius_norm": float(
            torch.linalg.matrix_norm(readout_matrix).cpu()
        ),
        "spectral_radius": float(torch.linalg.eigvals(model.A).abs().max().cpu()),
    }


def train(
    config: TrainConfig,
    *,
    device_name: str = "auto",
    resume: str | Path | None = None,
) -> dict[str, Any]:
    config = TrainConfig(
        **{
            **asdict(config),
            "trajectory_h5": Path(config.trajectory_h5).expanduser().resolve(),
            "feature_h5": Path(config.feature_h5).expanduser().resolve(),
            "output_dir": Path(config.output_dir).expanduser().resolve(),
        }
    )
    validate_train_config(config)
    if not config.trajectory_h5.is_file() or not config.feature_h5.is_file():
        raise FileNotFoundError("Trajectory and feature HDF5 files must exist")
    resume_path = None if resume is None else Path(resume).expanduser().resolve()
    if config.output_dir.exists() and resume_path is None:
        raise FileExistsError(
            f"Refusing to reuse an existing run directory: {config.output_dir}"
        )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    _seed_everything(config.seed)
    device = _resolve_device(device_name)

    trajectory_digest, feature_digest = validate_data_provenance(config)
    episodes = list_episode_ids(config.trajectory_h5)
    if config.max_episodes is not None:
        episodes = episodes[: config.max_episodes]
    splits = split_episode_ids(episodes, config.seed)
    if any(not split for split in splits.values()):
        raise ValueError("Training requires non-empty train/val/test episode splits")
    normalizers = fit_normalizers(
        config.trajectory_h5,
        config.feature_h5,
        splits["train"],
        feature_key=config.feature_key,
    )
    serialized_config = {
        **asdict(config),
        "trajectory_h5": str(config.trajectory_h5),
        "feature_h5": str(config.feature_h5),
        "output_dir": str(config.output_dir),
        "device": str(device),
        "episode_splits": splits,
        "trajectory_sha256": trajectory_digest,
        "feature_sha256": feature_digest,
        "resume_checkpoint": None if resume_path is None else str(resume_path),
    }
    datasets: dict[str, VisualDataset] = {}
    evaluation_datasets: dict[int, VisualDataset] = {}
    wandb_run: Any = None
    try:
        for name, episode_ids in splits.items():
            lazy_dataset = VisualWindowDataset(
                config.trajectory_h5,
                config.feature_h5,
                episode_ids,
                config.horizon,
                normalizers,
                feature_key=config.feature_key,
            )
            datasets[name] = (
                preload_dataset(lazy_dataset) if config.preload else lazy_dataset
            )
        loaders = {
            name: _loader(dataset, config=config, shuffle=name == "train")
            for name, dataset in datasets.items()
        }
        model = VisualLinearKoopman(
            robot_dim=int(datasets["train"].robot_dim),
            action_dim=int(datasets["train"].action_dim),
            visual_feature_dim=int(datasets["train"].feature_dim),
            visual_latent_dim=config.visual_latent_dim,
            encoder_hidden_dims=config.encoder_hidden_dims,
            transform_mode=config.transform_mode,
        ).to(device)
        _assert_finite_parameters(model)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.min_lr,
        )
        metrics_path = config.output_dir / "metrics.jsonl"
        history = _read_history(metrics_path) if resume_path is not None else []
        start_epoch = 0
        best_validation_observable = float("inf")
        best_validation_total_at_best = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        elapsed_before = 0.0
        if resume_path is not None:
            resume_model, resume_payload = load_checkpoint(
                resume_path, map_location=device
            )
            if resume_model.architecture() != model.architecture():
                raise ValueError(
                    "Resume checkpoint architecture does not match the requested run"
                )
            checkpoint_config = resume_payload.get("config", {})
            if checkpoint_config.get("trajectory_sha256") != trajectory_digest:
                raise ValueError("Resume checkpoint trajectory provenance differs")
            if checkpoint_config.get("feature_sha256") != feature_digest:
                raise ValueError("Resume checkpoint feature provenance differs")
            model.load_state_dict(resume_payload["model"])
            if resume_payload.get("optimizer") is None:
                raise ValueError("Resume checkpoint does not contain optimizer state")
            optimizer.load_state_dict(resume_payload["optimizer"])
            if resume_payload.get("scheduler") is not None:
                scheduler.load_state_dict(resume_payload["scheduler"])
            training_state = resume_payload.get("training_state") or {}
            # The loop uses zero-based ``epoch_index`` and reports
            # ``epoch_number = epoch_index + 1``.  Resuming from checkpoint N
            # therefore starts the loop at index N, which produces epoch N+1
            # without skipping an epoch.
            start_epoch = int(resume_payload["epoch"])
            best_validation_observable = float(
                training_state.get(
                    "best_validation_observable",
                    resume_payload["best_validation"],
                )
            )
            best_validation_total_at_best = float(
                training_state.get("best_validation_total_at_best", float("inf"))
            )
            best_epoch = int(training_state.get("best_epoch", resume_payload["epoch"]))
            epochs_without_improvement = int(
                training_state.get("epochs_without_improvement", 0)
            )
            elapsed_before = float(resume_payload.get("elapsed_seconds", 0.0))
            if resume_payload.get("rng_state") is not None:
                _restore_rng_state(resume_payload["rng_state"])
            history = _reconcile_history(metrics_path, int(resume_payload["epoch"]))
        wandb_run = _start_wandb(config, serialized_config)
        stopped_early = False
        started = time.monotonic()
        for epoch_index in range(start_epoch, config.epochs):
            epoch_number = epoch_index + 1
            model.train()
            totals: dict[str, float] = {}
            gradient_norm_sum = 0.0
            gradient_norm_max = 0.0
            gradient_samples = 0
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            if not math.isfinite(epoch_lr) or epoch_lr < 0:
                raise FloatingPointError("Optimizer learning rate is invalid")
            for batch in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model, batch, device, config)
                loss.total.backward()
                gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip,
                    error_if_nonfinite=True,
                )
                gradient_norm = float(gradient_norm_tensor.detach())
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite gradient norm")
                optimizer.step()
                _assert_finite_parameters(model)
                batch_samples = int(batch["robot"].shape[0])
                gradient_norm_sum += gradient_norm * batch_samples
                gradient_norm_max = max(gradient_norm_max, gradient_norm)
                gradient_samples += batch_samples
                _aggregate_loss(totals, loss, batch_samples)
            train_values = _finish_loss(totals)
            validation_values = evaluate_loss(
                model, loaders["val"], device, config
            )
            _assert_finite_scalars(validation_values, "validation")
            validation_total = float(validation_values["total"])
            validation_observable = observable_validation(
                validation_values,
                config,
            )
            improved = validation_observable < (
                best_validation_observable - config.early_stopping_min_delta
            )
            if improved:
                best_validation_observable = validation_observable
                best_validation_total_at_best = validation_total
                best_epoch = epoch_number
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler.step(validation_observable)
            next_lr = float(optimizer.param_groups[0]["lr"])
            if not math.isfinite(next_lr) or next_lr < 0:
                raise FloatingPointError("Scheduler produced an invalid learning rate")
            if gradient_samples <= 0:
                raise RuntimeError("Training loader produced no samples")
            record = {
                "epoch": epoch_number,
                "train": train_values,
                "validation": validation_values,
                "lr": epoch_lr,
                "next_lr": next_lr,
                "gradient_norm": gradient_norm_sum / gradient_samples,
                "gradient_norm_max": gradient_norm_max,
                "validation_observable": validation_observable,
                "validation_total": validation_total,
                "best_validation_observable": best_validation_observable,
                "best_validation_total_at_best": best_validation_total_at_best,
                # Compatibility alias: this is the full training objective at
                # the observable-best checkpoint, not the selection metric.
                "best_validation_total": best_validation_total_at_best,
                "epochs_without_improvement": epochs_without_improvement,
            }
            _assert_finite_tree(record, "history")
            history.append(record)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            if wandb_run is not None:
                logged: dict[str, float | int] = {
                    "epoch": epoch_number,
                    "lr": epoch_lr,
                    "next_lr": next_lr,
                    "gradient_norm": record["gradient_norm"],
                    "gradient_norm_max": gradient_norm_max,
                    "validation/observable": validation_observable,
                }
                logged.update(_flatten_numeric("train", train_values))
                logged.update(_flatten_numeric("validation", validation_values))
                wandb_run.log(logged, step=epoch_number)

            elapsed = elapsed_before + time.monotonic() - started
            checkpoint_state = {
                "best_validation_observable": best_validation_observable,
                "best_validation_total_at_best": best_validation_total_at_best,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "stopped_early": False,
            }
            save_checkpoint(
                config.output_dir / "last.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_number,
                best_validation=best_validation_observable,
                config=serialized_config,
                normalizers=normalizers,
                elapsed_seconds=elapsed,
                rng_state=_capture_rng_state(),
                training_state=checkpoint_state,
            )
            if improved:
                save_checkpoint(
                    config.output_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch_number,
                    best_validation=best_validation_observable,
                    config=serialized_config,
                    normalizers=normalizers,
                    elapsed_seconds=elapsed,
                    rng_state=_capture_rng_state(),
                    training_state=checkpoint_state,
                )
            if epoch_number % config.checkpoint_interval == 0:
                save_checkpoint(
                    config.output_dir / f"recovery_epoch_{epoch_number:04d}.pt",
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch_number,
                    best_validation=best_validation_observable,
                    config=serialized_config,
                    normalizers=normalizers,
                    elapsed_seconds=elapsed,
                    rng_state=_capture_rng_state(),
                    training_state=checkpoint_state,
                )
            if epoch_index == start_epoch or epoch_number % 10 == 0:
                print(
                    f"epoch={epoch_number:04d} "
                    f"train={train_values['total']:.6f} "
                    f"val_obs={validation_observable:.6f} "
                    f"val_total={validation_total:.6f} "
                    f"lr={epoch_lr:.3e} "
                    f"grad={record['gradient_norm']:.4f} "
                    f"rho={validation_values['spectral_radius']:.4f}",
                    flush=True,
                )
            if epochs_without_improvement >= config.patience:
                stopped_early = True
                break

        if not (config.output_dir / "best.pt").is_file():
            raise RuntimeError("Training ended without producing best.pt")
        best_model, best_payload = load_checkpoint(
            config.output_dir / "best.pt", map_location=device
        )
        model.load_state_dict(best_model.state_dict())
        _assert_finite_parameters(model)
        test_loss = evaluate_loss(model, loaders["test"], device, config)
        evaluation_datasets = make_evaluation_datasets(
            config,
            splits["test"],
            normalizers,
            reuse={config.horizon: datasets["test"]},
        )
        evaluation_loaders = {
            horizon: _loader(dataset, config=config, shuffle=False)
            for horizon, dataset in evaluation_datasets.items()
        }
        test_metrics = rollout_metrics(
            model,
            evaluation_loaders,
            normalizers,
            device,
            primary_horizon=config.horizon,
            requested_horizons=evaluation_horizons(config.horizon),
        )
        _assert_finite_tree(test_loss, "test_loss")
        _assert_finite_tree(test_metrics, "test_metrics")
        elapsed = elapsed_before + time.monotonic() - started
        checkpoint_path = config.output_dir / "best.pt"
        summary = {
            "best_epoch": best_epoch,
            "validation_selection_metric": "observable_validation",
            "best_validation_observable": best_validation_observable,
            "best_validation_total_at_best": best_validation_total_at_best,
            # Compatibility alias retaining the full training objective.
            "best_validation_total": best_validation_total_at_best,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "stop_reason": (
                "early_stopping" if stopped_early else "epochs_completed"
            ),
            "final_lr": float(optimizer.param_groups[0]["lr"]),
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "elapsed_seconds": elapsed,
            "checkpoint": str(checkpoint_path),
        }
        report = {
            "architecture": model.architecture(),
            "config": serialized_config,
            "data": {
                "trajectory_sha256": trajectory_digest,
                "feature_sha256": feature_digest,
                "episode_splits": splits,
                "window_counts": {
                    name: len(value) for name, value in datasets.items()
                },
                "test_evaluation_window_counts": {
                    str(horizon): (
                        len(evaluation_datasets[horizon])
                        if horizon in evaluation_datasets
                        else 0
                    )
                    for horizon in evaluation_horizons(config.horizon)
                },
                "preloaded": config.preload,
            },
            **summary,
            "summary": summary,
            "history": history,
        }
        _assert_finite_tree(report, "report")
        (config.output_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (config.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if wandb_run is not None:
            wandb_run.summary.update(_flatten_numeric("final", summary))
        print(json.dumps(summary, indent=2))
        return report
    finally:
        closed: set[int] = set()
        for dataset in (*datasets.values(), *evaluation_datasets.values()):
            if id(dataset) not in closed:
                dataset.close()
                closed.add(id(dataset))
        if wandb_run is not None:
            wandb_run.finish()


def _parse_args() -> tuple[TrainConfig, str, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-h5", type=Path, required=True)
    parser.add_argument("--feature-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--visual-latent-dim", type=int, default=16)
    parser.add_argument("--encoder-hidden-dims", type=int, nargs="+", default=(256, 64))
    parser.add_argument(
        "--transform-mode",
        choices=("identity", "learned", "learned_inverse", "learned_orthogonal"),
        default="identity",
    )
    parser.add_argument("--feature-key", default="resnet18")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a visual Koopman checkpoint containing optimizer state.",
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Materialize normalized windows in RAM before repeated epochs.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "disabled", "online"),
        default="offline",
    )
    parser.add_argument("--wandb-project", default="acmpc-visual-koopman")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Use only the first N episodes for a smoke test; omit for all episodes.",
    )
    parser.add_argument("--rollout-discount", type=float, default=0.99)
    parser.add_argument("--linear-weight", type=float, default=10.0)
    parser.add_argument("--robot-rollout-weight", type=float, default=2.0)
    parser.add_argument("--feature-reconstruction-weight", type=float, default=0.2)
    parser.add_argument("--future-feature-reconstruction-weight", type=float, default=0.2)
    parser.add_argument("--latent-variance-weight", type=float, default=0.1)
    parser.add_argument("--stability-weight", type=float, default=0.02)
    parser.add_argument("--identity-weight", type=float, default=1e-4)
    parser.add_argument("--transform-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--transform-condition-weight", type=float, default=1e-3)
    parser.add_argument("--transform-singular-value-weight", type=float, default=1e-2)
    parser.add_argument("--transform-minimum-singular-value", type=float, default=0.25)
    parser.add_argument("--target-latent-std", type=float, default=1.0)
    parser.add_argument("--spectral-radius-limit", type=float, default=1.02)
    parser.add_argument(
        "--allow-unverified-data",
        action="store_true",
        help="Allow data without causal replay and feature-cache provenance metadata.",
    )
    args = parser.parse_args()
    config = TrainConfig(
        trajectory_h5=args.trajectory_h5,
        feature_h5=args.feature_h5,
        output_dir=args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        horizon=args.horizon,
        visual_latent_dim=args.visual_latent_dim,
        encoder_hidden_dims=tuple(args.encoder_hidden_dims),
        transform_mode=args.transform_mode,
        feature_key=args.feature_key,
        seed=args.seed,
        workers=args.workers,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
        early_stopping_min_delta=args.early_stopping_min_delta,
        checkpoint_interval=args.checkpoint_interval,
        preload=args.preload,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_group=args.wandb_group,
        max_episodes=args.max_episodes,
        rollout_discount=args.rollout_discount,
        linear_weight=args.linear_weight,
        robot_rollout_weight=args.robot_rollout_weight,
        feature_reconstruction_weight=args.feature_reconstruction_weight,
        future_feature_reconstruction_weight=(
            args.future_feature_reconstruction_weight
        ),
        latent_variance_weight=args.latent_variance_weight,
        stability_weight=args.stability_weight,
        identity_weight=args.identity_weight,
        transform_reconstruction_weight=args.transform_reconstruction_weight,
        transform_condition_weight=args.transform_condition_weight,
        transform_singular_value_weight=args.transform_singular_value_weight,
        transform_minimum_singular_value=(
            args.transform_minimum_singular_value
        ),
        target_latent_std=args.target_latent_std,
        spectral_radius_limit=args.spectral_radius_limit,
        require_provenance=not args.allow_unverified_data,
    )
    return config, args.device, args.resume


if __name__ == "__main__":
    parsed_config, parsed_device, parsed_resume = _parse_args()
    train(parsed_config, device_name=parsed_device, resume=parsed_resume)
