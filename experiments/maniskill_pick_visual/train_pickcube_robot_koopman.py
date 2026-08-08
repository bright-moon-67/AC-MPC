"""Train a robot-only K-step Deep Koopman model on PickCube transitions.

The PickCube causal dataset stores ``robot`` = [qpos9, qvel9, tcp_xyz3] (21)
and normalized ``actions`` (8, [-1, 1]) per trajectory group.  This trainer
fits a DeepKoopman on the robot dynamics only (no visual latent), producing
the frozen lift that the AC-MPC visual PPO actors (KMPC, AB-PQ) consume.

Mid-training artifacts (per project convention):
  * ``recovery_epoch_XXXX.pt`` every 25 epochs
  * ``metrics.jsonl`` appended+flushed every epoch

Run:
  python -m experiments.maniskill_pick_visual.train_pickcube_robot_koopman \
      --trajectory-h5 .data/maniskill/visual_pickcube_causal_v2_all997_seed43_retry1.h5 \
      --output-dir runs/pickcube_robot_koopman --epochs 100 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from antmaze_ac.koopman.losses import koopman_loss
from antmaze_ac.koopman.model import DeepKoopman

ROBOT_DIM = 21
ACTION_DIM = 8


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    lift_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 256)
    seed: int = 43
    k_step: int = 20
    history: int = 0
    linear_weight: float = 10.0
    rollout_weight: float = 1.0
    stability_weight: float = 0.1
    latent_std_weight: float = 0.1
    identity_weight: float = 1e-4
    spectral_radius_limit: float = 0.95
    target_latent_std: float = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_mask(episode_ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.isin(episode_ids, selected)


def _resolve_device(name: str) -> torch.device:
    return torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else ("cpu" if name == "auto" else name)
    )


def load_dataset(
    path: Path,
    split_seed: int = 31,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load PickCube transitions from either the causal h5 or a coverage npz.

    h5 (causal replay): per-trajectory groups with ``robot`` [T,21] and
    ``actions`` [T-1,8]; splits are created here.
    npz (coverage collector): already-flattened arrays with ``state``,
    ``action``, ``next_state``, ``episode_id``, ``step_index`` and the three
    ``*_episode_ids`` splits.

    Returns ``data`` (with all needed keys) and ``masks`` per split.
    """
    if path.suffix.lower() == ".npz":
        return _load_dataset_npz(path)
    return _load_dataset_h5(
        path, split_seed=split_seed,
        validation_fraction=validation_fraction, test_fraction=test_fraction,
    )


def _load_dataset_h5(
    path: Path,
    split_seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    episode_ids: list[int] = []
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys(), key=lambda k: int(k.split("_")[1]))
        for episode_index, key in enumerate(keys):
            group = handle[key]
            robot = np.asarray(group["robot"], dtype=np.float64)
            action = np.asarray(group["actions"], dtype=np.float64)
            if robot.ndim != 2 or robot.shape[1] != ROBOT_DIM:
                raise ValueError(
                    f"{key}: expected robot [T,{ROBOT_DIM}], got {robot.shape}"
                )
            if action.ndim != 2 or action.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"{key}: expected actions [T-1,{ACTION_DIM}], got {action.shape}"
                )
            if len(action) != len(robot) - 1:
                raise ValueError(f"{key}: action/robot length mismatch")
            states.append(robot[:-1])
            next_states.append(robot[1:])
            actions.append(action)
            episode_ids.extend([episode_index] * len(action))
    data = {
        "state": np.concatenate(states, axis=0).astype(np.float32),
        "next_state": np.concatenate(next_states, axis=0).astype(np.float32),
        "action": np.concatenate(actions, axis=0).astype(np.float32),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
    }
    if not all(
        np.isfinite(data[name]).all()
        for name in ("state", "action", "next_state")
    ):
        raise FloatingPointError("Dataset contains NaN or Inf")

    rng = np.random.RandomState(split_seed)
    all_episodes = np.unique(data["episode_id"])
    rng.shuffle(all_episodes)
    test_count = int(round(test_fraction * len(all_episodes)))
    validation_count = int(round(validation_fraction * len(all_episodes)))
    test_ids = all_episodes[:test_count]
    validation_ids = all_episodes[test_count : test_count + validation_count]
    train_ids = all_episodes[test_count + validation_count :]
    data.update(
        {
            "train_episode_ids": train_ids,
            "validation_episode_ids": validation_ids,
            "test_episode_ids": test_ids,
        }
    )
    masks = {
        split: _episode_mask(data["episode_id"], data[f"{split}_episode_ids"])
        for split in ("train", "validation", "test")
    }
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("Every episode split must contain transitions")
    return data, masks


def _load_dataset_npz(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load the flattened coverage npz produced by collect_pickcube_coverage."""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {
        "state",
        "action",
        "next_state",
        "episode_id",
        "train_episode_ids",
        "validation_episode_ids",
        "test_episode_ids",
    }
    missing = required - data.keys()
    if missing:
        raise KeyError(f"Coverage npz is missing fields: {sorted(missing)}")
    if data["state"].shape != data["next_state"].shape:
        raise ValueError("state and next_state shapes differ")
    if data["state"].shape[1:] != (ROBOT_DIM,) or data["action"].shape[1:] != (
        ACTION_DIM,
    ):
        raise ValueError(
            f"Expected state [N,{ROBOT_DIM}] and action [N,{ACTION_DIM}]"
        )
    if not all(
        np.isfinite(data[name]).all()
        for name in ("state", "action", "next_state")
    ):
        raise FloatingPointError("Coverage npz contains NaN or Inf")
    masks = {
        split: _episode_mask(data["episode_id"], data[f"{split}_episode_ids"])
        for split in ("train", "validation", "test")
    }
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("Every episode split must contain transitions")
    return data, masks


def fit_normalizer(
    state: np.ndarray,
    next_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.concatenate((state, next_state), axis=0).astype(np.float64)
    center = np.zeros(ROBOT_DIM, dtype=np.float64)
    center[:9] = samples[:, :9].mean(axis=0)
    center[18:21] = samples[:, 18:21].mean(axis=0)
    scale = np.sqrt(np.mean(np.square(samples - center), axis=0))
    scale = np.maximum(scale, 1e-4)
    return center.astype(np.float32), scale.astype(np.float32)


def build_windows(
    data: dict[str, np.ndarray],
    selected_episode_ids: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    k_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k_step < 1:
        raise ValueError("k_step must be positive")
    state_windows: list[np.ndarray] = []
    action_windows: list[np.ndarray] = []
    episode_id = data["episode_id"]
    for selected_episode in np.asarray(selected_episode_ids, dtype=np.int64):
        indices = np.flatnonzero(episode_id == selected_episode)
        if len(indices) < k_step:
            continue
        chain_error = np.max(
            np.abs(data["next_state"][indices[:-1]] - data["state"][indices[1:]])
        )
        if chain_error > 2e-5:
            raise ValueError(
                f"Episode {selected_episode} transition chain mismatch "
                f"{chain_error:.3e}"
            )
        for offset in range(len(indices) - k_step + 1):
            transition_indices = indices[offset : offset + k_step]
            physical_states = np.concatenate(
                (
                    data["state"][transition_indices],
                    data["next_state"][transition_indices[-1]][None],
                ),
                axis=0,
            )
            state_windows.append(((physical_states - center) / scale).astype(np.float32))
            action_windows.append(data["action"][transition_indices].astype(np.float32))
    states = np.asarray(state_windows, dtype=np.float32)
    actions = np.asarray(action_windows, dtype=np.float32)
    if states.shape[1:] != (k_step + 1, ROBOT_DIM):
        raise RuntimeError("Invalid Koopman state-window shape")
    if actions.shape[1:] != (k_step, ACTION_DIM):
        raise RuntimeError("Invalid Koopman action-window shape")
    return states, actions


def train(
    trajectory_h5: Path,
    output_dir: Path,
    config: TrainConfig,
    device_name: str = "auto",
) -> dict[str, Any]:
    config_used = config
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, masks = load_dataset(trajectory_h5, split_seed=config.seed)
    center, scale = fit_normalizer(
        data["state"][masks["train"]], data["next_state"][masks["train"]]
    )
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        states, actions = build_windows(
            data,
            data[f"{split}_episode_ids"],
            center,
            scale,
            k_step=config.k_step,
        )
        if len(states) == 0:
            raise RuntimeError(f"No {split} windows; k_step too large?")
        dataset = TensorDataset(
            torch.from_numpy(states), torch.from_numpy(actions)
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            drop_last=split == "train",
        )

    model = DeepKoopman(
        state_dim=ROBOT_DIM,
        action_dim=ACTION_DIM,
        lift_dim=config.lift_dim,
        hidden_dims=config.hidden_dims,
        activation="silu",
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = output_dir / "best.pt"
    best_validation = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        weighted_loss = 0.0
        sample_count = 0
        last_loss = None
        for states, actions in loaders["train"]:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = koopman_loss(
                model,
                states,
                actions,
                linear_weight=config.linear_weight,
                rollout_weight=config.rollout_weight,
                stability_weight=config.stability_weight,
                latent_std_weight=config.latent_std_weight,
                identity_weight=config.identity_weight,
                spectral_radius_limit=config.spectral_radius_limit,
                target_latent_std=config.target_latent_std,
            )
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite Koopman gradient")
            optimizer.step()
            count = len(states)
            weighted_loss += float(loss.total.detach()) * count
            sample_count += count
            last_loss = loss

        validation_mse = _rollout_normalized_mse(model, loaders["validation"], device)
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_total": weighted_loss / sample_count,
            "validation_rollout_normalized_mse": validation_mse,
            "spectral_radius": float(last_loss.spectral_radius.detach())
            if last_loss is not None
            else float("nan"),
            "stability_penalty": float(last_loss.stability.detach())
            if last_loss is not None
            else float("nan"),
        }
        history.append(epoch_record)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")
            handle.flush()
        is_best = validation_mse < best_validation
        if is_best:
            best_validation = validation_mse
            best_epoch = epoch
        checkpoint_payload = {
            "kind": "pickcube_robot_k_step_koopman",
            "model_state": model.state_dict(),
            "architecture": model.architecture(),
            "normalizer": {
                "center": torch.from_numpy(center),
                "scale": torch.from_numpy(scale),
                "fit": "train episodes only; qpos9 center=train mean; tcp_xyz3 center=train mean",
            },
            "state_kind": "q_qdot_tcp",
            "robot_dim": ROBOT_DIM,
            "action_dim": ACTION_DIM,
            "config": asdict(config),
            "dataset_path": str(trajectory_h5.resolve()),
            "dataset_sha256": _sha256(trajectory_h5),
            "best_epoch": best_epoch,
            "best_validation_rollout_normalized_mse": best_validation,
            "k_step": config.k_step,
            "history": config.history,
            "split_episode_ids": {
                split: torch.from_numpy(data[f"{split}_episode_ids"])
                for split in ("train", "validation", "test")
            },
        }
        if is_best:
            torch.save(checkpoint_payload, checkpoint_path)
        if epoch % 25 == 0:
            torch.save(
                {**checkpoint_payload, "epoch": epoch},
                output_dir / f"recovery_epoch_{epoch:04d}.pt",
            )
        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            elapsed = time.perf_counter() - start_time
            print(
                f"epoch={epoch:04d} train={epoch_record['train_total']:.6g} "
                f"val_nMSE={validation_mse:.6g} "
                f"rhoA={epoch_record['spectral_radius']:.6g} "
                f"stab={epoch_record['stability_penalty']:.6g} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    report = {
        "kind": "pickcube_robot_k_step_koopman_report",
        "dataset_path": str(trajectory_h5.resolve()),
        "dataset_sha256": _sha256(trajectory_h5),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "device": str(device),
        "config": asdict(config),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_rollout_normalized_mse": float(
            checkpoint["best_validation_rollout_normalized_mse"]
        ),
        "elapsed_seconds": time.perf_counter() - start_time,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "window_counts": {
            split: int(len(loaders[split].dataset))
            for split in ("train", "validation", "test")
        },
        "history": history,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _rollout_normalized_mse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    total = 0.0
    elements = 0
    model.eval()
    with torch.no_grad():
        for states, actions in loader:
            states = states.to(device)
            actions = actions.to(device)
            initial = states[:, 0]
            lifted = model.lift(initial)
            for step in range(actions.shape[1]):
                lifted = (
                    lifted @ model.A.mT + actions[:, step] @ model.B.mT
                )
            predicted = lifted @ model.C.mT
            error = (predicted - states[:, -1]).square().sum()
            total += float(error)
            elements += states.shape[0] * ROBOT_DIM
    model.train()
    return total / elements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pickcube_robot_koopman"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lift-dim", type=int, default=32)
    parser.add_argument("--k-step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = train(
        args.trajectory_h5,
        args.output_dir,
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lift_dim=args.lift_dim,
            k_step=args.k_step,
            seed=args.seed,
        ),
        device_name=args.device,
    )
    concise = {
        "checkpoint_path": report["checkpoint_path"],
        "best_epoch": report["best_epoch"],
        "best_validation_rollout_normalized_mse": report[
            "best_validation_rollout_normalized_mse"
        ],
        "elapsed_seconds": report["elapsed_seconds"],
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
