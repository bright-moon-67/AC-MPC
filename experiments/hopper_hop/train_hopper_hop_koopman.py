"""Train a K-step Deep Koopman model on the MS-HopperHop dataset.

Optimized checkpoint logic (vs. the PandaReach trainer):
  * Two files: ``best.pt`` (best validation, for downstream BC/PPO) and
    ``latest.pt`` (periodic, full resume state: model + optimizer + rng +
    epoch + history).
  * Atomic writes (temp file + ``os.replace``): a crash never leaves a
    corrupt checkpoint.
  * Resume: if ``latest.pt`` exists, training continues from that epoch with
    the same optimizer/RNG state.
  * Early stopping: training halts after ``patience`` epochs without
    validation improvement.

The ``best.pt`` payload keeps the PandaReach contract
(``model_state``/``architecture``/``state_kind``/``normalizer``/...) so the
existing ``load_koopman`` consumers work; ``state_kind`` is ``hopperhop`` and
the state/action dims are 15/4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from antmaze_ac.koopman.losses import koopman_loss
from antmaze_ac.koopman.model import DeepKoopman

STATE_DIM = 15
ACTION_DIM = 4


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 500
    batch_size: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    lift_dim: int = 48
    hidden_dims: tuple[int, ...] = (256, 256)
    seed: int = 43
    k_step: int = 20
    linear_weight: float = 10.0
    rollout_weight: float = 1.0
    stability_weight: float = 0.1
    latent_std_weight: float = 0.1
    identity_weight: float = 1e-4
    spectral_radius_limit: float = 0.95
    target_latent_std: float = 1.0
    # checkpoint / early stopping
    checkpoint_every: int = 25
    patience: int = 40
    max_windows: int = 1_000_000  # cap train windows (memory bound)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_mask(
    episode_ids: np.ndarray, selected_ids: np.ndarray
) -> np.ndarray:
    return np.isin(episode_ids, selected_ids)


def load_dataset(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {
        "state", "action", "next_state", "episode_id",
        "train_episode_ids", "validation_episode_ids", "test_episode_ids",
    }
    missing = required - data.keys()
    if missing:
        raise KeyError(f"Dataset is missing fields: {sorted(missing)}")
    if data["state"].shape != data["next_state"].shape:
        raise ValueError("state and next_state shapes differ")
    if data["state"].shape[1:] != (STATE_DIM,) or \
            data["action"].shape[1:] != (ACTION_DIM,):
        raise ValueError(
            f"Expected state [N,{STATE_DIM}] and action [N,{ACTION_DIM}]"
        )
    if not all(
        np.isfinite(data[name]).all()
        for name in ("state", "action", "next_state")
    ):
        raise FloatingPointError("Dataset contains NaN or Inf")

    masks = {
        split: _episode_mask(data["episode_id"], data[f"{split}_episode_ids"])
        for split in ("train", "validation", "test")
    }
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("Every episode split must contain transitions")
    if np.any(
        masks["train"].astype(np.int8)
        + masks["validation"].astype(np.int8)
        + masks["test"].astype(np.int8)
        != 1
    ):
        raise ValueError("Episode splits overlap or omit transitions")
    return data, masks


def fit_normalizer(
    state: np.ndarray, next_state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Center/scale every physical state dim by train-data mean/std."""
    samples = np.concatenate((state, next_state), axis=0).astype(np.float64)
    center = samples.mean(axis=0)
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
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [batch, K+1, 15] state windows and [batch, K, 4] action windows.

    Uses only the initial true state (open-loop model identification, matching
    the ``koopman_loss`` convention).
    """
    if k_step < 1:
        raise ValueError("k_step must be positive")
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    step_index = data.get("step_index")
    state_windows: list[np.ndarray] = []
    action_windows: list[np.ndarray] = []
    for selected_episode in np.asarray(selected_episode_ids, dtype=np.int64):
        indices = np.flatnonzero(episode_id == selected_episode)
        if len(indices) < k_step:
            continue
        if step_index is not None:
            episode_steps = np.asarray(step_index[indices], dtype=np.int64)
            if not np.array_equal(
                episode_steps, np.arange(len(indices), dtype=np.int64)
            ):
                raise ValueError(
                    f"Episode {selected_episode} has non-consecutive steps"
                )
        chain_error = np.max(
            np.abs(
                data["next_state"][indices[:-1]]
                - data["state"][indices[1:]]
            )
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
            state_windows.append(
                ((physical_states - center) / scale).astype(np.float32)
            )
            action_windows.append(
                data["action"][transition_indices].astype(np.float32)
            )
            if max_windows is not None and len(state_windows) >= max_windows:
                break
        if max_windows is not None and len(state_windows) >= max_windows:
            break
    states = np.asarray(state_windows, dtype=np.float32)
    actions = np.asarray(action_windows, dtype=np.float32)
    if states.shape[1:] != (k_step + 1, STATE_DIM):
        raise RuntimeError("Invalid Koopman state-window shape")
    if actions.shape[1:] != (k_step, ACTION_DIM):
        raise RuntimeError("Invalid Koopman action-window shape")
    return states, actions


def _make_loader(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(states), torch.from_numpy(actions))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def rollout_normalized_mse(
    model: DeepKoopman, batches: DataLoader, device: torch.device
) -> float:
    squared_error = 0.0
    elements = 0
    model.eval()
    for states, actions in batches:
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        prediction, _ = model.rollout(states[:, 0], actions)
        target = states[:, 1:]
        squared_error += float((prediction - target).square().sum())
        elements += target.numel()
    return squared_error / elements


@torch.no_grad()
def prediction_metrics(
    model: DeepKoopman,
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    state = data["state"][mask]
    action = data["action"][mask]
    target = data["next_state"][mask]
    predictions: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(state), batch_size):
        stop = start + batch_size
        normalized = torch.as_tensor(
            (state[start:stop] - center) / scale,
            dtype=torch.float32,
            device=device,
        )
        command = torch.as_tensor(
            action[start:stop], dtype=torch.float32, device=device
        )
        predicted_normalized, _ = model(normalized, command)
        predictions.append(
            predicted_normalized.cpu().numpy() * scale + center
        )
    predicted = np.concatenate(predictions, axis=0)
    residual = predicted - target
    hold_residual = state - target
    groups = {
        "qpos": slice(0, 6),
        "qvel": slice(6, 13),
        "contact": slice(13, 15),
        "all": slice(0, 15),
    }
    metrics: dict[str, Any] = {"transitions": int(len(state))}
    for name, indices in groups.items():
        group_residual = residual[:, indices]
        group_hold = hold_residual[:, indices]
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(np.square(group_residual)))),
            "mae": float(np.mean(np.abs(group_residual))),
            "hold_rmse": float(np.sqrt(np.mean(np.square(group_hold)))),
            "hold_mae": float(np.mean(np.abs(group_hold))),
            "residual_mean": np.mean(group_residual, axis=0).tolist(),
        }
    normalized_residual = residual / scale
    metrics["normalized_mse"] = float(np.mean(np.square(normalized_residual)))
    return metrics


@torch.no_grad()
def rollout_prediction_metrics(
    model: DeepKoopman,
    states: np.ndarray,
    actions: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(states), batch_size):
        stop = start + batch_size
        initial = torch.as_tensor(
            states[start:stop, 0], dtype=torch.float32, device=device
        )
        command = torch.as_tensor(
            actions[start:stop], dtype=torch.float32, device=device
        )
        predicted_normalized, _ = model.rollout(initial, command)
        predictions.append(predicted_normalized.cpu().numpy())
    predicted_normalized = np.concatenate(predictions, axis=0)
    target_normalized = states[:, 1:]
    predicted = predicted_normalized * scale + center
    target = target_normalized * scale + center
    initial = states[:, :1] * scale + center
    hold = np.broadcast_to(initial, target.shape)
    groups = {
        "qpos": slice(0, 6),
        "qvel": slice(6, 13),
        "contact": slice(13, 15),
        "all": slice(0, 15),
    }
    result: dict[str, Any] = {
        "windows": int(len(states)),
        "k_step": int(actions.shape[1]),
        "all_steps": {},
        "normalized_mse_all_steps": float(
            np.mean(np.square(predicted_normalized - target_normalized))
        ),
        "horizons": {},
    }
    for name, indices in groups.items():
        group_residual = (predicted - target)[..., indices]
        group_hold = (hold - target)[..., indices]
        result["all_steps"][name] = {
            "rmse": float(np.sqrt(np.mean(np.square(group_residual)))),
            "mae": float(np.mean(np.abs(group_residual))),
            "hold_rmse": float(np.sqrt(np.mean(np.square(group_hold)))),
            "hold_mae": float(np.mean(np.abs(group_hold))),
            "residual_mean": np.mean(
                group_residual, axis=tuple(range(group_residual.ndim - 1))
            ).tolist(),
        }
    requested_horizons = sorted({1, 5, 10, int(actions.shape[1])})
    for horizon in requested_horizons:
        if horizon > actions.shape[1]:
            continue
        result["horizons"][str(horizon)] = {}
        for name, indices in groups.items():
            group_residual = (predicted - target)[:, horizon - 1, indices]
            group_hold = (hold - target)[:, horizon - 1, indices]
            result["horizons"][str(horizon)][name] = {
                "rmse": float(np.sqrt(np.mean(np.square(group_residual)))),
                "mae": float(np.mean(np.abs(group_residual))),
                "hold_rmse": float(
                    np.sqrt(np.mean(np.square(group_hold)))
                ),
                "hold_mae": float(np.mean(np.abs(group_hold))),
                "residual_mean": np.mean(
                    group_residual, axis=0
                ).tolist(),
            }
    return result


def _capture_rng_state() -> dict[str, Any]:
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().cpu(),
        "cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    # torch.load(..., map_location=device) maps saved tensors to CUDA; the
    # RNG states must be CPU byte tensors, so force-copy to CPU explicitly.
    torch.random.set_rng_state(torch.as_tensor(state["torch"]).cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(device_state).cpu() for device_state in state["cuda"]]
        )


def train(
    dataset_path: Path,
    output_dir: Path,
    config: TrainConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )

    data, masks = load_dataset(dataset_path)
    center, scale = fit_normalizer(
        data["state"][masks["train"]], data["next_state"][masks["train"]]
    )
    window_arrays = {
        split: build_windows(
            data,
            data[f"{split}_episode_ids"],
            center,
            scale,
            k_step=config.k_step,
            max_windows=(
                config.max_windows if split == "train" else None
            ),
        )
        for split in ("train", "validation", "test")
    }
    loaders = {
        split: _make_loader(
            *window_arrays[split],
            batch_size=config.batch_size,
            shuffle=split == "train",
            seed=config.seed,
        )
        for split in ("train", "validation", "test")
    }

    model = DeepKoopman(
        state_dim=STATE_DIM,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    history_path = output_dir / "history.jsonl"

    # ---- resume from latest.pt if present ----
    start_epoch = 1
    best_validation = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    if latest_path.exists():
        payload = torch.load(
            latest_path, map_location=device, weights_only=False
        )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng_state(payload.get("rng_state"))
        start_epoch = int(payload["epoch"]) + 1
        best_validation = float(payload["best_validation"])
        best_epoch = int(payload.get("best_epoch", 0))
        history = list(payload.get("history", []))
        print(
            f"resumed Koopman training from epoch {start_epoch} "
            f"(best_validation={best_validation:.6g})",
            flush=True,
        )

    start_time = time.perf_counter()
    elapsed_base = 0.0
    if latest_path.exists() and history:
        elapsed_base = float(history[-1].get("elapsed_seconds", 0.0))
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.epochs + 1):
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

        validation_mse = rollout_normalized_mse(
            model, loaders["validation"], device
        )
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
            "elapsed_seconds": elapsed_base
            + time.perf_counter()
            - start_time,
        }
        history.append(epoch_record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            # best.pt: PandaReach-compatible payload for downstream BC/PPO.
            torch.save(
                {
                    "kind": "hopperhop_k_step_koopman",
                    "model_state": model.state_dict(),
                    "architecture": model.architecture(),
                    "normalizer": {
                        "center": torch.from_numpy(center),
                        "scale": torch.from_numpy(scale),
                        "fit": "train episodes only; per-dim mean/std",
                    },
                    "state_kind": "hopperhop",
                    "config": asdict(config),
                    "dataset_path": str(dataset_path.resolve()),
                    "dataset_sha256": _sha256(dataset_path),
                    "best_epoch": best_epoch,
                    "best_validation_rollout_normalized_mse": best_validation,
                    "k_step": config.k_step,
                    "history": config.history if hasattr(config, "history") else 0,
                    "split_episode_ids": {
                        split: torch.from_numpy(data[f"{split}_episode_ids"])
                        for split in ("train", "validation", "test")
                    },
                },
                best_path.with_name(best_path.name + ".tmp"),
            )
            os.replace(
                best_path.with_name(best_path.name + ".tmp"), best_path
            )
        else:
            epochs_without_improvement += 1

        # latest.pt: periodic resumable checkpoint (atomic write).
        if (
            epoch % config.checkpoint_every == 0
            or epoch == config.epochs
            or epochs_without_improvement == config.patience
        ):
            from antmaze_ac.koopman.checkpoint import save_checkpoint

            save_checkpoint(
                latest_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation=best_validation,
                config=asdict(config),
                normalizers={
                    "center": center.tolist(),
                    "scale": scale.tolist(),
                    "fit": "train episodes only; per-dim mean/std",
                },
                elapsed_seconds=epoch_record["elapsed_seconds"],
                rng_state=_capture_rng_state(),
                history=history,
            )

        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:04d} "
                f"train={epoch_record['train_total']:.6g} "
                f"val_nMSE={validation_mse:.6g} "
                f"rhoA={epoch_record['spectral_radius']:.6g} "
                f"elapsed={epoch_record['elapsed_seconds']:.1f}s",
                flush=True,
            )

        if epochs_without_improvement >= config.patience:
            print(
                f"early stopping at epoch {epoch} "
                f"(no improvement for {config.patience} epochs, "
                f"best={best_validation:.6g} @ epoch {best_epoch})",
                flush=True,
            )
            break

    # ---- final evaluation on the best checkpoint ----
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    split_metrics = {
        split: {
            "one_step": prediction_metrics(
                model,
                data,
                masks[split],
                center,
                scale,
                device,
                config.batch_size,
            ),
            "rollout": rollout_prediction_metrics(
                model,
                *window_arrays[split],
                center,
                scale,
                device,
                config.batch_size,
            ),
        }
        for split, mask in masks.items()
    }
    with torch.no_grad():
        spectral_radius = float(
            torch.linalg.eigvals(model.A).abs().max().cpu()
        )
    report = {
        "kind": "hopperhop_k_step_koopman_report",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint_path": str(best_path.resolve()),
        "device": str(device),
        "config": asdict(config),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_rollout_normalized_mse": float(
            checkpoint["best_validation_rollout_normalized_mse"]
        ),
        "elapsed_seconds": (
            elapsed_base + time.perf_counter() - start_time
        ),
        "spectral_radius_A": spectral_radius,
        "state_kind": "hopperhop",
        "normalizer": {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "fit": "train episodes only",
        },
        "metrics": split_metrics,
        "window_counts": {
            split: int(len(window_arrays[split][0]))
            for split in ("train", "validation", "test")
        },
        "history": history,
        "scope": (
            "K_step=20 open-loop model identification on MS-HopperHop "
            "PPO-collected multi-stage data (15-dim state, 4-dim action)"
        ),
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
        default=Path("runs/hopper_hop/data/hopperhop_koopman.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/hopper_hop/koopman_v2"),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lift-dim", type=int, default=48)
    parser.add_argument("--k-step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--max-windows", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lift_dim=args.lift_dim,
        k_step=args.k_step,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        patience=args.patience,
        max_windows=args.max_windows,
    )
    report = train(args.dataset, args.output_dir, config, device_name=args.device)
    print(
        f"best_epoch={report['best_epoch']} "
        f"best_val_nMSE={report['best_validation_rollout_normalized_mse']:.6g} "
        f"elapsed={report['elapsed_seconds']:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
