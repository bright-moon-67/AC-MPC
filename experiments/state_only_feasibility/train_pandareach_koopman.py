"""Train a K-step Deep Koopman model on the PandaReach pilot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
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


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 500
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


def _episode_mask(
    episode_ids: np.ndarray,
    selected_ids: np.ndarray,
) -> np.ndarray:
    return np.isin(episode_ids, selected_ids)


def load_dataset(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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
        raise KeyError(f"Dataset is missing fields: {sorted(missing)}")
    if data["state"].shape != data["next_state"].shape:
        raise ValueError("state and next_state shapes differ")
    if data["state"].shape[1:] != (17,) or data["action"].shape[1:] != (7,):
        raise ValueError("Expected state [N,17] and action [N,7]")
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
    state: np.ndarray,
    next_state: np.ndarray,
    *,
    state_kind: str = "q_qdot_error",
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.concatenate((state, next_state), axis=0).astype(np.float64)
    center = np.zeros(17, dtype=np.float64)
    center[:7] = samples[:, :7].mean(axis=0)
    if state_kind == "q_qdot_tcp":
        center[14:17] = samples[:, 14:17].mean(axis=0)
    elif state_kind != "q_qdot_error":
        raise ValueError(f"Unsupported state_kind {state_kind!r}")
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
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    step_index = data.get("step_index")
    for selected_episode in np.asarray(selected_episode_ids, dtype=np.int64):
        indices = np.flatnonzero(episode_id == selected_episode)
        if len(indices) < k_step:
            raise ValueError(
                f"Episode {selected_episode} has only {len(indices)} "
                f"transitions for K_step={k_step}"
            )
        if step_index is not None:
            episode_steps = np.asarray(step_index[indices], dtype=np.int64)
            if not np.array_equal(
                episode_steps,
                np.arange(len(indices), dtype=np.int64),
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
    states = np.asarray(state_windows, dtype=np.float32)
    actions = np.asarray(action_windows, dtype=np.float32)
    if states.shape[1:] != (k_step + 1, 17):
        raise RuntimeError("Invalid Koopman state-window shape")
    if actions.shape[1:] != (k_step, 7):
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
    model: DeepKoopman,
    batches: DataLoader,
    device: torch.device,
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
    tail_name: str = "e",
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
            action[start:stop],
            dtype=torch.float32,
            device=device,
        )
        predicted_normalized, _ = model(normalized, command)
        prediction = (
            predicted_normalized.cpu().numpy() * scale + center
        )
        predictions.append(prediction)
    predicted = np.concatenate(predictions, axis=0)
    residual = predicted - target
    hold_residual = state - target
    groups = {
        "q": slice(0, 7),
        "qdot": slice(7, 14),
        tail_name: slice(14, 17),
        "all": slice(0, 17),
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
    metrics["normalized_mse"] = float(
        np.mean(np.square(normalized_residual))
    )
    return metrics


def _physical_error_metrics(
    residual: np.ndarray,
    hold_residual: np.ndarray,
    tail_name: str = "e",
) -> dict[str, Any]:
    groups = {
        "q": slice(0, 7),
        "qdot": slice(7, 14),
        tail_name: slice(14, 17),
        "all": slice(0, 17),
    }
    metrics: dict[str, Any] = {}
    for name, indices in groups.items():
        group_residual = residual[..., indices]
        group_hold = hold_residual[..., indices]
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(np.square(group_residual)))),
            "mae": float(np.mean(np.abs(group_residual))),
            "hold_rmse": float(np.sqrt(np.mean(np.square(group_hold)))),
            "hold_mae": float(np.mean(np.abs(group_hold))),
            "residual_mean": np.mean(
                group_residual,
                axis=tuple(range(group_residual.ndim - 1)),
            ).tolist(),
        }
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
    tail_name: str = "e",
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(states), batch_size):
        stop = start + batch_size
        initial = torch.as_tensor(
            states[start:stop, 0],
            dtype=torch.float32,
            device=device,
        )
        command = torch.as_tensor(
            actions[start:stop],
            dtype=torch.float32,
            device=device,
        )
        predicted_normalized, _ = model.rollout(initial, command)
        predictions.append(predicted_normalized.cpu().numpy())
    predicted_normalized = np.concatenate(predictions, axis=0)
    target_normalized = states[:, 1:]
    predicted = predicted_normalized * scale + center
    target = target_normalized * scale + center
    initial = states[:, :1] * scale + center
    hold = np.broadcast_to(initial, target.shape)
    result: dict[str, Any] = {
        "windows": int(len(states)),
        "k_step": int(actions.shape[1]),
        "all_steps": _physical_error_metrics(
            predicted - target,
            hold - target,
            tail_name,
        ),
        "normalized_mse_all_steps": float(
            np.mean(np.square(predicted_normalized - target_normalized))
        ),
        "horizons": {},
    }
    requested_horizons = sorted(
        {1, 5, 10, int(actions.shape[1])}
    )
    for horizon in requested_horizons:
        if horizon > actions.shape[1]:
            continue
        result["horizons"][str(horizon)] = _physical_error_metrics(
            predicted[:, horizon - 1] - target[:, horizon - 1],
            hold[:, horizon - 1] - target[:, horizon - 1],
            tail_name,
        )
    return result


def train(
    dataset_path: Path,
    output_dir: Path,
    config: TrainConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    if config.history != 0:
        raise ValueError("This state-only experiment requires history=0")
    if config.k_step != 20:
        raise ValueError("The formal PandaReach configuration requires K_step=20")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )

    data, masks = load_dataset(dataset_path)
    state_kind = (
        str(np.asarray(data["state_kind"]).item())
        if "state_kind" in data
        else "q_qdot_error"
    )
    tail_name = "tcp" if state_kind == "q_qdot_tcp" else "e"
    center, scale = fit_normalizer(
        data["state"][masks["train"]],
        data["next_state"][masks["train"]],
        state_kind=state_kind,
    )
    window_arrays = {
        split: build_windows(
            data,
            data[f"{split}_episode_ids"],
            center,
            scale,
            k_step=config.k_step,
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
        state_dim=17,
        action_dim=7,
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
                model.parameters(),
                config.gradient_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite Koopman gradient")
            optimizer.step()
            count = len(states)
            weighted_loss += float(loss.total.detach()) * count
            sample_count += count
            last_loss = loss

        validation_mse = rollout_normalized_mse(
            model,
            loaders["validation"],
            device,
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
            "kind": "pandareach_k_step_koopman",
            "model_state": model.state_dict(),
            "architecture": model.architecture(),
            "normalizer": {
                "center": torch.from_numpy(center),
                "scale": torch.from_numpy(scale),
                "fit": (
                    "train episodes only; q/tcp center=train mean for "
                    "q_qdot_tcp, q center=train mean otherwise"
                ),
            },
            "state_kind": state_kind,
            "config": asdict(config),
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": _sha256(dataset_path),
            "best_epoch": best_epoch,
            "best_validation_rollout_normalized_mse": best_validation,
            "k_step": config.k_step,
            "history": config.history,
            "split_episode_ids": {
                split: torch.from_numpy(
                    data[f"{split}_episode_ids"]
                )
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
                f"epoch={epoch:04d} "
                f"train={epoch_record['train_total']:.6g} "
                f"val_nMSE={validation_mse:.6g} "
                f"rhoA={epoch_record['spectral_radius']:.6g} "
                f"stab={epoch_record['stability_penalty']:.6g} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
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
                tail_name,
            ),
            "rollout": rollout_prediction_metrics(
                model,
                *window_arrays[split],
                center,
                scale,
                device,
                config.batch_size,
                tail_name,
            ),
        }
        for split, mask in masks.items()
    }
    with torch.no_grad():
        spectral_radius = float(
            torch.linalg.eigvals(model.A).abs().max().cpu()
        )
    report = {
        "kind": "pandareach_k_step_koopman_report",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "device": str(device),
        "config": asdict(config),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_rollout_normalized_mse": float(
            checkpoint["best_validation_rollout_normalized_mse"]
        ),
        "elapsed_seconds": time.perf_counter() - start_time,
        "spectral_radius_A": spectral_radius,
        "state_kind": state_kind,
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
            "K_step=20 model-identification rollout loss with history=0; "
            "no multi-step MPC controller is used"
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "runs/pandareach_small/data/pandareach_dls_100.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/pandareach_small/koopman"),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--k-step", type=int, default=20)
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--stability-weight", type=float, default=0.1)
    parser.add_argument(
        "--spectral-radius-limit",
        type=float,
        default=0.95,
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train(
        args.dataset,
        args.output_dir,
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            k_step=args.k_step,
            history=args.history,
            stability_weight=args.stability_weight,
            spectral_radius_limit=args.spectral_radius_limit,
            seed=args.seed,
        ),
        device_name=args.device,
    )
    concise = {
        "checkpoint_path": report["checkpoint_path"],
        "best_epoch": report["best_epoch"],
        "elapsed_seconds": report["elapsed_seconds"],
        "spectral_radius_A": report["spectral_radius_A"],
        "test_metrics": report["metrics"]["test"],
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
