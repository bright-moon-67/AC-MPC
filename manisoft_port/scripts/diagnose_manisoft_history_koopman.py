#!/usr/bin/env python
"""Measure History Koopman prediction errors on expert and policy rollouts.

The diagnostic rebuilds history contexts from physical states and applied
actions using the checkpoint's own normalizer.  It therefore remains valid
when an old expert/BC trajectory is evaluated with a newly trained Koopman
checkpoint whose normalization statistics have changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.data.action_rate_sampling import rate_bin_indices
from antmaze_ac.koopman.checkpoint import load_checkpoint
from antmaze_ac.koopman.history_model import HistoryDeepKoopman


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sequences(
    path: str | Path,
    state_dim: int,
    action_dim: int,
) -> list[dict[str, np.ndarray]]:
    """Load one expert dataset or a directory of evaluation trajectories."""

    path = Path(path).expanduser().resolve()
    files = (
        sorted(
            [candidate for candidate in [path / "trajectory.npz"] if candidate.is_file()]
            + list(path.glob("trajectory_*.npz"))
        )
        if path.is_dir()
        else [path]
    )
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"No trajectory data found at {path}")
    sequences: list[dict[str, np.ndarray]] = []
    for file in files:
        with np.load(file, allow_pickle=False) as archive:
            if "observation" not in archive.files:
                raise KeyError(f"{file} does not contain observation")
            observations = np.asarray(archive["observation"], dtype=np.float32)
            action_key = (
                "applied_action"
                if "applied_action" in archive.files
                else "expert_action"
            )
            if action_key not in archive.files:
                raise KeyError(f"{file} has no applied/expert action")
            actions = np.asarray(archive[action_key], dtype=np.float32)
            episode_ids = (
                np.asarray(archive["episode_id"], dtype=np.int64)
                if "episode_id" in archive.files
                else np.zeros(len(observations), dtype=np.int64)
            )
        if observations.ndim != 2 or observations.shape[1] < state_dim:
            raise ValueError(f"{file}: observation does not contain {state_dim}-D state")
        if (
            actions.ndim != 2
            or actions.shape[1] != action_dim
            or len(actions) != len(observations)
        ):
            raise ValueError(f"{file}: action/observation rows do not match")
        if not np.isfinite(observations).all() or not np.isfinite(actions).all():
            raise ValueError(f"{file} contains NaN or Inf")
        for episode_id in np.unique(episode_ids):
            mask = episode_ids == episode_id
            sequences.append(
                {
                    "states": observations[mask, :state_dim],
                    "actions": actions[mask],
                }
            )
    return sequences


def build_contexts(
    states: np.ndarray,
    actions: np.ndarray,
    starts: np.ndarray,
    history_steps: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> np.ndarray:
    """Rebuild ``[normalized state history, past action history]``."""

    contexts = np.empty(
        (len(starts), history_steps * (states.shape[1] + actions.shape[1])),
        dtype=np.float32,
    )
    for row, start in enumerate(starts):
        state_indices = np.arange(start - history_steps + 1, start + 1)
        state_indices = np.maximum(state_indices, 0)
        state_history = (states[state_indices] - state_mean) / state_std
        action_indices = np.arange(start - history_steps, start)
        action_history = np.zeros(
            (history_steps, actions.shape[1]), dtype=np.float32
        )
        valid = action_indices >= 0
        action_history[valid] = actions[action_indices[valid]]
        contexts[row] = np.concatenate(
            (state_history.reshape(-1), action_history.reshape(-1))
        )
    return contexts


def summarize(values: np.ndarray, scale: float = 1.0) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64) * scale
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


@torch.no_grad()
def diagnose_sequences(
    model: HistoryDeepKoopman,
    sequences: list[dict[str, np.ndarray]],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    horizons: list[int],
    tip_indices: np.ndarray,
    rate_edges: list[float],
    ood_edges: list[float],
    device: torch.device,
    batch_size: int,
) -> dict:
    max_horizon = max(horizons)
    records: dict[int, dict[str, list[np.ndarray]]] = {
        horizon: {
            "state_rmse": [],
            "tip_error": [],
            "persistence_tip_error": [],
            "rate": [],
            "state_zmax": [],
        }
        for horizon in horizons
    }
    sequence_count = 0
    window_count = 0
    for sequence in sequences:
        states = sequence["states"]
        actions = sequence["actions"]
        if len(states) <= max_horizon:
            continue
        sequence_count += 1
        starts_all = np.arange(len(states) - max_horizon, dtype=np.int64)
        window_count += len(starts_all)
        for batch_start in range(0, len(starts_all), batch_size):
            starts = starts_all[batch_start:batch_start + batch_size]
            contexts = build_contexts(
                states,
                actions,
                starts,
                model.history_steps,
                state_mean,
                state_std,
            )
            normalized_states = (states[starts] - state_mean) / state_std
            future_actions = np.stack(
                [actions[start:start + max_horizon] for start in starts]
            )
            predicted_normalized, _ = model.rollout(
                torch.as_tensor(normalized_states, device=device),
                torch.as_tensor(future_actions, device=device),
                torch.as_tensor(contexts, device=device),
            )
            predicted = (
                predicted_normalized.cpu().numpy() * state_std + state_mean
            )
            history_actions = np.zeros(
                (len(starts), model.history_steps, actions.shape[1]),
                dtype=np.float32,
            )
            for row, start in enumerate(starts):
                history_start = max(0, start - model.history_steps)
                available = actions[history_start:start]
                if len(available):
                    history_actions[row, -len(available):] = available
            action_path = np.concatenate(
                (history_actions, future_actions), axis=1
            )
            all_rates = np.max(
                np.abs(np.diff(action_path, axis=1)), axis=2
            )
            history_peak_rate = (
                np.max(all_rates[:, : model.history_steps - 1], axis=1)
                if model.history_steps > 1
                else np.zeros(len(starts), dtype=np.float32)
            )
            prefix_rates = np.maximum(
                history_peak_rate[:, None],
                np.maximum.accumulate(
                    all_rates[:, model.history_steps - 1 :], axis=1
                ),
            )
            state_zmax = np.max(np.abs(normalized_states), axis=1)
            for horizon in horizons:
                truth = states[starts + horizon]
                prediction = predicted[:, horizon - 1]
                records[horizon]["state_rmse"].append(
                    np.sqrt(np.mean((prediction - truth) ** 2, axis=1))
                )
                records[horizon]["tip_error"].append(
                    np.linalg.norm(
                        prediction[:, tip_indices] - truth[:, tip_indices],
                        axis=1,
                    )
                )
                records[horizon]["persistence_tip_error"].append(
                    np.linalg.norm(
                        states[starts][:, tip_indices] - truth[:, tip_indices],
                        axis=1,
                    )
                )
                records[horizon]["rate"].append(prefix_rates[:, horizon - 1])
                records[horizon]["state_zmax"].append(state_zmax)

    if not sequence_count:
        raise ValueError(f"No sequence is longer than max horizon {max_horizon}")
    rate_labels = [
        f"le_{rate_edges[0]:g}",
        *[
            f"{left:g}_to_{right:g}"
            for left, right in zip(rate_edges[:-1], rate_edges[1:])
        ],
        f"gt_{rate_edges[-1]:g}",
    ]
    ood_labels = [
        f"le_{ood_edges[0]:g}",
        *[
            f"{left:g}_to_{right:g}"
            for left, right in zip(ood_edges[:-1], ood_edges[1:])
        ],
        f"gt_{ood_edges[-1]:g}",
    ]
    result = {"sequences": sequence_count, "max_horizon_windows": window_count, "horizons": {}}
    for horizon in horizons:
        values = {
            key: np.concatenate(chunks)
            for key, chunks in records[horizon].items()
        }
        rate_bins = rate_bin_indices(values["rate"], rate_edges)
        ood_bins = rate_bin_indices(values["state_zmax"], ood_edges)
        horizon_result = {
            "state_rmse": summarize(values["state_rmse"]),
            "tip_error_mm": summarize(values["tip_error"], 1000.0),
            "persistence_tip_error_mm": summarize(
                values["persistence_tip_error"], 1000.0
            ),
            "action_rate": summarize(values["rate"]),
            "state_zmax": summarize(values["state_zmax"]),
            "by_action_rate": {},
            "by_state_zmax": {},
        }
        for bin_index, label in enumerate(rate_labels):
            mask = rate_bins == bin_index
            if np.any(mask):
                horizon_result["by_action_rate"][label] = {
                    "tip_error_mm": summarize(values["tip_error"][mask], 1000.0),
                    "state_rmse": summarize(values["state_rmse"][mask]),
                }
        for bin_index, label in enumerate(ood_labels):
            mask = ood_bins == bin_index
            if np.any(mask):
                horizon_result["by_state_zmax"][label] = {
                    "tip_error_mm": summarize(values["tip_error"][mask], 1000.0),
                    "state_rmse": summarize(values["state_rmse"][mask]),
                }
        result["horizons"][str(horizon)] = horizon_result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-dataset", action="append", default=[])
    parser.add_argument("--rollout-root", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--tip-indices", type=int, nargs=3, default=[30, 31, 32])
    parser.add_argument("--action-rate-bins", type=float, nargs="+", default=[0.005, 0.05])
    parser.add_argument("--state-z-bins", type=float, nargs="+", default=[5.0, 20.0])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.expert_dataset and not args.rollout_root:
        parser.error("provide at least one --expert-dataset or --rollout-root")
    if any(value < 1 for value in args.horizons) or args.batch_size < 1:
        parser.error("horizons and batch-size must be positive")
    for name, edges in (
        ("action-rate-bins", args.action_rate_bins),
        ("state-z-bins", args.state_z_bins),
    ):
        if any(value <= 0 for value in edges) or any(
            right <= left for left, right in zip(edges[:-1], edges[1:])
        ):
            parser.error(f"--{name} must be positive and strictly increasing")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, payload = load_checkpoint(args.checkpoint, map_location=device)
    if not isinstance(model, HistoryDeepKoopman):
        raise TypeError("checkpoint is not a HistoryDeepKoopman model")
    model = model.to(device).eval()
    tip_indices = np.asarray(args.tip_indices, dtype=np.int64)
    if np.any(tip_indices < 0) or np.any(tip_indices >= model.state_dim):
        raise ValueError("tip indices lie outside the Koopman physical state")
    state_normalizer = payload["normalizers"]["state"]
    state_mean = np.asarray(state_normalizer["mean"], dtype=np.float32)
    state_std = np.maximum(
        np.asarray(state_normalizer["std"], dtype=np.float32), 1e-6
    )
    groups = {
        "expert": args.expert_dataset,
        "rollout": args.rollout_root,
    }
    result = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "history_steps": model.history_steps,
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "horizons": sorted(set(args.horizons)),
        "groups": {},
    }
    for group_name, sources in groups.items():
        for source_index, source in enumerate(sources):
            sequences = load_sequences(source, model.state_dim, model.action_dim)
            group_key = f"{group_name}:{source_index}:{Path(source).name}"
            result["groups"][group_key] = diagnose_sequences(
                model,
                sequences,
                state_mean,
                state_std,
                sorted(set(args.horizons)),
                tip_indices,
                args.action_rate_bins,
                args.state_z_bins,
                device,
                args.batch_size,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
