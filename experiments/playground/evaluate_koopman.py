"""Evaluate Playground Koopman exports on a common episode-safe test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.playground.collect_koopman import BEHAVIORS, STAGES
from experiments.playground.tasks import TASKS
from experiments.playground.train_ppo import _atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_model(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Models must use NAME=PATH syntax")
    return name, Path(path)


def _load_model(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        encoder_count = int(metadata["encoder_layer_count"])
        return {
            "A": np.asarray(archive["A"], dtype=np.float64),
            "B": np.asarray(archive["B"], dtype=np.float64),
            "center": np.asarray(archive["center"], dtype=np.float64),
            "scale": np.asarray(archive["scale"], dtype=np.float64),
            "weights": tuple(
                np.asarray(archive[f"encoder_{index}_weight"], dtype=np.float64)
                for index in range(encoder_count)
            ),
            "biases": tuple(
                np.asarray(archive[f"encoder_{index}_bias"], dtype=np.float64)
                for index in range(encoder_count)
            ),
            "metadata": metadata,
        }


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def _lift(model: dict[str, Any], normalized_state: np.ndarray) -> np.ndarray:
    encoded = normalized_state
    for index, (weight, bias) in enumerate(
        zip(model["weights"], model["biases"], strict=True)
    ):
        encoded = encoded @ weight.T + bias
        if index + 1 < len(model["weights"]):
            encoded = _silu(encoded)
    return np.concatenate((normalized_state, encoded), axis=-1)


def _reference_scale(data_dir: Path, state_dim: int) -> np.ndarray:
    count = 0
    total = np.zeros(state_dim, dtype=np.float64)
    total_square = np.zeros(state_dim, dtype=np.float64)
    for stage in STAGES:
        with np.load(data_dir / f"{stage}.npz", allow_pickle=False) as archive:
            states = np.asarray(archive["states"], dtype=np.float64)
        training = states[np.arange(states.shape[0]) % 10 < 8, :-1]
        count += int(np.prod(training.shape[:-1]))
        total += np.sum(training, axis=(0, 1))
        total_square += np.sum(training * training, axis=(0, 1))
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 1e-12)
    return np.sqrt(variance)


def _empty_metrics(state_dim: int) -> dict[str, Any]:
    return {
        "windows": 0,
        "weighted_sse": 0.0,
        "weighted_hold_sse": 0.0,
        "one_step_sse": 0.0,
        "final_step_sse": 0.0,
        "physical_sse": np.zeros(state_dim, dtype=np.float64),
        "reward_errors": 0,
        "reward_predictions": 0,
    }


def _update_metrics(
    metrics: dict[str, Any],
    error_square: np.ndarray,
    hold_error_square: np.ndarray,
    physical_error_square: np.ndarray,
    reward_mismatch: np.ndarray,
    weights: np.ndarray,
) -> None:
    metrics["windows"] += error_square.shape[0]
    metrics["weighted_sse"] += float(
        np.sum(error_square * weights[None, :, None])
    )
    metrics["weighted_hold_sse"] += float(
        np.sum(hold_error_square * weights[None, :, None])
    )
    metrics["one_step_sse"] += float(np.sum(error_square[:, 0]))
    metrics["final_step_sse"] += float(np.sum(error_square[:, -1]))
    metrics["physical_sse"] += np.sum(physical_error_square, axis=(0, 1))
    metrics["reward_errors"] += int(np.sum(reward_mismatch))
    metrics["reward_predictions"] += int(reward_mismatch.size)


def _finalize_metrics(
    metrics: dict[str, Any], state_dim: int, horizon: int, weights: np.ndarray
) -> dict[str, Any]:
    windows = metrics["windows"]
    weighted_denominator = windows * state_dim * float(np.sum(weights))
    nmse = metrics["weighted_sse"] / weighted_denominator
    hold_nmse = metrics["weighted_hold_sse"] / weighted_denominator
    return {
        "windows": windows,
        "weighted_rollout_nmse": nmse,
        "weighted_rollout_nrmse": math.sqrt(nmse),
        "hold_nmse": hold_nmse,
        "model_to_hold_mse_ratio": nmse / hold_nmse,
        "one_step_nmse": metrics["one_step_sse"] / (windows * state_dim),
        f"step_{horizon}_nmse": metrics["final_step_sse"]
        / (windows * state_dim),
        "physical_rmse_over_rollout": np.sqrt(
            metrics["physical_sse"] / (windows * horizon)
        ).tolist(),
        "exact_reward_mismatch_fraction": metrics["reward_errors"]
        / metrics["reward_predictions"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.task != "ReacherHard":
        raise ValueError("Exact-reward diagnostics currently support ReacherHard only")
    task = TASKS[args.task]
    if args.horizon < 1 or args.horizon > task.episode_steps:
        raise ValueError("Evaluation horizon is outside the episode")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    named_paths = dict(args.model)
    if len(named_paths) != len(args.model):
        raise ValueError("Model names must be unique")
    models = {
        name: _load_model(path.resolve()) for name, path in named_paths.items()
    }
    for name, model in models.items():
        architecture = model["metadata"]["architecture"]
        if int(architecture["state_dim"]) != task.observation_dim:
            raise ValueError(f"{name} state dimension does not match {args.task}")
        if int(architecture["action_dim"]) != task.action_dim:
            raise ValueError(f"{name} action dimension does not match {args.task}")

    common_scale = _reference_scale(args.data_dir, task.observation_dim)
    weights = 0.99 ** np.arange(args.horizon, dtype=np.float64)
    groups = ("all", "first_50", "unreached_start", *BEHAVIORS)
    accumulated = {
        model_name: {group: _empty_metrics(task.observation_dim) for group in groups}
        for model_name in models
    }
    behavior_counts = {name: 0 for name in BEHAVIORS}
    max_start = task.episode_steps - args.horizon
    for stage in STAGES:
        with np.load(args.data_dir / f"{stage}.npz", allow_pickle=False) as archive:
            states = np.asarray(archive["states"], dtype=np.float64)
            actions = np.asarray(archive["actions"], dtype=np.float64)
            rewards = np.asarray(archive["rewards"], dtype=np.float64)
            behavior = (
                np.asarray(archive["behavior_mode"], dtype=np.int32)
                if "behavior_mode" in archive.files
                else np.zeros(states.shape[0], dtype=np.int32)
            )
        test_episode = np.flatnonzero(np.arange(states.shape[0]) % 10 == 9)
        starts = np.arange(max_start + 1, dtype=np.int32)
        pairs = np.stack(
            np.meshgrid(test_episode, starts, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        for mode, name in enumerate(BEHAVIORS):
            behavior_counts[name] += int(np.sum(behavior[test_episode] == mode))
        for lower in range(0, pairs.shape[0], args.batch_size):
            pair = pairs[lower : lower + args.batch_size]
            episode_index, start_index = pair[:, 0], pair[:, 1]
            initial = states[episode_index, start_index]
            true = np.stack(
                [
                    states[episode_index, start_index + offset + 1]
                    for offset in range(args.horizon)
                ],
                axis=1,
            )
            true_reward = np.stack(
                [
                    rewards[episode_index, start_index + offset]
                    for offset in range(args.horizon)
                ],
                axis=1,
            )
            mode = behavior[episode_index]
            masks = {
                "all": np.ones(pair.shape[0], dtype=bool),
                "first_50": start_index < 50,
                "unreached_start": rewards[episode_index, start_index] < 0.5,
                **{
                    name: mode == index for index, name in enumerate(BEHAVIORS)
                },
            }
            for model_name, model in models.items():
                normalized = (initial - model["center"]) / model["scale"]
                lifted = _lift(model, normalized)
                predicted = np.empty_like(true)
                for offset in range(args.horizon):
                    action = actions[episode_index, start_index + offset]
                    lifted = lifted @ model["A"].T + action @ model["B"].T
                    predicted[:, offset] = (
                        lifted[:, : task.observation_dim] * model["scale"]
                        + model["center"]
                    )
                physical_error_square = (predicted - true) ** 2
                error_square = physical_error_square / (
                    common_scale[None, None, :] ** 2
                )
                hold_error_square = (
                    (initial[:, None, :] - true)
                    / common_scale[None, None, :]
                ) ** 2
                predicted_reward = (
                    np.linalg.norm(predicted[..., 2:4], axis=-1) <= 0.025
                )
                reward_mismatch = predicted_reward != (true_reward >= 0.5)
                for group, mask in masks.items():
                    if np.any(mask):
                        _update_metrics(
                            accumulated[model_name][group],
                            error_square[mask],
                            hold_error_square[mask],
                            physical_error_square[mask],
                            reward_mismatch[mask],
                            weights,
                        )

    result = {
        "kind": "mujoco_playground_koopman_test_evaluation_v1",
        "task": args.task,
        "data_dir": str(args.data_dir.resolve()),
        "data_manifest_sha256": _sha256(args.data_dir / "manifest.json"),
        "split": "episode_index_mod_10_equals_9",
        "horizon_steps": args.horizon,
        "horizon_seconds": args.horizon * task.control_timestep,
        "common_reference_scale": common_scale.tolist(),
        "test_behavior_episode_counts": behavior_counts,
        "models": {},
        "finished_unix_seconds": time.time(),
    }
    for name, model in models.items():
        result["models"][name] = {
            "path": str(named_paths[name].resolve()),
            "sha256": _sha256(named_paths[name]),
            "architecture": model["metadata"]["architecture"],
            "groups": {
                group: _finalize_metrics(
                    metrics, task.observation_dim, args.horizon, weights
                )
                for group, metrics in accumulated[name].items()
                if metrics["windows"]
            },
        }
    _atomic_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("ReacherHard",), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=_parse_model, action="append", required=True)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
