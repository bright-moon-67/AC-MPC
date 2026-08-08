"""Episode-safe HDF5 windows for visual controlled Koopman training."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SplitFractions = Mapping[str, float] | Sequence[float]
NormalizerDict = dict[str, np.ndarray]


def list_episode_ids(trajectory_h5: str | Path) -> list[str]:
    """Return trajectory group names in numeric-aware deterministic order."""

    with h5py.File(trajectory_h5, "r") as handle:
        names = [
            name
            for name, value in handle.items()
            if isinstance(value, h5py.Group)
            and name.startswith("traj_")
            and name[5:].isdigit()
        ]

    def key(name: str) -> tuple[int, int | str]:
        if name.startswith("traj_") and name[5:].isdigit():
            return (0, int(name[5:]))
        return (1, name)

    return sorted(names, key=key)


def split_episode_ids(
    episode_names: Sequence[str],
    seed: int,
    fractions: SplitFractions = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Split unique episode names without ever mixing windows across splits."""

    names = [str(name) for name in episode_names]
    if not names:
        raise ValueError("episode_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("episode_names must be unique")
    if isinstance(fractions, Mapping):
        split_names = list(fractions)
        values = np.asarray([fractions[name] for name in split_names], dtype=np.float64)
    else:
        if len(fractions) != 3:
            raise ValueError("Sequence fractions must contain train/val/test")
        split_names = ["train", "val", "test"]
        values = np.asarray(fractions, dtype=np.float64)
    if len(values) == 0 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Split fractions must be finite and non-negative")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError("Split fractions must sum to one")

    exact = values * len(names)
    counts = np.floor(exact).astype(np.int64)
    remainder = len(names) - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remainder]] += 1

    positive = np.flatnonzero(values > 0)
    if len(names) >= len(positive):
        for index in positive:
            if counts[index] != 0:
                continue
            donors = np.flatnonzero(counts > 1)
            if len(donors) == 0:
                break
            donor = donors[np.argmax(counts[donors])]
            counts[donor] -= 1
            counts[index] += 1

    permutation = np.random.default_rng(seed).permutation(len(names))
    shuffled = [names[index] for index in permutation]
    result: dict[str, list[str]] = {}
    start = 0
    for split_name, count in zip(split_names, counts.tolist(), strict=True):
        result[split_name] = shuffled[start : start + count]
        start += count
    if start != len(names):
        raise RuntimeError("Internal split accounting error")
    return result


def _feature_dataset(
    handle: h5py.File,
    episode_id: str,
    feature_key: str,
) -> h5py.Dataset:
    if episode_id not in handle:
        raise KeyError(f"Missing feature episode {episode_id}")
    node = handle[episode_id]
    if isinstance(node, h5py.Dataset):
        return node
    if feature_key not in node or not isinstance(node[feature_key], h5py.Dataset):
        raise KeyError(f"Missing {episode_id}/{feature_key} feature dataset")
    return node[feature_key]


def _update_moments(
    values: np.ndarray,
    total: np.ndarray | None,
    total_square: np.ndarray | None,
    count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Normalizer inputs must be finite rank-2 arrays")
    batch_total = values.sum(axis=0)
    batch_square = np.square(values).sum(axis=0)
    if total is None:
        total, total_square = batch_total, batch_square
    else:
        if total.shape != batch_total.shape:
            raise ValueError("Feature dimensions differ between episodes")
        total += batch_total
        assert total_square is not None
        total_square += batch_square
    return total, total_square, count + len(values)


def _finish_moments(
    total: np.ndarray | None,
    total_square: np.ndarray | None,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if total is None or total_square is None or count == 0:
        raise ValueError("Cannot fit a normalizer on an empty split")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def fit_normalizers(
    trajectory_h5: str | Path,
    feature_h5: str | Path,
    train_ids: Sequence[str],
    *,
    feature_key: str = "resnet18",
) -> NormalizerDict:
    """Fit robot/feature statistics using training episodes only."""

    episode_ids = [str(episode_id) for episode_id in train_ids]
    if not episode_ids:
        raise ValueError("train_ids must not be empty")
    robot_total = robot_square = feature_total = feature_square = None
    robot_count = feature_count = 0
    with h5py.File(trajectory_h5, "r") as trajectories, h5py.File(
        feature_h5, "r"
    ) as features:
        for episode_id in episode_ids:
            if episode_id not in trajectories:
                raise KeyError(f"Missing trajectory episode {episode_id}")
            robot = np.asarray(trajectories[episode_id]["robot"])
            actions = trajectories[episode_id]["actions"]
            feature = np.asarray(
                _feature_dataset(features, episode_id, feature_key)
            )
            if len(robot) != len(actions) + 1:
                raise ValueError(f"{episode_id}/robot must contain T+1 samples")
            if len(robot) != len(feature):
                raise ValueError(f"Robot/feature T+1 mismatch in {episode_id}")
            robot_total, robot_square, robot_count = _update_moments(
                robot, robot_total, robot_square, robot_count
            )
            feature_total, feature_square, feature_count = _update_moments(
                feature, feature_total, feature_square, feature_count
            )
    robot_mean, robot_std = _finish_moments(
        robot_total, robot_square, robot_count
    )
    feature_mean, feature_std = _finish_moments(
        feature_total, feature_square, feature_count
    )
    return {
        "robot_mean": robot_mean,
        "robot_std": robot_std,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


class VisualWindowDataset(Dataset[dict[str, Any]]):
    """Multi-step windows with ``H+1`` states/features and ``H`` actions.

    HDF5 files are opened lazily and reopened after a process fork, avoiding
    sharing unsafe h5py handles between PyTorch DataLoader workers.
    """

    def __init__(
        self,
        trajectory_h5: str | Path,
        feature_h5: str | Path,
        episode_ids: Sequence[str],
        horizon: int,
        normalizers: Mapping[str, np.ndarray] | None = None,
        normalize: bool = True,
        *,
        feature_key: str = "resnet18",
    ) -> None:
        super().__init__()
        self.trajectory_h5 = str(Path(trajectory_h5).expanduser().resolve())
        self.feature_h5 = str(Path(feature_h5).expanduser().resolve())
        self.episode_ids = tuple(str(episode_id) for episode_id in episode_ids)
        self.horizon = int(horizon)
        self.feature_key = str(feature_key)
        self.normalize = bool(normalize)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not self.episode_ids or len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("episode_ids must be non-empty and unique")
        if self.normalize and normalizers is None:
            raise ValueError("normalizers are required when normalize=True")

        self.normalizers = (
            {key: np.asarray(value, dtype=np.float32) for key, value in normalizers.items()}
            if normalizers is not None
            else None
        )
        self._trajectory_handle: h5py.File | None = None
        self._feature_handle: h5py.File | None = None
        self._handle_pid: int | None = None
        self._windows: list[tuple[str, int]] = []
        self.robot_dim: int | None = None
        self.feature_dim: int | None = None
        self.action_dim: int | None = None
        self._build_index_and_validate()
        self._validate_normalizers()

    def _build_index_and_validate(self) -> None:
        with h5py.File(self.trajectory_h5, "r") as trajectories, h5py.File(
            self.feature_h5, "r"
        ) as features:
            for episode_id in self.episode_ids:
                if episode_id not in trajectories:
                    raise KeyError(f"Missing trajectory episode {episode_id}")
                group = trajectories[episode_id]
                if "robot" not in group or "actions" not in group:
                    raise KeyError(f"{episode_id} needs robot and actions datasets")
                robot = group["robot"]
                actions = group["actions"]
                feature = _feature_dataset(features, episode_id, self.feature_key)
                if robot.ndim != 2 or actions.ndim != 2 or feature.ndim != 2:
                    raise ValueError(f"Rank mismatch in {episode_id}")
                transition_count = len(actions)
                if len(robot) != transition_count + 1:
                    raise ValueError(f"{episode_id}/robot must contain T+1 samples")
                if len(feature) != transition_count + 1:
                    raise ValueError(f"{episode_id}/features must contain T+1 samples")
                dims = (robot.shape[1], feature.shape[1], actions.shape[1])
                if self.robot_dim is None:
                    self.robot_dim, self.feature_dim, self.action_dim = dims
                elif dims != (self.robot_dim, self.feature_dim, self.action_dim):
                    raise ValueError("Robot, feature, or action dimensions differ")
                self._windows.extend(
                    (episode_id, start)
                    for start in range(transition_count - self.horizon + 1)
                )
        if not self._windows:
            raise ValueError("No complete windows exist for the requested horizon")

    def _validate_normalizers(self) -> None:
        if self.normalizers is None:
            return
        expected = {
            "robot_mean": self.robot_dim,
            "robot_std": self.robot_dim,
            "feature_mean": self.feature_dim,
            "feature_std": self.feature_dim,
        }
        missing = set(expected).difference(self.normalizers)
        if missing:
            raise KeyError(f"Missing normalizer entries: {sorted(missing)}")
        for name, dimension in expected.items():
            value = self.normalizers[name]
            if value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} shape or values")
            if name.endswith("_std") and np.any(value <= 0):
                raise ValueError(f"{name} must be strictly positive")

    def _ensure_handles(self) -> tuple[h5py.File, h5py.File]:
        pid = os.getpid()
        if self._handle_pid != pid:
            self.close()
        if self._trajectory_handle is None:
            self._trajectory_handle = h5py.File(self.trajectory_h5, "r")
            self._feature_handle = h5py.File(self.feature_h5, "r")
            self._handle_pid = pid
        assert self._feature_handle is not None
        return self._trajectory_handle, self._feature_handle

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def window_metadata(self) -> tuple[tuple[str, int], ...]:
        """Return the deterministic ``(episode_id, start)`` window index.

        Evaluation code uses this immutable view to construct action-sequence
        counterfactuals without relying on DataLoader batch boundaries.
        """

        return tuple(self._windows)

    def action_windows(self, indices: Sequence[int]) -> torch.Tensor:
        """Load complete, unnormalised action windows for dataset indices.

        A short-lived HDF5 handle is intentional: it avoids opening the lazy
        per-worker handle before a multiprocessing DataLoader is iterated.
        """

        resolved = [int(index) for index in indices]
        if any(index < 0 or index >= len(self) for index in resolved):
            raise IndexError("action-window index is out of range")
        windows: list[np.ndarray] = []
        with h5py.File(self.trajectory_h5, "r") as trajectories:
            for index in resolved:
                episode_id, start = self._windows[index]
                stop = start + self.horizon
                windows.append(
                    np.asarray(
                        trajectories[episode_id]["actions"][start:stop],
                        dtype=np.float32,
                    )
                )
        if not windows:
            return torch.empty((0, self.horizon, int(self.action_dim)), dtype=torch.float32)
        return torch.from_numpy(np.ascontiguousarray(np.stack(windows)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_id, start = self._windows[index]
        trajectories, features = self._ensure_handles()
        stop_state = start + self.horizon + 1
        stop_action = start + self.horizon
        robot = np.asarray(
            trajectories[episode_id]["robot"][start:stop_state], dtype=np.float32
        )
        feature = np.asarray(
            _feature_dataset(features, episode_id, self.feature_key)[
                start:stop_state
            ],
            dtype=np.float32,
        )
        actions = np.asarray(
            trajectories[episode_id]["actions"][start:stop_action],
            dtype=np.float32,
        )
        if not (
            np.isfinite(robot).all()
            and np.isfinite(feature).all()
            and np.isfinite(actions).all()
        ):
            raise ValueError(f"Non-finite training data in {episode_id} at {start}")
        if self.normalize:
            assert self.normalizers is not None
            robot = (robot - self.normalizers["robot_mean"]) / self.normalizers[
                "robot_std"
            ]
            feature = (
                feature - self.normalizers["feature_mean"]
            ) / self.normalizers["feature_std"]
        state = np.concatenate((robot, feature), axis=-1)
        return {
            "robot": torch.from_numpy(np.ascontiguousarray(robot)),
            "features": torch.from_numpy(np.ascontiguousarray(feature)),
            "state": torch.from_numpy(np.ascontiguousarray(state)),
            "actions": torch.from_numpy(np.ascontiguousarray(actions)),
            "episode_id": episode_id,
            "start": start,
        }

    def close(self) -> None:
        if self._trajectory_handle is not None:
            self._trajectory_handle.close()
        if self._feature_handle is not None:
            self._feature_handle.close()
        self._trajectory_handle = None
        self._feature_handle = None
        self._handle_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_trajectory_handle"] = None
        state["_feature_handle"] = None
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()


__all__ = [
    "VisualWindowDataset",
    "fit_normalizers",
    "list_episode_ids",
    "split_episode_ids",
]
