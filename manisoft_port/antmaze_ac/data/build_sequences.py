from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


REQUIRED_FIELDS = ("observations", "actions", "rewards", "terminals")


@dataclass
class AugmentedDataset:
    """Canonical offline transition dataset.

    The unambiguous transition convention is

    ``state_t=[observation_t, previous_action]``,
    ``action_t=delta_action_t=current_action-previous_action``, and
    ``next_state_t=[next_observation_t, current_action]``.

    ``x``, ``delta_action`` and ``next_x`` remain as read-only compatibility
    aliases for the Koopman code written before schema version 2.
    """

    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    terminal: np.ndarray
    timeout: np.ndarray
    episode_id: np.ndarray
    step_index: np.ndarray
    current_action: np.ndarray

    def __len__(self) -> int:
        return int(self.state.shape[0])

    @property
    def x(self) -> np.ndarray:
        return self.state

    @property
    def delta_action(self) -> np.ndarray:
        return self.action

    @property
    def next_x(self) -> np.ndarray:
        return self.next_state

    @property
    def previous_action(self) -> np.ndarray:
        return self.state[:, -self.action.shape[1] :]

    def take(self, indices: np.ndarray) -> "AugmentedDataset":
        return AugmentedDataset(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return only canonical schema-v2 fields for serialization."""

        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, np.ndarray]) -> "AugmentedDataset":
        """Load canonical schema-v2 or the legacy x/delta_action schema."""

        if {"state", "action", "next_state", "current_action"} <= set(values):
            terminal = np.asarray(values["terminal"], dtype=bool)
            timeout = np.asarray(values["timeout"], dtype=bool)
            return cls(
                state=np.asarray(values["state"]),
                action=np.asarray(values["action"]),
                next_state=np.asarray(values["next_state"]),
                reward=np.asarray(values["reward"]),
                done=np.asarray(
                    values.get("done", terminal | timeout),
                    dtype=bool,
                ),
                terminal=terminal,
                timeout=timeout,
                episode_id=np.asarray(values["episode_id"]),
                step_index=np.asarray(values["step_index"]),
                current_action=np.asarray(values["current_action"]),
            )
        legacy = {"x", "delta_action", "next_x", "action"}
        if legacy <= set(values):
            terminal = np.asarray(values["terminal"], dtype=bool)
            timeout = np.asarray(values["timeout"], dtype=bool)
            return cls(
                state=np.asarray(values["x"]),
                action=np.asarray(values["delta_action"]),
                next_state=np.asarray(values["next_x"]),
                reward=np.asarray(values["reward"]),
                done=terminal | timeout,
                terminal=terminal,
                timeout=timeout,
                episode_id=np.asarray(values["episode_id"]),
                step_index=np.asarray(values["step_index"]),
                current_action=np.asarray(values["action"]),
            )
        raise KeyError(
            "Dataset must use schema-v2 state/action/next_state/current_action "
            "or legacy x/delta_action/next_x/action fields"
        )

    def validate(self, atol: float = 1e-6) -> None:
        if not len(self):
            raise ValueError("Dataset is empty")
        action_dim = self.current_action.shape[1]
        expected_rows = len(self)
        arrays = {
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "done": self.done,
            "terminal": self.terminal,
            "timeout": self.timeout,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "current_action": self.current_action,
        }
        for name, values in arrays.items():
            if len(values) != expected_rows:
                raise ValueError(
                    f"{name} has {len(values)} rows; expected {expected_rows}"
                )
        if self.state.ndim != 2 or self.next_state.shape != self.state.shape:
            raise ValueError("state and next_state must be matching rank-2 arrays")
        if self.action.ndim != 2 or self.action.shape[1] != action_dim:
            raise ValueError("action must be rank-2 and match current_action width")
        if not np.array_equal(
            np.asarray(self.done, dtype=bool),
            np.asarray(self.terminal, dtype=bool)
            | np.asarray(self.timeout, dtype=bool),
        ):
            raise ValueError("done must equal terminal OR timeout")
        previous_action = self.previous_action
        error = np.max(
            np.abs(
                self.current_action
                - (previous_action + self.action)
            )
        )
        if error >= atol:
            raise ValueError(f"Action reconstruction error {error:.3e} is not < {atol:.1e}")
        if not np.allclose(
            self.next_state[:, -action_dim:],
            self.current_action,
            atol=atol,
        ):
            raise ValueError(
                "next_state action block does not equal current_action u_t"
            )
        starts = self.step_index == 0
        if not np.allclose(previous_action[starts], 0.0, atol=atol):
            raise ValueError("Episode starts do not use u_-1=0")


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, epsilon: float = 1e-6) -> "Normalizer":
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values.std(axis=0, dtype=np.float64).astype(np.float32)
        return cls(mean=mean, std=np.maximum(std, epsilon))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def _episode_layout(terminals: np.ndarray, timeouts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(terminals)
    episode_ids = np.empty(n, dtype=np.int64)
    step_indices = np.empty(n, dtype=np.int64)
    episode = 0
    step = 0
    for index in range(n):
        episode_ids[index] = episode
        step_indices[index] = step
        if bool(terminals[index]) or bool(timeouts[index]):
            episode += 1
            step = 0
        else:
            step += 1
    return episode_ids, step_indices


def build_augmented_dataset(raw: Mapping[str, np.ndarray]) -> AugmentedDataset:
    missing = [key for key in REQUIRED_FIELDS if key not in raw]
    if missing:
        raise KeyError(f"Missing D4RL fields: {missing}")
    observations = np.asarray(raw["observations"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.float32)
    rewards = np.asarray(raw["rewards"], dtype=np.float32).reshape(-1)
    terminals = np.asarray(raw["terminals"], dtype=bool).reshape(-1)
    timeouts = np.asarray(raw.get("timeouts", np.zeros(len(terminals), dtype=bool)), dtype=bool).reshape(-1)
    next_observations = raw.get("next_observations")
    if next_observations is None:
        raise KeyError("D4RL dataset must contain next_observations to avoid crossing episode boundaries")
    next_observations = np.asarray(next_observations, dtype=np.float32)

    n = len(observations)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must be rank-2 arrays")
    for name, values in {
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "next_observations": next_observations,
    }.items():
        if len(values) != n:
            raise ValueError(f"{name} has {len(values)} rows; expected {n}")
    if next_observations.shape != observations.shape:
        raise ValueError("next_observations shape must equal observations shape")

    episode_ids, step_indices = _episode_layout(terminals, timeouts)
    previous_actions = np.zeros_like(actions)
    continuation = step_indices > 0
    previous_actions[continuation] = actions[np.flatnonzero(continuation) - 1]
    delta_actions = actions - previous_actions
    dataset = AugmentedDataset(
        state=np.concatenate([observations, previous_actions], axis=1),
        action=delta_actions,
        next_state=np.concatenate([next_observations, actions], axis=1),
        reward=rewards,
        done=terminals | timeouts,
        terminal=terminals,
        timeout=timeouts,
        episode_id=episode_ids,
        step_index=step_indices,
        current_action=actions,
    )
    dataset.validate()
    return dataset


def split_by_episode(
    dataset: AugmentedDataset,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, AugmentedDataset]:
    fractions_array = np.asarray(fractions, dtype=np.float64)
    if len(fractions_array) != 3 or np.any(fractions_array <= 0):
        raise ValueError("fractions must contain three positive values")
    fractions_array /= fractions_array.sum()
    episodes = np.unique(dataset.episode_id)
    if len(episodes) < 3:
        raise ValueError("At least three episodes are required for train/validation/test")
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_validation = max(1, int(round(len(episodes) * fractions_array[1])))
    n_test = max(1, int(round(len(episodes) * fractions_array[2])))
    n_train = len(episodes) - n_validation - n_test
    if n_train < 1:
        raise ValueError("Split leaves no training episodes")
    episode_splits = {
        "train": episodes[:n_train],
        "validation": episodes[n_train : n_train + n_validation],
        "test": episodes[n_train + n_validation :],
    }
    return {
        name: dataset.take(np.flatnonzero(np.isin(dataset.episode_id, ids)))
        for name, ids in episode_splits.items()
    }


def valid_window_starts(dataset: AugmentedDataset, transitions: int) -> np.ndarray:
    """Return starts for windows with exactly ``transitions`` transitions."""

    if transitions < 1:
        raise ValueError("transitions must be positive")
    starts = []
    for start in range(len(dataset) - transitions + 1):
        stop = start + transitions
        if (
            np.all(dataset.episode_id[start:stop] == dataset.episode_id[start])
            and not np.any(dataset.terminal[start:stop] | dataset.timeout[start:stop])
            and np.array_equal(
                dataset.step_index[start:stop],
                np.arange(dataset.step_index[start], dataset.step_index[start] + transitions),
            )
        ):
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def load_d4rl_hdf5(path: str) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    """Load legacy D4RL HDF5 and construct its q-learning next observations.

    Legacy files contain observations at sample times but no
    ``next_observations``. As in D4RL's official ``qlearning_dataset``, the next
    row is used. Boundary rows remain labelled terminal/timeout and are excluded
    by ``valid_window_starts`` from Koopman windows.
    """

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install the 'd4rl-data' extra to read HDF5 files") from exc
    with h5py.File(path, "r") as handle:
        shapes = {
            name: tuple(handle[name].shape)
            for name in ("observations", "actions", "rewards", "terminals", "timeouts")
        }
        length = shapes["observations"][0]
        raw = {
            "observations": np.asarray(handle["observations"][:-1], dtype=np.float32),
            "next_observations": np.asarray(handle["observations"][1:], dtype=np.float32),
            "actions": np.asarray(handle["actions"][:-1], dtype=np.float32),
            "rewards": np.asarray(handle["rewards"][:-1], dtype=np.float32),
            "terminals": np.asarray(handle["terminals"][:-1], dtype=bool),
            "timeouts": np.asarray(handle["timeouts"][:-1], dtype=bool),
        }
    if any(shape[0] != length for shape in shapes.values()):
        raise ValueError(f"HDF5 fields have mismatched lengths: {shapes}")
    return raw, shapes
