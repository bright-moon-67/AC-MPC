from __future__ import annotations

import numpy as np
import pytest

from antmaze_ac.data.offline_episodes import (
    atomic_savez_compressed,
    merge_episode_files,
    validate_episode_arrays,
)


def _episode(episode_id: int, length: int, terminal: bool) -> dict[str, np.ndarray]:
    observations = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
    terminals = np.zeros(length, dtype=np.bool_)
    timeouts = np.zeros(length, dtype=np.bool_)
    terminals[-1] = terminal
    timeouts[-1] = not terminal
    return {
        "observations": observations,
        "actions": np.full((length, 2), episode_id, dtype=np.float32),
        "rewards": np.arange(length, dtype=np.float32),
        "next_observations": observations + 1,
        "terminals": terminals,
        "timeouts": timeouts,
        "episode_ids": np.full(length, episode_id, dtype=np.int64),
    }


def test_merge_episode_files_preserves_terminal_boundaries(tmp_path) -> None:
    paths = []
    for index, episode in enumerate((_episode(0, 2, True), _episode(1, 3, False))):
        path = tmp_path / f"episode_{index:06d}.npz"
        atomic_savez_compressed(path, episode)
        paths.append(path)

    total = merge_episode_files(paths, tmp_path / "dataset.npz")

    assert total == 5
    with np.load(tmp_path / "dataset.npz", allow_pickle=False) as dataset:
        assert dataset["observations"].shape == (5, 3)
        assert dataset["actions"].shape == (5, 2)
        assert dataset["terminals"].tolist() == [False, True, False, False, False]
        assert dataset["timeouts"].tolist() == [False, False, False, False, True]
        assert dataset["episode_ids"].tolist() == [0, 0, 1, 1, 1]


def test_validate_episode_rejects_internal_done() -> None:
    episode = _episode(0, 3, True)
    episode["terminals"][0] = True

    with pytest.raises(ValueError, match="Only the final transition"):
        validate_episode_arrays(episode)
