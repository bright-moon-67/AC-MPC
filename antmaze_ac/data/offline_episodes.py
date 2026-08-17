from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path

import numpy as np


OFFLINE_RL_KEYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
    "timeouts",
)


def validate_episode_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    """Validate one transition-aligned offline-RL episode.

    All arrays, including optional diagnostics, must use the transition axis as
    their first axis.  A collected episode must finish with exactly one of a
    terminal or timeout flag on its final transition.
    """

    missing = [key for key in OFFLINE_RL_KEYS if key not in arrays]
    if missing:
        raise KeyError(f"Episode is missing required arrays: {missing}")

    converted = {key: np.asarray(value) for key, value in arrays.items()}
    lengths = {
        key: int(value.shape[0])
        for key, value in converted.items()
        if value.ndim >= 1
    }
    if len(lengths) != len(converted):
        scalar_keys = [
            key for key, value in converted.items() if value.ndim == 0
        ]
        raise ValueError(f"Episode arrays must not be scalars: {scalar_keys}")
    transition_count = lengths["observations"]
    if transition_count < 1:
        raise ValueError("Episode must contain at least one transition")
    mismatched = {
        key: length
        for key, length in lengths.items()
        if length != transition_count
    }
    if mismatched:
        raise ValueError(
            f"Episode arrays are not transition-aligned: {mismatched}"
        )

    observations = converted["observations"]
    next_observations = converted["next_observations"]
    actions = converted["actions"]
    if observations.ndim != 2 or next_observations.shape != observations.shape:
        raise ValueError(
            "observations and next_observations must have the same [T,D] shape"
        )
    if actions.ndim != 2:
        raise ValueError("actions must have shape [T,A]")
    if converted["rewards"].shape != (transition_count,):
        raise ValueError("rewards must have shape [T]")

    terminals = converted["terminals"].astype(np.bool_, copy=False)
    timeouts = converted["timeouts"].astype(np.bool_, copy=False)
    if terminals.shape != (transition_count,) or timeouts.shape != (
        transition_count,
    ):
        raise ValueError("terminals and timeouts must have shape [T]")
    if np.any(terminals[:-1]) or np.any(timeouts[:-1]):
        raise ValueError("Only the final transition may end an episode")
    if bool(terminals[-1]) == bool(timeouts[-1]):
        raise ValueError(
            "The final transition must be exactly one of terminal or timeout"
        )
    if not all(
        np.isfinite(converted[key]).all()
        for key in ("observations", "actions", "rewards", "next_observations")
    ):
        raise ValueError("Core offline-RL arrays contain NaN or Inf")
    return transition_count


def atomic_savez_compressed(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Write a compressed NPZ without exposing a partially written file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def merge_episode_files(
    episode_paths: Sequence[str | Path],
    output_path: str | Path,
) -> int:
    """Validate and flatten episode shards into one D4RL-style NPZ."""

    paths = [Path(path) for path in episode_paths]
    if not paths:
        raise ValueError("At least one episode shard is required")

    chunks: dict[str, list[np.ndarray]] = {}
    expected_keys: tuple[str, ...] | None = None
    total_transitions = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            keys = tuple(sorted(archive.files))
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(f"Episode shard fields differ: {path}")
            episode = {key: np.asarray(archive[key]) for key in keys}
        total_transitions += validate_episode_arrays(episode)
        for key, value in episode.items():
            chunks.setdefault(key, []).append(value)

    merged = {
        key: np.concatenate(values, axis=0)
        for key, values in chunks.items()
    }
    atomic_savez_compressed(output_path, merged)
    return total_transitions
