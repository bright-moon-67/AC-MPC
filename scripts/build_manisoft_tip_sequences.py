#!/usr/bin/env python
"""Convert per-episode ManiSoft NPZ files to an 11-D tip-only Koopman dataset.

The full 41-D physical soft-robot state is compressed to the 11-D end-tip
state used for tracking:

    11-D = [tip_position(3), tip_speed(1), tip_velocity_direction(3),
            tip_quaternion_wxyz(4)]

The action dimension stays at 18, so the augmented (Koopman) state is
    29-D = [11-D tip state, 18-D previous_action]
and the Koopman action is the 18-D delta_action.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from antmaze_ac.config import load_config
from antmaze_ac.data.build_sequences import (
    AugmentedDataset,
    Normalizer,
    split_by_episode,
    valid_window_starts,
)

SOURCE_FIELDS = ("state", "action", "next_state")

SOURCE_OBSERVATION_DIM = 41
TIP_STATE_DIM = 11
ACTION_DIM = 18
AUGMENTED_STATE_DIM = TIP_STATE_DIM + ACTION_DIM

# Layout of the 41-D physical state produced by
# ManiSoft/scripts/collect_koopman_data.py:_build_compact_observation:
#   [0:33]  compact positions (nodes 0,2,...,20, coordinate-major: x,y,z)
#   [33:34] tip_speed
#   [34:37] tip_velocity_direction (unit vector, zero at rest)
#   [37:41] tip_quaternion_wxyz
# The tip is node 20 = the 11th sampled node, so within the coordinate-major
# layout its (x, y, z) live at indices (10, 21, 32).
TIP_POSITION_INDICES = (10, 21, 32)


@dataclass(frozen=True)
class SourceGroup:
    root: Path
    paths: tuple[Path, ...]
    ignored_paths: tuple[Path, ...]


def discover_source_groups(
    roots: Sequence[str | Path],
    episode_counts: Sequence[int] | None = None,
) -> list[SourceGroup]:
    """Resolve deterministic episode lists, optionally restricting each root to 0..N-1."""

    if not roots:
        raise ValueError("At least one --input-root is required")
    if episode_counts is not None and len(episode_counts) != len(roots):
        raise ValueError("--episode-counts must contain one value per --input-root")

    groups: list[SourceGroup] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Input root is not a directory: {root}")
        discovered = set(root.glob("episode_*.npz"))
        if episode_counts is None:
            paths = tuple(sorted(discovered, key=_episode_number))
            ignored_paths: tuple[Path, ...] = ()
        else:
            count = int(episode_counts[len(groups)])
            if count < 1:
                raise ValueError("Every --episode-counts value must be positive")
            paths = tuple(root / f"episode_{index:04d}.npz" for index in range(count))
            missing = [path for path in paths if path not in discovered]
            if missing:
                preview = ", ".join(path.name for path in missing[:5])
                raise FileNotFoundError(
                    f"{root} is missing {len(missing)} required episode files: {preview}"
                )
            ignored_paths = tuple(sorted(discovered.difference(paths), key=_episode_number))
        if not paths:
            raise ValueError(f"No episode_*.npz files found in {root}")
        groups.append(SourceGroup(root=root, paths=paths, ignored_paths=ignored_paths))
    return groups


def _episode_number(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("episode_"))
    except ValueError as exc:
        raise ValueError(f"Invalid episode filename: {path.name}") from exc


def extract_tip_state(physical: np.ndarray) -> np.ndarray:
    """Map a [T,41] full physical state to the [T,11] end-tip state.

    The 11-D layout mirrors the observation order used at collection time:
    tip position, tip speed, tip velocity direction, tip quaternion.
    """

    if physical.ndim != 2 or physical.shape[1] != SOURCE_OBSERVATION_DIM:
        raise ValueError(
            f"tip extraction requires [T,{SOURCE_OBSERVATION_DIM}] states, "
            f"got {physical.shape}"
        )
    tip_position = physical[:, TIP_POSITION_INDICES]
    tip_speed = physical[:, 33:34]
    tip_velocity_direction = physical[:, 34:37]
    tip_quaternion = physical[:, 37:41]
    tip_state = np.concatenate(
        (tip_position, tip_speed, tip_velocity_direction, tip_quaternion),
        axis=1,
    ).astype(np.float32, copy=False)
    if tip_state.shape[1] != TIP_STATE_DIM:
        raise RuntimeError(
            f"tip state must be {TIP_STATE_DIM}-D, got {tip_state.shape[1]}-D"
        )
    if not np.isfinite(tip_state).all():
        raise ValueError("tip state contains NaN or Inf")
    return tip_state


def load_manisoft_episode(
    path: Path,
    episode_id: int,
    continuity_atol: float = 1e-6,
) -> tuple[AugmentedDataset, float]:
    """Load one ``(s_t, u_t, s_{t+1})`` episode and build schema-v2 rows.

    The stored 41-D physical state is compressed to the 11-D tip state before
    the previous-action block is appended.
    """

    with np.load(path, allow_pickle=False) as archive:
        missing = [field for field in SOURCE_FIELDS if field not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing fields: {missing}")
        physical_state = np.asarray(archive["state"], dtype=np.float32)
        current_action = np.asarray(archive["action"], dtype=np.float32)
        next_physical_state = np.asarray(archive["next_state"], dtype=np.float32)

    if physical_state.ndim != 2 or physical_state.shape[1] != SOURCE_OBSERVATION_DIM:
        raise ValueError(
            f"{path}: state must have shape [T,{SOURCE_OBSERVATION_DIM}], "
            f"got {physical_state.shape}"
        )
    if next_physical_state.shape != physical_state.shape:
        raise ValueError(
            f"{path}: next_state shape {next_physical_state.shape} does not match "
            f"state shape {physical_state.shape}"
        )
    if current_action.ndim != 2 or current_action.shape != (
        len(physical_state),
        ACTION_DIM,
    ):
        raise ValueError(
            f"{path}: action must have shape [{len(physical_state)},{ACTION_DIM}], "
            f"got {current_action.shape}"
        )
    if not len(physical_state):
        raise ValueError(f"{path}: episode is empty")
    for name, values in (
        ("state", physical_state),
        ("action", current_action),
        ("next_state", next_physical_state),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {name} contains NaN or infinity")

    tip_state = extract_tip_state(physical_state)
    next_tip_state = extract_tip_state(next_physical_state)

    continuity_error = 0.0
    if len(tip_state) > 1:
        continuity_error = float(
            np.max(np.abs(next_tip_state[:-1] - tip_state[1:]))
        )
        if continuity_error > continuity_atol:
            raise ValueError(
                f"{path}: max |next_tip_state[t]-tip_state[t+1]|="
                f"{continuity_error:.3e} exceeds {continuity_atol:.1e}"
            )

    previous_action = np.zeros_like(current_action)
    previous_action[1:] = current_action[:-1]
    delta_action = current_action - previous_action
    rows = len(tip_state)
    timeout = np.zeros(rows, dtype=bool)
    timeout[-1] = True
    dataset = AugmentedDataset(
        state=np.concatenate((tip_state, previous_action), axis=1),
        action=delta_action,
        next_state=np.concatenate((next_tip_state, current_action), axis=1),
        reward=np.zeros(rows, dtype=np.float32),
        done=timeout.copy(),
        terminal=np.zeros(rows, dtype=bool),
        timeout=timeout,
        episode_id=np.full(rows, episode_id, dtype=np.int64),
        step_index=np.arange(rows, dtype=np.int64),
        current_action=current_action,
    )
    dataset.validate()
    return dataset, continuity_error


def concatenate_episodes(episodes: Sequence[AugmentedDataset]) -> AugmentedDataset:
    if not episodes:
        raise ValueError("No episodes were loaded")
    dataset = AugmentedDataset(
        **{
            field: np.concatenate([getattr(episode, field) for episode in episodes], axis=0)
            for field in AugmentedDataset.__dataclass_fields__
        }
    )
    dataset.validate()
    return dataset


def save_dataset(path: Path, dataset: AugmentedDataset) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **dataset.as_dict())
    temporary.replace(path)


def save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def convert(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)

    groups = discover_source_groups(args.input_root, args.episode_counts)
    total_files = sum(len(group.paths) for group in groups)
    if args.expected_episodes is not None and total_files != args.expected_episodes:
        raise ValueError(
            f"Expected {args.expected_episodes} episodes, resolved {total_files}"
        )

    episodes: list[AugmentedDataset] = []
    max_continuity_error = 0.0
    source_metadata: list[dict[str, object]] = []
    global_episode_id = 0
    for group in groups:
        group_rows = 0
        first_global_id = global_episode_id
        for source_index, path in enumerate(group.paths, start=1):
            episode, continuity_error = load_manisoft_episode(
                path,
                episode_id=global_episode_id,
                continuity_atol=args.continuity_atol,
            )
            episodes.append(episode)
            group_rows += len(episode)
            max_continuity_error = max(max_continuity_error, continuity_error)
            global_episode_id += 1
            if source_index % args.progress_every == 0 or source_index == len(group.paths):
                print(
                    f"loaded {source_index}/{len(group.paths)} from {group.root} "
                    f"({group_rows} transitions)",
                    flush=True,
                )
        source_metadata.append(
            {
                "root": str(group.root),
                "episodes": len(group.paths),
                "transitions": group_rows,
                "first_file": group.paths[0].name,
                "last_file": group.paths[-1].name,
                "global_episode_ids": [first_global_id, global_episode_id - 1],
                "ignored_extra_files": len(group.ignored_paths),
            }
        )

    episode_count = len(episodes)
    dataset = concatenate_episodes(episodes)
    episodes.clear()
    fractions = (
        config["data"]["train_fraction"],
        config["data"]["validation_fraction"],
        config["data"]["test_fraction"],
    )
    split_seed = int(config["experiment"]["seed"] if args.split_seed is None else args.split_seed)
    splits = split_by_episode(dataset, fractions=fractions, seed=split_seed)
    normalizer = Normalizer.fit(
        splits["train"].state,
        epsilon=float(config["data"]["normalization_epsilon"]),
    )
    k_step = int(config["koopman"]["K_step"])
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    reconstruction_error = float(
        np.max(
            np.abs(
                dataset.current_action
                - (dataset.state[:, -ACTION_DIM:] + dataset.action)
            )
        )
    )
    metadata: dict[str, object] = {
        "dataset_schema_version": 2,
        "state_layout": {
            "tip_position": {"slice": [0, 3], "description": "end-tip node position (m)"},
            "tip_speed": {"slice": [3, 4], "description": "end-tip velocity norm (m/s)"},
            "tip_velocity_direction": {
                "slice": [4, 7],
                "description": "end-tip velocity unit vector, zero at rest",
            },
            "tip_quaternion_wxyz": {
                "slice": [7, 11],
                "description": "end-tip orientation quaternion (wxyz)",
            },
            "previous_action": {"slice": [11, 29], "description": "18-D u_{t-1}"},
        },
        "transition_semantics": {
            "source_state": "41D physical state s_t (compressed to 11D tip state)",
            "state": "[tip_state_t, previous_action=u_{t-1}]",
            "action": "delta_action=u_t-u_{t-1} (physical units, not normalized)",
            "next_state": "[tip_state_{t+1}, current_action=u_t]",
            "current_action": "18D absolute ManiSoft command u_t",
            "episode_initial_previous_action": "u_-1=0",
            "done": "terminal OR timeout; final row of each source file is timeout",
        },
        "source_groups": source_metadata,
        "observation_dim": TIP_STATE_DIM,
        "source_observation_dim": SOURCE_OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "augmented_state_dim": AUGMENTED_STATE_DIM,
        "transitions": len(dataset),
        "episodes": episode_count,
        "split_seed": split_seed,
        "K_step": k_step,
        "max_source_temporal_continuity_error": max_continuity_error,
        "max_action_reconstruction_error": reconstruction_error,
        "normalizer": normalizer.state_dict(),
        "action_normalizer": "physical_units",
        "splits": {},
    }
    for name, split in splits.items():
        save_dataset(output / f"{name}.npz", split)
        starts = valid_window_starts(split, k_step)
        save_array(output / f"{name}_K{k_step}_starts.npy", starts)
        metadata["splits"][name] = {
            "rows": len(split),
            "episodes": len(np.unique(split.episode_id)),
            "valid_K_step_windows": len(starts),
        }
        print(
            f"saved {name}: rows={len(split)}, episodes={len(np.unique(split.episode_id))}, "
            f"K{k_step}_windows={len(starts)}",
            flush=True,
        )

    metadata_path = output / "metadata.json"
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge ManiSoft episode NPZ files into an 11-D tip-only Koopman "
            "dataset with 29-D augmented state / 18-D delta action."
        )
    )
    parser.add_argument("--config", default="configs/manisoft_coll.yaml")
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        help="Episode directory; repeat once for every source directory.",
    )
    parser.add_argument(
        "--episode-counts",
        type=int,
        nargs="+",
        default=None,
        help="Files episode_0000.npz through episode_(N-1).npz to use per root.",
    )
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--continuity-atol", type=float, default=1e-6)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.continuity_atol < 0:
        parser.error("--continuity-atol must be non-negative")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
