#!/usr/bin/env python
"""Merge multiple per-part KMPC offline collections into one dataset.

Each input root is a collection directory laid out as
``root/part_i/episodes/episode_*.npz`` (produced by
``collect_manisoft_kmpc_offline_dataset.py``).  Shards are merged in
``--roots`` order (root by root, part by part, episode by episode), episode
ids are renumbered to be globally unique across all roots, and the result is
written as one D4RL-style ``dataset.npz`` plus a combined ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from antmaze_ac.data.offline_episodes import (
    atomic_savez_compressed,
    validate_episode_arrays,
)


def _part_shards(part_dir: Path) -> list[Path]:
    episode_root = part_dir / "episodes"
    if not episode_root.is_dir():
        raise FileNotFoundError(f"Missing episode root: {episode_root}")
    shards = sorted(episode_root.glob("episode_*.npz"))
    expected = [
        episode_root / f"episode_{index:06d}.npz" for index in range(len(shards))
    ]
    if shards != expected:
        raise ValueError(
            f"Part {part_dir.name} shards are not a contiguous prefix"
        )
    return shards


def _episode_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    rewards = arrays["rewards"]
    return {
        "steps": int(len(rewards)),
        "return": float(rewards.sum()),
        "success": bool(np.asarray(arrays["terminals"])[-1]),
        "waypoints_completed": int(
            np.asarray(arrays["waypoints_completed"]).max(initial=0)
        ),
        "final_distance": float(np.asarray(arrays["next_active_distances"])[-1]),
        "minimum_distance": float(np.asarray(arrays["next_active_distances"]).min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = [Path(root).expanduser().resolve() for root in args.roots]
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    chunks: dict[str, list[np.ndarray]] = {}
    expected_keys: tuple[str, ...] | None = None
    total_transitions = 0
    episode_offset = 0
    summaries: list[dict[str, Any]] = []
    root_summaries: list[dict[str, Any]] = []

    for root in roots:
        root_episodes = 0
        root_transitions = 0
        for part_index in range(args.parts):
            part_dir = root / f"part_{part_index}"
            if not part_dir.is_dir():
                raise FileNotFoundError(f"Missing part dir: {part_dir}")
            shards = _part_shards(part_dir)
            for shard in shards:
                with np.load(shard, allow_pickle=False) as archive:
                    keys = tuple(sorted(archive.files))
                    if expected_keys is None:
                        expected_keys = keys
                    elif keys != expected_keys:
                        raise ValueError(f"Episode shard fields differ: {shard}")
                    episode = {key: np.asarray(archive[key]) for key in keys}
                total_transitions += validate_episode_arrays(episode)
                # Within a part local ids are 0..n-1, so add the fixed base
                # offset of all episodes from previously processed parts.
                episode["episode_ids"] = (
                    episode["episode_ids"].astype(np.int64, copy=False)
                    + episode_offset
                )
                root_episodes += 1
                root_transitions += int(episode["observations"].shape[0])
                summaries.append(
                    {
                        "episode_ids": int(episode["episode_ids"][0]),
                        "root": str(root),
                        **_episode_summary(episode),
                    }
                )
                for key, value in episode.items():
                    chunks.setdefault(key, []).append(value)
            episode_offset += len(shards)
        root_summaries.append(
            {
                "root": str(root),
                "episodes": root_episodes,
                "transitions": root_transitions,
            }
        )

    merged = {
        key: np.concatenate(values, axis=0)
        for key, values in chunks.items()
    }
    atomic_savez_compressed(output, merged)

    report = {
        "merged_dataset": str(output),
        "roots": root_summaries,
        "episodes": episode_offset,
        "transitions": total_transitions,
        "success_rate": float(np.mean([row["success"] for row in summaries])),
        "return_mean": float(np.mean([row["return"] for row in summaries])),
        "episode_steps_mean": float(np.mean([row["steps"] for row in summaries])),
        "waypoints_completed_mean": float(
            np.mean([row["waypoints_completed"] for row in summaries])
        ),
        "episode_summaries": summaries,
    }
    report_path = output.with_name("summary.json")
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
