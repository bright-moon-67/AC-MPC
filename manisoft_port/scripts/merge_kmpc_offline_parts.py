#!/usr/bin/env python
"""Combine per-part KMPC offline-collection shards into one dataset.

Each part directory (part_0/, part_1/, ...) contains episode_*.npz shards
whose local episode_ids start at 0.  This script:

  * validates every shard,
  * renumbers episode_ids so they are globally unique across parts,
  * concatenates all arrays into a single D4RL-style dataset.npz,
  * writes a combined summary.json with aggregated statistics.
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
    parser.add_argument("--root", required=True)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "dataset.npz"
    )
    part_dirs = [root / f"part_{i}" for i in range(args.parts)]
    missing = [path for path in part_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing part directories: {missing}")

    chunks: dict[str, list[np.ndarray]] = {}
    expected_keys: tuple[str, ...] | None = None
    total_transitions = 0
    episode_offset = 0
    summaries: list[dict[str, Any]] = []
    part_summaries: list[dict[str, Any]] = []

    for part_dir in part_dirs:
        shards = _part_shards(part_dir)
        part_transitions = 0
        for shard in shards:
            with np.load(shard, allow_pickle=False) as archive:
                keys = tuple(sorted(archive.files))
                if expected_keys is None:
                    expected_keys = keys
                elif keys != expected_keys:
                    raise ValueError(f"Episode shard fields differ: {shard}")
                episode = {
                    key: np.asarray(archive[key]) for key in keys
                }
            total_transitions += validate_episode_arrays(episode)
            part_transitions += int(episode["observations"].shape[0])

            # Renumber episode ids so they are globally unique after merge.
            # Within a part local ids are 0..n-1, so add the fixed base offset
            # of all episodes collected by earlier parts.
            episode["episode_ids"] = (
                episode["episode_ids"].astype(np.int64, copy=False) + episode_offset
            )
            summaries.append({"episode_ids": int(episode["episode_ids"][0]), **_episode_summary(episode)})
            for key, value in episode.items():
                chunks.setdefault(key, []).append(value)
        part_summaries.append(
            {"part": part_dir.name, "episodes": len(shards), "transitions": part_transitions}
        )
        episode_offset += len(shards)

    merged = {
        key: np.concatenate(values, axis=0)
        for key, values in chunks.items()
    }
    atomic_savez_compressed(output, merged)

    report = {
        "merged_dataset": str(output),
        "parts": part_summaries,
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
