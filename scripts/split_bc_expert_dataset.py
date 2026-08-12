#!/usr/bin/env python
"""Build the primary BC expert set from complete (all-waypoint) episodes.

Extracts the episodes that completed all three waypoints from the v5 and v6
expert datasets, merges them into a single primary BC expert set, then splits
the episodes into train/validation sets at the episode level (never at the
individual step level).

All original data (v5 + v6 = 270 episodes) is left untouched.

Usage:
    python scripts/split_bc_expert_dataset.py \
        --v5 data/processed/manisoft_bc_kmpc_three_waypoint_v5/expert.npz \
        --v6 data/processed/manisoft_bc_kmpc_three_waypoint_v6/expert.npz \
        --output-dir data/processed/manisoft_bc_kmpc_three_waypoint_v7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARRAY_NAMES = (
    "observation",
    "expert_action",
    "applied_action",
    "episode_id",
    "step_index",
    "expert_cost",
    "qp_iterations",
    "active_waypoint_index",
    "waypoint_passed",
    "waypoints_completed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5", required=True, help="v5 expert dataset .npz")
    parser.add_argument("--v6", required=True, help="v6 expert dataset .npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in ARRAY_NAMES}


def _complete_episodes(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, list]:
    """Mask of samples belonging to episodes that finished all waypoints.

    An episode is complete when its final sample has waypoints_completed
    equal to the waypoint count (3).
    """
    episode_ids = arrays["episode_id"]
    waypoint_count = int(arrays["waypoints_completed"].max())
    keep = np.zeros(len(episode_ids), dtype=bool)
    complete: list[int] = []
    for episode in np.unique(episode_ids):
        index = np.flatnonzero(episode_ids == episode)
        if arrays["waypoints_completed"][index[-1]] >= waypoint_count:
            keep[index] = True
            complete.append(int(episode))
    return keep, complete


def _report(path: Path) -> dict:
    report_path = path.with_suffix(".json")
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    v5_path = Path(args.v5).expanduser().resolve()
    v6_path = Path(args.v6).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not v5_path.is_file() or not v6_path.is_file():
        raise FileNotFoundError("Both v5 and v6 datasets must exist")
    v5_report = _report(v5_path)
    v6_report = _report(v6_path)
    if min(
        int(v5_report.get("schema_version", 0)),
        int(v6_report.get("schema_version", 0)),
    ) < 4:
        raise ValueError(
            "Input datasets predate absolute-action box BC-KMPC; recollect them"
        )
    if v5_report.get("reference_sha256") != v6_report.get("reference_sha256"):
        raise ValueError("v5 and v6 reference different waypoint sets")
    if v5_report.get("action_sha256") != v6_report.get("action_sha256"):
        raise ValueError("v5 and v6 reference different action sets")

    merged: dict[str, np.ndarray] | None = None
    global_offset = 0
    source_map: dict[str, dict] = {}
    complete_counts: list[int] = []
    for tag, path, report in (("v5", v5_path, v5_report), ("v6", v6_path, v6_report)):
        arrays = _load(path)
        keep, complete = _complete_episodes(arrays)
        selected = {name: arrays[name][keep] for name in ARRAY_NAMES}
        selected["episode_id"] = selected["episode_id"].astype(np.int64) + global_offset
        old_ids = [int(ep) for ep in complete]
        new_ids = [
            int(ep) + global_offset for ep in old_ids
        ]
        for old_id, new_id in zip(old_ids, new_ids):
            source_map[str(new_id)] = {"source": tag, "original_episode_id": old_id}
        if merged is None:
            merged = selected
        else:
            for name in ARRAY_NAMES:
                merged[name] = np.concatenate((merged[name], selected[name]), axis=0)
        global_offset = int(selected["episode_id"].max()) + 1
        complete_counts.append(len(complete))

    episode_ids = merged["episode_id"]
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episodes)
    if len(episodes) > 1:
        validation_count = max(
            1, int(round(len(episodes) * args.validation_fraction))
        )
        validation_count = min(validation_count, len(episodes) - 1)
        validation_episodes = episodes[:validation_count]
    else:
        raise ValueError("Need at least two complete episodes to split")
    validation_mask = np.isin(episode_ids, validation_episodes)
    train_arrays = {name: merged[name][~validation_mask] for name in ARRAY_NAMES}
    validation_arrays = {
        name: merged[name][validation_mask] for name in ARRAY_NAMES
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("bc_expert", "train", "val"):
        target = output_dir / f"{name}.npz"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")

    _save(output_dir / "bc_expert.npz", merged)
    _save(output_dir / "train.npz", train_arrays)
    _save(output_dir / "val.npz", validation_arrays)

    base_report = v6_report
    common = {
        "schema_version": base_report.get("schema_version"),
        "fixed_smoothness": base_report.get("fixed_smoothness"),
        "action_constraints": base_report.get("action_constraints"),
        "kind": base_report.get("kind"),
        "observation_dim": base_report.get("observation_dim"),
        "action_dim": base_report.get("action_dim"),
        "history_steps": base_report.get("history_steps"),
        "koopman_checkpoint": base_report.get("koopman_checkpoint"),
        "koopman_checkpoint_sha256": base_report.get("koopman_checkpoint_sha256"),
        "waypoint_root": base_report.get("waypoint_root"),
        "references": base_report.get("references"),
        "reference_sha256": base_report.get("reference_sha256"),
        "actions": base_report.get("actions"),
        "action_sha256": base_report.get("action_sha256"),
        "scenario": base_report.get("scenario"),
        "base_dataset": None,
        "base_dataset_sha256": None,
        "rollout_checkpoint": None,
        "rollout_checkpoint_sha256": None,
    }
    episode_return_estimate = (
        float(v5_report.get("episode_return_mean", 0.0)) * complete_counts[0]
        + float(v6_report.get("episode_return_mean", 0.0)) * complete_counts[1]
    ) / sum(complete_counts)

    for name, arrays in (
        ("bc_expert", merged),
        ("train", train_arrays),
        ("val", validation_arrays),
    ):
        report = dict(common)
        report.update(
            {
                "output": str(output_dir / f"{name}.npz"),
                "samples": int(len(arrays["observation"])),
                "episodes": int(len(np.unique(arrays["episode_id"]))),
                "episode_return_mean": float(episode_return_estimate),
                "episode_return_estimated": True,
                "complete_episodes_only": True,
                "runtime": {
                    "v5": {
                        "path": str(v5_path),
                        "complete_episodes": complete_counts[0],
                    },
                    "v6": {
                        "path": str(v6_path),
                        "complete_episodes": complete_counts[1],
                    },
                    "validation_fraction": args.validation_fraction,
                    "seed": args.seed,
                },
            }
        )
        _write_json(output_dir / f"{name}.json", report)

    split = {
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "total_complete_episodes": int(len(episodes)),
        "train_episodes": int(len(episodes) - validation_count),
        "validation_episodes": int(validation_count),
        "train_episode_ids": [int(e) for e in episodes[validation_count:]],
        "validation_episode_ids": [int(e) for e in validation_episodes],
        "episode_source_map": source_map,
    }
    _write_json(output_dir / "split.json", split)
    print(json.dumps(split, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
