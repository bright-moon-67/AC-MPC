#!/usr/bin/env python
"""Merge multiple expert dataset parts into a single BC-KMPC dataset.

Each part is produced by collect_manisoft_bc_kmpc_expert.py and contains
episode ids starting from 0.  This script re-bases the episode ids so the
merged dataset has globally unique ids, validates that all parts reference
the same waypoint set, and writes a merged expert.npz + expert.json.

Usage:
    python scripts/merge_expert_datasets.py \
        --inputs data/.../part0/expert.npz data/.../part1/expert.npz \
        --output data/.../expert.npz
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
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Expert dataset part .npz files to merge (in order).",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_report(npz_path: Path) -> dict:
    report_path = npz_path.with_suffix(".json")
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing metadata for part: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    inputs = [Path(p).expanduser().resolve() for p in args.inputs]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset part: {path}")

    reports = [_load_report(path) for path in inputs]
    base = reports[0]
    if int(base.get("schema_version", 0)) < 4:
        raise ValueError("Input dataset predates absolute-action box BC-KMPC")
    for idx, report in enumerate(reports[1:], start=1):
        if int(report.get("schema_version", 0)) < 4:
            raise ValueError(
                f"Part {idx} predates absolute-action box BC-KMPC"
            )
        if report.get("reference_sha256") != base.get("reference_sha256"):
            raise ValueError(f"Part {idx} references another waypoint set")
        if report.get("action_sha256") != base.get("action_sha256"):
            raise ValueError(f"Part {idx} references another action set")
        if report.get("observation_dim") != base.get("observation_dim"):
            raise ValueError(f"Part {idx} has an incompatible observation dim")
        if report.get("action_dim") != base.get("action_dim"):
            raise ValueError(f"Part {idx} has an incompatible action dim")

    merged: dict[str, np.ndarray] = {}
    episode_offset = 0
    total_samples = 0
    total_episodes = 0
    episode_return_sum = 0.0
    for idx, (path, report) in enumerate(zip(inputs, reports)):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in ARRAY_NAMES}
        count = len(arrays["observation"])
        if any(len(arrays[name]) != count for name in ARRAY_NAMES):
            raise ValueError(f"Part {idx} arrays have inconsistent lengths")
        episode_ids = arrays["episode_id"].astype(np.int64)
        if episode_offset:
            arrays["episode_id"] = episode_ids + episode_offset
        if idx == 0:
            merged = arrays
        else:
            for name in ARRAY_NAMES:
                merged[name] = np.concatenate((merged[name], arrays[name]), axis=0)
        episode_offset = int(arrays["episode_id"].max()) + 1
        total_samples += count
        total_episodes += int(report.get("episodes", 0))
        episode_return_sum += float(report.get("episode_return_mean", 0.0)) * int(
            report.get("episodes", 0)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **merged)
    temporary.replace(output)

    merged_report = {
        "schema_version": base.get("schema_version"),
        "fixed_smoothness": base.get("fixed_smoothness"),
        "action_constraints": base.get("action_constraints"),
        "kind": base.get("kind"),
        "output": str(output),
        "samples": total_samples,
        "base_samples": 0,
        "new_samples": total_samples,
        "episodes": total_episodes,
        "episode_return_mean": (
            episode_return_sum / total_episodes if total_episodes else 0.0
        ),
        "observation_dim": base.get("observation_dim"),
        "action_dim": base.get("action_dim"),
        "history_steps": base.get("history_steps"),
        "koopman_checkpoint": base.get("koopman_checkpoint"),
        "koopman_checkpoint_sha256": base.get("koopman_checkpoint_sha256"),
        "waypoint_root": base.get("waypoint_root"),
        "references": base.get("references"),
        "reference_sha256": base.get("reference_sha256"),
        "actions": base.get("actions"),
        "action_sha256": base.get("action_sha256"),
        "scenario": base.get("scenario"),
        "base_dataset": None,
        "base_dataset_sha256": None,
        "rollout_checkpoint": None,
        "rollout_checkpoint_sha256": None,
        "merged_parts": [str(path) for path in inputs],
        "runtime": {
            "parts": [
                {
                    "path": str(path),
                    "samples": report.get("samples"),
                    "episodes": report.get("episodes"),
                    "seed": (report.get("runtime") or {}).get("seed"),
                }
                for path, report in zip(inputs, reports)
            ]
        },
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(merged_report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(merged_report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
