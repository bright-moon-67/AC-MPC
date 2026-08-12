#!/usr/bin/env python
"""Merge multiple certified waypoint-bank shards into one manifest.

Each shard produced by generate_manisoft_waypoint_bank.py contains
triplet_XXXX/waypoint_{1,2,3}.npz files (and possibly a manifest.json when the
shard completed fully).  This script scans every input directory, copies all
certified triplets into a single output bank with renumbered triplet indices,
recomputes the per-waypoint metrics from the stored .npz data (so it also works
for shards that stopped early without writing a manifest), and writes a fresh
manifest.json that mirrors the generator's schema.

Triplets are ordered by input-directory order, then by triplet index inside
each directory, so the primary run (v1) comes first.

Usage:
  python scripts/merge_waypoint_banks.py \
      --inputs data/processed/manisoft_waypoint_bank_v1 \
               data/processed/manisoft_waypoint_bank_v1_shard1 ... \
      --output data/processed/manisoft_waypoint_bank_v1_merged \
      [--limit 200] [--scenario ...] [--seed 42] [--stable-steps 250]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

TIP_INDICES = np.asarray((30, 31, 32), dtype=np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_max(positions: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(positions - reference[None, :], axis=1).max())


def _load_triplet(bank_dir: Path, triplet_index: int) -> dict | None:
    """Reconstruct one triplet's metadata from its .npz files."""
    triplet_dir = bank_dir / f"triplet_{triplet_index:04d}"
    waypoint_rows = []
    for waypoint_index in (1, 2, 3):
        path = triplet_dir / f"waypoint_{waypoint_index}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            reference_tip = data["reference_tip_position"].astype(np.float64)
            initial_tip = data["initial_tip_position"].astype(np.float64)
            replay_tip = data["replay_reference_tip_position"].astype(np.float64)
            stable_positions = np.asarray(data["stable_window_tip_positions"])
            stable_speeds = np.asarray(data["stable_window_tip_speeds"])
            replay_positions = np.asarray(data["replay_stable_window_tip_positions"])
            replay_speeds = np.asarray(data["replay_stable_window_tip_speeds"])
            scale = float(data["action_scale"])
        waypoint_rows.append(
            {
                "index": waypoint_index - 1,
                "scale": scale,
                "reference": f"triplet_{triplet_index:04d}/waypoint_{waypoint_index}.npz",
                "sha256": _sha256(path),
                "tip_position_m": reference_tip.tolist(),
                "distance_from_initial_m": float(
                    np.linalg.norm(reference_tip - initial_tip)
                ),
                "generation_position_max_m": _norm_max(stable_positions, reference_tip),
                "generation_speed_max_m_per_s": float(stable_speeds.max()),
                "replay_position_max_m": _norm_max(replay_positions, replay_tip),
                "replay_speed_max_m_per_s": float(replay_speeds.max()),
                "replay_tip_error_m": float(np.linalg.norm(replay_tip - reference_tip)),
            }
        )
    return {"waypoints": waypoint_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scenario", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stable-steps", type=int, default=250)
    parser.add_argument("--control-hz", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)

    collected: list[dict] = []  # triplet dir metadata
    for bank_dir in args.inputs:
        bank_dir = bank_dir.expanduser().resolve()
        if not bank_dir.is_dir():
            print(f"[skip] missing bank dir: {bank_dir}")
            continue
        indices = sorted(
            int(p.name.split("_")[1])
            for p in bank_dir.glob("triplet_*")
            if p.is_dir()
        )
        for triplet_index in indices:
            triplet = _load_triplet(bank_dir, triplet_index)
            if triplet is None:
                print(
                    f"[skip] incomplete triplet {triplet_index} in {bank_dir}"
                )
                continue
            triplet["_source"] = bank_dir
            collected.append(triplet)

    if args.limit > 0:
        collected = collected[: args.limit]

    # Re-number and copy files.
    triplets = []
    for new_index, item in enumerate(collected):
        source_triplet = item["_source"]
        old_index = int(item["waypoints"][0]["reference"].split("/")[0].split("_")[1])
        src_dir = source_triplet / f"triplet_{old_index:04d}"
        dst_dir = output / f"triplet_{new_index:04d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for waypoint_index in (1, 2, 3):
            shutil.copy2(
                src_dir / f"waypoint_{waypoint_index}.npz",
                dst_dir / f"waypoint_{waypoint_index}.npz",
            )
        rows = []
        for row in item["waypoints"]:
            row = dict(row)
            row["reference"] = f"triplet_{new_index:04d}/waypoint_{row['index'] + 1}.npz"
            rows.append(row)
        triplets.append(
            {
                "index": new_index,
                "source_attempt": None,
                "base_action": None,
                "waypoints": rows,
            }
        )

    scenario = args.scenario
    if scenario is not None:
        scenario = scenario.expanduser().resolve()
        scenario_sha256 = _sha256(scenario)
    else:
        scenario = None
        scenario_sha256 = None

    manifest = {
        "schema_version": 1,
        "kind": "manisoft_certified_three_waypoint_reference_bank",
        "scenario": str(scenario) if scenario else None,
        "scenario_sha256": scenario_sha256,
        "triplet_count": len(triplets),
        "waypoint_count": 3,
        "state_dim": 45,
        "action_dim": 18,
        "action_limit": 0.30,
        "seed": args.seed,
        "certification": {
            "control_hz": args.control_hz,
            "stable_steps": args.stable_steps,
            "stable_seconds": args.stable_steps / args.control_hz,
            "merged_from": [str(p) for p in args.inputs],
        },
        "triplets": triplets,
    }
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "triplets": len(triplets)}))


if __name__ == "__main__":
    main()
