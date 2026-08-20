#!/usr/bin/env python
"""Extract a random subset of certified waypoint triplets into a new bank.

Sampling uses a fixed seed for reproducibility.  Waypoint .npz files are
copied byte-for-byte, so the per-waypoint sha256 hashes and metrics from the
source manifest remain valid; only triplet indices and reference paths are
re-numbered.

Usage:
  python scripts/extract_waypoint_bank_subset.py \
      --source data/processed/manisoft_waypoint_bank_v4_full_merged \
      --output data/processed/manisoft_waypoint_bank_v4_test20 \
      --count 20 --seed 42 \
      --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if args.count < 1:
        raise ValueError("--count must be positive")

    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    triplet_count = int(source_manifest["triplet_count"])
    if source_manifest.get("waypoint_count") != 3:
        raise ValueError("Source bank must contain exactly three waypoints per triplet")
    if args.count > triplet_count:
        raise ValueError(f"--count {args.count} exceeds source triplets {triplet_count}")

    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(triplet_count, size=args.count, replace=False))

    output.mkdir(parents=True)
    triplets = []
    for new_index, old_index in enumerate(chosen.tolist()):
        old = source_manifest["triplets"][old_index]
        src_dir = source / f"triplet_{old_index:04d}"
        dst_dir = output / f"triplet_{new_index:04d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for waypoint_index in (1, 2, 3):
            shutil.copy2(
                src_dir / f"waypoint_{waypoint_index}.npz",
                dst_dir / f"waypoint_{waypoint_index}.npz",
            )
        rows = []
        for row in old["waypoints"]:
            row = dict(row)
            row["reference"] = (
                f"triplet_{new_index:04d}/waypoint_{int(row['index']) + 1}.npz"
            )
            rows.append(row)
        triplets.append(
            {
                "index": new_index,
                "source_attempt": old.get("source_attempt"),
                "base_action": old.get("base_action"),
                "source_triplet_index": old_index,
                "waypoints": rows,
            }
        )

    scenario = (
        args.scenario.expanduser().resolve() if args.scenario else None
    )
    manifest = dict(source_manifest)
    manifest.update(
        {
            "scenario": str(scenario) if scenario else manifest.get("scenario"),
            "triplet_count": len(triplets),
            "seed": args.seed,
            "extracted_from": str(source),
            "triplets": triplets,
        }
    )
    if scenario is not None:
        import hashlib

        digest = hashlib.sha256()
        with scenario.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["scenario_sha256"] = digest.hexdigest()

    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "triplets": len(triplets),
                "chosen_source_indices": chosen.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
