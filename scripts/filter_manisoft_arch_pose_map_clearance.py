#!/usr/bin/env python
"""Filter a certified ManiSoft pose map by whole-arm table clearance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _centered_hull_radius(points: np.ndarray, center: np.ndarray) -> float:
    hull = ConvexHull(points)
    distances = -(
        hull.equations[:, :2] @ center + hull.equations[:, 2]
    ) / np.linalg.norm(hull.equations[:, :2], axis=1)
    return float(np.min(distances))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-clearance", type=float, required=True)
    args = parser.parse_args()
    if args.minimum_clearance <= 0:
        raise ValueError("minimum-clearance must be positive")
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if (
        str(payload["kind"].reshape(()).item())
        != "manisoft_table_arch_pose_map"
        or "minimum_table_clearances" not in payload
    ):
        raise ValueError("source is not a pose map with clearance certificates")
    tips = np.asarray(payload["tip_positions"], dtype=np.float64)
    clearances = np.asarray(
        payload["minimum_table_clearances"], dtype=np.float64
    )
    if clearances.shape != (len(tips),):
        raise ValueError("minimum_table_clearances does not match tip_positions")
    keep = clearances >= args.minimum_clearance
    # The entry bank independently certifies the first pose's complete path;
    # retain it even when an older map stored only a conservative placeholder.
    keep[0] = True
    if int(np.sum(keep)) < 4:
        raise RuntimeError("clearance filter leaves too few pose samples")
    for key, value in tuple(payload.items()):
        if value.ndim >= 1 and value.shape[0] == len(tips):
            payload[key] = value[keep]
    kept_tips = np.asarray(payload["tip_positions"], dtype=np.float64)
    center = np.asarray(payload["certified_center_xy"], dtype=np.float64)
    raw_radius = _centered_hull_radius(kept_tips[:, :2], center)
    if raw_radius <= 0:
        raise RuntimeError("entry point lies outside filtered pose-map hull")
    payload["raw_hull_radius"] = np.asarray(raw_radius, dtype=np.float32)
    payload["certified_radius"] = np.asarray(0.80 * raw_radius, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(output)
    summary = {
        "kind": "manisoft_table_arch_pose_map_clearance_manifest",
        "map": str(output),
        "map_sha256": _sha256(output),
        "source_map": str(source),
        "source_map_sha256": _sha256(source),
        "minimum_clearance": args.minimum_clearance,
        "source_samples": int(len(tips)),
        "kept_samples": int(len(kept_tips)),
        "raw_centered_radius": raw_radius,
        "certified_radius": 0.80 * raw_radius,
        "tip_minimum": kept_tips.min(axis=0).tolist(),
        "tip_maximum": kept_tips.max(axis=0).tolist(),
    }
    output.with_name("manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
