#!/usr/bin/env python
"""Restrict a certified arch pose map to a near-planar tip-height band."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--half-width", type=float, default=0.005)
    return parser.parse_args()


def _centered_hull_radius(points: np.ndarray, center: np.ndarray) -> float:
    hull = ConvexHull(points)
    distances = -(
        hull.equations[:, :2] @ center + hull.equations[:, 2]
    ) / np.linalg.norm(hull.equations[:, :2], axis=1)
    return float(np.min(distances))


def main() -> None:
    args = parse_args()
    if args.half_width <= 0:
        raise ValueError("half-width must be positive")
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    if (
        int(payload["schema_version"].reshape(()).item()) != 1
        or str(payload["kind"].reshape(()).item())
        != "manisoft_table_arch_pose_map"
    ):
        raise ValueError("unexpected source pose-map schema")
    tips = np.asarray(payload["tip_positions"], dtype=np.float64)
    actions = np.asarray(payload["physical_actions"], dtype=np.float64)
    indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
    if tips.ndim != 2 or tips.shape[1] != 3 or actions.shape != (len(tips), 18):
        raise ValueError("source pose map contains invalid arrays")
    target_z = float(tips[0, 2])
    keep = np.abs(tips[:, 2] - target_z) <= args.half_width + 1e-12
    keep[0] = True
    tips = tips[keep]
    actions = actions[keep]
    indices = indices[keep]
    if len(tips) < 4:
        raise RuntimeError("height band leaves too few pose samples")
    hull = ConvexHull(tips[:, :2])
    center = np.asarray(payload["certified_center_xy"], dtype=np.float64)
    raw_radius = _centered_hull_radius(tips[:, :2], center)
    if raw_radius <= 0:
        raise RuntimeError("entry point lies outside filtered pose-map hull")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int64),
            kind=np.asarray("manisoft_table_arch_pose_map"),
            entry_index=payload["entry_index"],
            entry_name=payload["entry_name"],
            candidate_indices=indices,
            tip_positions=tips.astype(np.float32),
            physical_actions=actions.astype(np.float32),
            certified_center_xy=center.astype(np.float32),
            certified_radius=np.asarray(0.85 * raw_radius, dtype=np.float32),
            raw_hull_radius=np.asarray(raw_radius, dtype=np.float32),
            scenario_sha256=payload["scenario_sha256"],
            entry_bank_sha256=payload["entry_bank_sha256"],
            source_report_sha256=payload["source_report_sha256"],
        )
    temporary.replace(output)
    summary = {
        "kind": "manisoft_table_arch_pose_map_height_band_manifest",
        "map": str(output),
        "map_sha256": _sha256(output),
        "source_map": str(source),
        "source_map_sha256": _sha256(source),
        "target_z": target_z,
        "half_width": float(args.half_width),
        "source_samples": int(len(keep)),
        "kept_samples": int(len(tips)),
        "xy_hull_area": float(hull.volume),
        "raw_centered_radius": raw_radius,
        "tip_minimum": tips.min(axis=0).tolist(),
        "tip_maximum": tips.max(axis=0).tolist(),
    }
    output.with_name("manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
