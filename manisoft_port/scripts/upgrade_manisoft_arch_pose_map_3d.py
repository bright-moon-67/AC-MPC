#!/usr/bin/env python
"""Upgrade certified arch poses to XYZ-to-action tetrahedral interpolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if (
        int(payload["schema_version"].reshape(()).item()) not in {1, 2}
        or str(payload["kind"].reshape(()).item())
        != "manisoft_table_arch_pose_map"
    ):
        raise ValueError("unexpected source pose-map schema")
    tips = np.asarray(payload["tip_positions"], dtype=np.float64)
    actions = np.asarray(payload["physical_actions"], dtype=np.float64)
    if tips.ndim != 2 or tips.shape[1] != 3 or actions.shape != (len(tips), 18):
        raise ValueError("source pose map contains invalid arrays")
    if len(tips) < 4 or np.linalg.matrix_rank(tips[1:] - tips[0]) < 3:
        raise ValueError("source poses do not span a three-dimensional volume")

    triangulation = Delaunay(tips)
    tetrahedra = tips[triangulation.simplices]
    maximum_edges = np.max(
        np.stack(
            [
                np.linalg.norm(
                    tetrahedra[:, first] - tetrahedra[:, second], axis=1
                )
                for first in range(4)
                for second in range(first)
            ],
            axis=1,
        ),
        axis=1,
    )
    if triangulation.find_simplex(tips[0], tol=1e-8) < 0:
        raise ValueError("entry pose lies outside the three-dimensional hull")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload["schema_version"] = np.asarray(2, dtype=np.int64)
    payload["interpolation_dimensions"] = np.asarray(3, dtype=np.int64)
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(output)

    hull = ConvexHull(tips)
    summary = {
        "kind": "manisoft_table_arch_pose_map_3d_manifest",
        "map": str(output),
        "map_sha256": _sha256(output),
        "source_map": str(source),
        "source_map_sha256": _sha256(source),
        "interpolation_dimensions": 3,
        "sample_count": int(len(tips)),
        "tip_minimum": tips.min(axis=0).tolist(),
        "tip_maximum": tips.max(axis=0).tolist(),
        "tip_span": np.ptp(tips, axis=0).tolist(),
        "hull_volume_m3": float(hull.volume),
        "tetrahedron_count": int(len(triangulation.simplices)),
        "maximum_edge_quantiles_m": {
            str(quantile): float(np.quantile(maximum_edges, quantile))
            for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        },
    }
    output.with_name("manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
