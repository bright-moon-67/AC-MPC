#!/usr/bin/env python
"""Propose locally regressed actions that close large pose-map simplices.

The output deliberately uses the search-report schema consumed by
``build_manisoft_arch_pose_map.py``.  Predictions are only proposals: the map
builder must still replay every action in ManiSoft and apply its long-hold,
orientation, height, and table-clearance certification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-simplex-edge", type=float, default=0.04)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--weight-floor", type=float, default=0.03)
    parser.add_argument("--minimum-query-separation", type=float, default=0.003)
    parser.add_argument("--absolute-action-limit", type=float, default=0.60)
    parser.add_argument(
        "--interpolation-dimensions",
        type=int,
        choices=(2, 3),
        help="Override the source map interpolation dimensionality.",
    )
    return parser.parse_args()


def _maximum_edge(points: np.ndarray) -> float:
    return max(
        float(np.linalg.norm(points[first] - points[second]))
        for first in range(len(points))
        for second in range(first)
    )


def _local_affine_action(
    query: np.ndarray,
    normalized_points: np.ndarray,
    actions: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    neighbors: int,
    ridge: float,
    weight_floor: float,
) -> np.ndarray:
    normalized_query = (query - center) / scale
    distances = np.linalg.norm(normalized_points - normalized_query, axis=1)
    indices = np.argsort(distances)[: min(neighbors, len(distances))]
    weights = 1.0 / np.maximum(distances[indices], weight_floor) ** 2
    design = np.column_stack(
        (normalized_points[indices], np.ones(len(indices), dtype=np.float64))
    )
    weighted_design = design * np.sqrt(weights[:, None])
    weighted_actions = actions[indices] * np.sqrt(weights[:, None])
    regularizer = np.diag(
        [ridge] * normalized_points.shape[1]
        + [max(ridge * 1e-3, 1e-12)]
    )
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + regularizer,
        weighted_design.T @ weighted_actions,
    )
    return np.append(normalized_query, 1.0) @ coefficients


def _report_row(
    index: int,
    action: np.ndarray,
    tip: np.ndarray,
    *,
    proposal_kind: str,
    source_simplex_maximum_edge: float,
    unclipped_action_maximum: float,
) -> dict:
    # The geometric fields below only pass the builder's cheap screening.
    # Its independent simulation subsequently replaces them with measured
    # values and is the sole source of certification.
    return {
        "candidate_index": int(index),
        "finite": True,
        "action": np.asarray(action, dtype=np.float64).tolist(),
        "final_tip": np.asarray(tip, dtype=np.float64).tolist(),
        "tip_downward_angle_degrees": 0.0,
        "minimum_table_clearance": 1.0,
        "hold_tip_span": 0.0,
        "proposal_kind": proposal_kind,
        "source_simplex_maximum_edge": float(source_simplex_maximum_edge),
        "unclipped_action_maximum": float(unclipped_action_maximum),
    }


def main() -> None:
    args = parse_args()
    if args.maximum_simplex_edge <= 0:
        raise ValueError("maximum-simplex-edge must be positive")
    if args.neighbors < 3:
        raise ValueError("neighbors must be at least three")
    if args.ridge <= 0 or args.weight_floor <= 0:
        raise ValueError("ridge and weight-floor must be positive")
    if args.minimum_query_separation <= 0 or args.absolute_action_limit <= 0:
        raise ValueError("separation and action limit must be positive")

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        if (
            int(np.asarray(data["schema_version"]).reshape(()).item())
            not in {1, 2}
            or str(np.asarray(data["kind"]).reshape(()).item())
            != "manisoft_table_arch_pose_map"
        ):
            raise ValueError("unexpected source pose-map schema")
        tips = np.asarray(data["tip_positions"], dtype=np.float64)
        actions = np.asarray(data["physical_actions"], dtype=np.float64)
        stored_dimensions = int(
            np.asarray(
                data["interpolation_dimensions"]
                if "interpolation_dimensions" in data.files
                else 2
            )
            .reshape(())
            .item()
        )
    if tips.ndim != 2 or tips.shape[1] != 3 or actions.shape != (len(tips), 18):
        raise ValueError("source pose map contains invalid arrays")
    if len(tips) < args.neighbors:
        raise ValueError("source pose map has fewer samples than requested neighbors")

    dimensions = (
        stored_dimensions
        if args.interpolation_dimensions is None
        else int(args.interpolation_dimensions)
    )
    if dimensions not in {2, 3}:
        raise ValueError("interpolation dimensions must be two or three")
    coordinates = tips[:, :dimensions]
    triangulation = Delaunay(coordinates)
    center = np.mean(coordinates, axis=0)
    scale = np.std(coordinates, axis=0)
    if np.any(scale <= 0):
        raise ValueError("pose-map interpolation samples have zero spread")
    normalized_points = (coordinates - center) / scale

    queries: list[tuple[np.ndarray, float, str]] = []
    bad_simplex_count = 0
    for vertices in triangulation.simplices:
        points = coordinates[vertices]
        maximum_edge = _maximum_edge(points)
        if maximum_edge <= args.maximum_simplex_edge:
            continue
        bad_simplex_count += 1
        centroid = np.mean(points, axis=0)
        edge_pairs = [
            (first, second)
            for first in range(dimensions + 1)
            for second in range(first)
        ]
        first, second = max(
            edge_pairs,
            key=lambda pair: np.linalg.norm(points[pair[0]] - points[pair[1]]),
        )
        midpoint = 0.5 * (points[first] + points[second])
        queries.extend(
            (
                (centroid, maximum_edge, "bad_simplex_centroid"),
                (midpoint, maximum_edge, "bad_simplex_longest_edge_midpoint"),
            )
        )

    accepted_queries: list[tuple[np.ndarray, float, str]] = []
    occupied = [row.copy() for row in coordinates]
    for query, maximum_edge, kind in sorted(
        queries, key=lambda row: row[1], reverse=True
    ):
        if min(np.linalg.norm(query - point) for point in occupied) < (
            args.minimum_query_separation
        ):
            continue
        accepted_queries.append((query, maximum_edge, kind))
        occupied.append(query.copy())

    rows = [
        _report_row(
            index,
            action,
            tip,
            proposal_kind="source_certified",
            source_simplex_maximum_edge=0.0,
            unclipped_action_maximum=float(np.max(np.abs(action))),
        )
        for index, (tip, action) in enumerate(zip(tips, actions))
    ]
    target_z = float(tips[0, 2])
    clipped_proposal_count = 0
    for query_index, (query, maximum_edge, kind) in enumerate(
        accepted_queries, start=len(rows)
    ):
        predicted = _local_affine_action(
            query,
            normalized_points,
            actions,
            center=center,
            scale=scale,
            neighbors=args.neighbors,
            ridge=args.ridge,
            weight_floor=args.weight_floor,
        )
        unclipped_maximum = float(np.max(np.abs(predicted)))
        clipped = np.clip(
            predicted, -args.absolute_action_limit, args.absolute_action_limit
        )
        clipped_proposal_count += int(not np.allclose(clipped, predicted))
        rows.append(
            _report_row(
                query_index,
                clipped,
                (
                    query
                    if dimensions == 3
                    else np.asarray([query[0], query[1], target_z])
                ),
                proposal_kind=kind,
                source_simplex_maximum_edge=maximum_edge,
                unclipped_action_maximum=unclipped_maximum,
            )
        )

    payload = {
        "kind": "manisoft_pose_map_gap_closure_proposals",
        "source_map": str(source),
        "source_sample_count": int(len(tips)),
        "bad_simplex_count": int(bad_simplex_count),
        "new_proposal_count": int(len(accepted_queries)),
        "clipped_proposal_count": int(clipped_proposal_count),
        "maximum_simplex_edge": float(args.maximum_simplex_edge),
        "neighbors": int(args.neighbors),
        "ridge": float(args.ridge),
        "weight_floor": float(args.weight_floor),
        "minimum_query_separation": float(args.minimum_query_separation),
        "interpolation_dimensions": dimensions,
        "best_candidates": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "best_candidates"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
