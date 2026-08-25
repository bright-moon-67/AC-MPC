#!/usr/bin/env python
"""Build a long-hold-certified local XY-to-pose map for the vertical arch."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from antmaze_ac.envs.kinematic_push_task import segment_aabb_distance
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.table_entry_bank import (
    load_table_entry_trajectory_bank,
    restore_rod_internal_state,
)


_WORKER: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--entry-bank", required=True)
    parser.add_argument("--search-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-count", type=int, default=96)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ramp-steps", type=int, default=400)
    parser.add_argument("--hold-steps", type=int, default=500)
    parser.add_argument("--tail-steps", type=int, default=200)
    parser.add_argument("--maximum-angle-degrees", type=float, default=20.0)
    parser.add_argument(
        "--maximum-screening-hold-span",
        type=float,
        default=0.003,
        help=(
            "Maximum short-search hold span accepted before the independent "
            "long-hold certification. The final map still uses "
            "--maximum-hold-span."
        ),
    )
    parser.add_argument(
        "--minimum-screening-clearance",
        type=float,
        default=0.0,
        help=(
            "Minimum clearance from the source search report used only for "
            "candidate pre-screening. A negative value is useful when a "
            "lower or repositioned table will be independently re-certified."
        ),
    )
    parser.add_argument("--maximum-hold-span", type=float, default=0.001)
    parser.add_argument(
        "--minimum-clearance",
        type=float,
        default=0.0,
        help="Required whole-arm/table clearance for every retained pose.",
    )
    parser.add_argument("--minimum-tip-z", type=float, default=0.440)
    parser.add_argument("--maximum-tip-z", type=float, default=0.467)
    parser.add_argument("--selection-radius", type=float, default=0.14)
    parser.add_argument("--radius-safety-factor", type=float, default=0.85)
    parser.add_argument(
        "--interpolation-dimensions",
        type=int,
        choices=(2, 3),
        default=2,
        help="Use XY triangles (legacy) or XYZ tetrahedra for interpolation.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimum_jerk(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def _init_worker(
    scenario: str,
    entry_bank: str,
    ramp_steps: int,
    hold_steps: int,
    tail_steps: int,
) -> None:
    bank = load_table_entry_trajectory_bank(entry_bank)
    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=(0.0, 0.0, 0.45),
        absolute_action_limit=bank.absolute_action_limit,
    )
    env.reset(seed=20260891)
    _WORKER.update(
        bank=bank,
        env=env,
        ramp_steps=ramp_steps,
        hold_steps=hold_steps,
        tail_steps=tail_steps,
    )


def _restore_entry() -> tuple[Any, Any]:
    bank = _WORKER["bank"]
    env = _WORKER["env"]
    rod = env.sim._backend._softrobot
    rod.position_collection[...] = bank.node_positions[0, -1].T
    rod.velocity_collection[...] = bank.node_velocities[0, -1].T
    rod.director_collection[...] = bank.element_directors[0, -1].transpose(1, 2, 0)
    rod.omega_collection[...] = bank.element_omegas[0, -1].T
    restore_rod_internal_state(rod, bank.rod_internal_states[0, -1])
    env.muscle.set_activation(bank.actions[0, -1].reshape(6, 3))
    return bank, env


def _certify(job: tuple[int, np.ndarray]) -> dict[str, Any]:
    candidate_index, target_action = job
    bank, env = _restore_entry()
    start_action = np.asarray(bank.actions[0, -1], dtype=np.float64)
    table_minimum = np.asarray(
        [bank.table_x_bounds[0], bank.table_y_bounds[0], -2.0]
    )
    table_maximum = np.asarray(
        [bank.table_x_bounds[1], bank.table_y_bounds[1], bank.table_surface_z]
    )
    tips: list[np.ndarray] = []
    minimum_clearance = float("inf")
    maximum_angle = 0.0
    finite = True
    try:
        total = _WORKER["ramp_steps"] + _WORKER["hold_steps"]
        for step in range(total):
            blend = _minimum_jerk((step + 1) / _WORKER["ramp_steps"])
            action = start_action + blend * (target_action - start_action)
            env.muscle.set_activation(action.reshape(6, 3))
            env.sim.step_with_torque_callback(
                lambda lengths: env.muscle.evaluate(lengths)
            )
            nodes = np.asarray(
                env.sim._backend.softrobot_state.element_positions,
                dtype=np.float64,
            )
            if not np.isfinite(nodes).all():
                finite = False
                break
            tangent = nodes[-1] - nodes[-2]
            tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
            maximum_angle = max(
                maximum_angle,
                float(np.rad2deg(np.arccos(np.clip(-tangent[2], -1.0, 1.0)))),
            )
            minimum_clearance = min(
                minimum_clearance,
                min(
                    segment_aabb_distance(a, b, table_minimum, table_maximum)
                    for a, b in zip(nodes[:-1], nodes[1:])
                )
                - bank.arm_radius
                - bank.safety_margin,
            )
            tips.append(nodes[-1].copy())
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        finite = False
    if not finite or not tips:
        return {"candidate_index": candidate_index, "finite": False}
    tip_rows = np.asarray(tips)
    return {
        "candidate_index": candidate_index,
        "finite": True,
        "action": np.asarray(target_action, dtype=np.float32),
        "tip": tip_rows[-1].astype(np.float32),
        "maximum_angle_degrees": maximum_angle,
        "minimum_clearance": minimum_clearance,
        "hold_span": float(
            np.max(np.ptp(tip_rows[-_WORKER["tail_steps"] :], axis=0))
        ),
    }


def _farthest_subset(
    rows: list[dict[str, Any]], count: int, dimensions: int
) -> list[dict[str, Any]]:
    points = np.asarray(
        [row["final_tip"][:dimensions] for row in rows], dtype=np.float64
    )
    scale = np.ptp(points, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    normalized = (points - np.mean(points, axis=0)) / scale
    start = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    chosen = [start]
    nearest = np.linalg.norm(normalized - normalized[start], axis=1)
    while len(chosen) < min(count, len(rows)):
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(
            nearest, np.linalg.norm(normalized - normalized[index], axis=1)
        )
    return [rows[index] for index in chosen]


def _centered_hull_radius(points: np.ndarray, center: np.ndarray) -> float:
    hull = ConvexHull(points)
    # scipy equations use normal.x + offset <= 0 inside the hull.
    distances = -(
        hull.equations[:, :2] @ center + hull.equations[:, 2]
    ) / np.linalg.norm(hull.equations[:, :2], axis=1)
    return float(np.min(distances))


def main() -> None:
    args = parse_args()
    if min(
        args.candidate_count,
        args.workers,
        args.ramp_steps,
        args.hold_steps,
        args.tail_steps,
        args.maximum_angle_degrees,
        args.maximum_screening_hold_span,
        args.maximum_hold_span,
        args.selection_radius,
        args.radius_safety_factor,
    ) <= 0:
        raise ValueError("counts and certification limits must be positive")
    if args.minimum_clearance < 0:
        raise ValueError("minimum-clearance cannot be negative")
    if args.tail_steps > args.hold_steps:
        raise ValueError("tail-steps cannot exceed hold-steps")
    scenario = Path(args.scenario).expanduser().resolve()
    entry_path = Path(args.entry_bank).expanduser().resolve()
    report_path = Path(args.search_report).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    bank = load_table_entry_trajectory_bank(entry_path)
    if bank.trajectory_count != 1:
        raise ValueError("pose-map builder currently requires exactly one entry")
    center = np.asarray(bank.physical_states[0, -1, 30:32], dtype=np.float64)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = []
    for row in report["best_candidates"]:
        if not row.get("finite", False):
            continue
        tip = np.asarray(row["final_tip"], dtype=np.float64)
        action = np.asarray(row["action"], dtype=np.float64)
        if (
            args.minimum_tip_z <= tip[2] <= args.maximum_tip_z
            and row["tip_downward_angle_degrees"] <= args.maximum_angle_degrees
            and row["minimum_table_clearance"] >= args.minimum_screening_clearance
            and row["hold_tip_span"] <= args.maximum_screening_hold_span
            and np.linalg.norm(tip[:2] - center) <= args.selection_radius
            and np.max(np.abs(action)) <= bank.absolute_action_limit + 1e-6
        ):
            candidates.append(row)
    selected = _farthest_subset(
        candidates, args.candidate_count, args.interpolation_dimensions
    )
    jobs = [
        (int(row["candidate_index"]), np.asarray(row["action"], dtype=np.float64))
        for row in selected
    ]
    context = mp.get_context("spawn")
    with context.Pool(
        min(args.workers, len(jobs)),
        initializer=_init_worker,
        initargs=(
            str(scenario),
            str(entry_path),
            args.ramp_steps,
            args.hold_steps,
            args.tail_steps,
        ),
    ) as pool:
        results = pool.map(_certify, jobs)
    passed = [
        row
        for row in results
        if row.get("finite", False)
        and row["maximum_angle_degrees"] <= args.maximum_angle_degrees
        and row["minimum_clearance"] >= args.minimum_clearance
        and row["hold_span"] <= args.maximum_hold_span
        and args.minimum_tip_z <= row["tip"][2] <= args.maximum_tip_z
    ]
    entry_tip = np.asarray(bank.physical_states[0, -1, 30:33], dtype=np.float32)
    entry_action = np.asarray(bank.actions[0, -1], dtype=np.float32)
    tip_positions = np.vstack((entry_tip, *(row["tip"] for row in passed)))
    actions = np.vstack((entry_action, *(row["action"] for row in passed)))
    indices = np.asarray([-1, *(row["candidate_index"] for row in passed)], dtype=np.int64)
    clearances = np.asarray(
        [
            bank.safety_margin,
            *(row["minimum_clearance"] for row in passed),
        ],
        dtype=np.float32,
    )
    hold_spans = np.asarray(
        [0.0, *(row["hold_span"] for row in passed)], dtype=np.float32
    )
    # Preserve distinct vertical layers for XYZ interpolation while retaining
    # the legacy XY duplicate rule for old planar maps.
    kept = []
    for index, point in enumerate(tip_positions):
        if not kept or min(
            np.linalg.norm(
                point[: args.interpolation_dimensions]
                - tip_positions[other, : args.interpolation_dimensions]
            )
            for other in kept
        ) >= 0.001:
            kept.append(index)
    tip_positions = tip_positions[kept]
    actions = actions[kept]
    indices = indices[kept]
    clearances = clearances[kept]
    hold_spans = hold_spans[kept]
    if len(tip_positions) < args.interpolation_dimensions + 1:
        raise RuntimeError("too few long-hold-certified pose samples")
    raw_radius = _centered_hull_radius(tip_positions[:, :2], center)
    certified_radius = args.radius_safety_factor * raw_radius
    if certified_radius <= 0:
        raise RuntimeError("entry tip lies outside certified pose-map hull")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(
                2 if args.interpolation_dimensions == 3 else 1,
                dtype=np.int64,
            ),
            kind=np.asarray("manisoft_table_arch_pose_map"),
            interpolation_dimensions=np.asarray(
                args.interpolation_dimensions, dtype=np.int64
            ),
            entry_index=np.asarray(0, dtype=np.int64),
            entry_name=np.asarray(bank.names[0]),
            candidate_indices=indices,
            minimum_table_clearances=clearances,
            hold_tip_spans=hold_spans,
            tip_positions=tip_positions.astype(np.float32),
            physical_actions=actions.astype(np.float32),
            certified_center_xy=center.astype(np.float32),
            certified_radius=np.asarray(certified_radius, dtype=np.float32),
            raw_hull_radius=np.asarray(raw_radius, dtype=np.float32),
            scenario_sha256=np.asarray(_sha256(scenario)),
            entry_bank_sha256=np.asarray(_sha256(entry_path)),
            source_report_sha256=np.asarray(_sha256(report_path)),
        )
    temporary.replace(output)
    summary = {
        "kind": "manisoft_table_arch_pose_map_manifest",
        "map": str(output),
        "map_sha256": _sha256(output),
        "selected_candidates": len(selected),
        "long_hold_passes": len(passed),
        "unique_samples": len(tip_positions),
        "interpolation_dimensions": args.interpolation_dimensions,
        "certified_center_xy": center.tolist(),
        "raw_hull_radius": raw_radius,
        "certified_radius": certified_radius,
        "tip_minimum": tip_positions.min(axis=0).tolist(),
        "tip_maximum": tip_positions.max(axis=0).tolist(),
        "maximum_hold_span": max((row["hold_span"] for row in passed), default=0.0),
        "maximum_screening_hold_span": args.maximum_screening_hold_span,
        "minimum_screening_clearance": args.minimum_screening_clearance,
        "minimum_clearance": min((row["minimum_clearance"] for row in passed), default=0.0),
        "required_minimum_clearance": args.minimum_clearance,
        "maximum_angle_degrees": max((row["maximum_angle_degrees"] for row in passed), default=0.0),
    }
    output.with_name("manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
