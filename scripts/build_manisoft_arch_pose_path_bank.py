#!/usr/bin/env python
"""Build and dynamically certify a non-collinear arch-pose path bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from antmaze_ac.envs.kinematic_push_task import segment_aabb_distance
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.table_entry_bank import (
    load_table_entry_trajectory_bank,
    restore_rod_internal_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--entry-bank", required=True)
    parser.add_argument("--search-report", required=True)
    parser.add_argument("--candidate-indices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ramp-steps", type=int, default=400)
    parser.add_argument("--hold-steps", type=int, default=600)
    parser.add_argument("--stability-tail-steps", type=int, default=200)
    parser.add_argument("--action-limit", type=float, default=0.60)
    parser.add_argument("--maximum-angle-degrees", type=float, default=15.0)
    parser.add_argument("--maximum-cross-track", type=float, default=0.030)
    parser.add_argument("--maximum-hold-span", type=float, default=0.001)
    parser.add_argument("--maximum-z-range", type=float, default=0.025)
    parser.add_argument("--video-stride", type=int, default=5)
    return parser.parse_args()


def _minimum_jerk(fraction: float) -> float:
    value = float(np.clip(fraction, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _turn_angles(points: np.ndarray) -> np.ndarray:
    vectors = np.diff(points[:, :2], axis=0)
    cosine = np.sum(vectors[:-1] * vectors[1:], axis=1) / (
        np.linalg.norm(vectors[:-1], axis=1)
        * np.linalg.norm(vectors[1:], axis=1)
    )
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def main() -> None:
    args = parse_args()
    if min(
        args.ramp_steps,
        args.hold_steps,
        args.stability_tail_steps,
        args.action_limit,
        args.maximum_angle_degrees,
        args.maximum_cross_track,
        args.maximum_hold_span,
        args.maximum_z_range,
        args.video_stride,
    ) <= 0:
        raise ValueError("all limits and rollout lengths must be positive")
    if args.stability_tail_steps > args.hold_steps:
        raise ValueError("stability-tail-steps cannot exceed hold-steps")
    scenario = Path(args.scenario).expanduser().resolve()
    entry_path = Path(args.entry_bank).expanduser().resolve()
    report_path = Path(args.search_report).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    candidate_indices = [
        int(value) for value in args.candidate_indices.split(",")
    ]
    if not 3 <= len(candidate_indices) <= 5:
        raise ValueError("candidate-indices must select three to five points")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates: dict[int, dict[str, Any]] = {
        int(row["candidate_index"]): row
        for row in report["best_candidates"]
    }
    missing = [value for value in candidate_indices if value not in candidates]
    if missing:
        raise ValueError(f"candidate indices absent from report: {missing}")
    actions = np.asarray(
        [candidates[value]["action"] for value in candidate_indices],
        dtype=np.float32,
    )
    bank = load_table_entry_trajectory_bank(entry_path)
    if bank.trajectory_count != 1:
        raise ValueError("arch path builder currently requires one entry")
    if np.max(np.abs(actions[0] - bank.actions[0, -1])) > 2e-6:
        raise ValueError("first pose action does not match the entry endpoint")

    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=(0.0, 0.0, 0.45),
        absolute_action_limit=args.action_limit,
    )
    env.reset(seed=20260839)
    rod = env.sim._backend._softrobot
    rod.position_collection[...] = bank.node_positions[0, -1].T
    rod.velocity_collection[...] = bank.node_velocities[0, -1].T
    rod.director_collection[...] = bank.element_directors[
        0, -1
    ].transpose(1, 2, 0)
    rod.omega_collection[...] = bank.element_omegas[0, -1].T
    restore_rod_internal_state(rod, bank.rod_internal_states[0, -1])
    env.muscle.set_activation(actions[0].reshape(6, 3))

    table_minimum = np.asarray(
        [bank.table_x_bounds[0], bank.table_y_bounds[0], -2.0]
    )
    table_maximum = np.asarray(
        [bank.table_x_bounds[1], bank.table_y_bounds[1], bank.table_surface_z]
    )

    def metrics() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, float, float
    ]:
        soft_state = env.sim._backend.softrobot_state
        nodes = np.asarray(
            soft_state.element_positions,
            dtype=np.float64,
        )
        directors = np.asarray(
            soft_state.element_directors, dtype=np.float64
        )
        tangent = nodes[-1] - nodes[-2]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        angle = float(
            np.rad2deg(np.arccos(np.clip(-tangent[2], -1.0, 1.0)))
        )
        clearance = float(
            min(
                segment_aabb_distance(start, end, table_minimum, table_maximum)
                for start, end in zip(nodes[:-1], nodes[1:])
            )
            - bank.arm_radius
            - bank.safety_margin
        )
        return (
            nodes[-1].copy(),
            nodes.copy(),
            directors.copy(),
            angle,
            clearance,
        )

    anchors = [metrics()[0]]
    transition_tips: list[np.ndarray] = []
    transition_nodes: list[np.ndarray] = []
    transition_directors: list[np.ndarray] = []
    transition_angles: list[np.ndarray] = []
    segment_reports: list[dict[str, Any]] = []
    for segment_index, (start_action, end_action) in enumerate(
        zip(actions[:-1], actions[1:])
    ):
        start_tip = metrics()[0]
        tip_rows: list[np.ndarray] = []
        node_rows: list[np.ndarray] = []
        director_rows: list[np.ndarray] = []
        angle_rows: list[float] = []
        clearance_rows: list[float] = []
        for step in range(args.ramp_steps + args.hold_steps):
            fraction = _minimum_jerk((step + 1) / args.ramp_steps)
            action = start_action + fraction * (end_action - start_action)
            env.muscle.set_activation(action.reshape(6, 3))
            env.sim.step_with_torque_callback(
                lambda lengths: env.muscle.evaluate(lengths)
            )
            tip, nodes, directors, angle, clearance = metrics()
            tip_rows.append(tip)
            node_rows.append(nodes)
            director_rows.append(directors)
            angle_rows.append(angle)
            clearance_rows.append(clearance)
        tips = np.asarray(tip_rows)
        nodes = np.asarray(node_rows)
        directors = np.asarray(director_rows)
        angles = np.asarray(angle_rows)
        end_tip = tips[-1]
        chord = end_tip[:2] - start_tip[:2]
        chord_squared = max(float(np.dot(chord, chord)), 1e-12)
        fractions = np.clip(
            ((tips[:, :2] - start_tip[:2]) @ chord) / chord_squared,
            0.0,
            1.0,
        )
        projected = start_tip[:2] + fractions[:, None] * chord
        cross_track = np.linalg.norm(tips[:, :2] - projected, axis=1)
        row = {
            "segment_index": segment_index,
            "start_tip": start_tip.tolist(),
            "end_tip": end_tip.tolist(),
            "horizontal_length": float(np.linalg.norm(chord)),
            "maximum_cross_track": float(
                np.max(cross_track[: args.ramp_steps])
            ),
            "rms_cross_track": float(
                np.sqrt(np.mean(np.square(cross_track[: args.ramp_steps])))
            ),
            "maximum_orientation_error_degrees": float(np.max(angles)),
            "minimum_table_clearance": float(np.min(clearance_rows)),
            "hold_tip_span": float(
                np.max(
                    np.ptp(tips[-args.stability_tail_steps :], axis=0)
                )
            ),
        }
        segment_reports.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        anchors.append(end_tip.copy())
        transition_tips.append(tips)
        transition_nodes.append(nodes)
        transition_directors.append(directors)
        transition_angles.append(angles)
    env.close()

    anchors_array = np.asarray(anchors)
    lengths = np.linalg.norm(np.diff(anchors_array[:, :2], axis=0), axis=1)
    turns = _turn_angles(anchors_array)
    checks = {
        "segment_lengths": bool(
            np.all((lengths >= 0.07) & (lengths <= 0.13))
            and 0.085 <= float(np.mean(lengths)) <= 0.11
        ),
        "turn_angles": bool(np.all((turns >= 60.0) & (turns <= 120.0))),
        "near_planar": float(np.ptp(anchors_array[:, 2]))
        <= args.maximum_z_range,
        "straight_transitions": max(
            row["maximum_cross_track"] for row in segment_reports
        )
        <= args.maximum_cross_track,
        "orientation": max(
            row["maximum_orientation_error_degrees"]
            for row in segment_reports
        )
        <= args.maximum_angle_degrees,
        "table_clearance": min(
            row["minimum_table_clearance"] for row in segment_reports
        )
        >= 0.0,
        "settled": max(row["hold_tip_span"] for row in segment_reports)
        <= args.maximum_hold_span,
    }
    summary = {
        "point_count": len(anchors_array),
        "segment_lengths": lengths.tolist(),
        "mean_segment_length": float(np.mean(lengths)),
        "turn_angles_degrees": turns.tolist(),
        "z_range": float(np.ptp(anchors_array[:, 2])),
        "checks": checks,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"arch pose path failed certification: {failed}")

    video_nodes = np.concatenate(
        (
            bank.node_positions[0, :: args.video_stride],
            *(
                rows[:: args.video_stride]
                for rows in transition_nodes
            ),
        ),
        axis=0,
    )
    video_directors = np.concatenate(
        (
            bank.element_directors[0, :: args.video_stride],
            *(
                rows[:: args.video_stride]
                for rows in transition_directors
            ),
        ),
        axis=0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int64),
            kind=np.asarray("manisoft_table_arch_pose_path_bank"),
            entry_index=np.asarray(0, dtype=np.int64),
            entry_name=np.asarray(bank.names[0]),
            candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
            tip_positions=anchors_array.astype(np.float32),
            physical_actions=actions.astype(np.float32),
            segment_lengths=lengths.astype(np.float32),
            turn_angles_degrees=turns.astype(np.float32),
            transition_tip_positions=np.asarray(
                transition_tips, dtype=np.float32
            ),
            transition_node_positions=np.asarray(
                transition_nodes, dtype=np.float32
            ),
            transition_element_directors=np.asarray(
                transition_directors, dtype=np.float32
            ),
            transition_orientation_errors_degrees=np.asarray(
                transition_angles, dtype=np.float32
            ),
            video_node_positions=video_nodes.astype(np.float32),
            video_element_directors=video_directors.astype(np.float32),
            video_stride=np.asarray(args.video_stride, dtype=np.int64),
            control_dt=np.asarray(bank.control_dt, dtype=np.float64),
            scenario_sha256=np.asarray(_sha256(scenario)),
            entry_bank_sha256=np.asarray(_sha256(entry_path)),
            source_report_sha256=np.asarray(_sha256(report_path)),
        )
    temporary.replace(output)
    manifest = {
        "kind": "manisoft_table_arch_pose_path_bank_manifest",
        "bank": str(output),
        "bank_sha256": _sha256(output),
        "scenario": str(scenario),
        "entry_bank": str(entry_path),
        "search_report": str(report_path),
        "candidate_indices": candidate_indices,
        "segments": segment_reports,
        **summary,
    }
    output.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
