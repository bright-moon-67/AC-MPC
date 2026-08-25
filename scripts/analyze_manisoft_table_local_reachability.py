#!/usr/bin/env python
"""Probe stable near-planar motions around every certified table-entry pose."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

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
    parser.add_argument(
        "--bank",
        default="data/processed/manisoft_table_entry_bank_v1/entry_bank.npz",
    )
    parser.add_argument(
        "--output",
        default="runs/manisoft_waypoint_sac_physical_smoke/local_reachability.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--transition-steps", type=int, default=100)
    parser.add_argument("--hold-steps", type=int, default=50)
    parser.add_argument(
        "--rotations",
        default="-4,-2,0,2,4",
        help="Comma-separated activation-plane rotations in degrees.",
    )
    parser.add_argument(
        "--scales",
        default="0.97,1.0,1.03",
        help="Comma-separated activation magnitude scales.",
    )
    parser.add_argument(
        "--entry-indices",
        default=None,
        help="Optional comma-separated entry indices; the default probes all.",
    )
    parser.add_argument("--workspace-low", default="-0.30,0.50,0.445")
    parser.add_argument("--workspace-high", default="0.30,0.80,0.54")
    parser.add_argument("--max-reach", type=float, default=0.91)
    parser.add_argument("--maximum-hold-tip-span", type=float, default=0.001)
    parser.add_argument(
        "--action-limit",
        type=float,
        default=0.30,
        help="Absolute SplineMuscle activation bound used by this probe.",
    )
    parser.add_argument(
        "--table-surface-z",
        type=float,
        default=None,
        help="Override the bank table height for geometric clearance checks.",
    )
    return parser.parse_args()


def _minimum_jerk(value: float) -> float:
    fraction = float(np.clip(value, 0.0, 1.0))
    return fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)


def _rotate_and_scale(
    action: np.ndarray,
    degrees: float,
    scale: float,
    action_limit: float,
) -> np.ndarray:
    values = np.asarray(action, dtype=np.float64).reshape(6, 3).copy()
    angle = np.deg2rad(degrees)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    values[:, :2] = scale * values[:, :2] @ rotation.T
    return np.clip(
        values.reshape(-1), -action_limit, action_limit
    ).astype(np.float32)


def _clearance(nodes: np.ndarray, bank, table_surface_z: float) -> float:
    minimum = np.asarray(
        [bank.table_x_bounds[0], bank.table_y_bounds[0], -2.0]
    )
    maximum = np.asarray(
        [bank.table_x_bounds[1], bank.table_y_bounds[1], table_surface_z]
    )
    return float(
        min(
            segment_aabb_distance(start, end, minimum, maximum)
            for start, end in zip(nodes[:-1], nodes[1:])
        )
        - bank.arm_radius
        - bank.safety_margin
    )


def _probe(
    arguments: tuple[
        str,
        str,
        int,
        float,
        float,
        int,
        int,
        tuple[float, float, float],
        tuple[float, float, float],
        float,
        float,
        float,
        float,
    ]
) -> dict:
    (
        scenario,
        bank_path,
        entry_index,
        degrees,
        scale,
        transition_steps,
        hold_steps,
        workspace_low,
        workspace_high,
        max_reach,
        maximum_hold_tip_span,
        action_limit,
        table_surface_z,
    ) = arguments
    bank = load_table_entry_trajectory_bank(bank_path)
    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=(0.0, 0.0, 0.5),
        absolute_action_limit=action_limit,
    )
    env.reset(seed=entry_index)
    rod = env.sim._backend._softrobot
    rod.position_collection[...] = bank.node_positions[entry_index, -1].T
    rod.velocity_collection[...] = bank.node_velocities[entry_index, -1].T
    rod.director_collection[...] = bank.element_directors[entry_index, -1].transpose(
        1, 2, 0
    )
    rod.omega_collection[...] = bank.element_omegas[entry_index, -1].T
    restore_rod_internal_state(rod, bank.rod_internal_states[entry_index, -1])
    env.sim._backend.time_tracker += bank.transition_count * bank.control_dt
    env.sim.current_step += bank.transition_count * int(
        round(bank.control_dt / env.sim._backend.dt)
    )
    start_action = bank.actions[entry_index, -1]
    env.muscle.set_activation(start_action.reshape(6, 3))
    target_action = _rotate_and_scale(
        start_action, degrees, scale, action_limit
    )
    start_tip = bank.tip_positions[entry_index, -1].copy()
    tips = []
    minimum_clearance = float("inf")
    maximum_action_delta = 0.0
    previous_action = start_action
    for step in range(transition_steps + hold_steps):
        blend = _minimum_jerk((step + 1) / transition_steps)
        action = start_action + blend * (target_action - start_action)
        maximum_action_delta = max(
            maximum_action_delta, float(np.max(np.abs(action - previous_action)))
        )
        previous_action = action
        env.muscle.set_activation(action.reshape(6, 3))
        env.sim.step_with_torque_callback(lambda lengths: env.muscle.evaluate(lengths))
        nodes = np.asarray(
            env.sim._backend.softrobot_state.element_positions, dtype=np.float64
        )
        tips.append(nodes[-1])
        minimum_clearance = min(
            minimum_clearance,
            _clearance(nodes, bank, table_surface_z),
        )
    env.close()
    tips_array = np.asarray(tips)
    final_tip = tips_array[-1]
    tip_tangent = nodes[-1] - nodes[-2]
    tip_tangent /= max(float(np.linalg.norm(tip_tangent)), 1e-12)
    downward_angle_degrees = float(
        np.rad2deg(
            np.arccos(np.clip(np.dot(tip_tangent, (0.0, 0.0, -1.0)), -1.0, 1.0))
        )
    )
    arch_height_above_tip = float(np.max(nodes[:, 2]) - final_tip[2])
    arch_peak_index = int(np.argmax(nodes[:, 2]))
    displacement = final_tip - start_tip
    hold_span = float(np.max(np.ptp(tips_array[-hold_steps:], axis=0)))
    endpoint_ok = bool(
        np.all(final_tip >= np.asarray(workspace_low))
        and np.all(final_tip <= np.asarray(workspace_high))
        and np.linalg.norm(final_tip) <= max_reach
    )
    return {
        "entry_index": entry_index,
        "entry_name": bank.names[entry_index],
        "rotation_degrees": degrees,
        "activation_scale": scale,
        "action_limit": action_limit,
        "table_surface_z": table_surface_z,
        "start_tip": start_tip.tolist(),
        "final_tip": final_tip.tolist(),
        "displacement": displacement.tolist(),
        "horizontal_displacement": float(np.linalg.norm(displacement[:2])),
        "vertical_displacement": float(displacement[2]),
        "tip_tangent": tip_tangent.tolist(),
        "tip_downward_angle_degrees": downward_angle_degrees,
        "arch_height_above_tip": arch_height_above_tip,
        "arch_peak_node_index": arch_peak_index,
        "minimum_table_clearance": minimum_clearance,
        "maximum_action_delta": maximum_action_delta,
        "hold_tip_span": hold_span,
        "passed": bool(
            endpoint_ok
            and minimum_clearance > 0
            and maximum_action_delta <= 0.015
            and hold_span <= maximum_hold_tip_span
        ),
    }


def main() -> None:
    args = parse_args()
    rotations = tuple(
        float(value) for value in args.rotations.split(",") if value.strip()
    )
    scales = tuple(float(value) for value in args.scales.split(",") if value.strip())
    workspace_low = tuple(
        float(value) for value in args.workspace_low.split(",") if value.strip()
    )
    workspace_high = tuple(
        float(value) for value in args.workspace_high.split(",") if value.strip()
    )
    if not rotations or not scales or min(scales) <= 0:
        raise ValueError("rotations must be nonempty and scales must be positive")
    if (
        len(workspace_low) != 3
        or len(workspace_high) != 3
        or np.any(np.asarray(workspace_low) >= np.asarray(workspace_high))
        or min(
            args.max_reach,
            args.maximum_hold_tip_span,
            args.action_limit,
        ) <= 0
    ):
        raise ValueError("workspace and reach/stability criteria are invalid")
    if min(args.workers, args.transition_steps, args.hold_steps) < 1:
        raise ValueError("workers and rollout step counts must be positive")
    scenario = str(Path(args.scenario).expanduser().resolve())
    bank_path = str(Path(args.bank).expanduser().resolve())
    bank = load_table_entry_trajectory_bank(bank_path)
    table_surface_z = (
        bank.table_surface_z
        if args.table_surface_z is None
        else float(args.table_surface_z)
    )
    entry_indices = (
        tuple(
            int(value)
            for value in args.entry_indices.split(",")
            if value.strip()
        )
        if args.entry_indices is not None
        else tuple(range(bank.trajectory_count))
    )
    if not entry_indices or any(
        index < 0 or index >= bank.trajectory_count for index in entry_indices
    ):
        raise ValueError("entry indices are empty or out of range")
    jobs = [
        (
            scenario,
            bank_path,
            entry_index,
            rotation,
            scale,
            args.transition_steps,
            args.hold_steps,
            workspace_low,
            workspace_high,
            args.max_reach,
            args.maximum_hold_tip_span,
            args.action_limit,
            table_surface_z,
        )
        for entry_index in entry_indices
        for rotation in rotations
        for scale in scales
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_probe, jobs))
    summaries = []
    for entry_index in entry_indices:
        name = bank.names[entry_index]
        selected = [row for row in rows if row["entry_index"] == entry_index]
        passed = [row for row in selected if row["passed"]]
        displacements = np.asarray([row["displacement"] for row in passed])
        summaries.append(
            {
                "entry_index": entry_index,
                "entry_name": name,
                "passed": len(passed),
                "total": len(selected),
                "displacement_min": displacements.min(axis=0).tolist()
                if len(passed)
                else None,
                "displacement_max": displacements.max(axis=0).tolist()
                if len(passed)
                else None,
                "maximum_horizontal_displacement": float(
                    max((row["horizontal_displacement"] for row in passed), default=0)
                ),
                "minimum_clearance": float(
                    min((row["minimum_table_clearance"] for row in passed), default=-np.inf)
                ),
                "minimum_tip_downward_angle_degrees": float(
                    min(
                        (
                            row["tip_downward_angle_degrees"]
                            for row in selected
                        ),
                        default=np.inf,
                    )
                ),
                "maximum_arch_height_above_tip": float(
                    max(
                        (row["arch_height_above_tip"] for row in selected),
                        default=0.0,
                    )
                ),
            }
        )
    report = {
        "kind": "manisoft_table_local_reachability",
        "scenario": scenario,
        "bank": bank_path,
        "transition_steps": args.transition_steps,
        "hold_steps": args.hold_steps,
        "rotations_degrees": list(rotations),
        "activation_scales": list(scales),
        "entry_indices": list(entry_indices),
        "workspace_low": list(workspace_low),
        "workspace_high": list(workspace_high),
        "max_reach": args.max_reach,
        "maximum_hold_tip_span": args.maximum_hold_tip_span,
        "action_limit": args.action_limit,
        "table_surface_z": table_surface_z,
        "passed": int(sum(row["passed"] for row in rows)),
        "total": len(rows),
        "summaries": summaries,
        "probes": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("kind", "passed", "total", "summaries")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
