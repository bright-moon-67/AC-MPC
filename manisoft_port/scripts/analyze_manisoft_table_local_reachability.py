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
from antmaze_ac.envs.table_entry_bank import load_table_entry_trajectory_bank


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
    return parser.parse_args()


def _minimum_jerk(value: float) -> float:
    fraction = float(np.clip(value, 0.0, 1.0))
    return fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)


def _rotate_and_scale(action: np.ndarray, degrees: float, scale: float) -> np.ndarray:
    values = np.asarray(action, dtype=np.float64).reshape(6, 3).copy()
    angle = np.deg2rad(degrees)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    values[:, :2] = scale * values[:, :2] @ rotation.T
    return np.clip(values.reshape(-1), -0.30, 0.30).astype(np.float32)


def _clearance(nodes: np.ndarray, bank) -> float:
    minimum = np.asarray(
        [bank.table_x_bounds[0], bank.table_y_bounds[0], -2.0]
    )
    maximum = np.asarray(
        [bank.table_x_bounds[1], bank.table_y_bounds[1], bank.table_surface_z]
    )
    return float(
        min(
            segment_aabb_distance(start, end, minimum, maximum)
            for start, end in zip(nodes[:-1], nodes[1:])
        )
        - bank.arm_radius
        - bank.safety_margin
    )


def _probe(arguments: tuple[str, str, int, float, float, int, int]) -> dict:
    scenario, bank_path, entry_index, degrees, scale, transition_steps, hold_steps = (
        arguments
    )
    bank = load_table_entry_trajectory_bank(bank_path)
    env = ManiSoftTipTrackingEnv(
        scenario, target_tip=(0.0, 0.0, 0.5), absolute_action_limit=0.30
    )
    env.reset(seed=entry_index)
    rod = env.sim._backend._softrobot
    rod.position_collection[...] = bank.node_positions[entry_index, -1].T
    rod.velocity_collection[...] = bank.node_velocities[entry_index, -1].T
    rod.director_collection[...] = bank.element_directors[entry_index, -1].transpose(
        1, 2, 0
    )
    rod.omega_collection[...] = bank.element_omegas[entry_index, -1].T
    start_action = bank.actions[entry_index, -1]
    target_action = _rotate_and_scale(start_action, degrees, scale)
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
        minimum_clearance = min(minimum_clearance, _clearance(nodes, bank))
    env.close()
    tips_array = np.asarray(tips)
    final_tip = tips_array[-1]
    displacement = final_tip - start_tip
    hold_span = float(np.max(np.ptp(tips_array[-hold_steps:], axis=0)))
    endpoint_ok = bool(
        -0.30 <= final_tip[0] <= 0.30
        and bank.table_y_bounds[0] + 0.08 <= final_tip[1]
        <= bank.table_y_bounds[1] - 0.06
        and bank.table_surface_z + bank.arm_radius + bank.safety_margin + 0.03
        <= final_tip[2]
        <= 0.54
        and np.linalg.norm(final_tip) <= 0.91
    )
    return {
        "entry_index": entry_index,
        "entry_name": bank.names[entry_index],
        "rotation_degrees": degrees,
        "activation_scale": scale,
        "start_tip": start_tip.tolist(),
        "final_tip": final_tip.tolist(),
        "displacement": displacement.tolist(),
        "horizontal_displacement": float(np.linalg.norm(displacement[:2])),
        "vertical_displacement": float(displacement[2]),
        "minimum_table_clearance": minimum_clearance,
        "maximum_action_delta": maximum_action_delta,
        "hold_tip_span": hold_span,
        "passed": bool(
            endpoint_ok
            and minimum_clearance > 0
            and maximum_action_delta <= 0.015
            and hold_span <= 0.001
        ),
    }


def main() -> None:
    args = parse_args()
    scenario = str(Path(args.scenario).expanduser().resolve())
    bank_path = str(Path(args.bank).expanduser().resolve())
    bank = load_table_entry_trajectory_bank(bank_path)
    rotations = (-4.0, -2.0, 0.0, 2.0, 4.0)
    scales = (0.97, 1.0, 1.03)
    jobs = [
        (
            scenario,
            bank_path,
            entry_index,
            rotation,
            scale,
            args.transition_steps,
            args.hold_steps,
        )
        for entry_index in range(bank.trajectory_count)
        for rotation in rotations
        for scale in scales
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_probe, jobs))
    summaries = []
    for entry_index, name in enumerate(bank.names):
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
            }
        )
    report = {
        "kind": "manisoft_table_local_reachability",
        "scenario": scenario,
        "bank": bank_path,
        "transition_steps": args.transition_steps,
        "hold_steps": args.hold_steps,
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
