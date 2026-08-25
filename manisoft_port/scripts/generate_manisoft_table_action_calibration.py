#!/usr/bin/env python
"""Calibrate a compact Cartesian table-action map around each entry pose.

The table entry bank contains six stable bent equilibria.  Around every
equilibrium this script measures the settled tip response to small activation
perturbations, solves bounded inverse-Jacobian problems for +/- global x/y
commands, and validates the resulting physical actions in the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import lsq_linear

from antmaze_ac.envs.manisoft_waypoint_sac_env import ManiSoftWaypointSACEnv
from antmaze_ac.envs.table_entry_bank import load_table_entry_trajectory_bank


_SCENARIO: str
_ENVIRONMENT: dict[str, Any]
_SETTLE_STEPS: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--config",
        default="configs/manisoft_waypoint_sac_table_local_line.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--perturbation", type=float, default=0.015)
    parser.add_argument("--command-distance", type=float, default=0.020)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--orientation-weight", type=float, default=0.15)
    parser.add_argument(
        "--maximum-calibration-action-delta", type=float, default=0.04
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=887000)
    return parser.parse_args()


def _initialize_worker(
    scenario: str, environment: dict[str, Any], settle_steps: int
) -> None:
    global _SCENARIO, _ENVIRONMENT, _SETTLE_STEPS
    _SCENARIO = scenario
    _ENVIRONMENT = environment
    _SETTLE_STEPS = settle_steps


def _rollout(task: tuple[int, int, float, int]) -> dict[str, Any]:
    entry_index, action_index, offset, seed = task
    env = ManiSoftWaypointSACEnv(_SCENARIO, **_ENVIRONMENT)
    calibration_speed = float(
        np.clip(0.025, env.min_desired_speed, env.max_desired_speed)
    )
    _, reset_info = env.reset(
        seed=seed,
        options={"entry_index": entry_index, "desired_speed": calibration_speed},
    )
    start_tip = np.asarray(reset_info["tip_position"], dtype=np.float64)
    start_tangent = np.asarray(reset_info["tip_tangent"], dtype=np.float64)
    equilibrium = np.asarray(reset_info["applied_action"], dtype=np.float64)
    requested = equilibrium.copy()
    requested[action_index] = np.clip(
        requested[action_index] + offset,
        -env.absolute_action_limit,
        env.absolute_action_limit,
    )
    minimum_clearance = float(reset_info["whole_arm_table_clearance"])
    violation = False
    info = reset_info
    for _ in range(_SETTLE_STEPS):
        _, _, _, _, info = env.step(requested.astype(np.float32))
        minimum_clearance = min(
            minimum_clearance, float(info["whole_arm_table_clearance"])
        )
        violation |= bool(
            info["table_violation"]
            or info["dynamics_violation"]
            or info["tip_orientation_violation"]
        )
    response = np.asarray(info["tip_position"], dtype=np.float64) - start_tip
    tangent_response = (
        np.asarray(info["tip_tangent"], dtype=np.float64) - start_tangent
    )
    actual_offset = float(requested[action_index] - equilibrium[action_index])
    env.close()
    return {
        "entry_index": entry_index,
        "action_index": action_index,
        "actual_offset": actual_offset,
        "response": response,
        "tangent_response": tangent_response,
        "orientation_error_degrees": float(
            info["tip_orientation_error_degrees"]
        ),
        "minimum_clearance": minimum_clearance,
        "violation": violation,
    }


def _validate_action(task: tuple[int, int, int, np.ndarray, int]) -> dict[str, Any]:
    entry_index, axis, sign, requested, seed = task
    env = ManiSoftWaypointSACEnv(_SCENARIO, **_ENVIRONMENT)
    calibration_speed = float(
        np.clip(0.025, env.min_desired_speed, env.max_desired_speed)
    )
    _, reset_info = env.reset(
        seed=seed,
        options={"entry_index": entry_index, "desired_speed": calibration_speed},
    )
    start_tip = np.asarray(reset_info["tip_position"], dtype=np.float64)
    minimum_clearance = float(reset_info["whole_arm_table_clearance"])
    violation = False
    info = reset_info
    for _ in range(_SETTLE_STEPS):
        _, _, _, _, info = env.step(np.asarray(requested, dtype=np.float32))
        minimum_clearance = min(
            minimum_clearance, float(info["whole_arm_table_clearance"])
        )
        violation |= bool(
            info["table_violation"]
            or info["dynamics_violation"]
            or info["tip_orientation_violation"]
        )
    response = np.asarray(info["tip_position"], dtype=np.float64) - start_tip
    env.close()
    return {
        "entry_index": entry_index,
        "axis": axis,
        "sign": sign,
        "response": response,
        "tip_tangent": np.asarray(info["tip_tangent"], dtype=np.float64),
        "orientation_error_degrees": float(
            info["tip_orientation_error_degrees"]
        ),
        "minimum_clearance": minimum_clearance,
        "violation": violation,
    }


def main() -> None:
    args = parse_args()
    if min(
        args.perturbation,
        args.command_distance,
        args.maximum_calibration_action_delta,
    ) <= 0:
        raise ValueError("perturbation and command distance must be positive")
    if args.orientation_weight < 0:
        raise ValueError("orientation-weight must be non-negative")
    if min(args.settle_steps, args.workers) < 1:
        raise ValueError("settle steps and workers must be positive")
    scenario = Path(args.scenario).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    environment = dict(payload["environment"])
    bank_path = Path(environment["entry_bank_path"]).expanduser()
    if not bank_path.is_absolute():
        bank_path = (config_path.parent.parent / bank_path).resolve()
    environment["entry_bank_path"] = str(bank_path)
    # Calibration directly requests physical actions and must not terminate
    # because an arbitrary sampled reference happened to be crossed.
    environment.update(
        {
            "action_mode": "absolute",
            "episode_steps": args.settle_steps + 20,
            "success_threshold": 0.001,
            "terminal_capture_radius": 0.002,
            "success_streak": args.settle_steps + 20,
            "terminal_settle_steps": args.settle_steps + 20,
            "stall_grace_steps": args.settle_steps + 20,
            "stall_window_steps": args.settle_steps + 20,
        }
    )
    bank = load_table_entry_trajectory_bank(bank_path)
    equilibrium_actions = np.asarray(bank.actions[:, -1], dtype=np.float64)
    tasks = [
        (entry, action, sign * args.perturbation, args.seed + entry)
        for entry in range(bank.trajectory_count)
        for action in range(18)
        for sign in (-1, 1)
    ]
    context = mp.get_context("spawn")
    with context.Pool(
        min(args.workers, len(tasks)),
        initializer=_initialize_worker,
        initargs=(str(scenario), environment, args.settle_steps),
    ) as pool:
        measurements = pool.map(_rollout, tasks)

    jacobians = np.zeros((bank.trajectory_count, 3, 18), dtype=np.float64)
    tangent_jacobians = np.zeros_like(jacobians)
    calibration_clearance = np.full(bank.trajectory_count, np.inf)
    calibration_violations = np.zeros(bank.trajectory_count, dtype=bool)
    for entry in range(bank.trajectory_count):
        for action in range(18):
            rows = [
                row
                for row in measurements
                if row["entry_index"] == entry and row["action_index"] == action
            ]
            denominator = sum(row["actual_offset"] ** 2 for row in rows)
            if denominator > 0:
                jacobians[entry, :, action] = sum(
                    row["actual_offset"] * row["response"] for row in rows
                ) / denominator
                tangent_jacobians[entry, :, action] = sum(
                    row["actual_offset"] * row["tangent_response"]
                    for row in rows
                ) / denominator
        entry_rows = [row for row in measurements if row["entry_index"] == entry]
        calibration_clearance[entry] = min(
            row["minimum_clearance"] for row in entry_rows
        )
        calibration_violations[entry] = any(row["violation"] for row in entry_rows)

    positive_deltas = np.zeros((bank.trajectory_count, 2, 18), dtype=np.float64)
    negative_deltas = np.zeros_like(positive_deltas)
    predicted = np.zeros((bank.trajectory_count, 2, 2, 3), dtype=np.float64)
    action_limit = float(environment["absolute_action_limit"])
    for entry, (jacobian, tangent_jacobian, equilibrium) in enumerate(
        zip(jacobians, tangent_jacobians, equilibrium_actions)
    ):
        lower = np.maximum(
            -action_limit - equilibrium,
            -args.maximum_calibration_action_delta,
        )
        upper = np.minimum(
            action_limit - equilibrium,
            args.maximum_calibration_action_delta,
        )
        augmented = np.vstack(
            (
                jacobian,
                args.orientation_weight * tangent_jacobian,
                np.sqrt(args.regularization) * np.eye(18),
            )
        )
        for axis in range(2):
            for sign_index, sign in enumerate((1, -1)):
                target = np.zeros(3, dtype=np.float64)
                target[axis] = sign * args.command_distance
                result = lsq_linear(
                    augmented,
                    np.concatenate((target, np.zeros(3), np.zeros(18))),
                    bounds=(lower, upper),
                    tol=1e-10,
                    lsmr_tol=1e-10,
                    max_iter=500,
                )
                if not result.success:
                    raise RuntimeError(
                        f"bounded calibration failed for entry {entry}, axis {axis}, sign {sign}"
                    )
                destination = positive_deltas if sign > 0 else negative_deltas
                destination[entry, axis] = result.x
                predicted[entry, axis, sign_index] = jacobian @ result.x

    validation_tasks = []
    for entry in range(bank.trajectory_count):
        for axis in range(2):
            for sign, deltas in ((1, positive_deltas), (-1, negative_deltas)):
                validation_tasks.append(
                    (
                        entry,
                        axis,
                        sign,
                        np.clip(
                            equilibrium_actions[entry] + deltas[entry, axis],
                            -action_limit,
                            action_limit,
                        ),
                        args.seed + 10_000 + entry * 10 + axis * 2 + (sign < 0),
                    )
                )
    with context.Pool(
        min(args.workers, len(validation_tasks)),
        initializer=_initialize_worker,
        initargs=(str(scenario), environment, args.settle_steps),
    ) as pool:
        validations = pool.map(_validate_action, validation_tasks)

    achieved = np.zeros_like(predicted)
    validation_clearance = np.full(bank.trajectory_count, np.inf)
    validation_violations = np.zeros(bank.trajectory_count, dtype=bool)
    validation_orientation_errors = np.zeros(
        (bank.trajectory_count, 2, 2), dtype=np.float64
    )
    for row in validations:
        sign_index = 0 if row["sign"] > 0 else 1
        achieved[row["entry_index"], row["axis"], sign_index] = row["response"]
        validation_clearance[row["entry_index"]] = min(
            validation_clearance[row["entry_index"]], row["minimum_clearance"]
        )
        validation_violations[row["entry_index"]] |= row["violation"]
        validation_orientation_errors[
            row["entry_index"], row["axis"], sign_index
        ] = row["orientation_error_degrees"]

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int64),
            kind=np.asarray("manisoft_table_cartesian_action_calibration"),
            entry_names=np.asarray(bank.names),
            equilibrium_actions=equilibrium_actions.astype(np.float32),
            tip_jacobians=jacobians.astype(np.float32),
            tip_tangent_jacobians=tangent_jacobians.astype(np.float32),
            positive_action_deltas=positive_deltas.astype(np.float32),
            negative_action_deltas=negative_deltas.astype(np.float32),
            predicted_displacements=predicted.astype(np.float32),
            achieved_displacements=achieved.astype(np.float32),
            command_distance=np.asarray(args.command_distance, dtype=np.float32),
            perturbation=np.asarray(args.perturbation, dtype=np.float32),
            settle_steps=np.asarray(args.settle_steps, dtype=np.int64),
            orientation_weight=np.asarray(
                args.orientation_weight, dtype=np.float32
            ),
            maximum_calibration_action_delta=np.asarray(
                args.maximum_calibration_action_delta, dtype=np.float32
            ),
            validation_orientation_errors_degrees=(
                validation_orientation_errors.astype(np.float32)
            ),
            scenario_sha256=np.asarray(hashlib.sha256(scenario.read_bytes()).hexdigest()),
            entry_bank_sha256=np.asarray(hashlib.sha256(bank_path.read_bytes()).hexdigest()),
            calibration_minimum_clearance=calibration_clearance.astype(np.float32),
            validation_minimum_clearance=validation_clearance.astype(np.float32),
            calibration_violations=calibration_violations,
            validation_violations=validation_violations,
        )
    temporary.replace(output)
    report = {
        "output": str(output),
        "entries": bank.trajectory_count,
        "command_distance": args.command_distance,
        "maximum_calibration_action_delta": (
            args.maximum_calibration_action_delta
        ),
        "singular_values": [
            np.linalg.svd(jacobian, compute_uv=False).tolist()
            for jacobian in jacobians
        ],
        "achieved_displacements": achieved.tolist(),
        "validation_orientation_errors_degrees": (
            validation_orientation_errors.tolist()
        ),
        "maximum_validation_orientation_error_degrees": float(
            np.max(validation_orientation_errors)
        ),
        "minimum_clearance": float(
            min(calibration_clearance.min(), validation_clearance.min())
        ),
        "violations": int(
            calibration_violations.sum() + validation_violations.sum()
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["violations"]:
        raise SystemExit("calibration produced a safety violation")


if __name__ == "__main__":
    main()
