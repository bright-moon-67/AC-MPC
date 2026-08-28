#!/usr/bin/env python
"""Generate a dynamically integrated smooth wall-route teacher episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.manisoft_wall_crossing_sac_env import wall_crossing_metrics
from antmaze_ac.envs.table_entry_bank import pack_rod_internal_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--seed-trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-scale", type=float, default=2.5)
    parser.add_argument("--action-scale", type=float, default=2.5)
    parser.add_argument("--maximum-hold-steps", type=int, default=600)
    parser.add_argument("--terminal-speed", type=float, default=0.10)
    parser.add_argument("--required-crossing-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20260864)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(env: ManiSoftTipTrackingEnv) -> dict[str, np.ndarray]:
    rod = env.sim._backend._softrobot
    return {
        "physical_state": np.asarray(env._physical_state(), dtype=np.float32),
        "node_positions": rod.position_collection.T.astype(np.float64, copy=True),
        "node_velocities": rod.velocity_collection.T.astype(np.float64, copy=True),
        "element_directors": rod.director_collection.transpose(2, 0, 1).astype(
            np.float64, copy=True
        ),
        "element_omegas": rod.omega_collection.T.astype(np.float64, copy=True),
        "rod_internal_state": pack_rod_internal_state(rod),
    }


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resample_actions(
    actions: np.ndarray, time_scale: float, action_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    count = int(round(len(actions) * time_scale))
    coordinates = np.linspace(0.0, len(actions) - 1, count)
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, len(actions) - 1)
    weight = (coordinates - lower)[:, None]
    values = (1.0 - weight) * actions[lower] + weight * actions[upper]
    return (action_scale * values).astype(np.float32), coordinates


def main() -> None:
    args = parse_args()
    scenario = Path(args.scenario).expanduser().resolve()
    task_config = Path(args.task_config).expanduser().resolve()
    seed_trajectory = Path(args.seed_trajectory).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (scenario, task_config, seed_trajectory):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    if (
        args.time_scale <= 1
        or args.action_scale <= 0
        or args.maximum_hold_steps < 1
        or args.terminal_speed <= 0
        or not 0 < args.required_crossing_fraction <= 1
    ):
        raise ValueError("invalid smooth-teacher generation settings")

    task_payload = yaml.safe_load(task_config.read_text())
    geometry = WallRouteGeometry.from_dict(task_payload["task"])
    scenario_payload = yaml.safe_load(scenario.read_text())
    control_dt = float(scenario_payload["backend"]["dt"]) * int(
        scenario_payload["environment"]["update_interval"]
    )
    with np.load(seed_trajectory, allow_pickle=False) as archive:
        source_actions = np.asarray(archive["actions"], dtype=np.float32)
        source_stages = np.asarray(archive["stages"], dtype=np.int8)
        source_control_dt = float(np.asarray(archive["control_dt"]).reshape(()))
    if source_actions.ndim != 2 or source_actions.shape[1] != 18:
        raise ValueError("seed trajectory actions must have shape [transition,18]")
    if source_stages.shape != (len(source_actions) + 1,):
        raise ValueError("seed trajectory stages must align with its states")
    if not np.isclose(source_control_dt, control_dt, atol=1e-12, rtol=0):
        raise ValueError("seed and teacher control time steps differ")
    actions, source_coordinates = _resample_actions(
        source_actions, args.time_scale, args.action_scale
    )
    if np.max(np.abs(actions)) > 0.60 + 1e-7:
        raise ValueError("scaled teacher action exceeds the 0.60 limit")

    maximum_steps = len(actions) + args.maximum_hold_steps
    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=geometry.target,
        episode_steps=maximum_steps,
        absolute_action_limit=0.60,
        muscle_torque_scale=45.0,
    )
    env.reset(seed=args.seed)
    initial = _capture(env)
    state_rows = {key: [value] for key, value in initial.items()}
    applied_actions: list[np.ndarray] = []
    stage_ids = [0]
    wall_clearances = []
    ground_clearances = []
    tip_speeds = []
    distal_fractions = []
    tip_x = []

    def append_metrics() -> object:
        rod = env.sim._backend._softrobot
        metrics = wall_crossing_metrics(
            geometry,
            rod.position_collection.T,
            rod.velocity_collection.T,
            1,
        )
        wall_clearances.append(metrics.wall_clearance)
        ground_clearances.append(metrics.ground_clearance)
        tip_speeds.append(metrics.tip_speed)
        distal_fractions.append(metrics.distal_crossed_fraction)
        tip_x.append(metrics.tip_x)
        return metrics

    append_metrics()
    terminal_step = None
    previous_tip_x = float(tip_x[-1])
    for step in range(maximum_steps):
        in_resampled_seed = step < len(actions)
        action = actions[step] if in_resampled_seed else actions[-1]
        env.muscle.set_activation(action.reshape(6, 3))
        try:
            env.sim.step_with_torque_callback(
                lambda lengths: env.muscle.evaluate(lengths)
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            env.close()
            raise RuntimeError(f"teacher dynamics failed at step {step + 1}") from error
        capture = _capture(env)
        for key, value in capture.items():
            state_rows[key].append(value)
        applied_actions.append(action.copy())
        if in_resampled_seed:
            source_index = min(
                int(round(source_coordinates[step])) + 1,
                len(source_stages) - 1,
            )
            stage_ids.append(int(source_stages[source_index]))
        else:
            stage_ids.append(5)
        metrics = append_metrics()
        if metrics.wall_clearance < 0:
            env.close()
            raise RuntimeError(f"teacher intersects the wall at step {step + 1}")
        if metrics.ground_clearance < -geometry.ground_violation_tolerance:
            env.close()
            raise RuntimeError(f"teacher violates the ground at step {step + 1}")
        safe_crossing = (
            not in_resampled_seed
            and metrics.distal_crossed_fraction
            >= args.required_crossing_fraction - 1e-8
            and previous_tip_x * metrics.tip_x <= 0
            and metrics.tip_speed <= args.terminal_speed
        )
        previous_tip_x = metrics.tip_x
        if safe_crossing:
            terminal_step = step + 1
            break
    env.close()
    if terminal_step is None:
        raise RuntimeError("teacher did not safely reach the yz plane")

    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "kind": np.asarray("manisoft_smooth_wall_teacher_episode"),
        "scenario_sha256": np.asarray(_sha256(scenario)),
        "task_config_sha256": np.asarray(_sha256(task_config)),
        "source_trajectory_sha256": np.asarray(_sha256(seed_trajectory)),
        "control_dt": np.asarray(control_dt, dtype=np.float64),
        "episode_seed": np.asarray(args.seed, dtype=np.int64),
        "route_side": np.asarray(1, dtype=np.int8),
        "time_scale": np.asarray(args.time_scale, dtype=np.float64),
        "action_scale": np.asarray(args.action_scale, dtype=np.float64),
        "terminal_step": np.asarray(terminal_step, dtype=np.int64),
        "stage_ids": np.asarray(stage_ids, dtype=np.int8),
        "actions": np.asarray(applied_actions, dtype=np.float32),
        "wall_clearances": np.asarray(wall_clearances, dtype=np.float32),
        "ground_clearances": np.asarray(ground_clearances, dtype=np.float32),
        "tip_speeds": np.asarray(tip_speeds, dtype=np.float32),
        "distal_crossed_fractions": np.asarray(
            distal_fractions, dtype=np.float32
        ),
        "tip_x": np.asarray(tip_x, dtype=np.float32),
        **{key: np.asarray(value) for key, value in state_rows.items()},
    }
    state_count = len(arrays["node_positions"])
    if len(arrays["actions"]) != state_count - 1:
        raise RuntimeError("teacher action/state alignment is invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(output, arrays)

    tip_velocities = arrays["node_velocities"][:, -1]
    tip_accelerations = np.diff(tip_velocities, axis=0) / control_dt
    tip_jerks = np.diff(tip_accelerations, axis=0) / control_dt
    action_deltas = np.diff(
        np.concatenate((np.zeros((1, 18), dtype=np.float32), arrays["actions"])),
        axis=0,
    )
    action_second_differences = np.diff(action_deltas, axis=0)
    summary = {
        "schema_version": 1,
        "kind": "manisoft_smooth_wall_teacher_summary",
        "episode": str(output),
        "episode_sha256": _sha256(output),
        "scenario": str(scenario),
        "scenario_sha256": _sha256(scenario),
        "task_config": str(task_config),
        "task_config_sha256": _sha256(task_config),
        "source_trajectory": str(seed_trajectory),
        "source_trajectory_sha256": _sha256(seed_trajectory),
        "time_scale": args.time_scale,
        "action_scale": args.action_scale,
        "control_steps": terminal_step,
        "duration_seconds": terminal_step * control_dt,
        "safe_yz_plane_crossing": True,
        "final_tip_xyz_m": arrays["node_positions"][-1, -1].tolist(),
        "final_tip_speed_mps": float(arrays["tip_speeds"][-1]),
        "final_distal_crossed_fraction": float(
            arrays["distal_crossed_fractions"][-1]
        ),
        "minimum_wall_clearance_m": float(np.min(arrays["wall_clearances"])),
        "minimum_ground_clearance_m": float(
            np.min(arrays["ground_clearances"])
        ),
        "maximum_tip_speed_mps": float(np.max(arrays["tip_speeds"])),
        "tip_speed_p95_mps": float(np.quantile(arrays["tip_speeds"], 0.95)),
        "maximum_tip_acceleration_mps2": float(
            np.max(np.linalg.norm(tip_accelerations, axis=1))
        ),
        "tip_jerk_p95_mps3": float(
            np.quantile(np.linalg.norm(tip_jerks, axis=1), 0.95)
        ),
        "maximum_absolute_action": float(np.max(np.abs(arrays["actions"]))),
        "maximum_action_delta": float(np.max(np.abs(action_deltas))),
        "action_delta_rms": float(np.sqrt(np.mean(np.square(action_deltas)))),
        "action_second_difference_rms": float(
            np.sqrt(np.mean(np.square(action_second_differences)))
        ),
        "dynamics_certification": {
            "method": "fresh ManiSoft integration from upright reset",
            "coordinate_interpolation_used": False,
            "wall_collision": False,
            "ground_violation": False,
        },
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
