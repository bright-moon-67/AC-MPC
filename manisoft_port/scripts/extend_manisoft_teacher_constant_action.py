#!/usr/bin/env python
"""Extend a traceable ManiSoft teacher by holding its final control action."""

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
from antmaze_ac.envs.table_entry_bank import (
    pack_rod_internal_state,
    restore_rod_internal_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-teacher", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--append-steps", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch-y-margin", type=float, default=0.05)
    parser.add_argument("--required-arch-height", type=float, default=0.30)
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


def _arch_height(
    nodes: np.ndarray, geometry: WallRouteGeometry, margin: float
) -> float:
    checked = nodes[geometry.mounting_exempt_nodes :]
    mask = (
        (checked[:, 1] >= geometry.wall_minimum[1] - margin)
        & (checked[:, 1] <= geometry.wall_maximum[1] + margin)
    )
    return float(np.min(checked[mask, 2])) if np.any(mask) else float("nan")


def main() -> None:
    args = parse_args()
    if args.append_steps < 1:
        raise ValueError("append-steps must be positive")
    source = Path(args.source_teacher).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    task_config = Path(args.task_config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (source, scenario, task_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    geometry = WallRouteGeometry.from_dict(
        yaml.safe_load(task_config.read_text(encoding="utf-8"))["task"]
    )
    with np.load(source, allow_pickle=False) as archive:
        teacher = {key: np.asarray(archive[key]) for key in archive.files}
    state_keys = (
        "physical_state",
        "node_positions",
        "node_velocities",
        "element_directors",
        "element_omegas",
        "rod_internal_state",
    )
    seed = int(np.asarray(teacher["episode_seed"]).reshape(()))
    control_dt = float(np.asarray(teacher["control_dt"]).reshape(()))
    final_action = np.asarray(teacher["actions"][-1], dtype=np.float32)
    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=geometry.target,
        episode_steps=len(teacher["actions"]) + args.append_steps + 1,
        absolute_action_limit=0.60,
        muscle_torque_scale=45.0,
    )
    env.reset(seed=seed)
    rod = env.sim._backend._softrobot
    rod.position_collection[...] = teacher["node_positions"][-1].T
    rod.velocity_collection[...] = teacher["node_velocities"][-1].T
    rod.director_collection[...] = teacher["element_directors"][-1].transpose(1, 2, 0)
    rod.omega_collection[...] = teacher["element_omegas"][-1].T
    restore_rod_internal_state(rod, teacher["rod_internal_state"][-1])
    source_steps = len(teacher["actions"])
    env.sim._backend.time_tracker += source_steps * control_dt
    env.sim.current_step += source_steps * int(round(control_dt / env.sim._backend.dt))
    env.muscle.set_activation(final_action.reshape(6, 3))
    appended = {key: [] for key in state_keys}
    try:
        for _ in range(args.append_steps):
            env.sim.step_with_torque_callback(
                lambda lengths: env.muscle.evaluate(lengths)
            )
            current = _capture(env)
            for key in state_keys:
                appended[key].append(current[key])
    finally:
        env.close()
    arrays = {
        key: np.concatenate((teacher[key], np.asarray(appended[key])), axis=0)
        for key in state_keys
    }
    arrays["actions"] = np.concatenate(
        (
            np.asarray(teacher["actions"], dtype=np.float32),
            np.repeat(final_action[None, :], args.append_steps, axis=0),
        ),
        axis=0,
    )
    stage_ids = np.concatenate(
        (
            np.asarray(teacher["stage_ids"][:-1], dtype=np.int8),
            np.full(args.append_steps + 1, 5, dtype=np.int8),
        )
    )
    wall, ground, speed, fraction, tip_x, arch = [], [], [], [], [], []
    for nodes, velocities in zip(arrays["node_positions"], arrays["node_velocities"]):
        metrics = wall_crossing_metrics(geometry, nodes, velocities, 1)
        wall.append(metrics.wall_clearance)
        ground.append(metrics.ground_clearance)
        speed.append(metrics.tip_speed)
        fraction.append(metrics.distal_crossed_fraction)
        tip_x.append(metrics.tip_x)
        arch.append(_arch_height(nodes, geometry, args.arch_y_margin))
    wall = np.asarray(wall, dtype=np.float32)
    ground = np.asarray(ground, dtype=np.float32)
    speed = np.asarray(speed, dtype=np.float32)
    fraction = np.asarray(fraction, dtype=np.float32)
    tip_x = np.asarray(tip_x, dtype=np.float32)
    arch = np.asarray(arch, dtype=np.float32)
    if np.min(wall) <= 0:
        raise RuntimeError("extended teacher contacts the moved wall")
    if np.min(ground) < -geometry.ground_violation_tolerance:
        raise RuntimeError("extended teacher violates the ground")
    enforcement_start = int(np.ceil(0.56 * len(arrays["actions"])))
    if np.nanmin(arch[enforcement_start:]) < args.required_arch_height:
        raise RuntimeError("extended teacher violates the required arch height")
    action_delta = np.diff(
        np.vstack((np.zeros((1, 18), dtype=np.float32), arrays["actions"])), axis=0
    )
    maximum_action_delta = float(np.max(np.abs(action_delta)))
    if maximum_action_delta > 0.003 + 1e-7:
        raise RuntimeError("extended teacher violates the action-rate limit")
    payload = dict(teacher)
    payload.update(
        {
            "task_config_sha256": np.asarray(_sha256(task_config)),
            "source_trajectory_sha256": np.asarray(_sha256(source)),
            "terminal_step": np.asarray(len(arrays["actions"]), dtype=np.int64),
            "constant_action_extension_steps": np.asarray(args.append_steps, dtype=np.int64),
            "stage_ids": stage_ids,
            "wall_clearances": wall,
            "ground_clearances": ground,
            "tip_speeds": speed,
            "distal_crossed_fractions": fraction,
            "tip_x": tip_x,
            "arch_heights": arch,
            **arrays,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(output)
    final_nodes = arrays["node_positions"][-1]
    tangent = final_nodes[-1] - final_nodes[-2]
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    summary = {
        "kind": "manisoft_constant_action_extended_teacher_summary",
        "episode": str(output),
        "episode_sha256": _sha256(output),
        "source_teacher": str(source),
        "source_teacher_sha256": _sha256(source),
        "task_config": str(task_config),
        "task_config_sha256": _sha256(task_config),
        "source_control_steps": source_steps,
        "constant_action_extension_steps": args.append_steps,
        "control_steps": len(arrays["actions"]),
        "duration_seconds": len(arrays["actions"]) * control_dt,
        "maximum_action_delta": maximum_action_delta,
        "minimum_wall_clearance_m": float(np.min(wall)),
        "minimum_ground_clearance_m": float(np.min(ground)),
        "minimum_enforced_arch_height_m": float(np.nanmin(arch[enforcement_start:])),
        "final_arch_height_m": float(arch[-1]),
        "final_tip_xyz_m": final_nodes[-1].tolist(),
        "final_tip_speed_mps": float(speed[-1]),
        "final_distal_crossed_fraction": float(fraction[-1]),
        "final_tip_angle_to_yz_deg": float(
            np.degrees(np.arcsin(np.clip(abs(tangent[0]), 0.0, 1.0)))
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
