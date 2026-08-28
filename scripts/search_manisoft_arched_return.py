#!/usr/bin/env python
"""Search a smooth distal control that returns to yz while preserving an arch."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--teacher-episode", required=True)
    parser.add_argument("--start-index", type=int, default=800)
    parser.add_argument("--controlled-points", type=int, default=3)
    parser.add_argument("--control-knots", type=int, default=1)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--transition-steps", type=int, default=160)
    parser.add_argument("--hold-steps", type=int, default=160)
    parser.add_argument("--maximum-action-delta", type=float, default=0.003)
    parser.add_argument("--arch-height", type=float, default=0.30)
    parser.add_argument("--arch-y-margin", type=float, default=0.05)
    parser.add_argument("--minimum-crossed-fraction", type=float, default=0.45)
    parser.add_argument("--minimum-tip-y-margin", type=float, default=0.05)
    parser.add_argument("--target-tip-z", type=float, default=None)
    parser.add_argument("--tip-z-tolerance", type=float, default=0.03)
    parser.add_argument("--tip-z-score-scale", type=float, default=2.0)
    parser.add_argument("--tip-speed-score-scale", type=float, default=0.02)
    parser.add_argument("--maximum-tip-speed", type=float, default=None)
    parser.add_argument("--plane-deficit-score-scale", type=float, default=0.0)
    parser.add_argument("--plane-distance-score-scale", type=float, default=1.0)
    parser.add_argument("--target-tip-x", type=float, default=None)
    parser.add_argument("--tip-x-tolerance", type=float, default=0.01)
    parser.add_argument("--tip-x-score-scale", type=float, default=1.0)
    parser.add_argument("--wall-clearance-target", type=float, default=0.0)
    parser.add_argument("--wall-clearance-score-scale", type=float, default=0.0)
    parser.add_argument(
        "--target-pose-episode",
        default=None,
        help="Episode whose final distal relative shape defines the pose target.",
    )
    parser.add_argument("--pose-node-count", type=int, default=6)
    parser.add_argument("--pose-score-scale", type=float, default=0.0)
    parser.add_argument("--pose-tolerance", type=float, default=None)
    parser.add_argument("--minimum-tip-angle-to-yz", type=float, default=None)
    parser.add_argument("--tip-angle-score-scale", type=float, default=0.0)
    parser.add_argument("--initial-std", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260866)
    parser.add_argument(
        "--initial-search-json",
        default=None,
        help="Warm-start the CEM mean from a previous search best target action.",
    )
    parser.add_argument(
        "--initial-knot-search-json",
        action="append",
        default=[],
        help="Warm-start each control knot from a previous search JSON.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-output", required=True)
    return parser.parse_args()


def _capture(env: ManiSoftTipTrackingEnv) -> dict[str, object]:
    rod = env.sim._backend._softrobot
    return {
        "positions": rod.position_collection.copy(),
        "velocities": rod.velocity_collection.copy(),
        "directors": rod.director_collection.copy(),
        "omegas": rod.omega_collection.copy(),
        "internal": pack_rod_internal_state(rod),
        "time": float(env.sim._backend.time_tracker),
        "step": int(env.sim.current_step),
    }


def _restore(
    env: ManiSoftTipTrackingEnv,
    state: dict[str, object],
    action: np.ndarray,
) -> None:
    rod = env.sim._backend._softrobot
    rod.position_collection[...] = state["positions"]
    rod.velocity_collection[...] = state["velocities"]
    rod.director_collection[...] = state["directors"]
    rod.omega_collection[...] = state["omegas"]
    restore_rod_internal_state(rod, np.asarray(state["internal"]))
    env.sim._backend.time_tracker = float(state["time"])
    env.sim.current_step = int(state["step"])
    env.muscle.set_activation(action.reshape(6, 3))


def _minimum_jerk(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def _arch_height(
    nodes: np.ndarray,
    geometry: WallRouteGeometry,
    margin: float,
) -> float:
    checked = nodes[geometry.mounting_exempt_nodes :]
    mask = (
        (checked[:, 1] >= geometry.wall_minimum[1] - margin)
        & (checked[:, 1] <= geometry.wall_maximum[1] + margin)
    )
    if not np.any(mask):
        return float("-inf")
    return float(np.min(checked[mask, 2]))


def _simulate(
    env: ManiSoftTipTrackingEnv,
    geometry: WallRouteGeometry,
    initial_state: dict[str, object],
    start_action: np.ndarray,
    target_vector: np.ndarray,
    *,
    controlled_points: int,
    control_knots: int,
    transition_steps: int,
    hold_steps: int,
    maximum_action_delta: float,
    arch_target: float,
    arch_y_margin: float,
    required_fraction: float,
    minimum_tip_y: float,
    target_tip_z: float | None,
    tip_z_tolerance: float,
    tip_z_score_scale: float,
    tip_speed_score_scale: float,
    maximum_tip_speed: float | None,
    plane_deficit_score_scale: float,
    plane_distance_score_scale: float,
    target_tip_x: float | None,
    tip_x_tolerance: float,
    tip_x_score_scale: float,
    wall_clearance_target: float,
    wall_clearance_score_scale: float,
    target_relative_pose: np.ndarray | None,
    pose_score_scale: float,
    pose_tolerance: float | None,
    minimum_tip_angle_to_yz: float | None,
    tip_angle_score_scale: float,
    record: bool = False,
) -> dict[str, object]:
    _restore(env, initial_state, start_action)
    targets = np.repeat(start_action[None, :], control_knots, axis=0).reshape(
        control_knots, 6, 3
    )
    targets[:, -controlled_points:] = target_vector.reshape(
        control_knots, controlled_points, 3
    )
    targets = targets.reshape(control_knots, -1)
    rod = env.sim._backend._softrobot
    best_score = float("inf")
    best_step = 0
    best_metrics: dict[str, float] = {}
    histories: dict[str, list] = {
        "physical_state": [],
        "node_positions": [],
        "node_velocities": [],
        "element_directors": [],
        "element_omegas": [],
        "rod_internal_state": [],
        "actions": [],
        "arch_heights": [],
        "wall_clearances": [],
        "ground_clearances": [],
        "distal_crossed_fractions": [],
        "tip_speeds": [],
    }
    termination_reason = "completed"
    transition_total = control_knots * transition_steps
    total_steps = transition_total + hold_steps
    for step in range(1, total_steps + 1):
        if step <= transition_total:
            knot = min((step - 1) // transition_steps, control_knots - 1)
            local_step = step - knot * transition_steps
            segment_start = start_action if knot == 0 else targets[knot - 1]
            desired = segment_start + _minimum_jerk(local_step / transition_steps) * (
                targets[knot] - segment_start
            )
        else:
            desired = targets[-1]
        previous = start_action if step == 1 else action
        action = previous + np.clip(
            desired - previous,
            -maximum_action_delta,
            maximum_action_delta,
        )
        action = np.clip(action, -0.60, 0.60).astype(np.float32)
        env.muscle.set_activation(action.reshape(6, 3))
        try:
            env.sim.step_with_torque_callback(
                lambda lengths: env.muscle.evaluate(lengths)
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            termination_reason = "dynamics_failure"
            break
        nodes = rod.position_collection.T.astype(np.float64, copy=False)
        velocities = rod.velocity_collection.T.astype(np.float64, copy=False)
        metrics = wall_crossing_metrics(geometry, nodes, velocities, 1)
        arch = _arch_height(nodes, geometry, arch_y_margin)
        if metrics.wall_clearance < 0:
            termination_reason = "wall_collision"
            break
        if metrics.ground_clearance < -geometry.ground_violation_tolerance:
            termination_reason = "ground_violation"
            break
        tip = nodes[-1]
        arch_deficit = max(0.0, arch_target - arch)
        crossing_deficit = max(0.0, required_fraction - metrics.distal_crossed_fraction)
        y_deficit = max(0.0, minimum_tip_y - float(tip[1]))
        plane_deficit = max(0.0, abs(float(tip[0])) - 0.01)
        tip_x_error = abs(
            float(tip[0]) - (0.0 if target_tip_x is None else target_tip_x)
        )
        wall_clearance_deficit = max(
            0.0, wall_clearance_target - metrics.wall_clearance
        )
        pose_rmse = 0.0
        if target_relative_pose is not None:
            current_pose = nodes[-len(target_relative_pose) :]
            current_relative_pose = current_pose - current_pose[-1]
            pose_rmse = float(
                np.sqrt(np.mean((current_relative_pose - target_relative_pose) ** 2))
            )
        tip_tangent = nodes[-1] - nodes[-2]
        tip_tangent /= max(float(np.linalg.norm(tip_tangent)), 1e-12)
        tip_angle_to_yz = float(
            np.degrees(np.arcsin(np.clip(abs(tip_tangent[0]), 0.0, 1.0)))
        )
        angle_deficit = (
            0.0
            if minimum_tip_angle_to_yz is None
            else max(0.0, minimum_tip_angle_to_yz - tip_angle_to_yz)
        )
        # Plane distance remains primary only after safety/arch constraints.
        score = (
            plane_distance_score_scale * abs(float(tip[0]))
            + (
                0.0
                if target_tip_x is None
                else tip_x_score_scale * tip_x_error
            )
            + plane_deficit_score_scale * plane_deficit**2
            + wall_clearance_score_scale * wall_clearance_deficit**2
            + pose_score_scale * pose_rmse
            + tip_angle_score_scale * angle_deficit**2
            + 2000.0 * arch_deficit**2
            + 4.0 * crossing_deficit
            + 10.0 * y_deficit
            + (
                0.0
                if target_tip_z is None
                else tip_z_score_scale * abs(float(tip[2]) - target_tip_z)
            )
            + tip_speed_score_scale * metrics.tip_speed
            + 0.002 * float(np.mean((action / 0.60) ** 2))
        )
        if score < best_score:
            best_score = float(score)
            best_step = step
            best_metrics = {
                "plane_distance": abs(float(tip[0])),
                "tip_x": float(tip[0]),
                "tip_x_error": float(tip_x_error),
                "tip_y": float(tip[1]),
                "tip_z": float(tip[2]),
                "tip_speed": float(metrics.tip_speed),
                "arch_height": float(arch),
                "distal_crossed_fraction": float(metrics.distal_crossed_fraction),
                "wall_clearance": float(metrics.wall_clearance),
                "ground_clearance": float(metrics.ground_clearance),
                "distal_pose_rmse": float(pose_rmse),
                "tip_angle_to_yz_deg": float(tip_angle_to_yz),
            }
        if record:
            histories["physical_state"].append(env._physical_state().copy())
            histories["node_positions"].append(nodes.copy())
            histories["node_velocities"].append(velocities.copy())
            histories["element_directors"].append(
                rod.director_collection.transpose(2, 0, 1).copy()
            )
            histories["element_omegas"].append(rod.omega_collection.T.copy())
            histories["rod_internal_state"].append(pack_rod_internal_state(rod))
            histories["actions"].append(action.copy())
            histories["arch_heights"].append(arch)
            histories["wall_clearances"].append(metrics.wall_clearance)
            histories["ground_clearances"].append(metrics.ground_clearance)
            histories["distal_crossed_fractions"].append(
                metrics.distal_crossed_fraction
            )
            histories["tip_speeds"].append(metrics.tip_speed)
    success = bool(
        best_metrics
        and (target_tip_x is not None or best_metrics["plane_distance"] <= 0.01)
        and (
            target_tip_x is None
            or best_metrics["tip_x_error"] <= tip_x_tolerance
        )
        and best_metrics["arch_height"] >= arch_target
        and best_metrics["distal_crossed_fraction"] >= required_fraction - 1e-8
        and best_metrics["tip_y"] >= minimum_tip_y
        and best_metrics["wall_clearance"] >= wall_clearance_target
        and (
            pose_tolerance is None
            or best_metrics["distal_pose_rmse"] <= pose_tolerance
        )
        and (
            minimum_tip_angle_to_yz is None
            or best_metrics["tip_angle_to_yz_deg"] >= minimum_tip_angle_to_yz
        )
        and (
            target_tip_z is None
            or abs(best_metrics["tip_z"] - target_tip_z) <= tip_z_tolerance
        )
        and (
            maximum_tip_speed is None
            or best_metrics["tip_speed"] <= maximum_tip_speed
        )
    )
    result: dict[str, object] = {
        "score": float(best_score),
        "best_step": int(best_step),
        "success": success,
        "termination_reason": termination_reason,
        "target_action": targets[-1],
        "target_actions": targets,
        **best_metrics,
    }
    if record:
        result["histories"] = {
            key: np.asarray(value)[:best_step] for key, value in histories.items()
        }
    return result


def main() -> None:
    args = parse_args()
    if not 1 <= args.controlled_points <= 6:
        raise ValueError("controlled-points must lie in [1,6]")
    if not 1 <= args.control_knots <= 3:
        raise ValueError("control-knots must lie in [1,3]")
    if args.initial_knot_search_json and args.initial_search_json is not None:
        raise ValueError("use either initial-search-json or knot warm starts")
    if args.initial_knot_search_json and len(args.initial_knot_search_json) not in (
        1,
        args.control_knots,
    ):
        raise ValueError("provide one or one-per-knot warm-start JSON")
    if args.samples < args.elite_count or args.elite_count < 2:
        raise ValueError("samples must be >= elite-count >= 2")
    if args.target_tip_z is not None and args.target_tip_z <= 0:
        raise ValueError("target-tip-z must be positive or omitted")
    if args.tip_z_tolerance <= 0 or args.tip_z_score_scale < 0:
        raise ValueError("tip-z tolerance/score scale is invalid")
    if args.tip_speed_score_scale < 0 or (
        args.maximum_tip_speed is not None and args.maximum_tip_speed <= 0
    ):
        raise ValueError("tip-speed settings are invalid")
    if min(
        args.plane_deficit_score_scale,
        args.plane_distance_score_scale,
        args.wall_clearance_target,
        args.wall_clearance_score_scale,
    ) < 0:
        raise ValueError("plane/wall scoring settings are invalid")
    if not 2 <= args.pose_node_count <= 21 or args.pose_score_scale < 0:
        raise ValueError("pose scoring settings are invalid")
    if args.pose_tolerance is not None and args.pose_tolerance <= 0:
        raise ValueError("pose tolerance must be positive or omitted")
    if args.tip_x_tolerance <= 0 or args.tip_x_score_scale < 0:
        raise ValueError("tip-x tolerance/score scale is invalid")
    if (
        args.minimum_tip_angle_to_yz is not None
        and not 0 <= args.minimum_tip_angle_to_yz <= 90
    ) or args.tip_angle_score_scale < 0:
        raise ValueError("tip angle settings are invalid")
    if args.initial_std <= 0:
        raise ValueError("initial-std must be positive")
    scenario = Path(args.scenario).expanduser().resolve()
    task_config = Path(args.task_config).expanduser().resolve()
    teacher_path = Path(args.teacher_episode).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    trajectory_output = Path(args.trajectory_output).expanduser().resolve()
    for path in (scenario, task_config, teacher_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() or trajectory_output.exists():
        raise FileExistsError("search output already exists")
    geometry = WallRouteGeometry.from_dict(
        yaml.safe_load(task_config.read_text(encoding="utf-8"))["task"]
    )
    with np.load(teacher_path, allow_pickle=False) as archive:
        state_count = len(archive["physical_state"])
        if not 0 < args.start_index < state_count - 1:
            raise ValueError("start-index is outside teacher episode")
        teacher = {
            key: np.asarray(archive[key])
            for key in (
                "physical_state",
                "node_positions",
                "node_velocities",
                "element_directors",
                "element_omegas",
                "rod_internal_state",
                "actions",
            )
        }
        episode_seed = int(np.asarray(archive["episode_seed"]).reshape(()))
        control_dt = float(np.asarray(archive["control_dt"]).reshape(()))
    target_relative_pose = None
    if args.target_pose_episode is not None:
        pose_path = Path(args.target_pose_episode).expanduser().resolve()
        if not pose_path.is_file():
            raise FileNotFoundError(pose_path)
        with np.load(pose_path, allow_pickle=False) as archive:
            pose_nodes = np.asarray(archive["node_positions"][-1], dtype=np.float64)
        pose_nodes = pose_nodes[-args.pose_node_count :]
        target_relative_pose = pose_nodes - pose_nodes[-1]

    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=geometry.target,
        episode_steps=args.control_knots * args.transition_steps + args.hold_steps,
        absolute_action_limit=0.60,
        muscle_torque_scale=45.0,
    )
    env.reset(seed=episode_seed)
    rod = env.sim._backend._softrobot
    index = args.start_index
    rod.position_collection[...] = teacher["node_positions"][index].T
    rod.velocity_collection[...] = teacher["node_velocities"][index].T
    rod.director_collection[...] = teacher["element_directors"][index].transpose(1, 2, 0)
    rod.omega_collection[...] = teacher["element_omegas"][index].T
    restore_rod_internal_state(rod, teacher["rod_internal_state"][index])
    env.sim._backend.time_tracker += index * control_dt
    env.sim.current_step += index * int(round(control_dt / env.sim._backend.dt))
    initial_state = _capture(env)
    start_action = teacher["actions"][index - 1].astype(np.float32)
    minimum_tip_y = geometry.wall_maximum[1] + args.minimum_tip_y_margin

    rng = np.random.default_rng(args.seed)
    distal_start = start_action.reshape(6, 3)[-args.controlled_points :].reshape(-1)
    distal_knots_start = np.tile(distal_start, args.control_knots)
    mean = distal_knots_start.copy()
    if args.initial_search_json is not None:
        initial_search = Path(args.initial_search_json).expanduser().resolve()
        if not initial_search.is_file():
            raise FileNotFoundError(initial_search)
        previous = json.loads(initial_search.read_text(encoding="utf-8"))
        previous_action = np.asarray(
            previous["best"]["target_action"], dtype=np.float32
        ).reshape(6, 3)
        previous_targets = previous["best"].get("target_actions")
        if previous_targets is not None:
            previous_targets = np.asarray(previous_targets, dtype=np.float32).reshape(
                -1, 6, 3
            )
            if len(previous_targets) == args.control_knots:
                mean = previous_targets[:, -args.controlled_points :].reshape(-1)
            else:
                mean = np.tile(
                    previous_action[-args.controlled_points :].reshape(-1),
                    args.control_knots,
                )
        else:
            mean = np.tile(
                previous_action[-args.controlled_points :].reshape(-1),
                args.control_knots,
            )
    if args.initial_knot_search_json:
        knot_paths = list(args.initial_knot_search_json)
        if len(knot_paths) == 1:
            knot_paths *= args.control_knots
        knot_values = []
        for value in knot_paths:
            knot_path = Path(value).expanduser().resolve()
            if not knot_path.is_file():
                raise FileNotFoundError(knot_path)
            previous = json.loads(knot_path.read_text(encoding="utf-8"))
            previous_action = np.asarray(
                previous["best"]["target_action"], dtype=np.float32
            ).reshape(6, 3)
            knot_values.append(previous_action[-args.controlled_points :].reshape(-1))
        mean = np.concatenate(knot_values)
    std = np.full_like(mean, args.initial_std)
    best_vector = mean.copy()
    best = _simulate(
        env,
        geometry,
        initial_state,
        start_action,
        best_vector,
        controlled_points=args.controlled_points,
        control_knots=args.control_knots,
        transition_steps=args.transition_steps,
        hold_steps=args.hold_steps,
        maximum_action_delta=args.maximum_action_delta,
        arch_target=args.arch_height,
        arch_y_margin=args.arch_y_margin,
        required_fraction=args.minimum_crossed_fraction,
        minimum_tip_y=minimum_tip_y,
        target_tip_z=args.target_tip_z,
        tip_z_tolerance=args.tip_z_tolerance,
        tip_z_score_scale=args.tip_z_score_scale,
        tip_speed_score_scale=args.tip_speed_score_scale,
        maximum_tip_speed=args.maximum_tip_speed,
        plane_deficit_score_scale=args.plane_deficit_score_scale,
        plane_distance_score_scale=args.plane_distance_score_scale,
        target_tip_x=args.target_tip_x,
        tip_x_tolerance=args.tip_x_tolerance,
        tip_x_score_scale=args.tip_x_score_scale,
        wall_clearance_target=args.wall_clearance_target,
        wall_clearance_score_scale=args.wall_clearance_score_scale,
        target_relative_pose=target_relative_pose,
        pose_score_scale=args.pose_score_scale,
        pose_tolerance=args.pose_tolerance,
        minimum_tip_angle_to_yz=args.minimum_tip_angle_to_yz,
        tip_angle_score_scale=args.tip_angle_score_scale,
    )
    rows = []
    for iteration in range(args.iterations):
        samples = np.clip(
            rng.normal(mean, std, size=(args.samples, len(mean))), -0.60, 0.60
        )
        samples[0] = mean
        samples[1] = distal_knots_start
        evaluated = []
        for sample in samples:
            result = _simulate(
                env,
                geometry,
                initial_state,
                start_action,
                sample,
                controlled_points=args.controlled_points,
                control_knots=args.control_knots,
                transition_steps=args.transition_steps,
                hold_steps=args.hold_steps,
                maximum_action_delta=args.maximum_action_delta,
                arch_target=args.arch_height,
                arch_y_margin=args.arch_y_margin,
                required_fraction=args.minimum_crossed_fraction,
                minimum_tip_y=minimum_tip_y,
                target_tip_z=args.target_tip_z,
                tip_z_tolerance=args.tip_z_tolerance,
                tip_z_score_scale=args.tip_z_score_scale,
                tip_speed_score_scale=args.tip_speed_score_scale,
                maximum_tip_speed=args.maximum_tip_speed,
                plane_deficit_score_scale=args.plane_deficit_score_scale,
                plane_distance_score_scale=args.plane_distance_score_scale,
                target_tip_x=args.target_tip_x,
                tip_x_tolerance=args.tip_x_tolerance,
                tip_x_score_scale=args.tip_x_score_scale,
                wall_clearance_target=args.wall_clearance_target,
                wall_clearance_score_scale=args.wall_clearance_score_scale,
                target_relative_pose=target_relative_pose,
                pose_score_scale=args.pose_score_scale,
                pose_tolerance=args.pose_tolerance,
                minimum_tip_angle_to_yz=args.minimum_tip_angle_to_yz,
                tip_angle_score_scale=args.tip_angle_score_scale,
            )
            evaluated.append((float(result["score"]), sample.copy(), result))
            if float(result["score"]) < float(best["score"]):
                best = result
                best_vector = sample.copy()
        evaluated.sort(key=lambda row: row[0])
        elites = np.stack([row[1] for row in evaluated[: args.elite_count]])
        mean = 0.25 * mean + 0.75 * np.mean(elites, axis=0)
        std = np.clip(0.25 * std + 0.75 * np.std(elites, axis=0), 0.01, 0.20)
        row = {
            "iteration": iteration + 1,
            "best_score": float(best["score"]),
            "best_plane_distance_m": float(best.get("plane_distance", np.inf)),
            "best_tip_x_m": float(best.get("tip_x", np.inf)),
            "best_arch_height_m": float(best.get("arch_height", -np.inf)),
            "best_tip_y_m": float(best.get("tip_y", -np.inf)),
            "best_tip_z_m": float(best.get("tip_z", np.inf)),
            "best_crossed_fraction": float(
                best.get("distal_crossed_fraction", 0.0)
            ),
            "best_distal_pose_rmse_m": float(
                best.get("distal_pose_rmse", np.inf)
            ),
            "best_tip_angle_to_yz_deg": float(
                best.get("tip_angle_to_yz_deg", -np.inf)
            ),
            "success": bool(best["success"]),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    recorded = _simulate(
        env,
        geometry,
        initial_state,
        start_action,
        best_vector,
        controlled_points=args.controlled_points,
        control_knots=args.control_knots,
        transition_steps=args.transition_steps,
        hold_steps=args.hold_steps,
        maximum_action_delta=args.maximum_action_delta,
        arch_target=args.arch_height,
        arch_y_margin=args.arch_y_margin,
        required_fraction=args.minimum_crossed_fraction,
        minimum_tip_y=minimum_tip_y,
        target_tip_z=args.target_tip_z,
        tip_z_tolerance=args.tip_z_tolerance,
        tip_z_score_scale=args.tip_z_score_scale,
        tip_speed_score_scale=args.tip_speed_score_scale,
        maximum_tip_speed=args.maximum_tip_speed,
        plane_deficit_score_scale=args.plane_deficit_score_scale,
        plane_distance_score_scale=args.plane_distance_score_scale,
        target_tip_x=args.target_tip_x,
        tip_x_tolerance=args.tip_x_tolerance,
        tip_x_score_scale=args.tip_x_score_scale,
        wall_clearance_target=args.wall_clearance_target,
        wall_clearance_score_scale=args.wall_clearance_score_scale,
        target_relative_pose=target_relative_pose,
        pose_score_scale=args.pose_score_scale,
        pose_tolerance=args.pose_tolerance,
        minimum_tip_angle_to_yz=args.minimum_tip_angle_to_yz,
        tip_angle_score_scale=args.tip_angle_score_scale,
        record=True,
    )
    env.close()
    histories = recorded.pop("histories")
    trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_output.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int64),
            kind=np.asarray("manisoft_arched_return_search_trajectory"),
            start_index=np.asarray(args.start_index, dtype=np.int64),
            start_physical_state=teacher["physical_state"][index],
            start_node_positions=teacher["node_positions"][index],
            start_node_velocities=teacher["node_velocities"][index],
            start_element_directors=teacher["element_directors"][index],
            start_element_omegas=teacher["element_omegas"][index],
            start_rod_internal_state=teacher["rod_internal_state"][index],
            start_action=start_action,
            **histories,
        )
    summary = {
        "kind": "manisoft_arched_return_search",
        "scenario": str(scenario),
        "task_config": str(task_config),
        "teacher_episode": str(teacher_path),
        "trajectory": str(trajectory_output),
        "settings": vars(args),
        "iterations": rows,
        "best": {
            key: (
                np.asarray(value).tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in recorded.items()
        },
    }
    summary["settings"]["output"] = str(output)
    summary["settings"]["trajectory_output"] = str(trajectory_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
