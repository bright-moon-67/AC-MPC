#!/usr/bin/env python
"""Probe distal-only tip reachability toward the yz plane after 40% crossing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from antmaze_ac.envs.manisoft_wall_crossing_sac_env import (
    ManiSoftWallCrossingSACEnv,
)
from antmaze_ac.envs.table_entry_bank import (
    pack_rod_internal_state,
    restore_rod_internal_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--snapshot-index", type=int, default=49)
    parser.add_argument("--crossing-fraction", type=float, default=0.40)
    parser.add_argument("--controlled-points", type=int, default=2)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--transition-steps", type=int, default=120)
    parser.add_argument("--hold-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _capture(env: ManiSoftWallCrossingSACEnv) -> dict[str, object]:
    rod = env.sim._backend._softrobot
    return {
        "positions": rod.position_collection.copy(),
        "velocities": rod.velocity_collection.copy(),
        "directors": rod.director_collection.copy(),
        "omegas": rod.omega_collection.copy(),
        "internal": pack_rod_internal_state(rod),
        "time": float(env.sim._backend.time_tracker),
        "step": int(env.sim.current_step),
        "action": env.previous_action.astype(np.float64, copy=True),
    }


def _restore(env: ManiSoftWallCrossingSACEnv, state: dict[str, object]) -> None:
    rod = env.sim._backend._softrobot
    rod.position_collection[...] = state["positions"]
    rod.velocity_collection[...] = state["velocities"]
    rod.director_collection[...] = state["directors"]
    rod.omega_collection[...] = state["omegas"]
    restore_rod_internal_state(rod, np.asarray(state["internal"]))
    env.sim._backend.time_tracker = float(state["time"])
    env.sim.current_step = int(state["step"])
    env.muscle.set_activation(np.asarray(state["action"]).reshape(6, 3))


def _simulate(
    env: ManiSoftWallCrossingSACEnv,
    state: dict[str, object],
    distal_target: np.ndarray,
    *,
    controlled_points: int,
    transition_steps: int,
    hold_steps: int,
    required_fraction: float,
    record: bool = False,
) -> dict[str, object]:
    _restore(env, state)
    rod = env.sim._backend._softrobot
    start_nodes = np.asarray(state["positions"]).T.copy()
    action = np.asarray(state["action"], dtype=np.float64).reshape(6, 3).copy()
    target = action.copy()
    target[-controlled_points:] = np.asarray(distal_target).reshape(
        controlled_points, 3
    )

    best_distance = abs(float(rod.position_collection[0, -1]))
    best_step = 0
    best_tip = rod.position_collection[:, -1].astype(np.float64, copy=True)
    best_nodes = rod.position_collection.T.astype(np.float64, copy=True)
    best_directors = rod.director_collection.transpose(2, 0, 1).astype(
        np.float64, copy=True
    )
    best_action = action.copy()
    minimum_wall_clearance = float("inf")
    minimum_ground_clearance = float("inf")
    maximum_tip_speed = 0.0
    maximum_proximal_rms_displacement = 0.0
    termination_reason = "completed"
    nodes_history = []
    directors_history = []
    action_history = []
    fraction_history = []
    wall_history = []
    ground_history = []
    speed_history = []

    total_steps = transition_steps + hold_steps
    for step in range(1, total_steps + 1):
        if step <= transition_steps:
            action += np.clip(target - action, -env.max_action_delta, env.max_action_delta)
        env.muscle.set_activation(action.reshape(6, 3))
        try:
            env.sim.step_with_torque_callback(
                lambda lengths: env.muscle.evaluate(lengths)
            )
            metrics = env._metrics()
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            termination_reason = "dynamics_violation"
            break
        nodes = rod.position_collection.T.astype(np.float64, copy=True)
        directors = rod.director_collection.transpose(2, 0, 1).astype(
            np.float64, copy=True
        )
        proximal_count = nodes.shape[0] - int(
            round(required_fraction * (nodes.shape[0] - 1))
        )
        proximal_rms = float(
            np.sqrt(np.mean((nodes[:proximal_count] - start_nodes[:proximal_count]) ** 2))
        )
        minimum_wall_clearance = min(
            minimum_wall_clearance, float(metrics.wall_clearance)
        )
        minimum_ground_clearance = min(
            minimum_ground_clearance, float(metrics.ground_clearance)
        )
        maximum_tip_speed = max(maximum_tip_speed, float(metrics.tip_speed))
        maximum_proximal_rms_displacement = max(
            maximum_proximal_rms_displacement, proximal_rms
        )
        if record:
            nodes_history.append(nodes)
            directors_history.append(directors)
            action_history.append(action.copy())
            fraction_history.append(float(metrics.distal_crossed_fraction))
            wall_history.append(float(metrics.wall_clearance))
            ground_history.append(float(metrics.ground_clearance))
            speed_history.append(float(metrics.tip_speed))

        unsafe_reason = None
        if metrics.wall_clearance < 0:
            unsafe_reason = "virtual_wall_collision"
        elif metrics.ground_clearance < -env.geometry.ground_violation_tolerance:
            unsafe_reason = "ground_violation"
        elif metrics.tip_speed > env.maximum_tip_speed:
            unsafe_reason = "tip_speed"
        if unsafe_reason is not None:
            termination_reason = unsafe_reason
            break

        tip = nodes[-1]
        plane_distance = abs(float(tip[0]))
        if (
            metrics.distal_crossed_fraction >= required_fraction - 1e-8
            and plane_distance < best_distance
        ):
            best_distance = plane_distance
            best_step = step
            best_tip = tip.copy()
            best_nodes = nodes.copy()
            best_directors = directors.copy()
            best_action = action.copy()

    # The primary objective is yz-plane distance.  A mild proximal-motion
    # penalty breaks ties in favor of a bend localized to the crossed suffix.
    score = best_distance + 0.05 * maximum_proximal_rms_displacement
    if best_step == 0:
        score += 1.0
    result: dict[str, object] = {
        "score": float(score),
        "best_plane_distance": float(best_distance),
        "best_step": int(best_step),
        "best_tip": best_tip,
        "best_nodes": best_nodes,
        "best_directors": best_directors,
        "best_action": best_action,
        "target_action": target,
        "minimum_wall_clearance": float(minimum_wall_clearance),
        "minimum_ground_clearance": float(minimum_ground_clearance),
        "maximum_tip_speed": float(maximum_tip_speed),
        "maximum_proximal_rms_displacement": float(
            maximum_proximal_rms_displacement
        ),
        "termination_reason": termination_reason,
    }
    if record:
        result.update(
            {
                "node_positions": np.asarray(nodes_history),
                "element_directors": np.asarray(directors_history),
                "actions": np.asarray(action_history),
                "distal_crossed_fractions": np.asarray(
                    fraction_history, dtype=np.float32
                ),
                "wall_clearances": np.asarray(wall_history, dtype=np.float32),
                "ground_clearances": np.asarray(ground_history, dtype=np.float32),
                "tip_speeds": np.asarray(speed_history, dtype=np.float32),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if not 0 < args.crossing_fraction < 1:
        raise ValueError("crossing-fraction must lie in (0,1)")
    if not 1 <= args.controlled_points <= 6:
        raise ValueError("controlled-points must lie in [1,6]")
    if args.elite_count < 2 or args.samples < args.elite_count:
        raise ValueError("samples must be at least elite-count >= 2")
    if min(args.iterations, args.transition_steps, args.hold_steps) < 1:
        raise ValueError("iteration and rollout lengths must be positive")

    run_path = Path(args.run_config).expanduser().resolve()
    run = json.loads(run_path.read_text(encoding="utf-8"))
    environment_config = dict(run["environment"])
    environment_config["success_crossed_fraction"] = max(
        args.crossing_fraction + 0.10, 0.50
    )

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make_env() -> ManiSoftWallCrossingSACEnv:
        return ManiSoftWallCrossingSACEnv(
            run["scenario"],
            task_config_path=run["task_config"],
            snapshot_bank_path=run["snapshot_bank"],
            **environment_config,
        )

    env = make_env()
    normalization = VecNormalize.load(
        str(Path(args.vecnormalize).expanduser().resolve()),
        DummyVecEnv([make_env]),
    )
    normalization.training = False
    normalization.norm_reward = False
    model = SAC.load(
        str(Path(args.model).expanduser().resolve()), device=args.device
    )
    observation, reset_info = env.reset(
        seed=args.seed, options={"snapshot_index": args.snapshot_index}
    )
    crossing_info = reset_info
    crossing_nodes = None
    crossing_directors = None
    while env.step_count < env.episode_steps:
        normalized = normalization.normalize_obs(observation[None, :])[0]
        policy_action, _ = model.predict(normalized, deterministic=True)
        observation, _, terminated, truncated, crossing_info = env.step(policy_action)
        if (
            crossing_info["distal_crossed_fraction"]
            >= args.crossing_fraction - 1e-8
            and crossing_info["wall_clearance"] >= 0
        ):
            rod = env.sim._backend._softrobot
            crossing_nodes = rod.position_collection.T.astype(
                np.float64, copy=True
            )
            crossing_directors = rod.director_collection.transpose(2, 0, 1).astype(
                np.float64, copy=True
            )
            break
        if terminated or truncated:
            raise RuntimeError(
                "policy terminated before reaching the requested crossing fraction: "
                f"{crossing_info['termination_reason']}"
            )
    if crossing_nodes is None or crossing_directors is None:
        raise RuntimeError("policy did not reach the requested crossing fraction")
    crossing_state = _capture(env)
    crossing_tip = crossing_nodes[-1].copy()
    start_action = np.asarray(crossing_state["action"]).reshape(6, 3)
    distal_start = start_action[-args.controlled_points :].reshape(-1)

    rng = np.random.default_rng(args.seed)
    mean = distal_start.copy()
    std = np.full_like(mean, 0.14)
    best = _simulate(
        env,
        crossing_state,
        distal_start,
        controlled_points=args.controlled_points,
        transition_steps=args.transition_steps,
        hold_steps=args.hold_steps,
        required_fraction=args.crossing_fraction,
    )
    best_vector = distal_start.copy()
    search_rows = []
    for iteration in range(args.iterations):
        samples = np.clip(
            rng.normal(mean, std, size=(args.samples, len(mean))), -0.30, 0.30
        )
        samples[0] = mean
        if iteration == 0:
            samples[1] = distal_start
        evaluated = []
        for sample_index, sample in enumerate(samples):
            result = _simulate(
                env,
                crossing_state,
                sample,
                controlled_points=args.controlled_points,
                transition_steps=args.transition_steps,
                hold_steps=args.hold_steps,
                required_fraction=args.crossing_fraction,
            )
            evaluated.append((float(result["score"]), sample, result))
            if float(result["score"]) < float(best["score"]):
                best = result
                best_vector = sample.copy()
        evaluated.sort(key=lambda item: item[0])
        elites = np.stack([item[1] for item in evaluated[: args.elite_count]])
        mean = 0.25 * mean + 0.75 * np.mean(elites, axis=0)
        std = np.clip(
            0.25 * std + 0.75 * np.std(elites, axis=0), 0.015, 0.20
        )
        row = {
            "iteration": iteration + 1,
            "best_plane_distance": float(best["best_plane_distance"]),
            "iteration_best_plane_distance": float(
                evaluated[0][2]["best_plane_distance"]
            ),
            "iteration_best_score": float(evaluated[0][0]),
        }
        search_rows.append(row)
        print(json.dumps(row), flush=True)

    recorded = _simulate(
        env,
        crossing_state,
        best_vector,
        controlled_points=args.controlled_points,
        transition_steps=args.transition_steps,
        hold_steps=args.hold_steps,
        required_fraction=args.crossing_fraction,
        record=True,
    )
    best_index = max(int(recorded["best_step"]) - 1, 0)
    safe_prefix = slice(0, best_index + 1)
    prefix_wall_clearances = np.asarray(recorded["wall_clearances"])[safe_prefix]
    prefix_ground_clearances = np.asarray(recorded["ground_clearances"])[safe_prefix]
    prefix_tip_speeds = np.asarray(recorded["tip_speeds"])[safe_prefix]
    trajectory_path = Path(args.trajectory_output).expanduser().resolve()
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trajectory_path,
        crossing_node_positions=crossing_nodes,
        crossing_element_directors=crossing_directors,
        node_positions=recorded["node_positions"][: best_index + 1],
        element_directors=recorded["element_directors"][: best_index + 1],
        actions=recorded["actions"][: best_index + 1],
        distal_crossed_fractions=recorded["distal_crossed_fractions"][
            : best_index + 1
        ],
        wall_clearances=recorded["wall_clearances"][: best_index + 1],
        ground_clearances=recorded["ground_clearances"][: best_index + 1],
        tip_speeds=recorded["tip_speeds"][: best_index + 1],
        control_dt=np.asarray(env.control_dt),
        snapshot_index=np.asarray(args.snapshot_index),
        crossing_policy_steps=np.asarray(env.step_count),
    )
    best_nodes = np.asarray(recorded["best_nodes"])
    proximal_count = best_nodes.shape[0] - int(
        round(args.crossing_fraction * (best_nodes.shape[0] - 1))
    )
    final_proximal_rms = float(
        np.sqrt(
            np.mean(
                (best_nodes[:proximal_count] - crossing_nodes[:proximal_count]) ** 2
            )
        )
    )
    summary = {
        "kind": "manisoft_postwall_yz_reachability_probe",
        "snapshot_index": args.snapshot_index,
        "route_side": int(reset_info["route_side"]),
        "required_crossing_fraction": args.crossing_fraction,
        "crossing_policy_steps": int(env.step_count),
        "crossing_tip": crossing_tip.tolist(),
        "crossing_wall_clearance": float(crossing_info["wall_clearance"]),
        "controlled_spatial_points": list(
            range(6 - args.controlled_points, 6)
        ),
        "start_plane_distance": abs(float(crossing_tip[0])),
        "best_plane_distance": float(recorded["best_plane_distance"]),
        "best_tip": np.asarray(recorded["best_tip"]).tolist(),
        "best_step": int(recorded["best_step"]),
        "best_time_seconds": float(recorded["best_step"] * env.control_dt),
        "best_distal_action": best_vector.reshape(args.controlled_points, 3).tolist(),
        "minimum_wall_clearance_to_best": float(
            np.min(prefix_wall_clearances)
        ),
        "minimum_ground_clearance_to_best": float(
            np.min(prefix_ground_clearances)
        ),
        "maximum_tip_speed_to_best": float(np.max(prefix_tip_speeds)),
        "final_proximal_rms_displacement": final_proximal_rms,
        "maximum_proximal_rms_displacement": float(
            recorded["maximum_proximal_rms_displacement"]
        ),
        "best_state_is_safe": bool(
            np.min(prefix_wall_clearances) >= 0
            and np.min(prefix_ground_clearances)
            >= -env.geometry.ground_violation_tolerance
            and np.max(prefix_tip_speeds) <= env.maximum_tip_speed
        ),
        "continuation_termination_reason": recorded["termination_reason"],
        "search": search_rows,
        "trajectory": str(trajectory_path),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    normalization.close()
    env.close()


if __name__ == "__main__":
    main()
