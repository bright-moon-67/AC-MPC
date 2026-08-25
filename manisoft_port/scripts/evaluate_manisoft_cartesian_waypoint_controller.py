#!/usr/bin/env python
"""Evaluate a calibrated Cartesian feedback controller on waypoint paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.envs.manisoft_waypoint_sac_env import ManiSoftWaypointSACEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=865000)
    parser.add_argument("--proportional-gain", type=float, default=20.0)
    parser.add_argument("--feedforward-scale", type=float, default=1.0)
    parser.add_argument("--waypoint-segment-count-range", default="3,3")
    parser.add_argument("--waypoint-segment-length-range", default=None)
    parser.add_argument("--waypoint-maximum-extent", type=float, default=None)
    parser.add_argument("--waypoint-minimum-turn-degrees", type=float, default=None)
    parser.add_argument("--waypoint-maximum-turn-degrees", type=float, default=60.0)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument(
        "--environment-prior",
        action="store_true",
        help="Drive with the environment's Cartesian prior and zero SAC residual.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    config_path = Path(args.config).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    environment_config = dict(payload["environment"])
    for key in ("entry_bank_path", "table_action_calibration_path"):
        value = environment_config.get(key)
        if value is not None:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (config_path.parent.parent / path).resolve()
            environment_config[key] = str(path)
    count_range = [
        int(value)
        for value in args.waypoint_segment_count_range.split(",")
        if value.strip()
    ]
    if len(count_range) != 2 or count_range[0] < 1 or count_range[0] > count_range[1]:
        raise ValueError("waypoint-segment-count-range must be an increasing pair")
    environment_config.update(
        {
            "curriculum": "table_waypoint_polyline",
            "waypoint_segment_count_range": count_range,
            "waypoint_maximum_turn_degrees": float(
                args.waypoint_maximum_turn_degrees
            ),
            "waypoint_single_line_probability": 0.0,
            "internal_waypoint_capture_radius": 0.010,
        }
    )
    if args.waypoint_segment_length_range is not None:
        length_range = [
            float(value)
            for value in args.waypoint_segment_length_range.split(",")
            if value.strip()
        ]
        if len(length_range) != 2 or length_range[0] <= 0 or length_range[0] > length_range[1]:
            raise ValueError("waypoint-segment-length-range must be an increasing pair")
        environment_config["waypoint_segment_length_range"] = length_range
    if args.waypoint_maximum_extent is not None:
        environment_config["waypoint_maximum_extent"] = float(
            args.waypoint_maximum_extent
        )
    if args.waypoint_minimum_turn_degrees is not None:
        environment_config["waypoint_minimum_turn_degrees"] = float(
            args.waypoint_minimum_turn_degrees
        )
    if args.episode_steps is not None:
        environment_config["episode_steps"] = int(args.episode_steps)
    if args.environment_prior:
        environment_config["cartesian_prior_weight"] = 1.0
        environment_config["cartesian_prior_residual_scale"] = 0.0
    environment_config.pop("waypoint_segment_count_probabilities", None)
    environment_config.pop("entry_sampling_weights", None)
    env = ManiSoftWaypointSACEnv(args.scenario, **environment_config)
    if env.cartesian_command_distance is None:
        raise RuntimeError("controller requires table_cartesian_delta action mode")
    steady_displacement = (
        env.cartesian_command_distance
        * env.cartesian_action_step_scale
        / env.cartesian_action_leak
    )
    speed_low, speed_high = env.desired_speed_bounds("table_waypoint_polyline")
    speeds = (speed_low, np.sqrt(speed_low * speed_high), speed_high)
    summaries = []
    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset(
                seed=args.seed + episode,
                options={
                    "curriculum": "table_waypoint_polyline",
                    "path_family": "waypoint_polyline",
                    "desired_speed": float(speeds[episode % len(speeds)]),
                },
            )
            del observation
            start_tip = np.asarray(reset_info["tip_position"], dtype=np.float64)
            terminal_info = reset_info
            episode_return = 0.0
            distances = []
            for step in range(int(environment_config["episode_steps"])):
                tip = np.asarray(terminal_info["tip_position"], dtype=np.float64)
                target = np.asarray(env.current_target, dtype=np.float64)
                feedforward = (target[:2] - start_tip[:2]) / steady_displacement
                feedback = args.proportional_gain * (target[:2] - tip[:2])
                action = (
                    np.zeros(2, dtype=np.float32)
                    if args.environment_prior
                    else np.clip(
                        args.feedforward_scale * feedforward + feedback,
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                )
                _, reward, terminated, truncated, terminal_info = env.step(action)
                episode_return += float(reward)
                distances.append(float(terminal_info["distance"]))
                if terminated or truncated:
                    break
            summaries.append(
                {
                    "seed": args.seed + episode,
                    "entry_index": terminal_info.get("entry_index"),
                    "desired_speed": float(terminal_info["desired_speed"]),
                    "success": bool(terminal_info["is_success"]),
                    "steps": step + 1,
                    "return": episode_return,
                    "final_progress": float(terminal_info["path_progress"]),
                    "internal_waypoints_completed": int(
                        terminal_info["internal_waypoints_completed"]
                    ),
                    "rmse_distance": float(
                        np.sqrt(np.mean(np.square(distances)))
                    ),
                    "table_violation": bool(terminal_info["table_violation"]),
                    "dynamics_violation": bool(
                        terminal_info["dynamics_violation"]
                    ),
                }
            )
    finally:
        env.close()
    output = {
        "kind": "manisoft_calibrated_cartesian_waypoint_controller_eval",
        "controller": {
            "proportional_gain": args.proportional_gain,
            "feedforward_scale": args.feedforward_scale,
            "steady_displacement_m": steady_displacement,
        },
        "episodes": summaries,
        "success_rate": float(np.mean([row["success"] for row in summaries])),
        "mean_final_progress": float(
            np.mean([row["final_progress"] for row in summaries])
        ),
        "mean_rmse_distance": float(
            np.mean([row["rmse_distance"] for row in summaries])
        ),
        "table_violations": int(sum(row["table_violation"] for row in summaries)),
        "dynamics_violations": int(
            sum(row["dynamics_violation"] for row in summaries)
        ),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
