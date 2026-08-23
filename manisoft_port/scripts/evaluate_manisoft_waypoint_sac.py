#!/usr/bin/env python
"""Evaluate ManiSoft waypoint SAC and export policy rollout transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from antmaze_ac.envs.manisoft_waypoint_sac_env import ManiSoftWaypointSACEnv
from antmaze_ac.envs.waypoint_paths import CURRICULUM_STAGES
from antmaze_ac.koopman.checkpoint import sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vec-normalize", default=None)
    parser.add_argument("--run-config", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional environment YAML override for cross-curriculum evaluation.",
    )
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--waypoint-maximum-extent", type=float, default=None)
    parser.add_argument(
        "--waypoint-segment-count-range",
        default=None,
        help="Inclusive segment-count pair, for example 2,3.",
    )
    parser.add_argument("--waypoint-maximum-turn-degrees", type=float, default=None)
    parser.add_argument(
        "--families",
        default="line,polyline,bezier,s_curve,reverse",
        help="Comma-separated deterministic evaluation cycle.",
    )
    parser.add_argument(
        "--speeds",
        default=None,
        help="Optional comma-separated speed cycle in m/s.",
    )
    parser.add_argument(
        "--entry-indices",
        default=None,
        help="Optional comma-separated entry-bank indices for a deterministic grid.",
    )
    parser.add_argument(
        "--warm-start-fractions",
        default=None,
        help="Optional comma-separated entry fractions for a deterministic grid.",
    )
    parser.add_argument(
        "--curriculum",
        choices=CURRICULUM_STAGES,
        default="mixed",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--cartesian-prior-weight",
        type=float,
        default=0.0,
        help="Blend weight for a calibrated Cartesian controller in [0, 1].",
    )
    parser.add_argument(
        "--cartesian-prior-proportional-gain",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--cartesian-prior-feedforward-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Keep only successful episodes in the exported NPZ.",
    )
    return parser.parse_args()


def _checkpoint_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    zipped = path.with_suffix(".zip")
    if zipped.is_file():
        return zipped
    raise FileNotFoundError(f"missing SAC checkpoint: {path}")


def _infer_file(explicit: str | None, model: Path, name: str) -> Path:
    path = Path(explicit).expanduser().resolve() if explicit else model.parent / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    return path


def _append_episode(
    destination: dict[str, list[np.ndarray | float | int | bool]],
    episode_rows: dict[str, list[np.ndarray | float | int | bool]],
) -> None:
    for key, values in episode_rows.items():
        destination[key].extend(values)


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0.0 <= args.cartesian_prior_weight <= 1.0:
        raise ValueError("cartesian-prior-weight must lie in [0, 1]")
    model_path = _checkpoint_path(args.model)
    run_config_path = _infer_file(args.run_config, model_path, "run_config.json")
    vecnormalize_path = _infer_file(
        args.vec_normalize, model_path, "vecnormalize.pkl"
    )
    runtime = json.loads(run_config_path.read_text(encoding="utf-8"))
    evaluation_config_path = None
    if args.config is None:
        environment_config = dict(runtime["resolved"]["environment"])
    else:
        evaluation_config_path = Path(args.config).expanduser().resolve()
        payload = yaml.safe_load(
            evaluation_config_path.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("environment"), dict
        ):
            raise ValueError("evaluation config must contain an environment mapping")
        environment_config = dict(payload["environment"])
        for key in ("entry_bank_path", "table_action_calibration_path"):
            value = environment_config.get(key)
            if value is None:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (evaluation_config_path.parent.parent / path).resolve()
            environment_config[key] = str(path)
    environment_config["curriculum"] = args.curriculum
    if args.episode_steps is not None:
        environment_config["episode_steps"] = int(args.episode_steps)
    if args.waypoint_maximum_extent is not None:
        if args.waypoint_maximum_extent <= 0:
            raise ValueError("waypoint-maximum-extent must be positive")
        environment_config["waypoint_maximum_extent"] = float(
            args.waypoint_maximum_extent
        )
    if args.waypoint_segment_count_range is not None:
        counts = [
            int(value)
            for value in args.waypoint_segment_count_range.split(",")
            if value.strip()
        ]
        if len(counts) != 2 or counts[0] < 1 or counts[0] > counts[1]:
            raise ValueError(
                "waypoint-segment-count-range must be an increasing positive pair"
            )
        environment_config["waypoint_segment_count_range"] = counts
        environment_config.pop("waypoint_segment_count_probabilities", None)
    if args.waypoint_maximum_turn_degrees is not None:
        if not 0.0 < args.waypoint_maximum_turn_degrees <= 180.0:
            raise ValueError(
                "waypoint-maximum-turn-degrees must lie in (0, 180]"
            )
        environment_config["waypoint_maximum_turn_degrees"] = float(
            args.waypoint_maximum_turn_degrees
        )
    scenario = Path(args.scenario or runtime["scenario"]).expanduser().resolve()
    if not scenario.is_file():
        raise FileNotFoundError(f"missing ManiSoft scenario: {scenario}")

    families = tuple(item.strip() for item in args.families.split(",") if item.strip())
    if not families:
        raise ValueError("at least one path family is required")

    entry_indices = (
        [int(item) for item in args.entry_indices.split(",") if item.strip()]
        if args.entry_indices is not None
        else []
    )
    warm_start_fractions = (
        [
            float(item)
            for item in args.warm_start_fractions.split(",")
            if item.strip()
        ]
        if args.warm_start_fractions is not None
        else []
    )
    if any(value < 0 for value in entry_indices):
        raise ValueError("--entry-indices must be non-negative")
    if any(not 0.0 <= value <= 1.0 for value in warm_start_fractions):
        raise ValueError("--warm-start-fractions must lie in [0, 1]")
    entry_grid = [
        (index, fraction)
        for fraction in (warm_start_fractions or [None])
        for index in (entry_indices or [None])
    ]

    from stable_baselines3 import SAC
    from stable_baselines3.common.save_util import load_from_pkl
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    base_env = ManiSoftWaypointSACEnv(scenario, **environment_config)
    monitored_env = Monitor(base_env)
    dummy = DummyVecEnv([lambda: monitored_env])
    normalizer = VecNormalize.load(str(vecnormalize_path), dummy)
    normalizer.training = False
    normalizer.norm_reward = False
    model = SAC.load(str(model_path), device=args.device)
    residual_config = runtime.get("frozen_base_residual")
    frozen_model = None
    frozen_normalizer = None
    residual_action_scale = 0.0
    if residual_config is not None:
        frozen_model_path = _checkpoint_path(
            residual_config["frozen_model_path"]
        )
        frozen_normalizer_path = Path(
            residual_config["frozen_vecnormalize_path"]
        ).expanduser().resolve()
        if not frozen_normalizer_path.is_file():
            raise FileNotFoundError(
                f"missing frozen VecNormalize state: {frozen_normalizer_path}"
            )
        residual_action_scale = float(
            residual_config["residual_action_scale"]
        )
        frozen_model = SAC.load(str(frozen_model_path), device="cpu")
        frozen_normalizer = load_from_pkl(str(frozen_normalizer_path))
        frozen_normalizer.training = False

    records: dict[str, list] = {
        key: []
        for key in (
            "observation",
            "normalized_observation",
            "next_observation",
            "normalized_next_observation",
            "frozen_base_action",
            "residual_policy_action",
            "raw_policy_action",
            "controller_prior_action",
            "policy_action",
            "applied_action",
            "applied_delta_action",
            "reward",
            "terminated",
            "truncated",
            "episode_id",
            "episode_start",
            "step_index",
            "tip_position",
            "target_tip",
            "lookahead_tip",
            "path_progress",
            "distance",
            "cross_track_distance",
            "desired_speed",
            "action_rate_clipped_ratio",
            "action_saturation_ratio",
            "table_violation",
            "terminal_timeout",
            "dynamics_violation",
            "waypoints_completed",
            "internal_waypoints_completed",
            "waypoint_passed",
        )
    }
    summaries: list[dict[str, Any]] = []
    if args.speeds is None:
        stage_low, stage_high = base_env.desired_speed_bounds(args.curriculum)
        speed_choices = np.asarray(
            [
                stage_low,
                np.sqrt(stage_low * stage_high),
                stage_high,
            ]
        )
    else:
        speed_choices = np.asarray(
            [float(item) for item in args.speeds.split(",") if item.strip()],
            dtype=np.float64,
        )
        if len(speed_choices) == 0 or np.any(speed_choices <= 0):
            raise ValueError("--speeds must contain positive values")
    kept_episode = 0
    steady_displacement = None
    if args.cartesian_prior_weight > 0:
        if (
            base_env.action_space.shape != (2,)
            or base_env.cartesian_command_distance is None
            or base_env.cartesian_action_leak <= 0
        ):
            raise ValueError(
                "Cartesian prior requires the leaky table_cartesian_delta action mode"
            )
        steady_displacement = (
            base_env.cartesian_command_distance
            * base_env.cartesian_action_step_scale
            / base_env.cartesian_action_leak
        )
    try:
        for episode in range(args.episodes):
            family = families[episode % len(families)]
            desired_speed = float(speed_choices[episode % len(speed_choices)])
            entry_index, warm_start_fraction = entry_grid[episode % len(entry_grid)]
            reset_options: dict[str, Any] = {
                "curriculum": args.curriculum,
                "path_family": family,
                "desired_speed": desired_speed,
            }
            if entry_index is not None:
                reset_options["entry_index"] = entry_index
            if warm_start_fraction is not None:
                reset_options["warm_start_fraction"] = warm_start_fraction
            observation, reset_info = monitored_env.reset(
                seed=args.seed + episode,
                options=reset_options,
            )
            start_tip = np.asarray(reset_info["tip_position"], dtype=np.float64)
            current_tip = start_tip.copy()
            episode_rows = {key: [] for key in records}
            distances: list[float] = []
            cross_track: list[float] = []
            episode_return = 0.0
            terminal_info = reset_info
            for step in range(int(environment_config["episode_steps"])):
                normalized = normalizer.normalize_obs(observation[None, :]).astype(
                    np.float32
                )
                requested_action, _ = model.predict(normalized, deterministic=True)
                residual_policy_action = np.asarray(
                    requested_action[0], dtype=np.float32
                )
                frozen_base_action = np.zeros_like(residual_policy_action)
                if frozen_model is not None:
                    frozen_normalized = frozen_normalizer.normalize_obs(
                        observation[None, :]
                    ).astype(np.float32)
                    frozen_prediction, _ = frozen_model.predict(
                        frozen_normalized, deterministic=True
                    )
                    frozen_base_action = np.asarray(
                        frozen_prediction[0], dtype=np.float32
                    )
                    raw_policy_action = np.clip(
                        frozen_base_action
                        + residual_action_scale * residual_policy_action,
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                else:
                    raw_policy_action = residual_policy_action
                controller_prior_action = np.zeros_like(raw_policy_action)
                if steady_displacement is not None:
                    target = np.asarray(base_env.current_target, dtype=np.float64)
                    feedforward = (target[:2] - start_tip[:2]) / steady_displacement
                    feedback = args.cartesian_prior_proportional_gain * (
                        target[:2] - current_tip[:2]
                    )
                    controller_prior_action = np.clip(
                        args.cartesian_prior_feedforward_scale * feedforward
                        + feedback,
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                requested_action = np.clip(
                    (1.0 - args.cartesian_prior_weight) * raw_policy_action
                    + args.cartesian_prior_weight * controller_prior_action,
                    -1.0,
                    1.0,
                ).astype(np.float32)
                next_observation, reward, terminated, truncated, info = monitored_env.step(
                    requested_action
                )
                normalized_next = normalizer.normalize_obs(
                    next_observation[None, :]
                ).astype(np.float32)
                values = {
                    "observation": observation.copy(),
                    "normalized_observation": normalized[0].copy(),
                    "next_observation": next_observation.copy(),
                    "normalized_next_observation": normalized_next[0].copy(),
                    "frozen_base_action": frozen_base_action.copy(),
                    "residual_policy_action": residual_policy_action.copy(),
                    "raw_policy_action": raw_policy_action.copy(),
                    "controller_prior_action": controller_prior_action.copy(),
                    "policy_action": requested_action.copy(),
                    "applied_action": np.asarray(info["applied_action"]).copy(),
                    "applied_delta_action": np.asarray(
                        info["applied_delta_action"]
                    ).copy(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "episode_id": kept_episode,
                    "episode_start": step == 0,
                    "step_index": step,
                    "tip_position": np.asarray(info["tip_position"]).copy(),
                    "target_tip": np.asarray(info["target_tip"]).copy(),
                    "lookahead_tip": np.asarray(info["lookahead_tip"]).copy(),
                    "path_progress": float(info["path_progress"]),
                    "distance": float(info["distance"]),
                    "cross_track_distance": float(info["cross_track_distance"]),
                    "desired_speed": float(info["desired_speed"]),
                    "action_rate_clipped_ratio": float(
                        info["action_rate_clipped_ratio"]
                    ),
                    "action_saturation_ratio": float(
                        info["action_saturation_ratio"]
                    ),
                    "table_violation": bool(info["table_violation"]),
                    "terminal_timeout": bool(info["terminal_timeout"]),
                    "dynamics_violation": bool(info["dynamics_violation"]),
                    "waypoints_completed": int(info["waypoints_completed"]),
                    "internal_waypoints_completed": int(
                        info["internal_waypoints_completed"]
                    ),
                    "waypoint_passed": bool(info["waypoint_passed"]),
                }
                for key, value in values.items():
                    episode_rows[key].append(value)
                distances.append(float(info["distance"]))
                cross_track.append(float(info["cross_track_distance"]))
                episode_return += float(reward)
                observation = next_observation
                terminal_info = info
                current_tip = np.asarray(info["tip_position"], dtype=np.float64)
                if terminated or truncated:
                    break
            success = bool(terminal_info.get("is_success", False))
            keep = success or not args.successful_only
            summary = {
                "episode": episode,
                "exported_episode": kept_episode if keep else None,
                "seed": args.seed + episode,
                "family": str(terminal_info.get("path_family", family)),
                "waypoint_count": int(len(reset_info["path_anchors"]) - 1),
                "path_length": float(reset_info["path_length"]),
                "desired_speed": desired_speed,
                "entry_index": reset_info.get("entry_index"),
                "entry_prefix_steps": int(reset_info.get("entry_prefix_steps", 0)),
                "warm_start_fraction": warm_start_fraction,
                "steps": len(distances),
                "return": episode_return,
                "success": success,
                "waypoints_completed": int(
                    terminal_info.get("waypoints_completed", int(success))
                ),
                "internal_waypoints_completed": int(
                    terminal_info.get("internal_waypoints_completed", 0)
                ),
                "table_violation": bool(terminal_info.get("table_violation", False)),
                "terminal_timeout": bool(
                    terminal_info.get("terminal_timeout", False)
                ),
                "dynamics_violation": bool(
                    terminal_info.get("dynamics_violation", False)
                ),
                "final_progress": float(terminal_info.get("path_progress", 0.0)),
                "mean_distance": float(np.mean(distances)),
                "rmse_distance": float(np.sqrt(np.mean(np.square(distances)))),
                "p95_distance": float(np.quantile(distances, 0.95)),
                "mean_cross_track_distance": float(np.mean(cross_track)),
                "path_anchors": np.asarray(reset_info["path_anchors"]).tolist(),
            }
            summaries.append(summary)
            if keep:
                _append_episode(records, episode_rows)
                kept_episode += 1
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        normalizer.close()

    output = Path(args.output).expanduser().resolve()
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(values) for key, values in records.items()}
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            control_hz=np.asarray(1.0 / base_env.control_dt, dtype=np.float32),
        )
    temporary.replace(output)
    successes = [row["success"] for row in summaries]
    report = {
        "kind": "manisoft_waypoint_sac_policy_rollouts",
        "dataset": str(output),
        "transition_count": len(records["reward"]),
        "evaluated_episodes": len(summaries),
        "exported_episodes": kept_episode,
        "successful_only": args.successful_only,
        "cartesian_prior": {
            "weight": args.cartesian_prior_weight,
            "proportional_gain": args.cartesian_prior_proportional_gain,
            "feedforward_scale": args.cartesian_prior_feedforward_scale,
            "steady_displacement_m": steady_displacement,
        },
        "success_rate": float(np.mean(successes)),
        "mean_rmse_distance": float(
            np.mean([row["rmse_distance"] for row in summaries])
        ),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "vecnormalize": str(vecnormalize_path),
        "vecnormalize_sha256": sha256(vecnormalize_path),
        "run_config": str(run_config_path),
        "frozen_base_residual": residual_config,
        "evaluation_config": (
            None
            if evaluation_config_path is None
            else str(evaluation_config_path)
        ),
        "scenario": str(scenario),
        "episodes": summaries,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
