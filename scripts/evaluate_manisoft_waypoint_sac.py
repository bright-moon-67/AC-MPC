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
    parser.add_argument(
        "--anchors-panel",
        default=None,
        help=(
            "Optional JSON fixed panel. It must contain a list (or a cases "
            "list) of objects with name and anchors=[[x,y,z],...]. Optional "
            "per-case fields are desired_speed, entry_index, and seed."
        ),
    )
    parser.add_argument(
        "--anchors-panel-case",
        default=None,
        help=(
            "Evaluate only the named case from --anchors-panel. This is useful "
            "for process-isolated parallel fixed-panel evaluation."
        ),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument(
        "--internal-waypoint-capture-radius",
        type=float,
        default=None,
        help="Optional internal-waypoint precision override in metres.",
    )
    parser.add_argument(
        "--target-lead-distance",
        type=float,
        default=None,
        help="Optional target lead-distance override in metres.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Optional terminal success radius override in metres.",
    )
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
        "--prior-only",
        action="store_true",
        help="Evaluate a zero SAC residual around the configured environment prior.",
    )
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
        "--cartesian-prior-internal-waypoints-only",
        action="store_true",
        help=(
            "Disable the externally blended Cartesian prior after all "
            "internal waypoints have been captured."
        ),
    )
    parser.add_argument(
        "--cartesian-prior-residual-scale",
        type=float,
        default=None,
        help="Override bounded SAC residual authority around the Cartesian prior.",
    )
    parser.add_argument(
        "--equilibrium-path-prior-weight",
        type=float,
        default=None,
        help="Override the certified equilibrium-path prior blend weight.",
    )
    parser.add_argument(
        "--equilibrium-path-residual-scale",
        type=float,
        default=None,
        help="Override SAC residual authority around the equilibrium-path prior.",
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


def _load_anchors_panel(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    panel_path = Path(path).expanduser().resolve()
    payload = json.loads(panel_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("anchors panel must contain a non-empty cases list")
    resolved: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"anchors panel case {index} must be an object")
        # Keep JSON coordinates at float64 precision. Certified panels often
        # use pose-map vertices or boundary chords; an eager float32 round trip
        # can move a valid boundary point into an adjacent rejected simplex.
        anchors = np.asarray(case.get("anchors"), dtype=np.float64)
        if anchors.ndim != 2 or anchors.shape[1] != 3 or len(anchors) < 2:
            raise ValueError(
                f"anchors panel case {index} must contain at least two [x,y,z] rows"
            )
        if not np.isfinite(anchors).all():
            raise ValueError(f"anchors panel case {index} contains non-finite values")
        desired_speed = case.get("desired_speed")
        if desired_speed is not None and float(desired_speed) <= 0:
            raise ValueError(
                f"anchors panel case {index} desired_speed must be positive"
            )
        entry_index = case.get("entry_index", 0)
        if int(entry_index) < 0:
            raise ValueError(
                f"anchors panel case {index} entry_index must be non-negative"
            )
        resolved.append(
            {
                "name": str(case.get("name", f"case_{index:02d}")),
                "anchors": anchors,
                "desired_speed": (
                    None if desired_speed is None else float(desired_speed)
                ),
                "entry_index": int(entry_index),
                "seed": None if case.get("seed") is None else int(case["seed"]),
            }
        )
    return resolved


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
        for key in (
            "entry_bank_path",
            "table_action_calibration_path",
            "table_equilibrium_path_bank_path",
            "table_pose_path_bank_path",
            "table_pose_map_path",
        ):
            value = environment_config.get(key)
            if value is None:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (evaluation_config_path.parent.parent / path).resolve()
            environment_config[key] = str(path)
    if args.equilibrium_path_prior_weight is not None:
        if not 0.0 <= args.equilibrium_path_prior_weight <= 1.0:
            raise ValueError("equilibrium-path-prior-weight must lie in [0, 1]")
        environment_config["equilibrium_path_prior_weight"] = float(
            args.equilibrium_path_prior_weight
        )
    if args.cartesian_prior_residual_scale is not None:
        if not 0.0 <= args.cartesian_prior_residual_scale <= 1.0:
            raise ValueError("cartesian-prior-residual-scale must lie in [0, 1]")
        environment_config["cartesian_prior_residual_scale"] = float(
            args.cartesian_prior_residual_scale
        )
    if args.equilibrium_path_residual_scale is not None:
        if not 0.0 <= args.equilibrium_path_residual_scale <= 1.0:
            raise ValueError("equilibrium-path-residual-scale must lie in [0, 1]")
        environment_config["equilibrium_path_residual_scale"] = float(
            args.equilibrium_path_residual_scale
        )
    environment_config["curriculum"] = args.curriculum
    if args.episode_steps is not None:
        environment_config["episode_steps"] = int(args.episode_steps)
    for argument_name, config_name in (
        ("internal_waypoint_capture_radius", "internal_waypoint_capture_radius"),
        ("target_lead_distance", "target_lead_distance"),
        ("success_threshold", "success_threshold"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            if value <= 0:
                raise ValueError(f"{argument_name.replace('_', '-')} must be positive")
            environment_config[config_name] = float(value)
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
    anchors_panel = _load_anchors_panel(args.anchors_panel)
    if args.anchors_panel_case is not None:
        if not anchors_panel:
            raise ValueError("--anchors-panel-case requires --anchors-panel")
        anchors_panel = [
            case for case in anchors_panel if case["name"] == args.anchors_panel_case
        ]
        if not anchors_panel:
            raise ValueError(
                f"unknown anchors panel case: {args.anchors_panel_case}"
            )

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
            "environment_controller_prior_action",
            "environment_blended_policy_action",
            "reference_policy_action",
            "effective_controller_prior_weight",
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
            "whole_arm_table_clearance",
            "table_violation",
            "terminal_timeout",
            "dynamics_violation",
            "waypoints_completed",
            "internal_waypoints_completed",
            "waypoint_passed",
            "active_waypoint_index",
            "active_waypoint_distance",
            "internal_waypoint_distance_delta",
            "normalized_internal_waypoint_progress",
            "normalized_internal_waypoint_capture_error",
            "tip_speed",
            "projected_speed",
            "geometric_path_end",
            "terminal_capture",
            "stalled",
            "path_progress_stalled",
            "waypoint_stalled",
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
        episode_count = len(anchors_panel) if anchors_panel else args.episodes
        for episode in range(episode_count):
            # A fixed custom panel must start every case from a genuinely
            # independent simulator. Reconstructing the soft-rod integrator
            # after a long prior rollout can leave sub-micrometre differences
            # at a certified pose-map boundary, which is enough to make a
            # boundary chord appear invalid and also weakens paired testing.
            if anchors_panel and episode > 0:
                monitored_env.close()
                base_env = ManiSoftWaypointSACEnv(scenario, **environment_config)
                monitored_env = Monitor(base_env)
            family = families[episode % len(families)]
            desired_speed = float(speed_choices[episode % len(speed_choices)])
            entry_index, warm_start_fraction = entry_grid[episode % len(entry_grid)]
            panel_case = anchors_panel[episode] if anchors_panel else None
            reset_options: dict[str, Any] = {
                "curriculum": args.curriculum,
                "path_family": family,
                "desired_speed": desired_speed,
            }
            episode_seed = args.seed + episode
            panel_name = None
            if panel_case is not None:
                panel_name = panel_case["name"]
                # Explicit tabletop coordinates always use the certified bent
                # entry snapshot. Leaving this at the evaluator's default
                # ``mixed`` curriculum can randomly choose an upright entry
                # mode and makes identical panels seed-dependent at reset.
                reset_options["curriculum"] = "table_waypoint_polyline"
                reset_options["anchors"] = panel_case["anchors"]
                entry_index = panel_case["entry_index"]
                if panel_case["desired_speed"] is not None:
                    desired_speed = panel_case["desired_speed"]
                    reset_options["desired_speed"] = desired_speed
                if panel_case["seed"] is not None:
                    episode_seed = panel_case["seed"]
            if entry_index is not None:
                reset_options["entry_index"] = entry_index
            if warm_start_fraction is not None:
                reset_options["warm_start_fraction"] = warm_start_fraction
            observation, reset_info = monitored_env.reset(
                seed=episode_seed,
                options=reset_options,
            )
            start_tip = np.asarray(reset_info["tip_position"], dtype=np.float64)
            current_tip = start_tip.copy()
            episode_rows = {key: [] for key in records}
            distances: list[float] = []
            cross_track: list[float] = []
            clearances: list[float] = []
            episode_return = 0.0
            terminal_info = reset_info
            for step in range(int(environment_config["episode_steps"])):
                normalized = normalizer.normalize_obs(observation[None, :]).astype(
                    np.float32
                )
                if args.prior_only:
                    residual_policy_action = np.zeros(
                        base_env.action_space.shape, dtype=np.float32
                    )
                else:
                    requested_action, _ = model.predict(
                        normalized, deterministic=True
                    )
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
                prior_active = not args.cartesian_prior_internal_waypoints_only or (
                    base_env.path is not None
                    and base_env.next_internal_waypoint_index
                    < len(base_env.path.anchors) - 1
                )
                if steady_displacement is not None and prior_active:
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
                effective_prior_weight = (
                    args.cartesian_prior_weight if prior_active else 0.0
                )
                requested_action = np.clip(
                    (1.0 - effective_prior_weight) * raw_policy_action
                    + effective_prior_weight * controller_prior_action,
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
                    "environment_controller_prior_action": np.asarray(
                        info["controller_prior_action"]
                    ).copy(),
                    "environment_blended_policy_action": np.asarray(
                        info["blended_policy_action"]
                    ).copy(),
                    "reference_policy_action": np.asarray(
                        info["reference_policy_action"]
                    ).copy(),
                    "effective_controller_prior_weight": float(
                        info["effective_controller_prior_weight"]
                    ),
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
                    "whole_arm_table_clearance": float(
                        info["whole_arm_table_clearance"]
                    ),
                    "table_violation": bool(info["table_violation"]),
                    "terminal_timeout": bool(info["terminal_timeout"]),
                    "dynamics_violation": bool(info["dynamics_violation"]),
                    "waypoints_completed": int(info["waypoints_completed"]),
                    "internal_waypoints_completed": int(
                        info["internal_waypoints_completed"]
                    ),
                    "waypoint_passed": bool(info["waypoint_passed"]),
                    "active_waypoint_index": int(info["active_waypoint_index"]),
                    "active_waypoint_distance": float(
                        info["active_waypoint_distance"]
                    ),
                    "internal_waypoint_distance_delta": float(
                        info["internal_waypoint_distance_delta"]
                    ),
                    "normalized_internal_waypoint_progress": float(
                        info["normalized_internal_waypoint_progress"]
                    ),
                    "normalized_internal_waypoint_capture_error": float(
                        info["normalized_internal_waypoint_capture_error"]
                    ),
                    "tip_speed": float(info.get("tip_speed", 0.0)),
                    "projected_speed": float(info.get("projected_speed", 0.0)),
                    "geometric_path_end": bool(
                        info.get("geometric_path_end", False)
                    ),
                    "terminal_capture": bool(
                        info.get("terminal_capture", False)
                    ),
                    "stalled": bool(info.get("stalled", False)),
                    "path_progress_stalled": bool(
                        info.get("path_progress_stalled", False)
                    ),
                    "waypoint_stalled": bool(
                        info.get("waypoint_stalled", False)
                    ),
                }
                for key, value in values.items():
                    episode_rows[key].append(value)
                distances.append(float(info["distance"]))
                cross_track.append(float(info["cross_track_distance"]))
                clearances.append(float(info["whole_arm_table_clearance"]))
                episode_return += float(reward)
                observation = next_observation
                terminal_info = info
                current_tip = np.asarray(info["tip_position"], dtype=np.float64)
                if terminated or truncated:
                    break
            success = bool(terminal_info.get("is_success", False))
            keep = success or not args.successful_only
            internal_completed = int(
                terminal_info.get("internal_waypoints_completed", 0)
            )
            internal_total = max(
                int(len(reset_info["path_anchors"]) - 2), 0
            )
            if success:
                failure_category = "success"
            elif terminal_info.get("dynamics_violation", False):
                failure_category = "dynamics_violation"
            elif terminal_info.get("table_violation", False):
                failure_category = "table_violation"
            elif terminal_info.get("terminal_timeout", False):
                failure_category = "terminal_timeout"
            elif internal_completed < internal_total:
                failure_category = f"before_internal_waypoint_{internal_completed + 1}"
            elif terminal_info.get("waypoint_stalled", False):
                failure_category = "waypoint_stall"
            elif terminal_info.get("path_progress_stalled", False):
                failure_category = "path_progress_stall"
            else:
                failure_category = "post_waypoint_incomplete"
            summary = {
                "episode": episode,
                "exported_episode": kept_episode if keep else None,
                "panel_case": panel_name,
                "seed": episode_seed,
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
                "failure_category": failure_category,
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
                "minimum_whole_arm_table_clearance": float(
                    np.min(clearances)
                ),
                "path_anchors": np.asarray(reset_info["path_anchors"]).tolist(),
                "path_generation_mode": str(
                    reset_info.get("path_generation_mode", "unknown")
                ),
                "path_segment_lengths": np.asarray(
                    reset_info.get("path_segment_lengths", [])
                ).tolist(),
                "path_turn_angles_degrees": np.asarray(
                    reset_info.get("path_turn_angles_degrees", [])
                ).tolist(),
            }
            summaries.append(summary)
            if keep:
                _append_episode(records, episode_rows)
                kept_episode += 1
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        monitored_env.close()
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
        "prior_only": bool(args.prior_only),
        "cartesian_prior": {
            "weight": args.cartesian_prior_weight,
            "proportional_gain": args.cartesian_prior_proportional_gain,
            "feedforward_scale": args.cartesian_prior_feedforward_scale,
            "internal_waypoints_only": (
                args.cartesian_prior_internal_waypoints_only
            ),
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
        "anchors_panel": (
            None
            if args.anchors_panel is None
            else str(Path(args.anchors_panel).expanduser().resolve())
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
