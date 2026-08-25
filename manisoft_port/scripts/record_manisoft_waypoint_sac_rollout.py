#!/usr/bin/env python
"""Record one complete, renderable ManiSoft waypoint-SAC rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.envs.manisoft_waypoint_sac_env import ManiSoftWaypointSACEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vec-normalize", required=True)
    parser.add_argument(
        "--run-config",
        help=(
            "Training run_config.json. Required to reproduce a frozen-base "
            "residual checkpoint exactly."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/manisoft_waypoint_sac_table_long10cm.yaml",
    )
    parser.add_argument(
        "--scenario",
        default="/root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--point-count",
        type=int,
        choices=tuple(range(8, 13)),
        default=10,
        help="Number of straight segments after the restored table start.",
    )
    parser.add_argument("--desired-speed", type=float, default=0.012)
    parser.add_argument("--episode-steps", type=int, default=6500)
    parser.add_argument("--seed", type=int, default=997500)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--anchors-json",
        help=(
            "Optional JSON array of explicit [x, y, z] path anchors. The first "
            "anchor should match the restored tabletop tip; otherwise the "
            "restored tip is prepended automatically."
        ),
    )
    parser.add_argument(
        "--allow-pose-map-hull",
        action="store_true",
        help=(
            "Evaluation-only: allow interpolation anywhere inside the pose-map "
            "convex hull instead of rejecting long Delaunay simplices."
        ),
    )
    return parser.parse_args()


def _resolve_checkpoint(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    zipped = path.with_suffix(".zip")
    if zipped.is_file():
        return zipped
    raise FileNotFoundError(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_config(config_path: Path, point_count: int) -> dict:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("environment"), dict
    ):
        raise ValueError("config must contain an environment mapping")
    config = dict(payload["environment"])
    for key in (
        "entry_bank_path",
        "table_action_calibration_path",
        "table_equilibrium_path_bank_path",
        "table_pose_path_bank_path",
        "table_pose_map_path",
    ):
        value = config.get(key)
        if value is None:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (config_path.parent.parent / path).resolve()
        config[key] = str(path)
    config.update(
        {
            "curriculum": "table_waypoint_polyline",
            "episode_steps": 6500,
            "waypoint_segment_count_range": [point_count, point_count],
        }
    )
    config.pop("waypoint_segment_count_probabilities", None)
    return config


def main() -> None:
    args = parse_args()
    model_path = _resolve_checkpoint(args.model)
    vec_path = Path(args.vec_normalize).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    scenario_path = Path(args.scenario).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path in (vec_path, config_path, scenario_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    environment_config = _environment_config(config_path, args.point_count)
    environment_config["episode_steps"] = int(args.episode_steps)
    if args.allow_pose_map_hull:
        environment_config["pose_map_maximum_simplex_edge"] = None
    anchors = None
    if args.anchors_json is not None:
        anchors = np.asarray(json.loads(args.anchors_json), dtype=np.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 3 or len(anchors) < 2:
            raise ValueError("--anchors-json must contain at least two [x, y, z] rows")
        if not np.isfinite(anchors).all():
            raise ValueError("--anchors-json must contain only finite coordinates")

    from stable_baselines3 import SAC
    from stable_baselines3.common.save_util import load_from_pkl
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    base_env = ManiSoftWaypointSACEnv(scenario_path, **environment_config)
    monitored_env = Monitor(base_env)
    dummy = DummyVecEnv([lambda: monitored_env])
    normalizer = VecNormalize.load(str(vec_path), dummy)
    normalizer.training = False
    normalizer.norm_reward = False
    model = SAC.load(str(model_path), device=args.device)
    runtime = None
    residual_config = None
    frozen_model = None
    frozen_normalizer = None
    residual_action_scale = 0.0
    if args.run_config is not None:
        run_config_path = Path(args.run_config).expanduser().resolve()
        if not run_config_path.is_file():
            raise FileNotFoundError(run_config_path)
        runtime = json.loads(run_config_path.read_text(encoding="utf-8"))
        residual_config = runtime.get("frozen_base_residual")
        if residual_config is not None:
            frozen_model_path = _resolve_checkpoint(
                residual_config["frozen_model_path"]
            )
            frozen_normalizer_path = Path(
                residual_config["frozen_vecnormalize_path"]
            ).expanduser().resolve()
            if not frozen_normalizer_path.is_file():
                raise FileNotFoundError(frozen_normalizer_path)
            frozen_model = SAC.load(str(frozen_model_path), device="cpu")
            frozen_normalizer = load_from_pkl(str(frozen_normalizer_path))
            frozen_normalizer.training = False
            residual_action_scale = float(
                residual_config["residual_action_scale"]
            )

    rows: dict[str, list] = {
        name: []
        for name in (
            "softrobot_positions",
            "softrobot_directors",
            "tip_position",
            "tip_orientation_error_degrees",
            "target_tip",
            "lookahead_tip",
            "path_progress",
            "active_waypoint_index",
            "waypoints_completed",
            "distance",
            "cross_track_distance",
            "whole_arm_table_clearance",
            "tip_speed",
            "frozen_base_action",
            "residual_policy_action",
            "requested_action",
            "applied_action",
            "reward",
        )
    }

    def append_frame(
        info: dict,
        requested_action: np.ndarray,
        reward: float,
        frozen_base_action: np.ndarray | None = None,
        residual_policy_action: np.ndarray | None = None,
    ) -> None:
        frame = base_env.trajectory_frame()
        for key in (
            "softrobot_positions",
            "softrobot_directors",
            "tip_position",
            "target_tip",
            "lookahead_tip",
            "path_progress",
        ):
            rows[key].append(frame[key])
        rows["tip_orientation_error_degrees"].append(
            float(frame["tip_orientation_error_degrees"])
        )
        rows["active_waypoint_index"].append(
            int(info.get("active_waypoint_index", 1))
        )
        rows["waypoints_completed"].append(
            int(info.get("waypoints_completed", 0))
        )
        rows["distance"].append(float(info.get("distance", 0.0)))
        rows["cross_track_distance"].append(
            float(info.get("cross_track_distance", 0.0))
        )
        rows["whole_arm_table_clearance"].append(
            float(info.get("whole_arm_table_clearance", np.nan))
        )
        rows["tip_speed"].append(float(info.get("tip_speed", 0.0)))
        rows["frozen_base_action"].append(
            np.zeros_like(requested_action)
            if frozen_base_action is None
            else frozen_base_action.copy()
        )
        rows["residual_policy_action"].append(
            requested_action.copy()
            if residual_policy_action is None
            else residual_policy_action.copy()
        )
        rows["requested_action"].append(requested_action.copy())
        rows["applied_action"].append(
            np.asarray(
                info.get("applied_action", base_env.previous_action),
                dtype=np.float32,
            ).copy()
        )
        rows["reward"].append(float(reward))

    terminal_info: dict = {}
    episode_return = 0.0
    try:
        reset_options = {"desired_speed": args.desired_speed}
        if anchors is not None:
            reset_options["anchors"] = anchors
            reset_options["curriculum"] = "table_waypoint_polyline"
        observation, reset_info = base_env.reset(
            seed=args.seed,
            options=reset_options,
        )
        initial_info = base_env._info(base_env.last_physical_state)
        append_frame(
            initial_info,
            np.zeros(base_env.action_space.shape, dtype=np.float32),
            0.0,
        )
        for step in range(args.episode_steps):
            normalized = normalizer.normalize_obs(
                observation[None, :]
            ).astype(np.float32)
            action, _ = model.predict(normalized, deterministic=True)
            residual_policy_action = np.asarray(
                action, dtype=np.float32
            ).reshape(-1)
            frozen_base_action = np.zeros_like(residual_policy_action)
            if frozen_model is not None:
                frozen_normalized = frozen_normalizer.normalize_obs(
                    observation[None, :]
                ).astype(np.float32)
                frozen_action, _ = frozen_model.predict(
                    frozen_normalized, deterministic=True
                )
                frozen_base_action = np.asarray(
                    frozen_action, dtype=np.float32
                ).reshape(-1)
                requested = np.clip(
                    frozen_base_action
                    + residual_action_scale * residual_policy_action,
                    -1.0,
                    1.0,
                ).astype(np.float32)
            else:
                requested = residual_policy_action
            observation, reward, terminated, truncated, terminal_info = (
                base_env.step(requested)
            )
            append_frame(
                terminal_info,
                requested,
                float(reward),
                frozen_base_action,
                residual_policy_action,
            )
            episode_return += float(reward)
            if terminated or truncated:
                break
    finally:
        normalizer.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    arrays = {name: np.asarray(values) for name, values in rows.items()}
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            path_anchors=np.asarray(reset_info["path_anchors"], dtype=np.float32),
            path_points=np.asarray(base_env.path.points, dtype=np.float32),
            path_segment_lengths=np.asarray(
                reset_info["path_segment_lengths"], dtype=np.float32
            ),
            table_x_bounds=np.asarray(base_env.table_x_bounds, dtype=np.float32),
            table_y_bounds=np.asarray(base_env.table_y_bounds, dtype=np.float32),
            table_surface_z=np.asarray(base_env.table_surface_z, dtype=np.float32),
            control_hz=np.asarray(1.0 / base_env.control_dt, dtype=np.float32),
            success=np.asarray(
                bool(terminal_info.get("is_success", False)), dtype=np.bool_
            ),
        )
    temporary.replace(output_path)

    report = {
        "kind": "manisoft_waypoint_sac_renderable_rollout",
        "trajectory": str(output_path),
        "frames": len(rows["tip_position"]),
        "steps": len(rows["tip_position"]) - 1,
        "simulation_seconds": (len(rows["tip_position"]) - 1)
        * base_env.control_dt,
        "point_count": int(len(reset_info["path_anchors"]) - 1),
        "path_anchors_m": np.asarray(reset_info["path_anchors"]).tolist(),
        "path_segment_lengths_m": np.asarray(
            reset_info["path_segment_lengths"]
        ).tolist(),
        "success": bool(terminal_info.get("is_success", False)),
        "return": episode_return,
        "final_distance_m": float(terminal_info.get("final_distance", np.nan)),
        "mean_cross_track_distance_m": float(
            np.mean(arrays["cross_track_distance"])
        ),
        "minimum_whole_arm_table_clearance_m": float(
            np.nanmin(arrays["whole_arm_table_clearance"])
        ),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "vecnormalize": str(vec_path),
        "vecnormalize_sha256": _sha256(vec_path),
        "run_config": (
            None
            if args.run_config is None
            else str(Path(args.run_config).expanduser().resolve())
        ),
        "frozen_base_residual": residual_config,
        "config": str(config_path),
        "scenario": str(scenario_path),
        "seed": args.seed,
        "allow_pose_map_hull": bool(args.allow_pose_map_hull),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["success"]:
        raise RuntimeError("recorded rollout did not complete successfully")


if __name__ == "__main__":
    main()
