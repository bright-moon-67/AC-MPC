#!/usr/bin/env python
"""Roll out and render the selected unified teacher-residual SAC policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.manisoft_teacher_tracking_sac_env import (
    ManiSoftTeacherTrackingSACEnv,
)
from manisoft.visualize.mujoco_viewer import update_softrobot_geoms
from render_manisoft_yz_plane_reachability import _display_model, _overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--teacher-episode", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-output", default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20290865)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _sac_badge(
    frame: np.ndarray,
    *,
    checkpoint_label: str,
    arch_height: float,
    tip_z: float,
    tip_speed: float,
    tip_x: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    right = frame.shape[1] - 25
    draw.rounded_rectangle(
        (right - 315, 94, right, 242),
        radius=10,
        fill=(18, 78, 132, 224),
    )
    draw.text(
        (right - 295, 104),
        f"UNIFIED RESIDUAL SAC · {checkpoint_label}",
        font=_font(16),
        fill="white",
    )
    arch_text = (
        "ARCH near wall: --"
        if not np.isfinite(arch_height)
        else f"ARCH near wall: {1000 * arch_height:5.1f} mm / 300 mm"
    )
    draw.text(
        (right - 295, 134),
        arch_text,
        font=_font(14),
        fill=(142, 250, 175) if arch_height >= 0.30 else (248, 214, 126),
    )
    draw.text(
        (right - 295, 160),
        f"TIP z: {1000 * tip_z:5.1f} mm",
        font=_font(14),
        fill=(142, 250, 175) if tip_z <= 0.18 else (248, 214, 126),
    )
    draw.text(
        (right - 295, 186),
        f"TIP speed: {tip_speed:5.3f} m/s",
        font=_font(14),
        fill=(142, 250, 175) if tip_speed <= 0.17 else (248, 214, 126),
    )
    draw.text(
        (right - 295, 212),
        f"SIGNED TIP x: {1000 * tip_x:+6.1f} mm",
        font=_font(14),
        fill=(142, 250, 175) if tip_x < 0 else (248, 214, 126),
    )
    return np.asarray(image)


def _rollout(args: argparse.Namespace, paths: dict[str, Path]):
    from stable_baselines3 import SAC
    from stable_baselines3.common.save_util import load_from_pkl

    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    environment = dict(payload["environment"])
    with np.load(paths["teacher_episode"], allow_pickle=False) as archive:
        transition_count = int(np.asarray(archive["actions"]).shape[0])
        stage_ids = np.asarray(archive["stage_ids"], dtype=np.int8)
    environment.update(
        {
            "reset_start_mode": "upright",
            "upright_start_probability": 1.0,
            "episode_steps": transition_count + 50,
        }
    )
    env = ManiSoftTeacherTrackingSACEnv(
        paths["scenario"],
        task_config_path=paths["task_config"],
        teacher_episode_path=paths["teacher_episode"],
        **environment,
    )
    model = SAC.load(paths["model"], device=args.device, print_system_info=False)
    normalizer = load_from_pkl(paths["vecnormalize"])
    observation, info = env.reset(seed=args.seed, options={"start_index": 0})
    rod = env.sim._backend._softrobot
    nodes = [rod.position_collection.T.astype(np.float64, copy=True)]
    residual_actions: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    fractions = [float(info["distal_crossed_fraction"])]
    wall_clearances = [float(info["wall_clearance"])]
    ground_clearances = [float(info["ground_clearance"])]
    plane_distances = [float(info["target_plane_distance"])]
    node_errors = [float(info["node_tracking_rmse"])]
    tip_errors = [float(info["tip_tracking_error"])]
    arch_heights = [float(info["arch_height"])]
    tip_speeds = [float(info["tip_speed"])]
    rewards: list[float] = []
    final_info = info
    while True:
        normalized = observation[None].astype(np.float32)
        if bool(normalizer.norm_obs):
            normalized = normalizer.normalize_obs(normalized)
        action, _ = model.predict(normalized, deterministic=True)
        observation, reward, terminated, truncated, final_info = env.step(action[0])
        nodes.append(rod.position_collection.T.astype(np.float64, copy=True))
        residual_actions.append(np.asarray(action[0], dtype=np.float32).copy())
        applied_actions.append(
            np.asarray(final_info["applied_action"], dtype=np.float32).copy()
        )
        fractions.append(float(final_info["distal_crossed_fraction"]))
        wall_clearances.append(float(final_info["wall_clearance"]))
        ground_clearances.append(float(final_info["ground_clearance"]))
        plane_distances.append(float(final_info["target_plane_distance"]))
        node_errors.append(float(final_info["node_tracking_rmse"]))
        tip_errors.append(float(final_info["tip_tracking_error"]))
        arch_heights.append(float(final_info["arch_height"]))
        tip_speeds.append(float(final_info["tip_speed"]))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    env.close()
    if not final_info.get("is_success", False):
        raise RuntimeError(
            "selected SAC replay failed: "
            f"{final_info.get('termination_reason')}, "
            f"progress={final_info.get('reference_progress')}"
        )
    arrays = {
        "control_dt": np.asarray(0.02, dtype=np.float64),
        "stage_ids": stage_ids[: len(nodes)],
        "node_positions": np.asarray(nodes, dtype=np.float64),
        "residual_actions": np.asarray(residual_actions, dtype=np.float32),
        "applied_actions": np.asarray(applied_actions, dtype=np.float32),
        "distal_crossed_fractions": np.asarray(fractions, dtype=np.float32),
        "wall_clearances": np.asarray(wall_clearances, dtype=np.float32),
        "ground_clearances": np.asarray(ground_clearances, dtype=np.float32),
        "plane_distances": np.asarray(plane_distances, dtype=np.float32),
        "node_tracking_errors": np.asarray(node_errors, dtype=np.float32),
        "tip_tracking_errors": np.asarray(tip_errors, dtype=np.float32),
        "arch_heights": np.asarray(arch_heights, dtype=np.float32),
        "tip_speeds": np.asarray(tip_speeds, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
    }
    verification = {
        "is_success": True,
        "termination_reason": final_info["termination_reason"],
        "control_steps": len(residual_actions),
        "final_distal_crossed_fraction": float(fractions[-1]),
        "final_target_plane_distance_m": float(plane_distances[-1]),
        "final_tip_speed_mps": float(final_info["tip_speed"]),
        "maximum_tip_speed_mps": float(np.max(tip_speeds)),
        "final_tip_xyz_m": arrays["node_positions"][-1, -1].astype(float).tolist(),
        "maximum_node_tracking_rmse_m": float(np.max(node_errors)),
        "maximum_tip_tracking_error_m": float(np.max(tip_errors)),
        "minimum_wall_clearance_m": float(np.min(wall_clearances)),
        "minimum_ground_clearance_m": float(np.min(ground_clearances)),
        "maximum_normalized_residual_action": float(
            np.max(np.abs(arrays["residual_actions"]))
        ),
        "return": float(np.sum(rewards)),
        "final_arch_height_m": float(arch_heights[-1]),
        "minimum_enforced_arch_height_m": float(
            np.nanmin(
                np.asarray(arch_heights)[
                    int(np.ceil(env.arch_enforcement_start_progress * transition_count)) :
                ]
            )
        ),
    }
    return arrays, verification


def _render(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    geometry: WallRouteGeometry,
) -> tuple[Path, Path, float]:
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_name(output.stem + "_preview.png")
    model, data = _display_model(geometry)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.43, 0.38]
    camera.distance = 1.62
    camera.azimuth = 143.0
    camera.elevation = -24.0

    nodes = arrays["node_positions"]
    control_dt = float(arrays["control_dt"])
    start_hold, end_hold = 0.75, 1.50
    motion_duration = (len(nodes) - 1) * control_dt / args.playback_speed
    duration = start_hold + motion_duration + end_hold
    frame_count = int(round(duration * args.fps)) + 1
    times = np.arange(frame_count, dtype=np.float64) / args.fps
    indices = np.rint(
        np.clip(
            (times - start_hold) * args.playback_speed / control_dt,
            0,
            len(nodes) - 1,
        )
    ).astype(np.int64)
    stage_names = {
        0: "unified SAC / upright approach and side bypass",
        1: "unified SAC / smooth activation transition",
        2: "unified SAC / increase distal crossing",
        3: "unified SAC / stabilize crossed body",
        4: "unified SAC / return toward yz plane",
        5: "unified SAC / damped terminal approach",
    }
    match = re.search(r"_(\d+)_steps", Path(args.model).stem)
    checkpoint_label = (
        f"{int(match.group(1)) // 1000}k" if match is not None else "selected"
    )
    writer = imageio.get_writer(
        output,
        format="FFMPEG",
        mode="I",
        fps=args.fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=[
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    threshold = geometry.wall_maximum[1] + geometry.postwall_y_margin
    final_frame = None
    try:
        for frame_index, (time_value, index) in enumerate(zip(times, indices)):
            phase = frame_index / max(frame_count - 1, 1)
            camera.azimuth = 143.0 - 18.0 * phase
            renderer.update_scene(data, camera=camera)
            current = nodes[index]
            update_softrobot_geoms(
                renderer.scene,
                current,
                radius=geometry.arm_radius,
                rgba=(0.20, 0.43, 0.76, 1.0),
                clear_existing=False,
            )
            checked = current[geometry.mounting_exempt_nodes :]
            distal_count = 0
            for value in (checked[:, 1] >= threshold)[::-1]:
                if not value:
                    break
                distal_count += 1
            if distal_count >= 2:
                update_softrobot_geoms(
                    renderer.scene,
                    checked[-distal_count:],
                    radius=geometry.arm_radius + 0.001,
                    rgba=(0.10, 0.84, 0.31, 1.0),
                    clear_existing=False,
                )
            rendered = renderer.render()
            success_frame = bool(
                index == len(nodes) - 1
                and time_value >= start_hold + motion_duration - 0.04
            )
            rendered = _overlay(
                rendered,
                {
                    "plane_distance": float(arrays["plane_distances"][index]),
                    "fraction": float(arrays["distal_crossed_fractions"][index]),
                    "wall_clearance": float(arrays["wall_clearances"][index]),
                    "stage": stage_names[int(arrays["stage_ids"][index])],
                },
                success_frame,
            )
            rendered = _sac_badge(
                rendered,
                checkpoint_label=checkpoint_label,
                arch_height=float(arrays["arch_heights"][index]),
                tip_z=float(current[-1, 2]),
                tip_speed=float(arrays["tip_speeds"][index]),
                tip_x=float(current[-1, 0]),
            )
            writer.append_data(rendered)
            final_frame = rendered
    finally:
        renderer.close()
        writer.close()
    if final_frame is None:
        raise RuntimeError("renderer produced no frames")
    imageio.imwrite(preview, final_frame)
    return output, preview, duration


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.fps) <= 0 or args.playback_speed <= 0:
        raise ValueError("render dimensions, fps and playback speed must be positive")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in (
            ("scenario", args.scenario),
            ("task_config", args.task_config),
            ("teacher_episode", args.teacher_episode),
            ("config", args.config),
            ("model", args.model),
            ("vecnormalize", args.vecnormalize),
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    trajectory = (
        output.with_name(output.stem + "_trajectory.npz")
        if args.trajectory_output is None
        else Path(args.trajectory_output).expanduser().resolve()
    )
    if trajectory.exists():
        raise FileExistsError(trajectory)
    geometry = WallRouteGeometry.from_dict(
        yaml.safe_load(paths["task_config"].read_text(encoding="utf-8"))["task"]
    )
    arrays, verification = _rollout(args, paths)
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    with trajectory.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    video, preview, duration = _render(args, arrays, geometry)
    result = {
        "kind": "manisoft_unified_teacher_residual_sac_replay",
        "model": str(paths["model"]),
        "model_sha256": _sha256(paths["model"]),
        "vecnormalize": str(paths["vecnormalize"]),
        "vecnormalize_sha256": _sha256(paths["vecnormalize"]),
        "teacher_episode": str(paths["teacher_episode"]),
        "teacher_episode_sha256": _sha256(paths["teacher_episode"]),
        "trajectory": str(trajectory),
        "trajectory_sha256": _sha256(trajectory),
        "video": str(video),
        "video_sha256": _sha256(video),
        "preview": str(preview),
        "rendered_duration_seconds": duration,
        "playback_speed": args.playback_speed,
        "verification": verification,
    }
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
