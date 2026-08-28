#!/usr/bin/env python
"""Independently replay, verify and render a smooth wall teacher episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml

from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.manisoft_wall_crossing_sac_env import wall_crossing_metrics
from manisoft.visualize.mujoco_viewer import update_softrobot_geoms
from render_manisoft_yz_plane_reachability import _display_model, _overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(archive, key: str):
    return np.asarray(archive[key]).reshape(()).item()


def _load_episode(path: Path, scenario: Path, task_config: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if int(_scalar(archive, "schema_version")) != 1 or str(
            _scalar(archive, "kind")
        ) != "manisoft_smooth_wall_teacher_episode":
            raise ValueError("unsupported smooth teacher episode")
        if str(_scalar(archive, "scenario_sha256")) != _sha256(scenario):
            raise ValueError("teacher episode scenario hash mismatch")
        if str(_scalar(archive, "task_config_sha256")) != _sha256(task_config):
            raise ValueError("teacher episode task hash mismatch")
        result = {key: np.asarray(archive[key]) for key in archive.files}
    actions = np.asarray(result["actions"], dtype=np.float32)
    nodes = np.asarray(result["node_positions"], dtype=np.float64)
    if actions.shape != (len(nodes) - 1, 18):
        raise ValueError("teacher actions and node states are not aligned")
    return result


def _replay(
    episode: dict,
    scenario: Path,
    geometry: WallRouteGeometry,
) -> tuple[np.ndarray, dict]:
    actions = np.asarray(episode["actions"], dtype=np.float32)
    expected = np.asarray(episode["node_positions"], dtype=np.float64)
    seed = int(np.asarray(episode["episode_seed"]).reshape(()))
    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=geometry.target,
        episode_steps=len(actions),
        absolute_action_limit=0.60,
        muscle_torque_scale=45.0,
    )
    env.reset(seed=seed)
    rod = env.sim._backend._softrobot
    replayed = [rod.position_collection.T.astype(np.float64, copy=True)]
    maximum_error = float(np.max(np.abs(replayed[0] - expected[0])))
    minimum_wall_clearance = geometry.whole_arm_wall_clearance(replayed[0])
    minimum_ground_clearance = geometry.whole_arm_ground_clearance(replayed[0])
    maximum_tip_speed = float(np.linalg.norm(rod.velocity_collection.T[-1]))
    for step, action in enumerate(actions, start=1):
        env.muscle.set_activation(action.reshape(6, 3))
        env.sim.step_with_torque_callback(lambda lengths: env.muscle.evaluate(lengths))
        nodes = rod.position_collection.T.astype(np.float64, copy=True)
        replayed.append(nodes)
        maximum_error = max(
            maximum_error, float(np.max(np.abs(nodes - expected[step])))
        )
        metrics = wall_crossing_metrics(
            geometry, nodes, rod.velocity_collection.T, 1
        )
        minimum_wall_clearance = min(
            minimum_wall_clearance, metrics.wall_clearance
        )
        minimum_ground_clearance = min(
            minimum_ground_clearance, metrics.ground_clearance
        )
        maximum_tip_speed = max(maximum_tip_speed, metrics.tip_speed)
    final_metrics = wall_crossing_metrics(
        geometry,
        rod.position_collection.T,
        rod.velocity_collection.T,
        1,
    )
    env.close()
    if maximum_error > 5e-8:
        raise RuntimeError(
            f"independent teacher replay diverged by {maximum_error:.3e} m"
        )
    verification = {
        "maximum_node_position_error_m": maximum_error,
        "minimum_wall_clearance_m": float(minimum_wall_clearance),
        "minimum_ground_clearance_m": float(minimum_ground_clearance),
        "maximum_tip_speed_mps": float(maximum_tip_speed),
        "final_tip_x_m": float(final_metrics.tip_x),
        "final_tip_speed_mps": float(final_metrics.tip_speed),
        "final_distal_crossed_fraction": float(
            final_metrics.distal_crossed_fraction
        ),
    }
    return np.asarray(replayed), verification


def _render(
    args: argparse.Namespace,
    episode: dict,
    nodes: np.ndarray,
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

    control_dt = float(np.asarray(episode["control_dt"]).reshape(()))
    start_hold = 0.75
    end_hold = 1.50
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
    fractions = np.asarray(episode["distal_crossed_fractions"], dtype=np.float32)
    clearances = np.asarray(episode["wall_clearances"], dtype=np.float32)
    stages = np.asarray(episode["stage_ids"], dtype=np.int8)
    stage_names = {
        0: "upright approach / side bypass",
        1: "smooth activation transition",
        2: "increase distal crossing",
        3: "stabilize crossed body",
        4: "distal return toward yz plane",
        5: "damped terminal approach",
    }
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
    final_rendered = None
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
            crossed = bool(
                index == len(nodes) - 1
                and time_value >= start_hold + motion_duration - 0.04
            )
            rendered = _overlay(
                rendered,
                {
                    "plane_distance": abs(float(current[-1, 0])),
                    "fraction": float(fractions[index]),
                    "wall_clearance": float(clearances[index]),
                    "stage": stage_names[int(stages[index])],
                },
                crossed,
            )
            writer.append_data(rendered)
            final_rendered = rendered
    finally:
        renderer.close()
        writer.close()
    if final_rendered is None:
        raise RuntimeError("renderer produced no frames")
    imageio.imwrite(preview, final_rendered)
    return output, preview, duration


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.fps) <= 0 or args.playback_speed <= 0:
        raise ValueError("render dimensions, fps and playback speed must be positive")
    episode_path = Path(args.episode).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    task_config = Path(args.task_config).expanduser().resolve()
    for path in (episode_path, scenario, task_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    geometry = WallRouteGeometry.from_dict(
        yaml.safe_load(task_config.read_text())["task"]
    )
    episode = _load_episode(episode_path, scenario, task_config)
    replayed_nodes, verification = _replay(episode, scenario, geometry)
    output, preview, duration = _render(args, episode, replayed_nodes, geometry)
    result = {
        "kind": "manisoft_smooth_wall_teacher_replay",
        "episode": str(episode_path),
        "episode_sha256": _sha256(episode_path),
        "video": str(output),
        "video_sha256": _sha256(output),
        "preview": str(preview),
        "duration_seconds": duration,
        "playback_speed": args.playback_speed,
        "control_steps": len(replayed_nodes) - 1,
        "verification": verification,
    }
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
