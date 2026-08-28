#!/usr/bin/env python
"""Render a certified approach prefix followed by the wall-crossing SAC policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# MuJoCo must select headless EGL before it is imported through ManiSoft.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from antmaze_ac.data.wall_crossing_snapshot_bank import (
    load_wall_crossing_snapshot_bank,
)
from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.manisoft_wall_crossing_sac_env import (
    ManiSoftWallCrossingSACEnv,
)
from manisoft.visualize.mujoco_viewer import (
    update_manisoft_softrobot_geoms,
    update_softrobot_geoms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--snapshot-index", type=int, default=49)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-output", default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--playback-speed", type=float, default=0.65)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _display_model(geometry: WallRouteGeometry):
    wall_center = 0.5 * (geometry.wall_minimum + geometry.wall_maximum)
    wall_half_size = 0.5 * (geometry.wall_maximum - geometry.wall_minimum)
    target = geometry.target
    xml = f"""
<mujoco model="wall_crossing_policy">
  <option timestep="0.0002"/>
  <statistic center="0 0.34 0.42" extent="1.15"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.75 0.75 0.75"
               specular="0.12 0.12 0.12"/>
    <rgba haze="0.90 0.93 0.97 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.72 0.79 0.88" rgb2="0.96 0.97 0.99"
             width="512" height="3072"/>
    <material name="floor_mat" rgba="0.82 0.84 0.86 1"/>
  </asset>
  <worldbody>
    <light pos="-0.8 -0.4 2.2" dir="0.3 0.4 -1"
           diffuse="0.85 0.85 0.85"/>
    <light pos="0.9 1.0 1.6" dir="-0.3 -0.5 -1"
           diffuse="0.45 0.48 0.55"/>
    <geom name="ground" type="plane" size="1.4 1.4 0.1"
          material="floor_mat"/>
    <geom name="wall" type="box"
          pos="{wall_center[0]} {wall_center[1]} {wall_center[2]}"
          size="{wall_half_size[0]} {wall_half_size[1]} {wall_half_size[2]}"
          rgba="0.88 0.23 0.14 0.78"/>
    <geom name="target" type="sphere"
          pos="{target[0]} {target[1]} {target[2]}"
          size="0.024" rgba="0.12 0.82 0.30 0.75"/>
    <geom name="base" type="cylinder" pos="0 0 0.018"
          size="0.075 0.018" rgba="0.16 0.19 0.24 1"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _distal_fraction(nodes: np.ndarray, geometry: WallRouteGeometry) -> float:
    checked = nodes[geometry.mounting_exempt_nodes :]
    threshold = geometry.wall_maximum[1] + geometry.postwall_y_margin
    count = 0
    for value in (checked[:, 1] >= threshold)[::-1]:
        if not value:
            break
        count += 1
    return float(count / len(checked))


def _font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _overlay(
    frame: np.ndarray,
    *,
    stage: int,
    fraction: float,
    wall_clearance: float,
    source_time: float,
    success: bool,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(25)
    body_font = _font(18)
    small_font = _font(15)
    draw.rounded_rectangle((20, 18, 530, 158), radius=12, fill=(8, 14, 24, 194))
    draw.text((38, 31), "ManiSoft: distal-body wall crossing", font=title_font, fill="white")
    stage_name = (
        "Candidate approach prefix (replay)"
        if stage == 0
        else "SAC policy control (20k checkpoint)"
    )
    stage_color = (225, 225, 225) if stage == 0 else (102, 210, 255)
    draw.text((38, 68), stage_name, font=body_font, fill=stage_color)
    draw.text(
        (38, 98),
        f"Contiguous distal body beyond wall: {100 * fraction:4.0f}% / 30%",
        font=body_font,
        fill=(136, 245, 159) if fraction >= 0.30 - 1e-8 else "white",
    )
    draw.text(
        (38, 127),
        f"Wall clearance: {1000 * wall_clearance:6.1f} mm    t={source_time:4.2f} s",
        font=small_font,
        fill=(224, 230, 238),
    )
    bar_left, bar_top, bar_width, bar_height = 38, 174, 360, 18
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_left + bar_width, bar_top + bar_height),
        radius=7,
        fill=(18, 25, 36, 210),
        outline=(220, 225, 232, 220),
        width=2,
    )
    filled = int(round(bar_width * min(fraction / 0.30, 1.0)))
    if filled:
        draw.rounded_rectangle(
            (bar_left, bar_top, bar_left + filled, bar_top + bar_height),
            radius=7,
            fill=(48, 203, 92, 235),
        )
    if success:
        draw.rounded_rectangle(
            (frame.shape[1] - 300, 26, frame.shape[1] - 24, 78),
            radius=12,
            fill=(20, 126, 58, 224),
        )
        draw.text(
            (frame.shape[1] - 280, 39),
            "DISTAL CROSSING SUCCESS",
            font=body_font,
            fill="white",
        )
    draw.rounded_rectangle(
        (20, frame.shape[0] - 68, 620, frame.shape[0] - 18),
        radius=10,
        fill=(8, 14, 24, 178),
    )
    draw.text(
        (37, frame.shape[0] - 55),
        "Red: virtual wall    Green: crossed distal arm    Green sphere: inactive target",
        font=small_font,
        fill=(238, 240, 244),
    )
    return np.asarray(image)


def _rollout(args: argparse.Namespace):
    run_path = Path(args.run_config).expanduser().resolve()
    run = json.loads(run_path.read_text(encoding="utf-8"))
    bank = load_wall_crossing_snapshot_bank(run["snapshot_bank"])
    index = int(args.snapshot_index)
    if not 0 <= index < bank.snapshot_count:
        raise ValueError("snapshot-index is outside the bank")
    source_episode = int(bank.source_episodes[index])
    source_frame = int(bank.source_frames[index])
    candidate_path = (
        Path(args.candidates).expanduser().resolve()
        / f"episode_{source_episode:05d}.npz"
    )
    with np.load(candidate_path, allow_pickle=False) as archive:
        prefix_nodes = np.asarray(
            archive["node_positions"][:source_frame], dtype=np.float64
        )
        prefix_directors = np.asarray(
            archive["element_directors"][:source_frame], dtype=np.float64
        )
        candidate_side = int(np.asarray(archive["route_side"]).reshape(()).item())
    if candidate_side != int(bank.route_sides[index]):
        raise RuntimeError("candidate and snapshot route sides differ")

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make_env():
        return ManiSoftWallCrossingSACEnv(
            run["scenario"],
            task_config_path=run["task_config"],
            snapshot_bank_path=run["snapshot_bank"],
            **run["environment"],
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
    observation, info = env.reset(
        seed=840049, options={"snapshot_index": index}
    )
    rod = env.sim._backend._softrobot
    policy_nodes = [rod.position_collection.T.astype(np.float64, copy=True)]
    policy_directors = [
        rod.director_collection.transpose(2, 0, 1).astype(np.float64, copy=True)
    ]
    policy_fractions = [float(info["distal_crossed_fraction"])]
    policy_clearances = [float(info["wall_clearance"])]
    policy_rewards: list[float] = []
    while True:
        normalized = normalization.normalize_obs(observation[None, :])[0]
        action, _ = model.predict(normalized, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        policy_nodes.append(rod.position_collection.T.astype(np.float64, copy=True))
        policy_directors.append(
            rod.director_collection.transpose(2, 0, 1).astype(
                np.float64, copy=True
            )
        )
        policy_fractions.append(float(info["distal_crossed_fraction"]))
        policy_clearances.append(float(info["wall_clearance"]))
        policy_rewards.append(float(reward))
        if terminated or truncated:
            break
    env.close()
    normalization.close()
    if not info["is_success"]:
        raise RuntimeError(
            f"selected rollout did not succeed: {info['termination_reason']}"
        )
    if not np.allclose(prefix_nodes[-1], bank.node_positions[index - 0], atol=0.08):
        # The prefix intentionally excludes the snapshot itself; this bound
        # only detects an unrelated candidate file rather than requiring two
        # adjacent dynamic frames to coincide.
        raise RuntimeError("candidate prefix does not lead to the snapshot")

    geometry = env.geometry
    policy_nodes_array = np.asarray(policy_nodes)
    policy_directors_array = np.asarray(policy_directors)
    nodes = np.concatenate((prefix_nodes, policy_nodes_array), axis=0)
    directors = np.concatenate((prefix_directors, policy_directors_array), axis=0)
    stage = np.concatenate(
        (
            np.zeros(len(prefix_nodes), dtype=np.int8),
            np.ones(len(policy_nodes_array), dtype=np.int8),
        )
    )
    fractions = np.asarray(
        [_distal_fraction(value, geometry) for value in nodes], dtype=np.float32
    )
    clearances = np.asarray(
        [geometry.whole_arm_wall_clearance(value) for value in nodes],
        dtype=np.float32,
    )
    return {
        "nodes": nodes,
        "directors": directors,
        "stage": stage,
        "fractions": fractions,
        "wall_clearances": clearances,
        "policy_start": len(prefix_nodes),
        "policy_steps": len(policy_nodes_array) - 1,
        "route_side": candidate_side,
        "geometry": geometry,
        "control_dt": float(bank.control_dt),
        "policy_return": float(np.sum(policy_rewards)),
        "snapshot_index": index,
        "source_episode": source_episode,
        "source_frame": source_frame,
    }


def _render(args: argparse.Namespace, rollout: dict):
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_name(output.stem + "_preview.png")
    trajectory_output = (
        output.with_name(output.stem + "_trajectory.npz")
        if args.trajectory_output is None
        else Path(args.trajectory_output).expanduser().resolve()
    )
    np.savez_compressed(
        trajectory_output,
        node_positions=rollout["nodes"],
        element_directors=rollout["directors"],
        stages=rollout["stage"],
        distal_crossed_fractions=rollout["fractions"],
        wall_clearances=rollout["wall_clearances"],
        policy_start=np.asarray(rollout["policy_start"]),
        snapshot_index=np.asarray(rollout["snapshot_index"]),
        source_episode=np.asarray(rollout["source_episode"]),
        source_frame=np.asarray(rollout["source_frame"]),
        route_side=np.asarray(rollout["route_side"]),
        control_dt=np.asarray(rollout["control_dt"]),
        policy_return=np.asarray(rollout["policy_return"]),
    )

    model, data = _display_model(rollout["geometry"])
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.34, 0.42]
    camera.distance = 1.72

    hold_start = 0.70
    hold_end = 1.25
    trajectory_duration = (
        (len(rollout["nodes"]) - 1)
        * rollout["control_dt"]
        / args.playback_speed
    )
    duration = hold_start + trajectory_duration + hold_end
    frame_count = int(round(duration * args.fps)) + 1
    times = np.arange(frame_count, dtype=np.float64) / args.fps
    source_floats = np.clip(
        (times - hold_start)
        * args.playback_speed
        / rollout["control_dt"],
        0,
        len(rollout["nodes"]) - 1,
    )
    source_indices = np.rint(source_floats).astype(np.int64)
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
    preview_written = False
    geometry = rollout["geometry"]
    threshold = geometry.wall_maximum[1] + geometry.postwall_y_margin
    try:
        for frame_index, (time_value, source_index) in enumerate(
            zip(times, source_indices)
        ):
            phase = frame_index / max(frame_count - 1, 1)
            camera.azimuth = 145.0 - 28.0 * phase
            camera.elevation = -22.0 - 5.0 * phase
            renderer.update_scene(data, camera=camera)
            stage = int(rollout["stage"][source_index])
            nodes = rollout["nodes"][source_index]
            directors = rollout["directors"][source_index]
            update_manisoft_softrobot_geoms(
                renderer.scene,
                nodes,
                directors,
                body_radius=0.024,
                band_radius=0.028,
                band_length=0.013,
                pipe_offset=0.027,
                pipe_radius=0.0035,
                body_rgba=(0.48, 0.51, 0.57, 1.0)
                if stage == 0
                else (0.28, 0.48, 0.72, 1.0),
                detail_rgba=(0.025, 0.025, 0.025, 1.0),
                clear_existing=False,
            )
            checked = nodes[geometry.mounting_exempt_nodes :]
            distal_count = 0
            for value in (checked[:, 1] >= threshold)[::-1]:
                if not value:
                    break
                distal_count += 1
            if distal_count >= 2:
                update_softrobot_geoms(
                    renderer.scene,
                    checked[-distal_count:],
                    radius=0.030,
                    rgba=(0.12, 0.88, 0.30, 1.0),
                    clear_existing=False,
                )
            rendered = renderer.render()
            fraction = float(rollout["fractions"][source_index])
            success = bool(
                fraction >= 0.30 - 1e-8
                and time_value
                >= hold_start + trajectory_duration - 0.05
            )
            rendered = _overlay(
                rendered,
                stage=stage,
                fraction=fraction,
                wall_clearance=float(rollout["wall_clearances"][source_index]),
                source_time=float(source_index * rollout["control_dt"]),
                success=success,
            )
            writer.append_data(rendered)
            if (
                not preview_written
                and stage == 1
                and fraction >= 0.25 - 1e-8
            ):
                imageio.imwrite(preview, rendered)
                preview_written = True
    finally:
        renderer.close()
        writer.close()
    if not preview_written:
        imageio.imwrite(preview, rendered)
    return output, preview, trajectory_output, duration


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.fps) <= 0 or args.playback_speed <= 0:
        raise ValueError("render sizes, fps, and playback speed must be positive")
    rollout = _rollout(args)
    output, preview, trajectory, duration = _render(args, rollout)
    print(
        json.dumps(
            {
                "video": str(output),
                "preview": str(preview),
                "trajectory": str(trajectory),
                "duration_seconds": duration,
                "snapshot_index": rollout["snapshot_index"],
                "source_episode": rollout["source_episode"],
                "source_frame": rollout["source_frame"],
                "policy_steps": rollout["policy_steps"],
                "policy_return": rollout["policy_return"],
                "final_distal_fraction": float(rollout["fractions"][-1]),
                "minimum_wall_clearance": float(
                    np.min(
                        rollout["wall_clearances"][rollout["policy_start"] :]
                    )
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
