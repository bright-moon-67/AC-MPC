#!/usr/bin/env python
"""Render a safe post-wall rollout until the tip crosses the yz plane."""

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
import yaml
from PIL import Image, ImageDraw, ImageFont

from antmaze_ac.data.wall_crossing_snapshot_bank import (
    load_wall_crossing_snapshot_bank,
)
from antmaze_ac.envs.manisoft_wall_crossing_sac_env import (
    ManiSoftWallCrossingSACEnv,
)
from manisoft.visualize.mujoco_viewer import update_softrobot_geoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--snapshot-index", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-output", default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--playback-speed", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--from-upright",
        action="store_true",
        help="Reconstruct the traceable continuous prefix from the upright state.",
    )
    return parser.parse_args()


def _font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _display_model(geometry):
    wall_center = 0.5 * (geometry.wall_minimum + geometry.wall_maximum)
    wall_half_size = 0.5 * (geometry.wall_maximum - geometry.wall_minimum)
    xml = f"""
<mujoco model="yz_plane_reachability">
  <option timestep="0.0002"/>
  <statistic center="0 0.42 0.38" extent="1.12"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.56 0.56 0.56" diffuse="0.76 0.76 0.76"
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
    <geom name="yz_plane" type="box" pos="0 0.61 0.40"
          size="0.001 0.39 0.40" rgba="0.10 0.70 0.95 0.13"/>
    <geom name="wall" type="box"
          pos="{wall_center[0]} {wall_center[1]} {wall_center[2]}"
          size="{wall_half_size[0]} {wall_half_size[1]} {wall_half_size[2]}"
          rgba="0.88 0.23 0.14 0.82"/>
    <geom name="base" type="cylinder" pos="0 0 0.018"
          size="0.075 0.018" rgba="0.16 0.19 0.24 1"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _overlay(frame: np.ndarray, row: dict, crossed: bool) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(25)
    body_font = _font(18)
    small_font = _font(15)
    draw.rounded_rectangle((20, 18, 620, 194), radius=12, fill=(8, 14, 24, 198))
    draw.text((38, 31), "ManiSoft: safe return to the yz plane", font=title_font, fill="white")
    draw.text(
        (38, 68),
        "Target y ignored  |  cyan: x = 0 plane  |  red: wall",
        font=small_font,
        fill=(175, 224, 250),
    )
    draw.text(
        (38, 93),
        f"Stage: {row['stage']}",
        font=small_font,
        fill=(252, 206, 111),
    )
    draw.text(
        (38, 119),
        f"|tip x|: {1000 * row['plane_distance']:6.2f} mm",
        font=body_font,
        fill=(120, 245, 160) if crossed else "white",
    )
    draw.text(
        (38, 146),
        f"Distal body beyond wall: {100 * row['fraction']:4.0f}%",
        font=body_font,
        fill=(140, 238, 166),
    )
    draw.text(
        (38, 172),
        f"Wall clearance: {1000 * row['wall_clearance']:6.2f} mm",
        font=small_font,
        fill=(230, 234, 240),
    )
    if crossed:
        draw.rounded_rectangle(
            (frame.shape[1] - 340, 28, frame.shape[1] - 25, 82),
            radius=12,
            fill=(20, 126, 58, 230),
        )
        draw.text(
            (frame.shape[1] - 318, 43),
            "SAFE yz-PLANE CROSSING",
            font=body_font,
            fill="white",
        )
    draw.rounded_rectangle(
        (20, frame.shape[0] - 65, 690, frame.shape[0] - 18),
        radius=10,
        fill=(8, 14, 24, 180),
    )
    draw.text(
        (37, frame.shape[0] - 52),
        "Blue: complete arm (true 45 mm radius)    Green: contiguous crossed suffix",
        font=small_font,
        fill=(238, 240, 244),
    )
    return np.asarray(image)


def _policy_nodes(
    run_path: Path,
    model_path: Path,
    vecnormalize_path: Path,
    *,
    snapshot_index: int,
    steps: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    run = json.loads(run_path.read_text())

    def make_env():
        return ManiSoftWallCrossingSACEnv(
            run["scenario"],
            task_config_path=run["task_config"],
            snapshot_bank_path=run["snapshot_bank"],
            **run["environment"],
        )

    env = make_env()
    normalization = VecNormalize.load(str(vecnormalize_path), DummyVecEnv([make_env]))
    normalization.training = False
    normalization.norm_reward = False
    model = SAC.load(str(model_path), device=device)
    observation, _ = env.reset(options={"snapshot_index": snapshot_index})
    nodes = []
    actions = []
    for _ in range(steps):
        normalized = normalization.normalize_obs(observation[None, :])[0]
        action, _ = model.predict(normalized, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        nodes.append(
            env.sim._backend._softrobot.position_collection.T.astype(
                np.float64, copy=True
            )
        )
        actions.append(env.previous_action.astype(np.float32, copy=True))
        if (terminated or truncated) and len(nodes) < steps:
            raise RuntimeError(
                f"policy prefix terminated at step {len(nodes)}: "
                f"{info['termination_reason']}"
            )
    normalization.close()
    env.close()
    return np.asarray(nodes), np.asarray(actions)


def _full_upright_prefix(args: argparse.Namespace, return_rollout: dict, geometry):
    run_path = Path(args.run_config).expanduser().resolve()
    run = json.loads(run_path.read_text())
    experiment_root = Path(run["snapshot_bank"]).expanduser().resolve().parents[1]
    scenario = Path(run["scenario"])
    old_task = Path(
        "/root/autodl-tmp/AC-MPC/configs/"
        "manisoft_wall_route_collection_strong_bend_e2mpa_r45mm_t45_a060.yaml"
    )

    candidate_path = experiment_root / "candidate_collection_v2/episode_00038.npz"
    with np.load(candidate_path, allow_pickle=False) as archive:
        approach = np.asarray(archive["node_positions"][:57], dtype=np.float64)
        approach_actions = np.asarray(archive["actions"][:56], dtype=np.float32)
    approach[:, :, 0] *= -1.0
    approach_actions = approach_actions.reshape(-1, 6, 3)
    approach_actions[:, :, 1:] *= -1.0
    approach_actions = approach_actions.reshape(-1, 18)

    mirrored_bank_path = experiment_root / "snapshot_bank_05_30pct_mirrored.npz"
    mirrored_bank = load_wall_crossing_snapshot_bank(mirrored_bank_path)
    errors = {
        "approach_to_brake_m": float(
            np.max(np.abs(approach[-1] - mirrored_bank.node_positions[72]))
        )
    }
    brake_config = yaml.safe_load(
        Path(
            "/root/autodl-tmp/AC-MPC/configs/"
            "manisoft_wall_crossing_sac_strong_bend_return.yaml"
        ).read_text()
    )["environment"]
    brake_config["episode_steps"] = 200
    brake_env = ManiSoftWallCrossingSACEnv(
        scenario,
        task_config_path=old_task,
        snapshot_bank_path=mirrored_bank_path,
        **brake_config,
    )
    brake_env.reset(options={"snapshot_index": 72})
    braking = []
    braking_actions = []
    for _ in range(95):
        request = np.clip(
            -brake_env.previous_action / brake_env.max_action_delta, -1.0, 1.0
        ).astype(np.float32)
        _, _, terminated, truncated, info = brake_env.step(request)
        if terminated or truncated:
            raise RuntimeError(
                f"braking prefix terminated early: {info['termination_reason']}"
            )
        braking.append(
            brake_env.sim._backend._softrobot.position_collection.T.astype(
                np.float64, copy=True
            )
        )
        braking_actions.append(brake_env.previous_action.copy())
    brake_env.close()
    braking = np.asarray(braking)
    braking_actions = np.asarray(braking_actions, dtype=np.float32)
    braked_bank = load_wall_crossing_snapshot_bank(
        experiment_root / "snapshot_bank_braked_30_35pct.npz"
    )
    errors["brake_to_phase_a_m"] = float(
        np.max(np.abs(braking[-1] - braked_bank.node_positions[7]))
    )

    phase_a_root = experiment_root / "sac_crossing_right_40pct_v1"
    phase_a, phase_a_actions = _policy_nodes(
        phase_a_root / "run_config.json",
        phase_a_root / "checkpoints/wall_crossing_sac_2500_steps.zip",
        phase_a_root
        / "checkpoints/wall_crossing_sac_vecnormalize_2500_steps.pkl",
        snapshot_index=7,
        steps=60,
        device=args.device,
    )
    first_40_bank = load_wall_crossing_snapshot_bank(
        experiment_root / "snapshot_bank_postcrossing_right_40pct.npz"
    )
    errors["phase_a_to_stabilization_m"] = float(
        np.max(np.abs(phase_a[-1] - first_40_bank.node_positions[1]))
    )

    stabilize_root = experiment_root / "sac_right_postcrossing_joint_20cm_v1"
    stabilization, stabilization_actions = _policy_nodes(
        stabilize_root / "run_config.json",
        stabilize_root / "checkpoints/wall_crossing_sac_1250_steps.zip",
        stabilize_root
        / "checkpoints/wall_crossing_sac_vecnormalize_1250_steps.pkl",
        snapshot_index=1,
        steps=20,
        device=args.device,
    )
    stable_bank = load_wall_crossing_snapshot_bank(
        experiment_root / "snapshot_bank_postcrossing_right_40pct_streak20.npz"
    )
    errors["stabilization_to_return_m"] = float(
        np.max(np.abs(stabilization[-1] - stable_bank.node_positions[1]))
    )
    errors["return_reset_m"] = float(
        np.max(np.abs(return_rollout["nodes"][0] - stable_bank.node_positions[1]))
    )
    if max(errors.values()) > 5e-8:
        raise RuntimeError(f"full-rollout stage discontinuity: {errors}")

    nodes = np.concatenate(
        (approach, braking, phase_a, stabilization, return_rollout["nodes"][1:]),
        axis=0,
    )
    actions = np.concatenate(
        (
            approach_actions,
            braking_actions,
            phase_a_actions,
            stabilization_actions,
            return_rollout["actions"],
        ),
        axis=0,
    )
    if len(actions) != len(nodes) - 1:
        raise RuntimeError("full rollout must have one action per transition")
    stages = np.concatenate(
        (
            np.zeros(len(approach), dtype=np.int8),
            np.ones(len(braking), dtype=np.int8),
            np.full(len(phase_a), 2, dtype=np.int8),
            np.full(len(stabilization), 3, dtype=np.int8),
            np.full(len(return_rollout["nodes"]) - 1, 4, dtype=np.int8),
        )
    )
    threshold = geometry.wall_maximum[1] + geometry.postwall_y_margin
    fractions = []
    for current in nodes:
        checked = current[geometry.mounting_exempt_nodes :]
        count = 0
        for value in (checked[:, 1] >= threshold)[::-1]:
            if not value:
                break
            count += 1
        fractions.append(count / len(checked))
    prefix_count = len(nodes) - len(return_rollout["nodes"])
    tip_speeds = np.concatenate(
        (
            np.full(prefix_count, np.nan, dtype=np.float32),
            return_rollout["tip_speeds"],
        )
    )
    full = {
        "nodes": nodes,
        "actions": actions,
        "fractions": np.asarray(fractions, dtype=np.float32),
        "wall_clearances": np.asarray(
            [geometry.whole_arm_wall_clearance(value) for value in nodes],
            dtype=np.float32,
        ),
        "ground_clearances": np.asarray(
            [geometry.whole_arm_ground_clearance(value) for value in nodes],
            dtype=np.float32,
        ),
        "tip_speeds": tip_speeds,
        "tip_x": nodes[:, -1, 0].astype(np.float32),
        "stages": stages,
        "crossed_at": len(nodes) - 1,
        "control_dt": return_rollout["control_dt"],
        "snapshot_index": return_rollout["snapshot_index"],
        "continuity_errors_m": errors,
    }
    if np.min(full["wall_clearances"]) < 0:
        raise RuntimeError("full upright trajectory intersects the test wall")
    if np.min(full["ground_clearances"]) < -geometry.ground_violation_tolerance:
        raise RuntimeError("full upright trajectory violates the ground plane")
    return full


def _rollout(args: argparse.Namespace) -> tuple[dict, object]:
    run = json.loads(Path(args.run_config).expanduser().resolve().read_text())
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
    model = SAC.load(str(Path(args.model).expanduser().resolve()), device=args.device)
    observation, info = env.reset(
        seed=826270 + args.snapshot_index,
        options={"snapshot_index": args.snapshot_index},
    )
    rod = env.sim._backend._softrobot
    nodes = [rod.position_collection.T.astype(np.float64, copy=True)]
    fractions = [float(info["distal_crossed_fraction"])]
    wall_clearances = [float(info["wall_clearance"])]
    ground_clearances = [float(info["ground_clearance"])]
    tip_speeds = [float(info["tip_speed"])]
    tip_x = [float(info["tip_x"])]
    applied_actions = []
    crossed_at = None
    termination_reason = None
    while env.step_count < env.episode_steps:
        normalized = normalization.normalize_obs(observation[None, :])[0]
        action, _ = model.predict(normalized, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        nodes.append(rod.position_collection.T.astype(np.float64, copy=True))
        fractions.append(float(info["distal_crossed_fraction"]))
        wall_clearances.append(float(info["wall_clearance"]))
        ground_clearances.append(float(info["ground_clearance"]))
        tip_speeds.append(float(info["tip_speed"]))
        tip_x.append(float(info["tip_x"]))
        applied_actions.append(env.previous_action.astype(np.float32, copy=True))
        safe = bool(
            wall_clearances[-1] >= 0
            and ground_clearances[-1]
            >= -env.geometry.ground_violation_tolerance
            and fractions[-1] >= env.success_crossed_fraction - 1e-8
        )
        if safe and tip_x[-2] * tip_x[-1] <= 0:
            crossed_at = len(nodes) - 1
            break
        if terminated or truncated:
            termination_reason = info["termination_reason"] or "episode_limit"
            break
    normalization.close()
    if crossed_at is None:
        env.close()
        raise RuntimeError(
            "rollout did not safely cross x=0 before termination: "
            f"{termination_reason}"
        )
    result = {
        "nodes": np.asarray(nodes),
        "fractions": np.asarray(fractions, dtype=np.float32),
        "wall_clearances": np.asarray(wall_clearances, dtype=np.float32),
        "ground_clearances": np.asarray(ground_clearances, dtype=np.float32),
        "tip_speeds": np.asarray(tip_speeds, dtype=np.float32),
        "tip_x": np.asarray(tip_x, dtype=np.float32),
        "actions": np.asarray(applied_actions, dtype=np.float32),
        "crossed_at": int(crossed_at),
        "control_dt": float(env.control_dt),
        "snapshot_index": int(args.snapshot_index),
        "stages": np.full(len(nodes), 4, dtype=np.int8),
        "continuity_errors_m": {},
    }
    geometry = env.geometry
    env.close()
    return result, geometry


def _render(args: argparse.Namespace, rollout: dict, geometry):
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_name(output.stem + "_preview.png")
    trajectory = (
        output.with_name(output.stem + "_trajectory.npz")
        if args.trajectory_output is None
        else Path(args.trajectory_output).expanduser().resolve()
    )
    trajectory_arrays = {
        key: value
        for key, value in rollout.items()
        if key != "continuity_errors_m"
    }
    trajectory_arrays["continuity_errors_json"] = np.asarray(
        json.dumps(rollout["continuity_errors_m"], sort_keys=True)
    )
    np.savez_compressed(trajectory, **trajectory_arrays)
    model, data = _display_model(geometry)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.43, 0.38]
    camera.distance = 1.62
    camera.azimuth = 140.0
    camera.elevation = -24.0

    start_hold = 0.75
    end_hold = 1.50
    motion_duration = (
        rollout["crossed_at"] * rollout["control_dt"] / args.playback_speed
    )
    duration = start_hold + motion_duration + end_hold
    frame_count = int(round(duration * args.fps)) + 1
    times = np.arange(frame_count, dtype=np.float64) / args.fps
    indices = np.rint(
        np.clip(
            (times - start_hold) * args.playback_speed / rollout["control_dt"],
            0,
            rollout["crossed_at"],
        )
    ).astype(np.int64)
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
    try:
        for frame_index, (time_value, index) in enumerate(zip(times, indices)):
            phase = frame_index / max(frame_count - 1, 1)
            camera.azimuth = 143.0 - 18.0 * phase
            renderer.update_scene(data, camera=camera)
            current_nodes = rollout["nodes"][index]
            update_softrobot_geoms(
                renderer.scene,
                current_nodes,
                radius=geometry.arm_radius,
                rgba=(0.20, 0.43, 0.76, 1.0),
                clear_existing=False,
            )
            checked = current_nodes[geometry.mounting_exempt_nodes :]
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
                index == rollout["crossed_at"]
                and time_value >= start_hold + motion_duration - 0.04
            )
            row = {
                "plane_distance": abs(float(rollout["tip_x"][index])),
                "fraction": float(rollout["fractions"][index]),
                "wall_clearance": float(rollout["wall_clearances"][index]),
                "stage": {
                    0: "upright approach / side bypass",
                    1: "activation braking",
                    2: "Phase A: increase distal crossing",
                    3: "stabilize 40% crossing",
                    4: "distal return toward yz plane",
                }[int(rollout["stages"][index])],
            }
            rendered = _overlay(rendered, row, crossed)
            writer.append_data(rendered)
            if index == rollout["crossed_at"] and not preview.is_file():
                imageio.imwrite(preview, rendered)
    finally:
        renderer.close()
        writer.close()
    if not preview.is_file():
        imageio.imwrite(preview, rendered)
    return output, preview, trajectory, duration


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.fps) <= 0 or args.playback_speed <= 0:
        raise ValueError("render sizes, fps and playback speed must be positive")
    rollout, geometry = _rollout(args)
    if args.from_upright:
        rollout = _full_upright_prefix(args, rollout, geometry)
    output, preview, trajectory, duration = _render(args, rollout, geometry)
    final = rollout["crossed_at"]
    print(
        json.dumps(
            {
                "video": str(output),
                "preview": str(preview),
                "trajectory": str(trajectory),
                "duration_seconds": duration,
                "snapshot_index": rollout["snapshot_index"],
                "safe_crossing_step": final,
                "safe_crossing_time_seconds": final * rollout["control_dt"],
                "tip_xyz_at_crossing_m": rollout["nodes"][final, -1].tolist(),
                "distal_crossed_fraction": float(rollout["fractions"][final]),
                "wall_clearance_m": float(rollout["wall_clearances"][final]),
                "ground_clearance_m": float(rollout["ground_clearances"][final]),
                "tip_speed_mps": float(rollout["tip_speeds"][final]),
                "from_upright": bool(args.from_upright),
                "total_control_steps": int(len(rollout["nodes"]) - 1),
                "minimum_wall_clearance_m": float(
                    np.min(rollout["wall_clearances"])
                ),
                "minimum_ground_clearance_m": float(
                    np.min(rollout["ground_clearances"])
                ),
                "continuity_errors_m": rollout["continuity_errors_m"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
