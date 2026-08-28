#!/usr/bin/env python
"""Render the selected SAC rollout with a force-free terminal cube push."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import yaml

from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.kinematic_push_task import (
    point_aabb_distance,
    segment_aabb_distance,
)
from manisoft.visualize.mujoco_viewer import update_softrobot_geoms
from render_manisoft_teacher_tracking_sac import (
    _font,
    _rollout,
    _sac_badge,
    _sha256,
)


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
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20290865)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wall-negative-x-extension", type=float, default=0.20)
    parser.add_argument("--platform-x-half-size", type=float, default=0.12)
    parser.add_argument("--platform-y-center", type=float, default=0.615)
    parser.add_argument("--platform-y-half-size", type=float, default=0.055)
    parser.add_argument("--platform-top-z", type=float, default=0.110)
    parser.add_argument("--platform-thickness", type=float, default=0.015)
    parser.add_argument("--cube-size", type=float, default=0.060)
    parser.add_argument("--cube-initial-x", type=float, default=-0.055)
    parser.add_argument("--contact-margin", type=float, default=0.002)
    parser.add_argument("--maximum-push-distance", type=float, default=0.040)
    parser.add_argument("--required-push-distance", type=float, default=0.015)
    parser.add_argument(
        "--visual-push-target-distance",
        type=float,
        default=None,
        help=(
            "Optional force-free cube displacement reached by rescaling the "
            "post-contact tip progress; the tip/cube surface contact is verified."
        ),
    )
    parser.add_argument(
        "--grounded-platform",
        action="store_true",
        help="Extend the support platform down to the ground plane.",
    )
    parser.add_argument(
        "--no-platform",
        action="store_true",
        help="Remove the support entirely and place the cube directly on z=0.",
    )
    parser.add_argument(
        "--realistic-scene",
        action="store_true",
        help="Use photographic materials, shadows and no yz helper plane.",
    )
    parser.add_argument(
        "--clean-render",
        action="store_true",
        help="Hide diagnostic HUD overlays for a clean scene render.",
    )
    return parser.parse_args()


def _geometry_values(args: argparse.Namespace, geometry: WallRouteGeometry) -> dict:
    platform_minimum = np.asarray(
        [
            -args.platform_x_half_size,
            args.platform_y_center - args.platform_y_half_size,
            0.0
            if args.grounded_platform
            else args.platform_top_z - args.platform_thickness,
        ],
        dtype=np.float64,
    )
    platform_maximum = np.asarray(
        [
            args.platform_x_half_size,
            args.platform_y_center + args.platform_y_half_size,
            args.platform_top_z,
        ],
        dtype=np.float64,
    )
    cube_initial_center = np.asarray(
        [
            args.cube_initial_x,
            args.platform_y_center,
            0.5 * args.cube_size
            if args.no_platform
            else args.platform_top_z + 0.5 * args.cube_size,
        ],
        dtype=np.float64,
    )
    extended_wall_minimum = geometry.wall_minimum.copy()
    extended_wall_minimum[0] -= args.wall_negative_x_extension
    return {
        "platform_minimum": platform_minimum,
        "platform_maximum": platform_maximum,
        "cube_initial_center": cube_initial_center,
        "extended_wall_minimum": extended_wall_minimum,
        "extended_wall_maximum": geometry.wall_maximum.copy(),
    }


def _minimum_capsule_clearance(
    nodes: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    radius: float,
    *,
    first_node: int = 0,
) -> tuple[float, tuple[int, int]]:
    best = float("inf")
    best_at = (-1, -1)
    for step, current in enumerate(nodes):
        checked = current[first_node:]
        for segment, (start, end) in enumerate(zip(checked[:-1], checked[1:])):
            clearance = segment_aabb_distance(start, end, minimum, maximum) - radius
            if clearance < best:
                best = float(clearance)
                best_at = (step, segment + first_node)
    return best, best_at


def _cube_motion(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    geometry: WallRouteGeometry,
    values: dict,
) -> tuple[np.ndarray, int, float]:
    tips = arrays["node_positions"][:, -1]
    centers = np.repeat(values["cube_initial_center"][None, :], len(tips), axis=0)
    half = np.full(3, 0.5 * args.cube_size, dtype=np.float64)
    minimum = values["cube_initial_center"] - half
    maximum = values["cube_initial_center"] + half
    contact_index = -1
    for index in range(max(1, int(0.75 * len(tips))), len(tips)):
        approaching_negative_x = tips[index, 0] < tips[index - 1, 0]
        if approaching_negative_x and point_aabb_distance(
            tips[index], minimum, maximum
        ) <= geometry.arm_radius + args.contact_margin:
            contact_index = index
            break
    if contact_index < 0:
        raise RuntimeError("the terminal tip surface never reaches the visual cube")
    contact_tip_x = float(tips[contact_index, 0])
    available_tip_travel = contact_tip_x - float(tips[-1, 0])
    if available_tip_travel <= 0.0:
        raise RuntimeError("the terminal tip has no negative-x travel after contact")
    target_push = (
        min(args.maximum_push_distance, available_tip_travel)
        if args.visual_push_target_distance is None
        else args.visual_push_target_distance
    )
    for index in range(contact_index, len(tips)):
        progress = float(
            np.clip(
                (contact_tip_x - tips[index, 0]) / available_tip_travel,
                0.0,
                1.0,
            )
        )
        centers[index, 0] -= target_push * progress
    push_distance = float(values["cube_initial_center"][0] - centers[-1, 0])
    if push_distance < args.required_push_distance:
        raise RuntimeError(
            f"visual cube push {push_distance:.6f} m is below the requirement"
        )
    return centers, contact_index, push_distance


def _verify_visual_task(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    geometry: WallRouteGeometry,
    values: dict,
) -> tuple[np.ndarray, dict]:
    platform_clearance = None
    platform_at = (-1, -1)
    if not args.no_platform:
        platform_clearance, platform_at = _minimum_capsule_clearance(
            arrays["node_positions"],
            values["platform_minimum"],
            values["platform_maximum"],
            geometry.arm_radius,
        )
        if platform_clearance <= 0.0:
            raise RuntimeError("the soft arm contacts the visual platform")
    extended_wall_clearance, wall_at = _minimum_capsule_clearance(
        arrays["node_positions"],
        values["extended_wall_minimum"],
        values["extended_wall_maximum"],
        geometry.arm_radius + geometry.wall_safety_margin,
        first_node=geometry.mounting_exempt_nodes,
    )
    if extended_wall_clearance <= 0.0:
        raise RuntimeError("the soft arm contacts the extended visual wall")
    cube_centers, contact_index, push_distance = _cube_motion(
        args, arrays, geometry, values
    )
    tip_cube_separations = []
    cube_half_vector = np.full(3, 0.5 * args.cube_size, dtype=np.float64)
    for tip, center in zip(
        arrays["node_positions"][contact_index:, -1],
        cube_centers[contact_index:],
    ):
        tip_cube_separations.append(
            point_aabb_distance(
                tip,
                center - cube_half_vector,
                center + cube_half_vector,
            )
            - geometry.arm_radius
        )
    maximum_tip_cube_separation = float(np.max(tip_cube_separations))
    if maximum_tip_cube_separation > args.contact_margin:
        raise RuntimeError(
            "the tip surface loses contact with the visually pushed cube"
        )
    cube_half = 0.5 * args.cube_size
    support_margin = None
    if args.no_platform:
        cube_ground_clearance = float(cube_centers[-1, 2] - cube_half)
        if abs(cube_ground_clearance) > 1e-12:
            raise RuntimeError("the platform-free cube is not grounded")
    else:
        support_margins = np.asarray(
            [
                cube_centers[-1, 0] - cube_half - values["platform_minimum"][0],
                values["platform_maximum"][0] - (cube_centers[-1, 0] + cube_half),
                cube_centers[-1, 1] - cube_half - values["platform_minimum"][1],
                values["platform_maximum"][1] - (cube_centers[-1, 1] + cube_half),
            ],
            dtype=np.float64,
        )
        if np.min(support_margins) < 0.0:
            raise RuntimeError("the visually pushed cube leaves the platform footprint")
        support_margin = float(np.min(support_margins))
        cube_ground_clearance = float(values["platform_maximum"][2])
    verification = {
        "visual_push_is_success": True,
        "cube_contact_index": contact_index,
        "cube_contact_time_s": contact_index * float(arrays["control_dt"]),
        "cube_initial_center_m": values["cube_initial_center"].tolist(),
        "cube_final_center_m": cube_centers[-1].tolist(),
        "cube_push_distance_negative_x_m": push_distance,
        "visual_push_target_distance_m": args.visual_push_target_distance,
        "maximum_tip_cube_surface_separation_m": maximum_tip_cube_separation,
        "minimum_tip_cube_surface_separation_m": float(
            np.min(tip_cube_separations)
        ),
        "minimum_cube_platform_edge_margin_m": support_margin,
        "cube_ground_clearance_m": cube_ground_clearance,
        "cube_is_directly_grounded": args.no_platform,
        "minimum_arm_platform_clearance_m": platform_clearance,
        "minimum_arm_platform_clearance_step": platform_at[0],
        "minimum_extended_wall_clearance_m": extended_wall_clearance,
        "minimum_extended_wall_clearance_step": wall_at[0],
        "platform_minimum_m": values["platform_minimum"].tolist(),
        "platform_maximum_m": values["platform_maximum"].tolist(),
        "platform_yz_symmetric": bool(
            np.isclose(
                values["platform_minimum"][0],
                -values["platform_maximum"][0],
            )
        ),
        "extended_wall_minimum_m": values["extended_wall_minimum"].tolist(),
        "extended_wall_maximum_m": values["extended_wall_maximum"].tolist(),
        "wall_negative_x_extension_m": args.wall_negative_x_extension,
        "cube_motion_model": (
            "force_free_contact_verified_rescaled_translation_along_negative_x"
            if args.visual_push_target_distance is not None
            else "force_free_latched_visual_translation_along_negative_x"
        ),
    }
    return cube_centers, verification


def _display_model(
    geometry: WallRouteGeometry,
    values: dict,
    args: argparse.Namespace,
) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    wall_center = 0.5 * (
        values["extended_wall_minimum"] + values["extended_wall_maximum"]
    )
    wall_half = 0.5 * (
        values["extended_wall_maximum"] - values["extended_wall_minimum"]
    )
    platform_center = 0.5 * (
        values["platform_minimum"] + values["platform_maximum"]
    )
    platform_half = 0.5 * (
        values["platform_maximum"] - values["platform_minimum"]
    )
    cube = values["cube_initial_center"]
    goal = cube.copy()
    goal[0] -= args.required_push_distance
    yz_plane = (
        ""
        if args.realistic_scene
        else '<geom name="yz_plane" type="box" pos="0 0.61 0.40" '
        'size="0.001 0.39 0.40" rgba="0.10 0.70 0.95 0.10"/>'
    )
    floor_material = "real_floor" if args.realistic_scene else "floor_mat"
    wall_material = 'material="painted_wall"' if args.realistic_scene else 'rgba="0.96 0.30 0.12 0.84"'
    platform_material = 'material="support_mat"' if args.realistic_scene else 'rgba="0.24 0.27 0.31 1"'
    cube_material = 'material="wood_cube"' if args.realistic_scene else 'rgba="0.98 0.76 0.12 1"'
    platform_geom = (
        ""
        if args.no_platform
        else f'''<geom name="platform" type="box"
          pos="{platform_center[0]} {platform_center[1]} {platform_center[2]}"
          size="{platform_half[0]} {platform_half[1]} {platform_half[2]}"
          {platform_material}/>'''
    )
    xml = f"""
<mujoco model="terminal_visual_push">
  <option timestep="0.0002"/>
  <statistic center="0 0.43 0.38" extent="1.12"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.30 0.30 0.30" diffuse="0.62 0.62 0.62"
               specular="0.28 0.28 0.28"/>
    <rgba haze="0.78 0.82 0.88 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.34 0.42 0.55" rgb2="0.90 0.92 0.94"
             width="512" height="3072"/>
    <material name="floor_mat" rgba="0.82 0.84 0.86 1"/>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.31 0.32 0.33" rgb2="0.345 0.355 0.365"
             width="512" height="512"/>
    <material name="real_floor" texture="floor_tex" texrepeat="7 7"
              texuniform="true" specular="0.20" shininess="0.18"
              reflectance="0.0"/>
    <material name="painted_wall" rgba="0.78 0.20 0.075 1"
              specular="0.16" shininess="0.12" reflectance="0.02"/>
    <material name="support_mat" rgba="0.18 0.20 0.22 1"
              specular="0.62" shininess="0.52" reflectance="0.08"/>
    <texture name="wood_tex" type="2d" builtin="checker"
             rgb1="0.62 0.31 0.10" rgb2="0.82 0.49 0.18"
             width="256" height="256"/>
    <material name="wood_cube" texture="wood_tex" texrepeat="2 2"
              texuniform="true" specular="0.20" shininess="0.16"/>
    <material name="base_mat" rgba="0.09 0.105 0.12 1"
              specular="0.82" shininess="0.70" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light directional="true" castshadow="false" pos="-0.8 -0.7 2.4"
           dir="0.35 0.45 -1" diffuse="0.92 0.88 0.82" specular="0.35 0.35 0.35"/>
    <light directional="true" castshadow="false" pos="1.2 1.1 1.8"
           dir="-0.45 -0.55 -1" diffuse="0.38 0.43 0.52" specular="0.16 0.16 0.18"/>
    <geom name="ground" type="plane" size="1.4 1.4 0.1" material="{floor_material}"/>
    {yz_plane}
    <geom name="extended_wall" type="box"
          pos="{wall_center[0]} {wall_center[1]} {wall_center[2]}"
          size="{wall_half[0]} {wall_half[1]} {wall_half[2]}"
          {wall_material}/>
    <geom name="base" type="cylinder" pos="0 0 0.018"
          size="0.075 0.018" material="base_mat"/>
    <geom name="base_ring" type="cylinder" pos="0 0 0.037"
          size="0.058 0.008" material="support_mat"/>
    {platform_geom}
    <geom name="push_goal" type="box"
          pos="{goal[0]} {goal[1]} {goal[2]}"
          size="{0.5 * args.cube_size} {0.5 * args.cube_size} {0.5 * args.cube_size}"
          rgba="0.12 0.78 0.95 {0.0 if args.realistic_scene else 0.18}"
          contype="0" conaffinity="0"/>
    <body name="push_cube" mocap="true" pos="{cube[0]} {cube[1]} {cube[2]}">
      <geom name="push_cube_geom" type="box"
            size="{0.5 * args.cube_size} {0.5 * args.cube_size} {0.5 * args.cube_size}"
            {cube_material} contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_cube")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError("push cube body is not mocap-enabled")
    mujoco.mj_forward(model, data)
    return model, data, mocap_id


def _overlay(
    frame: np.ndarray,
    *,
    stage: str,
    push_distance: float,
    contact: bool,
    platform_clearance: float,
    extended_wall_clearance: float,
    success: bool,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((20, 18, 650, 194), radius=12, fill=(8, 14, 24, 205))
    draw.text(
        (38, 31),
        "ManiSoft: terminal visual push along -x",
        font=_font(24),
        fill="white",
    )
    draw.text(
        (38, 68),
        "platform symmetric about yz  |  cube motion: force-free visual",
        font=_font(14),
        fill=(175, 224, 250),
    )
    draw.text((38, 94), f"Stage: {stage}", font=_font(14), fill=(252, 206, 111))
    state = "PUSH" if push_distance > 1e-5 else ("CONTACT" if contact else "APPROACH")
    draw.text(
        (38, 120),
        f"Cube state: {state}   |   -x push: {1000 * push_distance:5.1f} mm",
        font=_font(17),
        fill=(120, 245, 160) if push_distance > 0 else "white",
    )
    draw.text(
        (38, 148),
        f"Arm-platform clearance: {1000 * platform_clearance:5.1f} mm",
        font=_font(15),
        fill=(140, 238, 166),
    )
    draw.text(
        (38, 173),
        f"Extended-wall clearance: {1000 * extended_wall_clearance:5.1f} mm",
        font=_font(15),
        fill=(230, 234, 240),
    )
    if success:
        right = frame.shape[1] - 25
        draw.rounded_rectangle((right - 315, 28, right, 82), radius=12, fill=(20, 126, 58, 230))
        draw.text((right - 292, 43), "VISUAL -x PUSH COMPLETE", font=_font(17), fill="white")
    return np.asarray(image)


def _render(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    geometry: WallRouteGeometry,
    values: dict,
    cube_centers: np.ndarray,
    visual_verification: dict,
) -> tuple[Path, Path, float]:
    output = Path(args.output).expanduser().resolve()
    preview = output.with_name(output.stem + "_preview.png")
    model, data, mocap_id = _display_model(geometry, values, args)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-0.01, 0.45, 0.34]
    camera.distance = 1.52
    camera.azimuth = 143.0
    camera.elevation = -23.0
    nodes = arrays["node_positions"]
    control_dt = float(arrays["control_dt"])
    start_hold, end_hold = 0.75, 2.00
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
        0: "upright approach and side bypass",
        1: "smooth activation transition",
        2: "increase distal crossing",
        3: "stabilize crossed body",
        4: "return toward cube",
        5: "low-speed visual push",
    }
    match = re.search(r"_(\d+)_steps", Path(args.model).stem)
    label = f"{int(match.group(1)) // 1000}k" if match is not None else "selected"
    writer = imageio.get_writer(
        output,
        format="FFMPEG",
        mode="I",
        fps=args.fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=[
            "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart"
        ],
    )
    contact_index = int(visual_verification["cube_contact_index"])
    initial_cube_x = float(values["cube_initial_center"][0])
    final_frame = None
    try:
        for frame_index, (time_value, index) in enumerate(zip(times, indices)):
            phase = frame_index / max(frame_count - 1, 1)
            # Start from the established wall-bypass view, then orbit toward
            # the x-z view so the short negative-x cube translation is legible.
            camera.azimuth = 143.0 + 47.0 * phase
            data.mocap_pos[mocap_id] = cube_centers[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            current = nodes[index]
            first_arm_geom = int(renderer.scene.ngeom)
            update_softrobot_geoms(
                renderer.scene,
                current,
                radius=geometry.arm_radius,
                rgba=(0.20, 0.43, 0.76, 1.0),
                clear_existing=False,
            )
            if args.realistic_scene:
                for geom_index in range(first_arm_geom, int(renderer.scene.ngeom)):
                    geom = renderer.scene.geoms[geom_index]
                    geom.rgba[:] = (0.055, 0.30, 0.56, 1.0)
                    geom.specular = 0.62
                    geom.shininess = 0.48
            rendered = renderer.render()
            push_distance = initial_cube_x - float(cube_centers[index, 0])
            success = bool(
                index == len(nodes) - 1
                and time_value >= start_hold + motion_duration - 0.04
            )
            if not args.clean_render:
                platform_value = visual_verification[
                    "minimum_arm_platform_clearance_m"
                ]
                rendered = _overlay(
                    rendered,
                    stage=stage_names[int(arrays["stage_ids"][index])],
                    push_distance=push_distance,
                    contact=index >= contact_index,
                    platform_clearance=(
                        0.0 if platform_value is None else float(platform_value)
                    ),
                    extended_wall_clearance=float(
                        visual_verification["minimum_extended_wall_clearance_m"]
                    ),
                    success=success,
                )
                rendered = _sac_badge(
                    rendered,
                    checkpoint_label=label,
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
    positive = (
        args.width,
        args.height,
        args.fps,
        args.playback_speed,
        args.platform_x_half_size,
        args.platform_y_half_size,
        args.platform_top_z,
        args.platform_thickness,
        args.cube_size,
        args.contact_margin,
        args.maximum_push_distance,
        args.required_push_distance,
    )
    if (
        min(positive) <= 0
        or args.wall_negative_x_extension < 0
        or (
            args.visual_push_target_distance is not None
            and args.visual_push_target_distance <= 0
        )
        or (
            args.visual_push_target_distance is not None
            and args.visual_push_target_distance > args.maximum_push_distance
        )
        or args.required_push_distance > args.maximum_push_distance
    ):
        raise ValueError("visual push geometry and render settings must be positive")
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
    trajectory = (
        output.with_name(output.stem + "_trajectory.npz")
        if args.trajectory_output is None
        else Path(args.trajectory_output).expanduser().resolve()
    )
    if output.exists() or trajectory.exists():
        raise FileExistsError("visual push output already exists")
    geometry = WallRouteGeometry.from_dict(
        yaml.safe_load(paths["task_config"].read_text(encoding="utf-8"))["task"]
    )
    arrays, policy_verification = _rollout(args, paths)
    values = _geometry_values(args, geometry)
    cube_centers, visual_verification = _verify_visual_task(
        args, arrays, geometry, values
    )
    arrays["visual_cube_centers"] = cube_centers
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    with trajectory.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    video, preview, duration = _render(
        args, arrays, geometry, values, cube_centers, visual_verification
    )
    result = {
        "kind": "manisoft_terminal_force_free_visual_push_replay",
        "model": str(paths["model"]),
        "model_sha256": _sha256(paths["model"]),
        "teacher_episode": str(paths["teacher_episode"]),
        "teacher_episode_sha256": _sha256(paths["teacher_episode"]),
        "trajectory": str(trajectory),
        "trajectory_sha256": _sha256(trajectory),
        "video": str(video),
        "video_sha256": _sha256(video),
        "preview": str(preview),
        "rendered_duration_seconds": duration,
        "playback_speed": args.playback_speed,
        "policy_verification": policy_verification,
        "visual_task_verification": visual_verification,
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
