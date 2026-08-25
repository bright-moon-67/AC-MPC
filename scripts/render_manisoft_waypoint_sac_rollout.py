#!/usr/bin/env python
"""Fast MuJoCo rendering for a recorded ManiSoft waypoint-SAC rollout."""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw

import manisoft.asset as manisoft_asset
from manisoft.asset import Asset
from manisoft.visualize.mujoco_viewer import update_manisoft_softrobot_geoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=2.5,
        help="Simulation-time acceleration in the output video.",
    )
    parser.add_argument("--camera-azimuth", type=float, default=138.0)
    parser.add_argument("--camera-elevation", type=float, default=-27.0)
    parser.add_argument("--camera-distance", type=float, default=1.55)
    parser.add_argument(
        "--softrobot-body-radius",
        type=float,
        default=0.024,
        help="Rendered soft-robot body radius in metres.",
    )
    parser.add_argument(
        "--softrobot-band-radius",
        type=float,
        default=0.028,
        help="Rendered soft-robot band radius in metres.",
    )
    parser.add_argument(
        "--obstacle-size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional full obstacle dimensions in metres in the obstacle's "
            "local x/y/z frame."
        ),
    )
    parser.add_argument(
        "--obstacle-center-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Optional obstacle centre in world x/y coordinates (metres).",
    )
    parser.add_argument(
        "--obstacle-yaw-degrees",
        type=float,
        default=0.0,
        help="Obstacle local-x yaw in world coordinates.",
    )
    return parser.parse_args()


def _vector(values) -> str:
    return " ".join(str(float(value)) for value in values)


def _mesh_scale(asset_path: Path, requested_size) -> np.ndarray:
    mesh = trimesh.load(asset_path, force="mesh")
    extents = np.asarray(mesh.extents, dtype=np.float64)
    size = np.asarray(requested_size, dtype=np.float64)
    if extents.shape != (3,) or np.any(extents <= 0):
        raise ValueError(f"invalid asset extents: {asset_path}")
    return size / extents


def _build_model(
    path_anchors: np.ndarray,
    table_x_bounds: np.ndarray,
    table_y_bounds: np.ndarray,
    table_surface_z: float,
    obstacle_size: np.ndarray | None = None,
    obstacle_center_xy: np.ndarray | None = None,
    obstacle_yaw_degrees: float = 0.0,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    table_center = np.asarray(
        [
            np.mean(table_x_bounds),
            np.mean(table_y_bounds),
            0.5 * table_surface_z,
        ],
        dtype=np.float64,
    )
    table_size = np.asarray(
        [
            np.ptp(table_x_bounds),
            np.ptp(table_y_bounds),
            table_surface_z,
        ],
        dtype=np.float64,
    )
    root = ET.Element("mujoco", model="manisoft_waypoint_sac_video")
    ET.SubElement(root, "compiler", angle="degree")
    ET.SubElement(
        root,
        "statistic",
        center=_vector([table_center[0], table_center[1], 0.44]),
        extent="1.15",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.58 0.58 0.58",
        diffuse="0.76 0.76 0.76",
        specular="0.12 0.12 0.12",
    )
    ET.SubElement(visual, "rgba", haze="0.88 0.91 0.96 1")

    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.74 0.77 0.82",
        rgb2="0.94 0.95 0.97",
        width="512",
        height="3072",
    )
    table_path = Path(Asset.TABLE.value).expanduser()
    if not table_path.is_absolute():
        table_path = (
            Path(manisoft_asset.__file__).resolve().parents[1] / table_path
        )
    table_path = table_path.resolve()
    if not table_path.is_file():
        raise FileNotFoundError(table_path)
    ET.SubElement(
        assets,
        "mesh",
        name="table_mesh",
        file=str(table_path),
        scale=_vector(_mesh_scale(table_path, table_size)),
    )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        pos="-0.8 -0.3 2.3",
        dir="0.35 0.45 -1",
        diffuse="0.88 0.88 0.88",
    )
    ET.SubElement(
        world,
        "light",
        pos="0.9 1.0 1.7",
        dir="-0.35 -0.45 -1",
        diffuse="0.48 0.50 0.56",
    )
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="2 2 0.1",
        rgba="0.80 0.79 0.76 1",
        contype="0",
        conaffinity="0",
    )
    table = ET.SubElement(
        world, "body", name="table", pos=_vector(table_center)
    )
    ET.SubElement(
        table,
        "geom",
        type="mesh",
        mesh="table_mesh",
        rgba="0.72 0.72 0.70 1",
        contype="0",
        conaffinity="0",
    )

    if obstacle_size is not None:
        if obstacle_center_xy is None:
            raise ValueError("obstacle centre is required when size is set")
        obstacle_center = np.asarray(
            [
                obstacle_center_xy[0],
                obstacle_center_xy[1],
                table_surface_z + 0.5 * obstacle_size[2],
            ],
            dtype=np.float64,
        )
        obstacle = ET.SubElement(
            world,
            "body",
            name="tabletop_obstacle",
            pos=_vector(obstacle_center),
            euler=_vector([0.0, 0.0, obstacle_yaw_degrees]),
        )
        ET.SubElement(
            obstacle,
            "geom",
            name="tabletop_obstacle_geom",
            type="box",
            size=_vector(0.5 * obstacle_size),
            rgba="0.86 0.24 0.08 1.0",
            contype="0",
            conaffinity="0",
        )

    # The dense reference consists of straight chords.  Render them as thin
    # capsules and show every commanded point as a numbered marker in the HUD.
    for index, (start, end) in enumerate(
        zip(path_anchors[:-1], path_anchors[1:])
    ):
        ET.SubElement(
            world,
            "geom",
            name=f"path_segment_{index}",
            type="capsule",
            fromto=_vector(np.concatenate((start, end))),
            size="0.004",
            rgba="0.10 0.55 0.95 0.72",
            contype="0",
            conaffinity="0",
        )
    for index, point in enumerate(path_anchors):
        is_start = index == 0
        is_final = index == len(path_anchors) - 1
        rgba = (
            "0.55 0.58 0.62 0.90"
            if is_start
            else ("0.20 0.90 0.35 0.95" if is_final else "0.10 0.55 0.95 0.92")
        )
        radius = 0.012 if is_start else (0.017 if is_final else 0.014)
        ET.SubElement(
            world,
            "site",
            name=f"waypoint_{index}",
            type="sphere",
            pos=_vector(point),
            size=str(radius),
            rgba=rgba,
        )

    target_body = ET.SubElement(world, "body", name="moving_reference")
    ET.SubElement(
        target_body,
        "geom",
        type="sphere",
        size="0.017",
        rgba="1.0 0.72 0.05 0.90",
        contype="0",
        conaffinity="0",
    )

    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode")
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _hud(
    frame: np.ndarray,
    *,
    point_count: int,
    completed: int,
    error_mm: float,
    cross_track_mm: float,
    simulation_time: float,
    final_frame: bool,
    success: bool,
    obstacle_size_cm: np.ndarray | None,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (18, 18, 690, 116), radius=10, fill=(10, 16, 25, 188)
    )
    draw.text(
        (32, 29),
        f"ManiSoft SAC | {point_count}-segment polyline tracking",
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (32, 57),
        f"captured {min(completed, point_count)}/{point_count}   "
        f"target error {error_mm:5.1f} mm",
        fill=(225, 232, 240, 255),
    )
    draw.text(
        (32, 83),
        f"cross-track {cross_track_mm:4.1f} mm   sim t={simulation_time:5.2f} s",
        fill=(205, 215, 228, 255),
    )
    draw.rectangle((470, 29, 484, 43), fill=(26, 140, 242, 255))
    draw.text((491, 27), "piecewise-linear reference", fill=(235, 240, 247, 255))
    draw.ellipse((470, 57, 485, 72), fill=(255, 184, 13, 255))
    draw.text((491, 55), "moving target", fill=(235, 240, 247, 255))
    if obstacle_size_cm is not None:
        draw.rectangle((470, 84, 484, 98), fill=(219, 61, 20, 255))
        draw.text(
            (491, 82),
            "obstacle "
            + " x ".join(f"{value:g}" for value in obstacle_size_cm)
            + " cm",
            fill=(235, 240, 247, 255),
        )
    if final_frame:
        color = (33, 194, 92, 225) if success else (224, 66, 55, 225)
        label = "TRACKING COMPLETE" if success else "TRACKING FAILED"
        width = 224
        left = (image.width - width) // 2
        draw.rounded_rectangle(
            (left, image.height - 72, left + width, image.height - 24),
            radius=10,
            fill=color,
        )
        draw.text(
            (left + 40, image.height - 57),
            label,
            fill=(255, 255, 255, 255),
        )
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.fps) <= 0 or args.playback_speed <= 0:
        raise ValueError("resolution, fps, and playback speed must be positive")
    if min(args.softrobot_body_radius, args.softrobot_band_radius) <= 0.0:
        raise ValueError("rendered soft-robot radii must be positive")
    if (args.obstacle_size is None) != (args.obstacle_center_xy is None):
        raise ValueError(
            "--obstacle-size and --obstacle-center-xy must be provided together"
        )
    obstacle_size = (
        None
        if args.obstacle_size is None
        else np.asarray(args.obstacle_size, dtype=np.float64)
    )
    obstacle_center_xy = (
        None
        if args.obstacle_center_xy is None
        else np.asarray(args.obstacle_center_xy, dtype=np.float64)
    )
    if obstacle_size is not None and np.any(obstacle_size <= 0.0):
        raise ValueError("obstacle dimensions must be positive")
    trajectory_path = Path(args.trajectory).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)

    with np.load(trajectory_path, allow_pickle=False) as archive:
        positions = np.asarray(archive["softrobot_positions"], dtype=np.float32)
        directors = np.asarray(archive["softrobot_directors"], dtype=np.float32)
        tips = np.asarray(archive["tip_position"], dtype=np.float32)
        targets = np.asarray(archive["target_tip"], dtype=np.float32)
        anchors = np.asarray(archive["path_anchors"], dtype=np.float32)
        completed = np.asarray(archive["waypoints_completed"], dtype=np.int64)
        distances = np.asarray(archive["distance"], dtype=np.float64)
        cross_track = np.asarray(
            archive["cross_track_distance"], dtype=np.float64
        )
        table_x_bounds = np.asarray(
            archive.get("table_x_bounds", [0.15, 0.65]), dtype=np.float64
        )
        table_y_bounds = np.asarray(
            archive.get("table_y_bounds", [0.28, 0.72]), dtype=np.float64
        )
        table_surface_z = float(
            np.asarray(archive.get("table_surface_z", 0.36))
        )
        control_hz = float(np.asarray(archive["control_hz"]))
        success = bool(np.asarray(archive["success"]))
    if (
        len(positions) < 2
        or len(positions) != len(directors)
        or len(positions) != len(targets)
    ):
        raise ValueError("trajectory frame arrays have inconsistent lengths")

    model, data = _build_model(
        anchors,
        table_x_bounds,
        table_y_bounds,
        table_surface_z,
        obstacle_size=obstacle_size,
        obstacle_center_xy=obstacle_center_xy,
        obstacle_yaw_degrees=args.obstacle_yaw_degrees,
    )
    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "moving_reference"
    )
    natural_duration = (len(positions) - 1) / control_hz
    video_duration = natural_duration / args.playback_speed
    frame_count = max(2, int(round(video_duration * args.fps)))
    source_indices = np.rint(
        np.linspace(0, len(positions) - 1, frame_count)
    ).astype(np.int64)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (
        float(np.mean(anchors[:, 0])),
        float(np.mean(anchors[:, 1])),
        float(max(table_surface_z + 0.08, np.mean(anchors[:, 2]))),
    )
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.with_name(output_path.stem + "_preview.png")
    final_frame_path = output_path.with_name(output_path.stem + "_final.png")
    writer = imageio.get_writer(
        output_path,
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
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        for frame_index, source_index in enumerate(source_indices):
            model.body_pos[target_body_id] = targets[source_index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            update_manisoft_softrobot_geoms(
                renderer.scene,
                positions[source_index],
                directors[source_index],
                body_radius=args.softrobot_body_radius,
                band_radius=args.softrobot_band_radius,
                band_length=0.013,
                pipe_offset=args.softrobot_band_radius - 0.001,
                pipe_radius=0.0035,
                body_rgba=(0.50, 0.50, 0.50, 1.0),
                detail_rgba=(0.025, 0.025, 0.025, 1.0),
                clear_existing=False,
            )
            frame = renderer.render()
            final_frame = frame_index == frame_count - 1
            frame = _hud(
                frame,
                point_count=len(anchors) - 1,
                completed=int(completed[source_index]),
                error_mm=float(distances[source_index] * 1000.0),
                cross_track_mm=float(cross_track[source_index] * 1000.0),
                simulation_time=float(source_index / control_hz),
                final_frame=final_frame,
                success=success,
                obstacle_size_cm=(
                    None if obstacle_size is None else 100.0 * obstacle_size
                ),
            )
            writer.append_data(frame)
            if frame_index == 0:
                imageio.imwrite(preview_path, frame)
            if final_frame:
                imageio.imwrite(final_frame_path, frame)
    finally:
        renderer.close()
        writer.close()

    report = {
        "kind": "manisoft_waypoint_sac_video",
        "video": str(output_path),
        "preview": str(preview_path),
        "final_frame": str(final_frame_path),
        "trajectory": str(trajectory_path),
        "source_frames": len(positions),
        "video_frames": frame_count,
        "simulation_seconds": natural_duration,
        "video_seconds": video_duration,
        "playback_speed": args.playback_speed,
        "resolution": [args.width, args.height],
        "fps": args.fps,
        "success": success,
        "point_count": len(anchors) - 1,
        "segment_lengths_m": np.linalg.norm(
            np.diff(anchors, axis=0), axis=1
        ).tolist(),
        "final_tip_position": tips[-1].tolist(),
        "obstacle": (
            None
            if obstacle_size is None
            else {
                "size_m": obstacle_size.tolist(),
                "center_m": [
                    float(obstacle_center_xy[0]),
                    float(obstacle_center_xy[1]),
                    float(table_surface_z + 0.5 * obstacle_size[2]),
                ],
                "yaw_degrees": float(args.obstacle_yaw_degrees),
            }
        ),
        "rendered_softrobot_body_radius_m": args.softrobot_body_radius,
        "rendered_softrobot_band_radius_m": args.softrobot_band_radius,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
