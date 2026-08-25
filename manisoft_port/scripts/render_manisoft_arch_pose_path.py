#!/usr/bin/env python
"""Fast video renderer for a certified arch-pose path bank."""

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
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--playback-speed", type=float, default=5.0)
    parser.add_argument("--camera-azimuth", type=float, default=138.0)
    parser.add_argument("--camera-elevation", type=float, default=-22.0)
    parser.add_argument("--camera-distance", type=float, default=1.85)
    parser.add_argument("--initial-camera-distance", type=float, default=2.65)
    return parser.parse_args()


def _vector(values) -> str:
    return " ".join(str(float(value)) for value in values)


def _mesh_scale(asset_path: Path, requested_size) -> np.ndarray:
    mesh = trimesh.load(asset_path, force="mesh")
    extents = np.asarray(mesh.extents, dtype=np.float64)
    return np.asarray(requested_size, dtype=np.float64) / extents


def _build_model(anchors: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData]:
    root = ET.Element("mujoco", model="manisoft_vertical_arch_path")
    ET.SubElement(root, "compiler", angle="degree")
    ET.SubElement(root, "statistic", center="0.24 0.32 0.52", extent="1.15")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.58 0.58 0.58",
        diffuse="0.76 0.76 0.76",
        specular="0.12 0.12 0.12",
    )
    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.72 0.76 0.82",
        rgb2="0.95 0.96 0.98",
        width="512",
        height="3072",
    )
    table_path = Path(Asset.TABLE.value).expanduser()
    if not table_path.is_absolute():
        table_path = (
            Path(manisoft_asset.__file__).resolve().parents[1] / table_path
        )
    table_path = table_path.resolve()
    ET.SubElement(
        assets,
        "mesh",
        name="table_mesh",
        file=str(table_path),
        scale=_vector(_mesh_scale(table_path, [0.50, 0.44, 0.36])),
    )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        pos="-0.8 -0.5 2.4",
        dir="0.35 0.45 -1",
        diffuse="0.90 0.90 0.90",
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
    table = ET.SubElement(world, "body", name="table", pos="0.40 0.50 0.18")
    ET.SubElement(
        table,
        "geom",
        type="mesh",
        mesh="table_mesh",
        rgba="0.72 0.72 0.70 1",
        contype="0",
        conaffinity="0",
    )
    # Raised mounting pedestal corresponding to softrobot.start z=0.75 m.
    ET.SubElement(
        world,
        "geom",
        name="pedestal",
        type="cylinder",
        pos="0 0 0.375",
        size="0.075 0.375",
        rgba="0.24 0.27 0.31 1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        world,
        "geom",
        name="base_mount",
        type="cylinder",
        pos="0 0 0.75",
        size="0.095 0.035",
        rgba="0.12 0.14 0.17 1",
        contype="0",
        conaffinity="0",
    )
    for index, (start, end) in enumerate(zip(anchors[:-1], anchors[1:])):
        ET.SubElement(
            world,
            "geom",
            name=f"path_{index}",
            type="capsule",
            fromto=_vector(np.concatenate((start, end))),
            size="0.004",
            rgba="0.10 0.55 0.95 0.78",
            contype="0",
            conaffinity="0",
        )
    for index, point in enumerate(anchors):
        ET.SubElement(
            world,
            "site",
            name=f"waypoint_{index}",
            type="sphere",
            pos=_vector(point),
            size="0.014",
            rgba=(
                "0.20 0.90 0.35 0.96"
                if index == len(anchors) - 1
                else "0.10 0.55 0.95 0.94"
            ),
        )
    target = ET.SubElement(world, "body", name="moving_reference")
    ET.SubElement(
        target,
        "geom",
        type="sphere",
        size="0.017",
        rgba="1.0 0.72 0.05 0.92",
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
    point_index: int,
    point_count: int,
    angle: float,
    time_seconds: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (18, 18, 650, 112), radius=10, fill=(10, 16, 25, 188)
    )
    draw.text(
        (32, 29),
        "ManiSoft | raised-base vertical arch manipulation",
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (32, 57),
        f"waypoint {point_index + 1}/{point_count}   "
        f"tip-down error {angle:4.1f} deg",
        fill=(225, 232, 240, 255),
    )
    draw.text(
        (32, 83),
        f"certified straight-segment baseline   sim t={time_seconds:5.1f} s",
        fill=(205, 215, 228, 255),
    )
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    if min(
        args.width,
        args.height,
        args.fps,
        args.playback_speed,
        args.camera_distance,
        args.initial_camera_distance,
    ) <= 0:
        raise ValueError("render settings must be positive")
    bank_path = Path(args.bank).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    with np.load(bank_path, allow_pickle=False) as archive:
        if str(np.asarray(archive["kind"]).reshape(()).item()) != (
            "manisoft_table_arch_pose_path_bank"
        ):
            raise ValueError("unexpected path-bank kind")
        nodes = np.asarray(archive["video_node_positions"], dtype=np.float32)
        directors = np.asarray(
            archive["video_element_directors"], dtype=np.float32
        )
        anchors = np.asarray(archive["tip_positions"], dtype=np.float32)
        transition_count = int(archive["transition_node_positions"].shape[0])
        transition_steps = int(archive["transition_node_positions"].shape[1])
        stride = int(np.asarray(archive["video_stride"]).reshape(()).item())
        control_dt = float(np.asarray(archive["control_dt"]).reshape(()).item())
        angles_full = np.asarray(
            archive["transition_orientation_errors_degrees"],
            dtype=np.float32,
        )
    transition_video_frames = (transition_steps + stride - 1) // stride
    entry_frames = len(nodes) - transition_count * transition_video_frames
    targets = np.repeat(anchors[0:1], len(nodes), axis=0)
    angles = np.zeros(len(nodes), dtype=np.float32)
    point_indices = np.zeros(len(nodes), dtype=np.int64)
    for segment in range(transition_count):
        start = entry_frames + segment * transition_video_frames
        end = start + transition_video_frames
        sampled_angles = angles_full[segment, ::stride]
        angles[start:end] = sampled_angles[: end - start]
        ramp_video_frames = min(400 // stride, end - start)
        fraction = np.ones(end - start, dtype=np.float64)
        if ramp_video_frames:
            raw = (np.arange(ramp_video_frames) + 1) / ramp_video_frames
            fraction[:ramp_video_frames] = raw**3 * (
                10.0 - 15.0 * raw + 6.0 * raw**2
            )
        targets[start:end] = (
            (1.0 - fraction[:, None]) * anchors[segment]
            + fraction[:, None] * anchors[segment + 1]
        )
        point_indices[start:end] = segment + 1
    if entry_frames:
        tip_tangent = nodes[entry_frames - 1, -1] - nodes[entry_frames - 1, -2]
        tip_tangent /= np.linalg.norm(tip_tangent)
        angles[:entry_frames] = np.rad2deg(
            np.arccos(np.clip(-tip_tangent[2], -1.0, 1.0))
        )

    source_hz = 1.0 / (control_dt * stride)
    natural_duration = (len(nodes) - 1) / source_hz
    video_duration = natural_duration / args.playback_speed
    frame_count = max(2, int(round(video_duration * args.fps)))
    source_indices = np.rint(
        np.linspace(0, len(nodes) - 1, frame_count)
    ).astype(np.int64)

    model, data = _build_model(anchors)
    target_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "moving_reference"
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.23, 0.30, 0.58)
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation

    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_name(output.stem + "_preview.png")
    final = output.with_name(output.stem + "_final.png")
    writer = imageio.get_writer(
        output,
        format="FFMPEG",
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
            entry_fraction = float(
                np.clip(source_index / max(entry_frames - 1, 1), 0.0, 1.0)
            )
            blend = entry_fraction**3 * (
                10.0 - 15.0 * entry_fraction + 6.0 * entry_fraction**2
            )
            camera.distance = (
                (1.0 - blend) * args.initial_camera_distance
                + blend * args.camera_distance
            )
            camera.lookat[:] = (
                0.23,
                0.30,
                (1.0 - blend) * 0.88 + blend * 0.58,
            )
            model.body_pos[target_id] = targets[source_index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            update_manisoft_softrobot_geoms(
                renderer.scene,
                nodes[source_index],
                directors[source_index],
                body_radius=0.024,
                band_radius=0.028,
                band_length=0.013,
                pipe_offset=0.027,
                pipe_radius=0.0035,
                body_rgba=(0.50, 0.50, 0.50, 1.0),
                detail_rgba=(0.025, 0.025, 0.025, 1.0),
                clear_existing=False,
            )
            frame = _hud(
                renderer.render(),
                point_index=int(point_indices[source_index]),
                point_count=len(anchors),
                angle=float(angles[source_index]),
                time_seconds=float(source_index / source_hz),
            )
            writer.append_data(frame)
            if frame_index == 0:
                imageio.imwrite(preview, frame)
            if frame_index == frame_count - 1:
                imageio.imwrite(final, frame)
    finally:
        renderer.close()
        writer.close()

    report = {
        "kind": "manisoft_vertical_arch_pose_video",
        "video": str(output),
        "preview": str(preview),
        "final_frame": str(final),
        "source_bank": str(bank_path),
        "point_count": len(anchors),
        "natural_duration_seconds": natural_duration,
        "video_duration_seconds": video_duration,
        "frame_count": frame_count,
        "playback_speed": args.playback_speed,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
