#!/usr/bin/env python
"""Stitch a safe smooth prefix to a dynamically searched arched return."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.manisoft_wall_crossing_sac_env import wall_crossing_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-teacher", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch-y-margin", type=float, default=0.05)
    parser.add_argument("--required-arch-height", type=float, default=0.30)
    parser.add_argument(
        "--maximum-tip-plane-distance", type=float, default=0.01
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arch_height(nodes: np.ndarray, geometry: WallRouteGeometry, margin: float) -> float:
    checked = nodes[geometry.mounting_exempt_nodes :]
    mask = (
        (checked[:, 1] >= geometry.wall_minimum[1] - margin)
        & (checked[:, 1] <= geometry.wall_maximum[1] + margin)
    )
    return float(np.min(checked[mask, 2])) if np.any(mask) else float("nan")


def main() -> None:
    args = parse_args()
    source = Path(args.source_teacher).expanduser().resolve()
    branch_path = Path(args.branch).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    task_config = Path(args.task_config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (source, branch_path, scenario, task_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    geometry = WallRouteGeometry.from_dict(
        yaml.safe_load(task_config.read_text(encoding="utf-8"))["task"]
    )
    with np.load(source, allow_pickle=False) as archive:
        teacher = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(branch_path, allow_pickle=False) as archive:
        branch = {key: np.asarray(archive[key]) for key in archive.files}
    start = int(np.asarray(branch["start_index"]).reshape(()))
    state_keys = (
        "physical_state",
        "node_positions",
        "node_velocities",
        "element_directors",
        "element_omegas",
        "rod_internal_state",
    )
    for key in state_keys:
        if not np.allclose(
            teacher[key][start], branch[f"start_{key}"], atol=1e-10, rtol=0
        ):
            raise RuntimeError(f"branch start does not match teacher {key}")
    arrays = {
        key: np.concatenate((teacher[key][: start + 1], branch[key]), axis=0)
        for key in state_keys
    }
    arrays["actions"] = np.concatenate(
        (teacher["actions"][:start], branch["actions"]), axis=0
    ).astype(np.float32)
    state_count = len(arrays["physical_state"])
    if len(arrays["actions"]) != state_count - 1:
        raise RuntimeError("stitched actions and states are not aligned")
    action_previous = np.vstack((np.zeros((1, 18)), arrays["actions"][:-1]))
    maximum_action_delta = float(np.max(np.abs(arrays["actions"] - action_previous)))
    if maximum_action_delta > 0.003 + 1e-7:
        raise RuntimeError("stitched teacher violates the 0.003 action-rate limit")

    wall, ground, speed, fraction, tip_x, arch = [], [], [], [], [], []
    for nodes, velocities in zip(arrays["node_positions"], arrays["node_velocities"]):
        metrics = wall_crossing_metrics(geometry, nodes, velocities, 1)
        wall.append(metrics.wall_clearance)
        ground.append(metrics.ground_clearance)
        speed.append(metrics.tip_speed)
        fraction.append(metrics.distal_crossed_fraction)
        tip_x.append(metrics.tip_x)
        arch.append(_arch_height(nodes, geometry, args.arch_y_margin))
    wall = np.asarray(wall, dtype=np.float32)
    ground = np.asarray(ground, dtype=np.float32)
    speed = np.asarray(speed, dtype=np.float32)
    fraction = np.asarray(fraction, dtype=np.float32)
    tip_x = np.asarray(tip_x, dtype=np.float32)
    arch = np.asarray(arch, dtype=np.float32)
    if np.min(wall) < 0 or np.min(ground) < -geometry.ground_violation_tolerance:
        raise RuntimeError("stitched teacher violates virtual geometry")
    branch_arch = arch[start:]
    if np.nanmin(branch_arch) < args.required_arch_height:
        raise RuntimeError("stitched branch falls below the required arch height")
    final_nodes = arrays["node_positions"][-1]
    if abs(float(final_nodes[-1, 0])) > args.maximum_tip_plane_distance:
        raise RuntimeError("stitched teacher exceeds the allowed terminal x distance")
    if float(final_nodes[-1, 1]) <= geometry.wall_maximum[1]:
        raise RuntimeError("stitched teacher tip is not beyond the wall")

    stage_ids = np.concatenate(
        (
            teacher["stage_ids"][: start + 1],
            np.full(len(branch["physical_state"]), 4, dtype=np.int8),
        )
    )
    stage_ids[-1] = 5
    payload = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "kind": np.asarray("manisoft_smooth_wall_teacher_episode"),
        "scenario_sha256": np.asarray(_sha256(scenario)),
        "task_config_sha256": np.asarray(_sha256(task_config)),
        "source_trajectory_sha256": np.asarray(_sha256(source)),
        "control_dt": np.asarray(float(teacher["control_dt"]), dtype=np.float64),
        "episode_seed": np.asarray(int(teacher["episode_seed"]), dtype=np.int64),
        "route_side": np.asarray(1, dtype=np.int8),
        "time_scale": np.asarray(1.0, dtype=np.float64),
        "action_scale": np.asarray(1.0, dtype=np.float64),
        "terminal_step": np.asarray(len(arrays["actions"]), dtype=np.int64),
        "branch_start_index": np.asarray(start, dtype=np.int64),
        "required_arch_height": np.asarray(args.required_arch_height, dtype=np.float64),
        "arch_y_margin": np.asarray(args.arch_y_margin, dtype=np.float64),
        "stage_ids": stage_ids,
        "wall_clearances": wall,
        "ground_clearances": ground,
        "tip_speeds": speed,
        "distal_crossed_fractions": fraction,
        "tip_x": tip_x,
        "arch_heights": arch,
        **arrays,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(output)
    summary = {
        "kind": "manisoft_arched_smooth_wall_teacher_summary",
        "episode": str(output),
        "episode_sha256": _sha256(output),
        "source_teacher": str(source),
        "source_teacher_sha256": _sha256(source),
        "searched_branch": str(branch_path),
        "searched_branch_sha256": _sha256(branch_path),
        "branch_start_index": start,
        "control_steps": len(arrays["actions"]),
        "duration_seconds": len(arrays["actions"]) * float(teacher["control_dt"]),
        "maximum_action_delta": maximum_action_delta,
        "minimum_wall_clearance_m": float(np.min(wall)),
        "minimum_ground_clearance_m": float(np.min(ground)),
        "minimum_branch_arch_height_m": float(np.nanmin(branch_arch)),
        "final_arch_height_m": float(arch[-1]),
        "final_tip_xyz_m": final_nodes[-1].tolist(),
        "final_tip_speed_mps": float(speed[-1]),
        "final_distal_crossed_fraction": float(fraction[-1]),
        "final_target_plane_distance_m": abs(float(tip_x[-1])),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
