#!/usr/bin/env python
"""Remove an unnecessary constant-action hold and reintegrate the teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.manisoft_wall_crossing_sac_env import wall_crossing_metrics
from antmaze_ac.envs.table_entry_bank import pack_rod_internal_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-teacher", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--hold-start", type=int, required=True)
    parser.add_argument("--hold-stop", type=int, required=True)
    parser.add_argument("--retained-hold-steps", type=int, default=0)
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=1,
        help="Optional odd Savitzky-Golay window; 1 disables filtering.",
    )
    parser.add_argument("--savgol-polyorder", type=int, default=3)
    parser.add_argument("--bridge-start", type=int, default=None)
    parser.add_argument("--bridge-stop", type=int, default=None)
    parser.add_argument("--arch-enforcement-index", type=int, required=True)
    parser.add_argument("--arch-height", type=float, default=0.30)
    parser.add_argument("--arch-y-margin", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(env: ManiSoftTipTrackingEnv) -> dict[str, np.ndarray]:
    rod = env.sim._backend._softrobot
    return {
        "physical_state": np.asarray(env._physical_state(), dtype=np.float32),
        "node_positions": rod.position_collection.T.astype(np.float64, copy=True),
        "node_velocities": rod.velocity_collection.T.astype(np.float64, copy=True),
        "element_directors": rod.director_collection.transpose(2, 0, 1).astype(
            np.float64, copy=True
        ),
        "element_omegas": rod.omega_collection.T.astype(np.float64, copy=True),
        "rod_internal_state": pack_rod_internal_state(rod),
    }


def _arch_height(
    nodes: np.ndarray, geometry: WallRouteGeometry, margin: float
) -> float:
    checked = nodes[geometry.mounting_exempt_nodes :]
    mask = (
        (checked[:, 1] >= geometry.wall_minimum[1] - margin)
        & (checked[:, 1] <= geometry.wall_maximum[1] + margin)
    )
    return float(np.min(checked[mask, 2])) if np.any(mask) else float("nan")


def _longest_constant_action_run(actions: np.ndarray, tolerance: float = 1e-8) -> int:
    if len(actions) < 2:
        return 0
    constant = np.linalg.norm(np.diff(actions.astype(np.float64), axis=0), axis=1) < tolerance
    longest = current = 0
    for value in constant:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def main() -> None:
    args = parse_args()
    source = Path(args.source_teacher).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    task_config = Path(args.task_config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (source, scenario, task_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    geometry = WallRouteGeometry.from_dict(
        yaml.safe_load(task_config.read_text(encoding="utf-8"))["task"]
    )
    with np.load(source, allow_pickle=False) as archive:
        teacher = {key: np.asarray(archive[key]) for key in archive.files}
    source_actions = np.asarray(teacher["actions"], dtype=np.float32)
    source_stages = np.asarray(teacher["stage_ids"], dtype=np.int8)
    start, stop = args.hold_start, args.hold_stop
    if not 0 <= start < stop <= len(source_actions):
        raise ValueError("hold interval is outside the source action sequence")
    if args.retained_hold_steps < 0:
        raise ValueError("retained-hold-steps must be non-negative")
    if args.savgol_window != 1 and (
        args.savgol_window < 3
        or args.savgol_window % 2 == 0
        or args.savgol_polyorder >= args.savgol_window
    ):
        raise ValueError("Savitzky-Golay window must be odd and exceed polyorder")
    if not np.allclose(source_actions[start:stop], source_actions[start], atol=1e-8):
        raise ValueError("selected interval is not a constant-action hold")

    retained = np.repeat(
        source_actions[start : start + 1], args.retained_hold_steps, axis=0
    )
    actions = np.concatenate(
        (source_actions[:start], retained, source_actions[stop:]), axis=0
    ).astype(np.float32)
    source_indices = np.concatenate(
        (
            np.arange(start, dtype=np.int64),
            np.full(args.retained_hold_steps, start, dtype=np.int64),
            np.arange(stop, len(source_actions), dtype=np.int64),
        )
    )
    if (args.bridge_start is None) != (args.bridge_stop is None):
        raise ValueError("bridge-start and bridge-stop must be supplied together")
    if args.bridge_start is not None:
        bridge_start, bridge_stop = args.bridge_start, args.bridge_stop
        if not 0 <= bridge_start < bridge_stop < len(actions):
            raise ValueError("bridge interval is outside the retimed actions")
        phase = np.linspace(0.0, 1.0, bridge_stop - bridge_start + 1)
        weight = (phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2))[:, None]
        actions[bridge_start : bridge_stop + 1] = (
            (1.0 - weight) * actions[bridge_start]
            + weight * actions[bridge_stop]
        ).astype(np.float32)
    if args.savgol_window != 1:
        from scipy.signal import savgol_filter

        filtered = savgol_filter(
            actions,
            args.savgol_window,
            args.savgol_polyorder,
            axis=0,
            mode="interp",
        ).astype(np.float32)
        # Savitzky-Golay can overshoot a rate-constrained signal slightly.
        # Project it back into the same per-channel 0.003/step control envelope.
        rate_limited = np.empty_like(filtered)
        previous = np.zeros(18, dtype=np.float32)
        for index, desired in enumerate(filtered):
            current = previous + np.clip(desired - previous, -0.003, 0.003)
            rate_limited[index] = np.clip(current, -0.60, 0.60)
            previous = rate_limited[index]
        actions = rate_limited
    action_delta = np.diff(
        np.vstack((np.zeros((1, 18), dtype=np.float32), actions)), axis=0
    )
    maximum_action_delta = float(np.max(np.abs(action_delta)))
    if maximum_action_delta > 0.003 + 1e-7:
        raise RuntimeError("retimed actions violate the 0.003 action-rate limit")

    seed = int(np.asarray(teacher["episode_seed"]).reshape(()))
    control_dt = float(np.asarray(teacher["control_dt"]).reshape(()))
    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=geometry.target,
        episode_steps=len(actions) + 1,
        absolute_action_limit=0.60,
        muscle_torque_scale=45.0,
    )
    env.reset(seed=seed)
    state_keys = (
        "physical_state",
        "node_positions",
        "node_velocities",
        "element_directors",
        "element_omegas",
        "rod_internal_state",
    )
    initial = _capture(env)
    states = {key: [initial[key]] for key in state_keys}
    wall, ground, speed, fraction, tip_x, arch = [], [], [], [], [], []

    def append_metrics() -> None:
        rod = env.sim._backend._softrobot
        nodes = rod.position_collection.T
        metrics = wall_crossing_metrics(
            geometry, nodes, rod.velocity_collection.T, 1
        )
        wall.append(metrics.wall_clearance)
        ground.append(metrics.ground_clearance)
        speed.append(metrics.tip_speed)
        fraction.append(metrics.distal_crossed_fraction)
        tip_x.append(metrics.tip_x)
        arch.append(_arch_height(nodes, geometry, args.arch_y_margin))

    append_metrics()
    try:
        for step, action in enumerate(actions, start=1):
            env.muscle.set_activation(action.reshape(6, 3))
            env.sim.step_with_torque_callback(lambda lengths: env.muscle.evaluate(lengths))
            capture = _capture(env)
            for key in state_keys:
                states[key].append(capture[key])
            append_metrics()
            if wall[-1] < 0:
                raise RuntimeError(f"retimed teacher intersects wall at step {step}")
            if ground[-1] < -geometry.ground_violation_tolerance:
                raise RuntimeError(f"retimed teacher violates ground at step {step}")
    finally:
        env.close()

    arch_values = np.asarray(arch, dtype=np.float32)
    if np.nanmin(arch_values[args.arch_enforcement_index :]) < args.arch_height:
        raise RuntimeError("retimed teacher violates the enforced arch height")
    stage_ids = np.concatenate(
        (
            source_stages[:1],
            source_stages[np.minimum(source_indices + 1, len(source_stages) - 1)],
        )
    ).astype(np.int8)
    payload = {
        "schema_version": np.asarray(1, dtype=np.int64),
        # Keep the established on-disk interface; the retiming/filter provenance
        # is carried by the additional fields below.
        "kind": np.asarray("manisoft_smooth_wall_teacher_episode"),
        "scenario_sha256": np.asarray(_sha256(scenario)),
        "task_config_sha256": np.asarray(_sha256(task_config)),
        "source_trajectory_sha256": np.asarray(_sha256(source)),
        "control_dt": np.asarray(control_dt, dtype=np.float64),
        "episode_seed": np.asarray(seed, dtype=np.int64),
        "route_side": np.asarray(1, dtype=np.int8),
        "time_scale": np.asarray(1.0, dtype=np.float64),
        "action_scale": np.asarray(1.0, dtype=np.float64),
        "terminal_step": np.asarray(len(actions), dtype=np.int64),
        "retimed_hold_start": np.asarray(start, dtype=np.int64),
        "retimed_hold_stop": np.asarray(stop, dtype=np.int64),
        "retained_hold_steps": np.asarray(args.retained_hold_steps, dtype=np.int64),
        "savgol_window": np.asarray(args.savgol_window, dtype=np.int64),
        "savgol_polyorder": np.asarray(args.savgol_polyorder, dtype=np.int64),
        "bridge_start": np.asarray(
            -1 if args.bridge_start is None else args.bridge_start, dtype=np.int64
        ),
        "bridge_stop": np.asarray(
            -1 if args.bridge_stop is None else args.bridge_stop, dtype=np.int64
        ),
        "required_arch_height": np.asarray(args.arch_height, dtype=np.float64),
        "arch_y_margin": np.asarray(args.arch_y_margin, dtype=np.float64),
        "stage_ids": stage_ids,
        "source_action_indices": source_indices,
        "actions": actions,
        "wall_clearances": np.asarray(wall, dtype=np.float32),
        "ground_clearances": np.asarray(ground, dtype=np.float32),
        "tip_speeds": np.asarray(speed, dtype=np.float32),
        "distal_crossed_fractions": np.asarray(fraction, dtype=np.float32),
        "tip_x": np.asarray(tip_x, dtype=np.float32),
        "arch_heights": arch_values,
        **{key: np.asarray(values) for key, values in states.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(output)
    final_tip = payload["node_positions"][-1, -1]
    summary = {
        "kind": "manisoft_retimed_smooth_wall_teacher_summary",
        "episode": str(output),
        "episode_sha256": _sha256(output),
        "source_teacher": str(source),
        "source_teacher_sha256": _sha256(source),
        "removed_hold_interval": [start, stop],
        "removed_hold_steps": stop - start - args.retained_hold_steps,
        "savgol_window": args.savgol_window,
        "savgol_polyorder": args.savgol_polyorder,
        "minimum_jerk_bridge": (
            None
            if args.bridge_start is None
            else [args.bridge_start, args.bridge_stop]
        ),
        "longest_constant_action_run_steps": _longest_constant_action_run(actions),
        "control_steps": len(actions),
        "duration_seconds": len(actions) * control_dt,
        "maximum_action_delta": maximum_action_delta,
        "minimum_wall_clearance_m": float(np.min(wall)),
        "minimum_ground_clearance_m": float(np.min(ground)),
        "minimum_enforced_arch_height_m": float(
            np.nanmin(arch_values[args.arch_enforcement_index :])
        ),
        "final_tip_xyz_m": final_tip.tolist(),
        "final_tip_speed_mps": float(speed[-1]),
        "final_distal_crossed_fraction": float(fraction[-1]),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
