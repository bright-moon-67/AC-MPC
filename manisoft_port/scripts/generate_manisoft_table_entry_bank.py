#!/usr/bin/env python
"""Generate and certify upright-to-table trajectories in the real simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from antmaze_ac.envs.kinematic_push_task import segment_aabb_distance
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.table_entry_bank import pack_rod_internal_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--seeds", default="configs/manisoft_table_entry_seeds.yaml"
    )
    parser.add_argument(
        "--output",
        default="data/processed/manisoft_table_entry_bank_v1/entry_bank.npz",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _minimum_jerk(fraction: float) -> float:
    value = float(np.clip(fraction, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def _rotated_action(seed: dict[str, Any]) -> np.ndarray:
    action = np.asarray(seed["action"], dtype=np.float64).reshape(6, 3)
    angle = np.deg2rad(
        float(seed["align_rotation_degrees"])
        + float(seed.get("rotation_offset_degrees", 0.0))
    )
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    result = action.copy()
    result[:, :2] = action[:, :2] @ rotation.T
    return np.clip(result.reshape(-1), -0.30, 0.30).astype(np.float32)


def _table_clearance(
    nodes: np.ndarray,
    *,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    surface_z: float,
    padding: float,
) -> float:
    minimum = np.asarray([x_bounds[0], y_bounds[0], -2.0])
    maximum = np.asarray([x_bounds[1], y_bounds[1], surface_z])
    distance = min(
        segment_aabb_distance(start, end, minimum, maximum)
        for frame in nodes
        for start, end in zip(frame[:-1], frame[1:])
    )
    return float(distance - padding)


def _rollout(
    scenario: Path,
    target_action: np.ndarray,
    *,
    ramp_steps: int,
    hold_steps: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=(0.0, 0.0, 0.5),
        absolute_action_limit=0.30,
    )
    state, _ = env.reset(seed=seed)
    states = [np.asarray(state, dtype=np.float32)]
    nodes = [
        np.asarray(
            env.sim._backend.softrobot_state.element_positions, dtype=np.float64
        ).copy()
    ]
    rod = env.sim._backend._softrobot
    velocities = [rod.velocity_collection.T.astype(np.float64, copy=True)]
    directors = [rod.director_collection.transpose(2, 0, 1).astype(np.float64, copy=True)]
    omegas = [rod.omega_collection.T.astype(np.float64, copy=True)]
    internal_states = [pack_rod_internal_state(rod)]
    actions = []
    for step in range(ramp_steps + hold_steps):
        scale = _minimum_jerk((step + 1) / ramp_steps)
        action = np.asarray(scale * target_action, dtype=np.float32)
        env.muscle.set_activation(action.reshape(6, 3))
        env.sim.step_with_torque_callback(
            lambda lengths: env.muscle.evaluate(lengths)
        )
        actions.append(action)
        states.append(np.asarray(env._physical_state(), dtype=np.float32))
        nodes.append(
            np.asarray(
                env.sim._backend.softrobot_state.element_positions,
                dtype=np.float64,
            ).copy()
        )
        velocities.append(rod.velocity_collection.T.astype(np.float64, copy=True))
        directors.append(
            rod.director_collection.transpose(2, 0, 1).astype(np.float64, copy=True)
        )
        omegas.append(rod.omega_collection.T.astype(np.float64, copy=True))
        internal_states.append(pack_rod_internal_state(rod))
    env.close()
    return (
        np.stack(states),
        np.stack(actions),
        np.stack(nodes),
        np.stack(velocities),
        np.stack(directors),
        np.stack(omegas),
        np.stack(internal_states),
    )


def main() -> None:
    args = parse_args()
    scenario = Path(args.scenario).expanduser().resolve()
    seeds_path = Path(args.seeds).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not scenario.is_file() or not seeds_path.is_file():
        raise FileNotFoundError("scenario and seed configuration must exist")
    payload = yaml.safe_load(seeds_path.read_text(encoding="utf-8"))
    table = payload["table"]
    workspace = payload["workspace"]
    motion = payload["motion"]
    x_bounds = np.asarray(table["x_bounds"], dtype=np.float64)
    y_bounds = np.asarray(table["y_bounds"], dtype=np.float64)
    workspace_low = np.asarray(workspace["low"], dtype=np.float64)
    workspace_high = np.asarray(workspace["high"], dtype=np.float64)
    surface_z = float(table["surface_z"])
    arm_radius = float(table["arm_radius"])
    safety_margin = float(table["safety_margin"])
    ramp_steps = int(motion["ramp_steps"])
    hold_steps = int(motion["hold_steps"])
    max_delta = float(motion["maximum_action_delta"])
    max_hold_span = float(motion["maximum_hold_tip_span"])
    max_replay_error = float(motion["maximum_replay_tip_error"])
    max_reach = float(workspace["max_reach"])

    names: list[str] = []
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    node_rows: list[np.ndarray] = []
    velocity_rows: list[np.ndarray] = []
    director_rows: list[np.ndarray] = []
    omega_rows: list[np.ndarray] = []
    internal_state_rows: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for index, seed_config in enumerate(payload["seeds"]):
        name = str(seed_config["name"])
        target_action = _rotated_action(seed_config)
        states, actions, nodes, velocities, directors, omegas, internal_states = _rollout(
            scenario,
            target_action,
            ramp_steps=ramp_steps,
            hold_steps=hold_steps,
            seed=args.seed + index,
        )
        (
            replay_states,
            replay_actions,
            replay_nodes,
            replay_velocities,
            replay_directors,
            replay_omegas,
            replay_internal_states,
        ) = _rollout(
            scenario,
            target_action,
            ramp_steps=ramp_steps,
            hold_steps=hold_steps,
            seed=args.seed + 10_000 + index,
        )
        tip = states[-1, 30:33]
        action_delta = float(np.max(np.abs(np.diff(actions, axis=0))))
        hold_span = np.ptp(states[-hold_steps:, 30:33], axis=0)
        replay_error = max(
            float(np.max(np.abs(states - replay_states))),
            float(np.max(np.abs(velocities - replay_velocities))),
            float(np.max(np.abs(directors - replay_directors))),
            float(np.max(np.abs(omegas - replay_omegas))),
            float(np.max(np.abs(internal_states - replay_internal_states))),
        )
        replay_node_error = float(np.max(np.abs(nodes - replay_nodes)))
        clearance = _table_clearance(
            nodes,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            surface_z=surface_z,
            padding=arm_radius + safety_margin,
        )
        endpoint_ok = bool(
            np.all(tip >= workspace_low)
            and np.all(tip <= workspace_high)
            and np.linalg.norm(tip) <= max_reach
        )
        checks = {
            "endpoint_in_workspace": endpoint_ok,
            "whole_arm_table_clearance": clearance >= 0.0,
            "stable_hold": float(np.max(hold_span)) <= max_hold_span,
            "action_delta": action_delta <= max_delta + 1e-8,
            "deterministic_replay": max(replay_error, replay_node_error)
            <= max_replay_error,
        }
        report = {
            "name": name,
            "source": str(seed_config.get("source", "unspecified")),
            "endpoint": tip.tolist(),
            "path_length": float(
                np.sum(np.linalg.norm(np.diff(states[:, 30:33], axis=0), axis=1))
            ),
            "minimum_table_clearance": clearance,
            "maximum_action_delta": action_delta,
            "maximum_hold_tip_span": float(np.max(hold_span)),
            "maximum_replay_state_error": replay_error,
            "maximum_replay_node_error": replay_node_error,
            "checks": checks,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        if not all(checks.values()):
            failed = [key for key, passed in checks.items() if not passed]
            raise RuntimeError(f"trajectory {name!r} failed certification: {failed}")
        names.append(name)
        state_rows.append(states)
        action_rows.append(actions)
        node_rows.append(nodes)
        velocity_rows.append(velocities)
        director_rows.append(directors)
        omega_rows.append(omegas)
        internal_state_rows.append(internal_states)
        reports.append(report)

    scenario_config = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    control_dt = float(scenario_config["backend"]["dt"]) * int(
        scenario_config["environment"]["update_interval"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(3, dtype=np.int64),
            kind=np.asarray("manisoft_table_entry_trajectory_bank"),
            names=np.asarray(names),
            physical_states=np.stack(state_rows),
            actions=np.stack(action_rows),
            node_positions=np.stack(node_rows),
            node_velocities=np.stack(velocity_rows),
            element_directors=np.stack(director_rows),
            element_omegas=np.stack(omega_rows),
            rod_internal_states=np.stack(internal_state_rows),
            control_dt=np.asarray(control_dt, dtype=np.float64),
            scenario_sha256=np.asarray(_sha256(scenario)),
            table_x_bounds=x_bounds,
            table_y_bounds=y_bounds,
            table_surface_z=np.asarray(surface_z, dtype=np.float64),
            arm_radius=np.asarray(arm_radius, dtype=np.float64),
            safety_margin=np.asarray(safety_margin, dtype=np.float64),
        )
    temporary.replace(output)
    manifest = {
        "schema_version": 3,
        "kind": "manisoft_table_entry_trajectory_bank_manifest",
        "bank": str(output),
        "bank_sha256": _sha256(output),
        "scenario": str(scenario),
        "scenario_sha256": _sha256(scenario),
        "seed_config": str(seeds_path),
        "seed_config_sha256": _sha256(seeds_path),
        "trajectory_count": len(names),
        "transition_count_per_trajectory": ramp_steps + hold_steps,
        "control_dt": control_dt,
        "reports": reports,
    }
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
