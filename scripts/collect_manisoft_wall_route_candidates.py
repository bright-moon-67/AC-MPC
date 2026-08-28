#!/usr/bin/env python
"""Collect safe free-space episodes annotated by a virtual wall-route task.

The simulator contains only the bare soft arm.  The virtual wall and z=0
ground are collection-time filters: an unsafe transition is not written to the
episode.  Both left and right route phase trackers observe every rollout, and
the more advanced side is stored as the episode's candidate route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from antmaze_ac.data.wall_route_episodes import (
    WALL_ROUTE_PHASE_NAMES,
    WallRouteGeometry,
    WallRoutePhase,
    WallRoutePhaseTracker,
    generate_smooth_wall_route_actions,
    validate_wall_route_episode,
)
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.table_entry_bank import pack_rod_internal_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--config",
        default="configs/manisoft_wall_route_collection.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _capture(env: ManiSoftTipTrackingEnv, save_internal_states: bool) -> dict[str, np.ndarray]:
    rod = env.sim._backend._softrobot
    result = {
        "physical_state": np.asarray(env._physical_state(), dtype=np.float32),
        "node_positions": rod.position_collection.T.astype(np.float64, copy=True),
        "node_velocities": rod.velocity_collection.T.astype(np.float64, copy=True),
        "element_directors": rod.director_collection.transpose(2, 0, 1).astype(
            np.float64, copy=True
        ),
        "element_omegas": rod.omega_collection.T.astype(np.float64, copy=True),
    }
    if save_internal_states:
        result["rod_internal_state"] = pack_rod_internal_state(rod)
    return result


def _select_route_side(
    phase_ids_left: list[int],
    phase_ids_right: list[int],
    tip_positions: np.ndarray,
    geometry: WallRouteGeometry,
) -> int:
    left_phase = max(phase_ids_left)
    right_phase = max(phase_ids_right)
    if left_phase != right_phase:
        return -1 if left_phase > right_phase else 1
    left_progress = float(
        np.max(geometry.side_gate_x(-1) - tip_positions[:, 0])
    )
    right_progress = float(
        np.max(tip_positions[:, 0] - geometry.side_gate_x(1))
    )
    return -1 if left_progress > right_progress else 1


def _scenario_checks(
    scenario_payload: dict[str, Any], geometry: WallRouteGeometry
) -> tuple[float, float, int]:
    backend = scenario_payload["backend"]
    environment = scenario_payload["environment"]
    softrobot = scenario_payload["softrobot"]
    if scenario_payload.get("objects"):
        raise ValueError("wall-route collection scenario must have objects: []")
    base = np.asarray(softrobot["start"], dtype=np.float64)
    if not np.allclose(base, geometry.base, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"scenario softrobot.start {base.tolist()} does not match task base "
            f"{geometry.base.tolist()}"
        )
    scenario_radius = float(softrobot["radius"])
    if not np.isclose(scenario_radius, geometry.arm_radius, atol=1e-9, rtol=0.0):
        raise ValueError("scenario and task arm radii differ")
    arm_length = float(softrobot["length"])
    if geometry.wall_minimum[2] > geometry.ground_surface_z:
        raise ValueError("virtual wall must extend down to the virtual ground")
    required_wall_top = geometry.base[2] + arm_length + geometry.wall_padding
    if geometry.wall_maximum[2] < required_wall_top:
        raise ValueError(
            "virtual wall is low enough to permit an unintended over-the-top route"
        )
    control_dt = float(backend["dt"]) * int(environment["update_interval"])
    return control_dt, arm_length, int(softrobot["num_elements"])


def main() -> None:
    args = parse_args()
    scenario = Path(args.scenario).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not scenario.is_file() or not config_path.is_file():
        raise FileNotFoundError("scenario and collection config must exist")
    if output.exists() and (
        (output / "manifest.json").exists() or any(output.glob("episode_*.npz"))
    ):
        raise FileExistsError(
            f"output already contains a wall-route collection: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("wall-route collection config schema_version must be 1")
    geometry = WallRouteGeometry.from_dict(payload["task"])
    collection = dict(payload["collection"])
    scenario_payload = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    control_dt, arm_length, element_count = _scenario_checks(
        scenario_payload, geometry
    )

    episodes = int(
        collection["episodes"] if args.episodes is None else args.episodes
    )
    episode_steps = int(
        collection["episode_steps"]
        if args.episode_steps is None
        else args.episode_steps
    )
    seed = int(collection["seed"] if args.seed is None else args.seed)
    minimum_saved_steps = int(collection.get("minimum_saved_steps", 32))
    action_limit = float(collection.get("absolute_action_limit", 0.30))
    muscle_torque_scale = float(collection.get("muscle_torque_scale", 30.0))
    maximum_action_delta = float(collection["maximum_action_delta"])
    configured_maximum_tip_speed = collection.get("maximum_tip_speed", 1.0)
    maximum_tip_speed = (
        None
        if configured_maximum_tip_speed is None
        else float(configured_maximum_tip_speed)
    )
    maximum_tip_displacement = float(
        collection.get("maximum_tip_displacement", arm_length + 0.05)
    )
    save_internal_states = bool(collection.get("save_internal_states", True))
    stop_on_success = bool(collection.get("stop_on_success", True))
    if min(
        episodes,
        episode_steps,
        minimum_saved_steps,
        action_limit,
        muscle_torque_scale,
        maximum_action_delta,
        maximum_tip_displacement,
    ) <= 0:
        raise ValueError("collection counts and limits must be positive")
    if maximum_tip_speed is not None and maximum_tip_speed <= 0:
        raise ValueError("maximum_tip_speed must be positive or null")
    if minimum_saved_steps > episode_steps:
        raise ValueError("minimum_saved_steps cannot exceed episode_steps")

    rng = np.random.default_rng(seed)
    manifest_rows: list[dict[str, Any]] = []
    termination_counts: dict[str, int] = {}
    saved_count = 0
    success_count = 0
    phase_counts = np.zeros(len(WALL_ROUTE_PHASE_NAMES), dtype=np.int64)

    for episode_id in range(episodes):
        episode_seed = seed + episode_id
        env = ManiSoftTipTrackingEnv(
            scenario,
            target_tip=geometry.target,
            episode_steps=episode_steps,
            absolute_action_limit=action_limit,
            muscle_torque_scale=muscle_torque_scale,
        )
        initial_state, _ = env.reset(seed=episode_seed)
        initial = _capture(env, save_internal_states)
        if not np.allclose(initial["physical_state"], initial_state, atol=1e-7):
            raise RuntimeError("captured initial state differs from environment reset")
        initial_wall_clearance = geometry.whole_arm_wall_clearance(
            initial["node_positions"]
        )
        initial_ground_clearance = geometry.whole_arm_ground_clearance(
            initial["node_positions"]
        )
        if initial_wall_clearance < 0:
            raise RuntimeError("initial upright state intersects the virtual wall")
        if initial_ground_clearance < -geometry.ground_violation_tolerance:
            raise RuntimeError(
                "initial upright state violates the virtual ground by "
                f"{-initial_ground_clearance:.6f} m"
            )

        actions = generate_smooth_wall_route_actions(
            rng,
            episode_steps,
            control_dt,
            knot_seconds_range=collection["action_knot_seconds_range"],
            hold_seconds_range=collection["action_hold_seconds_range"],
            peak_range=collection["action_peak_range"],
            spatial_modes=int(collection.get("spatial_modes", 4)),
            absolute_action_limit=action_limit,
            maximum_action_delta=maximum_action_delta,
            zero_anchor_probability=float(
                collection.get("zero_anchor_probability", 0.10)
            ),
        )
        trackers = {
            -1: WallRoutePhaseTracker(geometry, -1),
            1: WallRoutePhaseTracker(geometry, 1),
        }
        rows: dict[str, list[np.ndarray]] = {
            key: [initial[key]]
            for key in (
                "physical_state",
                "node_positions",
                "node_velocities",
                "element_directors",
                "element_omegas",
            )
        }
        if save_internal_states:
            rows["rod_internal_state"] = [initial["rod_internal_state"]]
        saved_actions: list[np.ndarray] = []
        phase_ids = {-1: [0], 1: [0]}
        wall_clearances = [initial_wall_clearance]
        ground_clearances = [initial_ground_clearance]
        target_distances = [
            float(
                np.linalg.norm(initial["node_positions"][-1] - geometry.target)
            )
        ]
        termination_reason = "episode_limit"
        violating_clearance: float | None = None

        for action in actions:
            env.muscle.set_activation(action.reshape(6, 3))
            try:
                env.sim.step_with_torque_callback(
                    lambda lengths: env.muscle.evaluate(lengths)
                )
                current = _capture(env, save_internal_states)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                termination_reason = "dynamics_violation"
                break
            nodes = current["node_positions"]
            wall_clearance = geometry.whole_arm_wall_clearance(nodes)
            ground_clearance = geometry.whole_arm_ground_clearance(nodes)
            tip_velocity = current["node_velocities"][-1]
            tip_speed = float(np.linalg.norm(tip_velocity))
            tip_displacement = float(
                np.linalg.norm(nodes[-1] - initial["node_positions"][-1])
            )
            if wall_clearance < 0:
                termination_reason = "virtual_wall_collision"
                violating_clearance = wall_clearance
                break
            if ground_clearance < -geometry.ground_violation_tolerance:
                termination_reason = "ground_violation"
                violating_clearance = ground_clearance
                break
            if maximum_tip_speed is not None and tip_speed > maximum_tip_speed:
                termination_reason = "tip_speed"
                break
            if tip_displacement > maximum_tip_displacement:
                termination_reason = "tip_displacement"
                break

            saved_actions.append(np.asarray(action, dtype=np.float32).copy())
            for key in rows:
                rows[key].append(current[key])
            wall_clearances.append(wall_clearance)
            ground_clearances.append(ground_clearance)
            target_distances.append(
                float(np.linalg.norm(nodes[-1] - geometry.target))
            )
            updates = {
                side: tracker.update(nodes, tip_velocity)
                for side, tracker in trackers.items()
            }
            for side in (-1, 1):
                phase_ids[side].append(updates[side].phase)
            if stop_on_success and any(update.success for update in updates.values()):
                termination_reason = "target_reached"
                break

        env.close()
        transition_count = len(saved_actions)
        termination_counts[termination_reason] = (
            termination_counts.get(termination_reason, 0) + 1
        )
        if transition_count < minimum_saved_steps:
            report = {
                "attempt_index": episode_id,
                "episode_seed": episode_seed,
                "saved": False,
                "valid_transition_count": transition_count,
                "termination_reason": termination_reason,
                "violating_clearance": violating_clearance,
            }
            print(json.dumps(report, sort_keys=True), flush=True)
            continue

        tip_positions = np.asarray(rows["node_positions"], dtype=np.float64)[
            :, -1
        ]
        route_side = _select_route_side(
            phase_ids[-1], phase_ids[1], tip_positions, geometry
        )
        selected_phases = np.asarray(phase_ids[route_side], dtype=np.int8)
        maximum_phase = int(np.max(selected_phases))
        route_success = maximum_phase == int(WallRoutePhase.TARGET_REACHED)
        arrays = {
            "physical_states": np.stack(rows["physical_state"]),
            "actions": np.asarray(saved_actions, dtype=np.float32),
            "node_positions": np.stack(rows["node_positions"]),
            "node_velocities": np.stack(rows["node_velocities"]),
            "element_directors": np.stack(rows["element_directors"]),
            "element_omegas": np.stack(rows["element_omegas"]),
            "phase_ids": selected_phases,
            "phase_ids_left": np.asarray(phase_ids[-1], dtype=np.int8),
            "phase_ids_right": np.asarray(phase_ids[1], dtype=np.int8),
            "wall_clearances": np.asarray(wall_clearances, dtype=np.float32),
            "ground_clearances": np.asarray(ground_clearances, dtype=np.float32),
            "target_distances": np.asarray(target_distances, dtype=np.float32),
        }
        if save_internal_states:
            arrays["rod_internal_states"] = np.stack(rows["rod_internal_state"])
        validate_wall_route_episode(arrays)
        episode_path = output / f"episode_{saved_count:05d}.npz"
        _atomic_savez(
            episode_path,
            {
                "schema_version": np.asarray(1, dtype=np.int64),
                "kind": np.asarray("manisoft_virtual_wall_route_candidate"),
                "attempt_index": np.asarray(episode_id, dtype=np.int64),
                "episode_seed": np.asarray(episode_seed, dtype=np.int64),
                "route_side": np.asarray(route_side, dtype=np.int8),
                "route_success": np.asarray(route_success, dtype=np.bool_),
                "termination_reason": np.asarray(termination_reason),
                "control_dt": np.asarray(control_dt, dtype=np.float64),
                **arrays,
            },
        )
        action_rows = arrays["actions"]
        previous_actions = np.vstack(
            (np.zeros((1, 18), dtype=np.float32), action_rows[:-1])
        )
        report = {
            "index": saved_count,
            "attempt_index": episode_id,
            "path": episode_path.name,
            "sha256": _sha256(episode_path),
            "episode_seed": episode_seed,
            "route_side": route_side,
            "route_success": route_success,
            "maximum_phase": maximum_phase,
            "maximum_phase_name": WALL_ROUTE_PHASE_NAMES[maximum_phase],
            "transition_count": transition_count,
            "termination_reason": termination_reason,
            "minimum_wall_clearance": float(np.min(arrays["wall_clearances"])),
            "minimum_ground_clearance": float(
                np.min(arrays["ground_clearances"])
            ),
            "minimum_target_distance": float(np.min(arrays["target_distances"])),
            "final_target_distance": float(arrays["target_distances"][-1]),
            "maximum_action_magnitude": float(np.max(np.abs(action_rows))),
            "maximum_action_delta": float(
                np.max(np.abs(action_rows - previous_actions))
            ),
        }
        manifest_rows.append(report)
        phase_counts[maximum_phase] += 1
        saved_count += 1
        success_count += int(route_success)
        print(json.dumps(report, sort_keys=True), flush=True)

    manifest = {
        "schema_version": 1,
        "kind": "manisoft_virtual_wall_route_candidate_collection",
        "scenario": str(scenario),
        "scenario_sha256": _sha256(scenario),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "attempted_episodes": episodes,
        "saved_episodes": saved_count,
        "successful_routes": success_count,
        "control_dt": control_dt,
        "element_count": element_count,
        "action_limit": action_limit,
        "muscle_torque_scale": muscle_torque_scale,
        "maximum_action_delta": maximum_action_delta,
        "speed_termination_enabled": maximum_tip_speed is not None,
        "maximum_tip_speed": maximum_tip_speed,
        "phase_names": list(WALL_ROUTE_PHASE_NAMES),
        "maximum_phase_counts": phase_counts.tolist(),
        "termination_reason_counts": termination_counts,
        "task": payload["task"],
        "collection": {
            **collection,
            "episodes": episodes,
            "episode_steps": episode_steps,
            "seed": seed,
        },
        "episodes": manifest_rows,
    }
    manifest_path = output / "manifest.json"
    temporary_manifest = manifest_path.with_name("manifest.json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "saved_episodes": saved_count,
                "successful_routes": success_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
