#!/usr/bin/env python
"""Replay safe phase-2 candidates into certified wall-crossing reset states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.data.wall_route_episodes import WallRouteGeometry, WallRoutePhase
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.table_entry_bank import pack_rod_internal_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-fraction", type=float, default=0.05)
    parser.add_argument("--maximum-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-wall-clearance", type=float, default=0.002)
    parser.add_argument("--maximum-tip-speed", type=float, default=0.60)
    parser.add_argument("--per-bin-per-side", type=int, default=8)
    parser.add_argument(
        "--mirror-augment",
        action="store_true",
        help=(
            "Replay a reflected counterpart for every selected trajectory. "
            "The x reflection flips local torque axes 1 and 2."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distal_crossed_fraction(
    nodes: np.ndarray, geometry: WallRouteGeometry
) -> float:
    checked = np.asarray(nodes, dtype=np.float64)[geometry.mounting_exempt_nodes :]
    beyond = checked[:, 1] >= geometry.wall_maximum[1] + geometry.postwall_y_margin
    suffix_count = 0
    for value in beyond[::-1]:
        if not value:
            break
        suffix_count += 1
    return float(suffix_count / len(checked))


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


def _mirror_actions(actions: np.ndarray) -> np.ndarray:
    """Reflect activation about x=0 for the configured upright rod frame."""

    mirrored = np.asarray(actions, dtype=np.float32).reshape(-1, 6, 3).copy()
    # Torque is an axial vector.  Under x -> -x, its local y and z
    # components change sign while x is unchanged.
    mirrored[:, :, 1:] *= -1.0
    return mirrored.reshape(-1, 18)


def _select_frames(
    candidate_dir: Path,
    manifest: dict,
    geometry: WallRouteGeometry,
    *,
    minimum_fraction: float,
    maximum_fraction: float,
    minimum_wall_clearance: float,
    maximum_tip_speed: float,
    per_bin_per_side: int,
) -> list[dict]:
    proposed: list[dict] = []
    for row in manifest["episodes"]:
        if int(row["maximum_phase"]) < int(WallRoutePhase.TIP_BEYOND_WALL):
            continue
        path = candidate_dir / row["path"]
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"candidate hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            phases = np.asarray(archive["phase_ids"], dtype=np.int8)
            nodes = np.asarray(archive["node_positions"], dtype=np.float64)
            velocities = np.asarray(archive["node_velocities"], dtype=np.float64)
            clearances = np.asarray(archive["wall_clearances"], dtype=np.float64)
            grounds = np.asarray(archive["ground_clearances"], dtype=np.float64)
            side = int(np.asarray(archive["route_side"]).reshape(()).item())
        seen_bins: set[int] = set()
        for frame in range(1, len(phases)):
            if phases[frame] < int(WallRoutePhase.TIP_BEYOND_WALL):
                continue
            fraction = _distal_crossed_fraction(nodes[frame], geometry)
            # There are 20 checked nodes, so the exact fraction is also a
            # natural discrete curriculum bin (0.05, 0.10, ...).
            bin_index = int(round(fraction * len(nodes[frame][geometry.mounting_exempt_nodes :])))
            if bin_index in seen_bins:
                continue
            if not minimum_fraction - 1e-8 <= fraction <= maximum_fraction + 1e-8:
                continue
            tip_speed = float(np.linalg.norm(velocities[frame, -1]))
            if (
                clearances[frame] < minimum_wall_clearance
                or grounds[frame] < -geometry.ground_violation_tolerance
                or tip_speed > maximum_tip_speed
            ):
                continue
            seen_bins.add(bin_index)
            proposed.append(
                {
                    "episode": int(row["index"]),
                    "path": path,
                    "frame": frame,
                    "side": side,
                    "fraction": fraction,
                    "bin": bin_index,
                    "wall_clearance": float(clearances[frame]),
                    "tip_speed": tip_speed,
                }
            )

    # Retain diverse episodes while preferring a larger safety buffer and
    # slower moving snapshots in every side/progress curriculum cell.
    selected: list[dict] = []
    cells = sorted({(row["side"], row["bin"]) for row in proposed})
    for cell in cells:
        rows = [row for row in proposed if (row["side"], row["bin"]) == cell]
        rows.sort(key=lambda row: (-row["wall_clearance"], row["tip_speed"]))
        selected.extend(rows[:per_bin_per_side])
    selected.sort(key=lambda row: (row["episode"], row["frame"]))
    if not selected:
        raise RuntimeError("no safe phase-2 frames satisfy the snapshot filters")
    if set(row["side"] for row in selected) != {-1, 1}:
        raise RuntimeError("snapshot filters did not retain both wall-route sides")
    return selected


def main() -> None:
    args = parse_args()
    scenario = Path(args.scenario).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    candidate_dir = Path(args.candidates).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = candidate_dir / "manifest.json"
    for path in (scenario, config_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    if not (
        0 < args.minimum_fraction <= args.maximum_fraction < 1
        and args.minimum_wall_clearance >= 0
        and args.maximum_tip_speed > 0
        and args.per_bin_per_side > 0
    ):
        raise ValueError("invalid snapshot selection settings")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    geometry = WallRouteGeometry.from_dict(payload["task"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "manisoft_virtual_wall_route_candidate_collection":
        raise ValueError("candidate directory has an unsupported manifest")
    if manifest["scenario_sha256"] != _sha256(scenario):
        raise ValueError("candidate manifest scenario differs from --scenario")
    if manifest["config_sha256"] != _sha256(config_path):
        raise ValueError("candidate manifest config differs from --config")

    selected = _select_frames(
        candidate_dir,
        manifest,
        geometry,
        minimum_fraction=args.minimum_fraction,
        maximum_fraction=args.maximum_fraction,
        minimum_wall_clearance=args.minimum_wall_clearance,
        maximum_tip_speed=args.maximum_tip_speed,
        per_bin_per_side=args.per_bin_per_side,
    )
    grouped: dict[Path, list[dict]] = {}
    for row in selected:
        grouped.setdefault(row["path"], []).append(row)

    captured_rows: list[dict] = []
    replay_count = len(grouped) * (2 if args.mirror_augment else 1)
    replay_index = 0
    for source, requested in grouped.items():
        with np.load(source, allow_pickle=False) as archive:
            actions = np.asarray(archive["actions"], dtype=np.float32)
            expected_states = np.asarray(archive["physical_states"], dtype=np.float32)
            expected_nodes = np.asarray(archive["node_positions"], dtype=np.float64)
            episode_seed = int(np.asarray(archive["episode_seed"]).reshape(()).item())
        by_frame = {int(row["frame"]): row for row in requested}
        variants = [(False, actions)]
        if args.mirror_augment:
            variants.append((True, _mirror_actions(actions)))
        for mirrored, replay_actions in variants:
            replay_index += 1
            env = ManiSoftTipTrackingEnv(
                scenario,
                target_tip=geometry.target,
                episode_steps=max(by_frame),
                absolute_action_limit=float(manifest["action_limit"]),
                muscle_torque_scale=float(
                    manifest.get("muscle_torque_scale", 30.0)
                ),
            )
            initial, _ = env.reset(seed=episode_seed)
            if not mirrored and not np.allclose(
                initial, expected_states[0], atol=2e-6, rtol=1e-6
            ):
                raise RuntimeError(
                    f"candidate initial state is not reproducible: {source}"
                )
            for action_index, action in enumerate(
                replay_actions[: max(by_frame)]
            ):
                env.muscle.set_activation(action.reshape(6, 3))
                env.sim.step_with_torque_callback(
                    lambda lengths: env.muscle.evaluate(lengths)
                )
                frame = action_index + 1
                if frame not in by_frame:
                    continue
                capture = _capture(env)
                expected_frame_nodes = expected_nodes[frame].copy()
                if mirrored:
                    expected_frame_nodes[:, 0] *= -1.0
                    state_error = 0.0
                else:
                    state_error = float(
                        np.max(
                            np.abs(
                                capture["physical_state"]
                                - expected_states[frame]
                            )
                        )
                    )
                node_error = float(
                    np.max(
                        np.abs(
                            capture["node_positions"] - expected_frame_nodes
                        )
                    )
                )
                if state_error > 5e-5 or node_error > 5e-8:
                    raise RuntimeError(
                        f"candidate replay diverged at {source.name}:{frame}: "
                        f"state={state_error:.3e}, nodes={node_error:.3e}"
                    )
                row = dict(by_frame[frame])
                row.update(capture)
                row["side"] = -row["side"] if mirrored else row["side"]
                row["previous_action"] = replay_actions[frame - 1].copy()
                row["mirrored"] = mirrored
                captured_rows.append(row)
            env.close()
            print(
                json.dumps(
                    {
                        "replayed": replay_index,
                        "replay_count": replay_count,
                        "source": source.name,
                        "mirrored": mirrored,
                        "captured": len(requested),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    captured_rows.sort(key=lambda row: (row["side"], row["fraction"], row["episode"]))
    names = np.asarray(
        [
            f"episode_{row['episode']:05d}_frame_{row['frame']:04d}"
            f"{'_mirrored' if row['mirrored'] else ''}"
            for row in captured_rows
        ]
    )
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "kind": np.asarray("manisoft_wall_crossing_snapshot_bank"),
        "names": names,
        "source_episodes": np.asarray([row["episode"] for row in captured_rows]),
        "source_frames": np.asarray([row["frame"] for row in captured_rows]),
        "route_sides": np.asarray([row["side"] for row in captured_rows], dtype=np.int8),
        "mirrored": np.asarray([row["mirrored"] for row in captured_rows], dtype=np.bool_),
        "crossed_fractions": np.asarray(
            [row["fraction"] for row in captured_rows], dtype=np.float32
        ),
        "physical_states": np.stack([row["physical_state"] for row in captured_rows]),
        "previous_actions": np.stack([row["previous_action"] for row in captured_rows]),
        "node_positions": np.stack([row["node_positions"] for row in captured_rows]),
        "node_velocities": np.stack([row["node_velocities"] for row in captured_rows]),
        "element_directors": np.stack([row["element_directors"] for row in captured_rows]),
        "element_omegas": np.stack([row["element_omegas"] for row in captured_rows]),
        "rod_internal_states": np.stack([row["rod_internal_state"] for row in captured_rows]),
        "control_dt": np.asarray(manifest["control_dt"], dtype=np.float64),
        "scenario_sha256": np.asarray(manifest["scenario_sha256"]),
        "collection_config_sha256": np.asarray(manifest["config_sha256"]),
        "absolute_action_limit": np.asarray(manifest["action_limit"], dtype=np.float64),
        "muscle_torque_scale": np.asarray(
            manifest.get("muscle_torque_scale", 30.0), dtype=np.float64
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    summary = {
        "output": str(output),
        "snapshot_count": len(captured_rows),
        "left_count": sum(row["side"] < 0 for row in captured_rows),
        "right_count": sum(row["side"] > 0 for row in captured_rows),
        "mirrored_count": sum(row["mirrored"] for row in captured_rows),
        "fraction_counts": {
            f"{value:.2f}": int(
                sum(np.isclose(row["fraction"], value) for row in captured_rows)
            )
            for value in sorted(
                {float(row["fraction"]) for row in captured_rows}
            )
        },
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
