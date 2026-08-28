#!/usr/bin/env python
"""Brake moving wall-crossing snapshots into low-speed curriculum states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.envs.manisoft_wall_crossing_sac_env import (
    ManiSoftWallCrossingSACEnv,
)
from antmaze_ac.envs.table_entry_bank import pack_rod_internal_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--snapshot-bank", required=True)
    parser.add_argument("--sac-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--minimum-candidate-step", type=int, default=40)
    parser.add_argument("--minimum-fraction", type=float, default=0.30)
    parser.add_argument("--maximum-fraction", type=float, default=0.35)
    parser.add_argument("--maximum-tip-speed", type=float, default=0.50)
    parser.add_argument("--minimum-wall-clearance", type=float, default=0.002)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(env: ManiSoftWallCrossingSACEnv) -> dict[str, np.ndarray]:
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


def main() -> None:
    args = parse_args()
    scenario = Path(args.scenario).expanduser().resolve()
    task_config = Path(args.task_config).expanduser().resolve()
    source_bank = Path(args.snapshot_bank).expanduser().resolve()
    sac_config = Path(args.sac_config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (scenario, task_config, source_bank, sac_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    if not (
        args.steps >= args.minimum_candidate_step >= 1
        and 0 < args.minimum_fraction <= args.maximum_fraction < 1
        and args.maximum_tip_speed > 0
        and args.minimum_wall_clearance >= 0
    ):
        raise ValueError("invalid stabilization limits")

    environment_config = dict(
        yaml.safe_load(sac_config.read_text(encoding="utf-8"))["environment"]
    )
    environment_config["episode_steps"] = max(
        int(environment_config["episode_steps"]), args.steps
    )
    env = ManiSoftWallCrossingSACEnv(
        scenario,
        task_config_path=task_config,
        snapshot_bank_path=source_bank,
        **environment_config,
    )
    bank = env.snapshot_bank
    rows: list[dict[str, object]] = []
    for ordinal, index in enumerate(env.eligible_snapshot_indices):
        _, reset_info = env.reset(
            seed=910000 + ordinal, options={"snapshot_index": int(index)}
        )
        best: dict[str, object] | None = None
        stop_reason = "step_limit"
        for step in range(1, args.steps + 1):
            # Policy actions are normalized activation increments.  This
            # request reaches exactly zero without overshooting once the
            # remaining activation is smaller than max_action_delta.
            brake = np.clip(
                -env.previous_action / env.max_action_delta, -1.0, 1.0
            ).astype(np.float32)
            _, _, terminated, truncated, info = env.step(brake)
            if (
                step >= args.minimum_candidate_step
                and args.minimum_fraction - 1e-8
                <= info["distal_crossed_fraction"]
                <= args.maximum_fraction + 1e-8
                and info["tip_speed"] <= args.maximum_tip_speed
                and info["wall_clearance"] >= args.minimum_wall_clearance
                and info["ground_clearance"]
                >= -env.geometry.ground_violation_tolerance
            ):
                score = (
                    float(info["tip_speed"]),
                    -float(info["wall_clearance"]),
                )
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "step": step,
                        "metrics": dict(info),
                        "capture": _capture(env),
                        "previous_action": env.previous_action.copy(),
                    }
            if terminated or truncated:
                stop_reason = info["termination_reason"] or "episode_limit"
                break

        report = {
            "source_index": int(index),
            "source_name": bank.names[index],
            "side": int(bank.route_sides[index]),
            "source_fraction": float(bank.crossed_fractions[index]),
            "stop_reason": stop_reason,
            "selected": best is not None,
        }
        if best is not None:
            metrics = best["metrics"]
            row = {
                **report,
                "step": int(best["step"]),
                "fraction": float(metrics["distal_crossed_fraction"]),
                "tip_speed": float(metrics["tip_speed"]),
                "wall_clearance": float(metrics["wall_clearance"]),
                "ground_clearance": float(metrics["ground_clearance"]),
                "target_plane_distance": float(
                    metrics["target_plane_distance"]
                ),
                "capture": best["capture"],
                "previous_action": best["previous_action"],
            }
            rows.append(row)
            report.update(
                {
                    "step": row["step"],
                    "fraction": row["fraction"],
                    "tip_speed": row["tip_speed"],
                    "target_plane_distance": row["target_plane_distance"],
                }
            )
        print(json.dumps(report, sort_keys=True), flush=True)

    if not rows:
        raise RuntimeError("no low-speed stabilization state satisfied the filters")
    if {int(row["side"]) for row in rows} != {-1, 1}:
        raise RuntimeError("stabilized bank must retain both route sides")
    rows.sort(
        key=lambda row: (
            int(row["side"]),
            float(row["fraction"]),
            float(row["tip_speed"]),
        )
    )
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "kind": np.asarray("manisoft_wall_crossing_snapshot_bank"),
        "names": np.asarray(
            [
                f"{bank.names[int(row['source_index'])]}_braked_{int(row['step']):03d}"
                for row in rows
            ]
        ),
        "source_episodes": np.asarray(
            [bank.source_episodes[int(row["source_index"])] for row in rows]
        ),
        "source_frames": np.asarray(
            [
                bank.source_frames[int(row["source_index"])] + int(row["step"])
                for row in rows
            ]
        ),
        "route_sides": np.asarray([row["side"] for row in rows], dtype=np.int8),
        "crossed_fractions": np.asarray(
            [row["fraction"] for row in rows], dtype=np.float32
        ),
        "physical_states": np.stack(
            [row["capture"]["physical_state"] for row in rows]
        ),
        "previous_actions": np.stack([row["previous_action"] for row in rows]),
        "node_positions": np.stack(
            [row["capture"]["node_positions"] for row in rows]
        ),
        "node_velocities": np.stack(
            [row["capture"]["node_velocities"] for row in rows]
        ),
        "element_directors": np.stack(
            [row["capture"]["element_directors"] for row in rows]
        ),
        "element_omegas": np.stack(
            [row["capture"]["element_omegas"] for row in rows]
        ),
        "rod_internal_states": np.stack(
            [row["capture"]["rod_internal_state"] for row in rows]
        ),
        "control_dt": np.asarray(bank.control_dt, dtype=np.float64),
        "scenario_sha256": np.asarray(bank.scenario_sha256),
        "collection_config_sha256": np.asarray(
            bank.collection_config_sha256
        ),
        "absolute_action_limit": np.asarray(
            bank.absolute_action_limit, dtype=np.float64
        ),
        "muscle_torque_scale": np.asarray(
            bank.muscle_torque_scale, dtype=np.float64
        ),
        "source_snapshot_bank_sha256": np.asarray(_sha256(source_bank)),
        "stabilization_steps": np.asarray(args.steps, dtype=np.int64),
        "stabilization_maximum_tip_speed": np.asarray(
            args.maximum_tip_speed, dtype=np.float64
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
        "sha256": _sha256(output),
        "source_snapshot_bank": str(source_bank),
        "source_snapshot_bank_sha256": _sha256(source_bank),
        "snapshot_count": len(rows),
        "left_count": sum(int(row["side"]) < 0 for row in rows),
        "right_count": sum(int(row["side"]) > 0 for row in rows),
        "minimum_tip_speed": min(float(row["tip_speed"]) for row in rows),
        "maximum_tip_speed": max(float(row["tip_speed"]) for row in rows),
        "fraction_counts": {
            f"{fraction:.2f}": int(
                sum(
                    np.isclose(float(row["fraction"]), fraction)
                    for row in rows
                )
            )
            for fraction in sorted({float(row["fraction"]) for row in rows})
        },
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
