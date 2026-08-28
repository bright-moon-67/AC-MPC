#!/usr/bin/env python
"""Capture exact states when a frozen crossing policy first reaches a threshold."""

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
    parser.add_argument(
        "--rollout-model",
        default=None,
        help="Optional SAC checkpoint used instead of the environment base policy.",
    )
    parser.add_argument(
        "--rollout-vecnormalize",
        default=None,
        help="VecNormalize file paired with --rollout-model.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--crossed-fraction", type=float, default=0.40)
    parser.add_argument(
        "--threshold-streak",
        type=int,
        default=1,
        help="Consecutive safe threshold steps required before capture.",
    )
    parser.add_argument("--maximum-steps", type=int, default=150)
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
    rollout_model_path = (
        None
        if args.rollout_model is None
        else Path(args.rollout_model).expanduser().resolve()
    )
    rollout_normalizer_path = (
        None
        if args.rollout_vecnormalize is None
        else Path(args.rollout_vecnormalize).expanduser().resolve()
    )
    if (rollout_model_path is None) != (rollout_normalizer_path is None):
        raise ValueError(
            "rollout-model and rollout-vecnormalize must be supplied together"
        )
    for path in (scenario, task_config, source_bank, sac_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (rollout_model_path, rollout_normalizer_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    if (
        not 0 < args.crossed_fraction < 1
        or args.maximum_steps < 1
        or args.threshold_streak < 1
    ):
        raise ValueError("invalid crossing capture limits")

    environment_config = dict(
        yaml.safe_load(sac_config.read_text(encoding="utf-8"))["environment"]
    )
    environment_config["episode_steps"] = max(
        int(environment_config["episode_steps"]), args.maximum_steps
    )
    env = ManiSoftWallCrossingSACEnv(
        scenario,
        task_config_path=task_config,
        snapshot_bank_path=source_bank,
        **environment_config,
    )
    bank = env.snapshot_bank
    rollout_model = None
    rollout_normalizer = None
    if rollout_model_path is not None:
        from stable_baselines3 import SAC
        from stable_baselines3.common.save_util import load_from_pkl

        rollout_model = SAC.load(str(rollout_model_path), device="cpu")
        rollout_normalizer = load_from_pkl(rollout_normalizer_path)

    def rollout_action(observation: np.ndarray) -> np.ndarray:
        if rollout_model is None or rollout_normalizer is None:
            return np.zeros(18, dtype=np.float32)
        value = np.asarray(observation, dtype=np.float32)
        if bool(rollout_normalizer.norm_obs):
            value = np.clip(
                (value - rollout_normalizer.obs_rms.mean)
                / np.sqrt(
                    rollout_normalizer.obs_rms.var
                    + rollout_normalizer.epsilon
                ),
                -rollout_normalizer.clip_obs,
                rollout_normalizer.clip_obs,
            ).astype(np.float32)
        action, _ = rollout_model.predict(value, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(18)

    rows: list[dict[str, object]] = []
    for ordinal, index in enumerate(env.eligible_snapshot_indices):
        observation, reset_info = env.reset(
            seed=930000 + ordinal, options={"snapshot_index": int(index)}
        )
        selected = None
        stop_reason = "step_limit"
        threshold_count = 0
        for step in range(1, args.maximum_steps + 1):
            observation, _, terminated, truncated, info = env.step(
                rollout_action(observation)
            )
            safe_at_threshold = (
                info["distal_crossed_fraction"]
                >= args.crossed_fraction - 1e-8
                and info["tip_beyond_distance"] >= 0
                and info["wall_clearance"] >= 0
                and info["ground_clearance"]
                >= -env.geometry.ground_violation_tolerance
            )
            threshold_count = threshold_count + 1 if safe_at_threshold else 0
            if threshold_count >= args.threshold_streak:
                selected = {
                    "source_index": int(index),
                    "step": step,
                    "metrics": dict(info),
                    "capture": _capture(env),
                    "previous_action": env.previous_action.copy(),
                }
                stop_reason = "crossing_threshold"
                break
            if terminated or truncated:
                stop_reason = info["termination_reason"] or "episode_limit"
                break
        report = {
            "source_index": int(index),
            "source_name": bank.names[index],
            "side": int(bank.route_sides[index]),
            "selected": selected is not None,
            "stop_reason": stop_reason,
        }
        if selected is not None:
            metrics = selected["metrics"]
            row = {
                **selected,
                "side": int(bank.route_sides[index]),
                "fraction": float(metrics["distal_crossed_fraction"]),
                "tip_speed": float(metrics["tip_speed"]),
                "tip_x": float(metrics["tip_x"]),
                "wall_clearance": float(metrics["wall_clearance"]),
                "ground_clearance": float(metrics["ground_clearance"]),
            }
            rows.append(row)
            report.update(
                {
                    "step": row["step"],
                    "fraction": row["fraction"],
                    "tip_speed": row["tip_speed"],
                    "tip_x": row["tip_x"],
                    "ground_clearance": row["ground_clearance"],
                }
            )
        print(json.dumps(report, sort_keys=True), flush=True)
    env.close()

    if not rows:
        raise RuntimeError("the frozen base policy did not reach the threshold")
    rows.sort(key=lambda row: (int(row["side"]), int(row["source_index"])))
    policy_label = "rollout" if rollout_model_path is not None else "base"
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "kind": np.asarray("manisoft_wall_crossing_snapshot_bank"),
        "names": np.asarray(
            [
                f"{bank.names[int(row['source_index'])]}_{policy_label}_crossed_"
                f"{int(round(100 * float(row['fraction']))):02d}pct_"
                f"{int(row['step']):03d}"
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
        "collection_config_sha256": np.asarray(bank.collection_config_sha256),
        "absolute_action_limit": np.asarray(
            bank.absolute_action_limit, dtype=np.float64
        ),
        "muscle_torque_scale": np.asarray(
            bank.muscle_torque_scale, dtype=np.float64
        ),
        "source_snapshot_bank_sha256": np.asarray(_sha256(source_bank)),
        "rollout_policy_model_sha256": np.asarray(
            _sha256(rollout_model_path)
            if rollout_model_path is not None
            else _sha256(Path(environment_config["base_policy_model_path"]))
        ),
        "capture_crossed_fraction": np.asarray(
            args.crossed_fraction, dtype=np.float64
        ),
        "capture_threshold_streak": np.asarray(
            args.threshold_streak, dtype=np.int64
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
        "threshold_streak": args.threshold_streak,
        "snapshots": [
            {
                "name": arrays["names"][ordinal].item(),
                "source_index": int(row["source_index"]),
                "side": int(row["side"]),
                "step": int(row["step"]),
                "fraction": float(row["fraction"]),
                "tip_speed": float(row["tip_speed"]),
                "tip_x": float(row["tip_x"]),
                "wall_clearance": float(row["wall_clearance"]),
                "ground_clearance": float(row["ground_clearance"]),
            }
            for ordinal, row in enumerate(rows)
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
