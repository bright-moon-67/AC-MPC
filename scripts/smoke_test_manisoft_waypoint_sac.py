#!/usr/bin/env python
"""End-to-end physical smoke checks for the corrected waypoint SAC setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from antmaze_ac.envs.manisoft_waypoint_sac_env import ManiSoftWaypointSACEnv
from antmaze_ac.envs.table_entry_bank import load_table_entry_trajectory_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--config", default="configs/manisoft_waypoint_sac_physical.yaml"
    )
    parser.add_argument(
        "--output",
        default="runs/manisoft_waypoint_sac_physical_smoke/environment_report.json",
    )
    parser.add_argument("--seed", type=int, default=731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = Path(args.scenario).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = dict(payload["environment"])
    bank_path = Path(config["entry_bank_path"]).expanduser()
    if not bank_path.is_absolute():
        bank_path = (config_path.parent.parent / bank_path).resolve()
    config["entry_bank_path"] = str(bank_path)
    bank = load_table_entry_trajectory_bank(bank_path)
    entries = []
    for entry_index, name in enumerate(bank.names):
        entry_config = {**config, "curriculum": "entry"}
        env = ManiSoftWaypointSACEnv(scenario, **entry_config)
        observation, _ = env.reset(
            seed=args.seed + entry_index,
            options={
                "entry_index": entry_index,
                "desired_speed": 0.065,
            },
        )
        del observation
        episode_return = 0.0
        started = time.perf_counter()
        terminal_info = {}
        for step, action in enumerate(bank.actions[entry_index], start=1):
            _, reward, terminated, truncated, terminal_info = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        elapsed = time.perf_counter() - started
        env.close()
        row = {
            "entry_index": entry_index,
            "entry_name": name,
            "steps": step,
            "wall_seconds": elapsed,
            "return": episode_return,
            "success": bool(terminal_info.get("is_success", False)),
            "table_violation": bool(
                terminal_info.get("table_violation", False)
            ),
            "final_progress": float(terminal_info.get("path_progress", 0.0)),
            "final_distance": float(terminal_info.get("final_distance", np.inf)),
            "whole_arm_table_clearance": float(
                terminal_info.get("whole_arm_table_clearance", -np.inf)
            ),
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        entries.append(row)

    warm_config = {**config, "curriculum": "table_local"}
    warm_env = ManiSoftWaypointSACEnv(scenario, **warm_config)
    started = time.perf_counter()
    _, warm_info = warm_env.reset(
        seed=args.seed + 1000, options={"entry_index": 0}
    )
    reset_seconds = time.perf_counter() - started
    start_tip = np.asarray(warm_info["tip_position"])
    hold_action = np.asarray(warm_info["applied_action"])
    hold_info = warm_info
    for _ in range(5):
        _, _, terminated, truncated, hold_info = warm_env.step(hold_action)
        if terminated or truncated:
            break
    warm_env.close()
    snapshot = {
        "reset_wall_seconds": reset_seconds,
        "five_step_tip_drift": float(
            np.linalg.norm(np.asarray(hold_info["tip_position"]) - start_tip)
        ),
        "whole_arm_table_clearance": float(
            hold_info["whole_arm_table_clearance"]
        ),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    moving_snapshots = []
    for stage, fraction in (("entry_mid", 0.50), ("entry_tail", 0.80)):
        for entry_index, name in enumerate(bank.names):
            prefix_env = ManiSoftWaypointSACEnv(
                scenario, **{**config, "curriculum": stage}
            )
            _, prefix_info = prefix_env.reset(
                seed=args.seed + 2000 + entry_index,
                options={
                    "entry_index": entry_index,
                    "warm_start_fraction": fraction,
                },
            )
            prefix = int(prefix_info["entry_prefix_steps"])
            _, _, prefix_terminated, prefix_truncated, next_info = prefix_env.step(
                bank.actions[entry_index, min(prefix, bank.transition_count - 1)]
            )
            expected_tip = bank.tip_positions[
                entry_index, min(prefix + 1, bank.transition_count)
            ]
            row = {
                "curriculum": stage,
                "entry_name": name,
                "prefix_steps": prefix,
                "one_step_tip_error": float(
                    np.linalg.norm(np.asarray(next_info["tip_position"]) - expected_tip)
                ),
                "dynamics_violation": bool(next_info["dynamics_violation"]),
                "table_violation": bool(next_info["table_violation"]),
                "terminated": bool(prefix_terminated),
                "truncated": bool(prefix_truncated),
            }
            moving_snapshots.append(row)
            prefix_env.close()
    passed = bool(
        all(row["success"] and not row["table_violation"] for row in entries)
        and snapshot["five_step_tip_drift"] < 5e-4
        and snapshot["whole_arm_table_clearance"] > 0
        and not snapshot["terminated"]
        and not snapshot["truncated"]
        and all(
            not row["dynamics_violation"]
            and not row["table_violation"]
            and not row["terminated"]
            and not row["truncated"]
            and row["one_step_tip_error"] < 2e-5
            for row in moving_snapshots
        )
    )
    report = {
        "kind": "manisoft_waypoint_sac_physical_smoke",
        "passed": passed,
        "scenario": str(scenario),
        "config": str(config_path),
        "bank": str(bank_path),
        "entries": entries,
        "snapshot_warm_start": snapshot,
        "moving_snapshot_checks": moving_snapshots,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
