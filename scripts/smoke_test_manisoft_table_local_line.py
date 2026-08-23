#!/usr/bin/env python
"""Physical smoke test for the low-dimensional table-local line skill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.envs.manisoft_waypoint_sac_env import ManiSoftWaypointSACEnv
from antmaze_ac.envs.table_entry_bank import load_table_entry_trajectory_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--config",
        default="configs/manisoft_waypoint_sac_table_local_line.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=970000)
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
    rows = []
    for entry_index, entry_name in enumerate(bank.names):
        env = ManiSoftWaypointSACEnv(scenario, **config)
        observation, reset_info = env.reset(
            seed=args.seed + entry_index,
            options={"entry_index": entry_index, "desired_speed": 0.025},
        )
        start_tip = np.asarray(reset_info["tip_position"], dtype=np.float64)
        start_action = np.asarray(reset_info["applied_action"], dtype=np.float64)
        path_anchors = np.asarray(reset_info["path_anchors"], dtype=np.float64)
        minimum_clearance = float(reset_info["whole_arm_table_clearance"])
        maximum_action_delta = 0.0
        maximum_hold_drift = 0.0
        table_violation = False
        dynamics_violation = False
        resets = 0
        for step in range(args.probe_steps):
            if step < 10:
                policy_action = np.zeros(2, dtype=np.float32)
            else:
                phase = 2.0 * np.pi * (step - 10) / max(args.probe_steps - 10, 1)
                policy_action = np.asarray(
                    [np.sin(phase), np.cos(0.7 * phase)], dtype=np.float32
                )
            _, _, terminated, truncated, info = env.step(policy_action)
            maximum_action_delta = max(
                maximum_action_delta,
                float(np.max(np.abs(info["applied_delta_action"]))),
            )
            minimum_clearance = min(
                minimum_clearance, float(info["whole_arm_table_clearance"])
            )
            if step < 10:
                maximum_hold_drift = max(
                    maximum_hold_drift,
                    float(np.linalg.norm(np.asarray(info["tip_position"]) - start_tip)),
                )
            table_violation |= bool(info["table_violation"])
            dynamics_violation |= bool(info["dynamics_violation"])
            if terminated or truncated:
                if table_violation or dynamics_violation:
                    break
                resets += 1
                _, info = env.reset(
                    seed=args.seed + 1000 + entry_index * 100 + resets,
                    options={"entry_index": entry_index, "desired_speed": 0.025},
                )
        env.close()
        row = {
            "entry_index": entry_index,
            "entry_name": str(entry_name),
            "observation_shape": list(observation.shape),
            "policy_action_shape": [2],
            "physical_action_shape": list(start_action.shape),
            "path_length": float(np.linalg.norm(path_anchors[-1] - path_anchors[0])),
            "path_vertical_delta": float(path_anchors[-1, 2] - path_anchors[0, 2]),
            "maximum_hold_tip_drift": maximum_hold_drift,
            "maximum_applied_action_delta": maximum_action_delta,
            "minimum_table_clearance": minimum_clearance,
            "table_violation": table_violation,
            "dynamics_violation": dynamics_violation,
            "safe_resets": resets,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    passed = bool(
        all(
            row["policy_action_shape"] == [2]
            and row["physical_action_shape"] == [18]
            and 0.0179 <= row["path_length"] <= 0.0301
            and abs(row["path_vertical_delta"]) <= 0.0061
            and row["maximum_hold_tip_drift"] <= 5e-4
            and row["maximum_applied_action_delta"] <= config["max_action_delta"] + 1e-7
            and row["minimum_table_clearance"] > 0.0
            and not row["table_violation"]
            and not row["dynamics_violation"]
            for row in rows
        )
    )
    report = {
        "kind": "manisoft_table_local_line_physical_smoke",
        "passed": passed,
        "scenario": str(scenario),
        "config": str(config_path),
        "bank": str(bank_path),
        "probe_steps_per_entry": args.probe_steps,
        "entries": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": passed, "entries": len(rows)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
