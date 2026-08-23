#!/usr/bin/env python
"""Physical smoke test for short-segment table waypoint polylines."""

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
        default="configs/manisoft_waypoint_sac_table_waypoint_polyline.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=980000)
    parser.add_argument("--cartesian-action-leak", type=float, default=None)
    parser.add_argument("--waypoint-maximum-extent", type=float, default=None)
    parser.add_argument("--waypoint-maximum-turn-degrees", type=float, default=None)
    parser.add_argument(
        "--waypoint-segment-count-range",
        default=None,
        help="Inclusive segment-count pair, for example 4,4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = Path(args.scenario).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = dict(payload["environment"])
    if args.cartesian_action_leak is not None:
        config["cartesian_action_leak"] = float(args.cartesian_action_leak)
    if args.waypoint_maximum_extent is not None:
        config["waypoint_maximum_extent"] = float(args.waypoint_maximum_extent)
    if args.waypoint_maximum_turn_degrees is not None:
        if not 0.0 < args.waypoint_maximum_turn_degrees <= 180.0:
            raise ValueError("waypoint-maximum-turn-degrees must lie in (0, 180]")
        config["waypoint_maximum_turn_degrees"] = float(
            args.waypoint_maximum_turn_degrees
        )
    if args.waypoint_segment_count_range is not None:
        counts = [
            int(value)
            for value in args.waypoint_segment_count_range.split(",")
            if value.strip()
        ]
        if len(counts) != 2 or counts[0] < 1 or counts[0] > counts[1]:
            raise ValueError(
                "waypoint-segment-count-range must be an increasing positive pair"
            )
        config["waypoint_segment_count_range"] = counts
        config.pop("waypoint_segment_count_probabilities", None)
    for key in ("entry_bank_path", "table_action_calibration_path"):
        path = Path(config[key]).expanduser()
        if not path.is_absolute():
            path = (config_path.parent.parent / path).resolve()
        config[key] = str(path)
    bank = load_table_entry_trajectory_bank(Path(config["entry_bank_path"]))

    rows = []
    for entry_index, entry_name in enumerate(bank.names):
        env = ManiSoftWaypointSACEnv(scenario, **config)
        _, reset_info = env.reset(
            seed=args.seed + entry_index,
            options={
                "entry_index": entry_index,
                "desired_speed": 0.025,
                # Rehearsal configurations may randomly replace a waypoint
                # route with a one-segment line.  A waypoint smoke test must
                # explicitly exercise the harder multi-segment branch.
                "path_family": "waypoint_polyline",
            },
        )
        anchors = np.asarray(reset_info["path_anchors"], dtype=np.float64)
        start_tip = np.asarray(reset_info["tip_position"], dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
        minimum_clearance = float(reset_info["whole_arm_table_clearance"])
        maximum_action_delta = 0.0
        maximum_hold_drift = 0.0
        table_violation = False
        dynamics_violation = False
        for step in range(args.probe_steps):
            if step < 10:
                policy_action = np.zeros(2, dtype=np.float32)
            else:
                phase = 2.0 * np.pi * (step - 10) / max(args.probe_steps - 10, 1)
                policy_action = np.asarray(
                    [0.6 * np.sin(phase), 0.6 * np.cos(0.8 * phase)],
                    dtype=np.float32,
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
                break
        env.close()
        row = {
            "entry_index": entry_index,
            "entry_name": str(entry_name),
            "waypoint_count": int(len(anchors) - 1),
            "path_length": float(reset_info["path_length"]),
            "minimum_segment_length": float(np.min(segment_lengths)),
            "maximum_segment_length": float(np.max(segment_lengths)),
            "maximum_extent": float(
                np.max(np.linalg.norm(anchors - anchors[0], axis=1))
            ),
            "maximum_vertical_delta": float(
                np.max(np.abs(anchors[:, 2] - anchors[0, 2]))
            ),
            "maximum_hold_tip_drift": maximum_hold_drift,
            "maximum_applied_action_delta": maximum_action_delta,
            "minimum_table_clearance": minimum_clearance,
            "table_violation": table_violation,
            "dynamics_violation": dynamics_violation,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    passed = bool(
        all(
            int(config["waypoint_segment_count_range"][0])
            <= row["waypoint_count"]
            <= int(config["waypoint_segment_count_range"][1])
            and row["minimum_segment_length"]
            >= float(config["waypoint_segment_length_range"][0]) - 1e-4
            and row["maximum_segment_length"]
            <= float(config["waypoint_segment_length_range"][1]) + 1e-4
            and row["maximum_extent"]
            <= float(config["waypoint_maximum_extent"]) + 1e-4
            and row["maximum_vertical_delta"] <= 1e-6
            and row["maximum_hold_tip_drift"] <= 5e-4
            and row["maximum_applied_action_delta"]
            <= config["max_action_delta"] + 1e-7
            and row["minimum_table_clearance"] > 0.0
            and not row["table_violation"]
            and not row["dynamics_violation"]
            for row in rows
        )
    )
    report = {
        "kind": "manisoft_table_waypoint_polyline_physical_smoke",
        "passed": passed,
        "scenario": str(scenario),
        "config": str(config_path),
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
