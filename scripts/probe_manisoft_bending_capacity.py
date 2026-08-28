#!/usr/bin/env python
"""Compare baseline and strong-bending ManiSoft mechanics from upright reset.

The probe deliberately has no tip-speed termination.  It records speed as a
diagnostic while retaining non-finite dynamics and z=0 ground safeguards.
Every output includes scenario/config hashes and the exact activation profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv


@dataclass(frozen=True)
class Case:
    name: str
    scenario: Path
    action_limit: float
    muscle_torque_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-scenario",
        default="/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml",
    )
    parser.add_argument(
        "--strong-scenario",
        default="configs/manisoft_strong_bend_e2mpa_r45mm.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ramp-steps", type=int, default=120)
    parser.add_argument("--hold-steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def bend_metrics(nodes: np.ndarray) -> dict[str, float]:
    segments = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    unit = segments / np.maximum(lengths[:, None], np.finfo(float).eps)
    adjacent = np.sum(unit[:-1] * unit[1:], axis=1)
    local_angles = np.arccos(np.clip(adjacent, -1.0, 1.0))
    chord = nodes[-1] - nodes[0]
    chord_length = float(np.linalg.norm(chord))
    overall_angle = float(
        np.arccos(
            np.clip(chord[2] / max(chord_length, np.finfo(float).eps), -1.0, 1.0)
        )
    )
    distal_start = max(0, len(local_angles) - 8)
    return {
        "tip_horizontal_reach_m": float(np.linalg.norm(chord[:2])),
        "tip_height_m": float(nodes[-1, 2]),
        "overall_bend_deg": float(np.degrees(overall_angle)),
        "total_turn_deg": float(np.degrees(np.sum(local_angles))),
        "distal_turn_deg": float(np.degrees(np.sum(local_angles[distal_start:]))),
    }


def activation_patterns() -> list[tuple[str, np.ndarray]]:
    profiles = {
        "uniform": np.ones(6, dtype=np.float64),
        "distal": np.asarray([0.0, 0.0, 0.10, 0.40, 0.75, 1.0]),
    }
    patterns: list[tuple[str, np.ndarray]] = []
    for profile_name, weights in profiles.items():
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                target = np.zeros((6, 3), dtype=np.float64)
                target[:, axis] = sign * weights
                patterns.append(
                    (f"{profile_name}_axis{axis}_{'pos' if sign > 0 else 'neg'}", target)
                )
    return patterns


def run_trial(
    case: Case,
    pattern_name: str,
    normalized_target: np.ndarray,
    ramp_steps: int,
    hold_steps: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    scenario_payload = yaml.safe_load(case.scenario.read_text(encoding="utf-8"))
    radius = float(scenario_payload["softrobot"]["radius"])
    env = ManiSoftTipTrackingEnv(
        case.scenario,
        target_offset=(0.0, 0.005, 0.0),
        episode_steps=ramp_steps + hold_steps,
        absolute_action_limit=case.action_limit,
        muscle_torque_scale=case.muscle_torque_scale,
    )
    env.reset(seed=seed)
    rod = env.sim._backend._softrobot
    initial_nodes = rod.position_collection.T.astype(np.float64, copy=True)
    nodes_history = [initial_nodes]
    action_history: list[np.ndarray] = []
    maximum_tip_speed = float(np.linalg.norm(rod.velocity_collection.T[-1]))
    minimum_ground_clearance = float(
        np.min(initial_nodes[1:, 2]) - radius
    )
    status = "completed"
    ground_violation_tolerance = 0.0005

    total_steps = ramp_steps + hold_steps
    for step in range(total_steps):
        fraction = min(1.0, (step + 1) / ramp_steps)
        action = normalized_target * (case.action_limit * fraction)
        env.muscle.set_activation(action)
        try:
            env.sim.step_with_torque_callback(
                lambda lengths: env.muscle.evaluate(lengths)
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            status = "dynamics_violation"
            break
        nodes = rod.position_collection.T.astype(np.float64, copy=True)
        velocities = rod.velocity_collection.T.astype(np.float64, copy=False)
        if not np.isfinite(nodes).all() or not np.isfinite(velocities).all():
            status = "dynamics_violation"
            break
        nodes_history.append(nodes)
        action_history.append(action.astype(np.float32, copy=True))
        maximum_tip_speed = max(
            maximum_tip_speed, float(np.linalg.norm(velocities[-1]))
        )
        ground_clearance = float(np.min(nodes[1:, 2]) - radius)
        minimum_ground_clearance = min(minimum_ground_clearance, ground_clearance)
        if ground_clearance < -ground_violation_tolerance:
            status = "ground_violation"
            break

    final_nodes = nodes_history[-1]
    metrics: dict[str, object] = {
        "case": case.name,
        "pattern": pattern_name,
        "status": status,
        "completed_steps": len(action_history),
        "requested_steps": total_steps,
        "action_limit": case.action_limit,
        "muscle_torque_scale": case.muscle_torque_scale,
        "maximum_tip_speed_mps": maximum_tip_speed,
        "minimum_ground_clearance_m": minimum_ground_clearance,
        "tip_displacement_m": float(np.linalg.norm(final_nodes[-1] - initial_nodes[-1])),
        **bend_metrics(final_nodes),
    }
    env.close()
    return (
        metrics,
        np.asarray(nodes_history, dtype=np.float64),
        np.asarray(action_history, dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    if args.ramp_steps < 1 or args.hold_steps < 0:
        raise ValueError("ramp_steps must be positive and hold_steps non-negative")
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cases = [
        Case(
            "baseline_e10mpa_r50mm_t30_a030",
            Path(args.baseline_scenario).expanduser().resolve(),
            0.30,
            30.0,
        ),
        Case(
            "strong_e2mpa_r45mm_t45_a060",
            Path(args.strong_scenario).expanduser().resolve(),
            0.60,
            45.0,
        ),
    ]
    for case in cases:
        if not case.scenario.is_file():
            raise FileNotFoundError(case.scenario)

    rows: list[dict[str, object]] = []
    patterns = activation_patterns()
    for case_index, case in enumerate(cases):
        case_dir = output / case.name
        case_dir.mkdir()
        for pattern_index, (name, normalized_target) in enumerate(patterns):
            metrics, nodes, actions = run_trial(
                case,
                name,
                normalized_target,
                args.ramp_steps,
                args.hold_steps,
                args.seed + 100 * case_index + pattern_index,
            )
            rows.append(metrics)
            trial_path = case_dir / f"{pattern_index:02d}_{name}.npz"
            with trial_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    nodes=nodes,
                    actions=actions,
                    normalized_target=normalized_target,
                )
            print(json.dumps(metrics, sort_keys=True), flush=True)

    safe_rows = [row for row in rows if row["status"] == "completed"]
    best_by_case: dict[str, dict[str, object] | None] = {}
    for case in cases:
        eligible = [row for row in safe_rows if row["case"] == case.name]
        best_by_case[case.name] = (
            max(eligible, key=lambda row: float(row["distal_turn_deg"]))
            if eligible
            else None
        )

    baseline = yaml.safe_load(cases[0].scenario.read_text(encoding="utf-8"))["softrobot"]
    strong = yaml.safe_load(cases[1].scenario.read_text(encoding="utf-8"))["softrobot"]
    passive_ei_ratio = (
        float(strong["youngs_modulus"])
        / float(baseline["youngs_modulus"])
        * (float(strong["radius"]) / float(baseline["radius"])) ** 4
    )
    provenance = {
        "schema_version": 1,
        "kind": "manisoft_bending_capacity_ablation",
        "git_head": git_head(repo),
        "seed": args.seed,
        "ramp_steps": args.ramp_steps,
        "hold_steps": args.hold_steps,
        "speed_termination_enabled": False,
        "ground_violation_tolerance_m": 0.0005,
        "cases": [
            {
                "name": case.name,
                "scenario": str(case.scenario),
                "scenario_sha256": sha256(case.scenario),
                "action_limit": case.action_limit,
                "muscle_torque_scale": case.muscle_torque_scale,
            }
            for case in cases
        ],
        "theory": {
            "strong_to_baseline_passive_EI_ratio": passive_ei_ratio,
            "strong_to_baseline_torque_per_unit_activation_ratio": 45.0 / 30.0,
            "strong_to_baseline_max_torque_ratio": (45.0 * 0.60) / (30.0 * 0.30),
            "strong_to_baseline_max_linear_curvature_authority_ratio": (
                (45.0 * 0.60) / (30.0 * 0.30) / passive_ei_ratio
            ),
        },
        "best_completed_trial_by_case": best_by_case,
        "trials": rows,
    }
    result_path = output / "results.json"
    result_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"results": str(result_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
