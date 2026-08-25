#!/usr/bin/env python
"""Search stable arch poses whose terminal tangent points toward a table."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np

from antmaze_ac.envs.kinematic_push_task import segment_aabb_distance
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.table_entry_bank import (
    load_table_entry_trajectory_bank,
    restore_rod_internal_state,
)


_WORKER: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--bank",
        default="data/processed/manisoft_table_entry_bank_v1/entry_bank.npz",
    )
    parser.add_argument("--entry-index", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--action-limit", type=float, default=0.30)
    parser.add_argument("--target-tip-z", type=float, default=0.46)
    parser.add_argument("--target-tip-x", type=float)
    parser.add_argument("--target-tip-y", type=float)
    parser.add_argument("--tip-xy-weight", type=float, default=0.0)
    parser.add_argument("--tip-z-weight", type=float, default=240.0)
    parser.add_argument("--orientation-weight", type=float, default=2.0)
    parser.add_argument(
        "--arch-height-weight",
        type=float,
        default=80.0,
        help="Penalty weight for arch height below --minimum-arch-height.",
    )
    parser.add_argument("--minimum-arch-height", type=float, default=0.08)
    parser.add_argument(
        "--maximum-angle-degrees",
        type=float,
        default=35.0,
        help="Maximum terminal downward angle used by pass/fail labels.",
    )
    parser.add_argument("--clearance-weight", type=float, default=0.0)
    parser.add_argument(
        "--hold-span-weight",
        type=float,
        default=500.0,
        help=(
            "Penalty on short-rollout tip motion above 3 mm. Use zero for "
            "broad transient screening, then long-rollout certification."
        ),
    )
    parser.add_argument("--search-scale", type=float, default=1.0)
    parser.add_argument("--initial-report")
    parser.add_argument("--keep-candidates", type=int, default=100)
    parser.add_argument("--transition-steps", type=int, default=180)
    parser.add_argument("--hold-steps", type=int, default=80)
    parser.add_argument(
        "--table-surface-z",
        type=float,
        default=None,
        help="Override the entry-bank table height for clearance checks.",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def _minimum_jerk(value: float) -> float:
    fraction = float(np.clip(value, 0.0, 1.0))
    return fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)


def _initialize_worker(
    scenario: str,
    bank_path: str,
    entry_index: int,
    transition_steps: int,
    hold_steps: int,
    action_limit: float,
    target_tip_z: float,
    target_tip_x: float | None,
    target_tip_y: float | None,
    tip_xy_weight: float,
    tip_z_weight: float,
    orientation_weight: float,
    arch_height_weight: float,
    minimum_arch_height: float,
    maximum_angle_degrees: float,
    clearance_weight: float,
    hold_span_weight: float,
    table_surface_z: float,
) -> None:
    bank = load_table_entry_trajectory_bank(bank_path)
    env = ManiSoftTipTrackingEnv(
        scenario,
        target_tip=(0.0, 0.0, 0.5),
        absolute_action_limit=action_limit,
    )
    env.reset(seed=entry_index)
    _WORKER.update(
        {
            "bank": bank,
            "env": env,
            "entry_index": entry_index,
            "transition_steps": transition_steps,
            "hold_steps": hold_steps,
            "action_limit": action_limit,
            "target_tip_z": target_tip_z,
            "target_tip_x": target_tip_x,
            "target_tip_y": target_tip_y,
            "tip_xy_weight": tip_xy_weight,
            "tip_z_weight": tip_z_weight,
            "orientation_weight": orientation_weight,
            "arch_height_weight": arch_height_weight,
            "minimum_arch_height": minimum_arch_height,
            "maximum_angle_degrees": maximum_angle_degrees,
            "clearance_weight": clearance_weight,
            "hold_span_weight": hold_span_weight,
            "table_surface_z": table_surface_z,
            "base_time": bank.transition_count * bank.control_dt,
            "base_step": bank.transition_count
            * int(round(bank.control_dt / env.sim._backend.dt)),
        }
    )


def _restore_entry() -> tuple[np.ndarray, Any, ManiSoftTipTrackingEnv]:
    bank = _WORKER["bank"]
    env = _WORKER["env"]
    entry_index = _WORKER["entry_index"]
    rod = env.sim._backend._softrobot
    rod.position_collection[...] = bank.node_positions[entry_index, -1].T
    rod.velocity_collection[...] = bank.node_velocities[entry_index, -1].T
    rod.director_collection[...] = bank.element_directors[
        entry_index, -1
    ].transpose(1, 2, 0)
    rod.omega_collection[...] = bank.element_omegas[entry_index, -1].T
    restore_rod_internal_state(rod, bank.rod_internal_states[entry_index, -1])
    env.sim._backend.time_tracker = _WORKER["base_time"]
    env.sim.current_step = _WORKER["base_step"]
    start_action = np.asarray(bank.actions[entry_index, -1], dtype=np.float32)
    env.muscle.set_activation(start_action.reshape(6, 3))
    return start_action, bank, env


def _clearance(nodes: np.ndarray, bank) -> float:
    minimum = np.asarray(
        [bank.table_x_bounds[0], bank.table_y_bounds[0], -2.0]
    )
    maximum = np.asarray(
        [
            bank.table_x_bounds[1],
            bank.table_y_bounds[1],
            _WORKER["table_surface_z"],
        ]
    )
    return float(
        min(
            segment_aabb_distance(start, end, minimum, maximum)
            for start, end in zip(nodes[:-1], nodes[1:])
        )
        - bank.arm_radius
        - bank.safety_margin
    )


def _score(row: dict[str, Any]) -> float:
    if not row["finite"]:
        return 1e9
    tip = np.asarray(row["final_tip"])
    xy_penalty = (
        max(-0.55 - tip[0], 0.0)
        + max(tip[0] - 0.55, 0.0)
        + max(0.35 - tip[1], 0.0)
        + max(tip[1] - 0.90, 0.0)
    )
    target_xy_penalty = 0.0
    if (
        _WORKER["target_tip_x"] is not None
        and _WORKER["target_tip_y"] is not None
    ):
        target_xy_penalty = float(
            np.linalg.norm(
                tip[:2]
                - np.asarray(
                    [_WORKER["target_tip_x"], _WORKER["target_tip_y"]]
                )
            )
        )
    return float(
        _WORKER["orientation_weight"]
        * row["tip_downward_angle_degrees"]
        + 120.0 * xy_penalty
        + _WORKER["tip_xy_weight"] * target_xy_penalty
        + _WORKER["tip_z_weight"]
        * abs(tip[2] - _WORKER["target_tip_z"])
        + _WORKER["arch_height_weight"]
        * max(
            _WORKER["minimum_arch_height"] - row["arch_height_above_tip"],
            0.0,
        )
        + _WORKER["clearance_weight"]
        * max(-row["minimum_table_clearance"], 0.0)
        + _WORKER["hold_span_weight"]
        * max(row["hold_tip_span"] - 0.003, 0.0)
    )


def _probe(job: tuple[int, np.ndarray]) -> dict[str, Any]:
    candidate_index, target_action = job
    start_action, bank, env = _restore_entry()
    transition_steps = _WORKER["transition_steps"]
    hold_steps = _WORKER["hold_steps"]
    tips: list[np.ndarray] = []
    minimum_clearance = float("inf")
    finite = True
    nodes = np.asarray(bank.node_positions[_WORKER["entry_index"], -1])
    try:
        for step in range(transition_steps + hold_steps):
            blend = _minimum_jerk((step + 1) / transition_steps)
            action = start_action + blend * (target_action - start_action)
            env.muscle.set_activation(action.reshape(6, 3))
            env.sim.step_with_torque_callback(
                lambda lengths: env.muscle.evaluate(lengths)
            )
            nodes = np.asarray(
                env.sim._backend.softrobot_state.element_positions,
                dtype=np.float64,
            )
            if not np.isfinite(nodes).all():
                finite = False
                break
            tips.append(nodes[-1].copy())
            minimum_clearance = min(minimum_clearance, _clearance(nodes, bank))
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        finite = False
    if not finite or not tips:
        return {
            "candidate_index": candidate_index,
            "finite": False,
            "action": np.asarray(target_action).tolist(),
            "score": 1e9,
        }
    tips_array = np.asarray(tips)
    final_tip = tips_array[-1]
    tip_tangent = nodes[-1] - nodes[-2]
    tip_tangent /= max(float(np.linalg.norm(tip_tangent)), 1e-12)
    downward_angle = float(
        np.rad2deg(np.arccos(np.clip(-tip_tangent[2], -1.0, 1.0)))
    )
    row = {
        "candidate_index": candidate_index,
        "finite": True,
        "action": np.asarray(target_action).tolist(),
        "final_tip": final_tip.tolist(),
        "tip_tangent": tip_tangent.tolist(),
        "tip_downward_angle_degrees": downward_angle,
        "arch_height_above_tip": float(np.max(nodes[:, 2]) - final_tip[2]),
        "arch_peak_node_index": int(np.argmax(nodes[:, 2])),
        "minimum_table_clearance": minimum_clearance,
        "hold_tip_span": float(
            np.max(np.ptp(tips_array[-hold_steps:], axis=0))
        ),
    }
    row["score"] = _score(row)
    row["orientation_pass"] = bool(
        downward_angle <= _WORKER["maximum_angle_degrees"]
        and row["arch_height_above_tip"] >= 0.08
    )
    row["current_table_pass"] = bool(
        row["orientation_pass"]
        and minimum_clearance >= 0.0
        and 0.415 <= final_tip[2] <= 0.58
        and -0.55 <= final_tip[0] <= 0.55
        and 0.35 <= final_tip[1] <= 0.90
        and row["hold_tip_span"] <= 0.003
    )
    return row


def _structured_candidates(
    rng: np.random.Generator,
    base_action: np.ndarray,
    count: int,
    action_limit: float,
    search_scale: float,
) -> np.ndarray:
    base = np.asarray(base_action, dtype=np.float64).reshape(6, 3)
    rows = [base.copy()]
    control_fraction = np.linspace(0.0, 1.0, 6)
    while len(rows) < count:
        if search_scale < 1.0:
            distal_weight = np.linspace(0.35, 1.0, 6)[:, None]
            noise = rng.normal(
                0.0,
                0.40 * action_limit * search_scale,
                size=(6, 3),
            ) * distal_weight
            noise[:, 2] *= 0.5
            rows.append(np.clip(base + noise, -action_limit, action_limit))
            continue
        mode = int(rng.integers(0, 3))
        if mode == 0:
            noise_scale = np.linspace(0.035, 0.80 * action_limit, 6)[:, None]
            noise = rng.normal(size=(6, 3)) * noise_scale
            noise[:, 2] *= 0.45
            candidate = base + noise
        elif mode == 1:
            start_angle = rng.uniform(-np.pi, np.pi)
            angle_sweep = rng.uniform(np.deg2rad(80), np.deg2rad(300))
            angles = start_angle + angle_sweep * control_fraction
            magnitude = rng.uniform(0.50 * action_limit, 1.40 * action_limit)
            candidate = np.zeros((6, 3), dtype=np.float64)
            candidate[:, 0] = magnitude * np.cos(angles)
            candidate[:, 1] = magnitude * np.sin(angles)
            candidate[:, 2] = rng.normal(0.0, 0.06, size=6)
        else:
            distal = rng.uniform(-action_limit, action_limit, size=(6, 3))
            blend = np.linspace(0.0, 1.0, 6)[:, None] ** rng.uniform(0.7, 2.0)
            candidate = (1.0 - blend) * base + blend * distal
            candidate[:, 2] *= 0.65
        rows.append(np.clip(candidate, -action_limit, action_limit))
    return np.asarray(rows, dtype=np.float32).reshape(count, 18)


def _mutated_candidates(
    rng: np.random.Generator,
    elites: np.ndarray,
    count: int,
    round_index: int,
    action_limit: float,
    search_scale: float,
) -> np.ndarray:
    rows = [row.copy() for row in elites]
    sigma = (
        0.37
        * action_limit
        * search_scale
        * (0.65 ** max(round_index - 1, 0))
    )
    distal_weight = np.linspace(0.45, 1.0, 6)[:, None]
    while len(rows) < count:
        parent = elites[int(rng.integers(0, len(elites)))].reshape(6, 3)
        noise = rng.normal(0.0, sigma, size=(6, 3)) * distal_weight
        noise[:, 2] *= 0.5
        rows.append(
            np.clip(parent + noise, -action_limit, action_limit).reshape(-1)
        )
    return np.asarray(rows[:count], dtype=np.float32)


def main() -> None:
    args = parse_args()
    if min(
        args.samples,
        args.rounds,
        args.elite_count,
        args.workers,
        args.transition_steps,
        args.hold_steps,
        args.keep_candidates,
    ) < 1 or min(
        args.action_limit,
        args.search_scale,
        args.maximum_angle_degrees,
    ) <= 0:
        raise ValueError("search sizes and rollout lengths must be positive")
    if args.elite_count >= args.samples:
        raise ValueError("elite-count must be smaller than samples")
    if (args.target_tip_x is None) != (args.target_tip_y is None):
        raise ValueError("target-tip-x and target-tip-y must be supplied together")
    if min(
        args.tip_xy_weight,
        args.tip_z_weight,
        args.orientation_weight,
        args.arch_height_weight,
        args.minimum_arch_height,
        args.clearance_weight,
        args.hold_span_weight,
    ) < 0:
        raise ValueError("search score weights must be non-negative")
    scenario = Path(args.scenario).expanduser().resolve()
    bank_path = Path(args.bank).expanduser().resolve()
    bank = load_table_entry_trajectory_bank(bank_path)
    table_surface_z = (
        bank.table_surface_z
        if args.table_surface_z is None
        else float(args.table_surface_z)
    )
    if not 0 <= args.entry_index < bank.trajectory_count:
        raise ValueError("entry-index is out of range")
    base_action = bank.actions[args.entry_index, -1]
    rng = np.random.default_rng(args.seed)
    candidates = _structured_candidates(
        rng,
        base_action,
        args.samples,
        args.action_limit,
        args.search_scale,
    )
    if args.initial_report is not None:
        initial_report_path = Path(args.initial_report).expanduser().resolve()
        initial_report = json.loads(
            initial_report_path.read_text(encoding="utf-8")
        )
        initial_rows = [
            row
            for row in initial_report.get("best_candidates", [])
            if row.get("finite", False)
        ]

        # A report may be reused for a different XY target. Stored scores then
        # describe the old objective, so rerank its seeds for this invocation.
        def initial_score(row: dict[str, Any]) -> float:
            tip = np.asarray(row["final_tip"], dtype=np.float64)
            target_xy_penalty = 0.0
            if args.target_tip_x is not None and args.target_tip_y is not None:
                target_xy_penalty = float(
                    np.linalg.norm(
                        tip[:2]
                        - np.asarray(
                            [args.target_tip_x, args.target_tip_y],
                            dtype=np.float64,
                        )
                    )
                )
            return float(
                args.tip_xy_weight * target_xy_penalty
                + args.tip_z_weight * abs(tip[2] - args.target_tip_z)
                + args.orientation_weight
                * float(row.get("tip_downward_angle_degrees", 180.0))
                + args.clearance_weight
                * max(-float(row.get("minimum_table_clearance", -1.0)), 0.0)
                + args.hold_span_weight
                * max(float(row.get("hold_tip_span", 1.0)) - 0.003, 0.0)
            )

        initial_rows.sort(key=initial_score)
        initial_actions = np.asarray(
            [row["action"] for row in initial_rows[: args.elite_count]],
            dtype=np.float32,
        )
        if initial_actions.ndim != 2 or initial_actions.shape[1] != 18:
            raise ValueError("initial report contains no finite 18-D candidates")
        candidates = _mutated_candidates(
            rng,
            initial_actions,
            args.samples,
            3,
            args.action_limit,
            args.search_scale,
        )
    rounds: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(
            str(scenario),
            str(bank_path),
            args.entry_index,
            args.transition_steps,
            args.hold_steps,
            args.action_limit,
            args.target_tip_z,
            args.target_tip_x,
            args.target_tip_y,
            args.tip_xy_weight,
            args.tip_z_weight,
            args.orientation_weight,
            args.arch_height_weight,
            args.minimum_arch_height,
            args.maximum_angle_degrees,
            args.clearance_weight,
            args.hold_span_weight,
            table_surface_z,
        ),
    ) as executor:
        for round_index in range(args.rounds):
            jobs = list(enumerate(candidates))
            rows = list(executor.map(_probe, jobs, chunksize=1))
            rows.sort(key=lambda row: row["score"])
            all_rows.extend(rows)
            finite = [row for row in rows if row["finite"]]
            summary = {
                "round": round_index,
                "finite": len(finite),
                "minimum_score": float(rows[0]["score"]),
                "minimum_downward_angle_degrees": float(
                    min(
                        (
                            row["tip_downward_angle_degrees"]
                            for row in finite
                        ),
                        default=np.inf,
                    )
                ),
                "orientation_passes": sum(
                    bool(row.get("orientation_pass", False)) for row in rows
                ),
                "current_table_passes": sum(
                    bool(row.get("current_table_pass", False)) for row in rows
                ),
                "best": rows[0],
            }
            rounds.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
            elites = np.asarray(
                [row["action"] for row in rows[: args.elite_count]],
                dtype=np.float32,
            )
            candidates = _mutated_candidates(
                rng,
                elites,
                args.samples,
                round_index + 1,
                args.action_limit,
                args.search_scale,
            )

    all_rows.sort(key=lambda row: row["score"])
    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for row in all_rows:
        key = tuple(np.round(row["action"], 6))
        if key not in seen:
            unique.append(row)
            seen.add(key)
        if len(unique) == args.keep_candidates:
            break
    report = {
        "kind": "manisoft_arch_pose_search",
        "scenario": str(scenario),
        "bank": str(bank_path),
        "entry_index": args.entry_index,
        "entry_name": bank.names[args.entry_index],
        "seed": args.seed,
        "samples_per_round": args.samples,
        "round_count": args.rounds,
        "transition_steps": args.transition_steps,
        "hold_steps": args.hold_steps,
        "action_limit": args.action_limit,
        "target_tip_z": args.target_tip_z,
        "target_tip_x": args.target_tip_x,
        "target_tip_y": args.target_tip_y,
        "tip_xy_weight": args.tip_xy_weight,
        "tip_z_weight": args.tip_z_weight,
        "orientation_weight": args.orientation_weight,
        "arch_height_weight": args.arch_height_weight,
        "minimum_arch_height": args.minimum_arch_height,
        "maximum_angle_degrees": args.maximum_angle_degrees,
        "clearance_weight": args.clearance_weight,
        "hold_span_weight": args.hold_span_weight,
        "table_surface_z": table_surface_z,
        "search_scale": args.search_scale,
        "initial_report": (
            str(Path(args.initial_report).expanduser().resolve())
            if args.initial_report is not None
            else None
        ),
        "rounds": rounds,
        "best_candidates": unique,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
