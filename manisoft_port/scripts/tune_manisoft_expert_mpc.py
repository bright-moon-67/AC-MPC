#!/usr/bin/env python
"""Evaluate fixed-cost history MPC weight candidates on a waypoint bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.control.history_reference_mpc import FixedCostHistoryKoopmanMPC
from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_tracking_env import (
    ManiSoftThreeWaypointTrackingEnv,
    load_manisoft_waypoint_reference_bank,
)
from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256
from antmaze_ac.koopman.history_model import HistoryDeepKoopman


TIP_INDICES = (30, 31, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--waypoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME,STATE,TIP,ACTION,CONTROL",
    )
    parser.add_argument("--triplet-indices", type=int, nargs="+", required=True)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--qp-max-iterations", type=int, default=4000)
    parser.add_argument("--rollout-noise-std", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_candidate(value: str) -> dict:
    fields = value.split(",")
    if len(fields) != 5:
        raise ValueError(
            f"Candidate must be NAME,STATE,TIP,ACTION,CONTROL, got {value!r}"
        )
    name = fields[0].strip()
    if not name:
        raise ValueError("Candidate name cannot be empty")
    state, tip, action, control = map(float, fields[1:])
    if min(state, tip, action, control) <= 0:
        raise ValueError("Candidate weights must be positive")
    return {
        "name": name,
        "state_weight": state,
        "tip_state_scale": tip,
        "action_weight": action,
        "control_weight": control,
    }


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.koopman_checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    waypoint_root = Path(args.waypoint_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not checkpoint.is_file() or not scenario.is_file():
        raise FileNotFoundError("Koopman checkpoint and scenario must exist")
    bank = load_manisoft_waypoint_reference_bank(waypoint_root)
    if bank.scenario_sha256 != sha256(scenario):
        raise ValueError("Waypoint bank was certified with another scenario")
    if any(index < 0 or index >= bank.triplet_count for index in args.triplet_indices):
        raise ValueError("triplet-indices contains an out-of-range index")
    candidates = [parse_candidate(value) for value in args.candidate]
    if args.rollout_noise_std < 0:
        raise ValueError("rollout-noise-std must be non-negative")
    if len({row["name"] for row in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")

    device = torch.device(args.device)
    model, model_payload = load_checkpoint(checkpoint, map_location=device)
    if not isinstance(model, HistoryDeepKoopman):
        raise ValueError("Checkpoint must contain HistoryDeepKoopman")
    model = model.to(device).freeze_dynamics()
    state_stats = model_payload["normalizers"]["state"]
    state_mean = torch.as_tensor(state_stats["mean"], dtype=torch.float32, device=device)
    state_std = torch.as_tensor(state_stats["std"], dtype=torch.float32, device=device)
    result = {
        "koopman_checkpoint": str(checkpoint),
        "koopman_checkpoint_sha256": sha256(checkpoint),
        "scenario": str(scenario),
        "waypoint_root": str(waypoint_root),
        "waypoint_bank_sha256": bank.manifest_sha256,
        "triplet_indices": args.triplet_indices,
        "episode_steps": args.episode_steps,
        "horizon": args.horizon,
        "rollout_noise_std": args.rollout_noise_std,
        "candidates": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        expert = FixedCostHistoryKoopmanMPC(
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            action_low=np.full(18, -args.absolute_action_limit),
            action_high=np.full(18, args.absolute_action_limit),
            horizon=args.horizon,
            state_weight=candidate["state_weight"],
            tip_state_scale=candidate["tip_state_scale"],
            action_weight=candidate["action_weight"],
            control_weight=candidate["control_weight"],
            qp_max_iterations=args.qp_max_iterations,
        )
        episodes = []
        for run_index, triplet_index in enumerate(args.triplet_indices):
            base = ManiSoftThreeWaypointTrackingEnv(
                scenario,
                waypoint_tips=bank.states[:, :, np.asarray(TIP_INDICES)],
                episode_steps=args.episode_steps,
                absolute_action_limit=args.absolute_action_limit,
            )
            env = HistoryContextTrackingWrapper(
                base,
                history_steps=model.history_steps,
                state_mean=state_stats["mean"],
                state_std=state_stats["std"],
                tip_indices=TIP_INDICES,
            )
            actions = []
            reference_deviations = []
            distances = []
            rewards = []
            qp_iterations = []
            warm_start = None
            noise_rng = np.random.default_rng(args.seed + run_index)
            try:
                observation, info = env.reset(
                    seed=args.seed + run_index,
                    options={"waypoint_triplet_index": triplet_index},
                )
                distances.append(float(info["distance"]))
                terminated = truncated = False
                while not (terminated or truncated):
                    waypoint_index = int(base.active_waypoint_index)
                    reference_action = bank.actions[triplet_index, waypoint_index]
                    state = observation[: model.state_dim]
                    context = observation[
                        model.state_dim : model.state_dim + model.context_dim
                    ]
                    plan = expert.solve(
                        state=state,
                        context=context,
                        reference_state=bank.states[triplet_index, waypoint_index],
                        reference_action=reference_action,
                        initial_actions=warm_start,
                    )
                    action = np.asarray(plan["action"], dtype=np.float32)
                    applied_action = action.copy()
                    if args.rollout_noise_std:
                        applied_action += noise_rng.normal(
                            0.0, args.rollout_noise_std, size=action.shape
                        ).astype(np.float32)
                    observation, reward, terminated, truncated, info = env.step(
                        applied_action
                    )
                    actions.append(action)
                    reference_deviations.append(action - reference_action)
                    distances.append(float(info["distance"]))
                    rewards.append(float(reward))
                    qp_iterations.append(int(plan["qp_iterations"]))
                    sequence = np.asarray(plan["actions"], dtype=np.float32)
                    warm_start = np.concatenate((sequence[1:], sequence[-1:]))
            finally:
                env.close()
            action_array = np.asarray(actions)
            deviation_array = np.asarray(reference_deviations)
            distances_array = np.asarray(distances)
            episode = {
                "triplet_index": triplet_index,
                "steps": len(rewards),
                "return": float(np.sum(rewards)),
                "success": bool(terminated),
                "waypoints_completed": int(base.waypoints_completed),
                "initial_distance_m": float(distances_array[0]),
                "minimum_active_distance_m": float(distances_array.min()),
                "final_active_distance_m": float(distances_array[-1]),
                "action_abs_max": float(np.abs(action_array).max()),
                "action_l2_max": float(np.linalg.norm(action_array, axis=1).max()),
                "action_l2_mean": float(np.linalg.norm(action_array, axis=1).mean()),
                "all_dimension_saturation_rate": float(
                    np.mean(np.all(np.abs(action_array) >= 0.299, axis=1))
                ),
                "element_saturation_rate": float(np.mean(np.abs(action_array) >= 0.299)),
                "reference_deviation_l2_mean": float(
                    np.linalg.norm(deviation_array, axis=1).mean()
                ),
                "qp_iterations_mean": float(np.mean(qp_iterations)),
            }
            episodes.append(episode)
            print(json.dumps({"candidate": candidate["name"], **episode}), flush=True)
        candidate_result = {
            **candidate,
            "success_rate": float(np.mean([row["success"] for row in episodes])),
            "waypoints_completed_mean": float(
                np.mean([row["waypoints_completed"] for row in episodes])
            ),
            "return_mean": float(np.mean([row["return"] for row in episodes])),
            "action_l2_max": float(max(row["action_l2_max"] for row in episodes)),
            "element_saturation_rate": float(
                np.mean([row["element_saturation_rate"] for row in episodes])
            ),
            "episodes": episodes,
        }
        result["candidates"].append(candidate_result)
        write_json(output, result)

    result["ranking"] = [
        row["name"]
        for row in sorted(
            result["candidates"],
            key=lambda row: (
                row["success_rate"],
                row["waypoints_completed_mean"],
                row["return_mean"],
                -row["element_saturation_rate"],
            ),
            reverse=True,
        )
    ]
    write_json(output, result)
    print(json.dumps({"output": str(output), "ranking": result["ranking"]}))


if __name__ == "__main__":
    main()
