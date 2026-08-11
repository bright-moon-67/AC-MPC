#!/usr/bin/env python
"""Evaluate a BC-pretrained or PPO-fine-tuned soft-robot BC-KMPC policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_tracking_env import (
    ManiSoftThreeWaypointTrackingEnv,
    load_manisoft_waypoint_references,
)
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.serialization import load_history_mpc_checkpoint


TIP_INDICES = (30, 31, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--waypoint-root",
        default=None,
        help="Directory containing ref_4cm/ref_8cm/ref_12cm and actions/.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.episodes, args.episode_steps) < 1:
        raise ValueError("episodes and episode-steps must be positive")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    if not checkpoint.is_file() or not scenario.is_file():
        raise FileNotFoundError("Checkpoint and scenario must exist")
    policy, policy_payload, koopman_payload = load_history_mpc_checkpoint(
        checkpoint,
        device,
    )
    policy.eval()
    waypoint_root_value = args.waypoint_root or policy_payload.get("waypoint_root")
    if waypoint_root_value is None:
        raise ValueError(
            "--waypoint-root is required for a checkpoint without waypoint metadata"
        )
    waypoint_root = Path(waypoint_root_value).expanduser().resolve()
    reference_states, _, reference_paths, action_paths = (
        load_manisoft_waypoint_references(waypoint_root)
    )
    waypoint_tips = reference_states[:, np.asarray(TIP_INDICES)]
    reference_hashes = [sha256(path) for path in reference_paths]
    action_hashes = [sha256(path) for path in action_paths]
    if policy.waypoint_count != 3:
        raise ValueError("Checkpoint is not a three-waypoint BC-KMPC policy")
    if (
        policy_payload.get("reference_sha256") != reference_hashes
        or policy_payload.get("action_sha256") != action_hashes
    ):
        raise ValueError("Checkpoint references another waypoint set")
    runtime = policy_payload["runtime"]
    absolute_action_limit = float(runtime["absolute_action_limit"])
    max_delta = float(runtime["max_delta"])
    state_stats = koopman_payload["normalizers"]["state"]

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories: list[dict[str, np.ndarray]] = []
    episode_summaries: list[dict] = []
    inference_times: list[float] = []
    started = time.perf_counter()
    for episode in range(args.episodes):
        base = ManiSoftThreeWaypointTrackingEnv(
            scenario,
            waypoint_tips=waypoint_tips,
            episode_steps=args.episode_steps,
            absolute_action_limit=absolute_action_limit,
        )
        env = HistoryContextTrackingWrapper(
            base,
            history_steps=policy.history_steps,
            state_mean=state_stats["mean"],
            state_std=state_stats["std"],
            max_delta=max_delta,
            tip_indices=TIP_INDICES,
        )
        observations: list[np.ndarray] = []
        requested_actions: list[np.ndarray] = []
        applied_actions: list[np.ndarray] = []
        distances: list[float] = []
        rewards: list[float] = []
        residuals: list[float] = []
        active_waypoint_indices: list[int] = []
        completed_waypoints: list[int] = []
        waypoint_events: list[bool] = []
        all_waypoint_distances: list[np.ndarray] = []
        try:
            observation, info = env.reset(seed=args.seed + episode)
            distances.append(float(info["distance"]))
            all_waypoint_distances.append(
                np.asarray(info["all_waypoint_distances"], dtype=np.float32)
            )
            terminated = truncated = False
            while not (terminated or truncated):
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                )
                inference_started = time.perf_counter()
                with torch.no_grad():
                    action, _, _, policy_output = policy.act(
                        observation_tensor,
                        deterministic=True,
                        return_output=True,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_times.append(time.perf_counter() - inference_started)
                requested = action.detach().cpu().numpy()
                next_observation, reward, terminated, truncated, info = env.step(
                    requested
                )
                observations.append(observation.copy())
                requested_actions.append(requested.copy())
                applied_actions.append(
                    np.asarray(info["applied_action"], dtype=np.float32)
                )
                distances.append(float(info["distance"]))
                rewards.append(float(reward))
                residuals.append(
                    float(policy_output.mpc.projected_gradient_residual.mean())
                )
                active_waypoint_indices.append(int(info["active_waypoint_index"]))
                completed_waypoints.append(int(info["waypoints_completed"]))
                waypoint_events.append(bool(info["waypoint_passed"]))
                all_waypoint_distances.append(
                    np.asarray(
                        info["all_waypoint_distances"], dtype=np.float32
                    )
                )
                observation = next_observation
        finally:
            env.close()
        distance_array = np.asarray(distances, dtype=np.float32)
        all_distance_array = np.asarray(
            all_waypoint_distances, dtype=np.float32
        )
        final_completed = max(completed_waypoints, default=0)
        summary = {
            "episode": episode,
            "steps": len(rewards),
            "return": float(np.sum(rewards)),
            "initial_distance": float(distance_array[0]),
            "final_distance": float(distance_array[-1]),
            "minimum_distance": float(distance_array.min()),
            "success": bool(terminated),
            "waypoints_completed": final_completed,
            "waypoint_minimum_distances": all_distance_array.min(axis=0).tolist(),
            "waypoint_pass_events": int(np.sum(waypoint_events)),
            "projected_gradient_residual_mean": float(np.mean(residuals)),
        }
        episode_summaries.append(summary)
        trajectories.append(
            {
                "observation": np.asarray(observations, dtype=np.float32),
                "requested_action": np.asarray(requested_actions, dtype=np.float32),
                "applied_action": np.asarray(applied_actions, dtype=np.float32),
                "distance": distance_array,
                "reward": np.asarray(rewards, dtype=np.float32),
                "active_waypoint_index": np.asarray(
                    active_waypoint_indices, dtype=np.int64
                ),
                "waypoints_completed": np.asarray(
                    completed_waypoints, dtype=np.int64
                ),
                "waypoint_passed": np.asarray(waypoint_events, dtype=np.bool_),
                "all_waypoint_distances": all_distance_array,
            }
        )
        print(json.dumps(summary, sort_keys=True), flush=True)

    for episode, trajectory in enumerate(trajectories):
        np.savez_compressed(
            output / f"trajectory_{episode:04d}.npz",
            **trajectory,
        )
    inference_array = np.asarray(inference_times, dtype=np.float64)
    report = {
        "method": policy_payload["method"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "scenario": str(scenario),
        "waypoint_root": str(waypoint_root),
        "references": [str(path) for path in reference_paths],
        "reference_sha256": reference_hashes,
        "actions": [str(path) for path in action_paths],
        "action_sha256": action_hashes,
        "episodes": args.episodes,
        "success_rate": float(np.mean([row["success"] for row in episode_summaries])),
        "waypoints_completed_mean": float(
            np.mean([row["waypoints_completed"] for row in episode_summaries])
        ),
        "return_mean": float(np.mean([row["return"] for row in episode_summaries])),
        "final_distance_mean": float(
            np.mean([row["final_distance"] for row in episode_summaries])
        ),
        "minimum_distance_mean": float(
            np.mean([row["minimum_distance"] for row in episode_summaries])
        ),
        "inference_seconds_mean": float(inference_array.mean()),
        "inference_seconds_p95": float(np.quantile(inference_array, 0.95)),
        "total_wall_seconds": float(time.perf_counter() - started),
        "runtime": runtime,
        "episode_summaries": episode_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
