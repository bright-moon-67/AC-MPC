#!/usr/bin/env python
"""Evaluate a ManiSoft PPO or offline IQL policy checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_tracking_env import (
    MANISOFT_WAYPOINT_SUCCESS_STREAK,
    MANISOFT_WAYPOINT_SUCCESS_THRESHOLD,
    ManiSoftThreeWaypointTrackingEnv,
    load_manisoft_waypoint_reference_bank,
)
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.iql import load_manisoft_iql_checkpoint
from antmaze_ac.rl.manisoft_ppo_policies import load_manisoft_ppo_checkpoint


TIP_INDICES = (30, 31, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--waypoint-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-other-waypoint-bank",
        action="store_true",
        help="Permit an independent waypoint bank for generalization testing.",
    )
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

    checkpoint_header = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    loader = (
        load_manisoft_iql_checkpoint
        if checkpoint_header.get("method") == "manisoft_kmpc_iql"
        else load_manisoft_ppo_checkpoint
    )
    policy, payload, koopman_payload = loader(checkpoint, device)
    policy.eval()
    waypoint_root_value = args.waypoint_root or payload.get("waypoint_root")
    if waypoint_root_value is None:
        raise ValueError("--waypoint-root is required")
    waypoint_root = Path(waypoint_root_value).expanduser().resolve()
    waypoint_bank = load_manisoft_waypoint_reference_bank(waypoint_root)
    if waypoint_bank.scenario_sha256 != sha256(scenario):
        raise ValueError("Waypoint bank was certified with another scenario")
    if (
        not args.allow_other_waypoint_bank
        and payload.get("waypoint_bank_sha256")
        != waypoint_bank.manifest_sha256
    ):
        raise ValueError("Checkpoint references another waypoint bank")
    waypoint_tips = waypoint_bank.states[:, :, np.asarray(TIP_INDICES)]
    runtime = payload["runtime"]
    state_stats = koopman_payload["normalizers"]["state"]

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    schedule = np.concatenate(
        [
            rng.permutation(waypoint_bank.triplet_count)
            for _ in range(
                (args.episodes + waypoint_bank.triplet_count - 1)
                // waypoint_bank.triplet_count
            )
        ]
    )[: args.episodes]
    summaries: list[dict] = []
    inference_times: list[float] = []
    started = time.perf_counter()

    for episode, triplet_index in enumerate(schedule):
        base = ManiSoftThreeWaypointTrackingEnv(
            scenario,
            waypoint_tips=waypoint_tips,
            episode_steps=args.episode_steps,
            absolute_action_limit=float(runtime["absolute_action_limit"]),
            progress_reward_scale=float(runtime.get("progress_reward_scale", 1.0)),
            success_threshold=float(
                runtime.get(
                    "success_threshold", MANISOFT_WAYPOINT_SUCCESS_THRESHOLD
                )
            ),
            success_streak=int(
                runtime.get(
                    "required_success_streak", MANISOFT_WAYPOINT_SUCCESS_STREAK
                )
            ),
        )
        env = HistoryContextTrackingWrapper(
            base,
            history_steps=policy.history_steps,
            state_mean=state_stats["mean"],
            state_std=state_stats["std"],
            tip_indices=TIP_INDICES,
            max_delta=(
                None
                if payload["actor_name"] != "ppo_kmpc"
                or runtime.get("max_delta") is None
                else float(runtime["max_delta"])
            ),
        )
        observations: list[np.ndarray] = []
        requested_actions: list[np.ndarray] = []
        policy_actions: list[np.ndarray] = []
        applied_actions: list[np.ndarray] = []
        rewards: list[float] = []
        distances: list[float] = []
        all_distances: list[np.ndarray] = []
        active_indices: list[int] = []
        completed: list[int] = []
        waypoint_events: list[bool] = []
        residuals: list[float] = []
        try:
            observation, info = env.reset(
                seed=args.seed + episode,
                options={"waypoint_triplet_index": int(triplet_index)},
            )
            distances.append(float(info["distance"]))
            all_distances.append(
                np.asarray(info["all_waypoint_distances"], dtype=np.float32)
            )
            terminated = truncated = False
            while not (terminated or truncated):
                tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                )
                inference_started = time.perf_counter()
                with torch.no_grad():
                    action, _, _, policy_output = policy.act(
                        tensor,
                        deterministic=True,
                        return_output=True,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_times.append(time.perf_counter() - inference_started)
                requested = action.detach().cpu().numpy()
                following, reward, terminated, truncated, info = env.step(
                    requested
                )
                observations.append(observation.copy())
                policy_actions.append(requested.copy())
                requested_actions.append(
                    np.asarray(
                        info["requested_absolute_action"], dtype=np.float32
                    )
                )
                applied_actions.append(
                    np.asarray(info["applied_action"], dtype=np.float32)
                )
                rewards.append(float(reward))
                distances.append(float(info["distance"]))
                all_distances.append(
                    np.asarray(
                        info["all_waypoint_distances"], dtype=np.float32
                    )
                )
                active_indices.append(int(info["active_waypoint_index"]))
                completed.append(int(info["waypoints_completed"]))
                waypoint_events.append(bool(info["waypoint_passed"]))
                if payload["actor_name"] == "ppo_kmpc":
                    residuals.append(
                        float(
                            policy_output.mpc.projected_gradient_residual.mean()
                        )
                    )
                observation = following
        finally:
            env.close()

        action_array = np.asarray(applied_actions, dtype=np.float32)
        delta_array = np.diff(
            action_array,
            axis=0,
            prepend=np.zeros((1, action_array.shape[1]), dtype=np.float32),
        )
        all_distance_array = np.asarray(all_distances, dtype=np.float32)
        summary = {
            "episode": episode,
            "waypoint_triplet_index": int(triplet_index),
            "steps": len(rewards),
            "return": float(np.sum(rewards)),
            "success": bool(terminated),
            "waypoints_completed": max(completed, default=0),
            "final_distance": float(distances[-1]),
            "minimum_distance": float(np.min(distances)),
            "waypoint_minimum_distances": all_distance_array.min(axis=0).tolist(),
            "waypoint_pass_events": int(np.sum(waypoint_events)),
            "applied_action_abs_mean": float(np.mean(np.abs(action_array))),
            "applied_delta_action_l2_mean": float(
                np.linalg.norm(delta_array, axis=1).mean()
            ),
            "applied_delta_action_abs_max": float(
                np.max(np.abs(delta_array))
            ),
            "normalized_delta_abs_mean": (
                None
                if payload["actor_name"] != "ppo_kmpc"
                or runtime.get("max_delta") is None
                else float(np.mean(np.abs(np.asarray(policy_actions))))
            ),
            "action_bound_rate": float(
                np.mean(
                    np.abs(action_array)
                    >= float(runtime["absolute_action_limit"]) - 1e-6
                )
            ),
            "projected_gradient_residual_mean": (
                None if not residuals else float(np.mean(residuals))
            ),
        }
        summaries.append(summary)
        np.savez_compressed(
            output / f"trajectory_{episode:04d}.npz",
            observation=np.asarray(observations, dtype=np.float32),
            policy_action=np.asarray(policy_actions, dtype=np.float32),
            requested_action=np.asarray(requested_actions, dtype=np.float32),
            applied_action=action_array,
            reward=np.asarray(rewards, dtype=np.float32),
            distance=np.asarray(distances, dtype=np.float32),
            all_waypoint_distances=all_distance_array,
            active_waypoint_index=np.asarray(active_indices, dtype=np.int64),
            waypoints_completed=np.asarray(completed, dtype=np.int64),
            waypoint_passed=np.asarray(waypoint_events, dtype=np.bool_),
        )
        print(json.dumps(summary, sort_keys=True), flush=True)

    inference = np.asarray(inference_times, dtype=np.float64)
    report = {
        "method": payload["method"],
        "actor_name": payload["actor_name"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "scenario": str(scenario),
        "waypoint_root": str(waypoint_root),
        "waypoint_bank_sha256": waypoint_bank.manifest_sha256,
        "episodes": args.episodes,
        "success_rate": float(np.mean([row["success"] for row in summaries])),
        "waypoints_completed_mean": float(
            np.mean([row["waypoints_completed"] for row in summaries])
        ),
        "return_mean": float(np.mean([row["return"] for row in summaries])),
        "episode_steps_mean": float(np.mean([row["steps"] for row in summaries])),
        "final_distance_mean": float(
            np.mean([row["final_distance"] for row in summaries])
        ),
        "applied_action_abs_mean": float(
            np.mean([row["applied_action_abs_mean"] for row in summaries])
        ),
        "applied_delta_action_l2_mean": float(
            np.mean([row["applied_delta_action_l2_mean"] for row in summaries])
        ),
        "applied_delta_action_abs_max": float(
            np.max([row["applied_delta_action_abs_max"] for row in summaries])
        ),
        "normalized_delta_abs_mean": (
            None
            if payload["actor_name"] != "ppo_kmpc"
            or runtime.get("max_delta") is None
            else float(
                np.mean(
                    [row["normalized_delta_abs_mean"] for row in summaries]
                )
            )
        ),
        "action_bound_rate": float(
            np.mean([row["action_bound_rate"] for row in summaries])
        ),
        "inference_seconds_mean": float(inference.mean()),
        "inference_seconds_p95": float(np.quantile(inference, 0.95)),
        "wall_seconds": float(time.perf_counter() - started),
        "runtime": runtime,
        "episode_summaries": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
