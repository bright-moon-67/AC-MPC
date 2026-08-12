#!/usr/bin/env python
"""Evaluate a BC- or PPO-trained H=10 History-MLP baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.serialization import load_history_mlp_checkpoint


TIP_INDICES = (30, 31, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--reference", default=None)
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
    policy, policy_payload, koopman_payload = load_history_mlp_checkpoint(
        checkpoint,
        device,
    )
    policy.eval()
    reference_value = args.reference or policy_payload.get("reference")
    if reference_value is None:
        raise ValueError(
            "--reference is required for a BC checkpoint without reference metadata"
        )
    reference = Path(reference_value).expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"Missing reference: {reference}")
    with np.load(reference, allow_pickle=False) as archive:
        reference_state = np.asarray(
            archive["reference_state"],
            dtype=np.float32,
        ).reshape(-1)
    if reference_state.shape != (45,):
        raise ValueError("Reference state must be 45-D")
    target_tip = reference_state[np.asarray(TIP_INDICES)]
    runtime = policy_payload["runtime"]
    absolute_action_limit = float(runtime["absolute_action_limit"])
    state_stats = koopman_payload["normalizers"]["state"]

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories: list[dict[str, np.ndarray]] = []
    episode_summaries: list[dict] = []
    inference_times: list[float] = []
    started = time.perf_counter()
    for episode in range(args.episodes):
        base = ManiSoftTipTrackingEnv(
            scenario,
            target_tip=target_tip,
            episode_steps=args.episode_steps,
            absolute_action_limit=absolute_action_limit,
        )
        env = HistoryContextTrackingWrapper(
            base,
            history_steps=policy.history_steps,
            state_mean=state_stats["mean"],
            state_std=state_stats["std"],
            tip_indices=TIP_INDICES,
        )
        observations: list[np.ndarray] = []
        requested_actions: list[np.ndarray] = []
        applied_actions: list[np.ndarray] = []
        distances: list[float] = []
        rewards: list[float] = []
        try:
            observation, info = env.reset(seed=args.seed + episode)
            distances.append(float(info["distance"]))
            terminated = truncated = False
            while not (terminated or truncated):
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                )
                inference_started = time.perf_counter()
                with torch.no_grad():
                    action, _, _ = policy.act(
                        observation_tensor,
                        deterministic=True,
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
                observation = next_observation
        finally:
            env.close()
        distance_array = np.asarray(distances, dtype=np.float32)
        applied_array = np.asarray(applied_actions, dtype=np.float32)
        action_deltas = np.diff(
            np.concatenate(
                (
                    np.zeros((1, policy.action_dim), dtype=np.float32),
                    applied_array,
                ),
                axis=0,
            ),
            axis=0,
        )
        summary = {
            "episode": episode,
            "steps": len(rewards),
            "return": float(np.sum(rewards)),
            "initial_distance": float(distance_array[0]),
            "final_distance": float(distance_array[-1]),
            "minimum_distance": float(distance_array.min()),
            "success": bool(terminated),
            "action_delta_abs_mean": float(np.abs(action_deltas).mean()),
            "action_delta_abs_max": float(np.abs(action_deltas).max()),
        }
        episode_summaries.append(summary)
        trajectories.append(
            {
                "observation": np.asarray(observations, dtype=np.float32),
                "requested_action": np.asarray(requested_actions, dtype=np.float32),
                "applied_action": applied_array,
                "distance": distance_array,
                "reward": np.asarray(rewards, dtype=np.float32),
            }
        )
        print(json.dumps(summary, sort_keys=True), flush=True)

    for episode, trajectory in enumerate(trajectories):
        np.savez_compressed(output / f"trajectory_{episode:04d}.npz", **trajectory)
    inference_array = np.asarray(inference_times, dtype=np.float64)
    report = {
        "method": policy_payload["method"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "scenario": str(scenario),
        "reference": str(reference),
        "reference_sha256": sha256(reference),
        "episodes": args.episodes,
        "success_rate": float(np.mean([row["success"] for row in episode_summaries])),
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
