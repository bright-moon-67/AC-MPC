#!/usr/bin/env python
"""Collect an offline-RL dataset from a trained ManiSoft PPO-KMPC policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from antmaze_ac.data.offline_episodes import (
    atomic_savez_compressed,
    merge_episode_files,
    validate_episode_arrays,
)
from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_tracking_env import (
    MANISOFT_WAYPOINT_SUCCESS_STREAK,
    MANISOFT_WAYPOINT_SUCCESS_THRESHOLD,
    ManiSoftThreeWaypointTrackingEnv,
    load_manisoft_waypoint_reference_bank,
)
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.manisoft_ppo_policies import load_manisoft_ppo_checkpoint


TIP_INDICES = (30, 31, 32)
DATASET_SCHEMA_VERSION = 1
DATASET_KIND = "manisoft_ppo_kmpc_offline_transitions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--scenario",
        default=None,
        help="Defaults to the scenario recorded by the checkpoint.",
    )
    parser.add_argument(
        "--waypoint-root",
        default=None,
        help="Defaults to the certified waypoint bank in the checkpoint.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the KMPC mean. The default samples its trained distribution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a compatible interrupted collection in --output.",
    )
    parser.add_argument(
        "--no-merged-dataset",
        action="store_true",
        help="Keep episode shards only; useful for datasets too large for one NPZ.",
    )
    parser.add_argument(
        "--allow-other-waypoint-bank",
        action="store_true",
        help="Allow collection on a bank different from checkpoint training.",
    )
    return parser.parse_args()


def _resolve_recorded_path(
    argument: str | None,
    payload: dict[str, Any],
    key: str,
) -> Path:
    value = argument or payload.get(key)
    if value is None:
        raise ValueError(f"--{key.replace('_', '-')} is required")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _waypoint_schedule(
    triplet_count: int,
    episode_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = episode_count
    while remaining:
        permutation = rng.permutation(triplet_count)
        chunks.append(permutation[:remaining])
        remaining -= min(remaining, triplet_count)
    return np.concatenate(chunks).astype(np.int64, copy=False)


def _existing_episode_paths(episode_root: Path) -> list[Path]:
    paths = sorted(episode_root.glob("episode_*.npz"))
    expected = [episode_root / f"episode_{index:06d}.npz" for index in range(len(paths))]
    if paths != expected:
        raise ValueError("Existing episode shards are not a contiguous prefix")
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            validate_episode_arrays(
                {key: np.asarray(archive[key]) for key in archive.files}
            )
    return paths


def _episode_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as episode:
        rewards = np.asarray(episode["rewards"])
        terminal = bool(np.asarray(episode["terminals"])[-1])
        completed = np.asarray(episode["waypoints_completed"])
        distances = np.asarray(episode["next_active_distances"])
        return {
            "episode": int(np.asarray(episode["episode_ids"])[0]),
            "waypoint_triplet_index": int(
                np.asarray(episode["waypoint_triplet_indices"])[0]
            ),
            "steps": int(len(rewards)),
            "return": float(rewards.sum()),
            "success": terminal,
            "waypoints_completed": int(completed.max(initial=0)),
            "final_distance": float(distances[-1]),
            "minimum_distance": float(distances.min()),
        }


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.episode_steps < 1:
        raise ValueError("episodes and episode-steps must be positive")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    policy, payload, koopman_payload = load_manisoft_ppo_checkpoint(
        checkpoint, device
    )
    if payload.get("actor_name") != "ppo_kmpc":
        raise ValueError("Offline KMPC collection requires a ppo_kmpc checkpoint")
    policy.eval()
    runtime = payload["runtime"]
    max_delta_value = runtime.get("max_delta")
    if max_delta_value is None:
        raise ValueError("Checkpoint does not define normalized-delta actions")
    max_delta = float(max_delta_value)

    scenario = _resolve_recorded_path(args.scenario, payload, "scenario")
    waypoint_root = _resolve_recorded_path(
        args.waypoint_root, payload, "waypoint_root"
    )
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
    state_stats = koopman_payload["normalizers"]["state"]

    output = Path(args.output).expanduser().resolve()
    episode_root = output / "episodes"
    config_path = output / "collection_config.json"
    output.mkdir(parents=True, exist_ok=True)
    collection_config = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "kind": DATASET_KIND,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "scenario": str(scenario),
        "scenario_sha256": sha256(scenario),
        "waypoint_root": str(waypoint_root),
        "waypoint_bank_sha256": waypoint_bank.manifest_sha256,
        "episode_steps": args.episode_steps,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "observation_semantics": "history_context_tracking_wrapper_v1",
        "action_semantics": runtime["policy_action_semantics"],
        "applied_action_semantics": "absolute_muscle_activation",
        "max_delta": max_delta,
        "runtime": runtime,
    }
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(
                f"Collection already exists at {output}; pass --resume to continue"
            )
        if existing_config != collection_config:
            raise ValueError("Resume settings do not match collection_config.json")
    else:
        existing_entries = list(output.iterdir())
        if existing_entries:
            raise FileExistsError(
                f"Refusing to use non-empty output without collection metadata: {output}"
            )
        _atomic_write_json(config_path, collection_config)
    episode_root.mkdir(parents=True, exist_ok=True)

    existing_paths = _existing_episode_paths(episode_root)
    if len(existing_paths) > args.episodes:
        raise ValueError(
            f"Output already has {len(existing_paths)} episodes, more than requested"
        )
    schedule = _waypoint_schedule(
        waypoint_bank.triplet_count, args.episodes, args.seed
    )
    started = time.perf_counter()

    for episode_index in range(len(existing_paths), args.episodes):
        episode_seed = args.seed + episode_index
        torch.manual_seed(episode_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(episode_seed)
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
            max_delta=max_delta,
        )
        rows: dict[str, list[Any]] = {
            key: []
            for key in (
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "terminals",
                "timeouts",
                "episode_ids",
                "step_indices",
                "waypoint_triplet_indices",
                "requested_absolute_actions",
                "applied_actions",
                "applied_delta_actions",
                "behavior_action_means",
                "behavior_log_probs",
                "behavior_values",
                "active_waypoint_indices",
                "waypoints_completed",
                "waypoint_passed",
                "next_active_distances",
                "next_all_waypoint_distances",
                "projected_gradient_residuals",
            )
        }
        try:
            observation, _ = env.reset(
                seed=episode_seed,
                options={
                    "waypoint_triplet_index": int(schedule[episode_index])
                },
            )
            terminated = truncated = False
            step_index = 0
            while not (terminated or truncated):
                tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    action, log_prob, value, policy_output = policy.act(
                        tensor,
                        deterministic=args.deterministic,
                        return_output=True,
                    )
                policy_action = action.detach().cpu().numpy().astype(
                    np.float32, copy=False
                )
                following, reward, terminated, truncated, info = env.step(
                    policy_action
                )
                rows["observations"].append(observation.copy())
                rows["actions"].append(policy_action.copy())
                rows["rewards"].append(float(reward))
                rows["next_observations"].append(following.copy())
                rows["terminals"].append(bool(terminated))
                rows["timeouts"].append(bool(truncated))
                rows["episode_ids"].append(episode_index)
                rows["step_indices"].append(step_index)
                rows["waypoint_triplet_indices"].append(
                    int(info["waypoint_triplet_index"])
                )
                rows["requested_absolute_actions"].append(
                    np.asarray(info["requested_absolute_action"], dtype=np.float32)
                )
                rows["applied_actions"].append(
                    np.asarray(info["applied_action"], dtype=np.float32)
                )
                rows["applied_delta_actions"].append(
                    np.asarray(info["applied_delta_action"], dtype=np.float32)
                )
                rows["behavior_action_means"].append(
                    policy_output.mean.detach().cpu().numpy()
                )
                rows["behavior_log_probs"].append(float(log_prob.item()))
                rows["behavior_values"].append(float(value.item()))
                rows["active_waypoint_indices"].append(
                    int(info["active_waypoint_index"])
                )
                rows["waypoints_completed"].append(
                    int(info["waypoints_completed"])
                )
                rows["waypoint_passed"].append(bool(info["waypoint_passed"]))
                rows["next_active_distances"].append(float(info["distance"]))
                rows["next_all_waypoint_distances"].append(
                    np.asarray(info["all_waypoint_distances"], dtype=np.float32)
                )
                rows["projected_gradient_residuals"].append(
                    float(policy_output.mpc.projected_gradient_residual.mean())
                )
                observation = following
                step_index += 1
        finally:
            env.close()

        episode_arrays = {
            "observations": np.asarray(rows["observations"], dtype=np.float32),
            "actions": np.asarray(rows["actions"], dtype=np.float32),
            "rewards": np.asarray(rows["rewards"], dtype=np.float32),
            "next_observations": np.asarray(
                rows["next_observations"], dtype=np.float32
            ),
            "terminals": np.asarray(rows["terminals"], dtype=np.bool_),
            "timeouts": np.asarray(rows["timeouts"], dtype=np.bool_),
            "episode_ids": np.asarray(rows["episode_ids"], dtype=np.int64),
            "step_indices": np.asarray(rows["step_indices"], dtype=np.int64),
            "waypoint_triplet_indices": np.asarray(
                rows["waypoint_triplet_indices"], dtype=np.int64
            ),
            "requested_absolute_actions": np.asarray(
                rows["requested_absolute_actions"], dtype=np.float32
            ),
            "applied_actions": np.asarray(
                rows["applied_actions"], dtype=np.float32
            ),
            "applied_delta_actions": np.asarray(
                rows["applied_delta_actions"], dtype=np.float32
            ),
            "behavior_action_means": np.asarray(
                rows["behavior_action_means"], dtype=np.float32
            ),
            "behavior_log_probs": np.asarray(
                rows["behavior_log_probs"], dtype=np.float32
            ),
            "behavior_values": np.asarray(
                rows["behavior_values"], dtype=np.float32
            ),
            "active_waypoint_indices": np.asarray(
                rows["active_waypoint_indices"], dtype=np.int64
            ),
            "waypoints_completed": np.asarray(
                rows["waypoints_completed"], dtype=np.int64
            ),
            "waypoint_passed": np.asarray(
                rows["waypoint_passed"], dtype=np.bool_
            ),
            "next_active_distances": np.asarray(
                rows["next_active_distances"], dtype=np.float32
            ),
            "next_all_waypoint_distances": np.asarray(
                rows["next_all_waypoint_distances"], dtype=np.float32
            ),
            "projected_gradient_residuals": np.asarray(
                rows["projected_gradient_residuals"], dtype=np.float32
            ),
        }
        validate_episode_arrays(episode_arrays)
        episode_path = episode_root / f"episode_{episode_index:06d}.npz"
        atomic_savez_compressed(episode_path, episode_arrays)
        print(json.dumps(_episode_summary(episode_path), sort_keys=True), flush=True)

    episode_paths = _existing_episode_paths(episode_root)
    summaries = [_episode_summary(path) for path in episode_paths]
    if args.no_merged_dataset:
        transition_count = int(sum(row["steps"] for row in summaries))
        merged_path = None
    else:
        merged_path = output / "dataset.npz"
        transition_count = merge_episode_files(episode_paths, merged_path)
    report = {
        **collection_config,
        "requested_episodes": args.episodes,
        "collected_episodes": len(summaries),
        "transitions": transition_count,
        "success_rate": float(np.mean([row["success"] for row in summaries])),
        "return_mean": float(np.mean([row["return"] for row in summaries])),
        "episode_steps_mean": float(np.mean([row["steps"] for row in summaries])),
        "waypoints_completed_mean": float(
            np.mean([row["waypoints_completed"] for row in summaries])
        ),
        "merged_dataset": None if merged_path is None else str(merged_path),
        "elapsed_seconds_this_run": float(time.perf_counter() - started),
        "episode_summaries": summaries,
    }
    _atomic_write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
