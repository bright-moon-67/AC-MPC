#!/usr/bin/env python
"""Evaluate a kinematic-push policy and save a renderable trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_kinematic_push_env import ManiSoftKinematicPushEnv
from antmaze_ac.rl.serialization import (
    load_history_mpc_checkpoint,
    make_history_mpc_policy,
)


TIP_INDICES = (30, 31, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--koopman-checkpoint", default=None)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--solver-iterations", type=int, default=20)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.policy_checkpoint is None and args.koopman_checkpoint is None:
        parser.error("provide --policy-checkpoint or --koopman-checkpoint")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    scenario = Path(args.scenario).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if args.policy_checkpoint is not None:
        policy, policy_payload, koopman_payload = load_history_mpc_checkpoint(
            Path(args.policy_checkpoint).expanduser().resolve(), device
        )
        if policy.task_mode != "kinematic_push":
            raise ValueError("Policy checkpoint is not for kinematic push")
    else:
        policy, koopman_payload = make_history_mpc_policy(
            Path(args.koopman_checkpoint).expanduser().resolve(),
            device,
            horizon=args.horizon,
            solver_iterations=args.solver_iterations,
            absolute_action_limit=args.absolute_action_limit,
            task_mode="kinematic_push",
        )
        policy_payload = None
    policy.eval()
    stats = koopman_payload["normalizers"]["state"]
    base_env = ManiSoftKinematicPushEnv(
        scenario,
        episode_steps=args.episode_steps,
        absolute_action_limit=args.absolute_action_limit,
    )
    env = HistoryContextTrackingWrapper(
        base_env,
        history_steps=policy.history_steps,
        state_mean=stats["mean"],
        state_std=stats["std"],
        tip_indices=TIP_INDICES,
    )
    records: dict[str, list] = {
        name: []
        for name in (
            "softrobot_positions",
            "softrobot_directors",
            "target_center",
            "goal_center",
            "active_tip_target",
            "phase",
            "contact_locked",
            "episode_id",
            "step_index",
            "action",
            "reward",
        )
    }
    summaries = []

    def append_frame(episode: int, step: int, action: np.ndarray, reward: float) -> None:
        frame = base_env.trajectory_frame()
        for key in (
            "softrobot_positions",
            "softrobot_directors",
            "target_center",
            "goal_center",
            "active_tip_target",
            "phase",
            "contact_locked",
        ):
            records[key].append(frame[key])
        records["episode_id"].append(episode)
        records["step_index"].append(step)
        records["action"].append(action.copy())
        records["reward"].append(reward)

    try:
        for episode in range(args.episodes):
            side = -1 if episode % 2 == 0 else 1
            observation, _ = env.reset(
                seed=args.seed + episode, options={"route_side": side}
            )
            append_frame(episode, -1, np.zeros(18, dtype=np.float32), 0.0)
            episode_return = 0.0
            info = {}
            for step in range(args.episode_steps):
                with torch.no_grad():
                    action, _, _ = policy.act(
                        torch.as_tensor(observation, device=device),
                        deterministic=True,
                    )
                action_array = action.detach().cpu().numpy().astype(np.float32)
                observation, reward, terminated, truncated, info = env.step(
                    action_array
                )
                append_frame(episode, step, action_array, float(reward))
                episode_return += float(reward)
                if terminated or truncated:
                    break
            summary = {
                "episode": episode,
                "steps": step + 1,
                "return": episode_return,
                "route_side": side,
                "phase": int(base_env.task.phase),
                "contact_locked": bool(base_env.task.contact_locked),
                "success": bool(info.get("is_success", False)),
                "collision": bool(info.get("collision", False)),
                "goal_distance": float(info.get("goal_distance", np.nan)),
            }
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{
                name: np.asarray(values)
                for name, values in records.items()
            },
            control_hz=np.asarray(50.0, dtype=np.float32),
        )
    temporary.replace(output)
    report = {
        "kind": "manisoft_kinematic_push_trajectory",
        "trajectory": str(output),
        "frames": len(records["phase"]),
        "episodes": summaries,
        "success_rate": float(np.mean([row["success"] for row in summaries])),
        "policy_checkpoint": args.policy_checkpoint,
        "policy_method": None if policy_payload is None else policy_payload["method"],
        "scenario": str(scenario),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

