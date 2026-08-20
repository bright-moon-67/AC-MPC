#!/usr/bin/env python
"""Train the augmented-state, incremental-action PPO baseline."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.envs.factory import make_antmaze_env
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.delta_policy import DeltaPolicy
from antmaze_ac.rl.ppo import collect_rollout, ppo_update


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--koopman-checkpoint",
        required=True,
        help="Used only for the train-split state normalization and common resolved config",
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--output", default="runs/antmaze_umaze/delta_ppo")
    parser.add_argument("--backend", choices=["auto", "legacy", "modern"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    koopman_checkpoint = Path(args.koopman_checkpoint).resolve()
    koopman_sha = sha256(koopman_checkpoint)
    payload = torch.load(koopman_checkpoint, map_location="cpu", weights_only=False)
    stats = payload["normalizers"]["state"]
    config = payload["config"]
    state_dim = payload["architecture"]["state_dim"]
    action_dim = payload["architecture"]["action_dim"]
    baseline = config["delta_ppo_baseline"]
    policy = DeltaPolicy(
        state_dim,
        action_dim,
        torch.tensor(stats["mean"]),
        torch.tensor(stats["std"]),
        baseline["hidden_dims"],
        baseline["log_std_init"],
        baseline["activation"],
    ).to(device)
    ppo = config["ppo"]
    total_timesteps = int(args.total_timesteps or ppo["total_timesteps"])
    rollout_steps = int(args.rollout_steps or ppo["rollout_steps"])
    checkpoint_interval = int(
        args.checkpoint_interval_updates or ppo["checkpoint_interval_updates"]
    )
    seed = int(config["experiment"]["seed"] if args.seed is None else args.seed)
    set_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo["learning_rate"])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    if args.resume is None and history_path.exists():
        raise FileExistsError(
            f"{history_path} already exists; use a fresh output or pass --resume"
        )
    timesteps = 0
    update = 0
    elapsed_before = 0.0
    best_episode_return = -float("inf")
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        if resume_payload["method"] != "delta_ppo":
            raise ValueError("Resume checkpoint is not a Delta-PPO run")
        if resume_payload["koopman_checkpoint_sha256"] != koopman_sha:
            raise ValueError("Resume checkpoint references different normalization/config")
        if int(resume_payload["seed"]) != seed:
            raise ValueError("Resume seed does not match --seed")
        policy.load_state_dict(resume_payload["policy"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        timesteps = int(resume_payload["timesteps"])
        update = int(resume_payload["update"])
        elapsed_before = float(resume_payload.get("elapsed_seconds", 0.0))
        best_episode_return = float(
            resume_payload.get("best_completed_episode_return_mean", -float("inf"))
        )

    env = make_antmaze_env(config["experiment"]["env_id"], backend=args.backend)
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    env._ppo_observation = observation
    started = time.monotonic()

    def checkpoint_payload(elapsed_seconds: float) -> dict:
        return {
            "format_version": 2,
            "method": "delta_ppo",
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "koopman_checkpoint": str(koopman_checkpoint),
            "koopman_checkpoint_sha256": koopman_sha,
            "config": config,
            "timesteps": timesteps,
            "update": update,
            "seed": seed,
            "best_completed_episode_return_mean": best_episode_return,
            "elapsed_seconds": elapsed_seconds,
            "backend": args.backend,
            "device": str(device),
        }

    while timesteps < total_timesteps:
        steps = min(rollout_steps, total_timesteps - timesteps)
        rollout = collect_rollout(env, policy, steps, ppo["gamma"], ppo["gae_lambda"], device)
        metrics = ppo_update(
            policy,
            optimizer,
            rollout,
            update_epochs=ppo["update_epochs"],
            minibatch_size=ppo["minibatch_size"],
            clip_range=ppo["clip_range"],
            value_coefficient=ppo["value_coefficient"],
            entropy_coefficient=ppo["entropy_coefficient"],
            max_grad_norm=ppo["max_grad_norm"],
        )
        timesteps += steps
        update += 1
        elapsed = elapsed_before + time.monotonic() - started
        completed_mean = (
            float(rollout.episode_returns.mean()) if len(rollout.episode_returns) else None
        )
        if completed_mean is not None and completed_mean > best_episode_return:
            best_episode_return = completed_mean
            is_best = True
        else:
            is_best = False
        row = {
            "method": "delta_ppo",
            "seed": seed,
            "update": update,
            "timesteps": timesteps,
            "elapsed_seconds": elapsed,
            **metrics,
            "reward_mean": float(rollout.rewards.mean()),
            "completed_episodes": len(rollout.episode_returns),
            "completed_episode_return_mean": completed_mean,
            "success_rate": float(np.mean(rollout.episode_returns > 0))
            if len(rollout.episode_returns)
            else None,
            "episode_length_mean": float(rollout.episode_lengths.mean())
            if len(rollout.episode_lengths)
            else None,
            "action_saturation_rate": float(rollout.saturation.mean()),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        current_payload = checkpoint_payload(elapsed)
        torch.save(current_payload, output / "last.pt")
        if is_best:
            torch.save(current_payload, output / "best_completed_return.pt")
        if update % checkpoint_interval == 0:
            torch.save(current_payload, output / f"recovery_update_{update:06d}.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    elapsed = elapsed_before + time.monotonic() - started
    status = {
        "method": "delta_ppo",
        "seed": seed,
        "timesteps": timesteps,
        "updates": update,
        "elapsed_seconds": elapsed,
        "stop_reason": "total_timesteps",
        "best_completed_episode_return_mean": best_episode_return,
        "last_checkpoint_sha256": sha256(output / "last.pt"),
        "koopman_checkpoint_sha256": koopman_sha,
        "backend": args.backend,
        "device": str(device),
    }
    (output / "training_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))
    env.close()


if __name__ == "__main__":
    main()
