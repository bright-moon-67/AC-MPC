#!/usr/bin/env python
"""PPO fine-tuning for the force-free ManiSoft kinematic push task."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_kinematic_push_env import ManiSoftKinematicPushEnv
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.ppo import collect_rollout, ppo_update
from antmaze_ac.rl.serialization import (
    load_history_mpc_checkpoint,
    make_history_mpc_policy,
)


TIP_INDICES = (30, 31, 32)


def _device(specification: str) -> torch.device:
    if specification == "auto":
        specification = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(specification)


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--initial-checkpoint",
        default=None,
        help="Optional kinematic-push BC or PPO checkpoint.",
    )
    parser.add_argument("--expert-dataset", default=None)
    parser.add_argument("--bc-coefficient", type=float, default=0.1)
    parser.add_argument("--bc-updates-per-rollout", type=int, default=1)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--solver-iterations", type=int, default=20)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--log-std-init", type=float, default=-3.0)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = (
        args.total_timesteps,
        args.rollout_steps,
        args.episode_steps,
        args.update_epochs,
        args.minibatch_size,
        args.horizon,
        args.solver_iterations,
        args.checkpoint_interval,
    )
    if min(counts) < 1:
        parser.error("PPO counts must be positive")
    if args.minibatch_size > args.rollout_steps:
        parser.error("minibatch-size cannot exceed rollout-steps")
    if args.bc_coefficient < 0 or args.bc_updates_per_rollout < 0:
        parser.error("BC settings must be non-negative")
    return args


def _load_expert(
    path: Path | None,
    observation_dim: int,
    checkpoint_sha: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Missing expert dataset: {path}")
    report_path = path.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("koopman_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Expert dataset references another Koopman checkpoint")
    with np.load(path, allow_pickle=False) as archive:
        observations = np.asarray(archive["observation"], dtype=np.float32)
        actions = np.asarray(archive["expert_action"], dtype=np.float32)
    if observations.shape[1:] != (observation_dim,) or actions.shape[1:] != (18,):
        raise ValueError("Expert dataset shape is incompatible with this policy")
    return torch.from_numpy(observations), torch.from_numpy(actions)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    koopman_path = Path(args.koopman_checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not koopman_path.is_file() or not scenario.is_file():
        raise FileNotFoundError("Koopman checkpoint or scenario is missing")
    koopman_sha = sha256(koopman_path)

    initialization = None
    if args.initial_checkpoint is not None:
        initial_path = Path(args.initial_checkpoint).expanduser().resolve()
        policy, initialization, koopman_payload = load_history_mpc_checkpoint(
            initial_path, device
        )
        if initialization.get("koopman_checkpoint_sha256") != koopman_sha:
            raise ValueError("Initial policy references another Koopman model")
        if policy.task_mode != "kinematic_push":
            raise ValueError("Initial checkpoint is not a kinematic-push policy")
    else:
        policy, koopman_payload = make_history_mpc_policy(
            koopman_path,
            device,
            horizon=args.horizon,
            solver_iterations=args.solver_iterations,
            absolute_action_limit=args.absolute_action_limit,
            task_mode="kinematic_push",
        )
    with torch.no_grad():
        policy.log_std.fill_(args.log_std_init)
    state_stats = koopman_payload["normalizers"]["state"]
    base_env = ManiSoftKinematicPushEnv(
        scenario,
        episode_steps=args.episode_steps,
        absolute_action_limit=args.absolute_action_limit,
    )
    env = HistoryContextTrackingWrapper(
        base_env,
        history_steps=policy.history_steps,
        state_mean=state_stats["mean"],
        state_std=state_stats["std"],
        tip_indices=TIP_INDICES,
    )
    if env.observation_space.shape != (policy.observation_dim,):
        raise RuntimeError("Environment and policy observation layouts differ")
    expert_path = (
        None
        if args.expert_dataset is None
        else Path(args.expert_dataset).expanduser().resolve()
    )
    expert = _load_expert(expert_path, policy.observation_dim, koopman_sha)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        eps=1e-5,
    )
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    best_score = -float("inf")
    completed_timesteps = 0
    update_count = math.ceil(args.total_timesteps / args.rollout_steps)
    rng = np.random.default_rng(args.seed)
    runtime = {
        "task_mode": "kinematic_push",
        "waypoint_count": 1,
        "horizon": int(policy.actor.horizon),
        "solver_iterations": int(policy.actor.solver_iterations),
        "absolute_action_limit": args.absolute_action_limit,
        "quadratic_log_scale": float(policy.actor.quadratic_log_scale),
        "linear_scale": float(policy.actor.linear_scale),
        "action_quadratic_scale": float(policy.actor.action_quadratic_scale),
        "task_context_dim": int(policy.task_context_dim),
        "observation_dim": int(policy.observation_dim),
    }

    def checkpoint_payload(update: int) -> dict:
        return {
            "format_version": 6,
            "method": "actor_critic_kinematic_push",
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "koopman_checkpoint": str(koopman_path),
            "koopman_checkpoint_sha256": koopman_sha,
            "scenario": str(scenario),
            "scenario_sha256": sha256(scenario),
            "update": update,
            "timesteps": completed_timesteps,
            "runtime": runtime,
            "config": vars(args),
        }

    try:
        for update in range(1, update_count + 1):
            policy.eval()
            rollout = collect_rollout(
                env,
                policy,
                args.rollout_steps,
                args.gamma,
                args.gae_lambda,
                device,
            )
            policy.train()
            metrics = ppo_update(
                policy,
                optimizer,
                rollout,
                update_epochs=args.update_epochs,
                minibatch_size=args.minibatch_size,
                clip_range=args.clip_range,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
                minimum_log_std=-5.0,
                maximum_log_std=-1.5,
            )
            bc_loss = 0.0
            if (
                expert is not None
                and args.bc_coefficient > 0
                and args.bc_updates_per_rollout > 0
            ):
                expert_observations, expert_actions = expert
                for _ in range(args.bc_updates_per_rollout):
                    indices = rng.integers(
                        0, len(expert_observations), size=args.bc_batch_size
                    )
                    observation_batch = expert_observations[indices].to(device)
                    action_batch = expert_actions[indices].to(device)
                    prediction = policy.actor_mean(observation_batch).action
                    loss = args.bc_coefficient * (
                        prediction - action_batch
                    ).square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy.actor.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    bc_loss += float(loss.detach())
                bc_loss /= args.bc_updates_per_rollout
            completed_timesteps += args.rollout_steps
            completed = len(rollout.episode_returns)
            success_rate = (
                float(np.mean(rollout.episode_successes)) if completed else 0.0
            )
            completed_return = (
                float(np.mean(rollout.episode_returns)) if completed else float("nan")
            )
            score = success_rate * 1000.0 + (
                completed_return if np.isfinite(completed_return) else 0.0
            )
            row = {
                "update": update,
                "timesteps": completed_timesteps,
                "reward_mean": float(np.mean(rollout.rewards)),
                "completed_episodes": completed,
                "completed_return_mean": completed_return,
                "success_rate": success_rate,
                "max_phase_mean": (
                    float(np.mean(rollout.episode_waypoints_completed))
                    if completed
                    else float("nan")
                ),
                "bc_loss": bc_loss,
                **{key: float(value) for key, value in metrics.items()},
            }
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            _save(output / "last.pt", checkpoint_payload(update))
            if score > best_score:
                best_score = score
                _save(output / "best.pt", checkpoint_payload(update))
            if update % args.checkpoint_interval == 0:
                _save(
                    output / f"update_{update:06d}.pt",
                    checkpoint_payload(update),
                )
    finally:
        env.close()
    status = {
        "status": "complete",
        "timesteps": completed_timesteps,
        "updates": update_count,
        "best_score": best_score,
        "best_checkpoint": str((output / "best.pt").resolve()),
    }
    (output / "training_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()

