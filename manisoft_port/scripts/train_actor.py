#!/usr/bin/env python
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
from antmaze_ac.rl.ppo import collect_rollout, collect_vector_rollout, ppo_update
from antmaze_ac.rl.serialization import make_policy


class SolverFailureLimitExceeded(RuntimeError):
    """Raised only after repeated rollouts exceed the configured fallback limit."""


def resolve_device(specification: str) -> torch.device:
    return torch.device(
        "cuda"
        if specification == "auto" and torch.cuda.is_available()
        else ("cpu" if specification == "auto" else specification)
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--output", default="runs/antmaze_umaze/actor")
    parser.add_argument("--backend", choices=["auto", "legacy", "modern"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--actor-init",
        default=None,
        help="Initialize CostActor weights from a TD3+BC checkpoint.",
    )
    args = parser.parse_args()
    if args.resume is not None and args.actor_init is not None:
        raise ValueError("--resume and --actor-init are mutually exclusive")

    device = resolve_device(args.device)
    policy, koopman_payload = make_policy(args.koopman_checkpoint, device)
    config = koopman_payload["config"]
    if config["control"]["gain_update_interval"] != 1:
        raise ValueError("Feed-forward PPO training requires gain_update_interval=1 for policy reconstruction")
    ppo = config["ppo"]
    total_timesteps = int(args.total_timesteps or ppo["total_timesteps"])
    rollout_steps = int(args.rollout_steps or ppo["rollout_steps"])
    num_envs = int(args.num_envs or ppo.get("num_envs", 1))
    minibatch_size = int(args.minibatch_size or ppo["minibatch_size"])
    update_epochs = int(args.update_epochs or ppo["update_epochs"])
    if num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if rollout_steps % num_envs:
        raise ValueError("rollout_steps must be divisible by num_envs")
    if total_timesteps % num_envs:
        raise ValueError("total_timesteps must be divisible by num_envs")
    if minibatch_size < 2 or minibatch_size > rollout_steps:
        raise ValueError("minibatch_size must be in [2, rollout_steps]")
    if update_epochs < 1:
        raise ValueError("update_epochs must be positive")
    checkpoint_interval = int(
        args.checkpoint_interval_updates or ppo["checkpoint_interval_updates"]
    )
    seed = int(config["experiment"]["seed"] if args.seed is None else args.seed)
    set_seed(seed)

    expected_koopman_sha = sha256(args.koopman_checkpoint)
    actor_initialization = None
    if args.actor_init is not None:
        initialization_payload = torch.load(
            args.actor_init,
            map_location=device,
            weights_only=False,
        )
        if initialization_payload.get("method") != "td3_bc_koopman_lqr":
            raise ValueError("--actor-init must be a Koopman-LQR TD3+BC checkpoint")
        if (
            initialization_payload["koopman_checkpoint_sha256"]
            != expected_koopman_sha
        ):
            raise ValueError("--actor-init references a different Koopman model")
        policy.actor.load_state_dict(initialization_payload["actor"])
        actor_initialization = {
            "method": initialization_payload["method"],
            "checkpoint": str(Path(args.actor_init).resolve()),
            "checkpoint_sha256": sha256(args.actor_init),
            "gradient_step": int(initialization_payload["gradient_step"]),
        }
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=ppo["learning_rate"],
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    status_path = output / "training_status.json"
    if args.resume is None and history_path.exists():
        raise FileExistsError(
            f"{history_path} already exists; use a fresh output or pass --resume"
        )
    if args.resume is not None and status_path.exists():
        # A shorter profiling target may have completed normally before the
        # same checkpoint is extended. Its terminal status must not remain
        # visible while the resumed process is active.
        status_path.unlink()

    timesteps = 0
    update = 0
    elapsed_before = 0.0
    best_episode_return = -float("inf")
    consecutive_solver_failure_rollouts = 0
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        if resume_payload["method"] != "actor_critic_koopman_lqr":
            raise ValueError("Resume checkpoint is not an Actor-Critic Koopman-LQR run")
        if resume_payload["koopman_checkpoint_sha256"] != expected_koopman_sha:
            raise ValueError("Resume checkpoint references a different Koopman model")
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
        consecutive_solver_failure_rollouts = int(
            resume_payload.get("consecutive_solver_failure_rollouts", 0)
        )

    envs = [
        make_antmaze_env(config["experiment"]["env_id"], backend=args.backend)
        for _ in range(num_envs)
    ]
    for environment_index, env in enumerate(envs):
        environment_seed = seed + environment_index
        observation, _ = env.reset(seed=environment_seed)
        env.action_space.seed(environment_seed)
        # Avoid an unseeded second reset inside rollout collection.
        env._ppo_observation = observation
    started = time.monotonic()

    def checkpoint_payload(elapsed_seconds: float) -> dict:
        return {
            "format_version": 2,
            "method": "actor_critic_koopman_lqr",
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "koopman_checkpoint": str(Path(args.koopman_checkpoint).resolve()),
            "koopman_checkpoint_sha256": expected_koopman_sha,
            "timesteps": timesteps,
            "update": update,
            "seed": seed,
            "best_completed_episode_return_mean": best_episode_return,
            "consecutive_solver_failure_rollouts": (
                consecutive_solver_failure_rollouts
            ),
            "elapsed_seconds": elapsed_seconds,
            "config": config,
            "backend": args.backend,
            "device": str(device),
            "runtime": {
                "num_envs": num_envs,
                "rollout_steps": rollout_steps,
                "minibatch_size": minibatch_size,
                "update_epochs": update_epochs,
            },
            "actor_initialization": actor_initialization,
        }

    control = config["control"]
    maximum_fallback_fraction = float(
        control.get("dare_max_fallback_fraction_per_rollout", 0.05)
    )
    maximum_consecutive_failure_rollouts = int(
        control.get("dare_max_consecutive_failure_rollouts", 3)
    )
    if not 0.0 <= maximum_fallback_fraction <= 1.0:
        raise ValueError("dare_max_fallback_fraction_per_rollout must be in [0,1]")
    if maximum_consecutive_failure_rollouts < 1:
        raise ValueError("dare_max_consecutive_failure_rollouts must be positive")

    while timesteps < total_timesteps:
        current_steps = min(rollout_steps, total_timesteps - timesteps)
        try:
            synchronize = (
                (lambda: torch.cuda.synchronize(device))
                if device.type == "cuda"
                else (lambda: None)
            )
            synchronize()
            rollout_started = time.perf_counter()
            rollout = (
                collect_rollout(
                    envs[0],
                    policy,
                    current_steps,
                    ppo["gamma"],
                    ppo["gae_lambda"],
                    device,
                )
                if num_envs == 1
                else collect_vector_rollout(
                    envs,
                    policy,
                    current_steps,
                    ppo["gamma"],
                    ppo["gae_lambda"],
                    device,
                )
            )
            synchronize()
            rollout_seconds = time.perf_counter() - rollout_started
            fallback_fraction = float(rollout.dare_fallback.mean())
            retry_fraction = float(rollout.dare_retry.mean())
            if fallback_fraction > maximum_fallback_fraction:
                consecutive_solver_failure_rollouts += 1
            else:
                consecutive_solver_failure_rollouts = 0
            if (
                consecutive_solver_failure_rollouts
                >= maximum_consecutive_failure_rollouts
            ):
                raise SolverFailureLimitExceeded(
                    "DARE fallback fraction exceeded "
                    f"{maximum_fallback_fraction:.3f} for "
                    f"{consecutive_solver_failure_rollouts} consecutive rollouts"
                )
            update_started = time.perf_counter()
            update_metrics = ppo_update(
                policy,
                optimizer,
                rollout,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
                clip_range=ppo["clip_range"],
                value_coefficient=ppo["value_coefficient"],
                entropy_coefficient=ppo["entropy_coefficient"],
                max_grad_norm=ppo["max_grad_norm"],
            )
            synchronize()
            update_seconds = time.perf_counter() - update_started
        except (RuntimeError, FloatingPointError) as error:
            elapsed = elapsed_before + time.monotonic() - started
            emergency_payload = checkpoint_payload(elapsed)
            emergency_payload["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            emergency_path = output / "emergency.pt"
            torch.save(emergency_payload, emergency_path)
            stop_reason = (
                "solver_failure_limit"
                if isinstance(error, SolverFailureLimitExceeded)
                else "training_numerical_emergency"
                if isinstance(error, FloatingPointError)
                else "training_runtime_exception"
            )
            failure_status = {
                "method": "actor_critic_koopman_lqr",
                "seed": seed,
                "timesteps": timesteps,
                "updates": update,
                "elapsed_seconds": elapsed,
                "stop_reason": stop_reason,
                "error_type": type(error).__name__,
                "error": str(error),
                "emergency_checkpoint": str(emergency_path.resolve()),
                "emergency_checkpoint_sha256": sha256(emergency_path),
                "consecutive_solver_failure_rollouts": (
                    consecutive_solver_failure_rollouts
                ),
            }
            status_path.write_text(
                json.dumps(failure_status, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(failure_status, indent=2), flush=True)
            for env in envs:
                env.close()
            raise
        timesteps += current_steps
        update += 1
        elapsed = elapsed_before + time.monotonic() - started
        with torch.no_grad():
            diagnostic = policy(rollout.observations[: min(16, current_steps)])
        completed_mean = (
            float(rollout.episode_returns.mean()) if len(rollout.episode_returns) else None
        )
        row = {
            "method": "actor_critic_koopman_lqr",
            "seed": seed,
            "update": update,
            "timesteps": timesteps,
            "elapsed_seconds": elapsed,
            "num_envs": num_envs,
            "minibatch_size": minibatch_size,
            "update_epochs": update_epochs,
            "rollout_seconds": rollout_seconds,
            "ppo_update_seconds": update_seconds,
            "rollout_steps_per_second": current_steps / rollout_seconds,
            "ppo_samples_per_second": (
                current_steps * update_epochs / update_seconds
            ),
            "iteration_steps_per_second": (
                current_steps / (rollout_seconds + update_seconds)
            ),
            **update_metrics,
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
            "dare_retry_fraction": retry_fraction,
            "dare_fallback_fraction": fallback_fraction,
            "dare_failure_count": int(rollout.dare_fallback.sum()),
            "consecutive_solver_failure_rollouts": (
                consecutive_solver_failure_rollouts
            ),
            "dare_post_update_retry_fraction": float(
                diagnostic.solver_retry_used.float().mean()
            ),
            "dare_post_update_fallback_fraction": float(
                diagnostic.solver_fallback_used.float().mean()
            ),
            "stage_hessian_mean": float(diagnostic.stage_hessian_diag.mean()),
            "stage_linear_abs_mean": float(diagnostic.stage_linear.abs().mean()),
            "dare_residual_max": float(torch.max(diagnostic.lqr.dare.residual)),
            "dare_relative_residual_max": float(
                torch.max(diagnostic.lqr.dare.relative_residual)
            ),
            "dare_condition_max": float(torch.max(diagnostic.lqr.dare.condition_number)),
            "closed_loop_spectral_radius_max": float(
                torch.max(diagnostic.lqr.dare.closed_loop_spectral_radius)
            ),
        }
        if completed_mean is not None and completed_mean > best_episode_return:
            best_episode_return = completed_mean
            is_best = True
        else:
            is_best = False
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        current_payload = checkpoint_payload(elapsed)
        torch.save(current_payload, output / "last.pt")
        if is_best:
            torch.save(current_payload, output / "best_completed_return.pt")
        if update % checkpoint_interval == 0:
            torch.save(
                current_payload,
                output / f"recovery_update_{update:06d}.pt",
            )
        print(json.dumps(row, sort_keys=True), flush=True)

    elapsed = elapsed_before + time.monotonic() - started
    status = {
        "method": "actor_critic_koopman_lqr",
        "seed": seed,
        "timesteps": timesteps,
        "updates": update,
        "elapsed_seconds": elapsed,
        "stop_reason": "total_timesteps",
        "best_completed_episode_return_mean": best_episode_return,
        "last_checkpoint_sha256": sha256(output / "last.pt"),
        "koopman_checkpoint_sha256": expected_koopman_sha,
        "backend": args.backend,
        "device": str(device),
    }
    status_path.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
