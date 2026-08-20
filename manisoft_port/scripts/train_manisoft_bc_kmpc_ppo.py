#!/usr/bin/env python
"""PPO fine-tune the history-context soft-robot BC-KMPC actor and critic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
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
from antmaze_ac.rl.ppo import collect_rollout, collect_vector_rollout, ppo_update
from antmaze_ac.rl.serialization import make_history_mpc_policy


TIP_INDICES = (30, 31, 32)


def _device(specification: str) -> torch.device:
    return torch.device(
        "cuda"
        if specification == "auto" and torch.cuda.is_available()
        else ("cpu" if specification == "auto" else specification)
    )


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _make_env(
    *,
    scenario: Path,
    waypoint_tips: np.ndarray,
    episode_steps: int,
    absolute_action_limit: float,
    history_steps: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> HistoryContextTrackingWrapper:
    base = ManiSoftThreeWaypointTrackingEnv(
        scenario,
        waypoint_tips=waypoint_tips,
        episode_steps=episode_steps,
        absolute_action_limit=absolute_action_limit,
    )
    return HistoryContextTrackingWrapper(
        base,
        history_steps=history_steps,
        state_mean=state_mean,
        state_std=state_std,
        tip_indices=TIP_INDICES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--bc-checkpoint", default=None)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--waypoint-root",
        required=True,
        help="Directory containing the certified waypoint-bank manifest.json.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=None,
        help="Override the BC checkpoint solver depth; otherwise inherit it.",
    )
    parser.add_argument("--quadratic-log-scale", type=float, default=None)
    parser.add_argument("--linear-scale", type=float, default=None)
    parser.add_argument("--action-quadratic-scale", type=float, default=None)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=None,
        help="Override the policy exploration log_std (default uses the "
        "BC-KMPC value, typically -3 for physical-action std ~= 0.05).",
    )
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=None)
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.resume is not None and args.bc_checkpoint is not None:
        parser.error("--resume and --bc-checkpoint are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    koopman_path = Path(args.koopman_checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    waypoint_root = Path(args.waypoint_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for name, path in (
        ("Koopman checkpoint", koopman_path),
        ("scenario", scenario),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    waypoint_bank = load_manisoft_waypoint_reference_bank(waypoint_root)
    if waypoint_bank.scenario_sha256 != sha256(scenario):
        raise ValueError("Waypoint bank was certified with another scenario")
    waypoint_tips = waypoint_bank.states[:, :, np.asarray(TIP_INDICES)]

    initialization_payload = None
    initialization_path = args.bc_checkpoint or args.resume
    if initialization_path is not None:
        initialization_payload = torch.load(
            Path(initialization_path).expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
    initialization_runtime = (
        initialization_payload.get("runtime", {})
        if initialization_payload is not None
        else {}
    )
    solver_iterations = (
        args.solver_iterations
        if args.solver_iterations is not None
        else initialization_runtime.get("solver_iterations")
    )
    quadratic_log_scale = (
        args.quadratic_log_scale
        if args.quadratic_log_scale is not None
        else initialization_runtime.get("quadratic_log_scale")
    )
    linear_scale = (
        args.linear_scale
        if args.linear_scale is not None
        else initialization_runtime.get("linear_scale")
    )
    action_quadratic_scale = (
        args.action_quadratic_scale
        if args.action_quadratic_scale is not None
        else initialization_runtime.get("action_quadratic_scale")
    )

    policy, koopman_payload = make_history_mpc_policy(
        koopman_path,
        device,
        horizon=args.horizon,
        absolute_action_limit=args.absolute_action_limit,
        solver_iterations=solver_iterations,
        quadratic_log_scale=quadratic_log_scale,
        linear_scale=linear_scale,
        action_quadratic_scale=action_quadratic_scale,
        waypoint_count=3,
    )
    config = koopman_payload["config"]
    ppo = config["ppo"]
    total_timesteps = int(args.total_timesteps or ppo["total_timesteps"])
    rollout_steps = int(args.rollout_steps or ppo["rollout_steps"])
    num_envs = int(args.num_envs)
    minibatch_size = int(args.minibatch_size or ppo["minibatch_size"])
    update_epochs = int(args.update_epochs or ppo["update_epochs"])
    learning_rate = float(args.learning_rate or ppo["learning_rate"])
    checkpoint_interval = int(
        args.checkpoint_interval_updates or ppo["checkpoint_interval_updates"]
    )
    if min(total_timesteps, rollout_steps, num_envs, minibatch_size, update_epochs) < 1:
        raise ValueError("PPO counts must be positive")
    if rollout_steps % num_envs or total_timesteps % num_envs:
        raise ValueError("rollout-steps and total-timesteps must divide by num-envs")
    if minibatch_size < 2 or minibatch_size > rollout_steps:
        raise ValueError("minibatch-size must lie in [2, rollout-steps]")
    if args.target_kl <= 0:
        raise ValueError("--target-kl must be positive")

    expected_koopman_sha = sha256(koopman_path)
    bc_initialization = None
    if args.bc_checkpoint is not None:
        bc_path = Path(args.bc_checkpoint).expanduser().resolve()
        bc_payload = initialization_payload
        if bc_payload is None:
            raise RuntimeError("BC initialization payload was not loaded")
        if bc_payload.get("method") != "bc_kmpc_bc":
            raise ValueError("--bc-checkpoint is not a BC-KMPC BC checkpoint")
        if int(bc_payload.get("format_version", 0)) < 5:
            raise ValueError(
                "BC checkpoint predates randomized waypoint-bank BC-KMPC and "
                "must be retrained"
            )
        if bc_payload["koopman_checkpoint_sha256"] != expected_koopman_sha:
            raise ValueError("BC checkpoint references another Koopman model")
        for key, expected in (
            ("horizon", policy.actor.horizon),
            ("solver_iterations", policy.actor.solver_iterations),
            ("absolute_action_limit", args.absolute_action_limit),
            ("waypoint_count", 3),
            ("solver", "absolute_box_fista_v1"),
            ("fixed_smoothness", False),
            ("quadratic_log_scale", policy.actor.quadratic_log_scale),
            ("linear_scale", policy.actor.linear_scale),
            ("action_quadratic_scale", policy.actor.action_quadratic_scale),
        ):
            actual = bc_payload["runtime"].get(key)
            if actual != expected:
                raise ValueError(
                    f"BC checkpoint runtime {key}={actual!r}, expected {expected!r}"
                )
        policy.actor.load_state_dict(bc_payload["actor"])
        if (
            bc_payload.get("waypoint_bank_sha256")
            != waypoint_bank.manifest_sha256
        ):
            raise ValueError("BC checkpoint references another waypoint set")
        bc_initialization = {
            "checkpoint": str(bc_path),
            "checkpoint_sha256": sha256(bc_path),
            "best_validation_mse": float(bc_payload["best_validation_mse"]),
        }

    if args.log_std_init is not None:
        with torch.no_grad():
            policy.log_std.fill_(float(args.log_std_init))
        print(
            f"Overriding policy log_std to {args.log_std_init} "
            f"(sigma={float(np.exp(args.log_std_init)):.4f})",
            flush=True,
        )

    actor_parameters = list(policy.actor.parameters())
    auxiliary_parameters = [*policy.critic.parameters(), policy.log_std]
    optimizer = torch.optim.Adam(
        [
            {"params": actor_parameters, "lr": args.actor_learning_rate},
            {"params": auxiliary_parameters, "lr": learning_rate},
        ],
        eps=1e-5,
    )
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    status_path = output / "training_status.json"
    if args.resume is None and history_path.exists():
        raise FileExistsError(
            f"{history_path} already exists; use a new output or --resume"
        )

    runtime = {
        "horizon": policy.actor.horizon,
        "solver_iterations": policy.actor.solver_iterations,
        "absolute_action_limit": args.absolute_action_limit,
        "success_threshold": MANISOFT_WAYPOINT_SUCCESS_THRESHOLD,
        "required_success_streak": MANISOFT_WAYPOINT_SUCCESS_STREAK,
        "observation_dim": policy.observation_dim,
        "history_steps": policy.history_steps,
        "waypoint_count": policy.waypoint_count,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "minibatch_size": minibatch_size,
        "update_epochs": update_epochs,
        "action_distribution": policy.ACTION_DISTRIBUTION,
        "actor_learning_rate": args.actor_learning_rate,
        "auxiliary_learning_rate": learning_rate,
        "target_kl": args.target_kl,
        "solver": "absolute_box_fista_v1",
        "fixed_smoothness": False,
        "quadratic_log_scale": policy.actor.quadratic_log_scale,
        "linear_scale": policy.actor.linear_scale,
        "action_quadratic_scale": policy.actor.action_quadratic_scale,
    }
    metadata = {
        "method": "actor_critic_bc_kmpc",
        "format_version": 5,
        "koopman_checkpoint": str(koopman_path),
        "koopman_checkpoint_sha256": expected_koopman_sha,
        "waypoint_root": str(waypoint_root),
        "waypoint_bank_manifest": str(waypoint_bank.manifest_path),
        "waypoint_bank_sha256": waypoint_bank.manifest_sha256,
        "waypoint_triplet_count": waypoint_bank.triplet_count,
        "references": [str(path) for path in waypoint_bank.reference_paths],
        "scenario": str(scenario),
        "seed": args.seed,
        "runtime": runtime,
        "bc_initialization": bc_initialization,
        "config": config,
    }
    timesteps = 0
    update = 0
    elapsed_before = 0.0
    best_completed_return = -float("inf")
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser().resolve()
        resume_payload = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        if resume_payload.get("method") != "actor_critic_bc_kmpc":
            raise ValueError("Resume checkpoint is not actor-critic BC-KMPC")
        if int(resume_payload.get("format_version", 0)) < 5:
            raise ValueError(
                "Resume checkpoint predates randomized waypoint-bank BC-KMPC "
                "and is incompatible"
            )
        if resume_payload["koopman_checkpoint_sha256"] != expected_koopman_sha:
            raise ValueError("Resume checkpoint references another Koopman model")
        if (
            resume_payload.get("waypoint_bank_sha256")
            != waypoint_bank.manifest_sha256
        ):
            raise ValueError("Resume checkpoint references another waypoint set")
        if resume_payload["runtime"] != runtime:
            raise ValueError("Resume runtime configuration is incompatible")
        if int(resume_payload["seed"]) != args.seed:
            raise ValueError("Resume seed does not match --seed")
        policy.load_state_dict(resume_payload["policy"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        timesteps = int(resume_payload["timesteps"])
        update = int(resume_payload["update"])
        elapsed_before = float(resume_payload.get("elapsed_seconds", 0.0))
        best_completed_return = float(
            resume_payload.get("best_completed_episode_return_mean", -float("inf"))
        )
        bc_initialization = resume_payload.get("bc_initialization")
        metadata["bc_initialization"] = bc_initialization

    _write_json(
        output / "run_config.json",
        {
            **metadata,
            "arguments": vars(args),
            "device": str(device),
        },
    )

    state_stats = koopman_payload["normalizers"]["state"]
    envs = [
        _make_env(
            scenario=scenario,
            waypoint_tips=waypoint_tips,
            episode_steps=args.episode_steps,
            absolute_action_limit=args.absolute_action_limit,
            history_steps=policy.history_steps,
            state_mean=np.asarray(state_stats["mean"], dtype=np.float32),
            state_std=np.asarray(state_stats["std"], dtype=np.float32),
        )
        for _ in range(num_envs)
    ]
    for environment_index, env in enumerate(envs):
        observation, _ = env.reset(seed=args.seed + environment_index)
        env._ppo_observation = observation
        env.action_space.seed(args.seed + environment_index)

    def checkpoint_payload(elapsed_seconds: float) -> dict:
        return {
            **metadata,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "timesteps": timesteps,
            "update": update,
            "elapsed_seconds": elapsed_seconds,
            "best_completed_episode_return_mean": best_completed_return,
        }

    started = time.monotonic()
    wall_time_reached = False
    try:
        while timesteps < total_timesteps:
            current_steps = min(rollout_steps, total_timesteps - timesteps)
            if current_steps % num_envs:
                raise ValueError("Final rollout is not divisible by num-envs")
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
            update_started = time.perf_counter()
            metrics = ppo_update(
                policy,
                optimizer,
                rollout,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
                clip_range=ppo["clip_range"],
                value_coefficient=ppo["value_coefficient"],
                entropy_coefficient=ppo["entropy_coefficient"],
                max_grad_norm=ppo["max_grad_norm"],
                target_kl=args.target_kl,
            )
            synchronize()
            update_seconds = time.perf_counter() - update_started
            timesteps += current_steps
            update += 1
            elapsed = elapsed_before + time.monotonic() - started
            with torch.no_grad():
                diagnostic = policy(
                    rollout.observations[: min(16, current_steps)]
                )
            completed_mean = (
                float(rollout.episode_returns.mean())
                if len(rollout.episode_returns)
                else None
            )
            finite_distances = rollout.distances[
                np.isfinite(rollout.distances)
            ]
            row = {
                "method": "actor_critic_bc_kmpc",
                "seed": args.seed,
                "update": update,
                "timesteps": timesteps,
                "elapsed_seconds": elapsed,
                "rollout_seconds": rollout_seconds,
                "ppo_update_seconds": update_seconds,
                "iteration_steps_per_second": current_steps
                / (rollout_seconds + update_seconds),
                **metrics,
                "reward_mean": float(rollout.rewards.mean()),
                "completed_episodes": len(rollout.episode_returns),
                "completed_episode_return_mean": completed_mean,
                "episode_length_mean": (
                    float(rollout.episode_lengths.mean())
                    if len(rollout.episode_lengths)
                    else None
                ),
                "distance_mean": (
                    float(finite_distances.mean())
                    if len(finite_distances)
                    else None
                ),
                "distance_minimum": (
                    float(finite_distances.min())
                    if len(finite_distances)
                    else None
                ),
                "completed_successes": int(rollout.episode_successes.sum()),
                "completed_success_rate": (
                    float(rollout.episode_successes.mean())
                    if len(rollout.episode_successes)
                    else None
                ),
                "waypoints_completed_mean": (
                    float(rollout.episode_waypoints_completed.mean())
                    if len(rollout.episode_waypoints_completed)
                    else None
                ),
                "action_saturation_rate": float(rollout.saturation.mean()),
                "quadratic_weight_mean": float(
                    diagnostic.mpc.quadratic_diagonal.mean()
                ),
                "linear_weight_abs_mean": float(
                    diagnostic.mpc.linear_term.abs().mean()
                ),
                "projected_gradient_residual_mean": float(
                    diagnostic.mpc.projected_gradient_residual.mean()
                ),
                "log_std": policy.log_std.detach().cpu().tolist(),
            }
            if completed_mean is not None and completed_mean > best_completed_return:
                best_completed_return = completed_mean
                is_best = True
            else:
                is_best = False
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            current_payload = checkpoint_payload(elapsed)
            _save(output / "last.pt", current_payload)
            if is_best:
                _save(output / "best_completed_return.pt", current_payload)
            if update % checkpoint_interval == 0:
                _save(
                    output / f"recovery_update_{update:06d}.pt",
                    current_payload,
                )
            _write_json(
                status_path,
                {"state": "running", **row},
            )
            print(json.dumps(row, sort_keys=True), flush=True)
            if (
                args.max_wall_time_hours is not None
                and elapsed >= args.max_wall_time_hours * 3600.0
            ):
                wall_time_reached = True
                break
        elapsed = elapsed_before + time.monotonic() - started
        _write_json(
            status_path,
            {
                "state": "wall_time_reached" if wall_time_reached else "complete",
                "method": "actor_critic_bc_kmpc",
                "timesteps": timesteps,
                "updates": update,
                "elapsed_seconds": elapsed,
                "best_completed_episode_return_mean": best_completed_return,
                "last_checkpoint_sha256": sha256(output / "last.pt"),
            },
        )
    except BaseException as error:
        elapsed = elapsed_before + time.monotonic() - started
        emergency = checkpoint_payload(elapsed)
        emergency["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _save(output / "emergency.pt", emergency)
        _write_json(
            status_path,
            {
                "state": "failed",
                "method": "actor_critic_bc_kmpc",
                "timesteps": timesteps,
                "updates": update,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
