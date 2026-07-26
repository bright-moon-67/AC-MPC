#!/usr/bin/env python
"""Offline TD3+BC for the state-conditioned Koopman-LQR cost actor."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from antmaze_ac.config import load_config
from antmaze_ac.data.windows import load_npz_dataset
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.serialization import make_policy
from antmaze_ac.rl.td3_bc import (
    TD3BCTrainer,
    TwinActionValueCritic,
    offline_validation_metrics,
    sample_transition_batch,
)


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


def finite_mean(
    values: list[torch.Tensor | float | int | None],
) -> float | None:
    selected = [value for value in values if value is not None]
    if not selected:
        return None
    tensor_value = next(
        (value for value in selected if isinstance(value, torch.Tensor)),
        None,
    )
    if tensor_value is None:
        return float(np.mean(selected))
    device = tensor_value.device
    stacked = torch.stack(
        [
            value.detach().to(device)
            if isinstance(value, torch.Tensor)
            else torch.tensor(float(value), device=device)
            for value in selected
        ]
    )
    return float(stacked.mean().cpu())


def reconcile_history(path: Path, gradient_step: int) -> int:
    """Remove rows newer than a resume checkpoint."""

    if not path.exists():
        return 0
    retained = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["gradient_step"]) <= gradient_step:
            retained.append(row)
        else:
            removed += 1
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
        encoding="utf-8",
    )
    return removed


def environment_evaluation_due(gradient_step: int, interval: int) -> bool:
    """Evaluate the near-initial policy and then at fixed gradient-step intervals."""

    return interval > 0 and (
        gradient_step == 1 or gradient_step % interval == 0
    )


def lightweight_evaluation_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields required to reconstruct and evaluate the TD3+BC policy."""

    retained = (
        "format_version",
        "method",
        "policy",
        "koopman_checkpoint",
        "koopman_checkpoint_sha256",
        "gradient_step",
        "seed",
        "elapsed_seconds",
        "best_validation_behavior_cloning_loss",
        "config",
        "runtime",
    )
    return {key: payload[key] for key in retained}


def _write_jsonl_replacing_step(path: Path, row: dict[str, Any]) -> None:
    step = int(row["gradient_step"])
    retained: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            previous = json.loads(line)
            if int(previous["gradient_step"]) < step:
                retained.append(previous)
    retained.append(row)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in retained),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_environment_evaluation_trend(history_path: Path, output: Path) -> None:
    """Update the compact wall-clock convergence plot from successful evaluations."""

    if not history_path.exists():
        return
    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("status") == "ok"]
    if not rows:
        if output.exists():
            output.unlink()
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elapsed_hours = np.asarray(
        [float(row["training_elapsed_seconds"]) / 3600.0 for row in rows]
    )
    success = np.asarray([float(row["success_mean"]) for row in rows])
    progress = np.asarray(
        [float(row["goal_progress_fraction_mean"]) for row in rows]
    )
    minimum_distance = np.asarray(
        [float(row["minimum_goal_distance_mean"]) for row in rows]
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), squeeze=False)
    panels = (
        (success, "success rate", (0.0, 1.02)),
        (progress, "goal progress fraction", None),
        (minimum_distance, "minimum goal distance", None),
    )
    for axis, (values, title, limits) in zip(axes.flat, panels):
        axis.plot(elapsed_hours, values, marker="o", linewidth=1.6)
        for x_value, y_value, row in zip(elapsed_hours, values, rows):
            axis.annotate(
                f"{int(row['gradient_step'])}",
                (x_value, y_value),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_xlabel("training process wall time (hours)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xlim(0.0, max(0.1, float(elapsed_hours.max()) * 1.05))
        if limits is not None:
            axis.set_ylim(*limits)
    figure.suptitle("Periodic legacy AntMaze evaluation (labels = gradient step)")
    figure.tight_layout()
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    figure.savefig(temporary, dpi=160, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def reconcile_environment_evaluations(
    evaluation_root: Path,
    gradient_step: int,
) -> int:
    """Remove evaluation artifacts newer than a resume checkpoint."""

    if not evaluation_root.exists():
        return 0
    step_pattern = re.compile(r"^step_(\d{8})(?:_|\.pt)")
    removed = 0
    for path in evaluation_root.iterdir():
        match = step_pattern.match(path.name)
        if match is not None and int(match.group(1)) > gradient_step:
            path.unlink()
            removed += 1

    history_path = evaluation_root / "history.jsonl"
    if history_path.exists():
        retained = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["gradient_step"]) <= gradient_step:
                retained.append(row)
        history_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
            encoding="utf-8",
        )
        trend_path = evaluation_root / "trend.png"
        if trend_path.exists() and not retained:
            trend_path.unlink()
        elif retained:
            save_environment_evaluation_trend(history_path, trend_path)
    return removed


def run_environment_evaluation(
    *,
    project_root: Path,
    output: Path,
    gradient_step: int,
    training_elapsed_seconds: float,
    checkpoint_payload: dict[str, Any],
    episodes: int,
    plot_paths: int,
    device: str,
    seed_offset: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Synchronously evaluate a frozen snapshot in the original legacy simulator."""

    evaluation_root = output / "periodic_evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    prefix = f"step_{gradient_step:08d}_legacy_{episodes}ep"
    checkpoint_path = evaluation_root / f"step_{gradient_step:08d}.pt"
    report_path = evaluation_root / f"{prefix}.json"
    plot_path = evaluation_root / f"{prefix}_paths.png"
    checkpoint_temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        lightweight_evaluation_checkpoint(checkpoint_payload),
        checkpoint_temporary,
    )
    checkpoint_temporary.replace(checkpoint_path)

    command = [
        str(project_root / "scripts/run_legacy.sh"),
        "python",
        str(project_root / "scripts/evaluate_actor.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--method",
        "td3_bc",
        "--episodes",
        str(episodes),
        "--backend",
        "legacy",
        "--device",
        device,
        "--seed-offset",
        str(seed_offset),
        "--plot-paths",
        str(plot_paths),
        "--path-plot-output",
        str(plot_path),
        "--output",
        str(report_path),
    ]
    print(
        json.dumps(
            {
                "event": "periodic_environment_evaluation_started",
                "gradient_step": gradient_step,
                "episodes": episodes,
                "checkpoint": str(checkpoint_path.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("resolved_backend") != "legacy":
            raise RuntimeError("Periodic evaluation did not use the legacy backend")
        if int(report.get("episodes", 0)) != episodes:
            raise RuntimeError("Periodic evaluation episode count mismatch")
        if report.get("checkpoint_sha256") != sha256(checkpoint_path):
            raise RuntimeError("Periodic evaluation checkpoint digest mismatch")
        required_finite = (
            "success_mean",
            "return_mean",
            "d4rl_normalized_score",
            "goal_progress_fraction_mean",
            "minimum_goal_distance_mean",
            "final_goal_distance_mean",
        )
        if not all(math.isfinite(float(report[key])) for key in required_finite):
            raise FloatingPointError("Periodic evaluation contains NaN or Inf")
        if plot_paths > 0:
            path_data = plot_path.with_suffix(".npz")
            if not plot_path.exists() or not path_data.exists():
                raise FileNotFoundError("Periodic trajectory PNG/NPZ was not created")
        summary = {
            "status": "ok",
            "gradient_step": gradient_step,
            "training_elapsed_seconds": training_elapsed_seconds,
            "episodes": episodes,
            "success_mean": float(report["success_mean"]),
            "return_mean": float(report["return_mean"]),
            "d4rl_normalized_score": float(report["d4rl_normalized_score"]),
            "goal_progress_fraction_mean": float(
                report["goal_progress_fraction_mean"]
            ),
            "minimum_goal_distance_mean": float(
                report["minimum_goal_distance_mean"]
            ),
            "final_goal_distance_mean": float(report["final_goal_distance_mean"]),
            "dare_failure_count": report["dare_failure_count"],
            "dare_retry_count": report["dare_retry_count"],
            "closed_loop_spectral_radius_max": report[
                "closed_loop_spectral_radius_max"
            ],
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "report": str(report_path.resolve()),
            "path_plot_png": report.get("path_plot_png"),
            "path_data_npz": report.get("path_data_npz"),
        }
    except (
        FileNotFoundError,
        FloatingPointError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        stderr = getattr(error, "stderr", None)
        summary = {
            "status": "error",
            "gradient_step": gradient_step,
            "training_elapsed_seconds": training_elapsed_seconds,
            "episodes": episodes,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "report": str(report_path.resolve()),
            "error_type": type(error).__name__,
            "error": str(error),
            "stderr_tail": stderr[-4000:] if isinstance(stderr, str) else None,
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary["evaluation_seconds"] = time.perf_counter() - started
    history_path = evaluation_root / "history.jsonl"
    _write_jsonl_replacing_step(history_path, summary)
    save_environment_evaluation_trend(
        history_path,
        evaluation_root / "trend.png",
    )
    print(
        json.dumps(
            {
                "event": "periodic_environment_evaluation_finished",
                "gradient_step": gradient_step,
                "status": summary["status"],
                "evaluation_seconds": summary["evaluation_seconds"],
                "success_mean": summary.get("success_mean"),
                "goal_progress_fraction_mean": summary.get(
                    "goal_progress_fraction_mean"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--config", default="configs/antmaze_umaze.yaml")
    parser.add_argument("--data", default="data/processed/antmaze-umaze-v2")
    parser.add_argument("--output", default="runs/antmaze_umaze_td3_bc/seed_0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bc-warmup-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--environment-evaluation-interval",
        type=int,
        default=None,
        help="Gradient-step interval for legacy rollouts; 0 disables them.",
    )
    parser.add_argument("--environment-evaluation-episodes", type=int, default=None)
    parser.add_argument("--environment-evaluation-plot-paths", type=int, default=None)
    parser.add_argument("--environment-evaluation-device", default=None)
    parser.add_argument(
        "--environment-evaluation-timeout-seconds",
        type=float,
        default=None,
    )
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    td3_config = config["td3_bc"]
    device = resolve_device(args.device)
    seed = int(config["experiment"]["seed"] if args.seed is None else args.seed)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(seed + 10_000)

    gradient_steps = int(
        args.gradient_steps
        if args.gradient_steps is not None
        else td3_config["gradient_steps"]
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else td3_config["batch_size"]
    )
    bc_warmup_steps = int(
        args.bc_warmup_steps
        if args.bc_warmup_steps is not None
        else td3_config["bc_warmup_steps"]
    )
    log_interval = int(
        args.log_interval
        if args.log_interval is not None
        else td3_config["log_interval"]
    )
    validation_interval = int(
        args.validation_interval
        if args.validation_interval is not None
        else td3_config["validation_interval"]
    )
    checkpoint_interval = int(
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else td3_config["checkpoint_interval"]
    )
    environment_evaluation_interval = int(
        args.environment_evaluation_interval
        if args.environment_evaluation_interval is not None
        else td3_config["environment_evaluation_interval"]
    )
    environment_evaluation_episodes = int(
        args.environment_evaluation_episodes
        if args.environment_evaluation_episodes is not None
        else td3_config["environment_evaluation_episodes"]
    )
    environment_evaluation_plot_paths = int(
        args.environment_evaluation_plot_paths
        if args.environment_evaluation_plot_paths is not None
        else td3_config["environment_evaluation_plot_paths"]
    )
    environment_evaluation_device = str(
        args.environment_evaluation_device
        if args.environment_evaluation_device is not None
        else td3_config["environment_evaluation_device"]
    )
    if environment_evaluation_device == "same":
        environment_evaluation_device = str(device)
    environment_evaluation_timeout_seconds = float(
        args.environment_evaluation_timeout_seconds
        if args.environment_evaluation_timeout_seconds is not None
        else td3_config["environment_evaluation_timeout_seconds"]
    )
    max_wall_time_hours = float(
        args.max_wall_time_hours
        if args.max_wall_time_hours is not None
        else td3_config["max_wall_time_hours"]
    )
    for name, value in {
        "gradient_steps": gradient_steps,
        "batch_size": batch_size,
        "log_interval": log_interval,
        "validation_interval": validation_interval,
        "checkpoint_interval": checkpoint_interval,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if bc_warmup_steps < 0 or max_wall_time_hours <= 0:
        raise ValueError("bc_warmup_steps must be non-negative and wall time positive")
    if environment_evaluation_interval < 0:
        raise ValueError("environment_evaluation_interval must be non-negative")
    if environment_evaluation_episodes < 1:
        raise ValueError("environment_evaluation_episodes must be positive")
    if not 0 <= environment_evaluation_plot_paths <= environment_evaluation_episodes:
        raise ValueError(
            "environment_evaluation_plot_paths must be between zero and episodes"
        )
    if environment_evaluation_timeout_seconds <= 0:
        raise ValueError("environment_evaluation_timeout_seconds must be positive")

    data_root = Path(args.data)
    train_data = load_npz_dataset(data_root / "train.npz")
    validation_data = load_npz_dataset(data_root / "validation.npz")
    max_delta_action = float(td3_config["max_delta_action"])
    observed_action_max = float(np.max(np.abs(train_data.action)))
    if observed_action_max > max_delta_action + 1e-6:
        raise ValueError(
            "D4RL delta action exceeds configured support: "
            f"{observed_action_max:.6f} > {max_delta_action:.6f}"
        )

    policy, koopman_payload = make_policy(
        args.koopman_checkpoint,
        device,
        mean_action_limit=max_delta_action,
    )
    if train_data.state.shape[1] != policy.koopman.state_dim:
        raise ValueError("Offline state dimension does not match Koopman checkpoint")
    if train_data.action.shape[1] != policy.koopman.action_dim:
        raise ValueError("Offline action dimension does not match Koopman checkpoint")
    policy.requires_grad_(False)
    policy.actor.requires_grad_(True)
    target_policy = copy.deepcopy(policy).eval()
    target_policy.requires_grad_(False)

    critic = TwinActionValueCritic(
        policy.koopman.state_dim,
        policy.koopman.action_dim,
        policy.state_mean,
        policy.state_std,
        hidden_dims=td3_config["critic_hidden_dims"],
        activation=td3_config["critic_activation"],
        action_scale=max_delta_action,
    ).to(device)
    target_critic = copy.deepcopy(critic).eval()
    target_critic.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(
        policy.actor.parameters(),
        lr=float(td3_config["actor_learning_rate"]),
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=float(td3_config["critic_learning_rate"]),
    )
    trainer = TD3BCTrainer(
        policy,
        target_policy,
        critic,
        target_critic,
        actor_optimizer,
        critic_optimizer,
        discount=float(td3_config["discount"]),
        tau=float(td3_config["tau"]),
        policy_noise=float(td3_config["policy_noise"]),
        noise_clip=float(td3_config["noise_clip"]),
        policy_frequency=int(td3_config["policy_frequency"]),
        alpha=float(td3_config["alpha"]),
        bc_weight=float(td3_config["bc_weight"]),
        bc_warmup_steps=bc_warmup_steps,
        max_delta_action=max_delta_action,
        reward_scale=float(td3_config["reward_scale"]),
        reward_bias=float(td3_config["reward_bias"]),
        max_grad_norm=float(td3_config["max_grad_norm"]),
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    status_path = output / "training_status.json"
    if args.resume is None and history_path.exists():
        raise FileExistsError(
            f"{history_path} already exists; use a new output or --resume"
        )

    expected_koopman_sha = sha256(args.koopman_checkpoint)
    gradient_step = 0
    elapsed_before = 0.0
    best_validation_bc = float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload.get("method") != "td3_bc_koopman_lqr":
            raise ValueError("Resume checkpoint is not Koopman-LQR TD3+BC")
        if payload["koopman_checkpoint_sha256"] != expected_koopman_sha:
            raise ValueError("Resume checkpoint references a different Koopman model")
        if int(payload["seed"]) != seed:
            raise ValueError("Resume seed does not match --seed")
        policy.load_state_dict(payload["policy"])
        target_policy.actor.load_state_dict(payload["target_actor"])
        critic.load_state_dict(payload["critic"])
        target_critic.load_state_dict(payload["target_critic"])
        actor_optimizer.load_state_dict(payload["actor_optimizer"])
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
        gradient_step = int(payload["gradient_step"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        best_validation_bc = float(
            payload.get("best_validation_behavior_cloning_loss", float("inf"))
        )
        if payload.get("numpy_rng_state") is not None:
            rng.bit_generator.state = payload["numpy_rng_state"]
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        reconcile_history(history_path, gradient_step)
        if status_path.exists():
            status_path.unlink()
        reconcile_environment_evaluations(
            output / "periodic_evaluation",
            gradient_step,
        )

    validation_batch = sample_transition_batch(
        validation_data,
        int(td3_config["validation_batch_size"]),
        validation_rng,
        device,
    )
    started = time.monotonic()
    interval_started = time.perf_counter()
    interval_metrics: dict[
        str,
        list[torch.Tensor | float | int | None],
    ] = {}
    interval_environment_evaluation_seconds = 0.0
    environment_evaluation_seconds_total = 0.0
    periodic_history_path = output / "periodic_evaluation/history.jsonl"
    if periodic_history_path.exists():
        environment_evaluation_seconds_total = sum(
            float(json.loads(line).get("evaluation_seconds", 0.0))
            for line in periodic_history_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )

    runtime = {
        "gradient_steps": gradient_steps,
        "batch_size": batch_size,
        "bc_warmup_steps": bc_warmup_steps,
        "max_delta_action": max_delta_action,
        "max_wall_time_hours": max_wall_time_hours,
        "dataset_schema_version": 2,
        "environment_evaluation_interval": environment_evaluation_interval,
        "environment_evaluation_episodes": environment_evaluation_episodes,
        "environment_evaluation_plot_paths": environment_evaluation_plot_paths,
        "environment_evaluation_device": environment_evaluation_device,
        "environment_evaluation_timeout_seconds": (
            environment_evaluation_timeout_seconds
        ),
        "environment_evaluation_seconds_total": (
            environment_evaluation_seconds_total
        ),
    }

    def checkpoint_payload(elapsed_seconds: float) -> dict[str, Any]:
        return {
            "format_version": 1,
            "method": "td3_bc_koopman_lqr",
            "policy": policy.state_dict(),
            "actor": policy.actor.state_dict(),
            "target_actor": target_policy.actor.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": target_critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "koopman_checkpoint": str(Path(args.koopman_checkpoint).resolve()),
            "koopman_checkpoint_sha256": expected_koopman_sha,
            "data": str(data_root.resolve()),
            "gradient_step": gradient_step,
            "seed": seed,
            "elapsed_seconds": elapsed_seconds,
            "best_validation_behavior_cloning_loss": best_validation_bc,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "config": config,
            "runtime": runtime,
        }

    stop_reason = "gradient_steps"
    try:
        while gradient_step < gradient_steps:
            elapsed = elapsed_before + time.monotonic() - started
            if elapsed >= max_wall_time_hours * 3600.0:
                stop_reason = "max_wall_time"
                break
            gradient_step += 1
            batch = sample_transition_batch(
                train_data,
                batch_size,
                rng,
                device,
            )
            metrics = trainer.update(batch, gradient_step)
            for key, value in metrics.items():
                interval_metrics.setdefault(key, []).append(value)

            should_validate = (
                gradient_step == 1
                or gradient_step % validation_interval == 0
                or gradient_step == gradient_steps
            )
            validation = (
                offline_validation_metrics(policy, critic, validation_batch)
                if should_validate
                else None
            )
            is_best = False
            if validation is not None:
                validation_bc = validation["behavior_cloning_loss"]
                if validation_bc < best_validation_bc:
                    best_validation_bc = validation_bc
                    is_best = True

            environment_evaluation = None
            if environment_evaluation_due(
                gradient_step,
                environment_evaluation_interval,
            ):
                elapsed_at_snapshot = (
                    elapsed_before + time.monotonic() - started
                )
                environment_evaluation = run_environment_evaluation(
                    project_root=project_root,
                    output=output,
                    gradient_step=gradient_step,
                    training_elapsed_seconds=elapsed_at_snapshot,
                    checkpoint_payload=checkpoint_payload(elapsed_at_snapshot),
                    episodes=environment_evaluation_episodes,
                    plot_paths=environment_evaluation_plot_paths,
                    device=environment_evaluation_device,
                    seed_offset=100_000 + seed * 1_000,
                    timeout_seconds=environment_evaluation_timeout_seconds,
                )
                evaluation_seconds = float(
                    environment_evaluation["evaluation_seconds"]
                )
                interval_environment_evaluation_seconds += evaluation_seconds
                environment_evaluation_seconds_total += evaluation_seconds
                runtime["environment_evaluation_seconds_total"] = (
                    environment_evaluation_seconds_total
                )

            should_log = (
                gradient_step == 1
                or gradient_step % log_interval == 0
                or gradient_step == gradient_steps
                or environment_evaluation is not None
            )
            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                interval_seconds = max(
                    time.perf_counter()
                    - interval_started
                    - interval_environment_evaluation_seconds,
                    1e-12,
                )
                elapsed = elapsed_before + time.monotonic() - started
                row = {
                    "method": "td3_bc_koopman_lqr",
                    "seed": seed,
                    "gradient_step": gradient_step,
                    "elapsed_seconds": elapsed,
                    "updates_per_second": (
                        len(next(iter(interval_metrics.values())))
                        / interval_seconds
                    ),
                    "train": {
                        key: finite_mean(values)
                        for key, values in interval_metrics.items()
                    },
                    "validation": validation,
                    "environment_evaluation": environment_evaluation,
                    "environment_evaluation_seconds_total": (
                        environment_evaluation_seconds_total
                    ),
                }
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                interval_metrics = {}
                interval_started = time.perf_counter()
                interval_environment_evaluation_seconds = 0.0

            should_checkpoint = (
                gradient_step % checkpoint_interval == 0
                or gradient_step == gradient_steps
            )
            if should_checkpoint or is_best:
                elapsed = elapsed_before + time.monotonic() - started
                payload = checkpoint_payload(elapsed)
                torch.save(payload, output / "last.pt")
                if should_checkpoint:
                    torch.save(
                        payload,
                        output / f"recovery_step_{gradient_step:08d}.pt",
                    )
                if is_best:
                    torch.save(payload, output / "best_bc_validation.pt")
    except (RuntimeError, FloatingPointError) as error:
        elapsed = elapsed_before + time.monotonic() - started
        emergency = checkpoint_payload(elapsed)
        emergency["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        torch.save(emergency, output / "emergency.pt")
        status = {
            "method": "td3_bc_koopman_lqr",
            "gradient_step": gradient_step,
            "elapsed_seconds": elapsed,
            "stop_reason": "training_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "emergency_checkpoint_sha256": sha256(output / "emergency.pt"),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise

    elapsed = elapsed_before + time.monotonic() - started
    final_payload = checkpoint_payload(elapsed)
    torch.save(final_payload, output / "last.pt")
    status = {
        "method": "td3_bc_koopman_lqr",
        "seed": seed,
        "gradient_step": gradient_step,
        "requested_gradient_steps": gradient_steps,
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
        "best_validation_behavior_cloning_loss": best_validation_bc,
        "environment_evaluation_seconds_total": (
            environment_evaluation_seconds_total
        ),
        "environment_evaluation_history": (
            str(periodic_history_path.resolve())
            if periodic_history_path.exists()
            else None
        ),
        "last_checkpoint_sha256": sha256(output / "last.pt"),
        "koopman_checkpoint_sha256": expected_koopman_sha,
        "device": str(device),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
