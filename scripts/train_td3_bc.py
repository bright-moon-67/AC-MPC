#!/usr/bin/env python
"""Offline TD3+BC for the state-conditioned Koopman-LQR cost actor."""
from __future__ import annotations

import argparse
import copy
import json
import random
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
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

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

    runtime = {
        "gradient_steps": gradient_steps,
        "batch_size": batch_size,
        "bc_warmup_steps": bc_warmup_steps,
        "max_delta_action": max_delta_action,
        "max_wall_time_hours": max_wall_time_hours,
        "dataset_schema_version": 2,
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

            should_log = (
                gradient_step == 1
                or gradient_step % log_interval == 0
                or gradient_step == gradient_steps
            )
            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                interval_seconds = time.perf_counter() - interval_started
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
                }
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                interval_metrics = {}
                interval_started = time.perf_counter()

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
        "last_checkpoint_sha256": sha256(output / "last.pt"),
        "koopman_checkpoint_sha256": expected_koopman_sha,
        "device": str(device),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
