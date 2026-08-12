#!/usr/bin/env python
"""Behavior-clone the learned history BC-KMPC actor from MPC demonstrations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.serialization import make_history_mpc_policy


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--sequence-weight", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--solver-iterations", type=int, default=20)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(
        args.epochs,
        args.batch_size,
        args.checkpoint_interval,
        args.horizon,
        args.solver_iterations,
    ) < 1:
        parser.error("Epoch, batch, horizon and solver counts must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must lie in (0,1)")
    if args.sequence_weight < 0:
        parser.error("sequence-weight must be non-negative")
    return args


def _split_indices(
    episode_ids: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    if len(episodes) > 1:
        validation_count = max(1, int(round(len(episodes) * validation_fraction)))
        validation_count = min(validation_count, len(episodes) - 1)
        validation_episodes = episodes[:validation_count]
        validation_mask = np.isin(episode_ids, validation_episodes)
        return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)
    order = rng.permutation(len(episode_ids))
    validation_count = max(1, int(round(len(order) * validation_fraction)))
    if validation_count >= len(order):
        raise ValueError("Dataset needs at least two samples")
    return order[validation_count:], order[:validation_count]


def _future_targets(
    actions: np.ndarray,
    episode_ids: np.ndarray,
    active_waypoint_index: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build receding-horizon expert targets without crossing task stages."""

    count, action_dim = actions.shape
    future = np.zeros((count, horizon, action_dim), dtype=np.float32)
    mask = np.zeros((count, horizon), dtype=np.float32)
    for index in range(count):
        for offset in range(horizon):
            following = index + offset
            if (
                following >= count
                or episode_ids[following] != episode_ids[index]
                or active_waypoint_index[following]
                != active_waypoint_index[index]
            ):
                break
            future[index, offset] = actions[following]
            mask[index, offset] = 1.0
    return future, mask


@torch.no_grad()
def _evaluate(policy, loader, device: torch.device) -> dict:
    policy.eval()
    squared_error_sum = 0.0
    residual_sum = 0.0
    elements = 0
    samples = 0
    sequence_squared_error_sum = 0.0
    sequence_elements = 0.0
    for observations, targets, future, future_mask in loader:
        observations = observations.to(device)
        targets = targets.to(device)
        future = future.to(device)
        future_mask = future_mask.to(device)
        output = policy.actor_mean(observations)
        squared_error = (output.action - targets).square()
        squared_error_sum += float(squared_error.sum())
        residual_sum += float(output.projected_gradient_residual.sum())
        elements += squared_error.numel()
        samples += len(observations)
        if output.action_sequence.shape[-2] > 1:
            future_error = (
                output.action_sequence[:, 1:] - future[:, 1:]
            ).square().mean(dim=-1)
            valid = future_mask[:, 1:]
            sequence_squared_error_sum += float((future_error * valid).sum())
            sequence_elements += float(valid.sum())
    return {
        "mse": squared_error_sum / elements,
        "rmse": (squared_error_sum / elements) ** 0.5,
        "projected_gradient_residual_mean": residual_sum / samples,
        "future_sequence_mse": (
            sequence_squared_error_sum / sequence_elements
            if sequence_elements
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    checkpoint = Path(args.koopman_checkpoint).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not checkpoint.is_file() or not dataset_path.is_file():
        raise FileNotFoundError("Koopman checkpoint and expert dataset must exist")
    dataset_report_path = dataset_path.with_suffix(".json")
    if not dataset_report_path.is_file():
        raise FileNotFoundError(
            f"Missing expert dataset metadata: {dataset_report_path}"
        )
    dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
    if dataset_report.get("kind") != (
        "manisoft_history_bc_kmpc_three_waypoint_expert"
    ):
        raise ValueError("Expert dataset is not the three-waypoint schema")
    if int(dataset_report.get("schema_version", 0)) < 5:
        raise ValueError(
            "Expert dataset predates randomized waypoint-bank BC-KMPC; "
            "recollect it with collect_manisoft_bc_kmpc_expert.py"
        )
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    if args.resume is None and history_path.exists():
        raise FileExistsError(
            f"{history_path} already exists; use a new output or --resume"
        )

    policy, _ = make_history_mpc_policy(
        checkpoint,
        device,
        horizon=args.horizon,
        absolute_action_limit=args.absolute_action_limit,
        solver_iterations=args.solver_iterations,
        waypoint_count=3,
    )
    policy.koopman.eval()
    policy.critic.requires_grad_(False)
    policy.log_std.requires_grad_(False)
    optimizer = torch.optim.Adam(
        policy.actor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    with np.load(dataset_path, allow_pickle=False) as archive:
        observations = np.asarray(archive["observation"], dtype=np.float32)
        targets = np.asarray(archive["expert_action"], dtype=np.float32)
        episode_ids = np.asarray(archive["episode_id"], dtype=np.int64)
        active_waypoint_index = np.asarray(
            archive["active_waypoint_index"], dtype=np.int64
        )
    if observations.ndim != 2 or observations.shape[1] != policy.observation_dim:
        raise ValueError(
            f"Dataset observation must be [N,{policy.observation_dim}], "
            f"got {observations.shape}"
        )
    if targets.shape != (len(observations), policy.action_dim):
        raise ValueError("Dataset expert_action shape is incompatible")
    if episode_ids.shape != (len(observations),):
        raise ValueError("Dataset episode_id shape is incompatible")
    if active_waypoint_index.shape != (len(observations),) or np.any(
        (active_waypoint_index < 0) | (active_waypoint_index >= 3)
    ):
        raise ValueError("Dataset active_waypoint_index is incompatible")
    if not np.isfinite(observations).all() or not np.isfinite(targets).all():
        raise ValueError("Dataset contains NaN or Inf")
    future_targets, future_mask = _future_targets(
        targets,
        episode_ids,
        active_waypoint_index,
        policy.actor.horizon,
    )
    train_indices, validation_indices = _split_indices(
        episode_ids,
        args.validation_fraction,
        args.seed,
    )
    if args.max_train_samples is not None:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_validation_samples is not None:
        validation_indices = validation_indices[: args.max_validation_samples]
    train_data = TensorDataset(
        torch.from_numpy(observations[train_indices]),
        torch.from_numpy(targets[train_indices]),
        torch.from_numpy(future_targets[train_indices]),
        torch.from_numpy(future_mask[train_indices]),
    )
    validation_data = TensorDataset(
        torch.from_numpy(observations[validation_indices]),
        torch.from_numpy(targets[validation_indices]),
        torch.from_numpy(future_targets[validation_indices]),
        torch.from_numpy(future_mask[validation_indices]),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    expected_koopman_sha = sha256(checkpoint)
    start_epoch = 0
    best_validation_mse = float("inf")
    elapsed_before = 0.0
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser().resolve()
        resume_payload = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        if resume_payload.get("method") != "bc_kmpc_bc":
            raise ValueError("Resume checkpoint is not BC-KMPC behavior cloning")
        if int(resume_payload.get("format_version", 0)) < 5:
            raise ValueError("Resume checkpoint predates randomized waypoint-bank BC-KMPC")
        if resume_payload["koopman_checkpoint_sha256"] != expected_koopman_sha:
            raise ValueError("Resume checkpoint references another Koopman model")
        if resume_payload.get("waypoint_bank_sha256") != dataset_report[
            "waypoint_bank_sha256"
        ]:
            raise ValueError("Resume checkpoint references another waypoint set")
        expected_runtime = {
            "horizon": policy.actor.horizon,
            "solver_iterations": policy.actor.solver_iterations,
            "absolute_action_limit": args.absolute_action_limit,
            "waypoint_count": policy.waypoint_count,
            "solver": "absolute_box_fista_v1",
            "fixed_smoothness": False,
            "sequence_weight": args.sequence_weight,
        }
        for key, expected in expected_runtime.items():
            actual = resume_payload["runtime"].get(key)
            if actual != expected:
                raise ValueError(
                    f"Resume runtime {key}={actual!r}, expected {expected!r}"
                )
        policy.actor.load_state_dict(resume_payload["actor"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_validation_mse = float(resume_payload["best_validation_mse"])
        elapsed_before = float(resume_payload.get("elapsed_seconds", 0.0))

    runtime = {
        "horizon": policy.actor.horizon,
        "solver_iterations": policy.actor.solver_iterations,
        "absolute_action_limit": args.absolute_action_limit,
        "observation_dim": policy.observation_dim,
        "history_steps": policy.history_steps,
        "waypoint_count": policy.waypoint_count,
        "solver": "absolute_box_fista_v1",
        "fixed_smoothness": False,
        "sequence_weight": args.sequence_weight,
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
    }
    metadata = {
        "method": "bc_kmpc_bc",
        "format_version": 5,
        "koopman_checkpoint": str(checkpoint),
        "koopman_checkpoint_sha256": expected_koopman_sha,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "waypoint_root": dataset_report["waypoint_root"],
        "waypoint_bank_manifest": dataset_report["waypoint_bank_manifest"],
        "waypoint_bank_sha256": dataset_report["waypoint_bank_sha256"],
        "waypoint_triplet_count": dataset_report["waypoint_triplet_count"],
        "references": dataset_report["references"],
        "seed": args.seed,
        "runtime": runtime,
    }
    _write_json(output / "run_config.json", {**metadata, "arguments": vars(args)})

    started = time.monotonic()
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs):
            policy.actor.train()
            train_squared_error = 0.0
            train_elements = 0
            train_loss_sum = 0.0
            gradient_norm_sum = 0.0
            batches = 0
            for (
                batch_observations,
                batch_targets,
                batch_future,
                batch_future_mask,
            ) in train_loader:
                batch_observations = batch_observations.to(device)
                batch_targets = batch_targets.to(device)
                batch_future = batch_future.to(device)
                batch_future_mask = batch_future_mask.to(device)
                output_actor = policy.actor_mean(batch_observations)
                squared_error = (output_actor.action - batch_targets).square()
                loss = squared_error.mean()
                if output_actor.action_sequence.shape[-2] > 1:
                    future_error = (
                        output_actor.action_sequence[:, 1:]
                        - batch_future[:, 1:]
                    ).square().mean(dim=-1)
                    valid = batch_future_mask[:, 1:]
                    future_loss = (future_error * valid).sum() / valid.sum().clamp_min(
                        1.0
                    )
                    loss = loss + args.sequence_weight * future_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    policy.actor.parameters(),
                    args.gradient_clip_norm,
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("BC-KMPC BC gradient is NaN or Inf")
                optimizer.step()
                train_squared_error += float(squared_error.detach().sum())
                train_elements += squared_error.numel()
                train_loss_sum += float(loss.detach())
                gradient_norm_sum += float(gradient_norm.detach())
                batches += 1
            validation = _evaluate(
                policy,
                validation_loader,
                device,
            )
            elapsed = elapsed_before + time.monotonic() - started
            row = {
                "epoch": epoch,
                "elapsed_seconds": elapsed,
                "train_mse": train_squared_error / train_elements,
                "train_rmse": (train_squared_error / train_elements) ** 0.5,
                "train_loss": train_loss_sum / batches,
                "gradient_norm": gradient_norm_sum / batches,
                **{f"validation_{key}": value for key, value in validation.items()},
            }
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            payload = {
                **metadata,
                "actor": policy.actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_validation_mse": min(
                    best_validation_mse,
                    validation["mse"],
                ),
                "elapsed_seconds": elapsed,
                "last_report": row,
            }
            _save(output / "last.pt", payload)
            if validation["mse"] < best_validation_mse:
                best_validation_mse = validation["mse"]
                _save(output / "best_validation.pt", payload)
            if (epoch + 1) % args.checkpoint_interval == 0:
                _save(output / f"recovery_epoch_{epoch:04d}.pt", payload)
            last_epoch = epoch
            print(json.dumps(row, sort_keys=True), flush=True)
        status = {
            "state": "complete",
            "method": "bc_kmpc_bc",
            "last_epoch": last_epoch,
            "best_validation_mse": best_validation_mse,
            "best_validation_checkpoint": str(
                (output / "best_validation.pt").resolve()
            ),
        }
        _write_json(output / "training_status.json", status)
    except BaseException as error:
        _write_json(
            output / "training_status.json",
            {
                "state": "failed",
                "method": "bc_kmpc_bc",
                "last_epoch": last_epoch,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


if __name__ == "__main__":
    main()
