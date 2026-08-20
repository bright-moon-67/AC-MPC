#!/usr/bin/env python
"""Behavior-clone the differentiable Koopman-MPC actor for kinematic push."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.serialization import make_history_mpc_policy


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
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--solver-iterations", type=int, default=20)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--phase-balanced-sampling", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(
        args.epochs,
        args.batch_size,
        args.horizon,
        args.solver_iterations,
        args.checkpoint_interval,
    ) < 1:
        parser.error("counts must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must lie in (0,1)")
    return args


def _episode_split(
    episode_ids: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_ids)
    rng.shuffle(episodes)
    if len(episodes) > 1:
        count = min(len(episodes) - 1, max(1, round(len(episodes) * fraction)))
        validation = np.isin(episode_ids, episodes[:count])
        return np.flatnonzero(~validation), np.flatnonzero(validation)
    order = rng.permutation(len(episode_ids))
    count = min(len(order) - 1, max(1, round(len(order) * fraction)))
    return order[count:], order[:count]


@torch.no_grad()
def _evaluate(policy, loader, device: torch.device) -> float:
    policy.eval()
    squared = count = 0
    for observations, actions, _ in loader:
        prediction = policy.actor_mean(observations.to(device)).action
        error = (prediction - actions.to(device)).square()
        squared += float(error.sum())
        count += error.numel()
    return squared / max(count, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    checkpoint = Path(args.koopman_checkpoint).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not checkpoint.is_file() or not dataset.is_file():
        raise FileNotFoundError("Koopman checkpoint or expert dataset is missing")
    metadata_path = dataset.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing expert metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("task_mode") != "kinematic_push":
        raise ValueError("Dataset is not a kinematic-push expert dataset")
    if metadata.get("koopman_checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("Dataset references another Koopman checkpoint")

    with np.load(dataset, allow_pickle=False) as archive:
        observations = np.asarray(archive["observation"], dtype=np.float32)
        actions = np.asarray(archive["expert_action"], dtype=np.float32)
        episode_ids = np.asarray(archive["episode_id"], dtype=np.int64)
        phases = np.asarray(archive["phase"], dtype=np.int64)
    if args.max_samples is not None:
        observations = observations[: args.max_samples]
        actions = actions[: args.max_samples]
        episode_ids = episode_ids[: args.max_samples]
        phases = phases[: args.max_samples]
    if len(observations) < 2 or actions.shape != (len(observations), 18):
        raise ValueError("Expert dataset has invalid or insufficient samples")

    policy, _ = make_history_mpc_policy(
        checkpoint,
        device,
        horizon=args.horizon,
        absolute_action_limit=args.absolute_action_limit,
        solver_iterations=args.solver_iterations,
        task_mode="kinematic_push",
    )
    if observations.shape[1:] != (policy.observation_dim,):
        raise ValueError(
            f"Dataset observation shape {observations.shape[1:]} does not match "
            f"policy {(policy.observation_dim,)}"
        )
    rng = np.random.default_rng(args.seed)
    train_indices, validation_indices = _episode_split(
        episode_ids, args.validation_fraction, rng
    )
    tensor_observations = torch.from_numpy(observations)
    tensor_actions = torch.from_numpy(actions)
    tensor_phases = torch.from_numpy(phases)
    train_dataset = TensorDataset(
        tensor_observations[train_indices],
        tensor_actions[train_indices],
        tensor_phases[train_indices],
    )
    sampler = None
    shuffle = True
    if args.phase_balanced_sampling:
        counts = np.bincount(phases[train_indices], minlength=4).clip(min=1)
        weights = 1.0 / counts[phases[train_indices]]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )
    validation_loader = DataLoader(
        TensorDataset(
            tensor_observations[validation_indices],
            tensor_actions[validation_indices],
            tensor_phases[validation_indices],
        ),
        batch_size=args.batch_size,
    )
    optimizer = torch.optim.AdamW(
        policy.actor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
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
        "solver": "absolute_box_fista_v1",
    }

    def checkpoint_payload(epoch: int, validation_mse: float) -> dict:
        return {
            "format_version": 6,
            "method": "kinematic_push_bc_kmpc",
            "actor": policy.actor.state_dict(),
            "koopman_checkpoint": str(checkpoint),
            "koopman_checkpoint_sha256": sha256(checkpoint),
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "scenario": metadata.get("scenario"),
            "scenario_sha256": metadata.get("scenario_sha256"),
            "epoch": epoch,
            "best_validation_mse": min(best, validation_mse),
            "runtime": runtime,
            "config": vars(args),
        }

    for epoch in range(1, args.epochs + 1):
        policy.train()
        squared = elements = 0
        for batch_observation, batch_action, _ in train_loader:
            batch_observation = batch_observation.to(device)
            batch_action = batch_action.to(device)
            prediction = policy.actor_mean(batch_observation).action
            loss = (prediction - batch_action).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.actor.parameters(), args.gradient_clip_norm
            )
            optimizer.step()
            squared += float(loss.detach()) * batch_action.numel()
            elements += batch_action.numel()
        validation_mse = _evaluate(policy, validation_loader, device)
        train_mse = squared / max(elements, 1)
        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "validation_rmse": validation_mse**0.5,
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation_mse < best:
            best = validation_mse
            _save(output / "best_validation.pt", checkpoint_payload(epoch, validation_mse))
        if epoch % args.checkpoint_interval == 0:
            _save(output / f"epoch_{epoch:04d}.pt", checkpoint_payload(epoch, validation_mse))
    _save(output / "last.pt", checkpoint_payload(args.epochs, validation_mse))
    status = {
        "status": "complete",
        "best_validation_mse": best,
        "best_checkpoint": str((output / "best_validation.pt").resolve()),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "runtime": runtime,
    }
    (output / "training_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
