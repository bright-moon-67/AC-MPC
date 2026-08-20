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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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
    parser.add_argument(
        "--converged-action-weight",
        type=float,
        default=0.0,
        help="Weight for matching the current expert action after a larger "
        "FISTA iteration budget.",
    )
    parser.add_argument("--solver-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--converged-solver-iterations",
        type=int,
        default=100,
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--stage-balanced-sampling",
        action="store_true",
        help="Sample stages 0/1/2 equally during training.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--solver-iterations", type=int, default=20)
    parser.add_argument("--quadratic-log-scale", type=float, default=None)
    parser.add_argument("--linear-scale", type=float, default=None)
    parser.add_argument("--action-quadratic-scale", type=float, default=None)
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
        args.converged_solver_iterations,
    ) < 1:
        parser.error("Epoch, batch, horizon and solver counts must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must lie in (0,1)")
    if min(
        args.sequence_weight,
        args.converged_action_weight,
        args.solver_consistency_weight,
    ) < 0:
        parser.error("BC loss weights must be non-negative")
    if args.converged_solver_iterations < args.solver_iterations:
        parser.error(
            "converged-solver-iterations must be at least solver-iterations"
        )
    if args.quadratic_log_scale is not None and args.quadratic_log_scale <= 0:
        parser.error("quadratic-log-scale must be positive")
    if args.linear_scale is not None and args.linear_scale <= 0:
        parser.error("linear-scale must be positive")
    if args.action_quadratic_scale is not None and args.action_quadratic_scale <= 0:
        parser.error("action-quadratic-scale must be positive")
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


def _converged_sequence(
    actor,
    output,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat, residual = actor.solve_condensed_qp(
        output.qp_hessian,
        output.qp_linear,
        iterations=iterations,
    )
    return flat.reshape(
        *flat.shape[:-1],
        actor.horizon,
        actor.action_dim,
    ), residual


@torch.no_grad()
def _evaluate(
    policy,
    loader,
    device: torch.device,
    *,
    sequence_weight: float,
    converged_action_weight: float,
    solver_consistency_weight: float,
    converged_solver_iterations: int,
) -> dict:
    policy.eval()
    squared_error_sum = 0.0
    residual_sum = 0.0
    elements = 0
    samples = 0
    sequence_squared_error_sum = 0.0
    sequence_elements = 0.0
    converged_squared_error_sum = 0.0
    converged_elements = 0.0
    converged_residual_sum = 0.0
    solver_consistency_squared_sum = 0.0
    stage_squared_error_sum = np.zeros(3, dtype=np.float64)
    stage_elements = np.zeros(3, dtype=np.float64)
    stage_converged_error_sum = np.zeros(3, dtype=np.float64)
    stage_consistency_error_sum = np.zeros(3, dtype=np.float64)
    for observations, targets, future, future_mask, stages in loader:
        observations = observations.to(device)
        targets = targets.to(device)
        future = future.to(device)
        future_mask = future_mask.to(device)
        stages = stages.to(device)
        output = policy.actor_mean(observations)
        squared_error = (output.action - targets).square()
        squared_error_sum += float(squared_error.sum())
        residual_sum += float(output.projected_gradient_residual.sum())
        elements += squared_error.numel()
        samples += len(observations)
        for stage in range(3):
            selected = stages == stage
            if bool(selected.any()):
                stage_squared_error_sum[stage] += float(
                    squared_error[selected].sum()
                )
                stage_elements[stage] += int(selected.sum()) * squared_error.shape[-1]
        if output.action_sequence.shape[-2] > 1:
            future_error = (
                output.action_sequence[:, 1:] - future[:, 1:]
            ).square().mean(dim=-1)
            valid = future_mask[:, 1:]
            sequence_squared_error_sum += float((future_error * valid).sum())
            sequence_elements += float(valid.sum())
        if converged_action_weight > 0 or solver_consistency_weight > 0:
            converged, converged_residual = _converged_sequence(
                policy.actor,
                output,
                converged_solver_iterations,
            )
            converged_error = (converged[:, 0] - targets).square()
            converged_squared_error_sum += float(converged_error.sum())
            converged_elements += converged_error.numel()
            solver_consistency_squared_sum += float(
                (converged[:, 0] - output.action).square().sum()
            )
            for stage in range(3):
                selected = stages == stage
                if bool(selected.any()):
                    stage_converged_error_sum[stage] += float(
                        converged_error[selected].sum()
                    )
                    stage_consistency_error_sum[stage] += float(
                        (
                            converged[selected, 0] - output.action[selected]
                        ).square().sum()
                    )
            converged_residual_sum += float(converged_residual.sum())
    mse = squared_error_sum / elements
    future_mse = (
        sequence_squared_error_sum / sequence_elements
        if sequence_elements
        else 0.0
    )
    converged_mse = (
        converged_squared_error_sum / converged_elements
        if converged_elements
        else 0.0
    )
    solver_consistency_mse = (
        solver_consistency_squared_sum / converged_elements
        if converged_elements
        else 0.0
    )
    stage_mse = stage_squared_error_sum / np.maximum(stage_elements, 1.0)
    stage_converged_mse = stage_converged_error_sum / np.maximum(
        stage_elements, 1.0
    )
    stage_consistency_mse = stage_consistency_error_sum / np.maximum(
        stage_elements, 1.0
    )
    stage_objective = (
        stage_mse
        + converged_action_weight * stage_converged_mse
        + solver_consistency_weight * stage_consistency_mse
    )
    result = {
        "objective": (
            mse
            + sequence_weight * future_mse
            + converged_action_weight * converged_mse
            + solver_consistency_weight * solver_consistency_mse
        ),
        "mse": mse,
        "rmse": mse ** 0.5,
        "projected_gradient_residual_mean": residual_sum / samples,
        "future_sequence_mse": future_mse,
        "converged_action_mse": converged_mse,
        "solver_consistency_mse": solver_consistency_mse,
        "converged_projected_gradient_residual_mean": (
            converged_residual_sum / samples
            if converged_action_weight > 0 or solver_consistency_weight > 0
            else 0.0
        ),
        "stage_macro_objective": float(stage_objective.mean()),
    }
    for stage in range(3):
        result[f"stage_{stage}_samples"] = int(
            stage_elements[stage] / policy.action_dim
        )
        result[f"stage_{stage}_mse"] = float(stage_mse[stage])
        result[f"stage_{stage}_rmse"] = float(stage_mse[stage] ** 0.5)
        result[f"stage_{stage}_converged_action_mse"] = float(
            stage_converged_mse[stage]
        )
        result[f"stage_{stage}_solver_consistency_mse"] = float(
            stage_consistency_mse[stage]
        )
        result[f"stage_{stage}_objective"] = float(stage_objective[stage])
    return result


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
        quadratic_log_scale=args.quadratic_log_scale,
        linear_scale=args.linear_scale,
        action_quadratic_scale=args.action_quadratic_scale,
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
        torch.from_numpy(active_waypoint_index[train_indices]),
    )
    validation_data = TensorDataset(
        torch.from_numpy(observations[validation_indices]),
        torch.from_numpy(targets[validation_indices]),
        torch.from_numpy(future_targets[validation_indices]),
        torch.from_numpy(future_mask[validation_indices]),
        torch.from_numpy(active_waypoint_index[validation_indices]),
    )
    train_sampler = None
    if args.stage_balanced_sampling:
        train_stages = active_waypoint_index[train_indices]
        stage_counts = np.bincount(train_stages, minlength=3)
        if np.any(stage_counts == 0):
            raise ValueError(
                "stage-balanced-sampling requires all three stages in training"
            )
        sample_weights = 1.0 / stage_counts[train_stages]
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    best_validation_objective = float("inf")
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
            "converged_action_weight": args.converged_action_weight,
            "solver_consistency_weight": args.solver_consistency_weight,
            "converged_solver_iterations": args.converged_solver_iterations,
            "quadratic_log_scale": policy.actor.quadratic_log_scale,
            "linear_scale": policy.actor.linear_scale,
            "action_quadratic_scale": policy.actor.action_quadratic_scale,
            "stage_balanced_sampling": args.stage_balanced_sampling,
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
        best_validation_objective = float(
            resume_payload.get(
                "best_validation_objective",
                best_validation_mse,
            )
        )
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
        "converged_action_weight": args.converged_action_weight,
        "solver_consistency_weight": args.solver_consistency_weight,
        "converged_solver_iterations": args.converged_solver_iterations,
        "quadratic_log_scale": policy.actor.quadratic_log_scale,
        "linear_scale": policy.actor.linear_scale,
        "action_quadratic_scale": policy.actor.action_quadratic_scale,
        "stage_balanced_sampling": args.stage_balanced_sampling,
        "train_stage_counts": np.bincount(
            active_waypoint_index[train_indices], minlength=3
        ).tolist(),
        "validation_stage_counts": np.bincount(
            active_waypoint_index[validation_indices], minlength=3
        ).tolist(),
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
            train_converged_action_sum = 0.0
            train_solver_consistency_sum = 0.0
            gradient_norm_sum = 0.0
            batches = 0
            for (
                batch_observations,
                batch_targets,
                batch_future,
                batch_future_mask,
                _,
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
                converged_action_loss = output_actor.action.sum() * 0.0
                solver_consistency_loss = output_actor.action.sum() * 0.0
                if (
                    args.converged_action_weight > 0
                    or args.solver_consistency_weight > 0
                ):
                    converged_sequence, _ = _converged_sequence(
                        policy.actor,
                        output_actor,
                        args.converged_solver_iterations,
                    )
                    converged_action = converged_sequence[:, 0]
                    converged_action_loss = (
                        converged_action - batch_targets
                    ).square().mean()
                    solver_consistency_loss = (
                        converged_action - output_actor.action
                    ).square().mean()
                    loss = (
                        loss
                        + args.converged_action_weight * converged_action_loss
                        + args.solver_consistency_weight
                        * solver_consistency_loss
                    )
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
                train_converged_action_sum += float(
                    converged_action_loss.detach()
                )
                train_solver_consistency_sum += float(
                    solver_consistency_loss.detach()
                )
                gradient_norm_sum += float(gradient_norm.detach())
                batches += 1
            validation = _evaluate(
                policy,
                validation_loader,
                device,
                sequence_weight=args.sequence_weight,
                converged_action_weight=args.converged_action_weight,
                solver_consistency_weight=args.solver_consistency_weight,
                converged_solver_iterations=args.converged_solver_iterations,
            )
            elapsed = elapsed_before + time.monotonic() - started
            row = {
                "epoch": epoch,
                "elapsed_seconds": elapsed,
                "train_mse": train_squared_error / train_elements,
                "train_rmse": (train_squared_error / train_elements) ** 0.5,
                "train_loss": train_loss_sum / batches,
                "train_converged_action_mse": (
                    train_converged_action_sum / batches
                ),
                "train_solver_consistency_mse": (
                    train_solver_consistency_sum / batches
                ),
                "gradient_norm": gradient_norm_sum / batches,
                **{f"validation_{key}": value for key, value in validation.items()},
            }
            selection_metric = (
                "stage_macro_objective"
                if args.stage_balanced_sampling
                else "objective"
            )
            selection_objective = validation[selection_metric]
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
                "best_validation_objective": min(
                    best_validation_objective,
                    selection_objective,
                ),
                "validation_selection_objective": selection_objective,
                "validation_selection_metric": selection_metric,
                "elapsed_seconds": elapsed,
                "last_report": row,
            }
            _save(output / "last.pt", payload)
            best_validation_mse = min(best_validation_mse, validation["mse"])
            if selection_objective < best_validation_objective:
                best_validation_objective = selection_objective
                payload["best_validation_objective"] = best_validation_objective
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
            "best_validation_objective": best_validation_objective,
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
