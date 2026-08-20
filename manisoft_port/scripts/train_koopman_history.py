#!/usr/bin/env python
"""Train AC-MPC Koopman dynamics with finite history and absolute actions.

This is the history-only variant of ``scripts/train_koopman.py`` for raw
ManiSoft episodes. Its transition convention is

    z[t+1] = A z[t] + B u[t]

where the physical state is the raw 45-D ``s[t]`` (no previous-action block),
``u[t]`` is the raw 18-D absolute action, and the lifting encoder sees
``[s[t-H+1:t+1], u[t-H:t]]``. Thus current ``u[t]`` and future ``s[t+1]`` do
not leak into the lift at time ``t``. At an episode start, missing states are
filled with repeated ``s[0]`` and missing past actions are filled with zeros.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from antmaze_ac.config import load_config
from antmaze_ac.data.action_rate_sampling import (
    ActionRateStratifiedSampler,
    rate_bin_indices,
    window_action_rates,
)
from antmaze_ac.data.build_sequences import Normalizer
from antmaze_ac.data.history_windows import AbsoluteActionHistoryWindowDataset
from antmaze_ac.koopman.checkpoint import load_checkpoint, save_checkpoint, sha256
from antmaze_ac.koopman.history_losses import history_koopman_loss
from antmaze_ac.koopman.history_model import HistoryDeepKoopman
from train_koopman import (
    capture_rng_state,
    initialize_wandb,
    reconcile_history,
    restore_rng_state,
    set_seed,
)


def discover_episodes(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")
    paths = sorted(root.glob("episode_*.npz")) + sorted(root.glob("worker_*/episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz files found under {root}")
    return sorted(paths, key=lambda path: str(path.relative_to(root)))


def load_episodes(
    paths: list[Path],
    path_source_ids: list[int],
    expected_state_dim: int,
    expected_action_dim: int,
    continuity_atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    if len(paths) != len(path_source_ids):
        raise ValueError("paths and path_source_ids must have matching lengths")
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    step_indices: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    max_continuity_error = 0.0
    lengths: list[int] = []

    for episode_id, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in ("state", "action", "next_state") if key not in archive.files]
            if missing:
                raise KeyError(f"{path} is missing fields: {missing}")
            state = np.asarray(archive["state"], dtype=np.float32)
            action = np.asarray(archive["action"], dtype=np.float32)
            next_state = np.asarray(archive["next_state"], dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != expected_state_dim:
            raise ValueError(f"{path}: state must have shape [T,{expected_state_dim}], got {state.shape}")
        if action.ndim != 2 or action.shape != (len(state), expected_action_dim):
            raise ValueError(
                f"{path}: action must have shape [{len(state)},{expected_action_dim}], got {action.shape}"
            )
        if next_state.shape != state.shape:
            raise ValueError(f"{path}: next_state shape {next_state.shape} != state shape {state.shape}")
        if not (np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(next_state).all()):
            raise ValueError(f"{path} contains NaN or Inf")
        continuity_error = 0.0
        if len(state) > 1:
            continuity_error = float(np.max(np.abs(next_state[:-1] - state[1:])))
            if continuity_error > continuity_atol:
                raise ValueError(
                    f"{path}: max |next_state[t]-state[t+1]|={continuity_error:.3e} "
                    f"exceeds {continuity_atol:.1e}"
                )
        max_continuity_error = max(max_continuity_error, continuity_error)
        states.append(state)
        actions.append(action)
        episode_ids.append(np.full(len(state), episode_id, dtype=np.int64))
        step_indices.append(np.arange(len(state), dtype=np.int64))
        source_ids.append(
            np.full(len(state), path_source_ids[episode_id], dtype=np.int64)
        )
        lengths.append(len(state))
        if (episode_id + 1) % 50 == 0 or episode_id + 1 == len(paths):
            print(f"loaded {episode_id + 1}/{len(paths)} episodes", flush=True)

    diagnostics = {
        "episodes": len(paths),
        "transitions": int(sum(lengths)),
        "episode_len_min": int(min(lengths)),
        "episode_len_max": int(max(lengths)),
        "max_temporal_continuity_error": max_continuity_error,
    }
    return (
        np.concatenate(states, axis=0),
        np.concatenate(actions, axis=0),
        np.concatenate(episode_ids, axis=0),
        np.concatenate(step_indices, axis=0),
        np.concatenate(source_ids, axis=0),
        diagnostics,
    )


def split_episode_masks(
    episode_ids: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions_array = np.asarray(fractions, dtype=np.float64)
    fractions_array /= fractions_array.sum()
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_validation = max(1, int(round(len(episodes) * fractions_array[1])))
    n_test = max(1, int(round(len(episodes) * fractions_array[2])))
    n_train = len(episodes) - n_validation - n_test
    if n_train < 1:
        raise ValueError("Episode split leaves no training episodes")
    train_ids = episodes[:n_train]
    validation_ids = episodes[n_train : n_train + n_validation]
    test_ids = episodes[n_train + n_validation :]
    return tuple(np.isin(episode_ids, ids) for ids in (train_ids, validation_ids, test_ids))


def split_episode_masks_by_source(
    episode_ids: np.ndarray,
    source_ids: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split episodes independently within every input dataset root."""

    masks = [np.zeros(len(episode_ids), dtype=bool) for _ in range(3)]
    for source_id in np.unique(source_ids):
        source_mask = source_ids == source_id
        source_splits = split_episode_masks(
            episode_ids[source_mask], fractions, seed + int(source_id)
        )
        source_rows = np.flatnonzero(source_mask)
        for combined, source_split in zip(masks, source_splits):
            combined[source_rows[source_split]] = True
    return tuple(masks)


def rate_bin_labels(edges: list[float]) -> list[str]:
    labels = [f"le_{edges[0]:g}"]
    labels.extend(
        f"{left:g}_to_{right:g}"
        for left, right in zip(edges[:-1], edges[1:])
    )
    labels.append(f"gt_{edges[-1]:g}")
    return labels


def loss_kwargs_from_config(
    koopman_config: dict,
    state_std: np.ndarray,
) -> dict:
    weights = koopman_config["loss_weights"]
    return {
        "rollout_discount": koopman_config["rollout_discount"],
        "linear_weight": weights["linear"],
        "rollout_weight": weights["rollout"],
        "stability_weight": weights["stability"],
        "latent_std_weight": weights["latent_std"],
        "identity_weight": weights["identity"],
        "controllability_svd_weight": weights["controllability_svd"],
        "augmentation_weight": weights["augmentation"],
        "reconstruction_weight": weights["reconstruction"],
        "tip_position_weight": weights["tip_position"],
        "tip_position_slice": tuple(koopman_config["tip_position_slice"]),
        "state_std": torch.as_tensor(state_std, dtype=torch.float32),
        "spectral_radius_limit": koopman_config["spectral_radius_limit"],
        "target_latent_std": koopman_config["target_latent_std"],
    }


@torch.no_grad()
def average_loss(model, loader, device, loss_kwargs):
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for contexts, states, actions in loader:
        contexts, states, actions = contexts.to(device), states.to(device), actions.to(device)
        losses = history_koopman_loss(model, contexts, states, actions, **loss_kwargs)
        batch = len(states)
        count += batch
        for key, value in losses.scalars().items():
            totals[key] = totals.get(key, 0.0) + value * batch
    return {key: value / count for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/manisoft_coll.yaml")
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help=(
            "Dataset root containing episode_*.npz or worker_*/episode_*.npz. "
            "Repeat --data to combine smooth and rate-coverage collections."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-validation-windows", type=int, default=None)
    parser.add_argument(
        "--rate-stratified-sampling",
        action="store_true",
        help="Resample train windows by their peak history/rollout action rate each epoch.",
    )
    parser.add_argument(
        "--action-rate-bins",
        type=float,
        nargs="+",
        default=[0.005, 0.05],
        help="Strictly increasing max-|delta u| bin edges (default 0.005 0.05).",
    )
    parser.add_argument(
        "--action-rate-fractions",
        type=float,
        nargs="+",
        default=[0.50, 0.35, 0.15],
        help="Sampling/selection weights for rate bins (default 0.50 0.35 0.15).",
    )
    parser.add_argument(
        "--max-validation-windows-per-rate",
        type=int,
        default=20000,
        help="Maximum deterministic validation windows evaluated in each rate slice.",
    )
    parser.add_argument("--continuity-atol", type=float, default=1e-6)
    parser.add_argument(
        "--tip-position-weight",
        type=float,
        default=10000.0,
        help=(
            "Weight for physical-unit tip-position rollout MSE in m^2 "
            "(default 10000, so a 10 mm RMSE contributes about 1.0)."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default=None,
    )
    args = parser.parse_args()
    if args.history_steps < 1:
        parser.error("--history-steps must be positive")
    if args.continuity_atol < 0:
        parser.error("--continuity-atol must be non-negative")
    if args.tip_position_weight < 0:
        parser.error("--tip-position-weight must be non-negative")
    if len(args.action_rate_fractions) != len(args.action_rate_bins) + 1:
        parser.error("--action-rate-fractions must have one more value than --action-rate-bins")
    if any(value <= 0 for value in args.action_rate_bins) or any(
        right <= left
        for left, right in zip(args.action_rate_bins[:-1], args.action_rate_bins[1:])
    ):
        parser.error("--action-rate-bins must be positive and strictly increasing")
    if any(value < 0 for value in args.action_rate_fractions) or not sum(
        args.action_rate_fractions
    ):
        parser.error("--action-rate-fractions must be non-negative and sum to > 0")
    for name in (
        "max_train_windows",
        "max_validation_windows",
        "max_validation_windows_per_rate",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(load_config(args.config))
    koopman_config = config["koopman"]
    config["koopman"]["architecture"] = "fullA_history_context_v1"
    config["koopman"]["history_steps"] = int(args.history_steps)
    config["koopman"]["exact_action_integrator"] = False
    config["koopman"]["tip_position_slice"] = [30, 33]
    config["koopman"]["loss_weights"]["tip_position"] = float(
        args.tip_position_weight
    )
    config["data"]["state_semantics"] = "raw_45d_state_without_previous_action"
    config["data"]["action_semantics"] = "absolute_action"
    set_seed(config["experiment"]["seed"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    data_roots = [Path(root).expanduser().resolve() for root in args.data]
    paths: list[Path] = []
    path_source_ids: list[int] = []
    seen_paths: set[Path] = set()
    source_episode_counts: dict[str, int] = {}
    for source_id, root in enumerate(data_roots):
        source_paths = discover_episodes(root)
        duplicates = seen_paths.intersection(source_paths)
        if duplicates:
            raise ValueError(
                f"dataset roots overlap; duplicate episode {next(iter(duplicates))}"
            )
        seen_paths.update(source_paths)
        paths.extend(source_paths)
        path_source_ids.extend([source_id] * len(source_paths))
        source_episode_counts[str(root)] = len(source_paths)
    expected_state_dim = int(config["experiment"]["expected_observation_dim"])
    expected_action_dim = int(config["experiment"]["expected_action_dim"])
    (
        states,
        actions,
        episode_ids,
        step_indices,
        source_ids,
        diagnostics,
    ) = load_episodes(
        paths,
        path_source_ids,
        expected_state_dim,
        expected_action_dim,
        args.continuity_atol,
    )
    diagnostics["sources"] = source_episode_counts
    fractions = (
        config["data"]["train_fraction"],
        config["data"]["validation_fraction"],
        config["data"]["test_fraction"],
    )
    train_mask, validation_mask, test_mask = split_episode_masks_by_source(
        episode_ids,
        source_ids,
        fractions,
        int(config["experiment"]["seed"]),
    )
    normalizer = Normalizer.fit(
        states[train_mask],
        epsilon=float(config["data"]["normalization_epsilon"]),
    )
    k_step = int(koopman_config["K_step"])
    train_windows = AbsoluteActionHistoryWindowDataset(
        states[train_mask],
        actions[train_mask],
        episode_ids[train_mask],
        step_indices[train_mask],
        normalizer,
        k_step,
        args.history_steps,
    )
    validation_windows = AbsoluteActionHistoryWindowDataset(
        states[validation_mask],
        actions[validation_mask],
        episode_ids[validation_mask],
        step_indices[validation_mask],
        normalizer,
        k_step,
        args.history_steps,
    )
    split_summary = {
        "train_episodes": int(np.unique(episode_ids[train_mask]).size),
        "validation_episodes": int(np.unique(episode_ids[validation_mask]).size),
        "test_episodes": int(np.unique(episode_ids[test_mask]).size),
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
    }
    train_window_rates = window_action_rates(
        train_windows.actions,
        train_windows.episode_ids,
        train_windows.starts,
        k_step,
        args.history_steps,
    )
    validation_window_rates = window_action_rates(
        validation_windows.actions,
        validation_windows.episode_ids,
        validation_windows.starts,
        k_step,
        args.history_steps,
    )
    rate_edges = [float(value) for value in args.action_rate_bins]
    rate_fractions = np.asarray(args.action_rate_fractions, dtype=np.float64)
    rate_fractions /= rate_fractions.sum()
    labels = rate_bin_labels(rate_edges)
    train_rate_bins = rate_bin_indices(train_window_rates, rate_edges)
    validation_rate_bins = rate_bin_indices(validation_window_rates, rate_edges)
    split_summary["train_windows_by_action_rate"] = {
        label: int(np.sum(train_rate_bins == index))
        for index, label in enumerate(labels)
    }
    split_summary["validation_windows_by_action_rate"] = {
        label: int(np.sum(validation_rate_bins == index))
        for index, label in enumerate(labels)
    }
    config["data"]["action_rate_sampling"] = {
        "enabled": bool(args.rate_stratified_sampling),
        "definition": (
            "peak max-component |u[t]-u[t-1]| over the history and K-step "
            "rollout window, with zero previous action at episode boundaries"
        ),
        "bin_edges": rate_edges,
        "fractions": rate_fractions.tolist(),
        "train_windows_per_epoch": (
            args.max_train_windows or len(train_windows)
        ),
        "validation_selection": (
            "rate_weighted" if args.rate_stratified_sampling else "natural"
        ),
    }

    seed = int(config["experiment"]["seed"])
    train_sampler = None
    if args.rate_stratified_sampling:
        train_sampler = ActionRateStratifiedSampler(
            train_window_rates,
            rate_edges,
            rate_fractions,
            args.max_train_windows or len(train_windows),
            seed,
        )
    elif args.max_train_windows and args.max_train_windows < len(train_windows):
        subset_rng = np.random.default_rng(seed)
        selected = subset_rng.choice(
            len(train_windows), size=args.max_train_windows, replace=False
        )
        train_windows.starts = train_windows.starts[selected]

    validation_slice_loaders: dict[str, DataLoader] = {}
    validation_slice_weights: dict[str, float] = {}
    for bin_index, label in enumerate(labels):
        candidate_indices = np.flatnonzero(validation_rate_bins == bin_index)
        if not len(candidate_indices):
            continue
        slice_rng = np.random.default_rng(seed + 1000 + bin_index)
        if len(candidate_indices) > args.max_validation_windows_per_rate:
            candidate_indices = slice_rng.choice(
                candidate_indices,
                size=args.max_validation_windows_per_rate,
                replace=False,
            )
        slice_windows = copy.copy(validation_windows)
        slice_windows.starts = validation_windows.starts[candidate_indices]
        validation_slice_loaders[label] = DataLoader(
            slice_windows,
            batch_size=koopman_config["eval_batch_size"],
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        validation_slice_weights[label] = float(rate_fractions[bin_index])

    if args.max_validation_windows and args.max_validation_windows < len(validation_windows):
        validation_rng = np.random.default_rng(seed + 500)
        selected = validation_rng.choice(
            len(validation_windows),
            size=args.max_validation_windows,
            replace=False,
        )
        validation_windows.starts = validation_windows.starts[selected]

    del (
        paths,
        path_source_ids,
        states,
        actions,
        episode_ids,
        step_indices,
        source_ids,
        train_mask,
        validation_mask,
        test_mask,
        train_window_rates,
        validation_window_rates,
        train_rate_bins,
        validation_rate_bins,
    )
    gc.collect()

    train_loader = DataLoader(
        train_windows,
        batch_size=koopman_config["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_windows,
        batch_size=koopman_config["eval_batch_size"],
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = HistoryDeepKoopman(
        expected_state_dim,
        expected_action_dim,
        koopman_config["lift_dim"],
        koopman_config["encoder_hidden_dims"],
        koopman_config["encoder_activation"],
        args.history_steps,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=koopman_config["learning_rate"],
        weight_decay=koopman_config["weight_decay"],
    )

    output = Path(args.output or config["experiment"]["output_dir"]) / "koopman_history"
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    if args.resume is None and history_path.exists():
        raise FileExistsError(f"{history_path} already exists; use a fresh output or pass --resume")
    (output / "resolved_config.json").write_text(
        json.dumps(
            {
                **config,
                "dataset": {
                    **diagnostics,
                    **split_summary,
                    "roots": [str(root) for root in data_roots],
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    wandb_run = initialize_wandb(config, output, args.wandb_mode)

    start_epoch = 0
    best_validation = float("inf")
    elapsed_before = 0.0
    history_best_epoch = None
    if args.resume:
        loaded_model, payload = load_checkpoint(args.resume, map_location=device)
        if not isinstance(loaded_model, HistoryDeepKoopman):
            raise ValueError("Resume checkpoint is not a history Koopman model")
        if loaded_model.history_steps != args.history_steps:
            raise ValueError("Resume checkpoint history_steps does not match")
        model = loaded_model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=koopman_config["learning_rate"],
            weight_decay=koopman_config["weight_decay"],
        )
        if payload["optimizer"] is not None:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_validation = float(payload["best_validation"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        if payload.get("rng_state") is not None:
            restore_rng_state(payload["rng_state"])
        history_best_epoch, _ = reconcile_history(history_path, start_epoch)

    loss_kwargs = loss_kwargs_from_config(koopman_config, normalizer.std)
    max_epochs = args.max_epochs or int(koopman_config["max_epochs"])
    max_hours = args.max_wall_time_hours or float(koopman_config["max_wall_time_hours"])
    wall_limit = max_hours * 3600.0
    started = time.monotonic()
    best_epoch = int(history_best_epoch) if history_best_epoch is not None else start_epoch - 1
    last_epoch = start_epoch - 1
    stop_reason = "max_epochs"
    normalizers = {"state": normalizer.state_dict(), "action": "absolute_physical_units"}

    for epoch in range(start_epoch, max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_sums: dict[str, float] = {}
        gradient_sums = {
            "gradient_norm_before_clip": 0.0,
            "A_gradient_norm": 0.0,
            "B_gradient_norm": 0.0,
            "encoder_gradient_norm": 0.0,
        }
        train_count = 0
        gradient_batches = 0
        for contexts, batch_states, batch_actions in train_loader:
            contexts = contexts.to(device)
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = history_koopman_loss(
                model,
                contexts,
                batch_states,
                batch_actions,
                **loss_kwargs,
            )
            losses.total.backward()
            encoder_squared = torch.zeros((), device=device)
            for parameter in model.encoder.parameters():
                if parameter.grad is not None:
                    encoder_squared += parameter.grad.detach().square().sum()
            gradient_sums["A_gradient_norm"] += float(model.A.grad.detach().norm())
            gradient_sums["B_gradient_norm"] += float(model.B.grad.detach().norm())
            gradient_sums["encoder_gradient_norm"] += float(torch.sqrt(encoder_squared))
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), koopman_config["gradient_clip_norm"]
            )
            gradient_sums["gradient_norm_before_clip"] += float(gradient_norm)
            gradient_batches += 1
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite Koopman gradient")
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("Koopman parameter became NaN or Inf")
            batch = len(batch_states)
            train_count += batch
            for key, value in losses.scalars().items():
                train_sums[key] = train_sums.get(key, 0.0) + value * batch
        train_metrics = {key: value / train_count for key, value in train_sums.items()}
        train_metrics.update(
            {key: value / max(gradient_batches, 1) for key, value in gradient_sums.items()}
        )
        validation_metrics = average_loss(model, validation_loader, device, loss_kwargs)
        validation_by_action_rate = {
            label: average_loss(model, loader, device, loss_kwargs)
            for label, loader in validation_slice_loaders.items()
        }
        if args.rate_stratified_sampling:
            available_weight = sum(
                validation_slice_weights[label]
                for label in validation_by_action_rate
            )
            if available_weight > 0:
                selection_validation = sum(
                    validation_slice_weights[label] * metrics["total"]
                    for label, metrics in validation_by_action_rate.items()
                ) / available_weight
                selection_tip_position = sum(
                    validation_slice_weights[label] * metrics["tip_position"]
                    for label, metrics in validation_by_action_rate.items()
                ) / available_weight
            else:
                selection_validation = float(
                    np.mean(
                        [
                            metrics["total"]
                            for metrics in validation_by_action_rate.values()
                        ]
                    )
                )
                selection_tip_position = float(
                    np.mean(
                        [
                            metrics["tip_position"]
                            for metrics in validation_by_action_rate.values()
                        ]
                    )
                )
        else:
            selection_validation = validation_metrics["total"]
            selection_tip_position = validation_metrics["tip_position"]
        elapsed = elapsed_before + time.monotonic() - started
        row = {
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_by_action_rate": validation_by_action_rate,
            "selection_validation": selection_validation,
            "selection_tip_position_rmse_mm": (
                max(selection_tip_position, 0.0) ** 0.5 * 1000.0
            ),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "elapsed_seconds": elapsed,
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"validation/{key}": value for key, value in validation_metrics.items()},
                    **{
                        f"validation_rate/{label}/{key}": value
                        for label, metrics in validation_by_action_rate.items()
                        for key, value in metrics.items()
                    },
                    "validation/selection": selection_validation,
                },
                step=epoch,
            )
        if selection_validation < best_validation:
            best_validation = selection_validation
            best_epoch = epoch
            save_checkpoint(
                output / "best_validation.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation=best_validation,
                config=config,
                normalizers=normalizers,
                elapsed_seconds=elapsed,
                rng_state=capture_rng_state(),
            )
        if (epoch + 1) % int(koopman_config["checkpoint_interval"]) == 0:
            save_checkpoint(
                output / f"recovery_epoch_{epoch:04d}.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation=best_validation,
                config=config,
                normalizers=normalizers,
                elapsed_seconds=elapsed,
                rng_state=capture_rng_state(),
            )
        last_epoch = epoch
        print(json.dumps(row, sort_keys=True), flush=True)
        if elapsed >= wall_limit:
            stop_reason = "max_wall_time"
            break

    elapsed = elapsed_before + time.monotonic() - started
    save_checkpoint(
        output / "last.pt",
        model,
        optimizer=optimizer,
        epoch=last_epoch,
        best_validation=best_validation,
        config=config,
        normalizers=normalizers,
        elapsed_seconds=elapsed,
        rng_state=capture_rng_state(),
    )
    if not (output / "best_validation.pt").exists():
        raise RuntimeError("Training ended without producing best_validation.pt")
    status = {
        "actual_epochs_this_run": max(0, last_epoch - start_epoch + 1),
        "last_epoch": last_epoch,
        "elapsed_seconds_total": elapsed,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "best_validation_metric": (
            "rate_weighted_total_including_physical_tip_loss"
            if args.rate_stratified_sampling
            else "natural_total_including_physical_tip_loss"
        ),
        "tip_position_weight": float(args.tip_position_weight),
        "tip_position_slice": [30, 33],
        "stop_reason": stop_reason,
        "best_checkpoint_sha256": sha256(output / "best_validation.pt"),
        "last_checkpoint_sha256": sha256(output / "last.pt"),
        "device": str(device),
        "batch_size": int(koopman_config["batch_size"]),
        "eval_batch_size": int(koopman_config["eval_batch_size"]),
        "architecture": "fullA_history_context_v1",
        "state_semantics": "s_t",
        "action_semantics": "absolute_u_t",
        "history_steps": int(args.history_steps),
    }
    (output / "training_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update(status)
        wandb_run.finish()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
