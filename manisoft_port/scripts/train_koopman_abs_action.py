#!/usr/bin/env python
"""Train a Koopman model on absolute-action dynamics.

The model learns ``z_{t+1} = A z_t + B u_t`` where
  - the Koopman state is the raw 45-D three-section physical state ``s_t``
    (NO previous-action block), and
  - the control input is the absolute 18-D action ``u_t`` (not delta).

Everything else mirrors ``train_koopman.py``: DeepKoopman architecture, the
same fullA_history_v2_adapted loss, normalizer, checkpoints, W&B and history
logging.  The episodes are loaded directly from the collector's
``episode_*.npz`` files (``state``/``action``/``next_state``, action already
absolute) so no delta-action build step is needed.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_AC_MPC_SCRIPTS = Path(__file__).resolve().parent
if str(_AC_MPC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_AC_MPC_SCRIPTS))

from antmaze_ac.config import load_config
from antmaze_ac.data.build_sequences import Normalizer
from antmaze_ac.koopman.checkpoint import load_checkpoint, save_checkpoint, sha256
from antmaze_ac.koopman.losses import koopman_loss
from antmaze_ac.koopman.model import DeepKoopman
from train_koopman import (
    average_loss,
    capture_rng_state,
    initialize_wandb,
    reconcile_history,
    restore_rng_state,
    set_seed,
)


def discover_episodes(roots: Sequence[str | Path]) -> list[Path]:
    """Return every episode_*.npz under the given root directories (sorted)."""

    if not roots:
        raise ValueError("At least one --input-root is required")
    paths: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Input root is not a directory: {root}")
        paths.extend(sorted(root.glob("episode_*.npz")))
    if not paths:
        raise ValueError("No episode_*.npz files found in any input root")
    return sorted(paths, key=lambda p: str(p))


def load_all(episode_paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate (state, action, next_state) with episode ids from NPZ files."""

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    for episode_id, path in enumerate(episode_paths):
        with np.load(path, allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=np.float32)
            action = np.asarray(archive["action"], dtype=np.float32)
            next_state = np.asarray(archive["next_state"], dtype=np.float32)
        if state.ndim != 2 or action.ndim != 2 or next_state.ndim != 2:
            raise ValueError(f"{path}: expected 2-D arrays")
        if state.shape != next_state.shape or len(action) != len(state):
            raise ValueError(f"{path}: inconsistent shapes {state.shape} {action.shape}")
        if not (np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(next_state).all()):
            raise ValueError(f"{path}: contains NaN or Inf")
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        episode_ids.append(np.full(len(state), episode_id, dtype=np.int64))
    return (
        np.concatenate(states, axis=0),
        np.concatenate(actions, axis=0),
        np.concatenate(next_states, axis=0),
        np.concatenate(episode_ids, axis=0),
    )


def split_episode_ids(
    episode_ids: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split rows by whole episode using a fixed seed."""

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    unique = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    n = len(order)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    train_ep = set(order[:n_train].tolist())
    val_ep = set(order[n_train:n_train + n_val].tolist())
    test_ep = set(order[n_train + n_val:].tolist())
    train = np.isin(episode_ids, list(train_ep))
    val = np.isin(episode_ids, list(val_ep))
    test = np.isin(episode_ids, list(test_ep))
    if not (train.any() and val.any() and test.any()):
        raise ValueError("split produced an empty partition")
    return train, val, test


def valid_window_starts(
    states: np.ndarray,
    episode_ids: np.ndarray,
    transitions: int,
    step_index: np.ndarray,
) -> np.ndarray:
    """Start indices that keep a K-transition window inside one episode.

    Vectorised: avoids the O(N) Python loop over the full dataset.
    """

    n = len(states)
    if n <= transitions:
        return np.asarray([], dtype=np.int64)
    starts = np.arange(n - transitions, dtype=np.int64)
    stops = starts + transitions
    valid = (
        (episode_ids[starts] == episode_ids[stops])
        & (step_index[stops] == step_index[starts] + transitions)
    )
    return starts[valid]


class AbsActionWindowDataset(Dataset):
    """Window of normalized states (K+1, nx) and absolute actions (K, nu)."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ids: np.ndarray,
        step_index: np.ndarray,
        normalizer: Normalizer,
        transitions: int,
        starts: np.ndarray | None = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.normalizer = normalizer
        self.transitions = int(transitions)
        self.starts = (
            valid_window_starts(states, episode_ids, transitions, step_index)
            if starts is None
            else starts
        )
        if not len(self.starts):
            raise ValueError(f"No valid {transitions}-transition windows")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        stop = start + self.transitions
        states = self.states[start : stop + 1]
        states = self.normalizer.normalize(states).astype(np.float32)
        actions = self.actions[start:stop].astype(np.float32)  # absolute u_t
        return torch.from_numpy(states), torch.from_numpy(actions)


def build_loss_kwargs(koopman_config: dict) -> dict:
    weights = koopman_config.get("loss_weights", {})
    return {
        "rollout_discount": float(koopman_config["rollout_discount"]),
        "linear_weight": float(weights.get("linear", 10.0)),
        "rollout_weight": float(weights.get("rollout", 1.0)),
        "stability_weight": float(weights.get("stability", 0.01)),
        "latent_std_weight": float(weights.get("latent_std", 0.1)),
        "identity_weight": float(weights.get("identity", 1e-4)),
        "controllability_svd_weight": float(weights.get("controllability_svd", 0.0)),
        "augmentation_weight": float(weights.get("augmentation", 0.0)),
        "reconstruction_weight": float(weights.get("reconstruction", 1.0)),
        "spectral_radius_limit": float(koopman_config["spectral_radius_limit"]),
        "target_latent_std": float(koopman_config["target_latent_std"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/manisoft_coll.yaml")
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        help="Episode directory (repeats); the collector's worker_XX dirs.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-validation-windows", type=int, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    koopman_config = config["koopman"]
    set_seed(config["experiment"]["seed"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    # ---- load episodes and split ----
    episode_paths = discover_episodes(args.input_root)
    print(f"loading {len(episode_paths)} episodes ...", flush=True)
    states, actions, next_states, episode_ids = load_all(episode_paths)
    step_index = np.concatenate(
        [
            np.arange(int((episode_ids == e).sum()), dtype=np.int64)
            for e in np.unique(episode_ids)
        ]
    )
    action_dim = actions.shape[1]
    state_dim = states.shape[1]
    print(f"transitions={len(states)} state_dim={state_dim} action_dim={action_dim}", flush=True)

    fractions = (
        config["data"]["train_fraction"],
        config["data"]["validation_fraction"],
        config["data"]["test_fraction"],
    )
    split_seed = int(config["experiment"]["seed"])
    train_mask, val_mask, test_mask = split_episode_ids(episode_ids, fractions, split_seed)
    del episode_paths
    gc.collect()

    normalizer = Normalizer.fit(
        states[train_mask],
        epsilon=float(config["data"]["normalization_epsilon"]),
    )
    k_step = int(koopman_config["K_step"])

    train_windows = AbsActionWindowDataset(
        states[train_mask], actions[train_mask],
        episode_ids[train_mask], step_index[train_mask],
        normalizer, k_step,
    )
    validation_windows = AbsActionWindowDataset(
        states[val_mask], actions[val_mask],
        episode_ids[val_mask], step_index[val_mask],
        normalizer, k_step,
    )
    if args.max_train_windows:
        train_windows.starts = train_windows.starts[: args.max_train_windows]
    if args.max_validation_windows:
        validation_windows.starts = validation_windows.starts[: args.max_validation_windows]
    train_loader = DataLoader(
        train_windows,
        batch_size=koopman_config["batch_size"],
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_windows,
        batch_size=koopman_config["eval_batch_size"],
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    print(
        f"train windows={len(train_windows)} val windows={len(validation_windows)}",
        flush=True,
    )

    # ---- model (no action integrator: pure [s_t] + absolute u_t) ----
    model = DeepKoopman(
        state_dim,
        action_dim,
        koopman_config["lift_dim"],
        koopman_config["encoder_hidden_dims"],
        koopman_config["encoder_activation"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=koopman_config["learning_rate"],
        weight_decay=koopman_config["weight_decay"],
    )
    loss_kwargs = build_loss_kwargs(koopman_config)

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    if history_path.exists():
        raise FileExistsError(f"{history_path} already exists; use a fresh output")
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    wandb_run = initialize_wandb(config, output, args.wandb_mode)

    max_epochs = args.max_epochs or int(koopman_config["max_epochs"])
    max_hours = args.max_wall_time_hours or float(koopman_config["max_wall_time_hours"])
    wall_limit = max_hours * 3600
    started = time.monotonic()
    best_validation = float("inf")
    best_epoch = -1
    stop_reason = "max_epochs"
    last_epoch = -1
    normalizers = {"state": normalizer.state_dict(), "action": "absolute_physical_units"}

    for epoch in range(max_epochs):
        model.train()
        train_sums: dict[str, float] = {}
        train_count = 0
        gradient_sums = {
            "gradient_norm_before_clip": 0.0,
            "A_gradient_norm": 0.0,
            "B_gradient_norm": 0.0,
            "encoder_gradient_norm": 0.0,
        }
        gradient_batches = 0
        for states_batch, actions_batch in train_loader:
            states_batch, actions_batch = states_batch.to(device), actions_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = koopman_loss(model, states_batch, actions_batch, **loss_kwargs)
            losses.total.backward()
            encoder_squared = torch.zeros((), device=device)
            for parameter in model.encoder.parameters():
                if parameter.grad is not None:
                    encoder_squared = encoder_squared + parameter.grad.detach().square().sum()
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
                raise FloatingPointError("Koopman parameter became NaN or Inf after optimizer step")
            batch = len(states_batch)
            train_count += batch
            for key, value in losses.scalars().items():
                train_sums[key] = train_sums.get(key, 0.0) + value * batch
        train_metrics = {key: value / train_count for key, value in train_sums.items()}
        train_metrics.update(
            {key: value / max(gradient_batches, 1) for key, value in gradient_sums.items()}
        )
        validation_metrics = average_loss(model, validation_loader, device, loss_kwargs)
        elapsed = time.monotonic() - started
        row = {
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "train": train_metrics,
            "validation": validation_metrics,
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
                },
                step=epoch,
            )
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
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

    save_checkpoint(
        output / "last.pt",
        model,
        optimizer=optimizer,
        epoch=last_epoch,
        best_validation=best_validation,
        config=config,
        normalizers=normalizers,
        elapsed_seconds=time.monotonic() - started,
        rng_state=capture_rng_state(),
    )
    if not (output / "best_validation.pt").exists():
        raise RuntimeError("Training ended without producing best_validation.pt")
    status = {
        "architecture": model.architecture(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "transition_semantics": {
            "state": "45-D three-section physical state s_t (no prev-action block)",
            "action": "absolute 18-D action u_t",
            "dynamics": "z_{t+1} = A z_t + B u_t",
        },
        "transitions": len(states),
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "stop_reason": stop_reason,
        "last_epoch": last_epoch,
        "best_checkpoint_sha256": sha256(output / "best_validation.pt"),
    }
    (output / "training_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
