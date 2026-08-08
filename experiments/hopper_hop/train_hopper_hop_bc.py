"""Train and evaluate the four HopperHop BC actors (PPO / KLQR / AB-PQ / BC-KMPC).

Behavior-cloning pretraining on expert transitions rolled out from the trained
50M-step PPO policies (``collect_hopperhop_expert.py``), followed by a
closed-loop evaluation in MS-HopperHop.

Actors (all cost-map / controller heads reuse the frozen global Koopman model
``A,B,C`` from ``runs/hopper_hop/koopman_v2/best.pt``):

    PPO      : standard Gaussian-mean MLP on the RAW 15-dim state (the same
               convention as the PPO baseline trainer, so BC weights transfer
               to PPO fine-tuning and the baseline comparison is apples-to-
               apples). No Koopman lift.
    KLQR     : cost-map (lift) -> Q_diag, p -> differentiable DARE ->
               time-varying closed-loop gain (replaces the H1-min series).
    AB-PQ    : low-rank quadratic value head greedified through A,B (AB-PQ).
    BC-KMPC  : cost-map (lift) -> finite-horizon box-QP Koopman MPC.

HopperHop has no task context (pure locomotion), so ``context_dim=0``
everywhere.  Actions are the normalized ``pd_joint_delta_pos`` in [-1, 1]
(``action_limit = 1.0``).

Closed-loop evaluation runs the real MS-HopperHop-v1 with
``reward_mode=normalized_dense`` (per-step max 1.0, episode 600 steps, so the
maximum episode return is 600) and additionally reports the mean standing and
hopping reward components separately for diagnosis.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from antmaze_ac.koopman.model import DeepKoopman
from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor
from antmaze_ac.rl.quadratic_actors import KoopmanLQRActor, LowRankValueActor

STATE_DIM = 15
ACTION_DIM = 4
STAND_HEIGHT = 0.6
HOP_SPEED = 2.0
MAX_EPISODE_STEPS = 600


@dataclass(frozen=True)
class BCConfig:
    epochs: int = 250
    batch_size: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    hidden_dim: int = 128
    ppo_hidden_dim: int = 256
    ab_rank: int = 4
    kmpc_horizon: int = 10
    kmpc_solver_iterations: int = 20
    kmpc_sequence_weight: float = 0.25
    action_limit: float = 1.0
    seed: int = 47
    validation_interval: int = 5
    early_stopping_patience: int = 50
    checkpoint_interval: int = 10
    evaluation_episodes: int = 64
    evaluation_num_envs: int = 64
    evaluation_seed_start: int = 20_270_804


def _orthogonal_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class StandardPPOActor(nn.Module):
    """Standard continuous-PPO Gaussian-mean MLP (no Koopman lift).

    Input is the RAW 15-dim HopperHop state (same convention as the PPO
    baseline trainer).  Linear mean scaled by ``action_limit``; the env clips
    the sampled action to [-1, 1].
    """

    def __init__(self, input_dim: int, hidden_dim: int, action_limit: float):
        super().__init__()
        self.action_limit = float(action_limit)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )
        _orthogonal_linear(self.network[0], math.sqrt(2.0))
        _orthogonal_linear(self.network[2], math.sqrt(2.0))
        _orthogonal_linear(self.network[4], 0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_limit * self.network(state)


BC_ACTOR_ORDER = ("PPO", "KLQR", "AB-PQ", "BC-KMPC")


def load_koopman(path: Path, device: torch.device) -> tuple[DeepKoopman, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint["architecture"]
    model = DeepKoopman(
        state_dim=int(architecture["state_dim"]),
        action_dim=int(architecture["action_dim"]),
        lift_dim=int(architecture["lift_dim"]),
        hidden_dims=tuple(architecture["hidden_dims"]),
        activation=str(architecture.get("activation", "silu")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.freeze_dynamics()
    if checkpoint.get("state_kind") != "hopperhop":
        raise ValueError("BC requires a hopperhop Koopman checkpoint")
    return model, checkpoint


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {
        "state", "action", "episode_id",
        "train_episode_ids", "validation_episode_ids",
    }
    missing = required - data.keys()
    if missing:
        raise KeyError(f"Dataset is missing fields: {sorted(missing)}")
    if data["state"].shape[1:] != (STATE_DIM,) or \
            data["action"].shape[1:] != (ACTION_DIM,):
        raise ValueError(
            f"Expected state [N,{STATE_DIM}] action [N,{ACTION_DIM}]"
        )
    if not (np.isfinite(data["state"]).all() and np.isfinite(data["action"]).all()):
        raise FloatingPointError("Dataset contains NaN or Inf")
    for split in ("train", "validation"):
        ids = data[f"{split}_episode_ids"]
        if len(ids) == 0:
            raise ValueError(f"{split} episode split is empty")
    return data


def _normalizers(checkpoint: dict) -> tuple[np.ndarray, np.ndarray]:
    def _cpu_array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    center = _cpu_array(checkpoint["normalizer"]["center"])
    scale = _cpu_array(checkpoint["normalizer"]["scale"])
    if center.shape != (STATE_DIM,) or scale.shape != (STATE_DIM,):
        raise ValueError("Koopman normalizer is not 15-dimensional")
    return center, scale


def _build_future_actions(
    data: dict[str, np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """[N, H, 4] future action windows + [N, H] validity mask (same-episode)."""
    count = len(data["action"])
    future = np.zeros((count, horizon, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((count, horizon), dtype=np.float32)
    episode = data["episode_id"]
    for episode_id in np.unique(episode):
        indices = np.flatnonzero(episode == episode_id)
        actions = data["action"][indices]
        length = len(indices)
        for offset in range(horizon):
            if offset >= length:
                break
            future[indices[: length - offset], offset] = actions[offset:]
            mask[indices[: length - offset], offset] = 1.0
    return future, mask


def _split_arrays(
    data: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    splits: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation"):
        ids = data[f"{split}_episode_ids"]
        mask = np.isin(data["episode_id"], ids)
        splits[split] = {
            "state": data["state"][mask],
            "action": data["action"][mask],
            "episode_id": data["episode_id"][mask],
        }
    return splits


def _actor_action(
    name: str,
    actor: nn.Module,
    normalized_state: torch.Tensor,
    lifted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if name == "PPO":
        # raw-state baseline (the BC actor was trained on the raw state)
        return actor(normalized_state), None
    if name == "KLQR":
        return actor(lifted).action, None
    if name == "AB-PQ":
        return actor(lifted, lifted).action, None
    output = actor(lifted)
    return output.action, output.action_sequence


def _evaluate_mse(
    name: str,
    actor: nn.Module,
    koopman: DeepKoopman,
    loader: DataLoader,
    device: torch.device,
) -> float:
    total = 0.0
    elements = 0
    actor.eval()
    with torch.no_grad():
        for state, action, _future, _future_mask in loader:
            state, action = state.to(device), action.to(device)
            lifted = koopman.lift(state)
            prediction, _ = _actor_action(name, actor, state, lifted)
            total += float((prediction - action).square().sum())
            elements += action.numel()
    return total / elements


def _make_loader(
    state: np.ndarray,
    action: np.ndarray,
    future: np.ndarray,
    future_mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(state),
        torch.from_numpy(action),
        torch.from_numpy(future),
        torch.from_numpy(future_mask),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _train_actor(
    name: str,
    actor: nn.Module,
    koopman: DeepKoopman,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: BCConfig,
    device: torch.device,
    checkpoint_payload_base: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    last_improvement_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        actor.train()
        for state, action, future, future_mask in train_loader:
            state, action = state.to(device), action.to(device)
            future, future_mask = future.to(device), future_mask.to(device)
            with torch.no_grad():
                lifted = koopman.lift(state)
            prediction, sequence = _actor_action(name, actor, state, lifted)
            loss = (prediction - action).square().mean()
            if sequence is not None and sequence.shape[-2] > 1:
                # sequence[:, 1:] aligns with future[:, 1:] (future[0] is the
                # action at the current step, already supervised by ``loss``).
                errors = (sequence[:, 1:] - future[:, 1:]).square().mean(-1)
                valid = future_mask[:, 1:]
                future_loss = (errors * valid).sum() / valid.sum().clamp_min(1.0)
                loss = loss + config.kmpc_sequence_weight * future_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"{name} produced a non-finite gradient")
            optimizer.step()
        should_validate = (
            epoch == 1
            or epoch % config.validation_interval == 0
            or epoch == config.epochs
        )
        validation = float("nan")
        if should_validate:
            validation = _evaluate_mse(
                name, actor, koopman, validation_loader, device
            )
            if validation < best_validation:
                best_validation = validation
                best_epoch = epoch
                last_improvement_epoch = epoch
                best_state = copy.deepcopy(actor.state_dict())
        if epoch == 1 or epoch % 50 == 0 or epoch == config.epochs:
            print(
                f"actor={name} epoch={epoch:04d} val_mse={validation:.6g}",
                flush=True,
            )
        if (
            output_dir is not None
            and checkpoint_payload_base is not None
            and best_state is not None
            and (epoch % config.checkpoint_interval == 0 or epoch == config.epochs)
        ):
            _atomic_torch_save(
                output_dir / f"{name}.pt",
                {
                    **checkpoint_payload_base,
                    "actor_state": best_state,
                    "report": {
                        "best_epoch": best_epoch,
                        "completed_epochs": epoch,
                        "best_validation_mse": best_validation,
                        "checkpoint_partial": epoch < config.epochs,
                    },
                },
            )
            (output_dir / f"{name}.status.json").write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "completed_epochs": epoch,
                        "best_validation_mse": best_validation,
                        "pid": os.getpid(),
                    }
                ),
                encoding="utf-8",
            )
        if (
            should_validate
            and epoch - last_improvement_epoch
            >= config.early_stopping_patience
        ):
            print(
                f"actor={name} early_stop={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break
    if best_state is None:
        raise RuntimeError(f"{name} training did not produce a checkpoint")
    actor.load_state_dict(best_state)
    return actor, {
        "best_epoch": best_epoch,
        "completed_epochs": epoch,
        "best_validation_mse": best_validation,
        "training_seconds": time.perf_counter() - started,
    }


def _make_builders(
    koopman: DeepKoopman,
    config: BCConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Actor constructors shared by BC training and checkpoint evaluation."""
    lifted_dim = koopman.lifted_dim
    return {
        "PPO": lambda: StandardPPOActor(
            koopman.state_dim, config.ppo_hidden_dim, config.action_limit
        ),
        "KLQR": lambda: KoopmanLQRActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            context_dim=0,
            hidden_dims=(config.hidden_dim,),
            max_action=config.action_limit,
        ),
        "AB-PQ": lambda: LowRankValueActor(
            observation_dim=lifted_dim,
            A=koopman.A,
            B=koopman.B,
            R=torch.eye(ACTION_DIM, device=device, dtype=koopman.A.dtype),
            base_hessian=torch.eye(
                lifted_dim, device=device, dtype=koopman.A.dtype
            ),
            rank=config.ab_rank,
            hidden_dims=(config.hidden_dim,),
            max_action=config.action_limit,
        ),
        "BC-KMPC": lambda: KoopmanMPCActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            horizon=config.kmpc_horizon,
            context_dim=0,
            hidden_dims=(config.hidden_dim,),
            action_low=-config.action_limit,
            action_high=config.action_limit,
            solver_iterations=config.kmpc_solver_iterations,
        ),
    }


def _make_env(num_envs: int, seed: int, device: torch.device):
    import gymnasium as gym
    import mani_skill.envs  # registers tasks into the gymnasium registry
    from mani_skill.envs.utils import rewards
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    base = gym.make(
        "MS-HopperHop-v1",
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        sim_backend="gpu" if device.type == "cuda" else "cpu",
        render_backend="none",
    )
    env = ManiSkillVectorEnv(
        base, num_envs, auto_reset=True, ignore_terminations=False,
        record_metrics=True,
    )
    env.reset(seed=[seed + i for i in range(num_envs)])
    return env


def _batch_features(
    observation: torch.Tensor,
    koopman: DeepKoopman,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw state (for PPO) + normalized lifted state (for Koopman actors)."""
    state = torch.as_tensor(observation, device=device, dtype=torch.float32)
    normalized_state = (state - torch.as_tensor(center, device=device)) / torch.as_tensor(
        scale, device=device
    )
    with torch.no_grad():
        lifted = koopman.lift(normalized_state)
    return state, lifted


def closed_loop_evaluation(
    name: str,
    actor: nn.Module,
    koopman: DeepKoopman,
    center: np.ndarray,
    scale: np.ndarray,
    config: BCConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Vectorized closed-loop evaluation in MS-HopperHop.

    HopperHop has no early termination, so every parallel env runs exactly
    ``MAX_EPISODE_STEPS`` steps and each env index is one episode.
    """
    from mani_skill.envs.utils import rewards

    num_envs = min(config.evaluation_episodes, config.evaluation_num_envs)
    env = _make_env(num_envs, config.evaluation_seed_start, device)
    reward_sum = torch.zeros(num_envs, device=device)
    standing_sum = torch.zeros(num_envs, device=device)
    hopping_sum = torch.zeros(num_envs, device=device)
    bound_counts = torch.zeros(num_envs, device=device)
    action_counts = torch.zeros(num_envs, device=device)
    actor.eval()
    try:
        observation, _ = env.reset(
            seed=[
                config.evaluation_seed_start + i for i in range(num_envs)
            ]
        )
        for _ in range(MAX_EPISODE_STEPS):
            state, lifted = _batch_features(
                observation, koopman, center, scale, device
            )
            with torch.no_grad():
                action, _ = _actor_action(name, actor, state, lifted)
            bound_counts += (action.abs() >= 0.99 * config.action_limit).sum(-1).float()
            action_counts += ACTION_DIM
            observation, reward, _terminated, _truncated, _info = env.step(action)
            reward_sum += torch.as_tensor(
                reward, device=device, dtype=torch.float32
            ).reshape(-1)
            # reward components for diagnosis (identical to the env formula)
            standing = rewards.tolerance(
                env.unwrapped.height, lower=STAND_HEIGHT, upper=2.0
            ).view(-1)
            hopping = rewards.tolerance(
                env.unwrapped.subtreelinvelx,
                lower=HOP_SPEED,
                upper=float("inf"),
                margin=HOP_SPEED / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            ).view(-1)
            standing_sum += standing.float()
            hopping_sum += hopping.float()
    finally:
        env.close()
    steps = float(MAX_EPISODE_STEPS)
    return {
        "episodes": num_envs,
        "mean_return": float(reward_sum.mean()),
        "mean_episode_length": float(MAX_EPISODE_STEPS),
        "mean_standing": float(standing_sum.mean() / steps),
        "mean_hopping": float(hopping_sum.mean() / steps),
        "mean_step_reward": float(reward_sum.mean() / steps),
        "action_bound_fraction": float(
            bound_counts.sum() / action_counts.sum()
        ),
    }


def train(
    dataset_path: Path,
    koopman_path: Path,
    output_dir: Path,
    config: BCConfig,
    device_name: str = "auto",
    actor_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    actor_names = actor_names or BC_ACTOR_ORDER
    for name in actor_names:
        if name not in BC_ACTOR_ORDER:
            raise ValueError(f"Unknown actor {name!r}")

    koopman, checkpoint = load_koopman(koopman_path, device)
    center, scale = _normalizers(checkpoint)
    data = load_dataset(dataset_path)
    splits = _split_arrays(data)
    future, future_mask = _build_future_actions(data, config.kmpc_horizon)
    future_by_split = {
        split: (
            future[mask],
            future_mask[mask],
        )
        for split, mask in {
            "train": np.isin(data["episode_id"], data["train_episode_ids"]),
            "validation": np.isin(
                data["episode_id"], data["validation_episode_ids"]
            ),
        }.items()
    }
    loaders = {
        split: _make_loader(
            (splits[split]["state"] - center) / scale,
            splits[split]["action"],
            future_by_split[split][0],
            future_by_split[split][1],
            config.batch_size,
            shuffle=split == "train",
            seed=config.seed,
        )
        for split in ("train", "validation")
    }
    # PPO route trains on the RAW state (matching the PPO baseline trainer).
    loaders["train_ppo"] = _make_loader(
        splits["train"]["state"],
        splits["train"]["action"],
        future_by_split["train"][0],
        future_by_split["train"][1],
        config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    loaders["validation_ppo"] = _make_loader(
        splits["validation"]["state"],
        splits["validation"]["action"],
        future_by_split["validation"][0],
        future_by_split["validation"][1],
        config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    builders = _make_builders(koopman, config, device)
    checkpoint_base = {
        "kind": "hopperhop_bc_actor",
        "state_kind": "hopperhop",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "koopman_path": str(koopman_path.resolve()),
        "normalizer": {
            "state_center": center.tolist(),
            "state_scale": scale.tolist(),
        },
        "config": asdict(config),
        "dataset_path": str(dataset_path.resolve()),
    }
    results: dict[str, Any] = {}
    for index, name in enumerate(actor_names):
        torch.manual_seed(config.seed + index)
        actor = builders[name]().to(device)
        use_ppo = name == "PPO"
        actor, train_info = _train_actor(
            name,
            actor,
            koopman,
            loaders["train_ppo" if use_ppo else "train"],
            loaders["validation_ppo" if use_ppo else "validation"],
            config,
            device,
            checkpoint_payload_base={
                **checkpoint_base,
                "name": name,
            },
            output_dir=output_dir,
        )
        # final full checkpoint (with closed-loop eval below)
        evaluation = closed_loop_evaluation(
            name, actor, koopman, center, scale, config, device
        )
        final_payload = {
            **checkpoint_base,
            "name": name,
            "actor_state": actor.state_dict(),
            "report": {
                **train_info,
                "evaluation": evaluation,
            },
        }
        _atomic_torch_save(output_dir / f"{name}.pt", final_payload)
        results[name] = {"training": train_info, "evaluation": evaluation}
        print(
            f"actor={name} bc_done val_mse={train_info['best_validation_mse']:.6g} "
            f"eval_return={evaluation['mean_return']:.1f} "
            f"standing={evaluation['mean_standing']:.3f} "
            f"hopping={evaluation['mean_hopping']:.3f}",
            flush=True,
        )
    report = {
        "kind": "hopperhop_bc_report",
        "dataset_path": str(dataset_path.resolve()),
        "koopman_path": str(koopman_path.resolve()),
        "config": asdict(config),
        "results": results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def evaluate_checkpoint(
    output_dir: Path,
    actor_name: str,
    koopman_path: Path,
    config: BCConfig,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Closed-loop re-evaluation of a saved BC actor without retraining."""
    if actor_name not in BC_ACTOR_ORDER:
        raise ValueError(f"Unknown actor {actor_name!r}")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    payload = torch.load(
        output_dir / f"{actor_name}.pt", map_location=device, weights_only=False
    )
    if payload.get("kind") != "hopperhop_bc_actor":
        raise ValueError(f"{actor_name}.pt is not a HopperHop BC actor")
    if payload.get("name") != actor_name:
        raise ValueError(f"Checkpoint actor name mismatch")
    koopman, _ = load_koopman(koopman_path, device)
    actor = _make_builders(koopman, config, device)[actor_name]().to(device)
    actor.load_state_dict(payload["actor_state"])
    normalizer = payload["normalizer"]
    result = closed_loop_evaluation(
        actor_name,
        actor,
        koopman,
        np.asarray(normalizer["state_center"], dtype=np.float32),
        np.asarray(normalizer["state_scale"], dtype=np.float32),
        config,
        device,
    )
    print(json.dumps({actor_name: result}, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("runs/hopper_hop/data/hopperhop_expert.npz"),
    )
    parser.add_argument(
        "--koopman",
        type=Path,
        default=Path("runs/hopper_hop/koopman_v2/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/hopper_hop/bc_v2"),
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--kmpc-horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument(
        "--actor",
        default=None,
        choices=list(BC_ACTOR_ORDER),
        help="train only this actor (default: all four sequentially)",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BCConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        kmpc_horizon=args.kmpc_horizon,
        seed=args.seed,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_num_envs=args.evaluation_episodes,
    )
    actor_names = (args.actor,) if args.actor is not None else None
    report = train(
        args.dataset,
        args.koopman,
        args.output_dir,
        config,
        args.device,
        actor_names=actor_names,
    )
    print(
        json.dumps(
            {
                name: {
                    "val_mse": info["training"]["best_validation_mse"],
                    "eval_return": info["evaluation"]["mean_return"],
                }
                for name, info in report["results"].items()
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
