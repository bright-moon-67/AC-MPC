"""Collect expert (state, action) BC data by rolling out trained PPO policies.

Loads the trained 50M-step PPO checkpoints (one per seed) and rolls them out
with their *stochastic* Gaussian policy in MS-HopperHop, collecting
``(state, action)`` transitions.  Episodes are split 8/1/1 into
train/validation/test (interleaved by seed) and written to a dataset ``.npz``
with the same schema as the Koopman dataset builder:

    state [N,15], action [N,4], episode_id [N] (globally unique, 0..E-1),
    step_index [N], train_episode_ids / validation_episode_ids /
    test_episode_ids, state_kind = "hopperhop", source (checkpoint paths)

This is the BC pretraining dataset for the 4-method comparison; the earlier
budgeted PPO transitions (from the 20M run) are used for Koopman model
identification, NOT for imitation (those policies were far weaker).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from experiments.hopper_hop.train_hopper_hop_ppo import Actor, _make_env

STATE_DIM = 15
ACTION_DIM = 4
EPISODE_LENGTH = 600


@dataclass(frozen=True)
class CollectConfig:
    checkpoint_root: Path
    output: Path
    seed_dirs: tuple[str, ...] = (
        "seed_20240201",
        "seed_20240202",
        "seed_20240203",
    )
    num_envs: int = 2048
    # episodes to keep per seed (randomly sampled from the full rollout)
    episodes_per_seed: int = 500
    # scale applied to the learned exploration std during rollouts
    exploration_scale: float = 1.0
    seed: int = 20250808


def _load_policy(
    checkpoint_root: Path, seed_dir: str, device: torch.device
) -> tuple[Actor, torch.Tensor]:
    checkpoint_path = checkpoint_root / seed_dir / "latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    actor = Actor(int(payload["actor_state"]["net.0.weight"].shape[1]), ACTION_DIM)
    actor.load_state_dict(payload["actor_state"])
    actor.to(device).eval()
    log_std = payload["log_std"].to(device)
    return actor, log_std


def _collect_seed(
    actor: Actor,
    log_std: torch.Tensor,
    config: CollectConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Vectorized rollout: every env runs exactly EPISODE_LENGTH steps.

    MS-HopperHop has no early termination (only the 600-step horizon
    truncation), so all parallel envs complete a full episode synchronously
    and each env index maps 1:1 to one episode.
    """
    env = _make_env(config.num_envs, config.seed)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    with torch.no_grad():
        observation, _ = env.reset(
            seed=[config.seed + i for i in range(config.num_envs)]
        )
        for _ in range(EPISODE_LENGTH):
            mean = actor(observation.to(device))
            std = (log_std.exp() * config.exploration_scale).expand_as(mean)
            action = Normal(mean, std).sample()
            states.append(observation.to(device).detach().cpu().numpy())
            actions.append(action.detach().cpu().numpy())
            observation, _reward, _terminated, _truncated, _info = env.step(
                action
            )
    env.close()

    state_stack = np.stack(states, axis=0)  # [600, num_envs, 15]
    action_stack = np.stack(actions, axis=0)  # [600, num_envs, 4]

    rng = np.random.RandomState(config.seed)
    selected = np.sort(
        rng.choice(config.num_envs, config.episodes_per_seed, replace=False)
    )

    states_out: list[np.ndarray] = []
    actions_out: list[np.ndarray] = []
    steps_out: list[np.ndarray] = []
    ids_out: list[np.ndarray] = []
    for episode_index, env_idx in enumerate(selected):
        states_out.append(state_stack[:, env_idx])  # [600, 15]
        actions_out.append(action_stack[:, env_idx])  # [600, 4]
        steps_out.append(np.arange(EPISODE_LENGTH, dtype=np.int64))
        ids_out.append(np.full(EPISODE_LENGTH, episode_index, dtype=np.int64))
    return {
        "state": np.concatenate(states_out, axis=0).astype(np.float32),
        "action": np.concatenate(actions_out, axis=0).astype(np.float32),
        "episode_id": np.concatenate(ids_out).astype(np.int64),
        "step_index": np.concatenate(steps_out).astype(np.int64),
    }


def _split_episodes(episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    order = rng.permutation(episodes)
    train = order[: int(episodes * 0.8)]
    validation = order[int(episodes * 0.8) : int(episodes * 0.9)]
    test = order[int(episodes * 0.9) :]
    return train, validation, test


def collect(config: CollectConfig) -> Path:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed: list[dict[str, np.ndarray]] = []
    source_paths: list[str] = []
    for seed_dir in config.seed_dirs:
        actor, log_std = _load_policy(config.checkpoint_root, seed_dir, device)
        print(f"rolling out {seed_dir} ...", flush=True)
        data = _collect_seed(actor, log_std, config, device)
        per_seed.append(data)
        source_paths.append(str(config.checkpoint_root / seed_dir / "latest.pt"))

    # Remap episode ids to be globally unique across seeds.
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    step_indices: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    episode_offset = 0
    total_episodes = 0
    for data in per_seed:
        episodes = int(np.max(data["episode_id"])) + 1
        states.append(data["state"])
        actions.append(data["action"])
        step_indices.append(data["step_index"])
        episode_ids.append(data["episode_id"] + episode_offset)
        episode_offset += episodes
        total_episodes += episodes

    train, validation, test = _split_episodes(total_episodes, config.seed)
    output = {
        "state": np.concatenate(states).astype(np.float32),
        "action": np.concatenate(actions).astype(np.float32),
        "episode_id": np.concatenate(episode_ids).astype(np.int64),
        "step_index": np.concatenate(step_indices).astype(np.int64),
        "train_episode_ids": train.astype(np.int64),
        "validation_episode_ids": validation.astype(np.int64),
        "test_episode_ids": test.astype(np.int64),
        "state_kind": np.asarray("hopperhop"),
        "state_dim": np.asarray(STATE_DIM),
        "action_dim": np.asarray(ACTION_DIM),
        "source": np.asarray(source_paths),
    }
    for name in ("state", "action"):
        if not np.isfinite(output[name]).all():
            raise FloatingPointError(f"{name} contains NaN/Inf")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.output, **output)
    return config.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("runs/hopper_hop/ppo_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/hopper_hop/data/hopperhop_expert.npz"),
    )
    parser.add_argument(
        "--seed-dirs",
        default="seed_20240201,seed_20240202,seed_20240203",
    )
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--episodes-per-seed", type=int, default=500)
    parser.add_argument("--exploration-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20250808)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CollectConfig(
        checkpoint_root=args.checkpoint_root,
        output=args.output,
        seed_dirs=tuple(args.seed_dirs.split(",")),
        num_envs=args.num_envs,
        episodes_per_seed=args.episodes_per_seed,
        exploration_scale=args.exploration_scale,
        seed=args.seed,
    )
    path = collect(config)
    with np.load(path, allow_pickle=False) as archive:
        print(
            f"expert dataset written: {path}",
            flush=True,
        )
        print(
            f"  transitions={len(archive['state']):,} "
            f"episodes(train/val/test)="
            f"{len(archive['train_episode_ids'])}/"
            f"{len(archive['validation_episode_ids'])}/"
            f"{len(archive['test_episode_ids'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
