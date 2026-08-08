"""Collect broad 21-dim state-action coverage data for PickCube Koopman ID.

The causal replay demonstrations only visit goal-directed trajectories, so a
Deep Koopman trained on them (val rollout nMSE ~0.149 vs ~0.005 for the
PandaReach coverage model) is unreliable off-manifold.  This collector drives
the PickCube Panda with randomized exploratory actions over the full
21-dim robot state (qpos9 + qvel9 + tcp_xyz3) and 8-dim normalized action
space, mirroring ``collect_pandareach_coverage.py``:

* ``uniform``: full-range random normalized actions (broad q/qdot/tcp coverage),
* ``small``:   low-amplitude noise near the current pose (near-static dynamics),
* ``sweep``:   large-magnitude moves in random directions (high velocity),
* ``hold``:    zero action (deceleration / static states).

The gripper dimension (action[7]) is sampled by the same modes as-is; its
binary nature is not treated specially.  The saved npz uses the exact schema
consumed by ``train_pickcube_robot_koopman.py``: ``state_kind=q_qdot_tcp``,
consecutive ``step_index`` per episode, and ``next_state[i] == state[i+1]``.

Run:
  python -m experiments.maniskill_pick_visual.collect_pickcube_coverage \
      --output runs/pickcube_coverage/data/pickcube_coverage_600k.npz \
      --episodes 12000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from experiments.state_only_feasibility.collect_pandareach import (
    _episode_splits,
)
from experiments.maniskill_pick_visual.visual_pick_cube import (
    VisualPickCubeEnv,  # registers ACMPC-VisualPickCube-v1
)

ROBOT_DIM = 21
ACTION_DIM = 8


@dataclass(frozen=True)
class CoverageConfig:
    episodes: int = 12000
    seed_start: int = 20_280_904
    split_seed: int = 31
    steps_per_episode: int = 50
    robot_init_qpos_noise: float = 0.05
    uniform_prob: float = 0.60
    small_prob: float = 0.15
    small_scale: float = 0.05
    sweep_prob: float = 0.20
    hold_prob: float = 0.05

    def validate(self) -> None:
        if self.episodes < 3:
            raise ValueError("At least three episodes are required")
        if self.steps_per_episode < 20:
            raise ValueError("steps_per_episode must be >= Koopman K_step (20)")
        if self.robot_init_qpos_noise < 0:
            raise ValueError("robot_init_qpos_noise must be non-negative")
        if self.small_scale <= 0:
            raise ValueError("small_scale must be positive")
        probs = (
            self.uniform_prob
            + self.small_prob
            + self.sweep_prob
            + self.hold_prob
        )
        if abs(probs - 1.0) > 1e-6:
            raise ValueError(f"Action-mode probabilities must sum to 1, got {probs}")
        if not all(
            0.0 <= p <= 1.0
            for p in (
                self.uniform_prob,
                self.small_prob,
                self.sweep_prob,
                self.hold_prob,
            )
        ):
            raise ValueError("Action-mode probabilities must lie in [0, 1]")


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    """Return robot[21] = qpos9 + qvel9 + tcp_xyz3."""

    qpos = _numpy(observation["agent"]["qpos"]).reshape(-1)[:9]
    qvel = _numpy(observation["agent"]["qvel"]).reshape(-1)[:9]
    tcp = _numpy(observation["extra"]["tcp_pose"]).reshape(-1)[:3]
    return np.concatenate((qpos, qvel, tcp)).astype(np.float32)


def _sample_env_action(
    rng: np.random.Generator,
    config: CoverageConfig,
) -> np.ndarray:
    """Return a normalized action in [-1, 1]^8 (PickCube env action space)."""

    draw = rng.random()
    if draw < config.uniform_prob:
        return rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
    if draw < config.uniform_prob + config.small_prob:
        return rng.uniform(
            -config.small_scale, config.small_scale, size=ACTION_DIM
        ).astype(np.float32)
    if draw < config.uniform_prob + config.small_prob + config.sweep_prob:
        magnitude = rng.uniform(0.8, 1.0, size=ACTION_DIM).astype(np.float32)
        sign = rng.choice([-1.0, 1.0], size=ACTION_DIM).astype(np.float32)
        return (magnitude * sign).astype(np.float32)
    return np.zeros(ACTION_DIM, dtype=np.float32)


def collect(
    config: CoverageConfig,
    output_path: Path,
) -> dict[str, Any]:
    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = gym.make(
        "ACMPC-VisualPickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode=None,
        render_backend="none",
        max_episode_steps=config.steps_per_episode,
        robot_init_qpos_noise=config.robot_init_qpos_noise,
    )
    base_env.reset(seed=config.seed_start)
    rng = np.random.default_rng(config.seed_start)
    transitions: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "state",
            "action",
            "env_action",
            "next_state",
            "tcp_position",
            "next_tcp_position",
        )
    }
    episode_seed: list[int] = []
    try:
        for episode in range(config.episodes):
            seed = config.seed_start + episode
            observation, _ = base_env.reset(seed=seed)
            episode_seed.append(seed)
            for _ in range(config.steps_per_episode):
                state = _state_from_observation(observation)
                tcp = state[-3:].copy()
                env_action = _sample_env_action(rng, config)
                next_observation, _, _, _, _ = base_env.step(env_action)
                next_state = _state_from_observation(next_observation)
                transitions["state"].append(state)
                transitions["action"].append(env_action.copy())
                transitions["env_action"].append(env_action)
                transitions["next_state"].append(next_state)
                transitions["tcp_position"].append(tcp)
                transitions["next_tcp_position"].append(next_state[-3:].copy())
                observation = next_observation
    finally:
        base_env.close()

    arrays: dict[str, np.ndarray] = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in transitions.items()
    }
    episode_id = np.repeat(
        np.arange(config.episodes, dtype=np.int64),
        config.steps_per_episode,
    )
    step_index = np.tile(
        np.arange(config.steps_per_episode, dtype=np.int64),
        config.episodes,
    )
    train_ids, validation_ids, test_ids = _episode_splits(
        config.episodes, config.split_seed
    )
    arrays.update(
        {
            "episode_id": episode_id,
            "step_index": step_index,
            "state_kind": np.asarray("q_qdot_tcp"),
            "episode_seed": np.asarray(episode_seed, dtype=np.int64),
            "episode_length": np.full(
                config.episodes, config.steps_per_episode, dtype=np.int64
            ),
            "train_episode_ids": train_ids,
            "validation_episode_ids": validation_ids,
            "test_episode_ids": test_ids,
        }
    )
    for name, value in arrays.items():
        if value.dtype.kind not in {"U", "S"} and not np.isfinite(value).all():
            raise RuntimeError(f"Coverage dataset field {name} contains NaN/Inf")

    not_last = step_index[:-1] != config.steps_per_episode - 1
    mismatch = float(
        np.max(
            np.abs(
                arrays["next_state"][:-1][not_last]
                - arrays["state"][1:][not_last]
            )
        )
    )
    if mismatch > 2e-5:
        raise RuntimeError(
            f"Coverage transition chain mismatch {mismatch:.3e}"
        )

    np.savez_compressed(output_path, **arrays)

    state = arrays["state"]
    q = state[:, :9]
    qdot = state[:, 9:18]
    tcp = state[:, 18:21]
    coverage = {
        "q_pos_rad": {
            "minimum": q.min(0).tolist(),
            "maximum": q.max(0).tolist(),
            "span": (q.max(0) - q.min(0)).tolist(),
        },
        "qdot_rad_s": {
            "minimum": qdot.min(0).tolist(),
            "maximum": qdot.max(0).tolist(),
            "rms": np.sqrt(np.mean(np.square(qdot), axis=0)).tolist(),
        },
        "tcp_m": {
            "minimum": tcp.min(0).tolist(),
            "maximum": tcp.max(0).tolist(),
            "span": (tcp.max(0) - tcp.min(0)).tolist(),
        },
        "action_span": {
            "minimum": arrays["action"].min(0).tolist(),
            "maximum": arrays["action"].max(0).tolist(),
        },
    }
    summary = {
        "dataset_path": str(output_path.resolve()),
        "episodes": config.episodes,
        "transitions": int(len(arrays["state"])),
        "exploration": {
            "uniform_prob": config.uniform_prob,
            "small_prob": config.small_prob,
            "small_scale": config.small_scale,
            "sweep_prob": config.sweep_prob,
            "hold_prob": config.hold_prob,
            "robot_init_qpos_noise": config.robot_init_qpos_noise,
        },
        "action_definition": (
            "u=normalized pd_joint_delta in [-1,1]^8 (7 arm + 1 gripper); "
            "env action space"
        ),
        "state_definition": "x=[q(9), qdot(9), tcp_xyz(3)]",
        "state_kind": "q_qdot_tcp",
        "coverage": coverage,
        "split": {
            "train": int(len(train_ids)),
            "validation": int(len(validation_ids)),
            "test": int(len(test_ids)),
        },
        "config": asdict(config),
        "policy": (
            "random exploratory normalized actions (uniform/small/sweep/hold); "
            "no task objective"
        ),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/pickcube_coverage/data/pickcube_coverage_600k.npz"),
    )
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--steps-per-episode", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=20_280_904)
    parser.add_argument("--split-seed", type=int, default=31)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=0.05)
    parser.add_argument("--uniform-prob", type=float, default=0.60)
    parser.add_argument("--small-prob", type=float, default=0.15)
    parser.add_argument("--small-scale", type=float, default=0.05)
    parser.add_argument("--sweep-prob", type=float, default=0.20)
    parser.add_argument("--hold-prob", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CoverageConfig(
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        seed_start=args.seed_start,
        split_seed=args.split_seed,
        robot_init_qpos_noise=args.robot_init_qpos_noise,
        uniform_prob=args.uniform_prob,
        small_prob=args.small_prob,
        small_scale=args.small_scale,
        sweep_prob=args.sweep_prob,
        hold_prob=args.hold_prob,
    )
    print(json.dumps(collect(config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
