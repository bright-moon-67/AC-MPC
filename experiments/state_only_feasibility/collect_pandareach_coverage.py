"""Collect broad state-space coverage data for global Koopman identification.

The DLS expert demonstrations only visit states along minimum-jerk tracking
trajectories, so a Deep Koopman model trained on them is only reliable near
those trajectories -- closed-loop policies that push the robot elsewhere then
see a broken lift. This collector drives the Panda arm with randomized
exploratory actions:

* ``uniform``: full-range random joint deltas (broad q / qdot / tcp coverage),
* ``small``:  low-amplitude noise near the current pose (near-static dynamics),
* ``sweep``:  large-magnitude moves in random directions (high velocity, TCP
  across the workspace),
* ``hold``:   zero action (deceleration / static states).

The saved npz uses the exact schema consumed by
``train_pandareach_koopman.py``: ``state_kind=q_qdot_tcp``, consecutive
``step_index`` per episode, and ``next_state[i] == state[i+1]`` so K-step
windows are chain-consistent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from experiments.state_only_feasibility.collect_pandareach import _episode_splits
from experiments.state_only_feasibility.collect_pandareach_threewaypoint import (
    _state_from_observation,
)
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
)


@dataclass(frozen=True)
class CoverageConfig:
    episodes: int = 3000
    seed_start: int = 20_280_904
    split_seed: int = 31
    steps_per_episode: int = 200
    action_limit_rad: float = 0.1
    robot_init_qpos_noise: float = 0.2
    uniform_prob: float = 0.60
    small_prob: float = 0.15
    small_scale_rad: float = 0.01
    sweep_prob: float = 0.20
    hold_prob: float = 0.05

    def validate(self) -> None:
        if self.episodes < 3:
            raise ValueError("At least three episodes are required")
        if self.steps_per_episode < 20:
            raise ValueError("steps_per_episode must be >= Koopman K_step (20)")
        if self.action_limit_rad <= 0:
            raise ValueError("action_limit_rad must be positive")
        if self.robot_init_qpos_noise < 0:
            raise ValueError("robot_init_qpos_noise must be non-negative")
        if self.small_scale_rad <= 0:
            raise ValueError("small_scale_rad must be positive")
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


def _sample_env_action(
    rng: np.random.Generator,
    config: CoverageConfig,
) -> np.ndarray:
    """Return a normalized joint-delta action in [-1, 1]^7."""

    draw = rng.random()
    if draw < config.uniform_prob:
        return rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
    if draw < config.uniform_prob + config.small_prob:
        scale = config.small_scale_rad / config.action_limit_rad
        return rng.uniform(-scale, scale, size=7).astype(np.float32)
    if (
        draw
        < config.uniform_prob + config.small_prob + config.sweep_prob
    ):
        magnitude = rng.uniform(0.8, 1.0, size=7).astype(np.float32)
        sign = rng.choice([-1.0, 1.0], size=7).astype(np.float32)
        return (magnitude * sign).astype(np.float32)
    return np.zeros(7, dtype=np.float32)


def collect(
    config: CoverageConfig,
    output_path: Path,
) -> dict[str, Any]:
    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = gym.make(
        "ACMPC-PandaReach3-v0",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode=None,
        render_backend="none",
        max_episode_steps=config.steps_per_episode,
        goal_threshold=0.01,
        robot_init_qpos_noise=config.robot_init_qpos_noise,
    )
    env = PandaArmOnlyActionWrapper(base_env)
    env.reset(seed=config.seed_start)
    if (
        int(env.unwrapped.control_freq),
        int(env.unwrapped.sim_freq),
        int(env.unwrapped._sim_steps_per_control),
    ) != (20, 100, 5):
        raise RuntimeError("Unexpected PandaReach control timing")
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
    episode_length: list[int] = []
    try:
        for episode in range(config.episodes):
            seed = config.seed_start + episode
            observation, _ = env.reset(seed=seed)
            episode_seed.append(seed)
            for step in range(config.steps_per_episode):
                state = _state_from_observation(observation)
                tcp = state[-3:].copy()
                env_action = _sample_env_action(rng, config)
                action = (env_action * config.action_limit_rad).astype(
                    np.float32
                )
                next_observation, _, _, _, _ = env.step(env_action)
                next_state = _state_from_observation(next_observation)
                transitions["state"].append(state)
                transitions["action"].append(action)
                transitions["env_action"].append(env_action)
                transitions["next_state"].append(next_state)
                transitions["tcp_position"].append(tcp)
                transitions["next_tcp_position"].append(
                    next_state[-3:].copy()
                )
                observation = next_observation
            episode_length.append(config.steps_per_episode)
    finally:
        env.close()

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
            "episode_length": np.asarray(episode_length, dtype=np.int64),
            "train_episode_ids": train_ids,
            "validation_episode_ids": validation_ids,
            "test_episode_ids": test_ids,
        }
    )
    for name, value in arrays.items():
        if value.dtype.kind not in {"U", "S"} and not np.isfinite(value).all():
            raise RuntimeError(f"Coverage dataset field {name} contains NaN/Inf")

    # Chain-consistency audit: next_state[i] must equal state[i+1] within an
    # episode, as required by the Koopman K-step window builder.
    not_last = (
        step_index[:-1] != config.steps_per_episode - 1
    )
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

    q = arrays["state"][:, :7]
    qdot = arrays["state"][:, 7:14]
    tcp = arrays["state"][:, 14:17]
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
    }
    summary = {
        "dataset_path": str(output_path.resolve()),
        "episodes": config.episodes,
        "transitions": int(len(arrays["state"])),
        "exploration": {
            "uniform_prob": config.uniform_prob,
            "small_prob": config.small_prob,
            "small_scale_rad": config.small_scale_rad,
            "sweep_prob": config.sweep_prob,
            "hold_prob": config.hold_prob,
            "robot_init_qpos_noise": config.robot_init_qpos_noise,
        },
        "action_definition": (
            f"u=joint_delta in radians, hard support [-{config.action_limit_rad},"
            f"{config.action_limit_rad}]; env_action=u/0.1"
        ),
        "state_definition": "x=[q(7), qdot(7), tcp_xyz(3)]",
        "state_kind": "q_qdot_tcp",
        "coverage": coverage,
        "timing": {
            "control_frequency_hz": 20,
            "simulation_frequency_hz": 100,
            "physics_steps_per_action": 5,
            "control_dt_seconds": 0.05,
        },
        "split": {
            "train": int(len(train_ids)),
            "validation": int(len(validation_ids)),
            "test": int(len(test_ids)),
        },
        "config": asdict(config),
        "policy": (
            "random exploratory joint-delta actions (uniform/small/sweep/hold); "
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
        default=Path(
            "runs/pandareach_threewaypoint/data/pandareach_coverage_1000.npz"
        ),
    )
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=20_280_904)
    parser.add_argument("--split-seed", type=int, default=31)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=0.2)
    parser.add_argument("--uniform-prob", type=float, default=0.60)
    parser.add_argument("--small-prob", type=float, default=0.15)
    parser.add_argument("--small-scale-rad", type=float, default=0.01)
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
        small_scale_rad=args.small_scale_rad,
        sweep_prob=args.sweep_prob,
        hold_prob=args.hold_prob,
    )
    print(json.dumps(collect(config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
