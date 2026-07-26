#!/usr/bin/env python
"""Verify the exact legacy D4RL AntMaze API used for formal evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from antmaze_ac.envs.factory import make_antmaze_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    env = make_antmaze_env("antmaze-umaze-v2", backend="legacy")
    observation, _ = env.reset(seed=0)
    next_observation, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_dim, dtype=np.float32)
    )
    import d4rl
    import d4rl.infos  # noqa: F401
    import gym
    import mujoco_py

    report = {
        "backend": "legacy",
        "env_id": "antmaze-umaze-v2",
        "d4rl_version": d4rl.__version__,
        "gym_version": gym.__version__,
        "mujoco_py_version": mujoco_py.__version__,
        "raw_observation_dim": int(observation.shape[0] - env.action_dim),
        "augmented_observation_dim": int(observation.shape[0]),
        "action_dim": env.action_dim,
        "observation_shape": list(observation.shape),
        "next_observation_shape": list(next_observation.shape),
        "reset_previous_action_is_zero": bool(
            np.all(observation[-env.action_dim :] == 0)
        ),
        "zero_step_previous_action_is_zero": bool(
            np.all(next_observation[-env.action_dim :] == 0)
        ),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "max_episode_steps": int(env.env.legacy_env.spec.max_episode_steps),
        "target_goal": np.asarray(
            env.env.legacy_env.unwrapped.target_goal,
            dtype=np.float64,
        ).tolist(),
        "normalized_score_for_return_one": 100.0
        * float(d4rl.get_normalized_score("antmaze-umaze-v2", 1.0)),
        "applied_action": np.asarray(info["applied_action"]).tolist(),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
