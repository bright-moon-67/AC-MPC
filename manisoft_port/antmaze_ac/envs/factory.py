from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .delta_action_wrapper import DeltaActionWrapper


class ModernAntMazeAdapter(gym.Wrapper):
    """Flatten Gymnasium-Robotics AntMaze to the legacy 29-D state.

    Modern AntMaze separates ``achieved_goal=(x,y)`` from a 27-D proprioceptive
    observation. Concatenating them restores the legacy ``expose_all_qpos``
    29-D ordering. Sparse reward is normalized to legacy 0/1 using ``success``.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("Expected Dict observation from Gymnasium-Robotics AntMaze")
        achieved = env.observation_space["achieved_goal"]
        proprioception = env.observation_space["observation"]
        self.observation_space = gym.spaces.Box(
            np.concatenate((achieved.low, proprioception.low)).astype(np.float32),
            np.concatenate((achieved.high, proprioception.high)).astype(np.float32),
            dtype=np.float32,
        )

    @staticmethod
    def _flatten(observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate((observation["achieved_goal"], observation["observation"])).astype(np.float32)

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        return self._flatten(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["backend_sparse_reward"] = float(reward)
        legacy_reward = float(bool(info.get("success", reward > 0)))
        return self._flatten(observation), legacy_reward, terminated, truncated, info


class LegacyGymAdapter(gym.Env):
    """Adapt an old Gym/D4RL environment to Gymnasium's API."""

    def __init__(self, env) -> None:
        self.legacy_env = env
        self.observation_space = gym.spaces.Box(
            np.asarray(env.observation_space.low, dtype=np.float32),
            np.asarray(env.observation_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            np.asarray(env.action_space.low, dtype=np.float32),
            np.asarray(env.action_space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # D4RL AntMaze v2 samples its per-rollout goal from the legacy
            # module-level NumPy RNG rather than env.np_random.
            np.random.seed(seed)
            try:
                self.legacy_env.seed(seed)
            except (AttributeError, TypeError):
                pass
        result = self.legacy_env.reset()
        observation = result[0] if isinstance(result, tuple) else result
        return np.asarray(observation, dtype=np.float32), {}

    def step(self, action):
        result = self.legacy_env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        else:
            observation, reward, done, info = result
            timeout = bool(info.get("TimeLimit.truncated", False))
            terminated, truncated = bool(done and not timeout), timeout
        return np.asarray(observation, dtype=np.float32), float(reward), terminated, truncated, info

    def close(self):
        return self.legacy_env.close()


def make_antmaze_env(
    env_id: str = "antmaze-umaze-v2",
    *,
    backend: str = "auto",
    max_episode_steps: int = 700,
) -> DeltaActionWrapper:
    errors = []
    if backend in ("auto", "legacy"):
        try:
            import d4rl  # noqa: F401
            import gym as old_gym

            return DeltaActionWrapper(
                LegacyGymAdapter(old_gym.make(env_id)),
                expected_observation_dim=29,
            )
        except Exception as error:
            errors.append(f"legacy D4RL: {error}")
            if backend == "legacy":
                raise RuntimeError(errors[-1]) from error
    if backend in ("auto", "modern"):
        try:
            import gymnasium_robotics  # noqa: F401

            env = gym.make(
                "AntMaze_UMaze-v5",
                continuing_task=False,
                reset_target=False,
                include_cfrc_ext_in_observation=False,
                max_episode_steps=max_episode_steps,
            )
            return DeltaActionWrapper(ModernAntMazeAdapter(env), expected_observation_dim=29)
        except Exception as error:
            errors.append(f"Gymnasium-Robotics: {error}")
            raise RuntimeError("No AntMaze backend available. " + "; ".join(errors)) from error
    raise ValueError("backend must be auto, legacy, or modern")
