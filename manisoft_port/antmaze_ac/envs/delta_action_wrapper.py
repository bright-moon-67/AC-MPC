from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class DeltaActionWrapper(gym.Wrapper):
    """Expose augmented observations and incremental actions.

    The wrapper follows Gymnasium's reset/step API. ``prev_action`` is kept in
    the observation, not hidden in a policy, so the transformed process remains
    Markov and can be reproduced during PPO ``evaluate_actions``.
    """

    def __init__(self, env: gym.Env, expected_observation_dim: int | None = None) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("DeltaActionWrapper requires a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("DeltaActionWrapper requires a Box action space")
        if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
            raise ValueError("Only flat observation and action spaces are supported")
        if expected_observation_dim is not None and env.observation_space.shape[0] != expected_observation_dim:
            raise ValueError(
                f"Expected observation dim {expected_observation_dim}, got {env.observation_space.shape[0]}"
            )

        self.action_dim = int(env.action_space.shape[0])
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        obs_low = np.asarray(env.observation_space.low, dtype=np.float32)
        obs_high = np.asarray(env.observation_space.high, dtype=np.float32)
        act_low = np.asarray(env.action_space.low, dtype=np.float32)
        act_high = np.asarray(env.action_space.high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([obs_low, act_low]),
            high=np.concatenate([obs_high, act_high]),
            dtype=np.float32,
        )
        # Any finite requested delta is accepted; the applied absolute action is clipped.
        self.action_space = gym.spaces.Box(
            low=np.full(self.action_dim, -np.inf, dtype=np.float32),
            high=np.full(self.action_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != self.env.observation_space.shape:
            raise ValueError(f"Observation shape changed: {observation.shape}")
        return np.concatenate([observation, self.prev_action]).astype(np.float32, copy=False)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        return self._augment(observation), info

    def step(self, delta_action: np.ndarray):
        requested_delta = np.asarray(delta_action, dtype=np.float32)
        if requested_delta.shape != (self.action_dim,):
            raise ValueError(f"Expected delta action shape {(self.action_dim,)}, got {requested_delta.shape}")
        if not np.isfinite(requested_delta).all():
            raise ValueError("Requested delta action contains NaN or Inf")
        previous = self.prev_action.copy()
        applied_action = np.clip(
            previous + requested_delta,
            np.asarray(self.env.action_space.low),
            np.asarray(self.env.action_space.high),
        ).astype(np.float32)
        applied_delta = applied_action - previous
        observation, reward, terminated, truncated, info = self.env.step(applied_action)
        self.prev_action = applied_action
        info = dict(info)
        tolerance = np.finfo(np.float32).eps * 4
        saturated = np.logical_or(
            applied_action <= np.asarray(self.env.action_space.low) + tolerance,
            applied_action >= np.asarray(self.env.action_space.high) - tolerance,
        )
        info.update(
            {
                "requested_delta_action": requested_delta.copy(),
                "applied_delta_action": applied_delta.copy(),
                "applied_action": applied_action.copy(),
                "action_saturation_ratio": float(np.mean(saturated)),
            }
        )
        return self._augment(observation), reward, terminated, truncated, info


def make_delta_vector_env(env_fns):
    """Create independently wrapped vector environments.

    Wrapping each sub-environment before vectorization makes autoresets reset
    the corresponding ``prev_action`` without affecting other workers.
    """

    return gym.vector.SyncVectorEnv([lambda fn=fn: DeltaActionWrapper(fn()) for fn in env_fns])
