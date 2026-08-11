from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import gymnasium as gym
import numpy as np


class HistoryContextTrackingWrapper(gym.Wrapper):
    """Expose a reconstructable history observation for history Koopman PPO.

    The flattened observation is

    ``[s_t, context_t, target_tip]``

    where ``context_t=[normalized s[t-H+1:t+1], u[t-H:t]]``.  The current
    absolute action is deliberately absent.  The last action in the context
    is ``u[t-1]`` and is therefore sufficient to reproduce the action-rate
    constraints when PPO later evaluates shuffled minibatches.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        history_steps: int,
        state_mean: Sequence[float] | np.ndarray,
        state_std: Sequence[float] | np.ndarray,
        max_delta: float | Sequence[float] = 0.002,
        tip_indices: Sequence[int] = (30, 31, 32),
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("HistoryContextTrackingWrapper requires Box observations")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("HistoryContextTrackingWrapper requires Box actions")
        if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
            raise ValueError("Only flat observations and actions are supported")
        if history_steps < 1:
            raise ValueError("history_steps must be positive")

        self.history_steps = int(history_steps)
        self.state_dim = int(env.observation_space.shape[0])
        self.action_dim = int(env.action_space.shape[0])
        self.context_dim = self.history_steps * (
            self.state_dim + self.action_dim
        )
        self.tip_indices = np.asarray(tuple(tip_indices), dtype=np.int64)
        if self.tip_indices.shape != (3,):
            raise ValueError("tip_indices must contain exactly three indices")
        if np.any(self.tip_indices < 0) or np.any(self.tip_indices >= self.state_dim):
            raise ValueError("tip_indices are outside the physical state")

        self.state_mean = np.asarray(state_mean, dtype=np.float32).reshape(-1)
        self.state_std = np.asarray(state_std, dtype=np.float32).reshape(-1)
        if self.state_mean.shape != (self.state_dim,) or self.state_std.shape != (
            self.state_dim,
        ):
            raise ValueError("State normalizer shape does not match the environment")
        if not np.isfinite(self.state_mean).all() or not np.isfinite(
            self.state_std
        ).all():
            raise ValueError("State normalizer contains NaN or Inf")
        self.state_std = np.maximum(self.state_std, 1e-6)

        delta = np.asarray(max_delta, dtype=np.float32)
        self.max_delta = np.broadcast_to(delta, (self.action_dim,)).copy()
        if not np.isfinite(self.max_delta).all() or np.any(self.max_delta <= 0):
            raise ValueError("max_delta must be finite and positive")

        self.state_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.action_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)

        observation_dim = self.state_dim + self.context_dim + 3
        self.observation_space = gym.spaces.Box(
            low=np.full(observation_dim, -np.inf, dtype=np.float32),
            high=np.full(observation_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        # The wrapper accepts requested absolute actions. It applies the same
        # absolute/rate limits as the differentiable MPC actor before stepping.
        self.action_space = gym.spaces.Box(
            low=np.full(self.action_dim, -np.inf, dtype=np.float32),
            high=np.full(self.action_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def target_tip(self) -> np.ndarray:
        target = getattr(self.env, "target_tip", None)
        if target is None:
            raise RuntimeError("Wrapped environment has not initialized target_tip")
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if target.shape != (3,):
            raise RuntimeError(f"Wrapped target_tip has wrong shape {target.shape}")
        return target

    def _context(self) -> np.ndarray:
        if len(self.state_history) != self.history_steps or len(
            self.action_history
        ) != self.history_steps:
            raise RuntimeError("History buffer is not initialized")
        states = np.asarray(self.state_history, dtype=np.float32)
        normalized_states = (states - self.state_mean) / self.state_std
        actions = np.asarray(self.action_history, dtype=np.float32)
        return np.concatenate(
            (normalized_states.reshape(-1), actions.reshape(-1))
        ).astype(np.float32, copy=False)

    def _observation(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Physical observation has wrong shape {state.shape}")
        observation = np.concatenate((state, self._context(), self.target_tip))
        if observation.shape != self.observation_space.shape:
            raise RuntimeError("History observation layout is inconsistent")
        if not np.isfinite(observation).all():
            raise FloatingPointError("History observation contains NaN or Inf")
        return observation.astype(np.float32, copy=False)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        state, info = self.env.reset(**kwargs)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        self.state_history.clear()
        self.action_history.clear()
        for _ in range(self.history_steps):
            self.state_history.append(state.copy())
            self.action_history.append(
                np.zeros(self.action_dim, dtype=np.float32)
            )
        self.previous_action.fill(0.0)
        info = dict(info)
        info["history_steps"] = self.history_steps
        return self._observation(state), info

    def step(self, absolute_action: np.ndarray):
        requested = np.asarray(absolute_action, dtype=np.float32).reshape(-1)
        if requested.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action shape {(self.action_dim,)}, got {requested.shape}"
            )
        if not np.isfinite(requested).all():
            raise ValueError("Requested action contains NaN or Inf")

        base_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        base_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        rate_low = self.previous_action - self.max_delta
        rate_high = self.previous_action + self.max_delta
        applied = np.maximum(
            np.minimum(requested, np.minimum(base_high, rate_high)),
            np.maximum(base_low, rate_low),
        ).astype(np.float32)
        state, reward, terminated, truncated, info = self.env.step(applied)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        previous = self.previous_action.copy()
        self.previous_action[:] = applied
        self.state_history.append(state.copy())
        self.action_history.append(applied.copy())

        tolerance = np.finfo(np.float32).eps * 8
        saturated = np.abs(applied - requested) > tolerance
        info = dict(info)
        info.update(
            {
                "requested_absolute_action": requested.copy(),
                "applied_action": applied.copy(),
                "applied_delta_action": (applied - previous).copy(),
                "action_saturation_ratio": float(np.mean(saturated)),
            }
        )
        return (
            self._observation(state),
            reward,
            terminated,
            truncated,
            info,
        )
