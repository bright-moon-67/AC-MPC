from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import gymnasium as gym
import numpy as np


class HistoryContextTrackingWrapper(gym.Wrapper):
    """Expose a reconstructable history observation for history Koopman PPO.

    The flattened observation is

    ``[s_t, context_t, task_context]``

    where ``context_t=[normalized s[t-H+1:t+1], u[t-H:t]]``.  The latest
    history action is the previous applied action.  Legacy/BC use accepts an
    absolute action; optional ``max_delta`` use accepts a normalized increment
    and reconstructs the absolute action from that history state.  For a single goal
    ``task_context=target_tip``.  For an ordered waypoint task it is
    ``[G1,G2,G3,one_hot(active_index)]``.  Environments with a richer task may
    expose a fixed ``task_context_dim`` and ``task_context`` property; that
    vector is appended unchanged while the Koopman physical/history layout
    remains identical.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        history_steps: int,
        state_mean: Sequence[float] | np.ndarray,
        state_std: Sequence[float] | np.ndarray,
        tip_indices: Sequence[int] = (30, 31, 32),
        max_delta: float | None = None,
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
        if max_delta is not None and max_delta <= 0:
            raise ValueError("max_delta must be positive when configured")

        self.history_steps = int(history_steps)
        self.state_dim = int(env.observation_space.shape[0])
        self.action_dim = int(env.action_space.shape[0])
        self.max_delta = None if max_delta is None else float(max_delta)
        self.context_dim = self.history_steps * (
            self.state_dim + self.action_dim
        )
        external_context_dim = getattr(env, "task_context_dim", None)
        self.has_external_task_context = external_context_dim is not None
        self.waypoint_count = int(getattr(env, "waypoint_count", 1))
        if self.waypoint_count < 1:
            raise ValueError("waypoint_count must be positive")
        self.task_context_dim = (
            int(external_context_dim)
            if self.has_external_task_context
            else (3 if self.waypoint_count == 1 else 4 * self.waypoint_count)
        )
        if self.task_context_dim < 1:
            raise ValueError("task_context_dim must be positive")
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

        self.state_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.action_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)

        observation_dim = (
            self.state_dim + self.context_dim + self.task_context_dim
        )
        self.observation_space = gym.spaces.Box(
            low=np.full(observation_dim, -np.inf, dtype=np.float32),
            high=np.full(observation_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        if self.max_delta is None:
            # Legacy/BC mode: accept requested absolute actions and apply only
            # the environment's physical bounds.
            self.action_space = gym.spaces.Box(
                low=np.full(self.action_dim, -np.inf, dtype=np.float32),
                high=np.full(self.action_dim, np.inf, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            # PPO-KMPC mode: the policy variable is a dimensionless normalized
            # increment.  Mapping it here keeps the action saved by PPO exactly
            # equal to the action whose log-probability is optimized.
            self.action_space = gym.spaces.Box(
                low=np.full(self.action_dim, -1.0, dtype=np.float32),
                high=np.full(self.action_dim, 1.0, dtype=np.float32),
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

    @property
    def task_context(self) -> np.ndarray:
        if self.has_external_task_context:
            context = np.asarray(
                getattr(self.env, "task_context", None), dtype=np.float32
            ).reshape(-1)
            if context.shape != (self.task_context_dim,):
                raise RuntimeError(
                    "Wrapped task_context has wrong shape "
                    f"{context.shape}; expected {(self.task_context_dim,)}"
                )
            if not np.isfinite(context).all():
                raise FloatingPointError("Wrapped task_context contains NaN or Inf")
            return context
        if self.waypoint_count == 1:
            return self.target_tip
        waypoints = np.asarray(
            getattr(self.env, "waypoints", None), dtype=np.float32
        )
        if waypoints.shape != (self.waypoint_count, 3):
            raise RuntimeError(
                f"Wrapped waypoints have wrong shape {waypoints.shape}"
            )
        active_index = int(getattr(self.env, "active_waypoint_index", -1))
        if not 0 <= active_index < self.waypoint_count:
            raise RuntimeError("Wrapped active_waypoint_index is invalid")
        stage = np.eye(self.waypoint_count, dtype=np.float32)[active_index]
        return np.concatenate((waypoints.reshape(-1), stage))

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
        observation = np.concatenate((state, self._context(), self.task_context))
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

    def step(self, action: np.ndarray):
        requested = np.asarray(action, dtype=np.float32).reshape(-1)
        if requested.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action shape {(self.action_dim,)}, got {requested.shape}"
            )
        if not np.isfinite(requested).all():
            raise ValueError("Requested action contains NaN or Inf")

        base_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        base_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        previous = self.previous_action.copy()
        if self.max_delta is None:
            requested_absolute = requested
            requested_normalized_delta = None
        else:
            requested_normalized_delta = requested
            if np.any(requested < -1.0 - 1e-6) or np.any(
                requested > 1.0 + 1e-6
            ):
                raise ValueError(
                    "Normalized delta action must lie in [-1, 1]"
                )
            requested_absolute = (
                previous
                + self.max_delta * np.clip(requested, -1.0, 1.0)
            )
        applied = np.clip(
            requested_absolute, base_low, base_high
        ).astype(np.float32)
        state, reward, terminated, truncated, info = self.env.step(applied)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        self.previous_action[:] = applied
        self.state_history.append(state.copy())
        self.action_history.append(applied.copy())

        tolerance = np.finfo(np.float32).eps * 8
        saturated = np.abs(applied - requested_absolute) > tolerance
        bound = np.logical_or(
            applied <= base_low + tolerance,
            applied >= base_high - tolerance,
        )
        info = dict(info)
        info.update(
            {
                "requested_absolute_action": np.asarray(
                    requested_absolute, dtype=np.float32
                ).copy(),
                "applied_action": applied.copy(),
                "applied_delta_action": (applied - previous).copy(),
                "action_saturation_ratio": float(np.mean(saturated)),
                "action_bound_ratio": float(np.mean(bound)),
                "max_delta": self.max_delta,
            }
        )
        if requested_normalized_delta is not None:
            applied_normalized_delta = (
                (applied - previous) / self.max_delta
            ).astype(np.float32)
            info.update(
                {
                    "requested_normalized_delta_action": (
                        requested_normalized_delta.copy()
                    ),
                    "applied_normalized_delta_action": (
                        applied_normalized_delta.copy()
                    ),
                    "normalized_delta_bound_ratio": float(
                        np.mean(
                            np.abs(applied_normalized_delta)
                            >= 1.0 - 1e-6
                        )
                    ),
                }
            )
        return (
            self._observation(state),
            reward,
            terminated,
            truncated,
            info,
        )
