"""Residual-action wrapper around a frozen, normalized SB3 policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np


class _Predictor(Protocol):
    def predict(
        self, observation: np.ndarray, *, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]: ...


class _ObservationNormalizer(Protocol):
    def normalize_obs(self, observation: np.ndarray) -> np.ndarray: ...


class FrozenBaseResidualActionWrapper(gym.Wrapper):
    """Add a bounded learned residual to a frozen SAC controller.

    The wrapped environment still owns any analytical-controller blend.  This
    wrapper only changes the learned part from ``a`` to
    ``clip(a_frozen + residual_scale * a_residual)``.  A newly initialized
    residual actor whose deterministic mean is zero therefore reproduces the
    frozen deployment exactly, while exploration and learning cannot move the
    command by more than ``residual_scale`` per axis.
    """

    def __init__(
        self,
        env: gym.Env,
        frozen_model_path: str | Path | None = None,
        frozen_vecnormalize_path: str | Path | None = None,
        *,
        residual_action_scale: float = 0.10,
        residual_action_penalty_scale: float = 0.0,
        residual_stall_activation_steps: int = 0,
        residual_stall_ramp_steps: int = 1,
        base_model: _Predictor | None = None,
        observation_normalizer: _ObservationNormalizer | None = None,
    ) -> None:
        super().__init__(env)
        if (
            not isinstance(self.action_space, gym.spaces.Box)
            or len(self.action_space.shape) != 1
            or self.action_space.shape[0] < 1
        ):
            raise ValueError(
                "frozen-base residual SAC requires a one-dimensional Box "
                "action vector"
            )
        if not 0.0 < residual_action_scale <= 1.0:
            raise ValueError("residual_action_scale must lie in (0, 1]")
        if residual_action_penalty_scale < 0:
            raise ValueError("residual_action_penalty_scale must be non-negative")
        if residual_stall_activation_steps < 0:
            raise ValueError(
                "residual_stall_activation_steps must be non-negative"
            )
        if residual_stall_ramp_steps < 1:
            raise ValueError("residual_stall_ramp_steps must be positive")
        if (base_model is None) != (observation_normalizer is None):
            raise ValueError(
                "base_model and observation_normalizer must be supplied together"
            )

        if base_model is None:
            if frozen_model_path is None or frozen_vecnormalize_path is None:
                raise ValueError(
                    "frozen model and VecNormalize paths are required"
                )
            model_path = Path(frozen_model_path).expanduser().resolve()
            normalizer_path = Path(frozen_vecnormalize_path).expanduser().resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"missing frozen SAC model: {model_path}")
            if not normalizer_path.is_file():
                raise FileNotFoundError(
                    f"missing frozen VecNormalize state: {normalizer_path}"
                )
            # Imports stay inside the runtime path so lightweight environment
            # tests do not require Stable-Baselines3.
            from stable_baselines3 import SAC
            from stable_baselines3.common.save_util import load_from_pkl

            base_model = SAC.load(str(model_path), device="cpu")
            observation_normalizer = load_from_pkl(str(normalizer_path))
            observation_normalizer.training = False

        self.base_model = base_model
        self.observation_normalizer = observation_normalizer
        self.residual_action_scale = float(residual_action_scale)
        self.residual_action_penalty_scale = float(
            residual_action_penalty_scale
        )
        self.residual_stall_activation_steps = int(
            residual_stall_activation_steps
        )
        self.residual_stall_ramp_steps = int(residual_stall_ramp_steps)
        self._raw_observation: np.ndarray | None = None

    def _residual_activation_factor(self) -> float:
        """Activate learned corrections only after base progress has stalled."""

        if self.residual_stall_activation_steps <= 0:
            return 1.0
        base_env = self.env.unwrapped
        step_count = int(getattr(base_env, "step_count", 0))
        last_improvement = int(
            getattr(base_env, "last_waypoint_improvement_step", step_count)
        )
        stalled_steps = max(step_count - last_improvement, 0)
        return float(
            np.clip(
                (
                    stalled_steps - self.residual_stall_activation_steps
                )
                / self.residual_stall_ramp_steps,
                0.0,
                1.0,
            )
        )

    def _frozen_action(self, raw_observation: np.ndarray) -> np.ndarray:
        batched = np.asarray(raw_observation, dtype=np.float32)[None, :]
        normalized = self.observation_normalizer.normalize_obs(batched)
        action, _ = self.base_model.predict(normalized, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != self.action_space.shape:
            raise RuntimeError(
                "frozen policy returned an action with shape "
                f"{action.shape}, expected {self.action_space.shape}"
            )
        if not np.isfinite(action).all():
            raise FloatingPointError("frozen policy action contains NaN or Inf")
        return np.clip(action, self.action_space.low, self.action_space.high)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._raw_observation = np.asarray(observation, dtype=np.float32).copy()
        return observation, info

    def step(self, action: np.ndarray):
        if self._raw_observation is None:
            raise RuntimeError("environment must be reset before step")
        residual = np.asarray(action, dtype=np.float32).reshape(-1)
        if residual.shape != self.action_space.shape:
            raise ValueError(
                f"expected a {self.action_space.shape[0]}-D residual action, "
                f"got {residual.shape}"
            )
        if not np.isfinite(residual).all():
            raise FloatingPointError("residual action contains NaN or Inf")
        residual = np.clip(residual, self.action_space.low, self.action_space.high)
        frozen = self._frozen_action(self._raw_observation)
        activation_factor = self._residual_activation_factor()
        scaled_residual = (
            self.residual_action_scale * activation_factor * residual
        )
        combined = np.clip(
            frozen + scaled_residual,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)
        observation, reward, terminated, truncated, info = self.env.step(combined)
        penalty = self.residual_action_penalty_scale * float(
            np.mean(np.square(activation_factor * residual))
        )
        info = dict(info)
        info.update(
            {
                "frozen_base_action": frozen.copy(),
                "residual_policy_action": residual.copy(),
                "residual_scaled_action": scaled_residual.copy(),
                "combined_policy_action": combined.copy(),
                "residual_action_scale": self.residual_action_scale,
                "residual_activation_factor": activation_factor,
                "residual_stall_activation_steps": (
                    self.residual_stall_activation_steps
                ),
                "residual_action_penalty": penalty,
            }
        )
        self._raw_observation = np.asarray(observation, dtype=np.float32).copy()
        return observation, float(reward) - penalty, terminated, truncated, info
