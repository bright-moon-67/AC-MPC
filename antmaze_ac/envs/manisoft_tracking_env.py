from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from manisoft.envs.vlm_env import VisionLanguageManipulationEnvironment
from manisoft.muscle import SplineMuscle
from manisoft.utils import (
    KOOPMAN_PHYSICAL_STATE_DIM,
    KOOPMAN_TIP_POSITION_SLICE,
    koopman_section_state,
    load_yaml,
)

from .delta_action_wrapper import DeltaActionWrapper


class ManiSoftTipTrackingEnv(gym.Env):
    """固定目标的ManiSoft末端跟踪环境，输出45维物理状态。"""

    def __init__(
        self,
        scenario_path: str | Path,
        target_offset: Sequence[float] = (0.0, 0.005, 0.0),
        target_tip: Sequence[float] | None = None,
        episode_steps: int = 300,
        success_threshold: float = 0.0015,
        success_streak: int = 5,
        absolute_action_limit: float = 0.30,
    ):
        super().__init__()

        self.scenario_path = Path(scenario_path).resolve()
        self.target_offset = np.asarray(target_offset, dtype=np.float32)
        self.fixed_target_tip = (
            None
            if target_tip is None
            else np.asarray(target_tip, dtype=np.float32)
        )

        if self.target_offset.shape != (3,):
            raise ValueError("target_offset必须是3维")
        if self.fixed_target_tip is not None and self.fixed_target_tip.shape != (3,):
            raise ValueError("target_tip必须是3维")
        if self.fixed_target_tip is None and np.linalg.norm(self.target_offset) <= 0:
            raise ValueError("target_offset不能为零")

        self.episode_steps = int(episode_steps)
        self.success_threshold = float(success_threshold)
        self.required_success_streak = int(success_streak)
        self.absolute_action_limit = float(absolute_action_limit)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(KOOPMAN_PHYSICAL_STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-self.absolute_action_limit,
            high=self.absolute_action_limit,
            shape=(18,),
            dtype=np.float32,
        )

        self.sim = None
        self.muscle = None
        self.target_tip = None
        self.previous_distance = None
        self.target_scale = None
        self.step_count = 0
        self.success_count = 0

    def _physical_state(self) -> np.ndarray:
        soft = self.sim._backend.softrobot_state
        state = koopman_section_state(soft)

        if state.shape != (KOOPMAN_PHYSICAL_STATE_DIM,):
            raise RuntimeError(f"错误状态维度：{state.shape}")
        if not np.isfinite(state).all():
            raise FloatingPointError("ManiSoft状态出现NaN或Inf")

        return state

    @staticmethod
    def _tip_position(state: np.ndarray) -> np.ndarray:
        return state[KOOPMAN_TIP_POSITION_SLICE]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        self.sim = None
        self.muscle = None
        gc.collect()

        config = load_yaml(self.scenario_path)
        config.pop("renderer", None)

        self.sim = VisionLanguageManipulationEnvironment.from_dict(config)

        soft = self.sim._backend.softrobot_state
        self.muscle = SplineMuscle(
            robot_length=float(np.sum(soft.element_lengths)),
            robot_num_elements=int(soft.num_elements),
            number_of_control_points=6,
            muscle_torque_scale=30,
        )

        # Settle with zero activation for 1 s at 50 Hz, matching the Koopman
        # data collector (settle-seconds=1).  This relaxes the rod to the
        # gravitational equilibrium the model was trained on; without it the
        # closed-loop starts from the upright initial pose (out of
        # distribution) and the MPC diverges.
        self.muscle.set_activation(np.zeros((6, 3), dtype=np.float64))

        def zero_torque(element_lengths: np.ndarray) -> np.ndarray:
            return self.muscle.evaluate(element_lengths)

        for _ in range(50):
            self.sim.step_with_torque_callback(zero_torque)

        observation = self._physical_state()
        self.target_tip = (
            self._tip_position(observation) + self.target_offset
            if self.fixed_target_tip is None
            else self.fixed_target_tip
        ).astype(np.float32)

        self.previous_distance = float(
            np.linalg.norm(
                self._tip_position(observation) - self.target_tip
            )
        )
        # Offset tasks retain their original normalization.  An explicit
        # reference tip may be much farther away, so normalize progress and
        # distance reward by its actual reset distance instead of the default
        # 5 mm offset.
        self.target_scale = max(
            float(np.linalg.norm(self.target_offset))
            if self.fixed_target_tip is None
            else self.previous_distance,
            np.finfo(np.float32).eps,
        )
        self.step_count = 0
        self.success_count = 0

        return observation, {
            "target_tip": self.target_tip.copy(),
            "distance": self.previous_distance,
        }

    def step(self, absolute_action: np.ndarray):
        action = np.asarray(
            absolute_action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape != (18,):
            raise ValueError(f"动作维度错误：{action.shape}")
        if not np.isfinite(action).all():
            raise FloatingPointError("动作出现NaN或Inf")

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        self.muscle.set_activation(action.reshape(6, 3))

        def current_torque(element_lengths: np.ndarray) -> np.ndarray:
            return self.muscle.evaluate(element_lengths)

        # Match the 50 Hz Koopman data collector: keep the activation fixed
        # while refreshing the length-dependent distributed torque at every
        # physics substep.
        self.sim.step_with_torque_callback(current_torque)

        observation = self._physical_state()
        distance = float(
            np.linalg.norm(
                self._tip_position(observation) - self.target_tip
            )
        )

        if self.target_scale is None:
            raise RuntimeError("Environment must be reset before step")
        target_scale = self.target_scale
        progress = (self.previous_distance - distance) / target_scale

        # 有界奖励，避免距离增大时产生过大的负回报。
        distance_ratio = distance / target_scale
        bounded_progress = float(np.clip(progress, -0.2, 0.2))
        normalized_action = action / self.absolute_action_limit

        reward = (
            float(np.exp(-distance_ratio) - np.exp(-1.0))
            + 2.0 * bounded_progress
            - 0.01 * float(np.mean(normalized_action ** 2))
        )

        self.step_count += 1

        if distance <= self.success_threshold:
            self.success_count += 1
        else:
            self.success_count = 0

        terminated = (
            self.success_count >= self.required_success_streak
        )
        truncated = (
            self.step_count >= self.episode_steps and not terminated
        )

        if terminated:
            reward += 5.0

        self.previous_distance = distance

        info = {
            "distance": distance,
            "target_tip": self.target_tip.copy(),
            "is_success": bool(terminated),
            "success_streak": self.success_count,
        }

        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.sim = None
        self.muscle = None
        gc.collect()


def make_manisoft_tracking_env(
    scenario_path: str | Path,
    *,
    target_offset=(0.0, 0.005, 0.0),
    target_tip=None,
    episode_steps: int = 300,
    absolute_action_limit: float = 0.30,
):
    base_env = ManiSoftTipTrackingEnv(
        scenario_path,
        target_offset=target_offset,
        target_tip=target_tip,
        episode_steps=episode_steps,
        absolute_action_limit=absolute_action_limit,
    )
    return DeltaActionWrapper(
        base_env,
        expected_observation_dim=KOOPMAN_PHYSICAL_STATE_DIM,
    )
