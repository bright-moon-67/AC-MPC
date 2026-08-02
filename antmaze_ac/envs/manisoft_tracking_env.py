from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from manisoft.envs.vlm_env import VisionLanguageManipulationEnvironment
from manisoft.muscle import SplineMuscle
from manisoft.utils import Rotation, load_yaml

from .delta_action_wrapper import DeltaActionWrapper


class ManiSoftTipTrackingEnv(gym.Env):
    """固定目标的ManiSoft末端跟踪环境，输出41维物理状态。"""

    def __init__(
        self,
        scenario_path: str | Path,
        target_offset: Sequence[float] = (0.0, 0.005, 0.0),
        episode_steps: int = 300,
        success_threshold: float = 0.0015,
        success_streak: int = 5,
        absolute_action_limit: float = 0.30,
    ):
        super().__init__()

        self.scenario_path = Path(scenario_path).resolve()
        self.target_offset = np.asarray(target_offset, dtype=np.float32)

        if self.target_offset.shape != (3,):
            raise ValueError("target_offset必须是3维")
        if np.linalg.norm(self.target_offset) <= 0:
            raise ValueError("target_offset不能为零")

        self.episode_steps = int(episode_steps)
        self.success_threshold = float(success_threshold)
        self.required_success_streak = int(success_streak)
        self.absolute_action_limit = float(absolute_action_limit)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(41,),
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
        self.step_count = 0
        self.success_count = 0

    def _physical_state(self) -> np.ndarray:
        soft = self.sim._backend.softrobot_state

        positions = np.asarray(
            soft.element_positions,
            dtype=np.float64,
        ).T

        sample_step = int(soft.num_elements) // 10

        compact_positions = np.concatenate(
            (
                positions[0, ::sample_step],
                positions[1, ::sample_step],
                positions[2, ::sample_step],
            )
        )

        tip_velocity = np.asarray(
            soft.element_velocities,
            dtype=np.float64,
        ).T[..., -1]

        speed = np.asarray([np.linalg.norm(tip_velocity)])

        if speed[0] > 0:
            velocity_direction = tip_velocity / speed[0]
        else:
            velocity_direction = np.zeros(3)

        tip_quaternion = np.asarray(
            Rotation.from_directions(
                soft.element_directors[-1]
            ).to_wxyz()
        )

        state = np.concatenate(
            (
                compact_positions,
                speed,
                velocity_direction,
                tip_quaternion,
            )
        ).astype(np.float32)

        if state.shape != (41,):
            raise RuntimeError(f"错误状态维度：{state.shape}")
        if not np.isfinite(state).all():
            raise FloatingPointError("ManiSoft状态出现NaN或Inf")

        return state

    @staticmethod
    def _tip_position(state: np.ndarray) -> np.ndarray:
        return state[[10, 21, 32]]

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
            muscle_torque_scale=75,
        )

        observation = self._physical_state()
        self.target_tip = (
            self._tip_position(observation) + self.target_offset
        ).astype(np.float32)

        self.previous_distance = float(
            np.linalg.norm(
                self._tip_position(observation) - self.target_tip
            )
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

        soft = self.sim._backend.softrobot_state

        torque = self.muscle.activate(
            action.reshape(6, 3),
            soft.element_lengths,
        )
        self.sim.step(list(map(tuple, torque)))

        observation = self._physical_state()
        distance = float(
            np.linalg.norm(
                self._tip_position(observation) - self.target_tip
            )
        )

        target_scale = float(np.linalg.norm(self.target_offset))
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


class TipPositionErrorObservationWrapper(gym.ObservationWrapper):
    """Append normalized 3-D tip-position error after the 59-D dynamics state."""

    def __init__(self, env: gym.Env, target_scale: float):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("末端误差包装器要求Box观测空间")
        if env.observation_space.shape != (59,):
            raise ValueError(
                f"末端误差包装器要求59维基础状态，实际为{env.observation_space.shape}"
            )
        if target_scale <= 0:
            raise ValueError("target_scale必须为正数")

        self.target_scale = float(target_scale)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                (env.observation_space.low, np.full(3, -np.inf, dtype=np.float32))
            ),
            high=np.concatenate(
                (env.observation_space.high, np.full(3, np.inf, dtype=np.float32))
            ),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (59,):
            raise ValueError(f"错误基础状态维度：{state.shape}")

        target_tip = self.env.unwrapped.target_tip
        if target_tip is None:
            raise RuntimeError("ManiSoft环境尚未设置目标末端位置")
        tip_position = state[[10, 21, 32]]
        tip_error = (
            np.asarray(target_tip, dtype=np.float32) - tip_position
        ) / self.target_scale
        policy_state = np.concatenate((state, tip_error)).astype(
            np.float32,
            copy=False,
        )
        if policy_state.shape != (62,):
            raise RuntimeError(f"错误策略状态维度：{policy_state.shape}")
        if not np.isfinite(policy_state).all():
            raise FloatingPointError("策略状态出现NaN或Inf")
        return policy_state


def make_manisoft_tracking_env(
    scenario_path: str | Path,
    *,
    target_offset=(0.0, 0.005, 0.0),
    episode_steps: int = 300,
    absolute_action_limit: float = 0.30,
):
    base_env = ManiSoftTipTrackingEnv(
        scenario_path,
        target_offset=target_offset,
        episode_steps=episode_steps,
        absolute_action_limit=absolute_action_limit,
    )
    delta_env = DeltaActionWrapper(
        base_env,
        expected_observation_dim=41,
    )
    return TipPositionErrorObservationWrapper(
        delta_env,
        target_scale=float(np.linalg.norm(base_env.target_offset)),
    )
