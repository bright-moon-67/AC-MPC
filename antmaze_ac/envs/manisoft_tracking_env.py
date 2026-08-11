from __future__ import annotations

import gc
import json
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


MANISOFT_WAYPOINT_REFERENCE_FILES = (
    "ref_4cm/reference.npz",
    "ref_8cm/reference.npz",
    "ref_12cm/reference.npz",
)
MANISOFT_WAYPOINT_ACTION_FILES = (
    "actions/u_scale_0p25.json",
    "actions/u_scale_0p50.json",
    "actions/u_scale_0p75.json",
)


def load_manisoft_waypoint_references(
    root: str | Path,
) -> tuple[np.ndarray, np.ndarray, tuple[Path, ...], tuple[Path, ...]]:
    """Load and cross-check the fixed 4/8/12 cm waypoint references."""

    root = Path(root).expanduser().resolve()
    reference_paths = tuple(root / name for name in MANISOFT_WAYPOINT_REFERENCE_FILES)
    action_paths = tuple(root / name for name in MANISOFT_WAYPOINT_ACTION_FILES)
    missing = [path for path in (*reference_paths, *action_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing ManiSoft waypoint reference files: "
            + ", ".join(map(str, missing))
        )

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for reference_path, action_path in zip(reference_paths, action_paths):
        with np.load(reference_path, allow_pickle=False) as archive:
            state = np.asarray(archive["reference_state"], dtype=np.float32).reshape(-1)
            action = np.asarray(archive["reference_action"], dtype=np.float32).reshape(-1)
        if state.shape != (KOOPMAN_PHYSICAL_STATE_DIM,) or action.shape != (18,):
            raise ValueError(
                f"{reference_path} must contain a 45-D state and 18-D action"
            )
        action_payload = json.loads(action_path.read_text(encoding="utf-8"))
        reproducible_action = np.asarray(
            action_payload.get("u"), dtype=np.float32
        ).reshape(-1)
        if reproducible_action.shape != (18,):
            raise ValueError(f"{action_path} must contain an 18-D 'u' action")
        if not np.allclose(action, reproducible_action, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"Reference action in {reference_path} does not match {action_path}"
            )
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{reference_path} contains NaN or Inf")
        states.append(state)
        actions.append(action)
    return (
        np.stack(states),
        np.stack(actions),
        reference_paths,
        action_paths,
    )


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


class ManiSoftThreeWaypointTrackingEnv(ManiSoftTipTrackingEnv):
    """按 4 cm -> 8 cm -> 12 cm 顺序连续跟踪三个固定末端目标。"""

    waypoint_count = 3

    def __init__(
        self,
        scenario_path: str | Path,
        waypoint_tips: Sequence[Sequence[float]] | np.ndarray,
        episode_steps: int = 300,
        success_threshold: float = 0.0015,
        success_streak: int = 5,
        waypoint_event_reward: float = 1.0,
        absolute_action_limit: float = 0.30,
    ) -> None:
        waypoints = np.asarray(waypoint_tips, dtype=np.float32)
        if waypoints.shape != (self.waypoint_count, 3):
            raise ValueError("waypoint_tips必须是[3,3]")
        if not np.isfinite(waypoints).all():
            raise ValueError("waypoint_tips出现NaN或Inf")
        if waypoint_event_reward < 0:
            raise ValueError("waypoint_event_reward必须非负")
        super().__init__(
            scenario_path,
            target_tip=waypoints[0],
            episode_steps=episode_steps,
            success_threshold=success_threshold,
            success_streak=success_streak,
            absolute_action_limit=absolute_action_limit,
        )
        self.fixed_waypoints = waypoints.copy()
        self.waypoint_event_reward = float(waypoint_event_reward)
        self.active_waypoint_index = 0
        self.waypoints_completed = 0

    @property
    def waypoints(self) -> np.ndarray:
        return self.fixed_waypoints

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self.active_waypoint_index = 0
        self.waypoints_completed = 0
        self.target_tip = self.fixed_waypoints[0].copy()
        tip = self._tip_position(observation)
        distances = np.linalg.norm(self.fixed_waypoints - tip[None, :], axis=1)
        self.previous_distance = float(distances[0])
        self.target_scale = max(self.previous_distance, np.finfo(np.float32).eps)
        info.update(
            {
                "target_tip": self.target_tip.copy(),
                "waypoints": self.fixed_waypoints.copy(),
                "active_waypoint_index": self.active_waypoint_index,
                "waypoints_completed": self.waypoints_completed,
                "waypoint_passed": False,
                "all_waypoint_distances": distances.astype(np.float32),
            }
        )
        return observation, info

    def step(self, absolute_action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(
            absolute_action
        )
        reached_distance = float(info["distance"])
        waypoint_passed = False

        # Intermediate goals use the same consecutive-hit condition as the
        # original single-goal task, but reaching them advances the stage
        # instead of terminating the episode.
        if terminated and self.active_waypoint_index < self.waypoint_count - 1:
            terminated = False
            waypoint_passed = True
            self.waypoints_completed += 1
            self.active_waypoint_index += 1
            self.success_count = 0
            self.target_tip = self.fixed_waypoints[
                self.active_waypoint_index
            ].copy()
            next_distance = float(
                np.linalg.norm(self._tip_position(observation) - self.target_tip)
            )
            self.previous_distance = next_distance
            self.target_scale = max(next_distance, np.finfo(np.float32).eps)
            # Replace the single-task terminal bonus with an intermediate
            # waypoint event reward.
            reward = reward - 5.0 + self.waypoint_event_reward
            truncated = self.step_count >= self.episode_steps

        if terminated:
            self.waypoints_completed = self.waypoint_count

        tip = self._tip_position(observation)
        distances = np.linalg.norm(self.fixed_waypoints - tip[None, :], axis=1)
        active_distance = float(distances[self.active_waypoint_index])
        info.update(
            {
                "distance": active_distance,
                "target_tip": self.target_tip.copy(),
                "waypoints": self.fixed_waypoints.copy(),
                "active_waypoint_index": self.active_waypoint_index,
                "waypoints_completed": self.waypoints_completed,
                "waypoint_passed": waypoint_passed,
                "success_streak": self.success_count,
                "reached_waypoint_distance": reached_distance,
                "all_waypoint_distances": distances.astype(np.float32),
                "is_success": bool(terminated),
            }
        )
        return observation, float(reward), terminated, truncated, info


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
