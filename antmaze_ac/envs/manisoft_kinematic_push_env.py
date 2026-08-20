from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from manisoft.utils import KOOPMAN_TIP_POSITION_SLICE, load_yaml

from .kinematic_push_task import (
    KINEMATIC_PUSH_TASK_CONTEXT_DIM,
    KinematicPushConfig,
    KinematicPushTask,
)
from .manisoft_tracking_env import ManiSoftTipTrackingEnv


class ManiSoftKinematicPushEnv(ManiSoftTipTrackingEnv):
    """Bare-arm obstacle-routing task with a force-free follower cube.

    The ManiSoft backend receives only the 18 muscle activations.  Obstacle
    collision, target contact, and cube motion live in :class:`KinematicPushTask`
    and never modify the PyElastica state.  Consequently every robot transition
    remains in the same object-free dynamics family used to train Koopman.
    """

    task_context_dim = KINEMATIC_PUSH_TASK_CONTEXT_DIM
    waypoint_count = 1

    def __init__(
        self,
        scenario_path: str | Path,
        *,
        episode_steps: int = 600,
        absolute_action_limit: float = 0.30,
    ) -> None:
        scenario_path = Path(scenario_path).expanduser().resolve()
        scenario = load_yaml(scenario_path)
        if "task" not in scenario:
            raise ValueError(f"Scenario has no task section: {scenario_path}")
        self.push_config = KinematicPushConfig.from_dict(scenario["task"])
        backend_dt = float(scenario["backend"]["dt"])
        update_interval = int(scenario["environment"]["update_interval"])
        actual_control_dt = backend_dt * update_interval
        if not np.isclose(actual_control_dt, self.push_config.control_dt):
            raise ValueError(
                "task.control_dt does not match backend.dt * update_interval: "
                f"{self.push_config.control_dt} != {actual_control_dt}"
            )
        super().__init__(
            scenario_path,
            target_tip=self.push_config.target_initial_center,
            episode_steps=episode_steps,
            success_threshold=self.push_config.goal_radius,
            success_streak=self.push_config.success_streak,
            absolute_action_limit=absolute_action_limit,
            progress_reward_scale=1.0,
        )
        self.task = KinematicPushTask(self.push_config)
        self._last_observation: np.ndarray | None = None

    @staticmethod
    def _tip_position(state: np.ndarray) -> np.ndarray:
        return np.asarray(state[KOOPMAN_TIP_POSITION_SLICE], dtype=np.float64)

    def _arm_nodes(self) -> np.ndarray:
        nodes = np.asarray(
            self.sim._backend.softrobot_state.element_positions,
            dtype=np.float64,
        )
        if nodes.ndim == 2 and nodes.shape[1] == 3:
            return nodes
        if nodes.ndim == 2 and nodes.shape[0] == 3:
            return nodes.T
        raise RuntimeError(f"Unexpected Elastica node layout: {nodes.shape}")

    @property
    def task_context(self) -> np.ndarray:
        if self._last_observation is None:
            raise RuntimeError("Environment must be reset before task_context")
        return self.task.context(self._tip_position(self._last_observation))

    @property
    def active_target_tip(self) -> np.ndarray:
        return self.task.active_tip_target.astype(np.float32)

    @property
    def target_center(self) -> np.ndarray:
        return self.task.target_center.astype(np.float32)

    @property
    def goal_center(self) -> np.ndarray:
        return self.task.goal_center.astype(np.float32)

    def trajectory_frame(self) -> dict[str, Any]:
        """Snapshot used by the evaluator and fast MuJoCo video renderer."""

        soft = self.sim._backend.softrobot_state
        return {
            "softrobot_positions": np.asarray(
                soft.element_positions, dtype=np.float32
            ).copy(),
            "softrobot_directors": np.asarray(
                soft.element_directors, dtype=np.float32
            ).copy(),
            "target_center": self.target_center.copy(),
            "goal_center": self.goal_center.copy(),
            "active_tip_target": self.active_target_tip.copy(),
            "phase": int(self.task.phase),
            "contact_locked": bool(self.task.contact_locked),
        }

    def _task_info(self, update: dict[str, Any] | None = None) -> dict[str, Any]:
        update = {} if update is None else update
        tip = self._tip_position(self._last_observation)
        return {
            "distance": float(
                update.get(
                    "active_distance",
                    np.linalg.norm(tip - self.task.active_tip_target),
                )
            ),
            "goal_distance": float(
                update.get("goal_distance", self.task.goal_distance)
            ),
            "target_tip": self.active_target_tip.copy(),
            "active_target_tip": self.active_target_tip.copy(),
            "target_center": self.target_center.copy(),
            "goal_center": self.goal_center.copy(),
            "phase": int(self.task.phase),
            "phase_name": self.task.phase_name,
            "waypoints_completed": int(self.task.phase),
            "route_side": int(self.task.route_side),
            "contact_locked": bool(self.task.contact_locked),
            "contact_event": bool(update.get("contact_event", False)),
            "collision": bool(update.get("collision", False)),
            "robot_collision": bool(update.get("robot_collision", False)),
            "tip_collision": bool(update.get("tip_collision", False)),
            "whole_arm_collision": bool(
                update.get("whole_arm_collision", False)
            ),
            "target_collision": bool(update.get("target_collision", False)),
            "is_success": bool(update.get("success", False)),
            "success_streak": int(self.task.success_count),
            "tip_in_work_band": bool(
                self.push_config.work_z_bounds[0]
                <= tip[2]
                <= self.push_config.work_z_bounds[1]
            ),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        observation, _ = super().reset(seed=seed, options=options)
        options = {} if options is None else dict(options)
        self.task.reset(
            self._tip_position(observation),
            route_side=options.get("route_side"),
            rng=self.np_random,
            target_center=options.get("target_center"),
            goal_center=options.get("goal_center"),
        )
        self._last_observation = np.asarray(observation, dtype=np.float32)
        self.target_tip = self.active_target_tip.copy()
        self.previous_distance = float(
            np.linalg.norm(
                self._tip_position(self._last_observation)
                - self.task.active_tip_target
            )
        )
        self.target_scale = max(self.previous_distance, np.finfo(np.float32).eps)
        self.step_count = 0
        self.success_count = 0
        return self._last_observation.copy(), self._task_info()

    def step(self, absolute_action: np.ndarray):
        action = np.asarray(absolute_action, dtype=np.float32).reshape(-1)
        if action.shape != (18,):
            raise ValueError(f"Expected an 18-D action, got {action.shape}")
        if not np.isfinite(action).all():
            raise FloatingPointError("Action contains NaN or Inf")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.muscle.set_activation(action.reshape(6, 3))

        def current_torque(element_lengths: np.ndarray) -> np.ndarray:
            return self.muscle.evaluate(element_lengths)

        self.sim.step_with_torque_callback(current_torque)
        observation = self._physical_state()
        normalized_action_energy = float(
            np.mean(np.square(action / self.absolute_action_limit))
        )
        update = self.task.update(
            self._tip_position(observation),
            self._arm_nodes(),
            normalized_action_energy,
        )
        self._last_observation = np.asarray(observation, dtype=np.float32)
        self.target_tip = self.active_target_tip.copy()
        self.previous_distance = float(update["active_distance"])
        self.success_count = self.task.success_count
        self.step_count += 1

        success = bool(update["success"])
        collision_termination = bool(
            update["collision"] and self.push_config.terminate_on_collision
        )
        terminated = success or collision_termination
        truncated = self.step_count >= self.episode_steps and not terminated
        return (
            self._last_observation.copy(),
            float(update["reward"]),
            terminated,
            truncated,
            self._task_info(update),
        )
