from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from manisoft.utils import KOOPMAN_PHYSICAL_STATE_DIM, KOOPMAN_TIP_POSITION_SLICE

from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.manisoft_wall_crossing_sac_env import (
    WallCrossingMetrics,
    wall_crossing_metrics,
)
from antmaze_ac.envs.table_entry_bank import restore_rod_internal_state


MANISOFT_TEACHER_TRACKING_OBSERVATION_DIM = 139


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(archive: np.lib.npyio.NpzFile, name: str, dtype):
    if name not in archive:
        raise ValueError(f"teacher episode is missing {name!r}")
    return dtype(np.asarray(archive[name]).reshape(()).item())


@dataclass(frozen=True)
class SmoothWallTeacherEpisode:
    path: Path
    scenario_sha256: str
    task_config_sha256: str
    control_dt: float
    route_side: int
    physical_states: np.ndarray
    actions: np.ndarray
    node_positions: np.ndarray
    node_velocities: np.ndarray
    element_directors: np.ndarray
    element_omegas: np.ndarray
    rod_internal_states: np.ndarray
    stage_ids: np.ndarray

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])


def load_smooth_wall_teacher_episode(
    path: str | Path,
) -> SmoothWallTeacherEpisode:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as archive:
        if _scalar(archive, "schema_version", int) != 1 or _scalar(
            archive, "kind", str
        ) != "manisoft_smooth_wall_teacher_episode":
            raise ValueError(f"unsupported smooth teacher episode: {resolved}")
        result = SmoothWallTeacherEpisode(
            path=resolved,
            scenario_sha256=_scalar(archive, "scenario_sha256", str),
            task_config_sha256=_scalar(archive, "task_config_sha256", str),
            control_dt=_scalar(archive, "control_dt", float),
            route_side=_scalar(archive, "route_side", int),
            physical_states=np.asarray(archive["physical_state"], dtype=np.float32),
            actions=np.asarray(archive["actions"], dtype=np.float32),
            node_positions=np.asarray(archive["node_positions"], dtype=np.float64),
            node_velocities=np.asarray(archive["node_velocities"], dtype=np.float64),
            element_directors=np.asarray(
                archive["element_directors"], dtype=np.float64
            ),
            element_omegas=np.asarray(archive["element_omegas"], dtype=np.float64),
            rod_internal_states=np.asarray(
                archive["rod_internal_state"], dtype=np.float64
            ),
            stage_ids=np.asarray(archive["stage_ids"], dtype=np.int8),
        )
    state_count = result.transition_count + 1
    expected = {
        "physical_states": (state_count, KOOPMAN_PHYSICAL_STATE_DIM),
        "node_positions": (state_count, 21, 3),
        "node_velocities": (state_count, 21, 3),
        "element_directors": (state_count, 20, 3, 3),
        "element_omegas": (state_count, 20, 3),
        "stage_ids": (state_count,),
    }
    for name, shape in expected.items():
        if getattr(result, name).shape != shape:
            raise ValueError(
                f"teacher {name} has shape {getattr(result, name).shape}, expected {shape}"
            )
    if result.actions.shape != (state_count - 1, 18):
        raise ValueError("teacher actions must have shape [transition,18]")
    if result.rod_internal_states.shape[0] != state_count:
        raise ValueError("teacher internal states do not align with physical states")
    for name in (
        "physical_states",
        "actions",
        "node_positions",
        "node_velocities",
        "element_directors",
        "element_omegas",
        "rod_internal_states",
    ):
        if not np.isfinite(getattr(result, name)).all():
            raise ValueError(f"teacher {name} contains NaN or Inf")
    if result.route_side not in {-1, 1}:
        raise ValueError("teacher route_side must be -1 or +1")
    return result


class ManiSoftTeacherTrackingSACEnv(ManiSoftTipTrackingEnv):
    """One SAC policy tracks the complete smooth wall-route teacher.

    Training resets may restore any certified teacher state so every portion
    of the 21.82 s motion appears early in replay.  The actor remains a single
    network and receives the same observation at every phase.  Evaluation
    uses ``reset_start_mode='upright'`` and therefore executes the entire
    route from state zero without stitching stage-specific policies.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_path: str | Path,
        *,
        task_config_path: str | Path,
        teacher_episode_path: str | Path,
        episode_steps: int = 320,
        reset_start_mode: str = "random_teacher_snapshot",
        upright_start_probability: float = 0.20,
        lookahead_steps: int = 15,
        absolute_action_limit: float = 0.60,
        muscle_torque_scale: float = 45.0,
        max_action_delta: float = 0.003,
        teacher_residual_action_scale: float = 0.03,
        tracking_reward_scale: float = 1.0,
        node_tracking_scale: float = 0.035,
        tip_tracking_scale: float = 0.025,
        velocity_tracking_scale: float = 0.20,
        velocity_tracking_weight: float = 0.15,
        action_imitation_penalty_scale: float = 0.05,
        action_rate_penalty_scale: float = 0.01,
        near_wall_clearance: float = 0.02,
        near_wall_penalty_scale: float = 0.20,
        collision_penalty: float = 50.0,
        lost_tracking_penalty: float = 20.0,
        maximum_node_tracking_error: float = 0.15,
        maximum_tip_tracking_error: float = 0.20,
        terminal_bonus: float = 50.0,
        terminal_tip_tolerance: float = 0.015,
        terminal_node_tolerance: float = 0.025,
        terminal_plane_tolerance: float = 0.005,
        terminal_minimum_crossed_fraction: float = 0.50,
        terminal_maximum_tip_speed: float = 0.12,
        arch_height_target: float | None = None,
        arch_y_margin: float = 0.05,
        arch_enforcement_start_progress: float = 0.75,
        arch_deficit_penalty_scale: float = 0.0,
    ) -> None:
        scenario = Path(scenario_path).expanduser().resolve()
        self.task_config_path = Path(task_config_path).expanduser().resolve()
        self.teacher = load_smooth_wall_teacher_episode(teacher_episode_path)
        if _sha256(scenario) != self.teacher.scenario_sha256:
            raise ValueError("teacher episode was generated from a different scenario")
        if _sha256(self.task_config_path) != self.teacher.task_config_sha256:
            raise ValueError("teacher episode was generated from a different task config")
        task_payload = yaml.safe_load(self.task_config_path.read_text(encoding="utf-8"))
        self.geometry = WallRouteGeometry.from_dict(task_payload["task"])
        scenario_payload = yaml.safe_load(scenario.read_text(encoding="utf-8"))
        self.control_dt = float(scenario_payload["backend"]["dt"]) * int(
            scenario_payload["environment"]["update_interval"]
        )
        if not np.isclose(self.control_dt, self.teacher.control_dt, atol=1e-12, rtol=0):
            raise ValueError("teacher and scenario control time steps differ")
        if reset_start_mode not in {"random_teacher_snapshot", "upright"}:
            raise ValueError("reset_start_mode must be random_teacher_snapshot or upright")
        if not 0 <= upright_start_probability <= 1:
            raise ValueError("upright_start_probability must lie in [0,1]")
        if lookahead_steps < 1 or episode_steps < 1:
            raise ValueError("lookahead_steps and episode_steps must be positive")
        positive = (
            absolute_action_limit,
            muscle_torque_scale,
            max_action_delta,
            teacher_residual_action_scale,
            tracking_reward_scale,
            node_tracking_scale,
            tip_tracking_scale,
            velocity_tracking_scale,
            maximum_node_tracking_error,
            maximum_tip_tracking_error,
            terminal_tip_tolerance,
            terminal_node_tolerance,
            terminal_plane_tolerance,
            terminal_minimum_crossed_fraction,
            terminal_maximum_tip_speed,
        )
        if min(positive) <= 0:
            raise ValueError("teacher tracking scales and limits must be positive")
        if max_action_delta > absolute_action_limit:
            raise ValueError("max_action_delta cannot exceed the action limit")
        if terminal_minimum_crossed_fraction > 1:
            raise ValueError("terminal crossing fraction cannot exceed one")
        if arch_height_target is not None and arch_height_target <= 0:
            raise ValueError("arch_height_target must be positive or null")
        if arch_y_margin < 0:
            raise ValueError("arch_y_margin must be non-negative")
        if not 0 <= arch_enforcement_start_progress <= 1:
            raise ValueError("arch_enforcement_start_progress must lie in [0,1]")
        if min(
            velocity_tracking_weight,
            action_imitation_penalty_scale,
            action_rate_penalty_scale,
            near_wall_clearance,
            near_wall_penalty_scale,
            collision_penalty,
            lost_tracking_penalty,
            terminal_bonus,
            arch_deficit_penalty_scale,
        ) < 0:
            raise ValueError("reward coefficients must be non-negative")
        if np.max(np.abs(self.teacher.actions)) > absolute_action_limit + 1e-7:
            raise ValueError("environment action limit is below the teacher action")

        super().__init__(
            scenario,
            target_tip=self.geometry.target,
            episode_steps=episode_steps,
            absolute_action_limit=absolute_action_limit,
            muscle_torque_scale=muscle_torque_scale,
        )
        self.reset_start_mode = reset_start_mode
        self.upright_start_probability = float(upright_start_probability)
        self.lookahead_steps = int(lookahead_steps)
        self.max_action_delta = float(max_action_delta)
        self.teacher_residual_action_scale = float(teacher_residual_action_scale)
        self.tracking_reward_scale = float(tracking_reward_scale)
        self.node_tracking_scale = float(node_tracking_scale)
        self.tip_tracking_scale = float(tip_tracking_scale)
        self.velocity_tracking_scale = float(velocity_tracking_scale)
        self.velocity_tracking_weight = float(velocity_tracking_weight)
        self.action_imitation_penalty_scale = float(action_imitation_penalty_scale)
        self.action_rate_penalty_scale = float(action_rate_penalty_scale)
        self.near_wall_clearance = float(near_wall_clearance)
        self.near_wall_penalty_scale = float(near_wall_penalty_scale)
        self.collision_penalty = float(collision_penalty)
        self.lost_tracking_penalty = float(lost_tracking_penalty)
        self.maximum_node_tracking_error = float(maximum_node_tracking_error)
        self.maximum_tip_tracking_error = float(maximum_tip_tracking_error)
        self.terminal_bonus = float(terminal_bonus)
        self.terminal_tip_tolerance = float(terminal_tip_tolerance)
        self.terminal_node_tolerance = float(terminal_node_tolerance)
        self.terminal_plane_tolerance = float(terminal_plane_tolerance)
        self.terminal_minimum_crossed_fraction = float(
            terminal_minimum_crossed_fraction
        )
        self.terminal_maximum_tip_speed = float(terminal_maximum_tip_speed)
        self.arch_height_target = (
            None if arch_height_target is None else float(arch_height_target)
        )
        self.arch_y_margin = float(arch_y_margin)
        self.arch_enforcement_start_progress = float(
            arch_enforcement_start_progress
        )
        self.arch_deficit_penalty_scale = float(arch_deficit_penalty_scale)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(MANISOFT_TEACHER_TRACKING_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.reference_index = 0
        self.start_index = 0
        self.previous_action = np.zeros(18, dtype=np.float32)
        self.previous_policy_action = np.zeros(18, dtype=np.float32)
        self.last_observation = np.zeros(
            MANISOFT_TEACHER_TRACKING_OBSERVATION_DIM, dtype=np.float32
        )

    def _rod_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        rod = self.sim._backend._softrobot
        return (
            rod.position_collection.T.astype(np.float64, copy=False),
            rod.velocity_collection.T.astype(np.float64, copy=False),
        )

    def _metrics(self) -> WallCrossingMetrics:
        nodes, velocities = self._rod_arrays()
        return wall_crossing_metrics(
            self.geometry, nodes, velocities, self.teacher.route_side
        )

    def _arch_height(self) -> float:
        nodes, _ = self._rod_arrays()
        checked = nodes[self.geometry.mounting_exempt_nodes :]
        mask = (
            (checked[:, 1] >= self.geometry.wall_minimum[1] - self.arch_y_margin)
            & (checked[:, 1] <= self.geometry.wall_maximum[1] + self.arch_y_margin)
        )
        return float(np.min(checked[mask, 2])) if np.any(mask) else float("nan")

    def _select_start_index(self, options: dict[str, Any]) -> int:
        supplied = options.get("start_index")
        if supplied is not None:
            index = int(supplied)
        elif self.reset_start_mode == "upright" or (
            self.upright_start_probability > 0
            and self.np_random.random() < self.upright_start_probability
        ):
            index = 0
        else:
            index = int(self.np_random.integers(0, self.teacher.transition_count))
        if not 0 <= index < self.teacher.transition_count:
            raise ValueError("start_index must select a teacher transition")
        return index

    def _restore_teacher_state(self, index: int) -> np.ndarray:
        rod = self.sim._backend._softrobot
        teacher = self.teacher
        rod.position_collection[...] = teacher.node_positions[index].T
        rod.velocity_collection[...] = teacher.node_velocities[index].T
        rod.director_collection[...] = teacher.element_directors[index].transpose(1, 2, 0)
        rod.omega_collection[...] = teacher.element_omegas[index].T
        restore_rod_internal_state(rod, teacher.rod_internal_states[index])
        self.sim._backend.time_tracker += index * self.control_dt
        self.sim.current_step += index * int(round(self.control_dt / self.sim._backend.dt))
        self.previous_action = (
            np.zeros(18, dtype=np.float32)
            if index == 0
            else teacher.actions[index - 1].copy()
        )
        self.muscle.set_activation(self.previous_action.reshape(6, 3))
        state = np.asarray(self._physical_state(), dtype=np.float32)
        error = float(np.max(np.abs(state - teacher.physical_states[index])))
        if error > 5e-4:
            raise RuntimeError(
                f"teacher snapshot restore mismatch at {index}: {error:.3e}"
            )
        return state

    def _tracking_errors(self, index: int) -> tuple[float, float, float]:
        nodes, velocities = self._rod_arrays()
        node_error = nodes - self.teacher.node_positions[index]
        velocity_error = velocities - self.teacher.node_velocities[index]
        node_rmse = float(np.sqrt(np.mean(np.sum(node_error**2, axis=1))))
        tip_error = float(np.linalg.norm(node_error[-1]))
        velocity_rmse = float(
            np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))
        )
        return node_rmse, tip_error, velocity_rmse

    def _observation(
        self, physical_state: np.ndarray, metrics: WallCrossingMetrics
    ) -> np.ndarray:
        index = self.reference_index
        lookahead = min(index + self.lookahead_steps, self.teacher.transition_count)
        reference_error = self.teacher.physical_states[index] - physical_state
        lookahead_tip_error = (
            self.teacher.node_positions[lookahead, -1]
            - self._rod_arrays()[0][-1]
        )
        phase = index / self.teacher.transition_count
        length_scale = max(
            float(np.linalg.norm(self.geometry.target - self.geometry.base)), 1e-6
        )
        clearance_scale = max(self.geometry.arm_radius, 1e-6)
        features = np.asarray(
            [
                float(self.teacher.route_side),
                metrics.side_gate_margin / length_scale,
                metrics.tip_beyond_distance / length_scale,
                metrics.threading_score,
                metrics.distal_crossed_fraction,
                metrics.wall_clearance / clearance_scale,
                metrics.ground_clearance / clearance_scale,
                metrics.tip_speed / 0.30,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate(
            (
                np.asarray(physical_state, dtype=np.float32),
                self.previous_action,
                np.asarray(reference_error, dtype=np.float32),
                np.asarray(lookahead_tip_error, dtype=np.float32),
                self.teacher.actions[
                    min(index, self.teacher.transition_count - 1)
                ]
                / self.absolute_action_limit,
                np.asarray((phase, 1.0 - phase), dtype=np.float32),
                features,
            )
        ).astype(np.float32, copy=False)
        if observation.shape != (MANISOFT_TEACHER_TRACKING_OBSERVATION_DIM,):
            raise RuntimeError(f"unexpected teacher observation shape {observation.shape}")
        if not np.isfinite(observation).all():
            raise FloatingPointError("teacher observation contains NaN or Inf")
        return observation

    def teacher_observation_batch(self) -> np.ndarray:
        """Return exact on-teacher observations for actor BC initialization."""

        rows = []
        previous_reference = self.reference_index
        previous_action = self.previous_action.copy()
        try:
            for index in range(self.teacher.transition_count):
                self.reference_index = index
                self.previous_action = (
                    np.zeros(18, dtype=np.float32)
                    if index == 0
                    else self.teacher.actions[index - 1].copy()
                )
                metrics = wall_crossing_metrics(
                    self.geometry,
                    self.teacher.node_positions[index],
                    self.teacher.node_velocities[index],
                    self.teacher.route_side,
                )
                physical = self.teacher.physical_states[index]
                lookahead = min(
                    index + self.lookahead_steps, self.teacher.transition_count
                )
                reference_error = np.zeros_like(physical)
                lookahead_tip_error = (
                    self.teacher.node_positions[lookahead, -1]
                    - self.teacher.node_positions[index, -1]
                )
                phase = index / self.teacher.transition_count
                length_scale = max(
                    float(np.linalg.norm(self.geometry.target - self.geometry.base)),
                    1e-6,
                )
                clearance_scale = max(self.geometry.arm_radius, 1e-6)
                features = np.asarray(
                    [
                        float(self.teacher.route_side),
                        metrics.side_gate_margin / length_scale,
                        metrics.tip_beyond_distance / length_scale,
                        metrics.threading_score,
                        metrics.distal_crossed_fraction,
                        metrics.wall_clearance / clearance_scale,
                        metrics.ground_clearance / clearance_scale,
                        metrics.tip_speed / 0.30,
                    ],
                    dtype=np.float32,
                )
                rows.append(
                    np.concatenate(
                        (
                            physical,
                            self.previous_action,
                            reference_error,
                            lookahead_tip_error.astype(np.float32),
                            self.teacher.actions[index]
                            / self.absolute_action_limit,
                            np.asarray((phase, 1.0 - phase), dtype=np.float32),
                            features,
                        )
                    )
                )
        finally:
            self.reference_index = previous_reference
            self.previous_action = previous_action
        result = np.asarray(rows, dtype=np.float32)
        if result.shape != (
            self.teacher.transition_count,
            MANISOFT_TEACHER_TRACKING_OBSERVATION_DIM,
        ):
            raise RuntimeError("teacher BC observation batch has an invalid shape")
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        options = {} if options is None else dict(options)
        self.start_index = self._select_start_index(options)
        self.reference_index = self.start_index
        state = self._restore_teacher_state(self.start_index)
        metrics = self._metrics()
        if (
            metrics.wall_clearance < 0
            or metrics.ground_clearance < -self.geometry.ground_violation_tolerance
        ):
            raise RuntimeError("certified teacher reset violates virtual geometry")
        self.step_count = 0
        self.previous_policy_action.fill(0.0)
        self.last_observation = self._observation(state, metrics)
        node_rmse, tip_error, velocity_rmse = self._tracking_errors(
            self.reference_index
        )
        return self.last_observation.copy(), self._info(
            metrics,
            node_rmse=node_rmse,
            tip_error=tip_error,
            velocity_rmse=velocity_rmse,
            is_success=False,
        )

    def _info(
        self,
        metrics: WallCrossingMetrics,
        *,
        node_rmse: float,
        tip_error: float,
        velocity_rmse: float,
        is_success: bool,
        termination_reason: str | None = None,
        teacher_action_error: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "is_success": bool(is_success),
            "termination_reason": termination_reason,
            "start_index": int(self.start_index),
            "reference_index": int(self.reference_index),
            "reference_progress": float(
                self.reference_index / self.teacher.transition_count
            ),
            "stage_id": int(self.teacher.stage_ids[self.reference_index]),
            "node_tracking_rmse": float(node_rmse),
            "tip_tracking_error": float(tip_error),
            "velocity_tracking_rmse": float(velocity_rmse),
            "teacher_action_error": float(teacher_action_error),
            "distal_crossed_fraction": metrics.distal_crossed_fraction,
            "wall_clearance": metrics.wall_clearance,
            "ground_clearance": metrics.ground_clearance,
            "tip_speed": metrics.tip_speed,
            "tip_x": metrics.tip_x,
            "target_plane_distance": abs(metrics.tip_x),
            "arch_height": self._arch_height(),
            "applied_action": self.previous_action.copy(),
        }

    def step(
        self, policy_action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        requested_normalized = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        if requested_normalized.shape != (18,) or not np.isfinite(
            requested_normalized
        ).all():
            raise ValueError("policy action must contain 18 finite values")
        requested_normalized = np.clip(requested_normalized, -1.0, 1.0)
        teacher_action = self.teacher.actions[
            min(self.reference_index, self.teacher.transition_count - 1)
        ]
        requested = (
            teacher_action
            + self.teacher_residual_action_scale * requested_normalized
        )
        applied = self.previous_action + np.clip(
            requested - self.previous_action,
            -self.max_action_delta,
            self.max_action_delta,
        )
        applied = np.clip(
            applied, -self.absolute_action_limit, self.absolute_action_limit
        ).astype(np.float32)
        action_error = float(
            np.sqrt(np.mean(((applied - teacher_action) / self.absolute_action_limit) ** 2))
        )
        action_rate = float(
            np.sqrt(
                np.mean(
                    ((applied - self.previous_action) / self.max_action_delta) ** 2
                )
            )
        )
        self.previous_action = applied
        self.previous_policy_action = requested_normalized
        self.muscle.set_activation(applied.reshape(6, 3))
        try:
            self.sim.step_with_torque_callback(
                lambda lengths: self.muscle.evaluate(lengths)
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            observation = self.last_observation.copy()
            info = {
                "is_success": False,
                "termination_reason": "dynamics_failure",
                "start_index": int(self.start_index),
                "reference_index": int(self.reference_index),
            }
            return observation, -self.collision_penalty, True, False, info

        self.reference_index = min(
            self.reference_index + 1, self.teacher.transition_count
        )
        self.step_count += 1
        state = np.asarray(self._physical_state(), dtype=np.float32)
        metrics = self._metrics()
        node_rmse, tip_error, velocity_rmse = self._tracking_errors(
            self.reference_index
        )
        tracking_exponent = (
            (node_rmse / self.node_tracking_scale) ** 2
            + (tip_error / self.tip_tracking_scale) ** 2
            + self.velocity_tracking_weight
            * (velocity_rmse / self.velocity_tracking_scale) ** 2
        )
        tracking_score = float(np.exp(-min(tracking_exponent, 80.0)))
        reward = (
            self.tracking_reward_scale * tracking_score
            - self.action_imitation_penalty_scale * action_error**2
            - self.action_rate_penalty_scale * action_rate**2
        )
        reference_progress = self.reference_index / self.teacher.transition_count
        arch_height = self._arch_height()
        if (
            self.arch_height_target is not None
            and reference_progress >= self.arch_enforcement_start_progress
            and np.isfinite(arch_height)
        ):
            reward -= self.arch_deficit_penalty_scale * max(
                0.0, self.arch_height_target - arch_height
            ) ** 2
        if metrics.wall_clearance < self.near_wall_clearance:
            reward -= self.near_wall_penalty_scale * (
                self.near_wall_clearance - metrics.wall_clearance
            ) / max(self.near_wall_clearance, 1e-9)

        unsafe_reason = None
        if metrics.wall_clearance < 0:
            unsafe_reason = "virtual_wall_collision"
        elif metrics.ground_clearance < -self.geometry.ground_violation_tolerance:
            unsafe_reason = "ground_violation"
        elif node_rmse > self.maximum_node_tracking_error:
            unsafe_reason = "node_tracking_lost"
        elif tip_error > self.maximum_tip_tracking_error:
            unsafe_reason = "tip_tracking_lost"
        if unsafe_reason in {"virtual_wall_collision", "ground_violation"}:
            reward -= self.collision_penalty
        elif unsafe_reason is not None:
            reward -= self.lost_tracking_penalty

        at_terminal = self.reference_index >= self.teacher.transition_count
        is_success = bool(
            unsafe_reason is None
            and at_terminal
            and tip_error <= self.terminal_tip_tolerance
            and node_rmse <= self.terminal_node_tolerance
            and abs(metrics.tip_x) <= self.terminal_plane_tolerance
            and metrics.distal_crossed_fraction
            >= self.terminal_minimum_crossed_fraction - 1e-8
            and metrics.tip_speed <= self.terminal_maximum_tip_speed
            and (
                self.arch_height_target is None
                or (
                    np.isfinite(arch_height)
                    and arch_height >= self.arch_height_target
                )
            )
        )
        if is_success:
            reward += self.terminal_bonus
        terminated = unsafe_reason is not None or is_success
        truncated = self.step_count >= self.episode_steps and not terminated
        termination_reason = "teacher_terminal_success" if is_success else unsafe_reason
        self.last_observation = self._observation(state, metrics)
        info = self._info(
            metrics,
            node_rmse=node_rmse,
            tip_error=tip_error,
            velocity_rmse=velocity_rmse,
            is_success=is_success,
            termination_reason=termination_reason,
            teacher_action_error=action_error,
        )
        return self.last_observation.copy(), float(reward), terminated, truncated, info
