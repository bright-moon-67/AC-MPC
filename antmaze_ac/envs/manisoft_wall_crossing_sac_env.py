from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from manisoft.utils import KOOPMAN_PHYSICAL_STATE_DIM

from antmaze_ac.data.wall_crossing_snapshot_bank import (
    WallCrossingSnapshotBank,
    load_wall_crossing_snapshot_bank,
)
from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.table_entry_bank import restore_rod_internal_state


MANISOFT_WALL_CROSSING_OBSERVATION_DIM = KOOPMAN_PHYSICAL_STATE_DIM + 18 + 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WallCrossingMetrics:
    wall_clearance: float
    ground_clearance: float
    tip_speed: float
    tip_beyond_distance: float
    side_gate_margin: float
    threading_score: float
    distal_crossed_fraction: float
    crossed_fraction: float
    tip_x: float


def wall_crossing_metrics(
    geometry: WallRouteGeometry,
    nodes: np.ndarray,
    node_velocities: np.ndarray,
    side: int,
) -> WallCrossingMetrics:
    """Measure continuous and discrete progress without using the goal point."""

    points = np.asarray(nodes, dtype=np.float64)
    velocities = np.asarray(node_velocities, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or velocities.shape != points.shape:
        raise ValueError("nodes and node_velocities must have matching shape [N,3]")
    if side not in {-1, 1}:
        raise ValueError("side must be -1 or +1")
    checked = points[geometry.mounting_exempt_nodes :]
    post_y = geometry.wall_maximum[1] + geometry.postwall_y_margin
    beyond = checked[:, 1] >= post_y
    distal_count = 0
    for value in beyond[::-1]:
        if not value:
            break
        distal_count += 1
    denominator = max(post_y - geometry.wall_minimum[1], 1e-9)
    threading = np.clip(
        (checked[:, 1] - geometry.wall_minimum[1]) / denominator,
        0.0,
        1.0,
    )
    return WallCrossingMetrics(
        wall_clearance=geometry.whole_arm_wall_clearance(points),
        ground_clearance=geometry.whole_arm_ground_clearance(points),
        tip_speed=float(np.linalg.norm(velocities[-1])),
        tip_beyond_distance=float(points[-1, 1] - post_y),
        side_gate_margin=float(
            side * (points[-1, 0] - geometry.side_gate_x(side))
        ),
        threading_score=float(np.mean(threading)),
        distal_crossed_fraction=float(distal_count / len(checked)),
        crossed_fraction=float(np.mean(beyond)),
        tip_x=float(points[-1, 0]),
    )


class ManiSoftWallCrossingSACEnv(ManiSoftTipTrackingEnv):
    """SAC subtask: move a contiguous distal portion safely beyond the wall.

    Every episode starts from a simulator-certified phase-2 moving snapshot:
    the tip has already rounded one finite x end of the virtual wall.  The
    policy controls bounded 18-D activation increments and succeeds when the
    configured distal body fraction remains beyond the far wall face.  No
    target-point or target-plane term appears in this subtask reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_path: str | Path,
        *,
        task_config_path: str | Path,
        snapshot_bank_path: str | Path,
        episode_steps: int = 300,
        absolute_action_limit: float = 0.30,
        muscle_torque_scale: float = 30.0,
        max_action_delta: float = 0.005,
        success_crossed_fraction: float = 0.30,
        success_streak: int = 5,
        maximum_tip_speed: float | None = 0.75,
        snapshot_minimum_fraction: float = 0.05,
        snapshot_maximum_fraction: float = 0.25,
        snapshot_progress_bias: float = 6.0,
        threading_progress_reward_scale: float = 60.0,
        crossed_progress_reward_scale: float = 20.0,
        crossing_deficit_penalty_scale: float = 0.0,
        required_crossing_margin: float = 0.0,
        crossing_margin_progress_reward_scale: float = 0.0,
        crossing_margin_deficit_penalty_scale: float = 0.0,
        tip_progress_reward_scale: float = 2.0,
        collision_penalty: float = 15.0,
        success_bonus: float = 15.0,
        near_wall_clearance: float = 0.01,
        near_wall_penalty_scale: float = 0.20,
        step_penalty: float = 0.01,
        policy_action_penalty_scale: float = 0.005,
        activation_penalty_scale: float = 0.001,
        return_tip_x_tolerance: float | None = None,
        return_reward_minimum_fraction: float | None = None,
        return_progress_reward_scale: float = 0.0,
        base_policy_model_path: str | Path | None = None,
        base_policy_vecnormalize_path: str | Path | None = None,
        base_policy_device: str = "cpu",
        residual_action_scale: float = 0.10,
        allowed_route_side: int | None = None,
        allowed_snapshot_indices: list[int] | tuple[int, ...] | None = None,
        base_policy_latch_off_fraction: float | None = None,
        latched_residual_action_scale: float | None = None,
    ) -> None:
        self.task_config_path = Path(task_config_path).expanduser().resolve()
        payload = yaml.safe_load(self.task_config_path.read_text(encoding="utf-8"))
        self.geometry = WallRouteGeometry.from_dict(payload["task"])
        self.snapshot_bank_path = Path(snapshot_bank_path).expanduser().resolve()
        self.snapshot_bank: WallCrossingSnapshotBank = (
            load_wall_crossing_snapshot_bank(self.snapshot_bank_path)
        )
        scenario = Path(scenario_path).expanduser().resolve()
        if self.snapshot_bank.scenario_sha256 != _sha256(scenario):
            raise ValueError("snapshot bank was generated from a different scenario")
        if self.snapshot_bank.collection_config_sha256 != _sha256(
            self.task_config_path
        ):
            raise ValueError("snapshot bank was generated from a different task config")
        if self.snapshot_bank.absolute_action_limit > absolute_action_limit + 1e-7:
            raise ValueError("environment action limit is below the snapshot bank limit")
        if not np.isclose(
            self.snapshot_bank.muscle_torque_scale,
            muscle_torque_scale,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("environment torque scale differs from snapshot bank")
        if not 0 < max_action_delta <= absolute_action_limit:
            raise ValueError("max_action_delta must lie in (0, action limit]")
        if not 0 < success_crossed_fraction <= 1:
            raise ValueError("success_crossed_fraction must lie in (0,1]")
        if success_streak < 1:
            raise ValueError("success_streak must be positive")
        if maximum_tip_speed is not None and maximum_tip_speed <= 0:
            raise ValueError("maximum_tip_speed must be positive or null")
        if return_tip_x_tolerance is not None and return_tip_x_tolerance <= 0:
            raise ValueError("return_tip_x_tolerance must be positive or null")
        if return_reward_minimum_fraction is not None and not (
            0 < return_reward_minimum_fraction <= success_crossed_fraction
        ):
            raise ValueError(
                "return_reward_minimum_fraction must lie in "
                "(0, success_crossed_fraction]"
            )
        if (base_policy_model_path is None) != (
            base_policy_vecnormalize_path is None
        ):
            raise ValueError(
                "base policy model and VecNormalize paths must be supplied together"
            )
        if base_policy_model_path is not None and not 0 < residual_action_scale <= 1:
            raise ValueError("residual_action_scale must lie in (0,1]")
        if allowed_route_side not in (None, -1, 1):
            raise ValueError("allowed_route_side must be null, -1 or +1")
        if base_policy_latch_off_fraction is not None and not (
            0 < base_policy_latch_off_fraction <= 1
        ):
            raise ValueError("base_policy_latch_off_fraction must lie in (0,1]")
        if latched_residual_action_scale is not None and not (
            0 < latched_residual_action_scale <= 1
        ):
            raise ValueError("latched_residual_action_scale must lie in (0,1]")
        valid_snapshot_curriculum = (
            0 < snapshot_minimum_fraction <= snapshot_maximum_fraction
            and (
                snapshot_maximum_fraction < success_crossed_fraction
                if return_tip_x_tolerance is None
                else snapshot_maximum_fraction
                <= success_crossed_fraction + 1e-8
            )
        )
        if not valid_snapshot_curriculum:
            raise ValueError(
                "crossing-only snapshots must start below the success fraction; "
                "return-stage snapshots may start at it"
            )
        if min(
            snapshot_progress_bias,
            threading_progress_reward_scale,
            crossed_progress_reward_scale,
            crossing_deficit_penalty_scale,
            required_crossing_margin,
            crossing_margin_progress_reward_scale,
            crossing_margin_deficit_penalty_scale,
            tip_progress_reward_scale,
            collision_penalty,
            success_bonus,
            near_wall_clearance,
            near_wall_penalty_scale,
            step_penalty,
            policy_action_penalty_scale,
            activation_penalty_scale,
            return_progress_reward_scale,
        ) < 0:
            raise ValueError("reward and curriculum settings must be non-negative")
        super().__init__(
            scenario,
            target_tip=self.geometry.target,
            episode_steps=episode_steps,
            absolute_action_limit=absolute_action_limit,
            muscle_torque_scale=muscle_torque_scale,
        )
        scenario_payload = yaml.safe_load(scenario.read_text(encoding="utf-8"))
        self.control_dt = float(scenario_payload["backend"]["dt"]) * int(
            scenario_payload["environment"]["update_interval"]
        )
        if not np.isclose(self.control_dt, self.snapshot_bank.control_dt):
            raise ValueError("snapshot bank and scenario control time steps differ")

        self.max_action_delta = float(max_action_delta)
        self.success_crossed_fraction = float(success_crossed_fraction)
        self.required_crossing_streak = int(success_streak)
        self.maximum_tip_speed = (
            None if maximum_tip_speed is None else float(maximum_tip_speed)
        )
        self.snapshot_minimum_fraction = float(snapshot_minimum_fraction)
        self.snapshot_maximum_fraction = float(snapshot_maximum_fraction)
        self.snapshot_progress_bias = float(snapshot_progress_bias)
        self.threading_progress_reward_scale = float(
            threading_progress_reward_scale
        )
        self.crossed_progress_reward_scale = float(crossed_progress_reward_scale)
        self.crossing_deficit_penalty_scale = float(
            crossing_deficit_penalty_scale
        )
        self.required_crossing_margin = float(required_crossing_margin)
        self.crossing_margin_progress_reward_scale = float(
            crossing_margin_progress_reward_scale
        )
        self.crossing_margin_deficit_penalty_scale = float(
            crossing_margin_deficit_penalty_scale
        )
        self.tip_progress_reward_scale = float(tip_progress_reward_scale)
        self.collision_penalty = float(collision_penalty)
        self.success_bonus = float(success_bonus)
        self.near_wall_clearance = float(near_wall_clearance)
        self.near_wall_penalty_scale = float(near_wall_penalty_scale)
        self.step_penalty = float(step_penalty)
        self.policy_action_penalty_scale = float(policy_action_penalty_scale)
        self.activation_penalty_scale = float(activation_penalty_scale)
        self.return_tip_x_tolerance = (
            None
            if return_tip_x_tolerance is None
            else float(return_tip_x_tolerance)
        )
        self.return_reward_minimum_fraction = float(
            success_crossed_fraction
            if return_reward_minimum_fraction is None
            else return_reward_minimum_fraction
        )
        self.return_progress_reward_scale = float(return_progress_reward_scale)
        self.base_policy_model_path = (
            None
            if base_policy_model_path is None
            else Path(base_policy_model_path).expanduser().resolve()
        )
        self.base_policy_vecnormalize_path = (
            None
            if base_policy_vecnormalize_path is None
            else Path(base_policy_vecnormalize_path).expanduser().resolve()
        )
        self.base_policy_device = str(base_policy_device)
        self.residual_action_scale = float(residual_action_scale)
        self.allowed_route_side = allowed_route_side
        self.allowed_snapshot_indices = (
            None
            if allowed_snapshot_indices is None
            else tuple(int(index) for index in allowed_snapshot_indices)
        )
        if self.allowed_snapshot_indices is not None:
            if not self.allowed_snapshot_indices or len(
                set(self.allowed_snapshot_indices)
            ) != len(self.allowed_snapshot_indices):
                raise ValueError("allowed_snapshot_indices must be nonempty and unique")
            if min(self.allowed_snapshot_indices) < 0 or max(
                self.allowed_snapshot_indices
            ) >= self.snapshot_bank.snapshot_count:
                raise ValueError("allowed_snapshot_indices contain an invalid index")
        self.base_policy_latch_off_fraction = base_policy_latch_off_fraction
        self.latched_residual_action_scale = (
            self.residual_action_scale
            if latched_residual_action_scale is None
            else float(latched_residual_action_scale)
        )
        self.base_policy_latched_off = False
        self.base_policy_model = None
        self.base_policy_normalizer = None
        if self.base_policy_model_path is not None:
            if not self.base_policy_model_path.is_file():
                raise FileNotFoundError(self.base_policy_model_path)
            if not self.base_policy_vecnormalize_path.is_file():
                raise FileNotFoundError(self.base_policy_vecnormalize_path)
            from stable_baselines3 import SAC
            from stable_baselines3.common.save_util import load_from_pkl

            self.base_policy_model = SAC.load(
                str(self.base_policy_model_path),
                device=self.base_policy_device,
                print_system_info=False,
            )
            self.base_policy_normalizer = load_from_pkl(
                self.base_policy_vecnormalize_path
            )
            if self.base_policy_model.observation_space.shape != (
                MANISOFT_WALL_CROSSING_OBSERVATION_DIM,
            ):
                raise ValueError("base policy has an incompatible observation space")
            if np.asarray(self.base_policy_normalizer.obs_rms.mean).shape != (
                MANISOFT_WALL_CROSSING_OBSERVATION_DIM,
            ):
                raise ValueError("base policy normalizer has an incompatible shape")

        eligible = np.flatnonzero(
            (self.snapshot_bank.crossed_fractions >= self.snapshot_minimum_fraction - 1e-8)
            & (self.snapshot_bank.crossed_fractions <= self.snapshot_maximum_fraction + 1e-8)
            & (
                True
                if self.allowed_route_side is None
                else self.snapshot_bank.route_sides == self.allowed_route_side
            )
            & (
                True
                if self.allowed_snapshot_indices is None
                else np.isin(
                    np.arange(self.snapshot_bank.snapshot_count),
                    self.allowed_snapshot_indices,
                )
            )
        )
        if len(eligible) < 1:
            raise ValueError("snapshot bank has no state in the requested curriculum")
        required_sides = (
            (-1, 1)
            if self.allowed_route_side is None
            else (self.allowed_route_side,)
        )
        if set(self.snapshot_bank.route_sides[eligible].tolist()) != set(
            required_sides
        ):
            raise ValueError("snapshot curriculum does not contain the requested sides")
        logits = self.snapshot_progress_bias * self.snapshot_bank.crossed_fractions[eligible]
        weights = np.exp(logits - np.max(logits)).astype(np.float64)
        # The raw candidate bank is left-heavy.  Normalize within each side so
        # neither homotopy can dominate the replay buffer merely because broad
        # collection happened to find it more often.
        for side in required_sides:
            side_mask = self.snapshot_bank.route_sides[eligible] == side
            side_total = float(np.sum(weights[side_mask]))
            if side_total <= 0:
                raise ValueError("snapshot curriculum must retain both sides")
            weights[side_mask] *= (1.0 / len(required_sides)) / side_total
        self.eligible_snapshot_indices = eligible
        self.snapshot_sampling_weights = weights / np.sum(weights)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(MANISOFT_WALL_CROSSING_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.previous_action = np.zeros(18, dtype=np.float32)
        self.previous_metrics: WallCrossingMetrics | None = None
        self.previous_required_crossing_margin = 0.0
        self.route_side = 1
        self.snapshot_index = -1
        self.crossing_success_count = 0
        self.last_base_policy_action = np.zeros(18, dtype=np.float32)
        self.last_residual_policy_action = np.zeros(18, dtype=np.float32)
        self.last_combined_policy_action = np.zeros(18, dtype=np.float32)
        self.last_observation = np.zeros(
            MANISOFT_WALL_CROSSING_OBSERVATION_DIM, dtype=np.float32
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
            self.geometry, nodes, velocities, self.route_side
        )

    def _required_distal_crossing_margin(self) -> float:
        """Continuous y margin of the critical node for the success fraction."""

        nodes, _ = self._rod_arrays()
        checked = nodes[self.geometry.mounting_exempt_nodes :]
        required_count = max(
            1,
            int(
                np.ceil(
                    self.success_crossed_fraction * len(checked) - 1e-9
                )
            ),
        )
        post_y = (
            self.geometry.wall_maximum[1]
            + self.geometry.postwall_y_margin
        )
        return float(np.min(checked[-required_count:, 1] - post_y))

    def _observation(
        self, physical_state: np.ndarray, metrics: WallCrossingMetrics
    ) -> np.ndarray:
        length_scale = max(
            float(np.linalg.norm(self.geometry.target - self.geometry.base)), 1e-6
        )
        clearance_scale = max(self.geometry.arm_radius, 1e-6)
        features = np.asarray(
            [
                float(self.route_side),
                metrics.side_gate_margin / length_scale,
                metrics.tip_beyond_distance / length_scale,
                metrics.threading_score,
                metrics.distal_crossed_fraction,
                metrics.crossed_fraction,
                metrics.wall_clearance / clearance_scale,
                metrics.ground_clearance / clearance_scale,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate(
            (
                np.asarray(physical_state, dtype=np.float32),
                self.previous_action.astype(np.float32, copy=False),
                features,
            )
        )
        if observation.shape != (MANISOFT_WALL_CROSSING_OBSERVATION_DIM,):
            raise RuntimeError(f"unexpected wall-crossing observation {observation.shape}")
        if not np.isfinite(observation).all():
            raise FloatingPointError("wall-crossing observation contains NaN or Inf")
        return observation

    def _select_snapshot(self, options: dict[str, Any]) -> int:
        supplied = options.get("snapshot_index")
        if supplied is None:
            return int(
                self.np_random.choice(
                    self.eligible_snapshot_indices,
                    p=self.snapshot_sampling_weights,
                )
            )
        index = int(supplied)
        if index not in self.eligible_snapshot_indices:
            raise ValueError("snapshot_index is outside the configured curriculum")
        return index

    def _restore_snapshot(self, index: int) -> np.ndarray:
        bank = self.snapshot_bank
        rod = self.sim._backend._softrobot
        rod.position_collection[...] = bank.node_positions[index].T
        rod.velocity_collection[...] = bank.node_velocities[index].T
        rod.director_collection[...] = bank.element_directors[index].transpose(1, 2, 0)
        rod.omega_collection[...] = bank.element_omegas[index].T
        restore_rod_internal_state(rod, bank.rod_internal_states[index])
        frame = int(bank.source_frames[index])
        self.sim._backend.time_tracker += frame * self.control_dt
        self.sim.current_step += frame * int(
            round(self.control_dt / self.sim._backend.dt)
        )
        self.previous_action = bank.previous_actions[index].copy()
        self.muscle.set_activation(self.previous_action.reshape(6, 3))
        state = np.asarray(self._physical_state(), dtype=np.float32)
        error = float(np.max(np.abs(state - bank.physical_states[index])))
        if error > 5e-4:
            raise RuntimeError(
                "wall-crossing snapshot restore mismatch: "
                f"max physical-state error {error:.3e}"
            )
        return state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        options = {} if options is None else dict(options)
        self.snapshot_index = self._select_snapshot(options)
        self.route_side = int(self.snapshot_bank.route_sides[self.snapshot_index])
        physical_state = self._restore_snapshot(self.snapshot_index)
        metrics = self._metrics()
        if (
            metrics.wall_clearance < 0
            or metrics.ground_clearance < -self.geometry.ground_violation_tolerance
        ):
            raise RuntimeError("certified reset snapshot violates virtual geometry")
        self.step_count = 0
        self.crossing_success_count = 0
        self.last_base_policy_action.fill(0.0)
        self.last_residual_policy_action.fill(0.0)
        self.last_combined_policy_action.fill(0.0)
        self.base_policy_latched_off = False
        self.previous_metrics = metrics
        self.previous_required_crossing_margin = (
            self._required_distal_crossing_margin()
        )
        self.last_observation = self._observation(physical_state, metrics)
        return self.last_observation.copy(), self._info(metrics, is_success=False)

    def _info(
        self,
        metrics: WallCrossingMetrics,
        *,
        is_success: bool,
        termination_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "is_success": bool(is_success),
            "termination_reason": termination_reason,
            "snapshot_index": int(self.snapshot_index),
            "snapshot_name": self.snapshot_bank.names[self.snapshot_index],
            "source_crossed_fraction": float(
                self.snapshot_bank.crossed_fractions[self.snapshot_index]
            ),
            "route_side": int(self.route_side),
            "distal_crossed_fraction": metrics.distal_crossed_fraction,
            "crossed_fraction": metrics.crossed_fraction,
            "threading_score": metrics.threading_score,
            "tip_beyond_distance": metrics.tip_beyond_distance,
            "side_gate_margin": metrics.side_gate_margin,
            "wall_clearance": metrics.wall_clearance,
            "ground_clearance": metrics.ground_clearance,
            "tip_speed": metrics.tip_speed,
            "tip_x": metrics.tip_x,
            "target_plane_distance": abs(
                metrics.tip_x - float(self.geometry.target[0])
            ),
            "required_crossing_margin": (
                self._required_distal_crossing_margin()
            ),
            "success_streak": int(self.crossing_success_count),
            "applied_action": self.previous_action.copy(),
            "base_policy_action": self.last_base_policy_action.copy(),
            "residual_policy_action": self.last_residual_policy_action.copy(),
            "combined_policy_action": self.last_combined_policy_action.copy(),
            "base_policy_latched_off": bool(self.base_policy_latched_off),
        }

    def _base_policy_action(self, observation: np.ndarray) -> np.ndarray:
        if self.base_policy_model is None or self.base_policy_normalizer is None:
            return np.zeros(18, dtype=np.float32)
        normalizer = self.base_policy_normalizer
        value = np.asarray(observation, dtype=np.float32)
        if bool(normalizer.norm_obs):
            value = np.clip(
                (value - normalizer.obs_rms.mean)
                / np.sqrt(normalizer.obs_rms.var + normalizer.epsilon),
                -normalizer.clip_obs,
                normalizer.clip_obs,
            ).astype(np.float32)
        action, _ = self.base_policy_model.predict(
            value, deterministic=True
        )
        return np.asarray(action, dtype=np.float32).reshape(18)

    def step(
        self, policy_action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual_requested = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        if residual_requested.shape != (18,) or not np.isfinite(
            residual_requested
        ).all():
            raise ValueError("policy action must contain 18 finite values")
        residual_requested = np.clip(residual_requested, -1.0, 1.0)
        if (
            self.base_policy_latch_off_fraction is not None
            and self.previous_metrics is not None
            and self.previous_metrics.distal_crossed_fraction
            >= self.base_policy_latch_off_fraction - 1e-8
        ):
            self.base_policy_latched_off = True
        base_requested = (
            np.zeros(18, dtype=np.float32)
            if self.base_policy_latched_off
            else self._base_policy_action(self.last_observation)
        )
        active_residual_scale = (
            self.latched_residual_action_scale
            if self.base_policy_latched_off
            else self.residual_action_scale
        )
        requested = np.clip(
            base_requested
            + (
                active_residual_scale * residual_requested
                if self.base_policy_model is not None
                else residual_requested
            ),
            -1.0,
            1.0,
        ).astype(np.float32)
        self.last_base_policy_action = base_requested.copy()
        self.last_residual_policy_action = residual_requested.copy()
        self.last_combined_policy_action = requested.copy()
        applied = np.clip(
            self.previous_action + self.max_action_delta * requested,
            -self.absolute_action_limit,
            self.absolute_action_limit,
        ).astype(np.float32)
        previous_metrics = self.previous_metrics
        previous_crossing_margin = self.previous_required_crossing_margin
        if previous_metrics is None:
            raise RuntimeError("environment must be reset before step")
        self.previous_action = applied
        self.muscle.set_activation(applied.reshape(6, 3))
        try:
            self.sim.step_with_torque_callback(
                lambda lengths: self.muscle.evaluate(lengths)
            )
            physical_state = self._physical_state()
            metrics = self._metrics()
            crossing_margin = self._required_distal_crossing_margin()
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            self.step_count += 1
            info = self._info(
                previous_metrics,
                is_success=False,
                termination_reason="dynamics_violation",
            )
            return (
                self.last_observation.copy(),
                -self.collision_penalty,
                True,
                False,
                info,
            )

        reward = (
            self.threading_progress_reward_scale
            * (metrics.threading_score - previous_metrics.threading_score)
            + self.crossed_progress_reward_scale
            * (
                metrics.distal_crossed_fraction
                - previous_metrics.distal_crossed_fraction
            )
            - self.crossing_deficit_penalty_scale
            * max(
                0.0,
                self.success_crossed_fraction
                - metrics.distal_crossed_fraction,
            )
            + self.crossing_margin_progress_reward_scale
            * (crossing_margin - previous_crossing_margin)
            - self.crossing_margin_deficit_penalty_scale
            * max(0.0, self.required_crossing_margin - crossing_margin)
            + self.tip_progress_reward_scale
            * (metrics.tip_beyond_distance - previous_metrics.tip_beyond_distance)
            - self.step_penalty
            - self.policy_action_penalty_scale
            * float(np.mean(residual_requested**2))
            - self.activation_penalty_scale
            * float(np.mean((applied / self.absolute_action_limit) ** 2))
        )
        if metrics.wall_clearance < self.near_wall_clearance:
            reward -= self.near_wall_penalty_scale * (
                self.near_wall_clearance - metrics.wall_clearance
            ) / self.near_wall_clearance

        # Only reward motion back toward the yz target plane after a
        # contiguous distal section is safely threaded through.  Gating this
        # term prevents the policy from taking the geometrically invalid
        # direct x=0 route through the wall.
        if (
            self.return_progress_reward_scale > 0
            and min(
                previous_metrics.distal_crossed_fraction,
                metrics.distal_crossed_fraction,
            )
            >= self.return_reward_minimum_fraction - 1e-8
            and previous_metrics.tip_beyond_distance >= 0
            and metrics.tip_beyond_distance >= 0
        ):
            target_x = float(self.geometry.target[0])
            reward += self.return_progress_reward_scale * (
                abs(previous_metrics.tip_x - target_x)
                - abs(metrics.tip_x - target_x)
            )

        self.step_count += 1
        unsafe_reason = None
        if metrics.wall_clearance < 0:
            unsafe_reason = "virtual_wall_collision"
        elif metrics.ground_clearance < -self.geometry.ground_violation_tolerance:
            unsafe_reason = "ground_violation"
        elif (
            self.maximum_tip_speed is not None
            and metrics.tip_speed > self.maximum_tip_speed
        ):
            unsafe_reason = "tip_speed"
        if unsafe_reason is not None:
            reward -= self.collision_penalty

        at_crossing_goal = bool(
            unsafe_reason is None
            and metrics.distal_crossed_fraction
            >= self.success_crossed_fraction - 1e-8
            and metrics.tip_beyond_distance >= 0
            and (
                self.return_tip_x_tolerance is None
                or abs(metrics.tip_x - float(self.geometry.target[0]))
                <= self.return_tip_x_tolerance
            )
        )
        self.crossing_success_count = (
            self.crossing_success_count + 1 if at_crossing_goal else 0
        )
        is_success = self.crossing_success_count >= self.required_crossing_streak
        terminated = unsafe_reason is not None or is_success
        truncated = self.step_count >= self.episode_steps and not terminated
        termination_reason = unsafe_reason
        if is_success:
            reward += self.success_bonus
            termination_reason = (
                "distal_body_crossed"
                if self.return_tip_x_tolerance is None
                else "distal_body_crossed_and_tip_returned"
            )

        self.previous_metrics = metrics
        self.previous_required_crossing_margin = crossing_margin
        self.last_observation = self._observation(physical_state, metrics)
        info = self._info(
            metrics,
            is_success=is_success,
            termination_reason=termination_reason,
        )
        return self.last_observation.copy(), float(reward), terminated, truncated, info
