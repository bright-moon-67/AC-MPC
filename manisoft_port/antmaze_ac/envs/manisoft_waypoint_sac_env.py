from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from manisoft.utils import (
    KOOPMAN_PHYSICAL_STATE_DIM,
    KOOPMAN_TIP_POSITION_SLICE,
    load_yaml,
)

from .manisoft_tracking_env import ManiSoftTipTrackingEnv
from .kinematic_push_task import segment_aabb_distance
from .table_entry_bank import (
    TableEntryTrajectoryBank,
    load_table_entry_trajectory_bank,
    restore_rod_internal_state,
)
from .waypoint_paths import (
    CURRICULUM_STAGES,
    ReferencePath,
    WaypointPathGenerator,
    WaypointWorkspace,
)


MANISOFT_WAYPOINT_SAC_OBSERVATION_DIM = KOOPMAN_PHYSICAL_STATE_DIM + 18 + 3 + 3 + 1
MANISOFT_WAYPOINT_SAC_GATE_OBSERVATION_DIM = (
    MANISOFT_WAYPOINT_SAC_OBSERVATION_DIM + 2 + 1 + 1
)


class ManiSoftWaypointSACEnv(ManiSoftTipTrackingEnv):
    """Goal-conditioned SAC environment for obstacle-free tip path tracking.

    The default observation is ``[physical_state(45), previous_action(18),
    target_error(3), lookahead_error(3), normalized_speed(1)]``.  The optional
    ``gate74`` mode only appends four controller/gate features after this
    unchanged 70-D prefix; in both modes the simulator's original 45-D state
    and all of its indices remain untouched.  The original action mode requests
    an absolute 18-D muscle activation.  The table equilibrium
    modes instead request a compact residual around a certified bent
    equilibrium and map it to the same physical 18-D action.  In particular,
    ``table_cartesian_delta`` integrates calibrated global x/y commands, so
    the policy sees the same table coordinates at every entry posture.
    Keeping the applied physical action in the observation makes either
    rate-limited process Markov.

    The reference is a distance-parameterized path. Progress is the tip's
    bounded geometric projection onto that path, never a clock or tolerance
    gate.  Entry curricula use simulator-certified bends from the natural
    upright pose; prefix curricula restore a certified simulator snapshot and
    ask SAC to solve progressively longer suffixes.  These snapshots are reset
    states, not state-action demonstrations and are never inserted into SAC's
    replay buffer.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_path: str | Path,
        *,
        curriculum: str = "mixed",
        episode_steps: int = 600,
        absolute_action_limit: float = 0.30,
        max_action_delta: float = 0.01,
        action_mode: str = "absolute",
        equilibrium_rotation_degrees: float = 4.0,
        equilibrium_xy_scale_delta: float = 0.03,
        table_action_calibration_path: str | Path | None = None,
        table_equilibrium_path_bank_path: str | Path | None = None,
        table_pose_path_bank_path: str | Path | None = None,
        table_pose_map_path: str | Path | None = None,
        pose_map_maximum_simplex_edge: float | None = None,
        pose_map_interpolation_mode: str = "barycentric",
        cartesian_action_step_scale: float = 0.035,
        cartesian_action_leak: float = 0.008,
        cartesian_prior_weight: float = 0.0,
        cartesian_prior_proportional_gain: float = 20.0,
        cartesian_prior_feedforward_scale: float = 1.0,
        cartesian_prior_internal_waypoints_only: bool = False,
        cartesian_prior_residual_scale: float = 0.0,
        equilibrium_path_prior_weight: float = 0.0,
        equilibrium_path_residual_scale: float = 0.0,
        observation_mode: str = "legacy70",
        workspace_low: Sequence[float] = (-0.30, 0.50, 0.445),
        workspace_high: Sequence[float] = (0.30, 0.80, 0.54),
        max_reach: float = 0.90,
        waypoint_segment_count_range: Sequence[int] = (2, 4),
        waypoint_segment_count_probabilities: Sequence[float] | None = None,
        waypoint_segment_length_range: Sequence[float] = (0.015, 0.030),
        waypoint_first_segment_length_range: Sequence[float] | None = None,
        waypoint_maximum_extent: float = 0.045,
        waypoint_minimum_turn_degrees: float = 0.0,
        waypoint_maximum_turn_degrees: float = 135.0,
        waypoint_hard_turn_probability: float = 0.0,
        waypoint_hard_turn_range_degrees: Sequence[float] = (120.0, 175.0),
        waypoint_vertical_delta_range: Sequence[float] = (0.0, 0.0),
        waypoint_single_line_probability: float = 0.0,
        tracking_guard: float = 0.050,
        tracking_vertical_tolerance: float = 0.0,
        lookahead_distance: float = 0.050,
        success_threshold: float = 0.025,
        success_streak: int = 8,
        table_surface_z: float = 0.36,
        minimum_tip_z: float = 0.415,
        min_desired_speed: float = 0.015,
        max_desired_speed: float = 0.12,
        entry_mid_anchor_probability: float = 0.0,
        entry_mid_advance_fraction_range: Sequence[float] = (0.30, 0.55),
        entry_mid_anchor_fraction_range: Sequence[float] = (0.55, 0.85),
        entry_anchor_probability: float = 0.0,
        entry_anchor_fraction_range: Sequence[float] = (0.30, 0.65),
        requested_rate_penalty_scale: float = 0.01,
        policy_action_penalty_scale: float = 0.0,
        cross_track_penalty_scale: float = 0.08,
        tracking_distance_penalty_scale: float = 0.04,
        endpoint_progress_scale: float = 0.0,
        terminal_precision_scale: float = 0.40,
        terminal_distance_penalty_scale: float = 0.15,
        terminal_capture_radius: float = 0.040,
        table_violation_penalty: float = 300.0,
        terminate_on_table_violation: bool = True,
        terminal_settle_steps: int = 150,
        terminal_timeout_penalty: float = 50.0,
        internal_waypoint_capture_radius: float = 0.0,
        internal_waypoint_bonus: float = 0.0,
        internal_waypoint_progress_scale: float = 0.0,
        internal_waypoint_distance_penalty_scale: float = 0.0,
        waypoint_stall_steps: int = 0,
        waypoint_stall_distance_epsilon: float = 0.0003,
        entry_bank_path: str | Path | None = None,
        entry_sampling_weights: Sequence[float] | None = None,
        target_lead_distance: float = 0.015,
        maximum_projection_advance: float = 0.040,
        projection_backtrack: float = 0.010,
        stall_grace_steps: int = 100,
        stall_window_steps: int = 250,
        stall_progress_epsilon: float = 0.001,
        table_x_bounds: Sequence[float] = (-0.55, 0.55),
        table_y_bounds: Sequence[float] = (0.42, 0.86),
        arm_radius: float = 0.050,
        table_safety_margin: float = 0.005,
        table_clearance_violation_tolerance: float = 0.0,
        enforce_whole_arm_table_clearance: bool = True,
        target_tip_tangent: Sequence[float] = (0.0, 0.0, -1.0),
        tip_orientation_penalty_scale: float = 0.0,
        maximum_tip_orientation_error_degrees: float = 180.0,
        tip_orientation_violation_penalty: float = 0.0,
        terminate_on_tip_orientation_violation: bool = False,
    ) -> None:
        if curriculum not in CURRICULUM_STAGES:
            raise ValueError(f"unknown curriculum stage: {curriculum}")
        if observation_mode not in {"legacy70", "gate74"}:
            raise ValueError("observation_mode must be 'legacy70' or 'gate74'")
        if observation_mode == "gate74" and action_mode in {
            "absolute",
            "table_pose_residual",
        }:
            raise ValueError(
                "gate74 observation mode requires a compact 2-D table action mode"
            )
        if max_action_delta <= 0:
            raise ValueError("max_action_delta must be positive")
        if action_mode not in {
            "absolute",
            "table_equilibrium",
            "table_cartesian_delta",
            "table_pose_residual",
        }:
            raise ValueError(
                "action_mode must be 'absolute', 'table_equilibrium', or "
                "'table_cartesian_delta', or 'table_pose_residual'"
            )
        if equilibrium_rotation_degrees <= 0 or equilibrium_xy_scale_delta <= 0:
            raise ValueError("table equilibrium action ranges must be positive")
        if cartesian_action_step_scale <= 0:
            raise ValueError("cartesian_action_step_scale must be positive")
        if (
            pose_map_maximum_simplex_edge is not None
            and pose_map_maximum_simplex_edge <= 0
        ):
            raise ValueError("pose_map_maximum_simplex_edge must be positive")
        if pose_map_maximum_simplex_edge is not None and table_pose_map_path is None:
            raise ValueError(
                "pose_map_maximum_simplex_edge requires table_pose_map_path"
            )
        if pose_map_interpolation_mode not in {"barycentric", "nearest"}:
            raise ValueError(
                "pose_map_interpolation_mode must be 'barycentric' or 'nearest'"
            )
        if not 0.0 <= cartesian_action_leak < 1.0:
            raise ValueError("cartesian_action_leak must lie in [0, 1)")
        if not 0.0 <= cartesian_prior_weight <= 1.0:
            raise ValueError("cartesian_prior_weight must lie in [0, 1]")
        if not 0.0 <= cartesian_prior_residual_scale <= 1.0:
            raise ValueError("cartesian_prior_residual_scale must lie in [0, 1]")
        if cartesian_prior_residual_scale > 0 and cartesian_prior_weight <= 0:
            raise ValueError(
                "cartesian_prior_residual_scale requires a positive Cartesian prior"
            )
        if not 0.0 <= equilibrium_path_prior_weight <= 1.0:
            raise ValueError("equilibrium_path_prior_weight must lie in [0, 1]")
        if not 0.0 <= equilibrium_path_residual_scale <= 1.0:
            raise ValueError("equilibrium_path_residual_scale must lie in [0, 1]")
        if equilibrium_path_residual_scale > 0 and equilibrium_path_prior_weight <= 0:
            raise ValueError(
                "equilibrium_path_residual_scale requires a positive path prior"
            )
        if cartesian_prior_weight > 0 and equilibrium_path_prior_weight > 0:
            raise ValueError("Cartesian and equilibrium path priors are mutually exclusive")
        if min(
            cartesian_prior_proportional_gain,
            cartesian_prior_feedforward_scale,
        ) < 0:
            raise ValueError("Cartesian prior gains must be non-negative")
        if cartesian_prior_weight > 0 and action_mode != "table_cartesian_delta":
            raise ValueError(
                "cartesian_prior_weight requires table_cartesian_delta action mode"
            )
        if cartesian_prior_weight > 0 and cartesian_action_leak <= 0:
            raise ValueError("Cartesian prior requires a positive action leak")
        if equilibrium_path_prior_weight > 0 and action_mode not in {
            "table_equilibrium",
            "table_pose_residual",
        }:
            raise ValueError(
                "equilibrium_path_prior_weight requires an equilibrium/pose "
                "path action mode"
            )
        if tracking_guard <= 0 or lookahead_distance <= 0:
            raise ValueError("tracking thresholds must be positive")
        if tracking_vertical_tolerance < 0:
            raise ValueError("tracking_vertical_tolerance must be non-negative")
        if target_lead_distance <= 0 or maximum_projection_advance <= 0:
            raise ValueError("projection target distances must be positive")
        if projection_backtrack < 0:
            raise ValueError("projection_backtrack must be non-negative")
        if min(stall_grace_steps, stall_window_steps, terminal_settle_steps) < 1:
            raise ValueError("stall step counts must be positive")
        if stall_progress_epsilon <= 0:
            raise ValueError("stall_progress_epsilon must be positive")
        if not 0 < min_desired_speed <= max_desired_speed:
            raise ValueError("desired speed bounds are invalid")
        if not 0.0 <= entry_mid_anchor_probability <= 1.0:
            raise ValueError("entry_mid_anchor_probability must lie in [0, 1]")
        if not 0.0 <= entry_anchor_probability <= 1.0:
            raise ValueError("entry_anchor_probability must lie in [0, 1]")
        if minimum_tip_z < table_surface_z:
            raise ValueError("minimum_tip_z cannot be below the table surface")
        if terminal_capture_radius < success_threshold:
            raise ValueError("terminal_capture_radius cannot be below success_threshold")
        if min(
            internal_waypoint_capture_radius,
            internal_waypoint_bonus,
            internal_waypoint_progress_scale,
            internal_waypoint_distance_penalty_scale,
            waypoint_stall_steps,
            waypoint_stall_distance_epsilon,
        ) < 0:
            raise ValueError("internal waypoint settings must be non-negative")
        if internal_waypoint_bonus > 0 and internal_waypoint_capture_radius <= 0:
            raise ValueError(
                "internal_waypoint_bonus requires a positive capture radius"
            )
        if (
            internal_waypoint_progress_scale > 0
            and internal_waypoint_capture_radius <= 0
        ):
            raise ValueError(
                "internal_waypoint_progress_scale requires a positive capture radius"
            )
        if (
            internal_waypoint_distance_penalty_scale > 0
            and internal_waypoint_capture_radius <= 0
        ):
            raise ValueError(
                "internal_waypoint_distance_penalty_scale requires a positive "
                "capture radius"
            )
        if waypoint_stall_steps > 0 and waypoint_stall_distance_epsilon <= 0:
            raise ValueError(
                "waypoint stall detection requires a positive distance epsilon"
            )
        if min(
            requested_rate_penalty_scale,
            policy_action_penalty_scale,
            cross_track_penalty_scale,
            tracking_distance_penalty_scale,
            endpoint_progress_scale,
            terminal_precision_scale,
            terminal_distance_penalty_scale,
            table_violation_penalty,
            terminal_timeout_penalty,
            tip_orientation_penalty_scale,
            tip_orientation_violation_penalty,
        ) < 0:
            raise ValueError("reward shaping scales must be non-negative")
        target_tangent = np.asarray(target_tip_tangent, dtype=np.float64)
        if (
            target_tangent.shape != (3,)
            or not np.isfinite(target_tangent).all()
            or np.linalg.norm(target_tangent) <= 0
        ):
            raise ValueError("target_tip_tangent must be a finite nonzero 3-D vector")
        if not 0 < maximum_tip_orientation_error_degrees <= 180:
            raise ValueError(
                "maximum_tip_orientation_error_degrees must lie in (0, 180]"
            )

        scenario_path = Path(scenario_path).expanduser().resolve()
        scenario = load_yaml(scenario_path)
        backend_dt = float(scenario["backend"]["dt"])
        update_interval = int(scenario["environment"]["update_interval"])
        self.control_dt = backend_dt * update_interval
        if not np.isclose(self.control_dt, 0.02):
            raise ValueError(
                "waypoint SAC currently requires the 50 Hz Koopman control rate; "
                f"got dt={self.control_dt}"
            )

        super().__init__(
            scenario_path,
            target_tip=(0.0, 0.0, 0.5),
            episode_steps=episode_steps,
            success_threshold=success_threshold,
            success_streak=success_streak,
            absolute_action_limit=absolute_action_limit,
        )
        self.curriculum = curriculum
        self.max_action_delta = float(max_action_delta)
        self.action_mode = str(action_mode)
        self.equilibrium_rotation_degrees = float(equilibrium_rotation_degrees)
        self.equilibrium_xy_scale_delta = float(equilibrium_xy_scale_delta)
        self.cartesian_action_step_scale = float(cartesian_action_step_scale)
        self.cartesian_action_leak = float(cartesian_action_leak)
        self.cartesian_prior_weight = float(cartesian_prior_weight)
        self.cartesian_prior_proportional_gain = float(
            cartesian_prior_proportional_gain
        )
        self.cartesian_prior_feedforward_scale = float(
            cartesian_prior_feedforward_scale
        )
        self.cartesian_prior_internal_waypoints_only = bool(
            cartesian_prior_internal_waypoints_only
        )
        self.cartesian_prior_residual_scale = float(
            cartesian_prior_residual_scale
        )
        self.equilibrium_path_prior_weight = float(equilibrium_path_prior_weight)
        self.equilibrium_path_residual_scale = float(
            equilibrium_path_residual_scale
        )
        self.observation_mode = str(observation_mode)
        self.pose_map_interpolation_mode = str(pose_map_interpolation_mode)
        self.tracking_guard = float(tracking_guard)
        self.tracking_vertical_tolerance = float(
            tracking_vertical_tolerance
        )
        self.lookahead_distance = float(lookahead_distance)
        self.table_surface_z = float(table_surface_z)
        self.minimum_tip_z = float(minimum_tip_z)
        self.min_desired_speed = float(min_desired_speed)
        self.max_desired_speed = float(max_desired_speed)
        self.entry_mid_anchor_probability = float(entry_mid_anchor_probability)
        self.entry_mid_advance_fraction_range = np.asarray(
            entry_mid_advance_fraction_range, dtype=np.float64
        )
        self.entry_mid_anchor_fraction_range = np.asarray(
            entry_mid_anchor_fraction_range, dtype=np.float64
        )
        self.entry_anchor_probability = float(entry_anchor_probability)
        self.entry_anchor_fraction_range = np.asarray(
            entry_anchor_fraction_range, dtype=np.float64
        )
        for name, bounds in (
            ("entry_mid_advance_fraction_range", self.entry_mid_advance_fraction_range),
            ("entry_mid_anchor_fraction_range", self.entry_mid_anchor_fraction_range),
            ("entry_anchor_fraction_range", self.entry_anchor_fraction_range),
        ):
            if (
                bounds.shape != (2,)
                or bounds[0] < 0.0
                or bounds[1] > 1.0
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"{name} must be an increasing pair inside [0, 1]")
        self.requested_rate_penalty_scale = float(requested_rate_penalty_scale)
        self.policy_action_penalty_scale = float(policy_action_penalty_scale)
        self.cross_track_penalty_scale = float(cross_track_penalty_scale)
        self.tracking_distance_penalty_scale = float(
            tracking_distance_penalty_scale
        )
        self.endpoint_progress_scale = float(endpoint_progress_scale)
        self.terminal_precision_scale = float(terminal_precision_scale)
        self.terminal_distance_penalty_scale = float(
            terminal_distance_penalty_scale
        )
        self.terminal_capture_radius = float(terminal_capture_radius)
        self.table_violation_penalty = float(table_violation_penalty)
        self.terminate_on_table_violation = bool(
            terminate_on_table_violation
        )
        self.terminal_settle_steps = int(terminal_settle_steps)
        self.terminal_timeout_penalty = float(terminal_timeout_penalty)
        self.internal_waypoint_capture_radius = float(
            internal_waypoint_capture_radius
        )
        self.internal_waypoint_bonus = float(internal_waypoint_bonus)
        self.internal_waypoint_progress_scale = float(
            internal_waypoint_progress_scale
        )
        self.internal_waypoint_distance_penalty_scale = float(
            internal_waypoint_distance_penalty_scale
        )
        self.waypoint_stall_steps = int(waypoint_stall_steps)
        self.waypoint_stall_distance_epsilon = float(
            waypoint_stall_distance_epsilon
        )
        self.target_lead_distance = float(target_lead_distance)
        self.maximum_projection_advance = float(maximum_projection_advance)
        self.projection_backtrack = float(projection_backtrack)
        self.stall_grace_steps = int(stall_grace_steps)
        self.stall_window_steps = int(stall_window_steps)
        self.stall_progress_epsilon = float(stall_progress_epsilon)
        self.table_x_bounds = np.asarray(table_x_bounds, dtype=np.float64)
        self.table_y_bounds = np.asarray(table_y_bounds, dtype=np.float64)
        if (
            self.table_x_bounds.shape != (2,)
            or self.table_y_bounds.shape != (2,)
            or self.table_x_bounds[0] >= self.table_x_bounds[1]
            or self.table_y_bounds[0] >= self.table_y_bounds[1]
        ):
            raise ValueError("table x/y bounds must be increasing pairs")
        self.arm_radius = float(arm_radius)
        self.table_safety_margin = float(table_safety_margin)
        self.table_clearance_violation_tolerance = float(
            table_clearance_violation_tolerance
        )
        if (
            self.arm_radius <= 0
            or self.table_safety_margin < 0
            or self.table_clearance_violation_tolerance < 0
        ):
            raise ValueError("arm radius and table safety margin are invalid")
        self.enforce_whole_arm_table_clearance = bool(
            enforce_whole_arm_table_clearance
        )
        self.target_tip_tangent = (
            target_tangent / np.linalg.norm(target_tangent)
        )
        self.tip_orientation_penalty_scale = float(
            tip_orientation_penalty_scale
        )
        self.maximum_tip_orientation_error_degrees = float(
            maximum_tip_orientation_error_degrees
        )
        self.tip_orientation_violation_penalty = float(
            tip_orientation_violation_penalty
        )
        self.terminate_on_tip_orientation_violation = bool(
            terminate_on_tip_orientation_violation
        )
        self.entry_bank: TableEntryTrajectoryBank | None = None
        self.entry_sampling_weights: np.ndarray | None = None
        entry_path: Path | None = None
        if entry_bank_path is not None:
            entry_path = Path(entry_bank_path).expanduser().resolve()
            self.entry_bank = load_table_entry_trajectory_bank(entry_path)
            scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
            if self.entry_bank.scenario_sha256 != scenario_hash:
                raise ValueError(
                    "table-entry bank was generated with a different scenario"
                )
            if not np.isclose(self.entry_bank.control_dt, self.control_dt):
                raise ValueError("table-entry bank control_dt does not match scenario")
            if self.entry_bank.absolute_action_limit > self.absolute_action_limit + 1e-6:
                raise ValueError(
                    "environment activation limit is below the entry-bank limit"
                )
            if not (
                np.allclose(self.entry_bank.table_x_bounds, self.table_x_bounds)
                and np.allclose(self.entry_bank.table_y_bounds, self.table_y_bounds)
                and np.isclose(self.entry_bank.table_surface_z, self.table_surface_z)
                and np.isclose(self.entry_bank.arm_radius, self.arm_radius)
                # A bank certified with a larger clearance margin is also
                # valid in an otherwise identical, less conservative runtime
                # environment.  Reject only the unsafe opposite direction.
                and self.entry_bank.safety_margin + 1e-9
                >= self.table_safety_margin
            ):
                raise ValueError(
                    "table-entry bank clearance geometry does not match environment"
                )
        if entry_sampling_weights is not None:
            if self.entry_bank is None:
                raise ValueError("entry_sampling_weights requires an entry bank")
            weights = np.asarray(entry_sampling_weights, dtype=np.float64)
            if (
                weights.shape != (self.entry_bank.trajectory_count,)
                or np.any(weights < 0)
                or not np.isfinite(weights).all()
                or float(np.sum(weights)) <= 0
            ):
                raise ValueError(
                    "entry_sampling_weights must contain one non-negative "
                    "value per entry trajectory"
                )
            self.entry_sampling_weights = weights / np.sum(weights)
        self.cartesian_positive_deltas: np.ndarray | None = None
        self.cartesian_negative_deltas: np.ndarray | None = None
        self.cartesian_command_distance: float | None = None
        if self.action_mode == "table_cartesian_delta":
            if self.entry_bank is None or entry_path is None:
                raise ValueError(
                    "table_cartesian_delta requires a certified entry_bank_path"
                )
            if table_action_calibration_path is None:
                raise ValueError(
                    "table_cartesian_delta requires table_action_calibration_path"
                )
            calibration_path = (
                Path(table_action_calibration_path).expanduser().resolve()
            )
            with np.load(calibration_path, allow_pickle=False) as calibration:
                kind = str(np.asarray(calibration["kind"]).reshape(()).item())
                if kind != "manisoft_table_cartesian_action_calibration":
                    raise ValueError("unexpected table-action calibration kind")
                names = tuple(str(value) for value in calibration["entry_names"])
                if names != self.entry_bank.names:
                    raise ValueError(
                        "table-action calibration entry names do not match the bank"
                    )
                scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
                stored_scenario_hash = str(
                    np.asarray(calibration["scenario_sha256"]).reshape(()).item()
                )
                stored_bank_hash = str(
                    np.asarray(calibration["entry_bank_sha256"]).reshape(()).item()
                )
                if stored_scenario_hash != scenario_hash:
                    raise ValueError(
                        "table-action calibration was generated for another scenario"
                    )
                if stored_bank_hash != hashlib.sha256(entry_path.read_bytes()).hexdigest():
                    raise ValueError(
                        "table-action calibration was generated for another entry bank"
                    )
                equilibria = np.asarray(
                    calibration["equilibrium_actions"], dtype=np.float32
                )
                expected = np.asarray(self.entry_bank.actions[:, -1], dtype=np.float32)
                if equilibria.shape != expected.shape or not np.allclose(
                    equilibria, expected, atol=1e-6
                ):
                    raise ValueError(
                        "table-action calibration equilibria do not match the entry bank"
                    )
                positive = np.asarray(
                    calibration["positive_action_deltas"], dtype=np.float32
                )
                negative = np.asarray(
                    calibration["negative_action_deltas"], dtype=np.float32
                )
                expected_shape = (self.entry_bank.trajectory_count, 2, 18)
                if positive.shape != expected_shape or negative.shape != expected_shape:
                    raise ValueError(
                        "table-action calibration deltas must have shape "
                        f"{expected_shape}"
                    )
                if not np.isfinite(positive).all() or not np.isfinite(negative).all():
                    raise ValueError("table-action calibration contains NaN or Inf")
                if bool(np.any(calibration["validation_violations"])):
                    raise ValueError("table-action calibration failed safety validation")
                self.cartesian_positive_deltas = positive.copy()
                self.cartesian_negative_deltas = negative.copy()
                self.cartesian_command_distance = float(
                    np.asarray(calibration["command_distance"]).reshape(()).item()
                )
        self.equilibrium_path_entry_index: int | None = None
        self.equilibrium_path_tip_positions: np.ndarray | None = None
        self.equilibrium_path_policy_actions: np.ndarray | None = None
        self.pose_map_tip_positions: np.ndarray | None = None
        self.pose_map_policy_actions: np.ndarray | None = None
        self.pose_map_triangulation: Delaunay | None = None
        self.pose_map_valid_simplices: np.ndarray | None = None
        self.pose_map_interpolation_dimensions = 2
        if table_equilibrium_path_bank_path is not None:
            if self.entry_bank is None or entry_path is None:
                raise ValueError(
                    "table_equilibrium_path_bank_path requires an entry bank"
                )
            if self.action_mode != "table_equilibrium":
                raise ValueError(
                    "table equilibrium path banks require table_equilibrium action mode"
                )
            path_bank_path = Path(
                table_equilibrium_path_bank_path
            ).expanduser().resolve()
            with np.load(path_bank_path, allow_pickle=False) as path_bank:
                kind = str(np.asarray(path_bank["kind"]).reshape(()).item())
                version = int(
                    np.asarray(path_bank["schema_version"]).reshape(()).item()
                )
                if version != 1 or kind != "manisoft_table_equilibrium_path_bank":
                    raise ValueError("unexpected table equilibrium path-bank schema")
                stored_scenario_hash = str(
                    np.asarray(path_bank["scenario_sha256"]).reshape(()).item()
                )
                stored_entry_hash = str(
                    np.asarray(path_bank["entry_bank_sha256"]).reshape(()).item()
                )
                if stored_scenario_hash != hashlib.sha256(
                    scenario_path.read_bytes()
                ).hexdigest():
                    raise ValueError("equilibrium path bank uses another scenario")
                if stored_entry_hash != hashlib.sha256(
                    entry_path.read_bytes()
                ).hexdigest():
                    raise ValueError("equilibrium path bank uses another entry bank")
                entry_index = int(
                    np.asarray(path_bank["entry_index"]).reshape(()).item()
                )
                entry_name = str(
                    np.asarray(path_bank["entry_name"]).reshape(()).item()
                )
                if (
                    entry_index < 0
                    or entry_index >= self.entry_bank.trajectory_count
                    or self.entry_bank.names[entry_index] != entry_name
                ):
                    raise ValueError("equilibrium path-bank entry does not match")
                rotations = np.asarray(
                    path_bank["rotation_degrees"], dtype=np.float64
                )
                tips = np.asarray(path_bank["tip_positions"], dtype=np.float32)
                activation_scale = float(
                    np.asarray(path_bank["activation_scale"]).reshape(()).item()
                )
                if (
                    rotations.ndim != 1
                    or len(rotations) < 3
                    or tips.shape != (len(rotations), 3)
                    or not np.isfinite(rotations).all()
                    or not np.isfinite(tips).all()
                    or np.any(np.diff(rotations) <= 0)
                ):
                    raise ValueError("equilibrium path bank contains invalid arrays")
                normalized_rotation = rotations / self.equilibrium_rotation_degrees
                normalized_scale = np.full(
                    len(rotations),
                    (activation_scale - 1.0) / self.equilibrium_xy_scale_delta,
                    dtype=np.float64,
                )
                policy_actions = np.column_stack(
                    (normalized_rotation, normalized_scale)
                )
                if np.max(np.abs(policy_actions)) > 1.0 + 1e-6:
                    raise ValueError(
                        "equilibrium path bank exceeds configured policy-action ranges"
                    )
                expected_tip = self.entry_bank.tip_positions[entry_index, -1]
                if np.linalg.norm(tips[0] - expected_tip) > 1e-4:
                    raise ValueError(
                        "equilibrium path bank does not start at its certified entry"
                    )
                self.equilibrium_path_entry_index = entry_index
                self.equilibrium_path_tip_positions = tips.copy()
                self.equilibrium_path_policy_actions = policy_actions.astype(
                    np.float32
                )
        if table_pose_path_bank_path is not None:
            if table_equilibrium_path_bank_path is not None:
                raise ValueError(
                    "table pose and legacy equilibrium path banks are mutually exclusive"
                )
            if self.entry_bank is None or entry_path is None:
                raise ValueError("table_pose_path_bank_path requires an entry bank")
            if self.action_mode != "table_pose_residual":
                raise ValueError(
                    "table pose path banks require table_pose_residual action mode"
                )
            pose_path = Path(table_pose_path_bank_path).expanduser().resolve()
            with np.load(pose_path, allow_pickle=False) as path_bank:
                kind = str(np.asarray(path_bank["kind"]).reshape(()).item())
                version = int(
                    np.asarray(path_bank["schema_version"]).reshape(()).item()
                )
                if version != 1 or kind != "manisoft_table_arch_pose_path_bank":
                    raise ValueError("unexpected table pose path-bank schema")
                stored_scenario_hash = str(
                    np.asarray(path_bank["scenario_sha256"]).reshape(()).item()
                )
                stored_entry_hash = str(
                    np.asarray(path_bank["entry_bank_sha256"]).reshape(()).item()
                )
                if stored_scenario_hash != hashlib.sha256(
                    scenario_path.read_bytes()
                ).hexdigest():
                    raise ValueError("table pose path bank uses another scenario")
                if stored_entry_hash != hashlib.sha256(
                    entry_path.read_bytes()
                ).hexdigest():
                    raise ValueError("table pose path bank uses another entry bank")
                entry_index = int(
                    np.asarray(path_bank["entry_index"]).reshape(()).item()
                )
                entry_name = str(
                    np.asarray(path_bank["entry_name"]).reshape(()).item()
                )
                tips = np.asarray(path_bank["tip_positions"], dtype=np.float32)
                physical_actions = np.asarray(
                    path_bank["physical_actions"], dtype=np.float32
                )
                if (
                    entry_index < 0
                    or entry_index >= self.entry_bank.trajectory_count
                    or self.entry_bank.names[entry_index] != entry_name
                ):
                    raise ValueError("table pose path-bank entry does not match")
                if (
                    tips.ndim != 2
                    or tips.shape[1] != 3
                    or len(tips) < 3
                    or physical_actions.shape != (len(tips), 18)
                    or not np.isfinite(tips).all()
                    or not np.isfinite(physical_actions).all()
                    or np.max(np.abs(physical_actions))
                    > self.absolute_action_limit + 1e-6
                ):
                    raise ValueError("table pose path bank contains invalid arrays")
                if (
                    np.linalg.norm(
                        tips[0]
                        - self.entry_bank.tip_positions[entry_index, -1]
                    )
                    > 1e-4
                    or np.max(
                        np.abs(
                            physical_actions[0]
                            - self.entry_bank.actions[entry_index, -1]
                        )
                    )
                    > 2e-6
                ):
                    raise ValueError(
                        "table pose path bank does not start at its certified entry"
                    )
                self.equilibrium_path_entry_index = entry_index
                self.equilibrium_path_tip_positions = tips.copy()
                self.equilibrium_path_policy_actions = physical_actions.copy()
        pose_map_hull_equations = None
        if table_pose_map_path is not None:
            if (
                table_pose_path_bank_path is not None
                or table_equilibrium_path_bank_path is not None
            ):
                raise ValueError("table pose maps and path banks are mutually exclusive")
            if self.entry_bank is None or entry_path is None:
                raise ValueError("table_pose_map_path requires an entry bank")
            if self.action_mode != "table_pose_residual":
                raise ValueError("table pose maps require table_pose_residual action mode")
            pose_map_path = Path(table_pose_map_path).expanduser().resolve()
            with np.load(pose_map_path, allow_pickle=False) as pose_map:
                kind = str(np.asarray(pose_map["kind"]).reshape(()).item())
                version = int(np.asarray(pose_map["schema_version"]).reshape(()).item())
                if version not in {1, 2} or kind != "manisoft_table_arch_pose_map":
                    raise ValueError("unexpected table pose-map schema")
                interpolation_dimensions = int(
                    np.asarray(
                        pose_map["interpolation_dimensions"]
                        if "interpolation_dimensions" in pose_map.files
                        else 2
                    )
                    .reshape(())
                    .item()
                )
                if interpolation_dimensions not in {2, 3}:
                    raise ValueError(
                        "pose-map interpolation dimensions must be two or three"
                    )
                if version == 1 and interpolation_dimensions != 2:
                    raise ValueError(
                        "schema-v1 pose maps only support two-dimensional interpolation"
                    )
                if str(np.asarray(pose_map["scenario_sha256"]).reshape(()).item()) != hashlib.sha256(
                    scenario_path.read_bytes()
                ).hexdigest():
                    raise ValueError("table pose map uses another scenario")
                if str(np.asarray(pose_map["entry_bank_sha256"]).reshape(()).item()) != hashlib.sha256(
                    entry_path.read_bytes()
                ).hexdigest():
                    raise ValueError("table pose map uses another entry bank")
                entry_index = int(np.asarray(pose_map["entry_index"]).reshape(()).item())
                entry_name = str(np.asarray(pose_map["entry_name"]).reshape(()).item())
                tips = np.asarray(pose_map["tip_positions"], dtype=np.float32)
                actions = np.asarray(pose_map["physical_actions"], dtype=np.float32)
                if (
                    entry_index < 0
                    or entry_index >= self.entry_bank.trajectory_count
                    or self.entry_bank.names[entry_index] != entry_name
                ):
                    raise ValueError("table pose-map entry does not match")
                if (
                    tips.ndim != 2
                    or tips.shape[1] != 3
                    or len(tips) < 4
                    or actions.shape != (len(tips), 18)
                    or not np.isfinite(tips).all()
                    or not np.isfinite(actions).all()
                    or np.max(np.abs(actions)) > self.absolute_action_limit + 1e-6
                ):
                    raise ValueError("table pose map contains invalid arrays")
                if (
                    np.linalg.norm(tips[0] - self.entry_bank.tip_positions[entry_index, -1]) > 1e-4
                    or np.max(np.abs(actions[0] - self.entry_bank.actions[entry_index, -1])) > 2e-6
                ):
                    raise ValueError("table pose map does not start at its certified entry")
                interpolation_points = np.asarray(
                    tips[:, :interpolation_dimensions], dtype=np.float64
                )
                triangulation = Delaunay(interpolation_points)
                if (
                    triangulation.find_simplex(
                        interpolation_points[0], tol=1e-8
                    )
                    < 0
                ):
                    raise ValueError("entry tip lies outside the table pose-map hull")
                self.equilibrium_path_entry_index = entry_index
                self.pose_map_tip_positions = tips.copy()
                self.pose_map_policy_actions = actions.copy()
                self.pose_map_triangulation = triangulation
                self.pose_map_interpolation_dimensions = interpolation_dimensions
                if pose_map_maximum_simplex_edge is not None:
                    simplex_points = np.asarray(
                        interpolation_points[triangulation.simplices],
                        dtype=np.float64,
                    )
                    edge_lengths = np.stack(
                        [
                            np.linalg.norm(
                                simplex_points[:, first]
                                - simplex_points[:, second],
                                axis=1,
                            )
                            for first in range(interpolation_dimensions + 1)
                            for second in range(first)
                        ],
                        axis=1,
                    )
                    self.pose_map_valid_simplices = (
                        np.max(edge_lengths, axis=1)
                        <= float(pose_map_maximum_simplex_edge) + 1e-12
                    )
                    entry_simplex = int(
                        triangulation.find_simplex(
                            interpolation_points[0], tol=1e-8
                        )
                    )
                    if not self.pose_map_valid_simplices[entry_simplex]:
                        raise ValueError(
                            "entry tip lies in a pose-map simplex rejected by "
                            "pose_map_maximum_simplex_edge"
                        )
                pose_map_hull_equations = ConvexHull(
                    interpolation_points
                ).equations.copy()
        self.path_generator = WaypointPathGenerator(
            WaypointWorkspace.from_bounds(
                workspace_low,
                workspace_high,
                max_reach=max_reach,
                xy_hull_equations=pose_map_hull_equations,
                xy_triangulation=(
                    self.pose_map_triangulation
                    if self.pose_map_valid_simplices is not None
                    else None
                ),
                xy_valid_simplices=self.pose_map_valid_simplices,
                interpolation_dimensions=self.pose_map_interpolation_dimensions,
            ),
            waypoint_segment_count_range=waypoint_segment_count_range,
            waypoint_segment_count_probabilities=waypoint_segment_count_probabilities,
            waypoint_segment_length_range=waypoint_segment_length_range,
            waypoint_first_segment_length_range=waypoint_first_segment_length_range,
            waypoint_maximum_extent=waypoint_maximum_extent,
            waypoint_minimum_turn_degrees=waypoint_minimum_turn_degrees,
            waypoint_maximum_turn_degrees=waypoint_maximum_turn_degrees,
            waypoint_hard_turn_probability=waypoint_hard_turn_probability,
            waypoint_hard_turn_range_degrees=waypoint_hard_turn_range_degrees,
            waypoint_vertical_delta_range=waypoint_vertical_delta_range,
            waypoint_single_line_probability=waypoint_single_line_probability,
        )

        observation_dim = (
            MANISOFT_WAYPOINT_SAC_OBSERVATION_DIM
            if self.observation_mode == "legacy70"
            else MANISOFT_WAYPOINT_SAC_GATE_OBSERVATION_DIM
        )
        self.observation_space = gym.spaces.Box(
            low=np.full(observation_dim, -np.inf, dtype=np.float32),
            high=np.full(observation_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        if self.action_mode in {"absolute", "table_pose_residual"}:
            self.action_space = gym.spaces.Box(
                low=np.full(
                    18,
                    -absolute_action_limit
                    if self.action_mode == "absolute"
                    else -1.0,
                    dtype=np.float32,
                ),
                high=np.full(
                    18,
                    absolute_action_limit
                    if self.action_mode == "absolute"
                    else 1.0,
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.full(2, -1.0, dtype=np.float32),
                high=np.full(2, 1.0, dtype=np.float32),
                dtype=np.float32,
            )

        self.path: ReferencePath | None = None
        self.path_progress = 0.0
        self.current_target = np.zeros(3, dtype=np.float32)
        self.lookahead_target = np.zeros(3, dtype=np.float32)
        self.desired_speed = self.min_desired_speed
        self.previous_action = np.zeros(18, dtype=np.float32)
        self.previous_tip = np.zeros(3, dtype=np.float32)
        self.previous_tracking_distance = 0.0
        self.previous_final_distance = 0.0
        self.equilibrium_action = np.zeros(18, dtype=np.float32)
        self.last_physical_state: np.ndarray | None = None
        self.table_violation_count = 0
        self.projected_point = np.zeros(3, dtype=np.float32)
        self.best_path_progress = 0.0
        self.last_progress_step = 0
        self.terminal_entry_step: int | None = None
        self.terminal_timeout = False
        self.active_curriculum = curriculum
        self.entry_index: int | None = None
        self.entry_prefix_steps = 0
        self.stalled = False
        self.path_progress_stalled = False
        self.waypoint_stalled = False
        self.dynamics_violation = False
        self.last_table_clearance = float("inf")
        self.anchor_cumulative_length = np.zeros(1, dtype=np.float64)
        self.next_internal_waypoint_index = 1
        self.internal_waypoints_completed = 0
        self.waypoint_passed = False
        self.active_waypoint_distance = float("nan")
        self.internal_waypoint_distance_delta = 0.0
        self.normalized_internal_waypoint_progress = 0.0
        self.normalized_internal_waypoint_capture_error = 0.0
        self.best_active_waypoint_distance = float("nan")
        self.last_waypoint_improvement_step = 0
        self.cartesian_prior_start_tip = np.zeros(3, dtype=np.float32)
        self.last_raw_policy_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )
        self.last_controller_prior_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )
        self.last_blended_policy_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )
        self.path_anchor_policy_actions: np.ndarray | None = None
        self.reference_policy_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )

    @staticmethod
    def _tip_position(state: np.ndarray) -> np.ndarray:
        return np.asarray(state[KOOPMAN_TIP_POSITION_SLICE], dtype=np.float32)

    def _tip_orientation(self) -> tuple[np.ndarray, float]:
        """Return terminal centreline tangent and target-angle error.

        This is a reward/safety diagnostic derived from the simulator nodes;
        it does not alter the original 45-D physical state or observation
        prefix.
        """

        nodes = np.asarray(
            self.sim._backend.softrobot_state.element_positions,
            dtype=np.float64,
        )
        tangent = nodes[-1] - nodes[-2]
        norm = float(np.linalg.norm(tangent))
        if not np.isfinite(tangent).all() or norm <= 1e-12:
            return np.zeros(3, dtype=np.float64), 180.0
        tangent /= norm
        angle = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        np.dot(tangent, self.target_tip_tangent), -1.0, 1.0
                    )
                )
            )
        )
        return tangent, angle

    def _blend_cartesian_prior(self, policy_action: np.ndarray) -> np.ndarray:
        """Blend the learned command with a calibrated Cartesian stabilizer."""

        raw = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        if raw.shape != self.action_space.shape:
            raise ValueError(
                f"expected a {self.action_space.shape[0]}-D action, got {raw.shape}"
            )
        if not np.isfinite(raw).all():
            raise FloatingPointError("action contains NaN or Inf")
        raw = np.clip(raw, self.action_space.low, self.action_space.high)
        self.last_raw_policy_action = raw.copy()
        effective_prior_weight = self._effective_cartesian_prior_weight()
        prior = self._cartesian_prior_action(
            getattr(self, "last_physical_state", None)
        )
        if (
            getattr(self, "equilibrium_path_prior_weight", 0.0) > 0
            and getattr(self, "equilibrium_path_residual_scale", 0.0) > 0
        ):
            blended = np.clip(
                effective_prior_weight * prior
                + self.equilibrium_path_residual_scale * raw,
                self.action_space.low,
                self.action_space.high,
            ).astype(np.float32)
        elif (
            getattr(self, "cartesian_prior_weight", 0.0) > 0
            and getattr(self, "cartesian_prior_residual_scale", 0.0) > 0
        ):
            blended = np.clip(
                effective_prior_weight * prior
                + self.cartesian_prior_residual_scale * raw,
                self.action_space.low,
                self.action_space.high,
            ).astype(np.float32)
        else:
            blended = np.clip(
                (1.0 - effective_prior_weight) * raw
                + effective_prior_weight * prior,
                self.action_space.low,
                self.action_space.high,
            ).astype(np.float32)
        self.last_controller_prior_action = prior.copy()
        self.last_blended_policy_action = blended.copy()
        return blended

    def _effective_cartesian_prior_weight(self) -> float:
        effective_prior_weight = (
            getattr(self, "equilibrium_path_prior_weight", 0.0)
            if getattr(self, "equilibrium_path_prior_weight", 0.0) > 0
            else self.cartesian_prior_weight
        )
        if (
            self.cartesian_prior_internal_waypoints_only
            and (
                self.path is None
                or self.next_internal_waypoint_index
                >= len(self.path.anchors) - 1
            )
        ):
            effective_prior_weight = 0.0
        return float(effective_prior_weight)

    def _cartesian_prior_action(
        self, physical_state: np.ndarray | None
    ) -> np.ndarray:
        prior = np.zeros(self.action_space.shape, dtype=np.float32)
        if getattr(self, "equilibrium_path_prior_weight", 0.0) > 0:
            return np.asarray(
                self.reference_policy_action, dtype=np.float32
            ).copy()
        if self._effective_cartesian_prior_weight() > 0:
            if (
                self.action_mode != "table_cartesian_delta"
                or self.cartesian_command_distance is None
                or physical_state is None
            ):
                raise RuntimeError(
                    "Cartesian action prior requires a reset calibrated environment"
                )
            steady_displacement = (
                self.cartesian_command_distance
                * self.cartesian_action_step_scale
                / self.cartesian_action_leak
            )
            tip = self._tip_position(physical_state)
            feedforward = (
                self.current_target[:2] - self.cartesian_prior_start_tip[:2]
            ) / steady_displacement
            feedback = self.cartesian_prior_proportional_gain * (
                self.current_target[:2] - tip[:2]
            )
            prior = np.clip(
                self.cartesian_prior_feedforward_scale * feedforward + feedback,
                -1.0,
                1.0,
            ).astype(np.float32)
        return prior

    def _physical_action_request(self, policy_action: np.ndarray) -> np.ndarray:
        """Map the policy action to an absolute 18-D muscle activation."""

        requested = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        if requested.shape != self.action_space.shape:
            raise ValueError(
                f"expected a {self.action_space.shape[0]}-D action, "
                f"got {requested.shape}"
            )
        if not np.isfinite(requested).all():
            raise FloatingPointError("action contains NaN or Inf")
        requested = np.clip(requested, self.action_space.low, self.action_space.high)
        if self.action_mode in {"absolute", "table_pose_residual"}:
            return requested.astype(np.float32, copy=False)

        if self.action_mode == "table_cartesian_delta":
            if (
                self.entry_index is None
                or self.cartesian_positive_deltas is None
                or self.cartesian_negative_deltas is None
            ):
                raise RuntimeError(
                    "Cartesian table actions require an active calibrated entry pose"
                )
            positive = np.maximum(requested, 0.0)
            negative = np.maximum(-requested, 0.0)
            physical_delta = (
                positive @ self.cartesian_positive_deltas[self.entry_index]
                + negative @ self.cartesian_negative_deltas[self.entry_index]
            )
            # The calibrated columns are settled actions for a one-centimetre
            # displacement. Integrating a small fraction at 50 Hz turns them
            # into smooth global x/y velocity commands. A weak pull toward the
            # certified equilibrium makes this integrator BIBO-stable: a tiny
            # policy bias cannot accumulate until an actuator saturates, while
            # a sustained command still spans roughly three centimetres.
            return np.clip(
                self.previous_action
                + self.cartesian_action_step_scale * physical_delta
                - self.cartesian_action_leak
                * (self.previous_action - self.equilibrium_action),
                -self.absolute_action_limit,
                self.absolute_action_limit,
            ).astype(np.float32)

        values = self.equilibrium_action.reshape(6, 3).astype(
            np.float64, copy=True
        )
        angle = np.deg2rad(self.equilibrium_rotation_degrees * requested[0])
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
        scale = 1.0 + self.equilibrium_xy_scale_delta * requested[1]
        values[:, :2] = scale * values[:, :2] @ rotation.T
        return np.clip(
            values.reshape(-1),
            -self.absolute_action_limit,
            self.absolute_action_limit,
        ).astype(np.float32)

    def desired_speed_bounds(self, curriculum: str) -> tuple[float, float]:
        """Return the effective speed curriculum used by a stage."""

        stage_bounds = {
            "point": (self.min_desired_speed, min(0.045, self.max_desired_speed)),
            "entry_tail": (
                max(0.020, self.min_desired_speed),
                min(0.060, self.max_desired_speed),
            ),
            "entry_bridge": (
                max(0.020, self.min_desired_speed),
                min(0.070, self.max_desired_speed),
            ),
            "entry_mid": (
                max(0.025, self.min_desired_speed),
                min(0.080, self.max_desired_speed),
            ),
            "entry": (
                max(0.035, self.min_desired_speed),
                min(0.10, self.max_desired_speed),
            ),
            "table_point": (
                max(0.035, self.min_desired_speed),
                min(0.10, self.max_desired_speed),
            ),
            "table_local_line": (
                self.min_desired_speed,
                min(0.040, self.max_desired_speed),
            ),
            "table_waypoint_polyline": (
                self.min_desired_speed,
                min(0.040, self.max_desired_speed),
            ),
            "table_long_waypoints": (
                self.min_desired_speed,
                min(0.050, self.max_desired_speed),
            ),
            "table_local": (self.min_desired_speed, min(0.07, self.max_desired_speed)),
            "entry_local": (
                max(0.025, self.min_desired_speed),
                min(0.09, self.max_desired_speed),
            ),
            "path": (max(0.025, self.min_desired_speed), min(0.10, self.max_desired_speed)),
            "recovery": (self.min_desired_speed, self.max_desired_speed),
            "mixed": (self.min_desired_speed, self.max_desired_speed),
        }
        if curriculum not in stage_bounds:
            raise ValueError(f"unknown speed curriculum: {curriculum}")
        return stage_bounds[curriculum]

    def _sample_speed(self, curriculum: str) -> float:
        low, high = self.desired_speed_bounds(curriculum)
        # Log-uniform sampling prevents the largest speeds from dominating and
        # guarantees substantial low-speed coverage.
        return float(np.exp(self.np_random.uniform(np.log(low), np.log(high))))

    def _update_reference_targets(self) -> None:
        if self.path is None:
            raise RuntimeError("environment must be reset before use")
        # ``ReferencePath.sample`` clips its own input, but the same progress is
        # also used below to interpolate a reference action.  Keep the shared
        # progress inside the path domain so the action cannot be extrapolated
        # past the final certified posture during terminal holding.
        target_progress = float(
            np.clip(
                self.path_progress + self.target_lead_distance,
                0.0,
                self.path.length,
            )
        )
        if (
            self.internal_waypoint_capture_radius > 0
            and self.next_internal_waypoint_index < len(self.path.anchors) - 1
        ):
            # Projection is gated at an ordered internal waypoint until the
            # tip enters its capture ball. Do not simultaneously ask the actor
            # to chase a target beyond that corner: on a sharp turn this
            # rewards cutting the corner while the gate requires the opposite.
            # The uncapped lookahead below still exposes the outgoing segment.
            target_progress = min(
                target_progress,
                float(
                    self.anchor_cumulative_length[
                        self.next_internal_waypoint_index
                    ]
                ),
            )
        self.current_target = self.path.sample(target_progress)
        self.lookahead_target = self.path.sample(
            self.path_progress + self.target_lead_distance + self.lookahead_distance
        )
        self.target_tip = self.current_target.copy()
        if getattr(self, "pose_map_triangulation", None) is not None:
            self.reference_policy_action = self._pose_map_action(
                self.current_target
            )
        elif getattr(self, "path_anchor_policy_actions", None) is None:
            if hasattr(self, "action_space"):
                self.reference_policy_action = np.zeros(
                    self.action_space.shape, dtype=np.float32
                )
        else:
            action_progress = float(
                np.clip(
                    target_progress,
                    self.anchor_cumulative_length[0],
                    self.anchor_cumulative_length[-1],
                )
            )
            upper = int(
                np.searchsorted(
                    self.anchor_cumulative_length, action_progress, side="right"
                )
            )
            upper = min(max(upper, 1), len(self.anchor_cumulative_length) - 1)
            lower = upper - 1
            start_arc = self.anchor_cumulative_length[lower]
            segment_length = self.anchor_cumulative_length[upper] - start_arc
            fraction = (
                0.0
                if segment_length <= 0
                else (action_progress - start_arc) / segment_length
            )
            self.reference_policy_action = np.asarray(
                (1.0 - fraction) * self.path_anchor_policy_actions[lower]
                + fraction * self.path_anchor_policy_actions[upper],
                dtype=np.float32,
            )

    def _pose_map_action(self, target: np.ndarray) -> np.ndarray:
        if (
            self.pose_map_triangulation is None
            or self.pose_map_policy_actions is None
        ):
            raise RuntimeError("table pose map is not configured")
        dimensions = int(getattr(self, "pose_map_interpolation_dimensions", 2))
        query = np.asarray(target, dtype=np.float64)[:dimensions]
        # Keep the interpolation helper usable by lightweight map-validation
        # tools/tests that construct an uninitialized environment shell.
        # Fully initialized environments always set this attribute explicitly.
        if getattr(self, "pose_map_interpolation_mode", "barycentric") == "nearest":
            nearest = int(
                np.argmin(
                    np.linalg.norm(
                        self.pose_map_tip_positions[:, :dimensions] - query,
                        axis=1,
                    )
                )
            )
            return np.asarray(
                self.pose_map_policy_actions[nearest], dtype=np.float32
            ).copy()
        simplex = int(self.pose_map_triangulation.find_simplex(query, tol=1e-8))
        if simplex < 0:
            raise ValueError("reference target lies outside the certified pose-map hull")
        if (
            self.pose_map_valid_simplices is not None
            and not self.pose_map_valid_simplices[simplex]
        ):
            valid_indices = np.flatnonzero(self.pose_map_valid_simplices)
            transforms = self.pose_map_triangulation.transform[valid_indices]
            delta = query - transforms[:, dimensions]
            first = np.einsum(
                "nij,nj->ni", transforms[:, :dimensions], delta
            )
            weights = np.column_stack((first, 1.0 - np.sum(first, axis=1)))
            candidates = np.flatnonzero(np.all(weights >= -1e-7, axis=1))
            if not len(candidates):
                raise ValueError(
                    "reference target lies in an under-sampled pose-map simplex"
                )
            simplex = int(valid_indices[candidates[0]])
        transform = self.pose_map_triangulation.transform[simplex]
        first = transform[:dimensions] @ (query - transform[dimensions])
        weights = np.concatenate((first, [1.0 - np.sum(first)]))
        vertices = self.pose_map_triangulation.simplices[simplex]
        action = weights @ self.pose_map_policy_actions[vertices]
        return np.asarray(action, dtype=np.float32)

    def _initialize_internal_waypoint_gate(self) -> None:
        if self.path is None:
            raise RuntimeError("path must exist before initializing waypoint gates")
        segment_lengths = np.linalg.norm(np.diff(self.path.anchors, axis=0), axis=1)
        self.anchor_cumulative_length = np.concatenate(
            ([0.0], np.cumsum(segment_lengths, dtype=np.float64))
        )
        self.next_internal_waypoint_index = 1
        self.internal_waypoints_completed = 0
        self.waypoint_passed = False
        self.active_waypoint_distance = (
            self._task_space_distance(
                self.path.anchors[1], self.path.anchors[0]
            )
            if len(self.path.anchors) > 1
            else float("nan")
        )
        self.internal_waypoint_distance_delta = 0.0
        self.normalized_internal_waypoint_progress = 0.0
        self.normalized_internal_waypoint_capture_error = 0.0
        self.best_active_waypoint_distance = self.active_waypoint_distance
        self.last_waypoint_improvement_step = 0

    def _task_space_distance(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        """XY-dominant distance with a dead band for vertical error.

        A zero vertical tolerance preserves the historical Euclidean metric.
        With a positive tolerance, height error is free inside the requested
        band and only the excess contributes to the distance.  The raw XYZ
        target error remains in the policy observation, so this changes no
        state dimensions and does not hide height from the controller.
        """

        delta = np.asarray(first, dtype=np.float64) - np.asarray(
            second, dtype=np.float64
        )
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("task-space points must be finite 3-D values")
        vertical_tolerance = float(
            getattr(self, "tracking_vertical_tolerance", 0.0)
        )
        if vertical_tolerance <= 0:
            return float(np.linalg.norm(delta))
        xy_error = float(np.linalg.norm(delta[:2]))
        vertical_excess = max(
            abs(float(delta[2])) - vertical_tolerance,
            0.0,
        )
        return float(np.hypot(xy_error, vertical_excess))

    def _within_task_tolerance(
        self,
        tip: np.ndarray,
        target: np.ndarray,
        xy_tolerance: float,
    ) -> bool:
        """Return whether a target is captured under the configured Z band."""

        delta = np.asarray(tip, dtype=np.float64) - np.asarray(
            target, dtype=np.float64
        )
        vertical_tolerance = float(
            getattr(self, "tracking_vertical_tolerance", 0.0)
        )
        if vertical_tolerance <= 0:
            return bool(np.linalg.norm(delta) <= xy_tolerance)
        return bool(
            np.linalg.norm(delta[:2]) <= xy_tolerance
            and abs(float(delta[2])) <= vertical_tolerance
        )

    def _path_projection(
        self,
        tip: np.ndarray,
        *,
        minimum_distance: float,
        maximum_distance: float,
    ) -> tuple[float, float, np.ndarray]:
        if self.path is None:
            raise RuntimeError("path must exist before projection")
        weights = (
            (1.0, 1.0, 0.0)
            if float(getattr(self, "tracking_vertical_tolerance", 0.0)) > 0
            else None
        )
        progress, _, projected = self.path.project(
            tip,
            minimum_distance=minimum_distance,
            maximum_distance=maximum_distance,
            coordinate_weights=weights,
        )
        return (
            progress,
            self._task_space_distance(tip, projected),
            projected,
        )

    def _internal_waypoint_distance(
        self,
        tip: np.ndarray,
        index: int | None = None,
    ) -> float:
        """Distance to one ordered internal waypoint, or NaN at the endpoint."""

        if self.path is None:
            return float("nan")
        active_index = (
            self.next_internal_waypoint_index if index is None else int(index)
        )
        if active_index < 1 or active_index >= len(self.path.anchors) - 1:
            return float("nan")
        return self._task_space_distance(
            tip, self.path.anchors[active_index]
        )

    def _internal_waypoint_progress_limit(self, proposed_limit: float) -> float:
        if (
            self.internal_waypoint_capture_radius <= 0
            or self.path is None
            or self.next_internal_waypoint_index >= len(self.path.anchors) - 1
        ):
            return float(proposed_limit)
        return float(
            min(
                proposed_limit,
                self.anchor_cumulative_length[self.next_internal_waypoint_index],
            )
        )

    def _capture_internal_waypoint(self, tip: np.ndarray) -> bool:
        self.waypoint_passed = False
        if self.path is None or self.next_internal_waypoint_index >= len(
            self.path.anchors
        ) - 1:
            self.active_waypoint_distance = float("nan")
            return False
        index = self.next_internal_waypoint_index
        anchor = np.asarray(self.path.anchors[index], dtype=np.float32)
        self.active_waypoint_distance = self._task_space_distance(tip, anchor)
        if self.internal_waypoint_capture_radius <= 0:
            return False
        anchor_progress = float(self.anchor_cumulative_length[index])
        close_in_arc = self.path_progress >= (
            anchor_progress - self.internal_waypoint_capture_radius
        )
        if (
            close_in_arc
            and self._within_task_tolerance(
                tip, anchor, self.internal_waypoint_capture_radius
            )
        ):
            self.internal_waypoints_completed += 1
            self.next_internal_waypoint_index += 1
            self.waypoint_passed = True
            if self.next_internal_waypoint_index < len(self.path.anchors) - 1:
                self.active_waypoint_distance = self._task_space_distance(
                    tip,
                    self.path.anchors[self.next_internal_waypoint_index],
                )
            else:
                self.active_waypoint_distance = float("nan")
        return self.waypoint_passed

    def _observation(self, physical_state: np.ndarray) -> np.ndarray:
        tip = self._tip_position(physical_state)
        legacy_observation = np.concatenate(
            (
                np.asarray(physical_state, dtype=np.float32),
                self.previous_action,
                self.current_target - tip,
                self.lookahead_target - tip,
                np.asarray([self.desired_speed / self.max_desired_speed], dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
        if self.observation_mode == "legacy70":
            observation = legacy_observation
            expected_dim = MANISOFT_WAYPOINT_SAC_OBSERVATION_DIM
        else:
            active_distance = self._internal_waypoint_distance(tip)
            capture_error = (
                np.clip(
                    max(
                        active_distance - self.internal_waypoint_capture_radius,
                        0.0,
                    )
                    / max(self.internal_waypoint_capture_radius, 1e-6),
                    0.0,
                    4.0,
                )
                if np.isfinite(active_distance)
                and self.internal_waypoint_capture_radius > 0
                else 0.0
            )
            prior = self._cartesian_prior_action(physical_state)
            observation = np.concatenate(
                (
                    legacy_observation,
                    prior,
                    np.asarray(
                        [
                            self._effective_cartesian_prior_weight(),
                            capture_error,
                        ],
                        dtype=np.float32,
                    ),
                )
            ).astype(np.float32, copy=False)
            expected_dim = MANISOFT_WAYPOINT_SAC_GATE_OBSERVATION_DIM
        if observation.shape != (expected_dim,):
            raise RuntimeError(f"unexpected SAC observation shape: {observation.shape}")
        if not np.isfinite(observation).all():
            raise FloatingPointError("SAC observation contains NaN or Inf")
        return observation

    def _cross_track_distance(self, tip: np.ndarray) -> float:
        if self.path is None:
            return float("nan")
        # Dense path spacing is 5 mm, so nearest-point distance is an adequate
        # and cheap approximation for diagnostics and reward monitoring.
        vertical_tolerance = float(
            getattr(self, "tracking_vertical_tolerance", 0.0)
        )
        if vertical_tolerance <= 0:
            return float(
                np.min(np.linalg.norm(self.path.points - tip[None, :], axis=1))
            )
        deltas = np.asarray(self.path.points, dtype=np.float64) - np.asarray(
            tip, dtype=np.float64
        )[None, :]
        xy_error = np.linalg.norm(deltas[:, :2], axis=1)
        vertical_excess = np.maximum(
            np.abs(deltas[:, 2]) - vertical_tolerance,
            0.0,
        )
        return float(np.min(np.hypot(xy_error, vertical_excess)))

    def _whole_arm_table_clearance(self) -> float:
        """Signed capsule clearance from the finite virtual table body."""

        if not self.enforce_whole_arm_table_clearance:
            return float("inf")
        if self.sim is None or not hasattr(self.sim, "_backend"):
            return float("inf")
        soft = self.sim._backend.softrobot_state
        nodes = np.asarray(soft.element_positions, dtype=np.float64)
        minimum = np.asarray(
            [self.table_x_bounds[0], self.table_y_bounds[0], -2.0],
            dtype=np.float64,
        )
        maximum = np.asarray(
            [self.table_x_bounds[1], self.table_y_bounds[1], self.table_surface_z],
            dtype=np.float64,
        )
        centerline_distance = min(
            segment_aabb_distance(start, end, minimum, maximum)
            for start, end in zip(nodes[:-1], nodes[1:])
        )
        return float(
            centerline_distance - self.arm_radius - self.table_safety_margin
        )

    def _require_entry_bank(self, curriculum: str) -> TableEntryTrajectoryBank:
        if self.entry_bank is None:
            raise ValueError(
                f"curriculum {curriculum!r} requires entry_bank_path; "
                "generate the certified bank first"
            )
        return self.entry_bank

    def _select_entry_index(self, options: dict[str, Any]) -> int:
        bank = self._require_entry_bank(self.active_curriculum)
        supplied = options.get("entry_index")
        index = (
            int(
                self.np_random.choice(
                    bank.trajectory_count,
                    p=self.entry_sampling_weights,
                )
            )
            if supplied is None
            else int(supplied)
        )
        if not 0 <= index < bank.trajectory_count:
            raise ValueError(f"entry_index must be in [0, {bank.trajectory_count})")
        return index

    def _sample_entry_fraction(
        self,
        curriculum: str,
        supplied_fraction: float | None,
    ) -> float:
        """Sample a reset fraction without letting long failures erase anchors.

        Early ``entry_mid`` failures contribute many more replay transitions
        than short successful suffixes.  Optional anchor oversampling balances
        transitions at the episode source while explicit evaluation fractions
        remain exact and deterministic.
        """

        if supplied_fraction is not None:
            fraction = float(supplied_fraction)
        elif curriculum == "entry_mid" and self.entry_mid_anchor_probability > 0:
            use_anchor = bool(
                self.np_random.random() < self.entry_mid_anchor_probability
            )
            bounds = (
                self.entry_mid_anchor_fraction_range
                if use_anchor
                else self.entry_mid_advance_fraction_range
            )
            fraction = float(self.np_random.uniform(bounds[0], bounds[1]))
        elif curriculum == "entry":
            # Full-start episodes expose the complete upright-to-table path.
            # Shorter certified suffixes keep those long early failures from
            # erasing the already reliable entry_mid behavior.
            use_anchor = bool(
                self.np_random.random() < self.entry_anchor_probability
            )
            fraction = (
                float(self.np_random.uniform(*self.entry_anchor_fraction_range))
                if use_anchor
                else 0.0
            )
        else:
            default_ranges = {
                "entry_tail": (0.72, 0.92),
                # Overlap substantially with entry_tail before exposing
                # the policy to the much earlier entry_mid snapshots.
                "entry_bridge": (0.55, 0.85),
                "entry_mid": (0.30, 0.75),
                "recovery": (0.45, 0.90),
            }
            low, high = default_ranges[curriculum]
            fraction = float(self.np_random.uniform(low, high))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("warm_start_fraction must lie in [0, 1]")
        return fraction

    def _restore_entry_snapshot(self, entry_index: int, steps: int) -> np.ndarray:
        bank = self._require_entry_bank(self.active_curriculum)
        steps = int(np.clip(steps, 0, bank.transition_count))
        # Restore the four primary Cosserat-rod state arrays in-place. This is
        # deterministic and cuts a 400-control-step reset to a memory copy;
        # the bank generator separately certifies the same trajectory by full
        # replay, and reset rejects any snapshot/state mismatch.
        rod = self.sim._backend._softrobot
        rod.position_collection[...] = bank.node_positions[entry_index, steps].T
        rod.velocity_collection[...] = bank.node_velocities[entry_index, steps].T
        rod.director_collection[...] = bank.element_directors[
            entry_index, steps
        ].transpose(1, 2, 0)
        rod.omega_collection[...] = bank.element_omegas[entry_index, steps].T
        restore_rod_internal_state(
            rod, bank.rod_internal_states[entry_index, steps]
        )
        # Keep time-based callbacks and the environment step budget consistent
        # with the restored trajectory point.
        self.sim._backend.time_tracker += steps * self.control_dt
        self.sim.current_step += steps * int(
            round(self.control_dt / self.sim._backend.dt)
        )
        if steps:
            self.muscle.set_activation(
                bank.actions[entry_index, steps - 1].reshape(6, 3)
            )
        state = self._physical_state()
        expected = bank.physical_states[entry_index, steps]
        maximum_error = float(np.max(np.abs(state - expected)))
        if maximum_error > 5e-4:
            raise RuntimeError(
                "entry warm-start snapshot diverged from the certified bank: "
                f"max state error {maximum_error:.3e}"
            )
        return np.asarray(state, dtype=np.float32)

    def _entry_path(self, entry_index: int, prefix_steps: int = 0) -> ReferencePath:
        bank = self._require_entry_bank(self.active_curriculum)
        points = bank.tip_positions[entry_index, prefix_steps:]
        anchors = points[
            np.unique(np.linspace(0, len(points) - 1, 8).astype(int))
        ]
        return ReferencePath.from_points(
            f"certified_entry:{bank.names[entry_index]}", anchors, points
        )

    def _entry_plus_local_path(
        self,
        entry_index: int,
        prefix_steps: int,
        family: str | None,
    ) -> ReferencePath:
        entry = self._entry_path(entry_index, prefix_steps)
        local = self.path_generator.generate(
            self.np_random,
            entry.points[-1],
            curriculum="entry_local",
            family=family,
        )
        points = np.vstack((entry.points, local.points[1:]))
        anchors = np.vstack((entry.anchors, local.anchors[1:]))
        return ReferencePath.from_points(
            f"{entry.family}+{local.family}", anchors, points
        )

    def _table_equilibrium_long_path(self, entry_index: int) -> ReferencePath:
        """Sample 3--5 commanded waypoints from a certified equilibrium path.

        The restored entry pose is the path origin and is deliberately not
        counted as a commanded waypoint.  When five targets are requested the
        certified bank is traversed to its far end and the final segment
        reverses to the preceding equilibrium.  This keeps every commanded
        chord close to 10 cm without extrapolating beyond the validated
        workspace.
        """

        if (
            self.equilibrium_path_entry_index is None
            or self.equilibrium_path_tip_positions is None
            or self.equilibrium_path_policy_actions is None
        ):
            raise ValueError(
                "table_long_waypoints requires table_equilibrium_path_bank_path"
            )
        if entry_index != self.equilibrium_path_entry_index:
            raise ValueError(
                "table_long_waypoints reset selected an entry not covered by "
                "the equilibrium path bank"
            )
        counts = np.arange(
            self.path_generator.waypoint_segment_count_range[0],
            self.path_generator.waypoint_segment_count_range[1] + 1,
        )
        maximum_unique_segments = len(self.equilibrium_path_tip_positions) - 1
        maximum_segments = (
            maximum_unique_segments * 2
            if maximum_unique_segments > 0
            else 0
        )
        counts = counts[counts <= maximum_segments]
        if len(counts) == 0:
            raise ValueError("path bank has too few points for configured segment counts")
        probabilities = self.path_generator.waypoint_segment_count_probabilities
        if probabilities is not None:
            configured_counts = np.arange(
                self.path_generator.waypoint_segment_count_range[0],
                self.path_generator.waypoint_segment_count_range[1] + 1,
            )
            mask = configured_counts <= maximum_segments
            probabilities = probabilities[mask]
            probabilities = probabilities / np.sum(probabilities)
        segment_count = int(self.np_random.choice(counts, p=probabilities))
        forward = np.arange(len(self.equilibrium_path_tip_positions), dtype=int)
        backward = np.arange(
            len(self.equilibrium_path_tip_positions) - 2,
            -1,
            -1,
            dtype=int,
        )
        traversal = np.concatenate((forward, backward))
        anchor_indices = traversal[: segment_count + 1]
        anchors = self.equilibrium_path_tip_positions[anchor_indices].copy()
        self.path_anchor_policy_actions = self.equilibrium_path_policy_actions[
            anchor_indices
        ].copy()
        dense_rows = [np.asarray(anchors[0], dtype=np.float64)]
        for start, end in zip(anchors[:-1], anchors[1:]):
            distance = float(np.linalg.norm(end - start))
            count = max(
                2,
                int(np.ceil(distance / self.path_generator.dense_spacing)) + 1,
            )
            dense_rows.extend(
                np.linspace(start, end, count, endpoint=True)[1:]
            )
        return ReferencePath.from_points(
            "long_waypoint_polyline",
            anchors,
            np.asarray(dense_rows),
            generation_mode="certified_equilibrium_bank",
        )

    def _info(
        self,
        physical_state: np.ndarray,
        *,
        requested_action: np.ndarray | None = None,
        applied_delta: np.ndarray | None = None,
        is_success: bool = False,
        table_violation: bool = False,
    ) -> dict[str, Any]:
        if self.path is None:
            raise RuntimeError("environment must be reset before info is requested")
        tip = self._tip_position(physical_state)
        tip_tangent, orientation_error = self._tip_orientation()
        requested = self.previous_action if requested_action is None else requested_action
        delta = np.zeros(18, dtype=np.float32) if applied_delta is None else applied_delta
        target_delta = np.asarray(tip, dtype=np.float64) - np.asarray(
            self.current_target, dtype=np.float64
        )
        tolerance = np.finfo(np.float32).eps * 4
        saturated = np.logical_or(
            self.previous_action <= -self.absolute_action_limit + tolerance,
            self.previous_action >= self.absolute_action_limit - tolerance,
        )
        rate_clipped = np.abs(requested - self.previous_action) > tolerance
        return {
            "tip_position": tip.copy(),
            "tip_tangent": tip_tangent.astype(np.float32),
            "tip_orientation_error_degrees": float(orientation_error),
            "tip_orientation_violation": bool(
                orientation_error
                > self.maximum_tip_orientation_error_degrees
            ),
            "target_tip": self.current_target.copy(),
            "lookahead_tip": self.lookahead_target.copy(),
            "distance": float(np.linalg.norm(target_delta)),
            "task_distance": self._task_space_distance(
                tip, self.current_target
            ),
            "target_xy_error": float(np.linalg.norm(target_delta[:2])),
            "target_z_error": abs(float(target_delta[2])),
            "cross_track_distance": self._cross_track_distance(tip),
            "tracking_vertical_tolerance": float(
                getattr(self, "tracking_vertical_tolerance", 0.0)
            ),
            "path_progress": float(self.path_progress / max(self.path.length, 1e-9)),
            "path_progress_m": float(self.path_progress),
            "path_length": float(self.path.length),
            "waypoint_count": int(len(self.path.anchors) - 1),
            "waypoints_completed": int(
                self.internal_waypoints_completed + int(is_success)
            ),
            "internal_waypoints_completed": int(
                self.internal_waypoints_completed
            ),
            "active_waypoint_index": int(self.next_internal_waypoint_index),
            "active_waypoint_distance": float(self.active_waypoint_distance),
            "internal_waypoint_distance_delta": float(
                self.internal_waypoint_distance_delta
            ),
            "normalized_internal_waypoint_progress": float(
                self.normalized_internal_waypoint_progress
            ),
            "normalized_internal_waypoint_capture_error": float(
                self.normalized_internal_waypoint_capture_error
            ),
            "last_waypoint_improvement_step": int(
                self.last_waypoint_improvement_step
            ),
            "waypoint_passed": bool(self.waypoint_passed),
            "desired_speed": float(self.desired_speed),
            "path_family": self.path.family,
            "path_generation_mode": self.path.generation_mode,
            "path_segment_lengths": np.linalg.norm(
                np.diff(self.path.anchors, axis=0), axis=1
            ).astype(np.float32),
            "path_turn_angles_degrees": self._path_turn_angles_degrees(),
            "curriculum": self.curriculum,
            "active_curriculum": self.active_curriculum,
            "action_mode": self.action_mode,
            "cartesian_prior_weight": float(self.cartesian_prior_weight),
            "cartesian_prior_residual_scale": float(
                self.cartesian_prior_residual_scale
            ),
            "equilibrium_path_prior_weight": float(
                self.equilibrium_path_prior_weight
            ),
            "equilibrium_path_residual_scale": float(
                self.equilibrium_path_residual_scale
            ),
            "effective_cartesian_prior_weight": float(
                self._effective_cartesian_prior_weight()
            ),
            "effective_controller_prior_weight": float(
                self._effective_cartesian_prior_weight()
            ),
            "reference_policy_action": self.reference_policy_action.copy(),
            "path_anchors": self.path.anchors.copy(),
            "projected_point": self.projected_point.copy(),
            "raw_policy_action": self.last_raw_policy_action.copy(),
            "controller_prior_action": self.last_controller_prior_action.copy(),
            "blended_policy_action": self.last_blended_policy_action.copy(),
            "requested_action": np.asarray(requested, dtype=np.float32).copy(),
            "applied_action": self.previous_action.copy(),
            "applied_delta_action": np.asarray(delta, dtype=np.float32).copy(),
            "action_saturation_ratio": float(np.mean(saturated)),
            "action_rate_clipped_ratio": float(np.mean(rate_clipped)),
            "table_violation": bool(table_violation),
            "whole_arm_table_clearance": float(self.last_table_clearance),
            "stalled": bool(self.stalled),
            "path_progress_stalled": bool(self.path_progress_stalled),
            "waypoint_stalled": bool(self.waypoint_stalled),
            "terminal_timeout": bool(self.terminal_timeout),
            "dynamics_violation": bool(self.dynamics_violation),
            "entry_index": self.entry_index,
            "entry_prefix_steps": int(self.entry_prefix_steps),
            "is_success": bool(is_success),
            "success_streak": int(self.success_count),
        }

    def _path_turn_angles_degrees(self) -> np.ndarray:
        if self.path is None or len(self.path.anchors) < 3:
            return np.zeros(0, dtype=np.float32)
        vectors = np.diff(self.path.anchors[:, :2], axis=0)
        headings = np.arctan2(vectors[:, 1], vectors[:, 0])
        signed_turns = np.arctan2(
            np.sin(np.diff(headings)), np.cos(np.diff(headings))
        )
        return np.rad2deg(signed_turns).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        physical_state, _ = super().reset(seed=seed, options=options)
        options = {} if options is None else dict(options)
        curriculum = str(options.get("curriculum", self.curriculum))
        if curriculum not in CURRICULUM_STAGES:
            raise ValueError(f"unknown curriculum stage: {curriculum}")
        family = options.get("path_family")
        anchors = options.get("anchors")
        self.curriculum = curriculum
        self.active_curriculum = curriculum
        bank_curricula = {
            "entry_tail",
            "entry_bridge",
            "entry_mid",
            "entry",
            "table_local_line",
            "table_waypoint_polyline",
            "table_long_waypoints",
            "table_local",
            "entry_local",
            "recovery",
        }
        if self.entry_bank is not None:
            if curriculum == "table_point":
                self.active_curriculum = "entry"
            elif curriculum == "path":
                self.active_curriculum = "table_local"
            elif curriculum == "mixed":
                self.active_curriculum = str(
                    self.np_random.choice(
                        ("entry", "table_local", "entry_local"),
                        p=np.asarray((0.25, 0.35, 0.40)),
                    )
                )
        self.entry_index = None
        self.entry_prefix_steps = 0
        self.path_anchor_policy_actions = None
        self.reference_policy_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )
        warm_action = np.zeros(18, dtype=np.float32)
        if anchors is not None:
            # Explicit tabletop evaluation paths must begin at the certified
            # bent posture, not at the simulator's natural upright reset.
            # This also makes fixed long-path acceptance panels reproducible.
            if (
                self.entry_bank is not None
                and self.active_curriculum
                in {
                    "table_local_line",
                    "table_waypoint_polyline",
                    "table_long_waypoints",
                    "table_local",
                }
            ):
                bank = self._require_entry_bank(self.active_curriculum)
                self.entry_index = self._select_entry_index(options)
                self.entry_prefix_steps = bank.transition_count
                physical_state = self._restore_entry_snapshot(
                    self.entry_index, self.entry_prefix_steps
                )
                warm_action = bank.actions[self.entry_index, -1].copy()
            tip = self._tip_position(physical_state)
            self.path = self.path_generator.generate(
                self.np_random,
                tip,
                curriculum=curriculum,
                family=family,
                anchors=anchors,
            )
            self.active_curriculum = "custom"
        elif self.active_curriculum in bank_curricula:
            bank = self._require_entry_bank(self.active_curriculum)
            self.entry_index = self._select_entry_index(options)
            if self.active_curriculum in {
                "table_local_line",
                "table_waypoint_polyline",
                "table_long_waypoints",
                "table_local",
            }:
                self.entry_prefix_steps = bank.transition_count
                physical_state = self._restore_entry_snapshot(
                    self.entry_index, self.entry_prefix_steps
                )
                warm_action = bank.actions[self.entry_index, -1].copy()
                tip = self._tip_position(physical_state)
                self.path = (
                    self._table_equilibrium_long_path(self.entry_index)
                    if self.active_curriculum == "table_long_waypoints"
                    else self.path_generator.generate(
                        self.np_random,
                        tip,
                        curriculum=self.active_curriculum,
                        family=family,
                    )
                )
            elif self.active_curriculum in {
                "entry_tail",
                "entry_bridge",
                "entry_mid",
                "entry",
                "recovery",
            }:
                fraction = self._sample_entry_fraction(
                    self.active_curriculum,
                    options.get("warm_start_fraction"),
                )
                self.entry_prefix_steps = int(
                    round(fraction * bank.transition_count)
                )
                physical_state = self._restore_entry_snapshot(
                    self.entry_index, self.entry_prefix_steps
                )
                if self.entry_prefix_steps:
                    warm_action = bank.actions[
                        self.entry_index, self.entry_prefix_steps - 1
                    ].copy()
                self.path = (
                    self._entry_plus_local_path(
                        self.entry_index, self.entry_prefix_steps, family
                    )
                    if self.active_curriculum == "recovery"
                    else self._entry_path(
                        self.entry_index, self.entry_prefix_steps
                    )
                )
            elif self.active_curriculum == "entry_local":
                self.path = self._entry_plus_local_path(
                    self.entry_index, 0, family
                )
            else:
                self.path = self._entry_path(self.entry_index)
        else:
            tip = self._tip_position(physical_state)
            self.path = self.path_generator.generate(
                self.np_random,
                tip,
                curriculum=curriculum,
                family=family,
            )
        speed = options.get("desired_speed")
        self.desired_speed = (
            self._sample_speed(
                self.active_curriculum
                if self.active_curriculum in CURRICULUM_STAGES
                else curriculum
            )
            if speed is None
            else float(speed)
        )
        if not self.min_desired_speed <= self.desired_speed <= self.max_desired_speed:
            raise ValueError(
                "desired_speed must lie inside the configured training speed range"
            )
        tip = self._tip_position(physical_state)
        self._initialize_internal_waypoint_gate()
        self.active_waypoint_distance = self._internal_waypoint_distance(tip)
        self.best_active_waypoint_distance = self.active_waypoint_distance
        projected_progress, _, projected_point = self._path_projection(
            tip,
            minimum_distance=0.0,
            maximum_distance=min(self.path.length, self.maximum_projection_advance),
        )
        self.path_progress = float(projected_progress)
        self.projected_point = projected_point
        self._update_reference_targets()
        self.previous_action = warm_action.astype(np.float32, copy=True)
        self.equilibrium_action = warm_action.astype(np.float32, copy=True)
        self.cartesian_prior_start_tip = tip.copy()
        self.last_raw_policy_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )
        self.last_controller_prior_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )
        self.last_blended_policy_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )
        self.previous_tip = tip.copy()
        self.previous_tracking_distance = self._task_space_distance(
            tip, self.current_target
        )
        self.previous_final_distance = self._task_space_distance(
            tip, self.path.points[-1]
        )
        self.last_physical_state = np.asarray(physical_state, dtype=np.float32)
        self.step_count = 0
        self.success_count = 0
        self.table_violation_count = 0
        self.best_path_progress = self.path_progress
        self.last_progress_step = 0
        self.terminal_entry_step = None
        self.terminal_timeout = False
        self.stalled = False
        self.path_progress_stalled = False
        self.waypoint_stalled = False
        self.dynamics_violation = False
        self.last_table_clearance = self._whole_arm_table_clearance()
        if self.last_table_clearance < -self.table_clearance_violation_tolerance:
            raise RuntimeError(
                "reset/warm-start posture violates the configured whole-arm "
                f"table clearance by {-self.last_table_clearance:.4f} m"
            )
        observation = self._observation(self.last_physical_state)
        return observation, self._info(self.last_physical_state)

    def step(self, requested_action: np.ndarray):
        if self.path is None or self.last_physical_state is None:
            raise RuntimeError("environment must be reset before step")
        raw_policy_action = np.asarray(requested_action, dtype=np.float32).reshape(-1)
        policy_action_penalty = self.policy_action_penalty_scale * float(
            np.mean(np.square(raw_policy_action))
        )
        policy_action = self._blend_cartesian_prior(raw_policy_action)
        requested = self._physical_action_request(policy_action)
        previous_action = self.previous_action.copy()
        applied = np.clip(
            requested,
            previous_action - self.max_action_delta,
            previous_action + self.max_action_delta,
        )
        applied = np.clip(
            applied,
            -self.absolute_action_limit,
            self.absolute_action_limit,
        ).astype(np.float32)
        applied_delta = applied - previous_action
        requested_rate_excess = np.clip(
            (requested - applied) / self.max_action_delta,
            -10.0,
            10.0,
        )
        self.muscle.set_activation(applied.reshape(6, 3))

        def current_torque(element_lengths: np.ndarray) -> np.ndarray:
            return self.muscle.evaluate(element_lengths)

        self.sim.step_with_torque_callback(current_torque)
        try:
            physical_state = self._physical_state()
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            # Pure SAC must be allowed to explore bad actions without one
            # numerically invalid rod state killing every vector worker.  The
            # simulator is reset by Gym after this terminal transition.
            self.step_count += 1
            self.dynamics_violation = True
            self.previous_action = applied
            observation = self._observation(self.last_physical_state)
            info = self._info(
                self.last_physical_state,
                requested_action=requested,
                applied_delta=applied_delta,
            )
            info.update(
                {
                    "tip_speed": 0.0,
                    "reference_advanced": False,
                    "reference_advance_m": 0.0,
                    "projected_speed": 0.0,
                    "signed_projection_delta_m": 0.0,
                    "final_distance": self._task_space_distance(
                        self._tip_position(self.last_physical_state),
                        self.path.points[-1],
                    ),
                    "requested_rate_penalty": float(
                        self.requested_rate_penalty_scale
                        * np.mean(np.square(requested_rate_excess))
                    ),
                    "policy_action_penalty": float(policy_action_penalty),
                    "table_violation_count": int(self.table_violation_count),
                    "dynamics_error": type(error).__name__,
                }
            )
            return observation, -25.0, True, False, info
        tip = self._tip_position(physical_state)
        tip_speed = float(np.linalg.norm(tip - self.previous_tip) / self.control_dt)
        previous_progress = self.path_progress
        active_waypoint_index_before = self.next_internal_waypoint_index
        previous_active_waypoint_distance = self.active_waypoint_distance
        projected_progress, cross_track_distance, projected_point = self._path_projection(
            tip,
            minimum_distance=max(0.0, previous_progress - self.projection_backtrack),
            maximum_distance=self._internal_waypoint_progress_limit(
                min(
                    self.path.length,
                    previous_progress + self.maximum_projection_advance,
                )
            ),
        )
        signed_projection_delta = projected_progress - previous_progress
        self.path_progress = max(previous_progress, projected_progress)
        advance = self.path_progress - previous_progress
        self.projected_point = projected_point
        current_active_waypoint_distance = self._internal_waypoint_distance(
            tip, active_waypoint_index_before
        )
        if np.isfinite(previous_active_waypoint_distance) and np.isfinite(
            current_active_waypoint_distance
        ):
            self.internal_waypoint_distance_delta = float(
                previous_active_waypoint_distance - current_active_waypoint_distance
            )
        else:
            self.internal_waypoint_distance_delta = 0.0
        self.normalized_internal_waypoint_progress = float(
            np.clip(
                self.internal_waypoint_distance_delta
                / max(self.desired_speed * self.control_dt, 1e-4),
                -2.0,
                2.0,
            )
        )
        self.normalized_internal_waypoint_capture_error = float(
            np.clip(
                max(
                    current_active_waypoint_distance
                    - self.internal_waypoint_capture_radius,
                    0.0,
                )
                / max(self.internal_waypoint_capture_radius, 1e-6)
                if np.isfinite(current_active_waypoint_distance)
                and self.internal_waypoint_capture_radius > 0
                else 0.0,
                0.0,
                4.0,
            )
        )
        waypoint_passed = self._capture_internal_waypoint(tip)
        # Capturing a waypoint advances the ordered gate, so update afterwards
        # to expose the next segment immediately in the returned observation.
        self._update_reference_targets()
        tracking_distance = self._task_space_distance(tip, self.current_target)
        final_target = np.asarray(self.path.points[-1], dtype=np.float32)
        final_delta = np.asarray(tip, dtype=np.float64) - final_target
        final_distance = self._task_space_distance(tip, final_target)
        final_euclidean_distance = float(np.linalg.norm(final_delta))
        final_xy_distance = float(np.linalg.norm(final_delta[:2]))
        final_z_error = abs(float(final_delta[2]))
        normalized_progress = float(
            np.clip(
                signed_projection_delta
                / max(self.desired_speed * self.control_dt, 1e-4),
                -2.0,
                2.0,
            )
        )
        projected_speed = max(signed_projection_delta, 0.0) / self.control_dt
        speed_error = min(
            abs(projected_speed - self.desired_speed)
            / max(self.max_desired_speed, 1e-9),
            3.0,
        )
        normalized_delta = applied_delta / self.max_action_delta
        normalized_action = applied / self.absolute_action_limit
        requested_rate_penalty = self.requested_rate_penalty_scale * float(
            np.mean(np.square(requested_rate_excess))
        )
        normalized_endpoint_progress = float(
            np.clip(
                (self.previous_final_distance - final_distance)
                / max(self.desired_speed * self.control_dt, 1e-4),
                -2.0,
                2.0,
            )
        )
        reward = (
            0.65 * normalized_progress
            + self.internal_waypoint_progress_scale
            * self.normalized_internal_waypoint_progress
            - self.internal_waypoint_distance_penalty_scale
            * self.normalized_internal_waypoint_capture_error
            + self.endpoint_progress_scale * normalized_endpoint_progress
            - self.cross_track_penalty_scale
            * min(cross_track_distance / self.tracking_guard, 8.0)
            - self.tracking_distance_penalty_scale
            * min(tracking_distance / self.tracking_guard, 6.0)
            - 0.02 * speed_error
            - 0.005
            - 0.002 * float(np.mean(np.square(normalized_delta)))
            - 0.0005 * float(np.mean(np.square(normalized_action)))
            - requested_rate_penalty
            - policy_action_penalty
            - self.tip_orientation_penalty_scale
            * min((self._tip_orientation()[1] / 30.0) ** 2, 4.0)
        )
        if waypoint_passed:
            reward += self.internal_waypoint_bonus

        self.step_count += 1
        geometric_path_end = self.path_progress >= (
            self.path.length - max(self.success_threshold, 0.010)
        )
        at_path_end = bool(
            geometric_path_end
            and self._within_task_tolerance(
                tip, final_target, self.terminal_capture_radius
            )
        )
        within_success_tolerance = self._within_task_tolerance(
            tip, final_target, self.success_threshold
        )
        if at_path_end and within_success_tolerance:
            self.success_count += 1
        else:
            self.success_count = 0
        success = self.success_count >= self.required_success_streak
        if at_path_end and self.terminal_entry_step is None:
            self.terminal_entry_step = self.step_count
        elif not at_path_end:
            # Terminal settling is a consecutive dwell requirement. Leaving
            # the capture ball starts a fresh attempt rather than inheriting
            # elapsed time from an earlier transient visit.
            self.terminal_entry_step = None
        self.terminal_timeout = bool(
            at_path_end
            and not success
            and self.terminal_entry_step is not None
            and self.step_count - self.terminal_entry_step
            >= self.terminal_settle_steps
        )

        self.last_table_clearance = self._whole_arm_table_clearance()
        table_violation = bool(
            tip[2]
            < self.minimum_tip_z - self.table_clearance_violation_tolerance
            or self.last_table_clearance
            < -self.table_clearance_violation_tolerance
        )
        orientation_error = self._tip_orientation()[1]
        orientation_violation = bool(
            orientation_error > self.maximum_tip_orientation_error_degrees
        )
        if table_violation:
            self.table_violation_count += 1
            # A hard safety termination must be substantially worse than
            # continuing an imperfect rollout.  With only the old -10 cost,
            # SAC learned to touch the virtual table near the path end and
            # terminate early, thereby avoiding the accumulated tracking
            # costs of a longer attempt.
            reward -= self.table_violation_penalty
        if orientation_violation:
            reward -= self.tip_orientation_violation_penalty
        if at_path_end:
            normalized_final_distance = final_distance / self.success_threshold
            reward += self.terminal_precision_scale * float(
                np.exp(-0.5 * normalized_final_distance ** 2)
            )
            reward -= self.terminal_distance_penalty_scale * min(
                normalized_final_distance, 4.0
            )
            if within_success_tolerance:
                # Eight consecutive precise frames are required, so this
                # dense dwell reward teaches the behavior before the sparse
                # terminal bonus becomes common.
                reward += 0.15
            reward -= 0.01 * min(tip_speed, 5.0)
        if success:
            reward += 10.0
        elif self.terminal_timeout:
            # Bound the final precision attempt. Previously a policy that
            # stayed 5--14 cm from the endpoint could accrue terminal costs
            # for all 1800 steps, making an unsafe early termination cheaper.
            reward -= self.terminal_timeout_penalty
        if self.path_progress >= self.best_path_progress + self.stall_progress_epsilon:
            self.best_path_progress = self.path_progress
            self.last_progress_step = self.step_count
        self.path_progress_stalled = bool(
            not at_path_end
            and self.step_count >= self.stall_grace_steps
            and self.step_count - self.last_progress_step >= self.stall_window_steps
        )
        if waypoint_passed:
            self.best_active_waypoint_distance = self.active_waypoint_distance
            self.last_waypoint_improvement_step = self.step_count
        elif (
            np.isfinite(current_active_waypoint_distance)
            and (
                not np.isfinite(self.best_active_waypoint_distance)
                or current_active_waypoint_distance
                <= self.best_active_waypoint_distance
                - self.waypoint_stall_distance_epsilon
            )
        ):
            self.best_active_waypoint_distance = current_active_waypoint_distance
            self.last_waypoint_improvement_step = self.step_count
        self.waypoint_stalled = bool(
            self.waypoint_stall_steps > 0
            and np.isfinite(self._internal_waypoint_distance(tip))
            and self.step_count - self.last_waypoint_improvement_step
            >= self.waypoint_stall_steps
        )
        self.stalled = bool(
            self.path_progress_stalled or self.waypoint_stalled
        )
        if self.stalled:
            reward -= 2.0
        terminated = bool(
            success
            or (self.terminate_on_table_violation and table_violation)
            or (
                self.terminate_on_tip_orientation_violation
                and orientation_violation
            )
        )
        truncated = bool(
            (
                self.step_count >= self.episode_steps
                or self.stalled
                or self.terminal_timeout
            )
            and not terminated
        )
        self.previous_action = applied
        self.previous_tip = tip.copy()
        self.previous_tracking_distance = tracking_distance
        self.previous_final_distance = final_distance
        self.last_physical_state = np.asarray(physical_state, dtype=np.float32)
        observation = self._observation(self.last_physical_state)
        info = self._info(
            self.last_physical_state,
            requested_action=requested,
            applied_delta=applied_delta,
            is_success=success,
            table_violation=table_violation,
        )
        info.update(
            {
                "tip_speed": tip_speed,
                "reference_advanced": bool(advance > 0),
                "reference_advance_m": float(advance),
                "projected_speed": float(projected_speed),
                "signed_projection_delta_m": float(signed_projection_delta),
                "final_distance": float(final_distance),
                "final_euclidean_distance": float(final_euclidean_distance),
                "final_xy_distance": float(final_xy_distance),
                "final_z_error": float(final_z_error),
                "geometric_path_end": bool(geometric_path_end),
                "terminal_capture": bool(at_path_end),
                "normalized_endpoint_progress": normalized_endpoint_progress,
                "requested_rate_penalty": float(requested_rate_penalty),
                "policy_action_penalty": float(policy_action_penalty),
                "table_violation_count": int(self.table_violation_count),
            }
        )
        return observation, float(reward), terminated, truncated, info

    def trajectory_frame(self) -> dict[str, Any]:
        if self.last_physical_state is None or self.path is None:
            raise RuntimeError("environment must be reset before recording")
        soft = self.sim._backend.softrobot_state
        tip_tangent, orientation_error = self._tip_orientation()
        return {
            "softrobot_positions": np.asarray(
                soft.element_positions, dtype=np.float32
            ).copy(),
            "softrobot_directors": np.asarray(
                soft.element_directors, dtype=np.float32
            ).copy(),
            "tip_position": self._tip_position(self.last_physical_state).copy(),
            "tip_tangent": tip_tangent.astype(np.float32),
            "tip_orientation_error_degrees": float(orientation_error),
            "target_tip": self.current_target.copy(),
            "lookahead_tip": self.lookahead_target.copy(),
            "path_points": self.path.points.copy(),
            "path_anchors": self.path.anchors.copy(),
            "path_progress": float(self.path_progress),
        }
