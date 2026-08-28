from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np

from antmaze_ac.envs.kinematic_push_task import segment_aabb_distance


class WallRoutePhase(IntEnum):
    """Ordered geometric milestones for one wall-bypass episode."""

    UPRIGHT = 0
    SIDE_GATE = 1
    TIP_BEYOND_WALL = 2
    DISTAL_BODY_BEYOND_WALL = 3
    RETURNED_TO_TARGET_PLANE = 4
    TARGET_REACHED = 5


WALL_ROUTE_PHASE_NAMES = tuple(phase.name.lower() for phase in WallRoutePhase)


def _minimum_jerk(fraction: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def _spatial_basis(control_points: int = 6) -> np.ndarray:
    point = np.arange(1, control_points + 1)[:, None]
    mode = np.arange(1, control_points + 1)[None, :]
    return np.sqrt(2.0 / (control_points + 1)) * np.sin(
        np.pi * point * mode / (control_points + 1)
    )


def generate_smooth_wall_route_actions(
    rng: np.random.Generator,
    transition_count: int,
    control_dt: float,
    *,
    knot_seconds_range: Sequence[float],
    hold_seconds_range: Sequence[float],
    peak_range: Sequence[float],
    spatial_modes: int,
    absolute_action_limit: float,
    maximum_action_delta: float,
    zero_anchor_probability: float = 0.10,
) -> np.ndarray:
    """Generate rate-limited, low-spatial-mode 18-D candidate actions."""

    if transition_count < 1 or control_dt <= 0:
        raise ValueError("transition_count and control_dt must be positive")
    knot_bounds = _vector(knot_seconds_range, 2, "knot_seconds_range")
    hold_bounds = _vector(hold_seconds_range, 2, "hold_seconds_range")
    peak_bounds = _vector(peak_range, 2, "peak_range")
    if (
        knot_bounds[0] <= 0
        or knot_bounds[1] < knot_bounds[0]
        or hold_bounds[0] < 0
        or hold_bounds[1] < hold_bounds[0]
        or peak_bounds[0] <= 0
        or peak_bounds[1] < peak_bounds[0]
        or peak_bounds[1] > absolute_action_limit
    ):
        raise ValueError("invalid action duration or peak bounds")
    if not 1 <= spatial_modes <= 6:
        raise ValueError("spatial_modes must lie in [1,6]")
    if absolute_action_limit <= 0 or maximum_action_delta <= 0:
        raise ValueError("action limits must be positive")
    if not 0.0 <= zero_anchor_probability <= 1.0:
        raise ValueError("zero_anchor_probability must lie in [0,1]")

    basis = _spatial_basis()[:, :spatial_modes]
    actions = np.zeros((transition_count, 18), dtype=np.float64)
    current = np.zeros(18, dtype=np.float64)
    cursor = 0
    while cursor < transition_count:
        if rng.random() < zero_anchor_probability:
            target = np.zeros(18, dtype=np.float64)
        else:
            coefficients = rng.normal(size=(spatial_modes, 3))
            # Higher modes receive less energy, matching the broad Koopman
            # collector while concentrating this task search on smooth bends.
            coefficients *= np.linspace(1.0, 0.30, spatial_modes)[:, None]
            candidate = basis @ coefficients
            raw_peak = float(np.max(np.abs(candidate)))
            if raw_peak <= np.finfo(np.float64).eps:
                continue
            requested_peak = float(rng.uniform(peak_bounds[0], peak_bounds[1]))
            target = np.clip(
                candidate.reshape(-1) * requested_peak / raw_peak,
                -absolute_action_limit,
                absolute_action_limit,
            )

        difference = target - current
        difference_peak = float(np.max(np.abs(difference)))
        duration_steps = max(
            2,
            round(float(rng.uniform(knot_bounds[0], knot_bounds[1])) / control_dt),
        )
        # The maximum derivative of 10a^3-15a^4+6a^5 is 1.875.  This lower
        # bound makes the discrete action changes respect the requested limit.
        rate_steps = max(
            2,
            int(np.ceil(1.875 * difference_peak / maximum_action_delta)),
        )
        requested_steps = max(duration_steps, rate_steps)
        ramp_steps = min(requested_steps, transition_count - cursor)
        fractions = np.arange(1, ramp_steps + 1, dtype=np.float64) / requested_steps
        blend = _minimum_jerk(fractions)[:, None]
        actions[cursor : cursor + ramp_steps] = current + blend * difference
        cursor += ramp_steps
        current = actions[cursor - 1].copy()
        if cursor >= transition_count or ramp_steps < requested_steps:
            break

        hold_steps = min(
            round(float(rng.uniform(hold_bounds[0], hold_bounds[1])) / control_dt),
            transition_count - cursor,
        )
        if hold_steps:
            actions[cursor : cursor + hold_steps] = current
            cursor += hold_steps

    actions = np.clip(actions, -absolute_action_limit, absolute_action_limit)
    previous = np.vstack((np.zeros((1, 18)), actions[:-1]))
    observed_delta = float(np.max(np.abs(actions - previous)))
    if observed_delta > maximum_action_delta + 1e-8:
        raise RuntimeError(
            "generated action sequence violates maximum_action_delta: "
            f"{observed_delta:.6g} > {maximum_action_delta:.6g}"
        )
    return actions.astype(np.float32)


def _vector(value: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True)
class WallRouteGeometry:
    """Virtual geometry and phase thresholds for collection-time filtering.

    The wall has no physical contact model during collection.  A rollout is
    stopped before saving any transition whose swept rod capsule intersects
    the wall or whose non-mounted body falls below the virtual ground plane.
    """

    base: np.ndarray
    target: np.ndarray
    wall_minimum: np.ndarray
    wall_maximum: np.ndarray
    arm_radius: float
    wall_safety_margin: float
    ground_surface_z: float
    ground_safety_margin: float
    ground_violation_tolerance: float
    mounting_exempt_nodes: int
    prewall_y_margin: float
    postwall_y_margin: float
    crossed_node_fraction: float
    return_x_tolerance: float
    goal_tolerance: float
    goal_speed_tolerance: float
    goal_hold_steps: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WallRouteGeometry":
        wall = value["wall"]
        route = value["route"]
        ground = value["ground"]
        result = cls(
            base=_vector(value["base"], 3, "task.base"),
            target=_vector(value["target"], 3, "task.target"),
            wall_minimum=_vector(wall["minimum"], 3, "task.wall.minimum"),
            wall_maximum=_vector(wall["maximum"], 3, "task.wall.maximum"),
            arm_radius=float(wall["arm_radius"]),
            wall_safety_margin=float(wall.get("safety_margin", 0.0)),
            ground_surface_z=float(ground.get("surface_z", 0.0)),
            ground_safety_margin=float(ground.get("safety_margin", 0.0)),
            ground_violation_tolerance=float(
                ground.get("violation_tolerance", 0.0)
            ),
            mounting_exempt_nodes=int(ground.get("mounting_exempt_nodes", 1)),
            prewall_y_margin=float(route.get("prewall_y_margin", 0.0)),
            postwall_y_margin=float(route.get("postwall_y_margin", 0.0)),
            crossed_node_fraction=float(route["crossed_node_fraction"]),
            return_x_tolerance=float(route["return_x_tolerance"]),
            goal_tolerance=float(route["goal_tolerance"]),
            goal_speed_tolerance=float(route["goal_speed_tolerance"]),
            goal_hold_steps=int(route["goal_hold_steps"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if np.any(self.wall_maximum <= self.wall_minimum):
            raise ValueError("virtual wall must have positive extent on every axis")
        if min(
            self.arm_radius,
            self.return_x_tolerance,
            self.goal_tolerance,
            self.goal_speed_tolerance,
            self.goal_hold_steps,
        ) <= 0:
            raise ValueError("route radii, tolerances and hold steps must be positive")
        if min(
            self.wall_safety_margin,
            self.ground_safety_margin,
            self.ground_violation_tolerance,
            self.prewall_y_margin,
            self.postwall_y_margin,
            self.mounting_exempt_nodes,
        ) < 0:
            raise ValueError("safety margins and mounting exemption must be non-negative")
        if not 0.0 < self.crossed_node_fraction <= 1.0:
            raise ValueError("crossed_node_fraction must lie in (0, 1]")
        if not self.base[1] < self.wall_minimum[1] < self.wall_maximum[1] < self.target[1]:
            raise ValueError("wall must lie strictly between base and target on y")
        if not self.wall_minimum[0] <= self.base[0] <= self.wall_maximum[0]:
            raise ValueError("base x must lie behind the finite wall span")
        if not self.wall_minimum[0] <= self.target[0] <= self.wall_maximum[0]:
            raise ValueError("target x must lie behind the finite wall span")
        minimum_target_z = (
            self.ground_surface_z
            + self.ground_safety_margin
            + self.arm_radius
            - self.ground_violation_tolerance
        )
        if self.target[2] < minimum_target_z:
            raise ValueError(
                "target centre is too low to keep the soft-arm body above ground"
            )

    @property
    def wall_padding(self) -> float:
        return self.arm_radius + self.wall_safety_margin

    def side_gate_x(self, side: int) -> float:
        if side not in {-1, 1}:
            raise ValueError("route side must be -1 or +1")
        return float(
            self.wall_minimum[0] - self.wall_padding
            if side < 0
            else self.wall_maximum[0] + self.wall_padding
        )

    def whole_arm_wall_clearance(self, nodes: np.ndarray) -> float:
        points = _node_array(nodes)
        return float(
            min(
                segment_aabb_distance(start, end, self.wall_minimum, self.wall_maximum)
                for start, end in zip(points[:-1], points[1:])
            )
            - self.wall_padding
        )

    def whole_arm_ground_clearance(self, nodes: np.ndarray) -> float:
        """Return physical-body clearance above z=0 outside the mount.

        The fixed base node and the segment attached to it form the mounting
        collar.  They are exempt because a radius-5 cm rod centred at z=0
        necessarily intersects the mounting plane.  Every remaining node is
        treated as the centre of a radius-``arm_radius`` body.
        """

        points = _node_array(nodes)
        if self.mounting_exempt_nodes >= len(points):
            raise ValueError("mounting_exempt_nodes must leave at least one node")
        checked = points[self.mounting_exempt_nodes :]
        return float(
            np.min(checked[:, 2])
            - self.arm_radius
            - self.ground_surface_z
            - self.ground_safety_margin
        )

    def crossed_fraction(self, nodes: np.ndarray) -> float:
        points = _node_array(nodes)
        threshold = self.wall_maximum[1] + self.postwall_y_margin
        checked = points[self.mounting_exempt_nodes :]
        return float(np.mean(checked[:, 1] >= threshold))


def _node_array(nodes: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(nodes, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[1] != 3
        or len(result) < 2
        or not np.isfinite(result).all()
    ):
        raise ValueError("soft-arm nodes must have finite shape [N,3], N >= 2")
    return result


@dataclass(frozen=True)
class WallRouteStep:
    phase: int
    phase_name: str
    phase_advanced: bool
    side_gate_reached: bool
    crossed_fraction: float
    target_distance: float
    goal_hold_count: int
    success: bool


class WallRoutePhaseTracker:
    """Monotone phase tracker used to label one candidate rollout."""

    def __init__(self, geometry: WallRouteGeometry, side: int):
        if side not in {-1, 1}:
            raise ValueError("route side must be -1 or +1")
        self.geometry = geometry
        self.side = int(side)
        self.phase = WallRoutePhase.UPRIGHT
        self.goal_hold_count = 0

    def update(
        self,
        nodes: np.ndarray,
        tip_velocity: Sequence[float] | np.ndarray,
    ) -> WallRouteStep:
        points = _node_array(nodes)
        velocity = _vector(tip_velocity, 3, "tip_velocity")
        tip = points[-1]
        phase_before = self.phase
        gate_x = self.geometry.side_gate_x(self.side)
        side_gate_reached = bool(
            self.side * tip[0] >= self.side * gate_x
            and tip[1]
            <= self.geometry.wall_minimum[1] - self.geometry.prewall_y_margin
        )
        crossed_fraction = self.geometry.crossed_fraction(points)
        target_distance = float(np.linalg.norm(tip - self.geometry.target))

        if self.phase == WallRoutePhase.UPRIGHT and side_gate_reached:
            self.phase = WallRoutePhase.SIDE_GATE
        elif self.phase == WallRoutePhase.SIDE_GATE and tip[1] >= (
            self.geometry.wall_maximum[1] + self.geometry.postwall_y_margin
        ):
            self.phase = WallRoutePhase.TIP_BEYOND_WALL
        elif (
            self.phase == WallRoutePhase.TIP_BEYOND_WALL
            and crossed_fraction >= self.geometry.crossed_node_fraction
        ):
            self.phase = WallRoutePhase.DISTAL_BODY_BEYOND_WALL
        elif (
            self.phase == WallRoutePhase.DISTAL_BODY_BEYOND_WALL
            and tip[1]
            >= self.geometry.wall_maximum[1] + self.geometry.postwall_y_margin
            and abs(float(tip[0] - self.geometry.target[0]))
            <= self.geometry.return_x_tolerance
        ):
            self.phase = WallRoutePhase.RETURNED_TO_TARGET_PLANE

        if self.phase == WallRoutePhase.RETURNED_TO_TARGET_PLANE:
            at_goal = bool(
                target_distance <= self.geometry.goal_tolerance
                and np.linalg.norm(velocity) <= self.geometry.goal_speed_tolerance
            )
            self.goal_hold_count = self.goal_hold_count + 1 if at_goal else 0
            if self.goal_hold_count >= self.geometry.goal_hold_steps:
                self.phase = WallRoutePhase.TARGET_REACHED
        elif self.phase != WallRoutePhase.TARGET_REACHED:
            self.goal_hold_count = 0

        return WallRouteStep(
            phase=int(self.phase),
            phase_name=WALL_ROUTE_PHASE_NAMES[int(self.phase)],
            phase_advanced=self.phase != phase_before,
            side_gate_reached=side_gate_reached,
            crossed_fraction=crossed_fraction,
            target_distance=target_distance,
            goal_hold_count=int(self.goal_hold_count),
            success=self.phase == WallRoutePhase.TARGET_REACHED,
        )


def validate_wall_route_episode(arrays: Mapping[str, np.ndarray]) -> int:
    """Validate a saved, transition-aligned candidate episode."""

    required = {
        "physical_states",
        "actions",
        "node_positions",
        "node_velocities",
        "element_directors",
        "element_omegas",
        "phase_ids",
        "wall_clearances",
        "ground_clearances",
        "target_distances",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"wall-route episode is missing fields: {missing}")
    values = {key: np.asarray(value) for key, value in arrays.items()}
    actions = values["actions"]
    states = values["physical_states"]
    transition_count = int(actions.shape[0])
    if actions.ndim != 2 or actions.shape[1] != 18 or transition_count < 1:
        raise ValueError("actions must have nonempty shape [T,18]")
    if states.ndim != 2 or states.shape != (transition_count + 1, 45):
        raise ValueError("physical_states must have shape [T+1,45]")
    state_keys = (
        "node_positions",
        "node_velocities",
        "phase_ids",
        "wall_clearances",
        "ground_clearances",
        "target_distances",
    )
    if any(values[key].shape[0] != transition_count + 1 for key in state_keys):
        raise ValueError("state diagnostics must all have a T+1 leading axis")
    nodes = values["node_positions"]
    if nodes.ndim != 3 or nodes.shape[2] != 3 or nodes.shape[1] < 2:
        raise ValueError("node_positions must have shape [T+1,N,3]")
    if values["node_velocities"].shape != nodes.shape:
        raise ValueError("node_velocities must match node_positions")
    element_count = nodes.shape[1] - 1
    if values["element_directors"].shape != (
        transition_count + 1,
        element_count,
        3,
        3,
    ):
        raise ValueError("element_directors have an unexpected shape")
    if values["element_omegas"].shape != (
        transition_count + 1,
        element_count,
        3,
    ):
        raise ValueError("element_omegas have an unexpected shape")
    phase_ids = values["phase_ids"].reshape(-1)
    if (
        phase_ids.shape != (transition_count + 1,)
        or np.any(phase_ids < int(WallRoutePhase.UPRIGHT))
        or np.any(phase_ids > int(WallRoutePhase.TARGET_REACHED))
        or np.any(np.diff(phase_ids) < 0)
    ):
        raise ValueError("phase_ids must be a monotone valid phase sequence")
    numeric = (
        states,
        actions,
        nodes,
        values["node_velocities"],
        values["element_directors"],
        values["element_omegas"],
        values["wall_clearances"],
        values["ground_clearances"],
        values["target_distances"],
    )
    if not all(np.isfinite(value).all() for value in numeric):
        raise ValueError("wall-route episode contains NaN or Inf")
    if "rod_internal_states" in values:
        internal = values["rod_internal_states"]
        if internal.ndim != 2 or internal.shape[0] != transition_count + 1:
            raise ValueError("rod_internal_states must have shape [T+1,C]")
        if not np.isfinite(internal).all():
            raise ValueError("rod_internal_states contain NaN or Inf")
    return transition_count
