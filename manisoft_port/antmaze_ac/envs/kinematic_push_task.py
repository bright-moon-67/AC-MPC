from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


KINEMATIC_PUSH_TASK_CONTEXT_DIM = 33
KINEMATIC_PUSH_PHASE_NAMES = (
    "route_high_before",
    "route_high_after",
    "route_low_after",
    "contact",
    "push",
)


def _vector(
    value: Sequence[float] | np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def point_aabb_distance(
    point: Sequence[float] | np.ndarray,
    minimum: Sequence[float] | np.ndarray,
    maximum: Sequence[float] | np.ndarray,
) -> float:
    """Euclidean distance from a point to a closed axis-aligned box."""

    point = _vector(point, 3, "point")
    minimum = _vector(minimum, 3, "minimum")
    maximum = _vector(maximum, 3, "maximum")
    if np.any(maximum < minimum):
        raise ValueError("AABB maximum must not be smaller than minimum")
    displacement = np.maximum(np.maximum(minimum - point, point - maximum), 0.0)
    return float(np.linalg.norm(displacement))


def segment_intersects_aabb(
    start: Sequence[float] | np.ndarray,
    end: Sequence[float] | np.ndarray,
    minimum: Sequence[float] | np.ndarray,
    maximum: Sequence[float] | np.ndarray,
) -> bool:
    """Return whether a closed line segment intersects a closed AABB."""

    start = _vector(start, 3, "segment start")
    end = _vector(end, 3, "segment end")
    minimum = _vector(minimum, 3, "minimum")
    maximum = _vector(maximum, 3, "maximum")
    direction = end - start
    lower_time = 0.0
    upper_time = 1.0
    for axis in range(3):
        if abs(direction[axis]) <= np.finfo(np.float64).eps:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
            continue
        inverse = 1.0 / direction[axis]
        enter = (minimum[axis] - start[axis]) * inverse
        leave = (maximum[axis] - start[axis]) * inverse
        if enter > leave:
            enter, leave = leave, enter
        lower_time = max(lower_time, enter)
        upper_time = min(upper_time, leave)
        if lower_time > upper_time:
            return False
    return True


def segment_aabb_distance(
    start: Sequence[float] | np.ndarray,
    end: Sequence[float] | np.ndarray,
    minimum: Sequence[float] | np.ndarray,
    maximum: Sequence[float] | np.ndarray,
) -> float:
    """Exact Euclidean distance between a line segment and an AABB.

    The squared point-to-box distance is piecewise quadratic along a segment.
    Splitting at all slab crossings and minimizing each quadratic avoids the
    false-positive square corners produced by simply expanding every AABB axis.
    """

    start = _vector(start, 3, "segment start")
    end = _vector(end, 3, "segment end")
    minimum = _vector(minimum, 3, "minimum")
    maximum = _vector(maximum, 3, "maximum")
    direction = end - start
    breaks = [0.0, 1.0]
    for axis in range(3):
        if abs(direction[axis]) <= np.finfo(np.float64).eps:
            continue
        for boundary in (minimum[axis], maximum[axis]):
            crossing = float((boundary - start[axis]) / direction[axis])
            if 0.0 < crossing < 1.0:
                breaks.append(crossing)
    breaks = sorted(set(breaks))
    candidates = list(breaks)
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        middle = 0.5 * (lower + upper)
        coordinate = start + middle * direction
        alpha = np.zeros(3, dtype=np.float64)
        beta = np.zeros(3, dtype=np.float64)
        below = coordinate < minimum
        above = coordinate > maximum
        alpha[below] = minimum[below] - start[below]
        beta[below] = -direction[below]
        alpha[above] = start[above] - maximum[above]
        beta[above] = direction[above]
        denominator = float(np.dot(beta, beta))
        if denominator > np.finfo(np.float64).eps:
            optimum = -float(np.dot(alpha, beta)) / denominator
            candidates.append(float(np.clip(optimum, lower, upper)))
    return min(
        point_aabb_distance(start + time * direction, minimum, maximum)
        for time in candidates
    )


def polyline_intersects_aabb(
    points: Sequence[Sequence[float]] | np.ndarray,
    minimum: Sequence[float] | np.ndarray,
    maximum: Sequence[float] | np.ndarray,
    padding: float = 0.0,
) -> bool:
    """Conservative capsule-vs-box check for all soft-arm segments.

    Comparing exact segment-to-box distance with the arm radius models a swept
    capsule and avoids tunnelling between Elastica nodes.  Unlike axis-wise box
    expansion, it does not create square false-positive regions at corners.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("polyline points must have shape [N,3] with N >= 2")
    if not np.isfinite(points).all() or padding < 0:
        raise ValueError("polyline must be finite and padding non-negative")
    minimum = _vector(minimum, 3, "minimum")
    maximum = _vector(maximum, 3, "maximum")
    return any(
        segment_aabb_distance(start, end, minimum, maximum)
        <= float(padding) + np.finfo(np.float64).eps
        for start, end in zip(points[:-1], points[1:])
    )


def aabbs_overlap(
    first_minimum: np.ndarray,
    first_maximum: np.ndarray,
    second_minimum: np.ndarray,
    second_maximum: np.ndarray,
) -> bool:
    return bool(
        np.all(first_maximum >= second_minimum)
        and np.all(second_maximum >= first_minimum)
    )


@dataclass(frozen=True)
class KinematicPushConfig:
    control_dt: float
    table_surface_z: float
    table_x_bounds: np.ndarray
    table_y_bounds: np.ndarray
    obstacle_minimum: np.ndarray
    obstacle_maximum: np.ndarray
    target_initial_center: np.ndarray
    target_size: np.ndarray
    goal_center: np.ndarray
    goal_radius: float
    work_z_bounds: np.ndarray
    route_side_x: float
    route_before_y: float
    route_after_y: float
    route_high_z: float
    route_after_high_z: float
    route_side_low_z: float
    route_contact_z: float
    waypoint_tolerances: np.ndarray
    tip_radius: float
    contact_margin: float
    latch_contact: bool
    arm_radius: float
    collision_safety_margin: float
    collision_scope: str
    terminate_on_collision: bool
    success_streak: int
    route_progress_reward: float
    object_progress_reward: float
    step_reward: float
    action_reward: float
    contact_reward: float
    success_reward: float
    collision_reward: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KinematicPushConfig":
        if value.get("type") != "kinematic_push_around_obstacle":
            raise ValueError("Unsupported or missing kinematic push task type")
        table = value["table"]
        obstacle = value["obstacle"]
        target = value["target"]
        route = value["route"]
        contact = value["contact"]
        collision = value["collision"]
        reward = value["reward"]
        config = cls(
            control_dt=float(value["control_dt"]),
            table_surface_z=float(table["surface_z"]),
            table_x_bounds=_vector(table["x_bounds"], 2, "table.x_bounds"),
            table_y_bounds=_vector(table["y_bounds"], 2, "table.y_bounds"),
            obstacle_minimum=_vector(obstacle["minimum"], 3, "obstacle.minimum"),
            obstacle_maximum=_vector(obstacle["maximum"], 3, "obstacle.maximum"),
            target_initial_center=_vector(
                target["initial_center"], 3, "target.initial_center"
            ),
            target_size=_vector(target["size"], 3, "target.size"),
            goal_center=_vector(value["goal_center"], 3, "goal_center"),
            goal_radius=float(value["goal_radius"]),
            work_z_bounds=_vector(value["work_z_bounds"], 2, "work_z_bounds"),
            route_side_x=float(route["side_x"]),
            route_before_y=float(route["before_y"]),
            route_after_y=float(route["after_y"]),
            route_high_z=float(route["high_z"]),
            route_after_high_z=float(route["after_high_z"]),
            route_side_low_z=float(route["side_low_z"]),
            route_contact_z=float(route["contact_z"]),
            waypoint_tolerances=_vector(
                route["waypoint_tolerances"], 3, "route.waypoint_tolerances"
            ),
            tip_radius=float(contact["tip_radius"]),
            contact_margin=float(contact["margin"]),
            latch_contact=bool(contact.get("latch", True)),
            arm_radius=float(collision["arm_radius"]),
            collision_safety_margin=float(collision["safety_margin"]),
            collision_scope=str(collision.get("scope", "whole_arm")),
            terminate_on_collision=bool(collision.get("terminate", True)),
            success_streak=int(value["success_streak"]),
            route_progress_reward=float(reward["route_progress"]),
            object_progress_reward=float(reward["object_progress"]),
            step_reward=float(reward["step"]),
            action_reward=float(reward["action"]),
            contact_reward=float(reward["contact"]),
            success_reward=float(reward["success"]),
            collision_reward=float(reward["collision"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "control_dt": self.control_dt,
            "goal_radius": self.goal_radius,
            "route_side_x": self.route_side_x,
            "tip_radius": self.tip_radius,
            "arm_radius": self.arm_radius,
        }
        if any(number <= 0 for number in positive.values()):
            raise ValueError("Task distances, radii and control_dt must be positive")
        if self.contact_margin < 0 or self.collision_safety_margin < 0:
            raise ValueError("Task margins must be non-negative")
        if self.collision_scope not in {"tip", "whole_arm"}:
            raise ValueError("collision.scope must be 'tip' or 'whole_arm'")
        if self.success_streak < 1:
            raise ValueError("success_streak must be positive")
        if np.any(self.target_size <= 0):
            raise ValueError("target.size must be positive")
        if np.any(self.waypoint_tolerances <= 0):
            raise ValueError("route.waypoint_tolerances must be positive")
        if np.any(self.obstacle_maximum <= self.obstacle_minimum):
            raise ValueError("Obstacle AABB must have positive extents")
        if not self.table_x_bounds[0] < self.table_x_bounds[1]:
            raise ValueError("table.x_bounds must be increasing")
        if not self.table_y_bounds[0] < self.table_y_bounds[1]:
            raise ValueError("table.y_bounds must be increasing")
        if not self.work_z_bounds[0] < self.work_z_bounds[1]:
            raise ValueError("work_z_bounds must be increasing")


class KinematicPushTask:
    """State machine for force-free contact and kinematic cube transport."""

    phase_count = len(KINEMATIC_PUSH_PHASE_NAMES)
    task_context_dim = KINEMATIC_PUSH_TASK_CONTEXT_DIM

    def __init__(self, config: KinematicPushConfig):
        self.config = config
        self.target_center = config.target_initial_center.copy()
        self.goal_center = config.goal_center.copy()
        self.target_velocity = np.zeros(3, dtype=np.float64)
        self.contact_locked = False
        self.target_tip_offset = np.zeros(3, dtype=np.float64)
        self.phase = 0
        self.route_side = 1
        self.success_count = 0
        self.previous_active_distance = 0.0
        self.previous_goal_distance = 0.0

    @property
    def phase_name(self) -> str:
        return KINEMATIC_PUSH_PHASE_NAMES[self.phase]

    @property
    def target_half_size(self) -> np.ndarray:
        return self.config.target_size * 0.5

    @property
    def obstacle_center(self) -> np.ndarray:
        return (self.config.obstacle_minimum + self.config.obstacle_maximum) * 0.5

    @property
    def obstacle_half_size(self) -> np.ndarray:
        return (self.config.obstacle_maximum - self.config.obstacle_minimum) * 0.5

    def _approach_direction(self) -> np.ndarray:
        direction = self.goal_center[:2] - self.target_center[:2]
        norm = float(np.linalg.norm(direction))
        if norm <= np.finfo(np.float64).eps:
            return np.array([0.0, 1.0], dtype=np.float64)
        return direction / norm

    def _approach_tip_target(self) -> np.ndarray:
        direction = self._approach_direction()
        # Approach from behind the cube.  The 0.8 radius factor puts the tip
        # just inside the configured contact shell without visual overlap.
        projected_half = float(np.sum(np.abs(direction) * self.target_half_size[:2]))
        offset = projected_half + 0.8 * self.config.tip_radius
        result = self.target_center.copy()
        result[:2] -= offset * direction
        result[2] = self.config.route_contact_z
        return result

    def route_waypoints(self) -> np.ndarray:
        sign = float(self.route_side)
        return np.asarray(
            [
                [
                    sign * self.config.route_side_x,
                    self.config.route_before_y,
                    self.config.route_high_z,
                ],
                [
                    sign * self.config.route_side_x,
                    self.config.route_after_y,
                    self.config.route_after_high_z,
                ],
                [
                    sign * self.config.route_side_x,
                    self.config.route_after_y,
                    self.config.route_side_low_z,
                ],
                self._approach_tip_target(),
            ],
            dtype=np.float64,
        )

    @property
    def active_tip_target(self) -> np.ndarray:
        if self.phase < 4:
            return self.route_waypoints()[self.phase].copy()
        result = self.goal_center - self.target_tip_offset
        return result.astype(np.float64, copy=True)

    @property
    def middle_section_target(self) -> np.ndarray:
        """Safe reference for Koopman's node-14 position during routing."""

        sign = float(self.route_side)
        if self.phase == 0:
            return np.asarray(
                [
                    sign * 0.55 * self.config.route_side_x,
                    0.5 * self.config.route_before_y,
                    max(self.config.obstacle_maximum[2] + 0.06, 0.64),
                ],
                dtype=np.float64,
            )
        if self.phase == 1:
            return np.asarray(
                [
                    sign * 0.65 * self.config.route_side_x,
                    self.config.obstacle_minimum[1] - 0.04,
                    self.config.obstacle_maximum[2] + 0.05,
                ],
                dtype=np.float64,
            )
        if self.phase == 2:
            return np.asarray(
                [
                    sign * 0.65 * self.config.route_side_x,
                    self.config.obstacle_minimum[1] - 0.02,
                    self.config.table_surface_z + 0.08,
                ],
                dtype=np.float64,
            )
        # Once the tip is safely behind the obstacle, let the middle section
        # wrap toward the near-side corner so the tip can reach the centerline.
        return np.asarray(
            [
                sign
                * (
                    abs(self.config.obstacle_minimum[0])
                    + self.config.arm_radius
                    + 0.006
                ),
                self.config.obstacle_minimum[1] - 0.02,
                self.config.table_surface_z + 0.08,
            ],
            dtype=np.float64,
        )

    @property
    def goal_distance(self) -> float:
        return float(np.linalg.norm(self.target_center[:2] - self.goal_center[:2]))

    def reset(
        self,
        tip_position: Sequence[float] | np.ndarray,
        *,
        route_side: int | None = None,
        rng: np.random.Generator | None = None,
        target_center: Sequence[float] | np.ndarray | None = None,
        goal_center: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        tip = _vector(tip_position, 3, "tip_position")
        if target_center is not None:
            self.target_center = _vector(target_center, 3, "target_center").copy()
        else:
            self.target_center = self.config.target_initial_center.copy()
        self.goal_center = (
            self.config.goal_center.copy()
            if goal_center is None
            else _vector(goal_center, 3, "goal_center").copy()
        )
        if route_side is not None and route_side not in {-1, 1}:
            raise ValueError("route_side must be -1 or +1")
        if route_side is None:
            rng = np.random.default_rng() if rng is None else rng
            route_side = -1 if int(rng.integers(2)) == 0 else 1
        self.route_side = int(route_side)
        self.target_velocity.fill(0.0)
        self.contact_locked = False
        self.target_tip_offset.fill(0.0)
        self.phase = 0
        self.success_count = 0
        self.previous_active_distance = float(
            np.linalg.norm(tip - self.active_tip_target)
        )
        self.previous_goal_distance = self.goal_distance

    def _target_contact(self, tip: np.ndarray) -> bool:
        minimum = self.target_center - self.target_half_size
        maximum = self.target_center + self.target_half_size
        return point_aabb_distance(tip, minimum, maximum) <= (
            self.config.tip_radius + self.config.contact_margin
        )

    def _clamp_target_to_table(self, center: np.ndarray) -> np.ndarray:
        result = center.copy()
        half = self.target_half_size
        result[0] = np.clip(
            result[0],
            self.config.table_x_bounds[0] + half[0],
            self.config.table_x_bounds[1] - half[0],
        )
        result[1] = np.clip(
            result[1],
            self.config.table_y_bounds[0] + half[1],
            self.config.table_y_bounds[1] - half[1],
        )
        result[2] = self.config.table_surface_z + half[2]
        return result

    def _target_hits_obstacle(self) -> bool:
        return aabbs_overlap(
            self.target_center - self.target_half_size,
            self.target_center + self.target_half_size,
            self.config.obstacle_minimum,
            self.config.obstacle_maximum,
        )

    def update(
        self,
        tip_position: Sequence[float] | np.ndarray,
        arm_nodes: Sequence[Sequence[float]] | np.ndarray,
        normalized_action_energy: float,
    ) -> dict[str, Any]:
        tip = _vector(tip_position, 3, "tip_position")
        nodes = np.asarray(arm_nodes, dtype=np.float64)
        old_target = self.target_center.copy()
        old_phase = self.phase
        contact_event = False

        if self.phase < 3:
            distance = float(np.linalg.norm(tip - self.active_tip_target))
            if distance <= self.config.waypoint_tolerances[self.phase]:
                self.phase += 1

        if self.phase >= 3 and not self.contact_locked and self._target_contact(tip):
            self.contact_locked = True
            self.target_tip_offset = self.target_center - tip
            self.phase = 4
            contact_event = True

        if self.contact_locked:
            followed = tip + self.target_tip_offset
            self.target_center = self._clamp_target_to_table(followed)
            self.target_velocity = (
                self.target_center - old_target
            ) / self.config.control_dt
        else:
            self.target_velocity.fill(0.0)

        whole_arm_collision = polyline_intersects_aabb(
            nodes,
            self.config.obstacle_minimum,
            self.config.obstacle_maximum,
            padding=(
                self.config.arm_radius + self.config.collision_safety_margin
            ),
        )
        tip_collision = point_aabb_distance(
            tip,
            self.config.obstacle_minimum,
            self.config.obstacle_maximum,
        ) <= self.config.tip_radius + self.config.collision_safety_margin
        robot_collision = (
            tip_collision
            if self.config.collision_scope == "tip"
            else whole_arm_collision
        )
        target_collision = self._target_hits_obstacle()
        collision = bool(robot_collision or target_collision)

        active_distance = float(np.linalg.norm(tip - self.active_tip_target))
        goal_distance = self.goal_distance
        route_progress = self.previous_active_distance - active_distance
        object_progress = self.previous_goal_distance - goal_distance
        # Do not mix distances belonging to two different route stages.
        if self.phase != old_phase:
            route_progress = 0.0
        reward = (
            self.config.step_reward
            + self.config.action_reward * float(normalized_action_energy)
            + self.config.route_progress_reward * route_progress
        )
        if self.contact_locked:
            reward += self.config.object_progress_reward * object_progress
        if contact_event:
            reward += self.config.contact_reward

        if goal_distance <= self.config.goal_radius:
            self.success_count += 1
        else:
            self.success_count = 0
        success = self.success_count >= self.config.success_streak
        if success:
            reward += self.config.success_reward
        if collision:
            reward += self.config.collision_reward

        self.previous_active_distance = active_distance
        self.previous_goal_distance = goal_distance
        return {
            "reward": float(reward),
            "success": bool(success),
            "collision": collision,
            "robot_collision": bool(robot_collision),
            "tip_collision": bool(tip_collision),
            "whole_arm_collision": bool(whole_arm_collision),
            "target_collision": bool(target_collision),
            "contact_event": contact_event,
            "active_distance": active_distance,
            "goal_distance": goal_distance,
        }

    def context(self, tip_position: Sequence[float] | np.ndarray) -> np.ndarray:
        tip = _vector(tip_position, 3, "tip_position")
        phase = np.eye(self.phase_count, dtype=np.float64)[self.phase]
        side = np.asarray(
            [float(self.route_side < 0), float(self.route_side > 0)],
            dtype=np.float64,
        )
        clearance = point_aabb_distance(
            tip,
            self.config.obstacle_minimum,
            self.config.obstacle_maximum,
        )
        context = np.concatenate(
            (
                self.active_tip_target,
                self.target_center,
                self.target_velocity,
                self.goal_center,
                self.obstacle_center,
                self.obstacle_half_size,
                self.target_center - tip,
                self.goal_center - self.target_center,
                np.asarray([clearance, float(self.contact_locked)]),
                phase,
                side,
            )
        )
        if context.shape != (self.task_context_dim,):
            raise RuntimeError("Kinematic push task context layout is inconsistent")
        return context.astype(np.float32)
