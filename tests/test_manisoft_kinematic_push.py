from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from antmaze_ac.envs.kinematic_push_task import (
    KinematicPushConfig,
    KinematicPushTask,
    point_aabb_distance,
    polyline_intersects_aabb,
    segment_aabb_distance,
)


def _config(success_streak: int = 1) -> KinematicPushConfig:
    return KinematicPushConfig.from_dict(
        {
            "type": "kinematic_push_around_obstacle",
            "control_dt": 0.02,
            "table": {
                "surface_z": 0.39,
                "x_bounds": [-0.60, 0.60],
                "y_bounds": [0.20, 0.80],
            },
            "obstacle": {
                "minimum": [-0.15, 0.38, 0.39],
                "maximum": [0.15, 0.54, 0.59],
            },
            "target": {
                "initial_center": [0.0, 0.64, 0.4225],
                "size": [0.065, 0.065, 0.065],
            },
            "goal_center": [0.0, 0.74, 0.4225],
            "goal_radius": 0.045,
            "work_z_bounds": [0.415, 0.58],
            "route": {
                "side_x": 0.54,
                "before_y": 0.16,
                "after_y": 0.59,
                "high_z": 0.82,
                "after_high_z": 0.57,
                "side_low_z": 0.43,
                "contact_z": 0.4225,
                "waypoint_tolerances": [0.060, 0.100, 0.080],
            },
            "contact": {"tip_radius": 0.024, "margin": 0.01, "latch": True},
            "collision": {
                "scope": "tip",
                "arm_radius": 0.024,
                "safety_margin": 0.0,
                "terminate": True,
            },
            "success_streak": success_streak,
            "reward": {
                "route_progress": 2.0,
                "object_progress": 8.0,
                "step": -0.01,
                "action": -0.001,
                "contact": 3.0,
                "success": 20.0,
                "collision": -20.0,
            },
        }
    )


def _safe_nodes(tip: np.ndarray) -> np.ndarray:
    return np.stack((tip + np.array([0.0, 0.0, 0.2]), tip))


def test_segment_collision_checks_between_nodes() -> None:
    minimum = np.array([-0.2, 0.38, 0.39])
    maximum = np.array([0.2, 0.54, 0.79])
    assert polyline_intersects_aabb(
        [[-0.4, 0.46, 0.5], [0.4, 0.46, 0.5]], minimum, maximum
    )
    assert not polyline_intersects_aabb(
        [[-0.4, 0.8, 0.5], [0.4, 0.8, 0.5]], minimum, maximum
    )
    assert np.isclose(
        point_aabb_distance([0.0, 0.6, 0.5], minimum, maximum), 0.06
    )
    # Axis-wise AABB expansion would incorrectly collide with this diagonal
    # corner point, whose true Euclidean clearance is sqrt(2)*0.029 > 0.03.
    corner = np.array([-0.229, 0.351, 0.48])
    assert segment_aabb_distance(corner, corner, minimum, maximum) > 0.03
    assert not polyline_intersects_aabb(
        [corner, corner + np.array([-0.01, -0.01, 0.0])],
        minimum,
        maximum,
        padding=0.03,
    )


def test_route_contact_follow_and_success() -> None:
    task = KinematicPushTask(_config())
    task.reset([0.0, 0.0, 1.0], route_side=1)
    assert task.context([0.0, 0.0, 1.0]).shape == (33,)
    for expected_phase in (1, 2, 3):
        tip = task.active_tip_target
        result = task.update(tip, _safe_nodes(tip), 0.0)
        assert not result["collision"]
        assert task.phase == expected_phase
    contact_tip = task.active_tip_target
    result = task.update(contact_tip, _safe_nodes(contact_tip), 0.0)
    assert result["contact_event"]
    assert task.contact_locked and task.phase == 4
    push_tip = task.active_tip_target
    result = task.update(push_tip, _safe_nodes(push_tip), 0.0)
    assert result["success"]
    np.testing.assert_allclose(task.target_center[:2], task.goal_center[:2])


def test_revised_project_scene_fits_table_and_routes_around_obstacle() -> None:
    scenario_path = Path(
        "/root/autodl-tmp/ManiSoft/configs/push_around_obstacle_kinematic.yaml"
    )
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    config = KinematicPushConfig.from_dict(payload["task"])
    np.testing.assert_allclose(
        config.obstacle_maximum - config.obstacle_minimum,
        [0.10, 0.15, 0.05],
    )
    assert config.obstacle_minimum[2] == config.table_surface_z
    target_half = config.target_size * 0.5
    for center in (config.target_initial_center, config.goal_center):
        assert config.table_x_bounds[0] <= center[0] - target_half[0]
        assert center[0] + target_half[0] <= config.table_x_bounds[1]
        assert config.table_y_bounds[0] <= center[1] - target_half[1]
        assert center[1] + target_half[1] <= config.table_y_bounds[1]
        assert np.isclose(center[2] - target_half[2], config.table_surface_z)
    for side in (-1, 1):
        task = KinematicPushTask(config)
        task.reset([0.0, 0.0, 1.0], route_side=side)
        route = np.vstack(([0.0, 0.0, 1.0], task.route_waypoints()))
        assert not polyline_intersects_aabb(
            route,
            config.obstacle_minimum,
            config.obstacle_maximum,
            padding=config.tip_radius,
        )
