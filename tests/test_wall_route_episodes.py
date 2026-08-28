from __future__ import annotations

import numpy as np
import pytest

from antmaze_ac.data.wall_route_episodes import (
    WallRouteGeometry,
    WallRoutePhase,
    WallRoutePhaseTracker,
    generate_smooth_wall_route_actions,
    validate_wall_route_episode,
)


def _geometry(goal_hold_steps: int = 2) -> WallRouteGeometry:
    return WallRouteGeometry.from_dict(
        {
            "base": [0.0, 0.0, 0.0],
            "target": [0.0, 0.70, 0.05],
            "wall": {
                "minimum": [-0.10, 0.40, 0.0],
                "maximum": [0.10, 0.42, 1.10],
                "arm_radius": 0.05,
                "safety_margin": 0.01,
            },
            "ground": {
                "surface_z": 0.0,
                "safety_margin": 0.0,
                "violation_tolerance": 0.0005,
                "mounting_exempt_nodes": 1,
            },
            "route": {
                "prewall_y_margin": 0.02,
                "postwall_y_margin": 0.02,
                "crossed_node_fraction": 0.30,
                "return_x_tolerance": 0.04,
                "goal_tolerance": 0.015,
                "goal_speed_tolerance": 0.03,
                "goal_hold_steps": goal_hold_steps,
            },
        }
    )


def test_ground_clearance_exempts_only_the_mounting_node() -> None:
    geometry = _geometry()
    upright = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.0, 0.0, 0.10]]
    )
    assert geometry.whole_arm_ground_clearance(upright) == pytest.approx(0.0)

    below_ground = upright.copy()
    below_ground[1, 2] = 0.04
    assert geometry.whole_arm_ground_clearance(below_ground) == pytest.approx(-0.01)


def test_wall_clearance_uses_whole_polyline_and_arm_radius() -> None:
    geometry = _geometry()
    direct = np.asarray([[0.0, 0.0, 0.20], [0.0, 0.70, 0.20]])
    safe_right = np.asarray(
        [
            [0.0, 0.0, 0.20],
            [0.17, 0.35, 0.20],
            [0.17, 0.47, 0.20],
            [0.0, 0.70, 0.20],
        ]
    )
    assert geometry.whole_arm_wall_clearance(direct) < 0.0
    assert geometry.whole_arm_wall_clearance(safe_right) > 0.0


def test_phase_tracker_requires_ordered_side_cross_return_and_hold() -> None:
    geometry = _geometry(goal_hold_steps=2)
    tracker = WallRoutePhaseTracker(geometry, side=1)
    zero_velocity = np.zeros(3)

    side_nodes = np.asarray(
        [[0.0, 0.0, 0.05], [0.08, 0.15, 0.20], [0.17, 0.30, 0.30]]
    )
    assert tracker.update(side_nodes, zero_velocity).phase == WallRoutePhase.SIDE_GATE

    tip_beyond = side_nodes.copy()
    tip_beyond[-1] = [0.17, 0.50, 0.25]
    assert (
        tracker.update(tip_beyond, zero_velocity).phase
        == WallRoutePhase.TIP_BEYOND_WALL
    )

    body_beyond = np.asarray(
        [
            [0.0, 0.0, 0.05],
            [0.10, 0.20, 0.15],
            [0.17, 0.35, 0.20],
            [0.17, 0.46, 0.15],
            [0.17, 0.55, 0.10],
        ]
    )
    assert (
        tracker.update(body_beyond, zero_velocity).phase
        == WallRoutePhase.DISTAL_BODY_BEYOND_WALL
    )

    returned = body_beyond.copy()
    returned[-1] = [0.02, 0.60, 0.08]
    assert (
        tracker.update(returned, zero_velocity).phase
        == WallRoutePhase.RETURNED_TO_TARGET_PLANE
    )

    at_goal = returned.copy()
    at_goal[-1] = geometry.target
    first = tracker.update(at_goal, zero_velocity)
    second = tracker.update(at_goal, zero_velocity)
    assert not first.success
    assert second.success
    assert second.phase == WallRoutePhase.TARGET_REACHED


def test_smooth_candidate_actions_obey_magnitude_and_rate_limits() -> None:
    actions = generate_smooth_wall_route_actions(
        np.random.default_rng(7),
        400,
        0.02,
        knot_seconds_range=(0.5, 1.0),
        hold_seconds_range=(0.0, 0.2),
        peak_range=(0.12, 0.30),
        spatial_modes=4,
        absolute_action_limit=0.30,
        maximum_action_delta=0.005,
    )
    previous = np.vstack((np.zeros((1, 18), dtype=np.float32), actions[:-1]))
    assert actions.shape == (400, 18)
    assert np.max(np.abs(actions)) <= 0.30 + 1e-7
    assert np.max(np.abs(actions - previous)) <= 0.005 + 1e-7
    assert np.max(np.abs(actions)) >= 0.10


def test_wall_route_episode_schema_is_transition_aligned() -> None:
    arrays = {
        "physical_states": np.zeros((3, 45), dtype=np.float32),
        "actions": np.zeros((2, 18), dtype=np.float32),
        "node_positions": np.zeros((3, 4, 3), dtype=np.float64),
        "node_velocities": np.zeros((3, 4, 3), dtype=np.float64),
        "element_directors": np.zeros((3, 3, 3, 3), dtype=np.float64),
        "element_omegas": np.zeros((3, 3, 3), dtype=np.float64),
        "phase_ids": np.asarray([0, 1, 1], dtype=np.int8),
        "wall_clearances": np.ones(3, dtype=np.float32),
        "ground_clearances": np.zeros(3, dtype=np.float32),
        "target_distances": np.ones(3, dtype=np.float32),
    }
    assert validate_wall_route_episode(arrays) == 2

    arrays["phase_ids"] = np.asarray([0, 2, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="monotone"):
        validate_wall_route_episode(arrays)
