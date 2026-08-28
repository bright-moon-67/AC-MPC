from __future__ import annotations

import numpy as np

from antmaze_ac.data.wall_route_episodes import WallRouteGeometry
from antmaze_ac.envs.manisoft_wall_crossing_sac_env import wall_crossing_metrics


def _geometry() -> WallRouteGeometry:
    return WallRouteGeometry.from_dict(
        {
            "base": [0.0, 0.0, 0.0],
            "target": [0.0, 0.65, 0.05],
            "wall": {
                "minimum": [-0.10, 0.32, 0.0],
                "maximum": [0.10, 0.34, 1.10],
                "arm_radius": 0.05,
                "safety_margin": 0.01,
            },
            "ground": {
                "surface_z": 0.0,
                "violation_tolerance": 0.0005,
                "mounting_exempt_nodes": 1,
            },
            "route": {
                "crossed_node_fraction": 0.30,
                "return_x_tolerance": 0.04,
                "goal_tolerance": 0.015,
                "goal_speed_tolerance": 0.03,
                "goal_hold_steps": 5,
                "postwall_y_margin": 0.02,
            },
        }
    )


def test_distal_fraction_only_counts_contiguous_tip_suffix() -> None:
    geometry = _geometry()
    nodes = np.zeros((21, 3), dtype=np.float64)
    nodes[:, 0] = -0.20
    nodes[:, 2] = np.linspace(0.05, 0.65, 21)
    nodes[:, 1] = 0.20
    nodes[5, 1] = 0.40  # Disconnected middle node does not count as distal.
    nodes[-3:, 1] = 0.40
    metrics = wall_crossing_metrics(
        geometry, nodes, np.zeros_like(nodes), side=-1
    )
    assert np.isclose(metrics.distal_crossed_fraction, 3 / 20)
    assert np.isclose(metrics.crossed_fraction, 4 / 20)


def test_tip_before_far_face_has_zero_distal_fraction() -> None:
    geometry = _geometry()
    nodes = np.zeros((21, 3), dtype=np.float64)
    nodes[:, 0] = 0.20
    nodes[:, 2] = np.linspace(0.05, 0.65, 21)
    nodes[:, 1] = 0.20
    nodes[-2, 1] = 0.40
    metrics = wall_crossing_metrics(
        geometry, nodes, np.zeros_like(nodes), side=1
    )
    assert metrics.distal_crossed_fraction == 0.0
    assert np.isclose(metrics.crossed_fraction, 1 / 20)


def test_threading_score_increases_continuously_before_node_crossing() -> None:
    geometry = _geometry()
    nodes = np.zeros((21, 3), dtype=np.float64)
    nodes[:, 0] = 0.20
    nodes[:, 2] = np.linspace(0.05, 0.65, 21)
    nodes[:, 1] = 0.321
    first = wall_crossing_metrics(
        geometry, nodes, np.zeros_like(nodes), side=1
    )
    nodes[:, 1] += 0.005
    second = wall_crossing_metrics(
        geometry, nodes, np.zeros_like(nodes), side=1
    )
    assert first.crossed_fraction == second.crossed_fraction == 0.0
    assert second.threading_score > first.threading_score
