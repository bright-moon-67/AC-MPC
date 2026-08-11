import json

import numpy as np
import pytest

from antmaze_ac.envs.manisoft_tracking_env import (
    MANISOFT_WAYPOINT_ACTION_FILES,
    MANISOFT_WAYPOINT_REFERENCE_FILES,
    ManiSoftThreeWaypointTrackingEnv,
    ManiSoftTipTrackingEnv,
    load_manisoft_waypoint_references,
)


def test_waypoint_reference_loader_orders_and_cross_checks_actions(tmp_path):
    expected_states = []
    expected_actions = []
    for index, (reference_name, action_name) in enumerate(
        zip(MANISOFT_WAYPOINT_REFERENCE_FILES, MANISOFT_WAYPOINT_ACTION_FILES)
    ):
        state = np.full(45, index + 1, dtype=np.float32)
        action = np.linspace(-0.1, 0.1, 18, dtype=np.float32) * (index + 1)
        reference_path = tmp_path / reference_name
        action_path = tmp_path / action_name
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        action_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(reference_path, reference_state=state, reference_action=action)
        action_path.write_text(
            json.dumps({"u": action.tolist()}), encoding="utf-8"
        )
        expected_states.append(state)
        expected_actions.append(action)

    states, actions, reference_paths, action_paths = (
        load_manisoft_waypoint_references(tmp_path)
    )
    np.testing.assert_allclose(states, expected_states)
    np.testing.assert_allclose(actions, expected_actions)
    assert [path.relative_to(tmp_path).as_posix() for path in reference_paths] == list(
        MANISOFT_WAYPOINT_REFERENCE_FILES
    )
    assert [path.relative_to(tmp_path).as_posix() for path in action_paths] == list(
        MANISOFT_WAYPOINT_ACTION_FILES
    )

    action_paths[1].write_text(json.dumps({"u": [0.0] * 18}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_manisoft_waypoint_references(tmp_path)


def test_intermediate_waypoints_advance_without_terminating(monkeypatch):
    waypoints = np.asarray(
        [[0.04, 0.0, 1.0], [0.08, 0.0, 1.0], [0.12, 0.0, 1.0]],
        dtype=np.float32,
    )
    observation = np.zeros(45, dtype=np.float32)
    observation[30:33] = waypoints[0]

    def reached_step(self, action):
        return observation.copy(), 5.5, True, False, {
            "distance": 0.0,
            "target_tip": self.target_tip.copy(),
            "is_success": True,
        }

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "step", reached_step)
    env = ManiSoftThreeWaypointTrackingEnv.__new__(
        ManiSoftThreeWaypointTrackingEnv
    )
    env.fixed_waypoints = waypoints
    env.waypoint_event_reward = 1.0
    env.active_waypoint_index = 0
    env.waypoints_completed = 0
    env.success_count = 1
    env.target_tip = waypoints[0].copy()
    env.previous_distance = 0.0
    env.target_scale = 1.0
    env.step_count = 1
    env.episode_steps = 10

    _, reward, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and not truncated
    assert reward == pytest.approx(1.5)
    assert info["waypoint_passed"]
    assert info["active_waypoint_index"] == 1
    assert info["waypoints_completed"] == 1

    env.active_waypoint_index = 2
    env.waypoints_completed = 2
    env.target_tip = waypoints[2].copy()
    _, _, terminated, _, info = env.step(np.zeros(18))
    assert terminated
    assert not info["waypoint_passed"]
    assert info["waypoints_completed"] == 3
    assert info["is_success"]
