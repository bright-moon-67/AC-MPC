import json
import hashlib
import inspect

import numpy as np
import pytest

from antmaze_ac.envs.manisoft_tracking_env import (
    MANISOFT_WAYPOINT_SUCCESS_STREAK,
    MANISOFT_WAYPOINT_SUCCESS_THRESHOLD,
    MANISOFT_WAYPOINT_ACTION_FILES,
    MANISOFT_WAYPOINT_REFERENCE_FILES,
    ManiSoftThreeWaypointTrackingEnv,
    ManiSoftTipTrackingEnv,
    load_manisoft_waypoint_reference_bank,
    load_manisoft_waypoint_references,
)


def test_waypoint_defaults_are_5mm_and_immediate():
    parameters = inspect.signature(
        ManiSoftThreeWaypointTrackingEnv.__init__
    ).parameters
    assert parameters["success_threshold"].default == pytest.approx(0.005)
    assert parameters["success_streak"].default == 1
    assert MANISOFT_WAYPOINT_SUCCESS_THRESHOLD == pytest.approx(0.005)
    assert MANISOFT_WAYPOINT_SUCCESS_STREAK == 1


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_certified_waypoint_bank_loader(tmp_path):
    triplets = []
    expected_states = []
    expected_actions = []
    for triplet_index in range(2):
        state_rows = []
        action_rows = []
        waypoint_rows = []
        for waypoint_index in range(3):
            state = np.zeros(45, dtype=np.float32)
            state[30:33] = (triplet_index + 1, waypoint_index + 1, 0.5)
            action = np.full(18, 0.01 * (waypoint_index + 1), dtype=np.float32)
            relative = f"triplet_{triplet_index:04d}/waypoint_{waypoint_index + 1}.npz"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                reference_state=state,
                reference_action=action,
                reference_tip_position=state[30:33],
            )
            waypoint_rows.append(
                {"index": waypoint_index, "reference": relative, "sha256": _sha256(path)}
            )
            state_rows.append(state)
            action_rows.append(action)
        triplets.append(
            {"index": triplet_index, "waypoints": waypoint_rows}
        )
        expected_states.append(state_rows)
        expected_actions.append(action_rows)
    manifest = {
        "schema_version": 1,
        "kind": "manisoft_certified_three_waypoint_reference_bank",
        "scenario_sha256": "scenario-hash",
        "triplet_count": 2,
        "waypoint_count": 3,
        "triplets": triplets,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bank = load_manisoft_waypoint_reference_bank(tmp_path)
    np.testing.assert_allclose(bank.states, expected_states)
    np.testing.assert_allclose(bank.actions, expected_actions)
    assert bank.triplet_count == 2
    assert bank.scenario_sha256 == "scenario-hash"

    first_reference = tmp_path / triplets[0]["waypoints"][0]["reference"]
    with first_reference.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manisoft_waypoint_reference_bank(tmp_path)


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
    env.active_waypoint_triplet_index = 0
    env.waypoint_event_reward = 3.0
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
    assert reward == pytest.approx(3.5)
    assert info["waypoint_passed"]
    assert info["active_waypoint_index"] == 1
    assert info["waypoints_completed"] == 1

    env.active_waypoint_index = 2
    env.waypoints_completed = 2
    env.target_tip = waypoints[2].copy()
    _, reward, terminated, _, info = env.step(np.zeros(18))
    assert terminated
    assert not info["waypoint_passed"]
    assert info["waypoints_completed"] == 3
    assert info["is_success"]
    assert reward == pytest.approx(8.5)


def test_waypoint_triplet_sampling_is_seeded_and_can_be_forced(monkeypatch):
    bank = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    observation = np.zeros(45, dtype=np.float32)

    def fake_reset(self, *, seed=None, options=None):
        self.np_random = np.random.default_rng(seed)
        return observation.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftThreeWaypointTrackingEnv.__new__(
        ManiSoftThreeWaypointTrackingEnv
    )
    env.waypoint_tip_bank = bank

    _, first = env.reset(seed=123)
    _, second = env.reset(seed=123)
    assert first["waypoint_triplet_index"] == second["waypoint_triplet_index"]
    np.testing.assert_array_equal(first["waypoints"], second["waypoints"])

    _, forced = env.reset(options={"waypoint_triplet_index": 3})
    assert forced["waypoint_triplet_index"] == 3
    np.testing.assert_array_equal(forced["waypoints"], bank[3])
