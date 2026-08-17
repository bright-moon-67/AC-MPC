from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("dm_control")

from experiments.dmc.tasks.adapter import DMCAdapter, make_dmc_adapter
from experiments.dmc.tasks.registry import (
    DMC_NATIVE_PROTOCOL,
    TASK_SPECS,
    verify_task_spec,
)


@pytest.mark.parametrize("task_name", tuple(TASK_SPECS))
def test_live_native_task_specs_match_registry(task_name: str) -> None:
    measured = verify_task_spec(task_name)
    spec = TASK_SPECS[task_name]
    assert measured["control_timestep"] == pytest.approx(spec.native_control_dt)
    assert measured["step_limit"] == spec.native_step_limit
    assert not hasattr(spec, "benchmark_control_dt")


def test_reacher_rejects_nonintegral_control_timestep() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        make_dmc_adapter("reacher_hard", control_timestep=0.05)


def test_step_preserves_discount_and_requested_applied_actions() -> None:
    env = make_dmc_adapter("cartpole_swingup", seed=7)
    try:
        observation = env.reset(seed=11)
        assert observation.shape == (5,)
        next_observation, reward, done, info = env.step(np.asarray([3.0]))
        assert next_observation.shape == (5,)
        assert np.isfinite(reward)
        assert not done
        np.testing.assert_array_equal(info["requested_action"], [3.0])
        np.testing.assert_array_equal(info["applied_action"], [1.0])
        np.testing.assert_array_equal(env.get_last_requested_action(), [3.0])
        np.testing.assert_array_equal(env.get_last_applied_action(), [1.0])
        assert info["discount"] == env.get_task_discount()
        assert info["discount"] == pytest.approx(1.0)
        assert info["terminated"] is False
        assert info["truncated"] is False
        assert info["step_type"] == "MID"
    finally:
        env.close()


def test_reset_with_seed_is_reproducible() -> None:
    env = make_dmc_adapter("reacher_hard", seed=0)
    try:
        first = env.reset(seed=123).copy()
        env.step(np.asarray([0.2, -0.3]))
        second = env.reset(seed=123).copy()
        np.testing.assert_array_equal(first, second)
    finally:
        env.close()


def test_native_protocol_metadata_is_complete() -> None:
    env = make_dmc_adapter("hopper_hop", seed=0)
    try:
        metadata = env.protocol_metadata()
        assert metadata["protocol_name"] == DMC_NATIVE_PROTOCOL
        assert metadata["control_dt"] == pytest.approx(0.02)
        assert metadata["physics_dt"] == pytest.approx(0.005)
        assert metadata["step_limit"] == 1000
        assert metadata["time_limit"] == pytest.approx(20.0)
        assert metadata["dm_control_version"] != "unknown"
        assert metadata["mujoco_version"] != "unknown"
        assert env.metadata()["seed"] == 0
    finally:
        env.close()


def test_contact_diagnostics_use_contact_force_api() -> None:
    class FakeData:
        ncon = 2

        def __init__(self) -> None:
            self.calls: list[int] = []

        def contact_force(self, contact_id: int) -> np.ndarray:
            self.calls.append(contact_id)
            if contact_id == 0:
                return np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
            return np.asarray([[0.0, 0.0, 12.0], [0.0, 0.0, 0.0]])

    data = FakeData()
    adapter = DMCAdapter.__new__(DMCAdapter)
    adapter._env = SimpleNamespace(physics=SimpleNamespace(data=data))
    diagnostics = adapter.get_contact_diagnostics()
    assert data.calls == [0, 1]
    assert diagnostics == {
        "n_contacts": 2,
        "n_force_components": 6,
        "total_force": 17.0,
        "max_force": 12.0,
        "total_torque": 2.0,
        "max_torque": 2.0,
    }
