from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.dmc.o2o.config import O2OConfig, SUPPORTED_O2O_TASKS
from experiments.dmc.o2o.formal_hopper import FORMAL_METHODS
from experiments.dmc.o2o.formal_hopper_hop import TASK
from experiments.dmc.o2o.train import (
    RECORDED_REWARD_O2O_TASKS,
    _dataset_action_repeat,
    _validate_dataset_environment_protocol,
)
from experiments.dmc.ppo.vector_env import ProcessDMCVectorEnv, SyncDMCVectorEnv
from experiments.dmc.tasks.adapter import make_dmc_adapter


def test_hopper_hop_is_an_explicit_recorded_reward_o2o_task() -> None:
    assert TASK == "hopper_hop"
    assert TASK in SUPPORTED_O2O_TASKS
    assert TASK in RECORDED_REWARD_O2O_TASKS


def test_hopper_hop_supports_the_formal_seven_method_matrix() -> None:
    for method in FORMAL_METHODS:
        config = O2OConfig(
            task=TASK,
            method=method,
            offline_updates=50_000,
            online_steps=20_000,
            kmpc_horizon=8,
            mpve_total_horizon=4,
        )
        config.validate()
        assert config.task == TASK
        expected_target_entropy = (
            -4.0
            if config.temperature_objective == "calql_log_alpha"
            else -2.0
        )
        assert config.target_entropy == expected_target_entropy


def test_hopper_hop_structured_methods_require_koopman() -> None:
    assert O2OConfig(task=TASK, method="Cal-RLPD-KMPC").requires_koopman
    assert O2OConfig(task=TASK, method="Cal-RLPD-Lift").requires_koopman
    assert not O2OConfig(task=TASK, method="Cal-RLPD").requires_koopman


def test_hopper_hop_o2o_pins_dataset_and_environment_to_ar2() -> None:
    dataset = SimpleNamespace(
        metadata={
            "action_repeat": 2,
            "control_dt": 0.04,
            "transitions_per_episode": 500,
        }
    )
    assert _dataset_action_repeat(TASK, dataset) == 2
    protocol = {
        "protocol_name": "tdmpc2_action_repeat2_v1",
        "action_repeat": 2,
        "control_dt": 0.04,
        "step_limit": 500,
    }
    _validate_dataset_environment_protocol(TASK, dataset, protocol)

    native_protocol = {
        "protocol_name": "dmc_native_v1",
        "control_dt": 0.02,
        "step_limit": 1000,
    }
    with pytest.raises(ValueError, match="dataset/environment protocol mismatch"):
        _validate_dataset_environment_protocol(TASK, dataset, native_protocol)


def test_hopper_hop_explicit_ar2_matches_two_native_steps() -> None:
    pytest.importorskip("dm_control")
    seed = 20260851
    action = np.asarray([0.2, -0.3, 0.4, -0.5], dtype=np.float32)
    native = make_dmc_adapter(TASK, seed=seed, action_repeat=1)
    ar2 = make_dmc_adapter(TASK, seed=seed, action_repeat=2)
    try:
        np.testing.assert_array_equal(native.reset(seed=seed), ar2.reset(seed=seed))
        _first_state, first_reward, _done, _info = native.step(action)
        second_state, second_reward, _done, _info = native.step(action)
        ar2_state, ar2_reward, _done, _info = ar2.step(action)
        np.testing.assert_array_equal(second_state, ar2_state)
        assert ar2_reward == pytest.approx(first_reward + second_reward, abs=0.0)
        assert ar2.protocol_metadata()["control_dt"] == 0.04
        assert ar2.protocol_metadata()["step_limit"] == 500
    finally:
        native.close()
        ar2.close()


def test_hopper_hop_process_vector_runner_preserves_explicit_ar2() -> None:
    pytest.importorskip("dm_control")
    seed = 20260851
    sync = SyncDMCVectorEnv(TASK, 2, seed, action_repeat=2)
    parallel = ProcessDMCVectorEnv(
        TASK, 2, seed, workers=2, action_repeat=2
    )
    try:
        np.testing.assert_array_equal(sync.reset(), parallel.reset())
        actions = np.asarray(
            [[0.2, -0.3, 0.4, -0.5], [-0.4, 0.1, 0.3, -0.2]],
            dtype=np.float32,
        )
        expected = sync.step(actions)
        actual = parallel.step(actions)
        np.testing.assert_array_equal(actual.observation, expected.observation)
        np.testing.assert_array_equal(actual.reward, expected.reward)
        assert parallel.protocol["protocol_name"] == "tdmpc2_action_repeat2_v1"
        assert parallel.protocol["control_dt"] == 0.04
        assert parallel.protocol["step_limit"] == 500
    finally:
        parallel.close()
        sync.close()
