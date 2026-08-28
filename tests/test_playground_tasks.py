from __future__ import annotations

import pytest

from experiments.playground.tasks import TASKS, PlaygroundTask


def test_six_task_contract() -> None:
    assert tuple(TASKS) == (
        "CartpoleSwingup",
        "ReacherHard",
        "HopperHop",
        "HopperStand",
        "WalkerRun",
        "HumanoidRun",
    )
    assert [(task.observation_dim, task.action_dim) for task in TASKS.values()] == [
        (5, 1),
        (6, 2),
        (15, 4),
        (15, 4),
        (24, 6),
        (67, 21),
    ]


def test_substeps_are_integral() -> None:
    assert [task.substeps for task in TASKS.values()] == [1, 4, 4, 8, 10, 5]


def test_rejects_fractional_substeps() -> None:
    task = PlaygroundTask("bad", 1, 1, 0.01, 0.003)
    with pytest.raises(ValueError, match="non-integral"):
        _ = task.substeps


def test_task_scaled_model_and_controller_horizons() -> None:
    expected = {
        "CartpoleSwingup": (10, 50, 20, 10),
        "ReacherHard": (10, 25, 10, 5),
        "HopperHop": (24, 25, 10, 5),
        "HopperStand": (48, 20, 8, 4),
        "WalkerRun": (32, 20, 8, 4),
        "HumanoidRun": (96, 20, 8, 4),
    }
    for name, task in TASKS.items():
        assert (
            task.koopman_lift_dim,
            task.koopman_horizon_steps,
            task.kmpc_horizon_steps,
            task.mpve_horizon_steps,
        ) == expected[name]
        maximum_lift_ratio = 3.5 if name == "HopperStand" else 2.0
        assert (
            1.0
            <= task.koopman_lift_dim / task.observation_dim
            <= maximum_lift_ratio
        )
        assert 0.5 <= task.koopman_horizon_steps * task.control_timestep <= 1.0
        expected_kmpc_duration = 0.32 if name == "HopperStand" else 0.2
        assert task.kmpc_horizon_steps * task.control_timestep == pytest.approx(
            expected_kmpc_duration
        )
        assert 0.1 <= task.mpve_horizon_steps * task.control_timestep <= 0.2
