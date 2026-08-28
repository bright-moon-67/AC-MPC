from __future__ import annotations

import numpy as np
import pytest

from experiments.playground.collect_koopman import (
    _allocate_behavior_modes,
    _behavior_fractions,
)
from experiments.playground.tasks import TASKS


def test_task_score_horizon_is_compatible_with_dmc_return_scale() -> None:
    assert {
        name: task.episode_steps for name, task in TASKS.items()
    } == {
        "CartpoleSwingup": 1000,
        "ReacherHard": 1000,
        "HopperHop": 1000,
        "HopperStand": 500,
        "WalkerRun": 1000,
        "HumanoidRun": 1000,
    }


def test_task_action_and_observation_dimensions_are_positive() -> None:
    assert all(
        task.observation_dim > 0 and task.action_dim > 0
        for task in TASKS.values()
    )


def test_global_coverage_behavior_allocation_is_exact_and_reproducible() -> None:
    fractions = _behavior_fractions(0.4, 0.3, 0.3)
    first = _allocate_behavior_modes(1000, fractions, seed=17)
    second = _allocate_behavior_modes(1000, fractions, seed=17)
    assert np.array_equal(first, second)
    assert np.bincount(first, minlength=3).tolist() == [400, 300, 300]


@pytest.mark.parametrize(
    "fractions",
    ((0.4, 0.3, 0.2), (-0.1, 0.6, 0.5), (float("nan"), 0.5, 0.5)),
)
def test_global_coverage_behavior_fractions_fail_closed(fractions) -> None:
    with pytest.raises(ValueError):
        _behavior_fractions(*fractions)
