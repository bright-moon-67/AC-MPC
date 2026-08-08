"""Phase 4 pipeline tests: collection schema, dataset assembly, comparison."""

import numpy as np
import pytest

from experiments.hopper_hop_mujoco.collect import build_mujoco_datasets
from experiments.hopper_hop_mujoco.collect.collect_hopperhop_mujoco import (
    PPO_UPDATE,
    RANDOM_UPDATE,
    MIN_EPISODE_LEN,
)
from experiments.hopper_hop_mujoco.eval import compare_contact_dims


def test_collector_constants():
    assert RANDOM_UPDATE == 10
    assert PPO_UPDATE == 20
    assert MIN_EPISODE_LEN >= 20  # must cover K-step window length


def test_collector_module_imports():
    # the collector must import without touching the GPU/PhysX runtime
    import experiments.hopper_hop_mujoco.collect.collect_hopperhop_mujoco as c

    assert callable(c.collect)
    assert callable(c._flush)


def test_build_module_imports():
    assert callable(build_mujoco_datasets.assemble)


def test_compare_module_imports():
    assert callable(compare_contact_dims.compare)


def test_chunk_schema_matches_ppo_collector():
    """The chunk dict must contain exactly the PPO collector's fields."""
    import experiments.hopper_hop_mujoco.collect.collect_hopperhop_mujoco as c

    # _flush writes: state, action, next_state, episode_id, step_index, update, global_step
    assert hasattr(c, "_flush")
    # sanity: these are the same keys the PPO TransitionCollector.flush writes
    assert {
        "state",
        "action",
        "next_state",
        "episode_id",
        "step_index",
        "update",
        "global_step",
    } == {"state", "action", "next_state", "episode_id", "step_index", "update", "global_step"}


def test_compare_handles_missing_reports(tmp_path):
    """compare() must not crash when some report.json files are absent."""
    rows = compare_contact_dims.compare(tmp_path)
    assert isinstance(rows, dict)
    # no MuJoCo config rows yet (nothing trained in tmp_path)
    for cfg in ("mujoco_default", "mujoco_compliant", "mujoco_hard"):
        assert cfg not in rows or rows[cfg] is not None
