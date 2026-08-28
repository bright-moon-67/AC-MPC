from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("mani_skill")

from experiments.maniskill_pick_visual.collect_visual_pickcube import _write_episode
from experiments.maniskill_pick_visual.upgrade_applied_actions import upgrade


def test_collector_schema_distinguishes_applied_and_raw_actions(tmp_path) -> None:
    path = tmp_path / "episode.h5"
    raw = np.asarray([[2.0, -3.0], [0.25, -0.5]], dtype=np.float32)
    applied = np.clip(raw, -1.0, 1.0)
    robots = [np.zeros(3, np.float32) for _ in range(3)]
    rgbs = [np.zeros((2, 2, 3), np.uint8) for _ in range(3)]
    depths = [np.zeros((2, 2, 1), np.uint16) for _ in range(3)]
    with h5py.File(path, "x") as output:
        _write_episode(
            output,
            "traj_0",
            robots=robots,
            rgbs=rgbs,
            depths=depths,
            actions=applied,
            raw_actions=raw,
            terminated=[False, False],
            truncated=[False, True],
            success=[False, True],
            source_episode={"episode_id": 9, "episode_seed": 10},
        )
    with h5py.File(path, "r") as output:
        np.testing.assert_array_equal(output["traj_0/actions"][:], applied)
        np.testing.assert_array_equal(output["traj_0/raw_actions"][:], raw)


def test_v1_upgrade_is_non_destructive_and_clips_actions(tmp_path) -> None:
    source_path = tmp_path / "v1.h5"
    output_path = tmp_path / "v2.h5"
    raw = np.asarray([[2.0, -3.0], [0.25, -0.5]], dtype=np.float32)
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 2, 2, 3)
    with h5py.File(source_path, "x") as source:
        source.attrs["causal_replay"] = True
        group = source.create_group("traj_0")
        group.create_dataset("actions", data=raw)
        group.create_dataset("rgb", data=rgb)

    summary = upgrade(source_path, output_path)

    assert summary["clipped_action_scalars"] == 2
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "r") as output:
        np.testing.assert_array_equal(source["traj_0/actions"][:], raw)
        np.testing.assert_array_equal(output["traj_0/raw_actions"][:], raw)
        np.testing.assert_array_equal(
            output["traj_0/actions"][:], np.clip(raw, -1, 1)
        )
        np.testing.assert_array_equal(output["traj_0/rgb"][:], rgb)
        assert bool(output.attrs["actions_are_applied"])
