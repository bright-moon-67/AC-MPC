from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

import experiments.maniskill_pick_visual.dataset as dataset_module
from experiments.maniskill_pick_visual.dataset import (
    VisualWindowDataset,
    fit_normalizers,
    list_episode_ids,
    split_episode_ids,
)


def _write_synthetic_files(tmp_path, lengths=(4, 5, 6, 4, 5, 6)):
    trajectory_path = tmp_path / "visual.h5"
    feature_path = tmp_path / "features.h5"
    with h5py.File(trajectory_path, "w") as trajectories, h5py.File(
        feature_path, "w"
    ) as features:
        for episode_index, transition_count in enumerate(lengths):
            name = f"traj_{episode_index}"
            time = np.arange(transition_count + 1, dtype=np.float32)[:, None]
            robot = np.concatenate(
                (
                    time + episode_index * 10,
                    2 * time + episode_index * 100,
                    -time,
                ),
                axis=1,
            )
            feature = np.concatenate(
                (time + episode_index * 1000, time**2 + episode_index), axis=1
            )
            actions = np.concatenate(
                (
                    np.arange(transition_count, dtype=np.float32)[:, None]
                    + episode_index * 100,
                    -np.ones((transition_count, 1), dtype=np.float32),
                ),
                axis=1,
            )
            group = trajectories.create_group(name)
            group.create_dataset("robot", data=robot)
            group.create_dataset("actions", data=actions)
            group.create_dataset(
                "rgb", data=np.zeros((transition_count + 1, 2, 2, 3), np.uint8)
            )
            group.create_dataset(
                "depth", data=np.zeros((transition_count + 1, 2, 2, 1), np.uint16)
            )
            feature_group = features.create_group(name)
            feature_group.create_dataset("resnet18", data=feature)
    return trajectory_path, feature_path


def test_episode_split_is_deterministic_disjoint_and_complete(tmp_path):
    trajectory_path, _ = _write_synthetic_files(tmp_path)
    names = list_episode_ids(trajectory_path)
    first = split_episode_ids(names, seed=17, fractions=(0.6, 0.2, 0.2))
    second = split_episode_ids(names, seed=17, fractions=(0.6, 0.2, 0.2))
    assert first == second
    assert set(first) == {"train", "val", "test"}
    sets = [set(first[key]) for key in ("train", "val", "test")]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert set().union(*sets) == set(names)
    assert all(sets)


def test_normalizers_use_only_requested_training_episodes(tmp_path):
    trajectory_path, feature_path = _write_synthetic_files(tmp_path)
    train_ids = ["traj_0", "traj_1"]
    normalizers = fit_normalizers(trajectory_path, feature_path, train_ids)
    with h5py.File(trajectory_path, "r") as trajectories, h5py.File(
        feature_path, "r"
    ) as features:
        expected_robot = np.concatenate(
            [trajectories[name]["robot"][:] for name in train_ids]
        )
        expected_feature = np.concatenate(
            [features[name]["resnet18"][:] for name in train_ids]
        )
    np.testing.assert_allclose(normalizers["robot_mean"], expected_robot.mean(0))
    np.testing.assert_allclose(normalizers["robot_std"], expected_robot.std(0))
    np.testing.assert_allclose(
        normalizers["feature_mean"], expected_feature.mean(0)
    )
    np.testing.assert_allclose(
        normalizers["feature_std"], expected_feature.std(0)
    )
    assert normalizers["feature_mean"][0] < 1000


def test_window_alignment_normalization_and_no_episode_crossing(tmp_path):
    trajectory_path, feature_path = _write_synthetic_files(tmp_path, lengths=(4, 3))
    ids = ["traj_0", "traj_1"]
    normalizers = fit_normalizers(trajectory_path, feature_path, ids)
    dataset = VisualWindowDataset(
        trajectory_path,
        feature_path,
        ids,
        horizon=2,
        normalizers=normalizers,
        normalize=True,
    )
    assert len(dataset) == (4 - 2 + 1) + (3 - 2 + 1)
    assert dataset.window_metadata == (
        ("traj_0", 0),
        ("traj_0", 1),
        ("traj_0", 2),
        ("traj_1", 0),
        ("traj_1", 1),
    )
    selected_actions = dataset.action_windows([3, 1])
    assert selected_actions.shape == (2, 2, 2)
    np.testing.assert_array_equal(selected_actions[0, :, 0], [100, 101])
    np.testing.assert_array_equal(selected_actions[1, :, 0], [1, 2])
    sample = dataset[1]
    assert sample["episode_id"] == "traj_0"
    assert sample["start"] == 1
    assert sample["robot"].shape == (3, 3)
    assert sample["features"].shape == (3, 2)
    assert sample["state"].shape == (3, 5)
    assert sample["actions"].shape == (2, 2)
    raw_robot = np.array([[1, 2, -1], [2, 4, -2], [3, 6, -3]], np.float32)
    expected_robot = (
        raw_robot - normalizers["robot_mean"]
    ) / normalizers["robot_std"]
    np.testing.assert_allclose(sample["robot"].numpy(), expected_robot)
    np.testing.assert_array_equal(sample["actions"][:, 0].numpy(), [1, 2])

    last = dataset[len(dataset) - 1]
    assert last["episode_id"] == "traj_1"
    assert last["start"] == 1
    dataset.close()


def test_dataset_reopens_hdf5_safely_for_workers_and_pickle(tmp_path, monkeypatch):
    trajectory_path, feature_path = _write_synthetic_files(tmp_path, lengths=(5,))
    normalizers = fit_normalizers(trajectory_path, feature_path, ["traj_0"])
    dataset = VisualWindowDataset(
        trajectory_path,
        feature_path,
        ["traj_0"],
        horizon=2,
        normalizers=normalizers,
    )
    _ = dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._trajectory_handle is None
    torch.testing.assert_close(restored[0]["state"], dataset[0]["state"])
    restored.close()

    # Simulate a forked worker PID. The inherited handles must be discarded
    # and reopened before reading in the child process.
    parent_pid = dataset._handle_pid
    assert parent_pid is not None
    monkeypatch.setattr(dataset_module.os, "getpid", lambda: parent_pid + 1)
    forked_sample = dataset[0]
    assert dataset._handle_pid == parent_pid + 1
    assert forked_sample["state"].shape == (3, 5)
    dataset.close()


def test_dataset_rejects_t_plus_one_mismatch(tmp_path):
    trajectory_path, feature_path = _write_synthetic_files(tmp_path, lengths=(4,))
    with h5py.File(feature_path, "a") as features:
        del features["traj_0/resnet18"]
        features["traj_0"].create_dataset(
            "resnet18", data=np.zeros((4, 2), dtype=np.float32)
        )
    with pytest.raises(ValueError, match=r"T\+1"):
        VisualWindowDataset(
            trajectory_path,
            feature_path,
            ["traj_0"],
            horizon=1,
            normalizers={
                "robot_mean": np.zeros(3, np.float32),
                "robot_std": np.ones(3, np.float32),
                "feature_mean": np.zeros(2, np.float32),
                "feature_std": np.ones(2, np.float32),
            },
        )
