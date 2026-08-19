import numpy as np

from antmaze_ac.data.build_sequences import (
    Normalizer,
    build_augmented_dataset,
    split_by_episode,
    valid_window_starts,
)
from antmaze_ac.data.windows import load_npz_dataset


def raw_dataset():
    observations = np.arange(24, dtype=np.float32).reshape(12, 2)
    actions = np.arange(12, dtype=np.float32)[:, None] / 10
    terminals = np.zeros(12, dtype=bool)
    timeouts = np.zeros(12, dtype=bool)
    timeouts[[3, 7, 11]] = True
    return {
        "observations": observations,
        "next_observations": observations + 1,
        "actions": actions,
        "rewards": np.zeros(12, dtype=np.float32),
        "terminals": terminals,
        "timeouts": timeouts,
    }


def test_boundaries_and_action_reconstruction():
    dataset = build_augmented_dataset(raw_dataset())
    dataset.validate()
    starts = dataset.step_index == 0
    np.testing.assert_array_equal(dataset.state[starts, -1], 0)
    np.testing.assert_array_equal(dataset.episode_id, np.repeat(np.arange(3), 4))
    np.testing.assert_array_equal(dataset.step_index, np.tile(np.arange(4), 3))
    np.testing.assert_allclose(
        dataset.current_action,
        dataset.state[:, -1:] + dataset.action,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        dataset.next_state[:, -1:],
        dataset.current_action,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        dataset.done,
        dataset.terminal | dataset.timeout,
    )
    assert dataset.x is dataset.state
    assert dataset.delta_action is dataset.action
    assert dataset.next_x is dataset.next_state
    np.testing.assert_array_equal(valid_window_starts(dataset, 3), [0, 4, 8])
    assert len(valid_window_starts(dataset, 4)) == 0


def test_episode_split_and_train_only_normalization():
    dataset = build_augmented_dataset(raw_dataset())
    splits = split_by_episode(dataset, seed=5)
    ids = [set(split.episode_id.tolist()) for split in splits.values()]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    normalizer = Normalizer.fit(splits["train"].state)
    recovered = normalizer.denormalize(normalizer.normalize(dataset.state))
    np.testing.assert_allclose(recovered, dataset.state, atol=1e-5)


def test_loader_upgrades_legacy_npz_without_changing_semantics(tmp_path):
    dataset = build_augmented_dataset(raw_dataset())
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        x=dataset.state,
        delta_action=dataset.action,
        next_x=dataset.next_state,
        reward=dataset.reward,
        terminal=dataset.terminal,
        timeout=dataset.timeout,
        episode_id=dataset.episode_id,
        step_index=dataset.step_index,
        action=dataset.current_action,
    )
    upgraded = load_npz_dataset(path)
    np.testing.assert_array_equal(upgraded.state, dataset.state)
    np.testing.assert_array_equal(upgraded.action, dataset.action)
    np.testing.assert_array_equal(upgraded.next_state, dataset.next_state)
    np.testing.assert_array_equal(
        upgraded.current_action,
        dataset.current_action,
    )
    np.testing.assert_array_equal(upgraded.done, dataset.done)
