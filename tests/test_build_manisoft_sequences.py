from argparse import Namespace
import json

import numpy as np

from antmaze_ac.data.windows import load_npz_dataset
from scripts.build_manisoft_sequences import (
    convert,
    discover_source_groups,
    load_manisoft_episode,
)


def write_episode(path, rows=4, state_offset=0.0):
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps({
            "schema_version": 3,
            "state_dim": 45,
            "action_dim": 18,
            "control_hz": 50.0,
            "control_dt": 0.02,
            "physics_dt": 0.0002,
            "muscle_torque_scale": 30.0,
            "absolute_action_limit": 0.30,
            "scenario_path": "/test/scenario.yaml",
            "scenario_sha256": "test-scenario",
            "backend_type": "ElasticaBackend",
        }))
    state = np.arange(rows * 45, dtype=np.float32).reshape(rows, 45) + state_offset
    next_state = np.concatenate((state[1:], state[-1:] + 1), axis=0)
    action = np.arange(rows * 18, dtype=np.float32).reshape(rows, 18) / 100
    np.savez_compressed(path, state=state, action=action, next_state=next_state)
    return state, action, next_state


def test_source_episode_becomes_63d_incremental_action_dataset(tmp_path):
    path = tmp_path / "episode_0000.npz"
    state, current_action, next_state = write_episode(path)

    dataset, continuity_error = load_manisoft_episode(path, 7, 45, 18)

    assert dataset.state.shape == dataset.next_state.shape == (4, 63)
    assert dataset.action.shape == dataset.current_action.shape == (4, 18)
    np.testing.assert_array_equal(dataset.state[:, :45], state)
    np.testing.assert_array_equal(dataset.next_state[:, :45], next_state)
    np.testing.assert_array_equal(dataset.state[0, 45:], 0.0)
    np.testing.assert_array_equal(dataset.state[1:, 45:], current_action[:-1])
    np.testing.assert_allclose(
        dataset.action,
        current_action - dataset.state[:, 45:],
        atol=1e-7,
    )
    np.testing.assert_array_equal(dataset.next_state[:, 45:], current_action)
    np.testing.assert_array_equal(dataset.episode_id, 7)
    np.testing.assert_array_equal(dataset.step_index, np.arange(4))
    np.testing.assert_array_equal(dataset.timeout, [False, False, False, True])
    assert continuity_error == 0.0


def test_explicit_counts_select_exact_ranges_and_report_extras(tmp_path):
    first = tmp_path / "seed42"
    second = tmp_path / "seed43"
    first.mkdir()
    second.mkdir()
    for index in range(3):
        write_episode(first / f"episode_{index:04d}.npz")
    for index in range(2):
        write_episode(second / f"episode_{index:04d}.npz")

    groups = discover_source_groups([first, second], [2, 2])

    assert [len(group.paths) for group in groups] == [2, 2]
    assert [path.name for path in groups[0].paths] == [
        "episode_0000.npz",
        "episode_0001.npz",
    ]
    assert [path.name for path in groups[0].ignored_paths] == ["episode_0002.npz"]
    assert not groups[1].ignored_paths


def test_saved_schema_is_accepted_by_training_loader(tmp_path):
    path = tmp_path / "episode_0000.npz"
    write_episode(path)
    dataset, _ = load_manisoft_episode(path, 0, 45, 18)
    converted = tmp_path / "converted.npz"
    np.savez_compressed(converted, **dataset.as_dict())

    loaded = load_npz_dataset(converted)

    assert loaded.state.shape == (4, 63)
    assert loaded.action.shape == (4, 18)


def test_four_roots_are_merged_split_and_saved_for_training(tmp_path):
    roots = []
    counts = [2, 1, 1, 2]
    for group_index, count in enumerate(counts):
        root = tmp_path / f"seed{42 + group_index}"
        root.mkdir()
        roots.append(str(root))
        for episode_index in range(count):
            write_episode(
                root / f"episode_{episode_index:04d}.npz",
                rows=25,
                state_offset=1000.0 * group_index + 100.0 * episode_index,
            )

    output = tmp_path / "processed"
    metadata = convert(
        Namespace(
            config="configs/manisoft_coll.yaml",
            input_root=roots,
            episode_counts=counts,
            expected_episodes=6,
            output=str(output),
            split_seed=42,
            continuity_atol=1e-6,
            progress_every=10,
        )
    )

    assert metadata["episodes"] == 6
    assert metadata["transitions"] == 150
    assert metadata["augmented_state_dim"] == 63
    assert metadata["action_dim"] == 18
    assert sum(part["episodes"] for part in metadata["splits"].values()) == 6
    assert sum(part["rows"] for part in metadata["splits"].values()) == 150
    assert sum(
        part["valid_K_step_windows"] for part in metadata["splits"].values()
    ) == 30
    for split_name in ("train", "validation", "test"):
        split = load_npz_dataset(output / f"{split_name}.npz")
        assert split.state.shape[1] == 63
        assert split.action.shape[1] == 18
        starts = np.load(output / f"{split_name}_K20_starts.npy")
        assert len(starts) == 5 * len(np.unique(split.episode_id))
    assert (output / "metadata.json").is_file()
