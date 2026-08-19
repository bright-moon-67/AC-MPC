#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from antmaze_ac.config import load_config
from antmaze_ac.data.build_sequences import (
    Normalizer,
    build_augmented_dataset,
    load_d4rl_hdf5,
    split_by_episode,
    valid_window_starts,
)


def save_dataset(path: Path, dataset) -> None:
    np.savez_compressed(path, **dataset.as_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/antmaze_umaze.yaml")
    parser.add_argument("--input", default="data/raw/antmaze-umaze-v2.hdf5")
    parser.add_argument("--output", default="data/processed/antmaze-umaze-v2")
    parser.add_argument("--expected-sha256", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.expected_sha256:
        import hashlib

        digest = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
        if digest != args.expected_sha256:
            raise ValueError(f"Dataset SHA256 mismatch: {digest}")
    raw, source_shapes = load_d4rl_hdf5(args.input)
    obs_dim = raw["observations"].shape[1]
    action_dim = raw["actions"].shape[1]
    expected_obs = config["experiment"]["expected_observation_dim"]
    expected_action = config["experiment"]["expected_action_dim"]
    if (obs_dim, action_dim) != (expected_obs, expected_action):
        raise ValueError(
            f"Expected AntMaze dimensions {(expected_obs, expected_action)}, got {(obs_dim, action_dim)}"
        )

    dataset = build_augmented_dataset(raw)
    fractions = (
        config["data"]["train_fraction"],
        config["data"]["validation_fraction"],
        config["data"]["test_fraction"],
    )
    splits = split_by_episode(dataset, fractions, config["experiment"]["seed"])
    normalizer = Normalizer.fit(splits["train"].x, config["data"]["normalization_epsilon"])
    k_step = config["koopman"]["K_step"]
    metadata = {
        "dataset_schema_version": 2,
        "transition_semantics": {
            "state": "[observation_t, previous_action=u_{t-1}]",
            "action": "delta_action=u_t-u_{t-1}",
            "next_state": "[next_observation=observation_{t+1}, current_action=u_t]",
            "current_action": "absolute D4RL action u_t",
            "done": "terminal OR timeout",
        },
        "source": str(Path(args.input).resolve()),
        "source_shapes": {key: list(value) for key, value in source_shapes.items()},
        "observation_dim": obs_dim,
        "action_dim": action_dim,
        "augmented_state_dim": obs_dim + action_dim,
        "transitions": len(dataset),
        "episodes": int(dataset.episode_id.max() + 1),
        "max_action_reconstruction_error": float(
            np.max(
                np.abs(
                    dataset.current_action
                    - (dataset.state[:, -action_dim:] + dataset.action)
                )
            )
        ),
        "normalizer": normalizer.state_dict(),
        "splits": {},
    }
    for name, split in splits.items():
        save_dataset(output / f"{name}.npz", split)
        starts = valid_window_starts(split, k_step)
        np.save(output / f"{name}_K{k_step}_starts.npy", starts)
        metadata["splits"][name] = {
            "rows": len(split),
            "episodes": len(np.unique(split.episode_id)),
            "valid_K_step_windows": len(starts),
        }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
