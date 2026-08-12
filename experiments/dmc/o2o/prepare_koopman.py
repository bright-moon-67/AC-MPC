"""Prepare the canonical ExORL 1M episodes for the GPU Koopman trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments.dmc.o2o.dataset import OfflineDataset


STAGE_RANGES = {
    "early": (0, 330),
    "mid": (330, 660),
    "late": (660, 1000),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_stage(
    path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
) -> str:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                states=states,
                actions=actions,
                rewards=rewards,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def prepare(dataset_path: Path, output_dir: Path) -> dict:
    dataset = OfflineDataset.load(dataset_path)
    episode_count = int(dataset.metadata["episodes"])
    if episode_count != 1000 or len(dataset) != 1_000_000:
        raise ValueError("Primary ExORL Koopman fit requires exactly 1000x1000 transitions")
    episode_id = dataset.arrays["episode_id"].reshape(episode_count, 1000)
    episode_step = dataset.arrays["episode_step"].reshape(episode_count, 1000)
    expected_episode_id = np.broadcast_to(
        np.arange(episode_count, dtype=episode_id.dtype)[:, None], episode_id.shape
    )
    if not np.array_equal(episode_id, expected_episode_id):
        raise ValueError("Dataset episode IDs are not canonical")
    expected_episode_step = np.broadcast_to(
        np.arange(1000, dtype=episode_step.dtype)[None, :], episode_step.shape
    )
    if not np.array_equal(episode_step, expected_episode_step):
        raise ValueError("Dataset episode steps are not canonical")
    observation = dataset.arrays["observation"].reshape(episode_count, 1000, 5)
    next_observation = dataset.arrays["next_observation"].reshape(
        episode_count, 1000, 5
    )
    if not np.array_equal(observation[:, 1:], next_observation[:, :-1]):
        raise ValueError("Dataset transitions are not contiguous within episodes")
    states = np.concatenate((observation[:, :1], next_observation), axis=1)
    actions = dataset.arrays["action"].reshape(episode_count, 1000, 1)
    rewards = dataset.arrays["reward"].reshape(episode_count, 1000)
    discounts = dataset.arrays["discount"].reshape(episode_count, 1000)
    if not np.array_equal(discounts, np.ones_like(discounts)):
        raise ValueError("Canonical ExORL Cartpole episodes must retain discount one")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_counts: dict[str, int] = {}
    stage_metadata: dict[str, dict] = {}
    for name, (left, right) in STAGE_RANGES.items():
        path = output_dir / f"{name}.npz"
        checksum = _write_stage(
            path,
            states[left:right],
            actions[left:right],
            rewards[left:right],
        )
        stage_counts[name] = int(right - left)
        stage_metadata[name] = {
            "path": str(path),
            "sha256": checksum,
            "episode_id_start_inclusive": left,
            "episode_id_end_exclusive": right,
            "episodes": right - left,
            "states_shape": list(states[left:right].shape),
            "actions_shape": list(actions[left:right].shape),
            "rewards_shape": list(rewards[left:right].shape),
        }
    manifest = {
        "kind": "exorl_cartpole_koopman_adapter_v1",
        "task": "CartpoleSwingup",
        "policy": "ExORL ProtoRL exploratory data",
        "source_dataset": str(dataset.path),
        "source_dataset_sha256": dataset.sha256,
        "canonical_transitions_npz": str(dataset.path),
        "canonical_transitions_npz_sha256": dataset.sha256,
        "total_transitions": len(dataset),
        "episodes": episode_count,
        "stage_episode_counts": stage_counts,
        "stages": stage_metadata,
        "episode_steps": 1000,
        "observation_dim": 5,
        "action_dim": 1,
        "trainer_episode_split": "per_stage_modulo_10_8_1_1",
        "trainer_split_episode_counts": {
            "train": 800,
            "validation": 100,
            "test": 100,
        },
        "reward": dataset.metadata.get("reward"),
        "source_episode_identity_sha256": dataset.metadata.get(
            "source_episode_identity_sha256"
        ),
        "note": "early/mid/late are deterministic dataset partitions, not policy stages",
    }
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
