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
from experiments.dmc.tasks.registry import get_task_spec


LEGACY_STAGE_RANGES = {
    "early": (0, 330),
    "mid": (330, 660),
    "late": (660, 1000),
}


def _adapter_stages(dataset: OfflineDataset) -> list[tuple[str, int, int, list[int]]]:
    """Resolve deterministic adapter stages, preserving source-time strata."""

    episode_count = int(dataset.metadata["episodes"])
    selected = dataset.metadata.get("source_episode_indices")
    selection = dataset.metadata.get("selection")
    if selected is None:
        return [
            (name, left, right, list(range(left, right)))
            for name, (left, right) in LEGACY_STAGE_RANGES.items()
        ]
    if not isinstance(selected, list) or len(selected) != episode_count:
        raise ValueError("Stratified dataset has an invalid source episode ID list")
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in selected):
        raise ValueError("Stratified source episode IDs must be integers")
    if selected != sorted(set(selected)):
        raise ValueError("Stratified source episode IDs must be strictly increasing")
    if not isinstance(selection, dict) or selection.get("kind") not in {
        "temporal_block_microstratum_start_v1",
        "hopper_temporal_block_v1",
        "hopper_mixed_temporal_block_v1",
        "hopper_mixed_quality_balanced_v1",
    }:
        raise ValueError("Stratified dataset selection contract is missing or unsupported")
    if selection.get("kind") == "hopper_mixed_quality_balanced_v1":
        quality_order = selection.get("quality_order")
        quality_counts = selection.get("selected_episodes_per_quality")
        if quality_order != ["low", "medium", "high"] or not isinstance(
            quality_counts, dict
        ):
            raise ValueError("Invalid mixed Hopper quality-balanced contract")
        stages: list[tuple[str, int, int, list[int]]] = []
        left = 0
        for quality in quality_order:
            count = quality_counts.get(quality)
            if isinstance(count, bool) or not isinstance(count, int) or count < 10:
                raise ValueError("Invalid quality-stage episode count")
            right = left + count
            stages.append(
                (
                    f"quality_{quality}",
                    left,
                    right,
                    selected[left:right],
                )
            )
            left = right
        if left != episode_count:
            raise ValueError("Quality-stage counts do not cover the corpus")
        return stages
    source_total = int(selection.get("source_total_episodes", -1))
    blocks = int(selection.get("temporal_blocks", -1))
    per_block = int(selection.get("selected_episodes_per_block", -1))
    episodes_per_block = int(selection.get("episodes_per_block", -1))
    if selection.get("kind") in {
        "hopper_temporal_block_v1",
        "hopper_mixed_temporal_block_v1",
    }:
        if blocks != 10 or per_block < 1 or episode_count != blocks * per_block:
            raise ValueError("Invalid mixed Hopper temporal-block contract")
        return [
            (
                f"decile_{block:02d}",
                block * per_block,
                (block + 1) * per_block,
                selected[block * per_block : (block + 1) * per_block],
            )
            for block in range(blocks)
        ]
    if (
        source_total != 10_000
        or blocks != 10
        or per_block < 1
        or episodes_per_block != 1000
        or episode_count != blocks * per_block
        or episodes_per_block % per_block
    ):
        raise ValueError("Expected a Proto10M ten-decile microstratum contract")

    stages: list[tuple[str, int, int, list[int]]] = []
    for block in range(blocks):
        left = block * per_block
        right = left + per_block
        source_ids = selected[left:right]
        micro_width = episodes_per_block // per_block
        expected_ids = list(
            range(
                block * episodes_per_block,
                (block + 1) * episodes_per_block,
                micro_width,
            )
        )
        if source_ids != expected_ids:
            raise ValueError(
                f"Temporal decile {block} does not contain canonical microstratum starts"
            )
        stages.append((f"decile_{block:02d}", left, right, source_ids))
    return stages


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
    task_name = str(dataset.metadata.get("task", "cartpole_swingup"))
    spec = get_task_spec(task_name)
    episode_count = int(dataset.metadata["episodes"])
    episode_steps = int(dataset.metadata.get("transitions_per_episode", 1000))
    if episode_count < 1 or len(dataset) != episode_count * episode_steps:
        raise ValueError("Koopman adapter requires complete fixed-length episodes")
    episode_id = dataset.arrays["episode_id"].reshape(episode_count, episode_steps)
    episode_step = dataset.arrays["episode_step"].reshape(episode_count, episode_steps)
    expected_episode_id = np.broadcast_to(
        np.arange(episode_count, dtype=episode_id.dtype)[:, None], episode_id.shape
    )
    if not np.array_equal(episode_id, expected_episode_id):
        raise ValueError("Dataset episode IDs are not canonical")
    expected_episode_step = np.broadcast_to(
        np.arange(episode_steps, dtype=episode_step.dtype)[None, :], episode_step.shape
    )
    if not np.array_equal(episode_step, expected_episode_step):
        raise ValueError("Dataset episode steps are not canonical")
    observation = dataset.arrays["observation"].reshape(
        episode_count, episode_steps, spec.obs_dim
    )
    next_observation = dataset.arrays["next_observation"].reshape(
        episode_count, episode_steps, spec.obs_dim
    )
    if not np.array_equal(observation[:, 1:], next_observation[:, :-1]):
        raise ValueError("Dataset transitions are not contiguous within episodes")
    states = np.concatenate((observation[:, :1], next_observation), axis=1)
    actions = dataset.arrays["action"].reshape(
        episode_count, episode_steps, spec.action_dim
    )
    rewards = dataset.arrays["reward"].reshape(episode_count, episode_steps)
    discounts = dataset.arrays["discount"].reshape(episode_count, episode_steps)
    if not np.array_equal(discounts, np.ones_like(discounts)):
        raise ValueError("Canonical ExORL episodes must retain discount one")

    stages = _adapter_stages(dataset)
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for _name, left, right, _source_ids in stages:
        local_modulo = np.arange(right - left) % 10
        split_counts["train"] += int(np.count_nonzero(local_modulo < 8))
        split_counts["validation"] += int(np.count_nonzero(local_modulo == 8))
        split_counts["test"] += int(np.count_nonzero(local_modulo == 9))

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_counts: dict[str, int] = {}
    stage_metadata: dict[str, dict] = {}
    for name, left, right, source_episode_ids in stages:
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
            "source_episode_indices": source_episode_ids,
            "source_episode_index_first": source_episode_ids[0],
            "source_episode_index_last": source_episode_ids[-1],
        }
    manifest = {
        "kind": (
            "exorl_cartpole_koopman_adapter_v1"
            if task_name == "cartpole_swingup"
            else "exorl_walker_run_koopman_adapter_v1"
            if task_name == "walker_run"
            else "tdmpc2_hopper_koopman_adapter_v1"
        ),
        "task": (
            "CartpoleSwingup"
            if task_name == "cartpole_swingup"
            else "WalkerRun"
            if task_name == "walker_run"
            else "HopperStand"
        ),
        "policy": (
            "TD-MPC2 MT30 mixed-quality replay"
            if task_name == "hopper_stand"
            else "ExORL ProtoRL exploratory data"
        ),
        "source_dataset": str(dataset.path),
        "source_dataset_sha256": dataset.sha256,
        "canonical_transitions_npz": str(dataset.path),
        "canonical_transitions_npz_sha256": dataset.sha256,
        "total_transitions": len(dataset),
        "episodes": episode_count,
        "stage_episode_counts": stage_counts,
        "stage_order": [name for name, _left, _right, _source_ids in stages],
        "stages": stage_metadata,
        "episode_steps": episode_steps,
        "observation_dim": spec.obs_dim,
        "action_dim": spec.action_dim,
        "trainer_episode_split": "per_stage_modulo_10_8_1_1",
        "trainer_split_episode_counts": split_counts,
        "reward": dataset.metadata.get("reward"),
        "source_episode_identity_sha256": dataset.metadata.get(
            "source_episode_identity_sha256"
        ),
        "source_episode_indices_sha256": dataset.metadata.get(
            "source_episode_indices_sha256"
        ),
        "source_episode_indices": dataset.metadata.get("source_episode_indices"),
        "selection": dataset.metadata.get("selection"),
        "source_archive": dataset.metadata.get("source_archive"),
        "source_archive_sha256": dataset.metadata.get("source_archive_sha256"),
        "environment_discount_values": dataset.metadata.get(
            "environment_discount_values"
        ),
        "recorded_reward_max_abs_error": dataset.metadata.get(
            "recorded_reward_max_abs_error"
        ),
        "reward_source": dataset.metadata.get("reward_source", "oracle"),
        "reward_training": "outside_koopman_contract",
        "note": (
            "Each quality stage contains the deterministic low/medium/high "
            "return-balanced selection; local episode index modulo 10 gives "
            "an 80/10/10 train/validation/test split."
            if dataset.metadata.get("selection", {}).get("kind")
            == "hopper_mixed_quality_balanced_v1"
            else "Each decile stage is a deterministic source-time stratum; local "
            "episode index modulo 10 gives an 80/10/10 train/validation/test split."
            if dataset.metadata.get("source_episode_indices") is not None
            else "early/mid/late are deterministic dataset partitions, not policy stages"
        ),
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
