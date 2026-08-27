"""Convert TD-MPC2 MT30 Hopper arrays into the canonical O2O transition NPZ.

The MT30 files store one dummy row followed by 500 outer (action-repeat=2)
transitions per episode.  This converter keeps complete episodes and performs
the same deterministic temporal-block sampling used by the Walker formal
protocol, adapted to 400 episodes (40 per decile) so that 500-step episodes
still yield exactly 200,000 transitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.dmc.o2o.dataset import (
    TDMPC2_DATASET_KIND,
    _mc_returns,
    validate_dataset_arrays,
)


EPISODE_STEPS = 500
SOURCE_EPISODES = 24_000
TEMPORAL_BLOCKS = 10
EPISODES_PER_BLOCK = SOURCE_EPISODES // TEMPORAL_BLOCKS
SELECTED_PER_BLOCK = 40
QUALITY_ORDER = ("low", "medium", "high")
QUALITY_SELECTED_PER_TASK = {"low": 133, "medium": 134, "high": 133}
QUALITY_RETURN_RANGES = {
    "hopper_stand": {
        "low": (None, 500.0),
        "medium": (500.0, 900.0),
        "high": (900.0, None),
    },
    "hopper_hop": {
        "low": (None, 300.0),
        "medium": (300.0, 500.0),
        "high": (500.0, None),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_ids(total: int = SOURCE_EPISODES) -> list[int]:
    block_size = total // TEMPORAL_BLOCKS
    width = block_size // SELECTED_PER_BLOCK
    if total % TEMPORAL_BLOCKS or block_size % SELECTED_PER_BLOCK:
        raise ValueError("TD-MPC2 source does not support the fixed temporal selection")
    return [
        block * block_size + offset * width
        for block in range(TEMPORAL_BLOCKS)
        for offset in range(SELECTED_PER_BLOCK)
    ]


def _source_manifest(source: Path) -> tuple[str, str]:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = source / "source_manifest.json"
    if not manifest_path.is_file():
        return str(source), ""
    return str(manifest_path), sha256(manifest_path)


def quality_balanced_ids(source: Path, task: str) -> tuple[dict[str, list[int]], dict]:
    """Select source-index-dispersed episodes from explicit return bands."""

    reward = np.load(source / "reward.npy", mmap_mode="r")
    if reward.shape != (SOURCE_EPISODES, EPISODE_STEPS + 1):
        raise ValueError(f"Unexpected reward shape {reward.shape}")
    episode_return = np.asarray(reward[:, 1:], dtype=np.float64).sum(axis=1)
    selected: dict[str, list[int]] = {}
    audit: dict[str, dict] = {}
    for quality in QUALITY_ORDER:
        lower, upper = QUALITY_RETURN_RANGES[task][quality]
        mask = np.ones(SOURCE_EPISODES, dtype=np.bool_)
        if lower is not None:
            mask &= episode_return >= lower
        if upper is not None:
            mask &= episode_return < upper
        candidates = np.flatnonzero(mask)
        count = QUALITY_SELECTED_PER_TASK[quality]
        if len(candidates) < count:
            raise ValueError(
                f"{task} {quality} only has {len(candidates)} candidates for {count} slots"
            )
        # The candidate IDs are already in source order.  Taking the center of
        # each equal-width microstratum covers the whole source sequence while
        # remaining deterministic and avoiding endpoint bias.
        positions = np.floor(
            (np.arange(count, dtype=np.float64) + 0.5) * len(candidates) / count
        ).astype(np.int64)
        ids = candidates[positions]
        returns = episode_return[ids]
        if len(np.unique(ids)) != count:
            raise ValueError(f"{task} {quality} sampling produced duplicate episodes")
        selected[quality] = ids.tolist()
        audit[quality] = {
            "return_lower_inclusive": lower,
            "return_upper_exclusive": upper,
            "available_episodes": int(len(candidates)),
            "selected_episodes": count,
            "selected_source_episode_indices": ids.tolist(),
            "selected_return_min": float(returns.min()),
            "selected_return_mean": float(returns.mean()),
            "selected_return_median": float(np.median(returns)),
            "selected_return_max": float(returns.max()),
        }
    flat = [index for quality in QUALITY_ORDER for index in selected[quality]]
    if len(flat) != 400 or len(set(flat)) != 400:
        raise ValueError(f"{task} quality-balanced selection must contain 400 episodes")
    return selected, audit


def _load_task(source: Path, ids: list[int]) -> dict[str, np.ndarray]:
    observation = np.load(source / "observation.npy", mmap_mode="r")
    action = np.load(source / "action.npy", mmap_mode="r")
    reward = np.load(source / "reward.npy", mmap_mode="r")
    if observation.shape[0] != SOURCE_EPISODES or observation.shape[1] != EPISODE_STEPS + 1:
        raise ValueError(f"Unexpected observation shape {observation.shape}")
    if action.shape[:2] != observation.shape[:2] or reward.shape[:2] != observation.shape[:2]:
        raise ValueError("TD-MPC2 arrays have inconsistent leading shapes")
    episodes = []
    actions = []
    rewards = []
    for index in ids:
        # Row zero is the uninitialized/dummy record.  Real transition t uses
        # obs[t] -> obs[t+1], action[t+1], reward[t+1].
        episodes.append(np.asarray(observation[index, :EPISODE_STEPS], dtype=np.float32))
        actions.append(np.asarray(action[index, 1 : EPISODE_STEPS + 1], dtype=np.float32))
        rewards.append(np.asarray(reward[index, 1 : EPISODE_STEPS + 1], dtype=np.float32))
    obs = np.stack(episodes)
    act = np.stack(actions)
    rew = np.stack(rewards)
    if not np.isfinite(obs).all() or not np.isfinite(act).all() or not np.isfinite(rew).all():
        raise FloatingPointError("Selected TD-MPC2 real rows contain NaN or Inf")
    if np.any(np.abs(act) > 1.00001):
        raise ValueError("TD-MPC2 actions exceed [-1,1]")
    next_obs = np.concatenate((obs[:, 1:], observation[ids, EPISODE_STEPS : EPISODE_STEPS + 1].astype(np.float32)), axis=1)
    # The concatenation above is intentionally replaced below with an exact
    # indexed gather; keeping this local avoids materializing all source rows.
    next_obs = np.stack([
        np.asarray(observation[index, 1 : EPISODE_STEPS + 1], dtype=np.float32)
        for index in ids
    ])
    return {"observation": obs, "action": act, "reward": rew, "next_observation": next_obs}


def _canonical(task: str, source: Path, ids: list[int], *, source_ids: list[int] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    loaded = _load_task(source, ids)
    episodes = len(ids)
    count = episodes * EPISODE_STEPS
    arrays = {
        key: value.reshape(count, value.shape[-1]) if value.ndim == 3 else value.reshape(count)
        for key, value in loaded.items()
    }
    arrays["episode_id"] = np.repeat(np.arange(episodes, dtype=np.int64), EPISODE_STEPS)
    arrays["episode_step"] = np.tile(np.arange(EPISODE_STEPS, dtype=np.int32), episodes)
    arrays["discount"] = np.ones(count, dtype=np.float32)
    arrays["terminated"] = np.zeros(count, dtype=np.bool_)
    arrays["truncated"] = np.zeros(count, dtype=np.bool_)
    arrays["truncated"][EPISODE_STEPS - 1 :: EPISODE_STEPS] = True
    arrays["mc_return"] = np.concatenate(
        [
            _mc_returns(
                arrays["reward"][i * EPISODE_STEPS : (i + 1) * EPISODE_STEPS],
                arrays["discount"][i * EPISODE_STEPS : (i + 1) * EPISODE_STEPS],
                0.99,
            )
            for i in range(episodes)
        ]
    )
    selected = list(ids if source_ids is None else source_ids)
    metadata = {
        "kind": TDMPC2_DATASET_KIND,
        "task": task,
        "source": f"TD-MPC2 MT30 {task} exploratory dataset",
        "source_directory": str(source.resolve()),
        "source_archive": _source_manifest(source)[0],
        "source_archive_sha256": _source_manifest(source)[1],
        "source_schema": "tdmpc2_mt30_episode_arrays_v1",
        "observation_dim": int(arrays["observation"].shape[1]),
        "action_dim": int(arrays["action"].shape[1]),
        "transitions": count,
        "episodes": episodes,
        "transitions_per_episode": EPISODE_STEPS,
        "source_episode_indices": selected,
        "source_episode_index_first": selected[0],
        "source_episode_index_last": selected[-1],
        "selection": {
            "kind": "hopper_temporal_block_v1",
            "source_total_episodes": SOURCE_EPISODES,
            "temporal_blocks": TEMPORAL_BLOCKS,
            "episodes_per_block": EPISODES_PER_BLOCK,
            "selected_episodes_per_block": SELECTED_PER_BLOCK,
            "microstratum_width_episodes": EPISODES_PER_BLOCK // SELECTED_PER_BLOCK,
            "microstratum_offset": 0,
        },
        "gamma_for_mc_return": 0.99,
        "alignment": "outer_obs[t],outer_action[t],sum(native_rewards[t:t+2]),outer_obs[t+1]",
        "reward": "tdmpc2_recorded_two_native_substep_reward_sum_v1",
        "reward_source": "recorded",
        "reward_oracle": None,
        "recorded_reward_is_training_target": True,
        "timeout_bootstrap": "preserve_environment_discount",
        "boundary_semantics": "all complete 500-step episodes end by timeout with discount one",
        "environment_discount_values": [1.0],
        "action_repeat": 2,
        "control_dt": 0.04,
        "physics_dt": 0.005,
    }
    validate_dataset_arrays(arrays, gamma_for_mc_return=0.99, expected_episode_count=episodes, task=task, reward_source="recorded")
    return arrays, metadata


def write_dataset(output: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    metadata["sha256"] = sha256(output)
    output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--stand-source", type=Path)
    parser.add_argument("--hop-source", type=Path)
    parser.add_argument("--task", choices=("hopper_stand", "hopper_hop", "hopper_mixed"), required=True)
    parser.add_argument(
        "--selection",
        choices=("temporal_block", "quality_balanced"),
        default="temporal_block",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = selected_ids()
    if args.selection == "quality_balanced" and args.task != "hopper_mixed":
        parser.error("--selection quality_balanced is currently defined for hopper_mixed")
    if args.task in {"hopper_stand", "hopper_hop"}:
        if args.source is None:
            parser.error("--source is required for a single Hopper task")
        arrays, metadata = _canonical(args.task, args.source, ids)
    elif args.selection == "temporal_block":
        if args.stand_source is None or args.hop_source is None:
            parser.error("--stand-source and --hop-source are required for hopper_mixed")
        per_task = []
        for block in range(TEMPORAL_BLOCKS):
            block_ids = ids[block * SELECTED_PER_BLOCK : (block + 1) * SELECTED_PER_BLOCK]
            per_task.append((args.stand_source, block_ids, block_ids))
            per_task.append((args.hop_source, block_ids, [SOURCE_EPISODES + i for i in block_ids]))
        chunks = [_canonical("hopper_stand", source, local, source_ids=source_ids)[0] for source, local, source_ids in per_task]
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}
        # Rebuild contiguous episode IDs/steps after interleaving the two tasks.
        episodes = len(ids) * 2
        arrays["episode_id"] = np.repeat(np.arange(episodes, dtype=np.int64), EPISODE_STEPS)
        arrays["episode_step"] = np.tile(np.arange(EPISODE_STEPS, dtype=np.int32), episodes)
        metadata = dict(_canonical("hopper_stand", args.stand_source, ids)[1])
        metadata.update({
            "task": "hopper_stand",
            "source": "TD-MPC2 MT30 Hopper hop+stand Koopman corpus",
            "source_directory": f"{args.stand_source.resolve()} + {args.hop_source.resolve()}",
            "episodes": episodes,
            "transitions": episodes * EPISODE_STEPS,
            # Synthetic monotone identities interleave stand/hop within each
            # source-time stratum; the per-task source directories are bound
            # separately above and in the reproducibility manifest.
            "source_episode_indices": [
                2 * i + task_bit
                for block in range(TEMPORAL_BLOCKS)
                for i in ids[block * SELECTED_PER_BLOCK : (block + 1) * SELECTED_PER_BLOCK]
                for task_bit in (0, 1)
            ],
            "selection": {
                "kind": "hopper_mixed_temporal_block_v1",
                "source_total_episodes": SOURCE_EPISODES * 2,
                "temporal_blocks": TEMPORAL_BLOCKS,
                "episodes_per_block": EPISODES_PER_BLOCK * 2,
                "selected_episodes_per_block": SELECTED_PER_BLOCK * 2,
                "microstratum_width_episodes": EPISODES_PER_BLOCK // SELECTED_PER_BLOCK,
                "microstratum_offset": 0,
            },
        })
        validate_dataset_arrays(arrays, gamma_for_mc_return=0.99, expected_episode_count=episodes, task="hopper_stand", reward_source="recorded")
    else:
        if args.stand_source is None or args.hop_source is None:
            parser.error("--stand-source and --hop-source are required for hopper_mixed")
        task_sources = {
            "hopper_stand": args.stand_source.resolve(),
            "hopper_hop": args.hop_source.resolve(),
        }
        selections: dict[str, dict[str, list[int]]] = {}
        quality_audit: dict[str, dict] = {}
        for task, source in task_sources.items():
            selections[task], quality_audit[task] = quality_balanced_ids(
                source, task
            )
        chunks = []
        quality_stage_counts: dict[str, int] = {}
        for quality in QUALITY_ORDER:
            for task in ("hopper_stand", "hopper_hop"):
                chunks.append(
                    _canonical(
                        task,
                        task_sources[task],
                        selections[task][quality],
                    )[0]
                )
            quality_stage_counts[quality] = sum(
                len(selections[task][quality]) for task in task_sources
            )
        arrays = {
            key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
            for key in chunks[0]
        }
        episodes = 800
        arrays["episode_id"] = np.repeat(
            np.arange(episodes, dtype=np.int64), EPISODE_STEPS
        )
        arrays["episode_step"] = np.tile(
            np.arange(EPISODE_STEPS, dtype=np.int32), episodes
        )
        metadata = dict(
            _canonical(
                "hopper_stand",
                args.stand_source,
                selections["hopper_stand"]["low"],
            )[1]
        )
        source_manifests = {}
        for task, source in task_sources.items():
            manifest_path, manifest_sha256 = _source_manifest(source)
            source_manifests[task] = {
                "path": manifest_path,
                "sha256": manifest_sha256,
            }
        metadata.update(
            {
                "task": "hopper_stand",
                "source": (
                    "TD-MPC2 MT30 Hopper hop+stand quality-balanced Koopman corpus"
                ),
                "source_directory": " + ".join(
                    str(task_sources[task])
                    for task in ("hopper_stand", "hopper_hop")
                ),
                "source_manifests": source_manifests,
                "episodes": episodes,
                "transitions": episodes * EPISODE_STEPS,
                # Canonical corpus identities are monotone.  Exact task-local
                # origins live in the quality audit below.
                "source_episode_indices": list(range(episodes)),
                "source_episode_index_first": 0,
                "source_episode_index_last": episodes - 1,
                "selection": {
                    "kind": "hopper_mixed_quality_balanced_v1",
                    "source_total_episodes": SOURCE_EPISODES * 2,
                    "source_episodes_per_task": SOURCE_EPISODES,
                    "quality_order": list(QUALITY_ORDER),
                    "selected_episodes_per_quality": quality_stage_counts,
                    "selected_episodes_per_task_quality": dict(
                        QUALITY_SELECTED_PER_TASK
                    ),
                    "sampling": "source_index_microstratum_midpoint_v1",
                    "task_quality_audit": quality_audit,
                },
            }
        )
        validate_dataset_arrays(
            arrays,
            gamma_for_mc_return=0.99,
            expected_episode_count=episodes,
            task="hopper_stand",
            reward_source="recorded",
        )
    write_dataset(args.output, arrays, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
