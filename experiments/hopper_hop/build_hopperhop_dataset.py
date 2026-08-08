"""Build the HopperHop Koopman dataset from budgeted PPO transition data.

Merges the per-seed collected ``.npz`` chunks
(``runs/hopper_hop/data_v2/seed_*/``), remaps episode ids to be globally
unique (each seed restarts episode ids at ``num_envs``), and writes a single
dataset ``.npz`` with train/validation/test episode splits.

Splitting strategy: episodes are sorted by their PPO ``update`` stage and then
interleaved (8/1/1) so every split covers early, mid and late policy behavior.
This avoids the classic failure where a Koopman model trained only on early
random-policy data goes out-of-distribution later (the lesson from PandaReach).

Output fields (matches the PandaReach dataset contract):
    state [N,15], action [N,4], next_state [N,15],
    episode_id [N] (remapped, globally unique),
    step_index [N] (consecutive within each episode),
    update [N] (original PPO update stage, kept for analysis),
    train_episode_ids / validation_episode_ids / test_episode_ids,
    state_kind = "hopperhop", state_dim = 15, action_dim = 4,
    source_files (list of the input chunk paths)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATE_DIM = 15
ACTION_DIM = 4


@dataclass(frozen=True)
class BuildConfig:
    collect_root: Path
    output: Path
    seed_dirs: tuple[str, ...] = (
        "seed_20240201",
        "seed_20240202",
        "seed_20240203",
    )
    # 8/1/1 interleave across stage-sorted episodes.
    validation_every: int = 10
    test_offset: int = 9


def _load_chunks(root: Path, seed_dir: str) -> list[dict[str, np.ndarray]]:
    source = root / seed_dir
    if not source.is_dir():
        raise FileNotFoundError(f"Collection directory not found: {source}")
    chunks = []
    for chunk in sorted(source.glob("*.npz")):
        if chunk.name == "collection_status.json":
            continue
        with np.load(chunk, allow_pickle=False) as archive:
            chunks.append({name: archive[name] for name in archive.files})
    if not chunks:
        raise FileNotFoundError(f"No .npz chunks under {source}")
    return chunks


def _validate_chunk(chunk: dict[str, np.ndarray]) -> None:
    required = {"state", "action", "next_state", "episode_id",
                "step_index", "update", "global_step"}
    missing = required - chunk.keys()
    if missing:
        raise KeyError(f"Chunk missing fields: {sorted(missing)}")
    if chunk["state"].shape[1:] != (STATE_DIM,) or \
            chunk["action"].shape[1:] != (ACTION_DIM,):
        raise ValueError(
            f"Expected state [N,{STATE_DIM}] action [N,{ACTION_DIM}], "
            f"got {chunk['state'].shape} / {chunk['action'].shape}"
        )


def _collect_episodes(
    chunks: list[dict[str, np.ndarray]],
) -> list[tuple[int, int, dict[str, np.ndarray], np.ndarray]]:
    """Return [(episode_id, update, chunk, mask)] for every episode."""
    episodes: list[tuple[int, int, dict[str, np.ndarray], np.ndarray]] = []
    for chunk in chunks:
        _validate_chunk(chunk)
        episode_ids = chunk["episode_id"]
        updates = chunk["update"]
        for episode in np.unique(episode_ids):
            mask = episode_ids == episode
            steps = chunk["step_index"][mask]
            if not np.array_equal(steps, np.arange(len(steps))):
                raise ValueError(
                    f"Episode {int(episode)} has non-consecutive step_index"
                )
            chain = chunk["next_state"][mask]
            states = chunk["state"][mask]
            if len(states) > 1 and np.max(
                np.abs(chain[:-1] - states[1:])
            ) > 2e-5:
                raise ValueError(f"Episode {int(episode)} chain mismatch")
            episodes.append((int(episode), int(updates[mask][0]), chunk, mask))
    return episodes


def build(config: BuildConfig) -> Path:
    episodes: list[tuple[int, int, dict[str, np.ndarray], np.ndarray]] = []
    source_files: list[str] = []
    for seed_dir in config.seed_dirs:
        chunks = _load_chunks(config.collect_root, seed_dir)
        source_files.extend(
            str(chunk_path)
            for chunk_path in sorted(
                (config.collect_root / seed_dir).glob("*.npz")
            )
        )
        episodes.extend(_collect_episodes(chunks))

    # Stage-sorted order -> interleaved train/val/test covering all stages.
    episodes.sort(key=lambda item: (item[1], item[0]))

    train_episode_ids: list[int] = []
    validation_episode_ids: list[int] = []
    test_episode_ids: list[int] = []
    remap: dict[int, int] = {}  # original -> globally unique
    for index, (original, _update, _chunk, _mask) in enumerate(episodes):
        remap[original] = index
        bucket = index % config.validation_every
        # 8/1/1: train buckets 0-7, validation bucket 8, test bucket 9.
        if bucket == config.validation_every - 1:
            test_episode_ids.append(index)
        elif bucket == config.validation_every - 2:
            validation_episode_ids.append(index)
        else:
            train_episode_ids.append(index)

    if not (train_episode_ids and validation_episode_ids and test_episode_ids):
        raise ValueError("Every split must contain at least one episode")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    step_indices: list[np.ndarray] = []
    updates: list[np.ndarray] = []
    global_steps: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    for index, (original, update, chunk, mask) in enumerate(episodes):
        count = int(mask.sum())
        states.append(chunk["state"][mask])
        actions.append(chunk["action"][mask])
        next_states.append(chunk["next_state"][mask])
        step_indices.append(chunk["step_index"][mask])
        updates.append(np.full(count, update, dtype=np.int64))
        global_steps.append(chunk["global_step"][mask])
        episode_ids.append(np.full(count, index, dtype=np.int64))

    output = {
        "state": np.concatenate(states).astype(np.float32),
        "action": np.concatenate(actions).astype(np.float32),
        "next_state": np.concatenate(next_states).astype(np.float32),
        "episode_id": np.concatenate(episode_ids),
        "step_index": np.concatenate(step_indices),
        "update": np.concatenate(updates),
        "global_step": np.concatenate(global_steps),
        "train_episode_ids": np.asarray(train_episode_ids, dtype=np.int64),
        "validation_episode_ids": np.asarray(
            validation_episode_ids, dtype=np.int64
        ),
        "test_episode_ids": np.asarray(test_episode_ids, dtype=np.int64),
        "state_kind": np.asarray("hopperhop"),
        "state_dim": np.asarray(STATE_DIM),
        "action_dim": np.asarray(ACTION_DIM),
        "source_files": np.asarray(source_files),
    }
    for name in ("state", "action", "next_state"):
        if not np.isfinite(output[name]).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.output, **output)
    return config.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-root",
        type=Path,
        default=Path("runs/hopper_hop/data_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/hopper_hop/data/hopperhop_koopman.npz"),
    )
    parser.add_argument(
        "--seed-dirs",
        default="seed_20240201,seed_20240202,seed_20240203",
        help="comma-separated seed dir names under collect-root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        collect_root=args.collect_root,
        output=args.output,
        seed_dirs=tuple(args.seed_dirs.split(",")),
    )
    path = build(config)
    with np.load(path, allow_pickle=True) as archive:
        print(
            f"dataset written: {path}",
            flush=True,
        )
        print(
            f"  transitions={len(archive['state']):,} "
            f"episodes(train/val/test)="
            f"{len(archive['train_episode_ids'])}/"
            f"{len(archive['validation_episode_ids'])}/"
            f"{len(archive['test_episode_ids'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
