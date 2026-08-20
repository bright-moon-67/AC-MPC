"""Build the immutable 100k Cartpole subset of Proto-Stratified-1M."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.dmc.o2o.dataset import convert_exorl, temporal_stratified_episode_indices
from experiments.dmc.o2o.formal_cartpole import validate_dataset


def selected_episode_indices() -> tuple[int, ...]:
    parent_1m = set(temporal_stratified_episode_indices(episodes_per_decile=100))
    selected = temporal_stratified_episode_indices(episodes_per_decile=10)
    if not set(selected).issubset(parent_1m):
        raise AssertionError("100k selection must be a strict subset of Proto-Stratified-1M")
    return selected


def build(source_dir: Path, source_archive: Path, output: Path) -> dict:
    selection = {
        "kind": "temporal_block_microstratum_start_v1",
        "source_total_episodes": 10_000,
        "temporal_blocks": 10,
        "episodes_per_block": 1_000,
        "selected_episodes_per_block": 10,
        "microstratum_width_episodes": 100,
        "microstratum_offset": 0,
        "parent_pool_kind": "Proto-Stratified-1M",
        "parent_pool_selected_episodes_per_block": 100,
        "parent_pool_transitions": 1_000_000,
    }
    metadata = convert_exorl(
        source_dir, output, task="cartpole_swingup", reward_source="oracle",
        max_transitions=100_000, gamma=0.99,
        selected_episode_indices=selected_episode_indices(),
        selection_metadata=selection, source_archive=source_archive,
        allow_unselected_episode_files=True,
    )
    validate_dataset(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Formal dataset output already exists; refusing overwrite")
    result = build(args.source_dir, args.source_archive, args.output)
    manifest_path = args.output.with_suffix(".manifest.json")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
