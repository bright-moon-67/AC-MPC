"""Build the immutable 10x20/200k formal Walker Run dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.dmc.o2o.dataset import (
    convert_exorl,
    temporal_stratified_episode_indices,
)
from experiments.dmc.o2o.formal_walker import validate_dataset


def build(source_dir: Path, source_archive: Path, output: Path) -> dict:
    indices = temporal_stratified_episode_indices(
        source_total_episodes=10_000,
        temporal_deciles=10,
        episodes_per_decile=20,
    )
    selection = {
        "kind": "temporal_block_microstratum_start_v1",
        "source_total_episodes": 10_000,
        "temporal_blocks": 10,
        "episodes_per_block": 1_000,
        "selected_episodes_per_block": 20,
        "microstratum_width_episodes": 50,
        "microstratum_offset": 0,
    }
    metadata = convert_exorl(
        source_dir,
        output,
        task="walker_run",
        reward_source="oracle",
        max_transitions=200_000,
        gamma=0.99,
        selected_episode_indices=indices,
        selection_metadata=selection,
        source_archive=source_archive,
        allow_unselected_episode_files=True,
    )
    # Reload through the strict formal gate before exposing the path.
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
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
