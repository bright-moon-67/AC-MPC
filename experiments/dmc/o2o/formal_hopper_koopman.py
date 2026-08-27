"""Prepare and train the Hopper Stand Koopman model from Hop+Stand data."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from experiments.dmc.o2o.dataset import OfflineDataset
from experiments.dmc.o2o.formal_hopper import FORMAL_TRAINING_SEEDS


KOOPMAN_SELECTION_KINDS = (
    "hopper_mixed_temporal_block_v1",
    "hopper_mixed_quality_balanced_v1",
)


def validate_corpus(path: Path, selection_kind: str) -> OfflineDataset:
    dataset = OfflineDataset.load(path.resolve())
    if dataset.metadata.get("task") != "hopper_stand" or len(dataset) != 400_000:
        raise ValueError("Koopman corpus must contain the mixed 400k Hop+Stand transitions")
    if dataset.metadata.get("selection", {}).get("kind") != selection_kind:
        raise ValueError(
            f"Koopman corpus selection differs from {selection_kind!r}"
        )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-seed", type=int, choices=FORMAL_TRAINING_SEEDS, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="mixed hopper hop+stand 400k canonical dataset")
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-kind",
        choices=KOOPMAN_SELECTION_KINDS,
        default="hopper_mixed_temporal_block_v1",
    )
    parser.add_argument("--jax-python", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--run-inline", action="store_true")
    args = parser.parse_args()
    if args.launch and args.run_inline:
        parser.error("--launch and --run-inline are mutually exclusive")
    dataset = validate_corpus(args.dataset, args.selection_kind)
    prepare = [sys.executable, "-m", "experiments.dmc.o2o.prepare_koopman", "--dataset", str(dataset.path), "--output-dir", str(args.prepared_data_dir.resolve())]
    executable = os.path.abspath(str(args.jax_python or sys.executable))
    train = [executable, "-m", "experiments.playground.train_koopman", "--task", "HopperStand", "--data-dir", str(args.prepared_data_dir.resolve()), "--output-dir", str(args.output_dir.resolve()), "--lift-dim", "48", "--k-step", "20", "--batch-size", "2048", "--max-windows", "500000", "--validation-windows", "10000", "--epochs", "500", "--patience", "40", "--learning-rate", "0.0003", "--stability-reference-dt", "0.04", "--seed", str(args.training_seed)]
    print("prepare:", shlex.join(prepare), flush=True)
    print("train:", shlex.join(train), flush=True)
    if args.run_inline:
        subprocess.run(prepare, check=True)
        subprocess.run(train, check=True)
        return
    if not args.launch:
        subprocess.run(prepare, check=True)
        return
    subprocess.run(prepare, check=True)
    session = f"hopper_koopman_{args.training_seed}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shlex.join(train)], check=True)
    print(f"tmux session: {session}")


if __name__ == "__main__":
    main()
