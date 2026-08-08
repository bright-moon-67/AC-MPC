"""Assemble MuJoCo Koopman datasets with the ORIGINAL build pipeline.

For every contact config (mujoco_default / mujoco_compliant / mujoco_hard)
merges the three seed chunk dirs collected by
``collect_hopperhop_mujoco.py`` into the standard dataset contract via the
existing ``build_hopperhop_dataset.build`` (8/1/1 stage-interleaved splits).

Outputs:
    runs/hopper_hop_mujoco/data/<contact>/hopperhop_koopman.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.hopper_hop.build_hopperhop_dataset import (
    BuildConfig,
    build,
)
from experiments.hopper_hop_mujoco.envs.contact_config import PRESET_CONTACT_CONFIGS

DEFAULT_SEED_DIRS = ("seed_20240201", "seed_20240202", "seed_20240203")


def assemble(contact: str, data_root: Path, seed_dirs: tuple[str, ...]) -> Path:
    if contact not in PRESET_CONTACT_CONFIGS:
        raise ValueError(f"unknown contact {contact!r}")
    collect_root = data_root / contact
    if not collect_root.is_dir():
        raise FileNotFoundError(f"collection dir missing: {collect_root}")
    for seed_dir in seed_dirs:
        if not (collect_root / seed_dir).is_dir():
            raise FileNotFoundError(f"missing seed dir: {collect_root / seed_dir}")
    output = collect_root / "hopperhop_koopman.npz"
    path = build(
        BuildConfig(
            collect_root=collect_root,
            output=output,
            seed_dirs=seed_dirs,
        )
    )
    with np.load(path, allow_pickle=True) as archive:
        print(
            f"dataset: {path}  transitions={len(archive['state']):,} "
            f"episodes(train/val/test)="
            f"{len(archive['train_episode_ids'])}/"
            f"{len(archive['validation_episode_ids'])}/"
            f"{len(archive['test_episode_ids'])}",
            flush=True,
        )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contacts",
        default="mujoco_default,mujoco_compliant,mujoco_hard",
        help="comma-separated contact configs",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("runs/hopper_hop_mujoco/data")
    )
    parser.add_argument(
        "--seed-dirs", default=",".join(DEFAULT_SEED_DIRS)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_dirs = tuple(args.seed_dirs.split(","))
    for contact in args.contacts.split(","):
        assemble(contact, args.data_root, seed_dirs)


if __name__ == "__main__":
    main()
