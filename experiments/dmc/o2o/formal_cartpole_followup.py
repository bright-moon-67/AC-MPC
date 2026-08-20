"""Launch Cartpole Lift/KMPC methods after the seed Koopman completes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.dmc.o2o.formal_cartpole import FORMAL_TRAINING_SEEDS


STRUCTURED_METHODS = ("Cal-RLPD-Lift", "Cal-RLPD-KMPC")


def _koopman_complete(output_dir: Path) -> bool:
    try:
        run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return run.get("completed") is True and (output_dir / "best.npz").is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, choices=FORMAL_TRAINING_SEEDS, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    seed_dir = (args.run_root / f"seed_{args.training_seed}").resolve()
    koopman_dir = seed_dir / "koopman"
    while not _koopman_complete(koopman_dir):
        print(f"waiting for completed Koopman seed={args.training_seed}", flush=True)
        time.sleep(args.poll_seconds)
    for method in STRUCTURED_METHODS:
        run_path = seed_dir / method / "run.json"
        try:
            if json.loads(run_path.read_text(encoding="utf-8")).get("completed"):
                print(f"already completed: {method}", flush=True)
                continue
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        subprocess.run([
            sys.executable, "-m", "experiments.dmc.o2o.formal_cartpole",
            "--training-seed", str(args.training_seed), "--method", method,
            "--dataset", str(args.dataset.resolve()),
            "--koopman", str(koopman_dir / "best.npz"),
            "--run-root", str(args.run_root.resolve()), "--device", args.device, "--launch",
        ], check=True)


if __name__ == "__main__":
    main()
