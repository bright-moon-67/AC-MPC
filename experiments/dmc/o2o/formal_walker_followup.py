"""Launch Koopman-dependent formal Walker methods after training completes.

This is a lightweight protected follow-up process, not a model-quality gate.
The formal launcher still checks the completed artifact's seed, dataset SHA,
dimensions, horizon, and reward-free contract before starting either method.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.dmc.o2o.formal_walker import FORMAL_TRAINING_SEEDS


STRUCTURED_METHODS = ("Cal-RLPD-Lift", "Cal-RLPD-KMPC")


def _koopman_complete(output_dir: Path) -> bool:
    run_path = output_dir / "run.json"
    best_path = output_dir / "best.npz"
    if not run_path.is_file() or not best_path.is_file():
        return False
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return run.get("completed") is True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-seed", type=int, choices=FORMAL_TRAINING_SEEDS, required=True
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    seed_dir = (args.run_root / f"seed_{args.training_seed}").resolve()
    koopman_dir = seed_dir / "koopman"
    while not _koopman_complete(koopman_dir):
        print(
            f"waiting for completed Koopman seed={args.training_seed}", flush=True
        )
        time.sleep(args.poll_seconds)

    koopman = koopman_dir / "best.npz"
    for method in STRUCTURED_METHODS:
        run_dir = seed_dir / method
        run_path = run_dir / "run.json"
        if run_path.is_file():
            try:
                if json.loads(run_path.read_text(encoding="utf-8")).get("completed"):
                    print(f"already completed: {method}", flush=True)
                    continue
            except json.JSONDecodeError:
                pass
        command = [
            sys.executable,
            "-m",
            "experiments.dmc.o2o.formal_walker",
            "--training-seed",
            str(args.training_seed),
            "--method",
            method,
            "--dataset",
            str(args.dataset.resolve()),
            "--koopman",
            str(koopman),
            "--run-root",
            str(args.run_root.resolve()),
            "--device",
            args.device,
            "--launch",
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
