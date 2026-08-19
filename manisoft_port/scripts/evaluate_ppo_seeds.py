#!/usr/bin/env python
"""Resume-safe formal legacy evaluation and cross-seed aggregation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--method", choices=["actor", "delta_ppo"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--backend", choices=["legacy", "modern"], default="legacy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gain-update-interval", type=int, default=1)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Formal evaluation requires at least five unique seeds")
    if args.method == "delta_ppo" and args.gain_update_interval != 1:
        raise ValueError("gain_update_interval only applies to actor checkpoints")

    input_root = Path(args.input_root)
    reports = []
    for seed in seeds:
        seed_root = input_root / f"seed_{seed}"
        checkpoint = seed_root / "last.pt"
        status_path = seed_root / "training_status.json"
        if not checkpoint.exists() or not status_path.exists():
            raise FileNotFoundError(f"Seed {seed} has no completed checkpoint/status")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("stop_reason") != "total_timesteps":
            raise RuntimeError(
                f"Seed {seed} is not complete: stop_reason={status.get('stop_reason')}"
            )
        report = seed_root / (
            f"evaluation_{args.backend}_interval"
            f"{args.gain_update_interval}_{args.episodes}ep.json"
        )
        checkpoint_digest = sha256(checkpoint)
        reusable = False
        if report.exists() and not args.force:
            previous = json.loads(report.read_text(encoding="utf-8"))
            reusable = (
                int(previous.get("episodes", 0)) >= args.episodes
                and previous.get("checkpoint_sha256") == checkpoint_digest
                and previous.get("resolved_backend") == args.backend
                and int(previous.get("gain_update_interval", 1))
                == args.gain_update_interval
            )
        if not reusable:
            command = [
                sys.executable,
                "scripts/evaluate_actor.py",
                "--checkpoint",
                str(checkpoint),
                "--method",
                args.method,
                "--episodes",
                str(args.episodes),
                "--backend",
                args.backend,
                "--device",
                args.device,
                "--gain-update-interval",
                str(args.gain_update_interval),
                "--plot-paths",
                "10",
                "--output",
                str(report),
            ]
            console_path = seed_root / "evaluation_console.log"
            with console_path.open("a", encoding="utf-8") as console:
                subprocess.run(
                    command,
                    check=True,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                )
        else:
            print(f"Reusing complete evaluation for seed {seed}: {report}", flush=True)
        reports.append(report)

    aggregate = input_root / (
        f"aggregate_{args.backend}_interval"
        f"{args.gain_update_interval}_{args.episodes}ep.json"
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_evaluations.py",
            *map(str, reports),
            "--minimum-episodes",
            str(args.episodes),
            "--output",
            str(aggregate),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
