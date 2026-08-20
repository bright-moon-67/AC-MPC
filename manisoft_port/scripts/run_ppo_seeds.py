#!/usr/bin/env python
"""Sequential formal 5-seed launcher with isolated output directories."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["actor", "delta_ppo"], required=True)
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--backend", choices=["legacy", "modern"], default="legacy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()
    koopman_payload = torch.load(
        args.koopman_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    configured_seeds = koopman_payload["config"]["experiment"]["formal_seeds"]
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Formal run requires at least five unique seeds")
    if args.seeds == "0,1,2,3,4" and seeds != configured_seeds:
        raise ValueError(
            f"Checkpoint config formal_seeds={configured_seeds}, launcher defaults={seeds}"
        )
    script = "scripts/train_actor.py" if args.method == "actor" else "scripts/train_delta_ppo.py"
    output_root = Path(args.output_root)
    for seed in seeds:
        output = output_root / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=True)
        status_path = output / "training_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                status.get("stop_reason") == "total_timesteps"
                and int(status.get("seed", -1)) == seed
            ):
                print(f"Skipping completed seed {seed}: {status_path}", flush=True)
                continue
        resume = None
        if (output / "last.pt").exists():
            resume = output / "last.pt"
        elif (output / "emergency.pt").exists():
            resume = output / "emergency.pt"
        elif (output / "history.jsonl").exists():
            raise RuntimeError(
                f"{output} has history but no recoverable checkpoint; inspect it manually"
            )
        command = [
            sys.executable,
            script,
            "--koopman-checkpoint",
            args.koopman_checkpoint,
            "--output",
            str(output),
            "--backend",
            args.backend,
            "--device",
            args.device,
            "--seed",
            str(seed),
        ]
        if resume is not None:
            command.extend(["--resume", str(resume)])
        if args.total_timesteps is not None:
            command.extend(["--total-timesteps", str(args.total_timesteps)])
        console_path = output / "console.log"
        print(
            f"{'Resuming' if resume else 'Starting'} seed {seed}; "
            f"console={console_path}",
            flush=True,
        )
        with console_path.open("a", encoding="utf-8") as console:
            subprocess.run(
                command,
                check=True,
                stdout=console,
                stderr=subprocess.STDOUT,
            )
        if not status_path.exists():
            raise RuntimeError(f"Seed {seed} exited without {status_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("stop_reason") != "total_timesteps":
            raise RuntimeError(
                f"Seed {seed} did not complete: stop_reason={status.get('stop_reason')}"
            )


if __name__ == "__main__":
    main()
