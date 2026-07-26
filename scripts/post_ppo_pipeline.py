#!/usr/bin/env python
"""Monitor formal PPO, recover its launcher, then run formal evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_is_alive(pid_path: Path, expected_command: str) -> bool:
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if len(stat_fields) > 2 and stat_fields[2] == "Z":
            return False
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        return expected_command.encode() in command
    except (OSError, ValueError):
        return False


def write_status(path: Path, stage: str, **details: object) -> None:
    payload = {
        "stage": stage,
        "updated_unix_seconds": time.time(),
        **details,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(payload, sort_keys=True), flush=True)


def completed_training_seeds(
    output_root: Path,
    seeds: list[int],
) -> tuple[list[int], list[dict]]:
    completed = []
    failures = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        status_path = seed_root / "training_status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("stop_reason") != "total_timesteps":
            failures.append({"seed": seed, **status})
            continue
        checkpoint = seed_root / "last.pt"
        if not checkpoint.exists():
            failures.append(
                {
                    "seed": seed,
                    "stop_reason": "missing_last_checkpoint",
                }
            )
            continue
        if sha256(checkpoint) != status.get("last_checkpoint_sha256"):
            failures.append(
                {
                    "seed": seed,
                    "stop_reason": "last_checkpoint_sha256_mismatch",
                }
            )
            continue
        completed.append(seed)
    return completed, failures


def evaluation_is_complete(
    output_root: Path,
    seeds: list[int],
    episodes: int,
    gain_update_interval: int,
) -> bool:
    reports = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        checkpoint = seed_root / "last.pt"
        report_path = seed_root / (
            f"evaluation_legacy_interval{gain_update_interval}_{episodes}ep.json"
        )
        if not checkpoint.exists() or not report_path.exists():
            return False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            int(report.get("episodes", 0)) < episodes
            or report.get("resolved_backend") != "legacy"
            or int(report.get("gain_update_interval", -1)) != gain_update_interval
            or report.get("checkpoint_sha256") != sha256(checkpoint)
        ):
            return False
        reports.append(report)
    aggregate_path = output_root / (
        f"aggregate_legacy_interval{gain_update_interval}_{episodes}ep.json"
    )
    if not aggregate_path.exists():
        return False
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    return (
        int(aggregate.get("seed_count", 0)) >= len(seeds)
        and sorted(int(seed) for seed in aggregate.get("training_seeds", []))
        == sorted(seeds)
        and all(
            int(value) >= episodes
            for value in aggregate.get("episodes_per_seed", [])
        )
        and len(reports) == len(seeds)
    )


def run(command: list[str], project_root: Path) -> None:
    print(json.dumps({"event": "run", "command": command}), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--ppo-output", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gain-update-interval", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--maximum-restarts", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.koopman_checkpoint).resolve()
    output_root = Path(args.ppo_output).resolve()
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Formal pipeline requires at least five unique seeds")
    if args.episodes < 100:
        raise ValueError("Formal evaluation requires at least 100 episodes per seed")

    status_path = output_root / "post_ppo_pipeline_status.json"
    training_pid_path = output_root / "formal_launcher.pid"
    evaluation_pid_path = output_root / "formal_evaluation.pid"
    training_restarts = 0
    evaluation_restarts = 0
    last_progress: tuple[int, ...] | None = None

    try:
        while True:
            completed, failures = completed_training_seeds(output_root, seeds)
            if failures:
                raise RuntimeError(f"Formal PPO reported failure: {failures}")
            progress = tuple(completed)
            if progress != last_progress:
                write_status(
                    status_path,
                    "waiting_for_formal_ppo",
                    completed_seeds=completed,
                    remaining_seeds=[seed for seed in seeds if seed not in completed],
                    training_restarts=training_restarts,
                )
                last_progress = progress
            if len(completed) == len(seeds):
                if process_is_alive(training_pid_path, "run_ppo_seeds.py"):
                    time.sleep(args.poll_seconds)
                    continue
                break
            if not process_is_alive(training_pid_path, "run_ppo_seeds.py"):
                if training_restarts >= args.maximum_restarts:
                    raise RuntimeError(
                        "Formal PPO launcher exited and exhausted automatic restarts"
                    )
                run(
                    [
                        str(project_root / "scripts/run_ppo_formal_detached.sh"),
                        "actor",
                        str(checkpoint),
                        str(output_root),
                        args.device,
                    ],
                    project_root,
                )
                training_restarts += 1
            time.sleep(args.poll_seconds)

        write_status(
            status_path,
            "formal_ppo_complete",
            completed_seeds=seeds,
            training_restarts=training_restarts,
        )
        while not evaluation_is_complete(
            output_root,
            seeds,
            args.episodes,
            args.gain_update_interval,
        ):
            if not process_is_alive(
                evaluation_pid_path,
                "evaluate_ppo_seeds.py",
            ):
                if evaluation_restarts >= args.maximum_restarts:
                    raise RuntimeError(
                        "Formal evaluation exited and exhausted automatic restarts"
                    )
                run(
                    [
                        str(project_root / "scripts/run_evaluation_formal_detached.sh"),
                        "actor",
                        str(output_root),
                        str(args.gain_update_interval),
                        args.device,
                    ],
                    project_root,
                )
                evaluation_restarts += 1
                write_status(
                    status_path,
                    "formal_evaluation_running",
                    evaluation_restarts=evaluation_restarts,
                    episodes_per_seed=args.episodes,
                )
            time.sleep(args.poll_seconds)

        aggregate = output_root / (
            f"aggregate_legacy_interval{args.gain_update_interval}_"
            f"{args.episodes}ep.json"
        )
        write_status(
            status_path,
            "complete",
            completed_seeds=seeds,
            episodes_per_seed=args.episodes,
            aggregate=str(aggregate),
            aggregate_sha256=sha256(aggregate),
            training_restarts=training_restarts,
            evaluation_restarts=evaluation_restarts,
        )
    except BaseException as error:
        write_status(
            status_path,
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            training_restarts=training_restarts,
            evaluation_restarts=evaluation_restarts,
        )
        raise


if __name__ == "__main__":
    main()
