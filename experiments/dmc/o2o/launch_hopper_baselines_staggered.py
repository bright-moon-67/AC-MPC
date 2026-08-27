"""Submit the four remaining formal Hopper baseline seeds one hour apart."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASELINE_METHODS = ("Cal-RLPD", "Cal-QL", "RLPD", "AWAC", "IQL")
REMAINING_SEEDS = (20260852, 20260853, 20260854, 20260855)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _has_tmux_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _completed(run_dir: Path) -> bool:
    try:
        payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return payload.get("completed") is True


def _launch_seed(
    *, seed: int, dataset: Path, run_root: Path, repository: Path
) -> list[dict[str, Any]]:
    submitted: list[dict[str, Any]] = []
    for method in BASELINE_METHODS:
        run_dir = run_root / f"seed_{seed}" / method
        training_session = f"hopper_formal_{seed}_{method}".lower().replace("-", "_")
        archive_session = f"{training_session}_archive"
        already_completed = _completed(run_dir)
        if already_completed:
            training_action = "already_completed"
        elif _has_tmux_session(training_session):
            training_action = "already_running"
        else:
            command = [
                sys.executable,
                "-m",
                "experiments.dmc.o2o.formal_hopper",
                "--training-seed",
                str(seed),
                "--method",
                method,
                "--dataset",
                str(dataset),
                "--run-root",
                str(run_root),
                "--device",
                "cuda",
                "--launch",
            ]
            subprocess.run(command, cwd=repository, check=True)
            training_action = "launched"

        archive_path = run_dir / "checkpoints.zip"
        if archive_path.is_file():
            archive_action = "already_archived"
        elif _has_tmux_session(archive_session):
            archive_action = "already_watching"
        else:
            watcher = [
                sys.executable,
                "-m",
                "experiments.dmc.o2o.archive_checkpoints",
                "--run-dir",
                str(run_dir),
                "--watch",
                "--interval",
                "30",
            ]
            subprocess.run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    archive_session,
                    "-c",
                    str(repository),
                    shlex.join(watcher),
                ],
                check=True,
            )
            archive_action = "launched"
        event = {
            "seed": seed,
            "method": method,
            "run_dir": str(run_dir),
            "training_session": training_session,
            "training_action": training_action,
            "archive_session": archive_session,
            "archive_action": archive_action,
            "submitted_unix_seconds": time.time(),
        }
        submitted.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")

    repository = Path(__file__).resolve().parents[3]
    dataset = args.dataset.resolve()
    run_root = args.run_root.resolve()
    status_path = run_root / "staggered_baseline_submission_seed52_55.json"
    started = time.time()
    status: dict[str, Any] = {
        "kind": "acmpc_hopper_stand_staggered_baseline_submission_v1",
        "seeds": list(REMAINING_SEEDS),
        "methods": list(BASELINE_METHODS),
        "interval_seconds": args.interval_seconds,
        "first_submission_unix_seconds": started,
        "dataset": str(dataset),
        "submission_complete": False,
        "events": [],
    }
    _atomic_json(status_path, status)

    for index, seed in enumerate(REMAINING_SEEDS):
        scheduled = started + index * args.interval_seconds
        while True:
            remaining = scheduled - time.time()
            if remaining <= 0:
                break
            time.sleep(min(30.0, remaining))
        events = _launch_seed(
            seed=seed,
            dataset=dataset,
            run_root=run_root,
            repository=repository,
        )
        status["events"].extend(events)
        status["last_submitted_seed"] = seed
        _atomic_json(status_path, status)

    status["submission_complete"] = True
    status["submission_completed_unix_seconds"] = time.time()
    _atomic_json(status_path, status)
    print(f"All staggered seed submissions are complete: {status_path}", flush=True)


if __name__ == "__main__":
    main()
