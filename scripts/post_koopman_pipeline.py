#!/usr/bin/env python
"""Wait for formal Koopman, run hard gates, then launch formal Actor-Critic PPO."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_is_alive(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if len(stat_fields) > 2 and stat_fields[2] == "Z":
            return False
    except (OSError, ValueError):
        return False
    return True


def require_finite_koopman_report(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("horizons"):
        raise RuntimeError("Koopman evaluation has no horizon metrics")
    for horizon, metrics in report["horizons"].items():
        if int(metrics.get("samples", 0)) < 1:
            raise RuntimeError(f"Koopman horizon {horizon} has no samples")
        values = [
            *metrics["koopman_mse"].values(),
            *metrics["naive_hold_mse"].values(),
            *metrics["per_step_all_mse"],
        ]
        if not all(math.isfinite(float(value)) for value in values):
            raise FloatingPointError(
                f"Koopman evaluation horizon {horizon} contains NaN or Inf"
            )


def write_status(path: Path, stage: str, **details) -> None:
    payload = {
        "stage": stage,
        "updated_unix_seconds": time.time(),
        **details,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(payload, sort_keys=True), flush=True)


def run(command: list[str], project_root: Path) -> None:
    print(json.dumps({"event": "run", "command": command}), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--koopman-output",
        default="runs/antmaze_umaze_fulla_formal",
    )
    parser.add_argument(
        "--ppo-output",
        default="runs/antmaze_umaze_formal/actor",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-restarts", type=int, default=3)
    parser.add_argument("--legacy-wait-hours", type=float, default=3.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    koopman_output = (project_root / args.koopman_output).resolve()
    koopman_root = koopman_output / "koopman"
    status_path = koopman_root / "post_koopman_pipeline_status.json"
    training_status_path = koopman_root / "training_status.json"
    training_pid_path = koopman_root / "formal_train.pid"
    restarts = 0
    absent_polls = 0
    write_status(status_path, "waiting_for_koopman", restarts=restarts)

    try:
        while True:
            training_complete = training_status_path.exists()
            training_alive = process_is_alive(training_pid_path)
            if training_complete and not training_alive:
                break
            if training_alive:
                absent_polls = 0
            else:
                absent_polls += 1
                if absent_polls >= 2 and not training_complete:
                    if restarts >= args.maximum_restarts:
                        raise RuntimeError(
                            "Koopman exited without training_status.json and exhausted "
                            f"{args.maximum_restarts} automatic restarts"
                        )
                    run(
                        [
                            str(project_root / "scripts/run_koopman_detached.sh"),
                            str(koopman_output),
                        ],
                        project_root,
                    )
                    restarts += 1
                    absent_polls = 0
                    write_status(
                        status_path,
                        "waiting_for_koopman",
                        restarts=restarts,
                        note="unexpected exit recovered from newest checkpoint",
                    )
            time.sleep(args.poll_seconds)

        training_status = json.loads(
            training_status_path.read_text(encoding="utf-8")
        )
        if training_status.get("stop_reason") not in {
            "max_wall_time",
            "max_epochs",
            "user_requested",
        }:
            raise RuntimeError(
                "Formal Koopman ended with unexpected stop_reason="
                f"{training_status.get('stop_reason')!r}"
            )
        best_checkpoint = koopman_root / "best_validation.pt"
        last_checkpoint = koopman_root / "last.pt"
        if sha256(best_checkpoint) != training_status["best_checkpoint_sha256"]:
            raise RuntimeError("best_validation.pt SHA256 does not match training status")
        if sha256(last_checkpoint) != training_status["last_checkpoint_sha256"]:
            raise RuntimeError("last.pt SHA256 does not match training status")
        write_status(
            status_path,
            "koopman_complete",
            restarts=restarts,
            stop_reason=training_status["stop_reason"],
            best_epoch=training_status["best_epoch"],
            best_validation=training_status["best_validation"],
        )

        write_status(status_path, "evaluating_koopman")
        run(
            [
                sys.executable,
                "scripts/evaluate_koopman.py",
                "--checkpoint",
                str(best_checkpoint),
                "--data",
                "data/processed/antmaze-umaze-v2",
                "--split",
                "test",
                "--device",
                "cuda",
                "--output",
                str(koopman_root / "evaluation_test.json"),
            ],
            project_root,
        )
        require_finite_koopman_report(koopman_root / "evaluation_test.json")

        write_status(status_path, "waiting_for_legacy_cuda")
        legacy_deadline = time.monotonic() + args.legacy_wait_hours * 3600.0
        legacy_probe = [
            str(project_root / "scripts/run_legacy.sh"),
            "python",
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(), 'legacy torch has no CUDA'; "
                "print(torch.__version__)"
            ),
        ]
        while True:
            probe = subprocess.run(
                legacy_probe,
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if probe.returncode == 0:
                break
            if time.monotonic() >= legacy_deadline:
                raise RuntimeError("Timed out waiting for CUDA PyTorch in antmaze_legacy")
            time.sleep(args.poll_seconds)

        write_status(status_path, "checking_legacy_environment")
        run(
            [
                str(project_root / "scripts/run_legacy.sh"),
                "python",
                "scripts/check_legacy_env.py",
                "--output",
                "runs/legacy_env_check.json",
            ],
            project_root,
        )

        write_status(status_path, "checking_fixed_lqr")
        run(
            [
                str(project_root / "scripts/run_legacy.sh"),
                "python",
                "scripts/test_fixed_lqr.py",
                "--koopman-checkpoint",
                str(best_checkpoint),
                "--backend",
                "legacy",
                "--episodes",
                "1",
                "--gain-update-interval",
                "1",
                "--device",
                "cuda",
                "--output",
                str(koopman_root / "fixed_lqr_legacy_interval1.json"),
            ],
            project_root,
        )

        write_status(status_path, "launching_formal_ppo")
        ppo_output = (project_root / args.ppo_output).resolve()
        run(
            [
                str(project_root / "scripts/run_ppo_formal_detached.sh"),
                "actor",
                str(best_checkpoint),
                str(ppo_output),
                "cuda",
            ],
            project_root,
        )
        ppo_pid_path = ppo_output / "formal_launcher.pid"
        if not process_is_alive(ppo_pid_path):
            raise RuntimeError("Formal PPO launcher did not remain alive after startup")
        write_status(
            status_path,
            "formal_ppo_launched",
            ppo_output=str(ppo_output),
            ppo_launcher_pid=int(ppo_pid_path.read_text(encoding="utf-8")),
        )
    except BaseException as error:
        write_status(
            status_path,
            "failed_before_ppo",
            error_type=type(error).__name__,
            error=str(error),
            restarts=restarts,
        )
        raise


if __name__ == "__main__":
    main()
