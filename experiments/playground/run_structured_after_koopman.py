"""Wait for a Koopman run, then train/evaluate structured Playground peers.

This is intentionally a small process supervisor, not an experiment approval
system.  It records the exact child commands and exit codes, starts every
method in its own session so SSH loss is harmless, and never overwrites an
existing method run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from experiments.playground.structured_networks import STRUCTURED_METHODS
from experiments.playground.tasks import TASKS
from experiments.playground.train_ppo import _atomic_json


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _wait_for_koopman(run_dir: Path, poll_seconds: float) -> Path:
    run_file = run_dir / "run.json"
    best = run_dir / "best.npz"
    while True:
        metadata = _read_json(run_file)
        if metadata and metadata.get("completed") is True:
            if not best.is_file():
                raise RuntimeError(f"Completed Koopman run has no best.npz: {run_dir}")
            return best.resolve()
        pid_file = run_dir / "pid"
        if pid_file.is_file():
            try:
                os.kill(int(pid_file.read_text().strip()), 0)
            except (ValueError, ProcessLookupError):
                raise RuntimeError(
                    f"Koopman process ended without a completed run: {run_dir}"
                )
            except PermissionError:
                pass
        print(f"waiting for Koopman completion: {run_dir}", flush=True)
        time.sleep(poll_seconds)


def _child_command(
    *, task: str, method: str, koopman: Path, output: Path, seed: int, timesteps: int | None
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.playground.train_structured",
        "--task",
        task,
        "--method",
        method,
        "--koopman",
        str(koopman),
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
    ]
    if timesteps is not None:
        command.extend(("--timesteps", str(timesteps)))
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    koopman = _wait_for_koopman(args.koopman_run.resolve(), args.poll_seconds)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    children: dict[str, tuple[subprocess.Popen[Any], Any, Path]] = {}
    manifest: dict[str, Any] = {
        "kind": "mujoco_playground_structured_supervisor_v1",
        "task": args.task,
        "seed": args.seed,
        "koopman": str(koopman),
        "methods": list(args.methods),
        "timesteps": args.timesteps,
        "started_unix_seconds": time.time(),
        "runs": {},
    }
    for method in args.methods:
        output = root / method
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run: {output}")
        output.mkdir(parents=True, exist_ok=True)
        command = _child_command(
            task=args.task,
            method=method,
            koopman=koopman,
            output=output,
            seed=args.seed,
            timesteps=args.timesteps,
        )
        log_handle = (output / "process.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children[method] = (process, log_handle, output)
        manifest["runs"][method] = {
            "command": command,
            "output": str(output),
            "pid": process.pid,
            "started_unix_seconds": time.time(),
        }
        _atomic_json(root / "supervisor.json", manifest)
        print(f"started {method}: pid={process.pid}", flush=True)
        if args.launch_stagger_seconds:
            time.sleep(args.launch_stagger_seconds)

    failed: list[str] = []
    for method, (process, log_handle, output) in children.items():
        returncode = process.wait()
        log_handle.close()
        run_metadata = _read_json(output / "run.json")
        completed = bool(run_metadata and run_metadata.get("completed") is True)
        manifest["runs"][method].update(
            returncode=returncode,
            completed=completed,
            finished_unix_seconds=time.time(),
        )
        _atomic_json(root / "supervisor.json", manifest)
        if returncode != 0 or not completed:
            failed.append(method)
            continue
        evaluation_output = output / "eval_latest_128.json"
        evaluation_log = output / "eval_latest.log"
        evaluation_command = [
            sys.executable,
            "-m",
            "experiments.playground.evaluate",
            "--task",
            args.task,
            "--method",
            method,
            "--checkpoint",
            str(output / "checkpoints"),
            "--koopman",
            str(koopman),
            "--episodes",
            "128",
            "--seed",
            str(args.eval_seed),
            "--output",
            str(evaluation_output),
        ]
        with evaluation_log.open("w", encoding="utf-8") as handle:
            evaluated = subprocess.run(
                evaluation_command,
                cwd=Path.cwd(),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        manifest["runs"][method].update(
            evaluation_command=evaluation_command,
            evaluation_returncode=evaluated.returncode,
            evaluation_output=str(evaluation_output),
        )
        _atomic_json(root / "supervisor.json", manifest)
        if evaluated.returncode != 0:
            failed.append(method)

    manifest["finished_unix_seconds"] = time.time()
    manifest["completed"] = not failed
    manifest["failed_methods"] = failed
    _atomic_json(root / "supervisor.json", manifest)
    if failed:
        raise RuntimeError(f"Structured methods failed: {failed}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--koopman-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=STRUCTURED_METHODS,
        default=list(STRUCTURED_METHODS),
    )
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--eval-seed", type=int, default=20260901)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--launch-stagger-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.timesteps is not None and args.timesteps < 1:
        parser.error("--timesteps must be positive")
    if args.poll_seconds <= 0 or args.launch_stagger_seconds < 0:
        parser.error("poll interval must be positive and stagger must be non-negative")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
