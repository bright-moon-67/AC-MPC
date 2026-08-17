"""Run the remaining Playground benchmark tasks after Cartpole completes.

For each task this lightweight supervisor runs the official tuned PPO,
collects one million complete transitions from each of early/mid/late policy
checkpoints, trains the task-scaled Koopman model, then launches KMPC, AB-PQ,
and AC-MPC-MPVE together.  Completed stages are skipped on supervisor restart.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from experiments.playground.tasks import TASKS
from experiments.playground.train_ppo import _atomic_json


DEFAULT_TASKS = ("ReacherHard", "HopperHop", "WalkerRun", "HumanoidRun")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_completed(path: Path) -> bool:
    value = _read_json(path)
    return bool(value and value.get("completed") is True)


def _wait_for_completed(path: Path, poll_seconds: float) -> None:
    while not _is_completed(path):
        print(f"waiting for prerequisite: {path}", flush=True)
        time.sleep(poll_seconds)


def _run_command(
    command: list[str],
    *,
    log_path: Path,
    environment: dict[str, str],
    completed: Callable[[], bool],
) -> None:
    if completed():
        print(f"skip completed stage: {' '.join(command)}", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"started pid={process.pid}: {' '.join(command)}", flush=True)
        returncode = process.wait()
    if returncode != 0 or not completed():
        raise RuntimeError(
            f"Stage failed with returncode={returncode}; see {log_path}"
        )


def _training_checkpoints(path: Path) -> tuple[Path, Path, Path]:
    candidates = sorted(
        (
            child
            for child in path.iterdir()
            if child.is_dir() and child.name.isdigit() and int(child.name) > 0
        ),
        key=lambda child: int(child.name),
    )
    if len(candidates) < 3:
        raise RuntimeError(f"Need at least three positive PPO checkpoints in {path}")
    selected = (candidates[0], candidates[len(candidates) // 2], candidates[-1])
    if len({int(item.name) for item in selected}) != 3:
        raise RuntimeError("Early/mid/late checkpoint selection is not distinct")
    return selected


def _evaluate_ppo(
    *,
    task: str,
    run_dir: Path,
    seed: int,
    environment: dict[str, str],
) -> None:
    output = run_dir / "eval_latest_128.json"
    command = [
        sys.executable,
        "-m",
        "experiments.playground.evaluate",
        "--task",
        task,
        "--method",
        "PPO",
        "--checkpoint",
        str(run_dir / "checkpoints"),
        "--episodes",
        "128",
        "--seed",
        str(seed + 101),
        "--output",
        str(output),
    ]
    _run_command(
        command,
        log_path=run_dir / "eval_latest.log",
        environment=environment,
        completed=output.is_file,
    )


def _run_task(
    task_name: str,
    *,
    root: Path,
    seed: int,
    environment: dict[str, str],
) -> dict[str, Any]:
    task = TASKS[task_name]
    train_root = root / "train" / task_name / f"seed_{seed}"
    ppo_dir = train_root / "PPO"
    ppo_command = [
        sys.executable,
        "-m",
        "experiments.playground.train_ppo",
        "--task",
        task_name,
        "--seed",
        str(seed),
        "--output-dir",
        str(ppo_dir),
    ]
    _run_command(
        ppo_command,
        log_path=ppo_dir / "process.log",
        environment=environment,
        completed=lambda: _is_completed(ppo_dir / "run.json"),
    )
    _evaluate_ppo(task=task_name, run_dir=ppo_dir, seed=seed, environment=environment)

    early, middle, late = _training_checkpoints(ppo_dir / "checkpoints")
    data_dir = root / "data" / task_name / f"seed_{seed}" / "full_3m"
    collect_command = [
        sys.executable,
        "-m",
        "experiments.playground.collect_koopman",
        "--task",
        task_name,
        "--checkpoint",
        str(early),
        "--checkpoint",
        str(middle),
        "--checkpoint",
        str(late),
        "--output-dir",
        str(data_dir),
        "--num-envs",
        "1000",
        "--episode-steps",
        str(task.episode_steps),
        "--seed",
        str(seed + 201),
    ]
    _run_command(
        collect_command,
        log_path=data_dir / "collect.log",
        environment=environment,
        completed=lambda: (_read_json(data_dir / "manifest.json") or {}).get(
            "total_transitions"
        )
        == 3_000_000,
    )

    koopman_dir = root / "koopman" / task_name / f"seed_{seed}" / "full_3m_jax"
    koopman_command = [
        sys.executable,
        "-m",
        "experiments.playground.train_koopman",
        "--task",
        task_name,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(koopman_dir),
        "--lift-dim",
        str(task.koopman_lift_dim),
        "--k-step",
        str(task.koopman_horizon_steps),
        "--epochs",
        "500",
        "--patience",
        "40",
        "--max-windows",
        "500000",
        "--validation-windows",
        "10000",
        "--batch-size",
        "2048",
        "--spectral-radius-limit",
        "0.95",
        "--seed",
        str(seed + 301),
    ]
    _run_command(
        koopman_command,
        log_path=koopman_dir / "train.log",
        environment=environment,
        completed=lambda: _is_completed(koopman_dir / "run.json"),
    )

    structured_command = [
        sys.executable,
        "-m",
        "experiments.playground.run_structured_after_koopman",
        "--task",
        task_name,
        "--koopman-run",
        str(koopman_dir),
        "--output-root",
        str(train_root),
        "--seed",
        str(seed),
        "--methods",
        "KMPC",
        "AB-PQ",
        "AC-MPC-MPVE",
    ]
    _run_command(
        structured_command,
        log_path=train_root / "supervisor.log",
        environment=environment,
        completed=lambda: _is_completed(train_root / "supervisor.json"),
    )
    return {
        "task": task_name,
        "seed": seed,
        "ppo": str(ppo_dir),
        "data": str(data_dir),
        "koopman": str(koopman_dir),
        "train": str(train_root),
        "checkpoint_steps": [int(item.name) for item in (early, middle, late)],
        "parameters": task.to_dict(),
        "completed": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    cartpole = root / "train" / "CartpoleSwingup" / f"seed_{args.seed}"
    _wait_for_completed(cartpole / "supervisor.json", args.poll_seconds)
    manifest_path = root / "pipeline" / f"remaining_seed_{args.seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": "mujoco_playground_remaining_tasks_pipeline_v1",
        "seed": args.seed,
        "tasks": list(args.tasks),
        "started_unix_seconds": time.time(),
        "task_runs": {},
        "completed": False,
    }
    _atomic_json(manifest_path, manifest)
    for task_name in args.tasks:
        manifest["task_runs"][task_name] = _run_task(
            task_name, root=root, seed=args.seed, environment=environment
        )
        _atomic_json(manifest_path, manifest)
    manifest["completed"] = True
    manifest["finished_unix_seconds"] = time.time()
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/playground"))
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(DEFAULT_TASKS))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
