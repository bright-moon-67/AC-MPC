"""Protected five-seed Hopper quality-mixed Koopman and structured O2O queue."""

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

from experiments.dmc.o2o.formal_hopper import (
    FORMAL_TRAINING_SEEDS,
    training_session_name,
    validate_dataset,
)
from experiments.dmc.o2o.formal_hopper_koopman import validate_corpus


SELECTION_KIND = "hopper_mixed_quality_balanced_v1"
METHODS = ("Cal-RLPD-KMPC", "Cal-RLPD-Lift")
OFFLINE_LEARNING_RATE = 1e-4
ONLINE_LEARNING_RATE = 3e-4


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _completed(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("completed") is True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _tmux_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _prepare_command(corpus: Path, prepared: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.dmc.o2o.prepare_koopman",
        "--dataset",
        str(corpus.resolve()),
        "--output-dir",
        str(prepared.resolve()),
    ]


def _koopman_train_command(seed: int, prepared: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.playground.train_koopman",
        "--task",
        "HopperStand",
        "--data-dir",
        str(prepared.resolve()),
        "--output-dir",
        str(output.resolve()),
        "--lift-dim",
        "48",
        "--k-step",
        "20",
        "--batch-size",
        "2048",
        "--max-windows",
        "500000",
        "--validation-windows",
        "10000",
        "--epochs",
        "500",
        "--patience",
        "40",
        "--learning-rate",
        "0.0003",
        "--stability-reference-dt",
        "0.04",
        "--seed",
        str(seed),
    ]


def _launch_method(
    *, seed: int, method: str, dataset: Path, koopman: Path, run_root: Path
) -> None:
    run_dir = run_root / f"seed_{seed}" / method
    session = training_session_name(run_root, seed, method)
    if not _completed(run_dir / "run.json") and not _tmux_exists(session):
        command = [
            sys.executable,
            "-m",
            "experiments.dmc.o2o.formal_hopper",
            "--training-seed",
            str(seed),
            "--method",
            method,
            "--dataset",
            str(dataset.resolve()),
            "--koopman",
            str(koopman.resolve()),
            "--koopman-selection-kind",
            SELECTION_KIND,
            "--offline-actor-learning-rate",
            str(OFFLINE_LEARNING_RATE),
            "--offline-critic-learning-rate",
            str(OFFLINE_LEARNING_RATE),
            "--run-root",
            str(run_root.resolve()),
            "--device",
            "cuda",
            "--launch",
        ]
        subprocess.run(command, check=True)

    watcher = f"archive_{session}"
    if not (run_dir / "checkpoints.zip").is_file() and not _tmux_exists(watcher):
        watch_command = [
            sys.executable,
            "-m",
            "experiments.dmc.o2o.archive_checkpoints",
            "--run-dir",
            str(run_dir.resolve()),
            "--watch",
            "--interval",
            "30",
        ]
        watch_log = run_dir / "archive.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        shell_command = (
            f"{shlex.join(watch_command)} >> {shlex.quote(str(watch_log))} 2>&1"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", watcher, shell_command],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--koopman-corpus", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stagger-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    if args.stagger_seconds < 0:
        parser.error("--stagger-seconds must be non-negative")

    dataset = validate_dataset(args.dataset).path
    corpus = validate_corpus(args.koopman_corpus, SELECTION_KIND).path
    run_root = args.run_root.resolve()
    prepared = run_root / "koopman_prepared" / "shared"
    state_path = run_root / "campaign_state.json"
    fresh_state: dict[str, Any] = {
        "kind": "hopper_qualitymix_structured_five_seed_campaign_v1",
        "seeds": list(FORMAL_TRAINING_SEEDS),
        "methods": list(METHODS),
        "dataset": str(dataset),
        "koopman_corpus": str(corpus),
        "koopman_selection_kind": SELECTION_KIND,
        "offline_actor_learning_rate": OFFLINE_LEARNING_RATE,
        "offline_critic_learning_rate": OFFLINE_LEARNING_RATE,
        "online_actor_learning_rate": ONLINE_LEARNING_RATE,
        "online_critic_learning_rate": ONLINE_LEARNING_RATE,
        "stagger_seconds": args.stagger_seconds,
        "started_unix_seconds": time.time(),
        "seed_status": {},
    }
    try:
        state = json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = fresh_state
    else:
        immutable = (
            "kind",
            "seeds",
            "methods",
            "dataset",
            "koopman_corpus",
            "koopman_selection_kind",
            "offline_actor_learning_rate",
            "offline_critic_learning_rate",
            "online_actor_learning_rate",
            "online_critic_learning_rate",
        )
        mismatches = {
            key: {"saved": state.get(key), "requested": fresh_state[key]}
            for key in immutable
            if state.get(key) != fresh_state[key]
        }
        if mismatches:
            raise ValueError(f"Existing campaign state differs: {mismatches}")
        if not isinstance(state.get("seed_status"), dict):
            raise ValueError("Existing campaign seed_status is invalid")
        # Scheduling cadence is intentionally mutable on a protected resume.
        state["stagger_seconds"] = args.stagger_seconds
        state.pop("all_methods_submitted", None)
        state.pop("submission_completed_unix_seconds", None)
    _atomic_json(state_path, state)
    _run_logged(
        _prepare_command(corpus, prepared), run_root / "koopman_prepared" / "prepare.log"
    )

    previous_launch_time: float | None = None
    for seed in FORMAL_TRAINING_SEEDS:
        model_dir = run_root / "koopman" / f"seed_{seed}"
        model = model_dir / "best.npz"
        if not (_completed(model_dir / "run.json") and model.is_file()):
            if model_dir.exists() and any(model_dir.iterdir()):
                retry = model_dir.with_name(
                    f"{model_dir.name}_incomplete_{int(time.time())}"
                )
                os.replace(model_dir, retry)
            _run_logged(
                _koopman_train_command(seed, prepared, model_dir),
                run_root / "koopman" / f"seed_{seed}.log",
            )
        if not (_completed(model_dir / "run.json") and model.is_file()):
            raise RuntimeError(f"Koopman seed {seed} did not complete")

        saved_seed = state["seed_status"].get(str(seed))
        if isinstance(saved_seed, dict) and saved_seed.get(
            "methods_submitted"
        ) == list(METHODS):
            for method in METHODS:
                _launch_method(
                    seed=seed,
                    method=method,
                    dataset=dataset,
                    koopman=model,
                    run_root=run_root,
                )
            submitted = saved_seed.get("submitted_unix_seconds")
            if not isinstance(submitted, (int, float)):
                raise ValueError(f"Seed {seed} has an invalid submission timestamp")
            previous_launch_time = float(submitted)
            continue

        if previous_launch_time is not None:
            remaining = previous_launch_time + args.stagger_seconds - time.time()
            if remaining > 0:
                time.sleep(remaining)
        for method in METHODS:
            _launch_method(
                seed=seed,
                method=method,
                dataset=dataset,
                koopman=model,
                run_root=run_root,
            )
        previous_launch_time = time.time()
        state["seed_status"][str(seed)] = {
            "koopman_completed": True,
            "koopman_model": str(model.resolve()),
            "methods_submitted": list(METHODS),
            "submitted_unix_seconds": previous_launch_time,
        }
        _atomic_json(state_path, state)

    state["all_methods_submitted"] = True
    state["submission_completed_unix_seconds"] = time.time()
    _atomic_json(state_path, state)


if __name__ == "__main__":
    main()
