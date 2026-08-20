"""Prepare and train one seed-specific formal Cartpole Koopman model."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from experiments.dmc.o2o.formal_cartpole import FORMAL_TRAINING_SEEDS, _atomic_json, validate_dataset
from experiments.dmc.o2o.koopman import file_sha256


def prepare_command(dataset: Path, prepared_data_dir: Path) -> list[str]:
    return [sys.executable, "-m", "experiments.dmc.o2o.prepare_koopman",
            "--dataset", str(dataset.resolve()), "--output-dir", str(prepared_data_dir.resolve())]


def training_command(*, training_seed: int, prepared_data_dir: Path,
                     output_dir: Path, python_executable: Path | None = None) -> list[str]:
    if training_seed not in FORMAL_TRAINING_SEEDS:
        raise ValueError(f"Training seed is outside {FORMAL_TRAINING_SEEDS}")
    executable = os.path.abspath(sys.executable if python_executable is None else python_executable)
    return [
        executable, "-m", "experiments.playground.train_koopman",
        "--task", "CartpoleSwingup", "--data-dir", str(prepared_data_dir.resolve()),
        "--output-dir", str(output_dir.resolve()), "--lift-dim", "10",
        "--k-step", "50", "--batch-size", "2048", "--max-windows", "500000",
        "--validation-windows", "10000", "--epochs", "500", "--patience", "40",
        "--learning-rate", "0.0003", "--spectral-radius-limit", "0.95",
        "--stability-reference-dt", "0.04", "--seed", str(training_seed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, choices=FORMAL_TRAINING_SEEDS, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jax-python", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    dataset = validate_dataset(args.dataset)
    prepare = prepare_command(dataset.path, args.prepared_data_dir)
    train = training_command(training_seed=args.training_seed,
                             prepared_data_dir=args.prepared_data_dir,
                             output_dir=args.output_dir,
                             python_executable=args.jax_python)
    print("prepare:", shlex.join(prepare), flush=True)
    print("train:", shlex.join(train), flush=True)
    if not args.launch:
        return
    manifest_path = args.prepared_data_dir.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("canonical_transitions_npz_sha256") != dataset.sha256:
        raise ValueError("Prepared Koopman data belongs to another offline dataset")
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cuda"
    probe = subprocess.run([os.path.abspath(args.jax_python), "-c",
                            "import jax; assert jax.default_backend() == 'gpu'; print(jax.devices())"],
                           cwd=Path(__file__).resolve().parents[3], env=environment,
                           check=False, capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError(f"Local JAX GPU preflight failed; no packages were changed. stderr={probe.stderr!r}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_name = f"cartpole_koopman_{args.training_seed}"
    if subprocess.run(["tmux", "has-session", "-t", session_name], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError(f"Protected tmux session already exists: {session_name}")
    log_path = output_dir / "train.log"
    repository = Path(__file__).resolve().parents[3]
    shell_command = (f"cd {shlex.quote(str(repository))} && exec env JAX_PLATFORMS=cuda "
                     f"XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 {shlex.join(train)} "
                     f">> {shlex.quote(str(log_path))} 2>&1")
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, shell_command], check=True)
    pane_pid = int(subprocess.check_output(
        ["tmux", "display-message", "-p", "-t", session_name, "#{pane_pid}"], text=True).strip())
    _atomic_json(output_dir / "launch.json", {
        "kind": "acmpc_protected_seed_koopman_tmux_launch_v1",
        "training_seed": args.training_seed, "tmux_session": session_name,
        "pane_pid": pane_pid, "command": train, "dataset_path": str(dataset.path),
        "dataset_sha256": dataset.sha256, "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": file_sha256(manifest_path),
        "jax_preflight": probe.stdout.strip(), "log_path": str(log_path),
        "launched_unix_seconds": time.time(),
    })
    print(f"launched tmux={session_name} pane_pid={pane_pid} log={log_path}", flush=True)


if __name__ == "__main__":
    main()
