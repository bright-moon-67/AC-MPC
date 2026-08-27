"""Archive completed O2O checkpoints and remove redundant loose copies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def archive(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    checkpoints = sorted(run_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    output = run_dir / "checkpoints.zip"
    temporary = run_dir / ".checkpoints.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for path in checkpoints:
            zf.write(path, arcname=path.name)
    temporary.replace(output)
    for path in checkpoints:
        path.unlink()
    return output


def _final_evaluate(run_dir: Path) -> None:
    for checkpoint in ("online_000000", "online_020000"):
        output = run_dir / f"evaluation_10x10_{checkpoint}.json"
        if output.exists():
            continue
        command = [
            sys.executable,
            "-m",
            "experiments.dmc.o2o.evaluate_10x10",
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            checkpoint,
            "--device",
            "cpu",
            "--parallel-workers",
            "10",
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True)


def watch(run_dir: Path, interval: float, final_evaluation: bool) -> None:
    run_dir = run_dir.resolve()
    while True:
        status_path = run_dir / "run.json"
        try:
            status = json.loads(status_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(interval)
            continue
        if status.get("completed") is True:
            output = run_dir / "checkpoints.zip"
            if not output.exists():
                if final_evaluation:
                    _final_evaluate(run_dir)
                print(f"archiving {run_dir}", flush=True)
                archive(run_dir)
            return
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--skip-final-evaluation", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    if args.watch:
        watch(args.run_dir, args.interval, not args.skip_final_evaluation)
    else:
        print(archive(args.run_dir))


if __name__ == "__main__":
    main()
