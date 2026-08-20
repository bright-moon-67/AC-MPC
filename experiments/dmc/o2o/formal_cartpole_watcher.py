"""Evaluate completed Cartpole methods, then archive and remove their .pt files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from experiments.dmc.o2o.formal_cartpole import (
    FORMAL_METHODS, FORMAL_OFFLINE_UPDATES, FORMAL_ONLINE_TRANSITIONS,
    FORMAL_TRAINING_SEEDS, _atomic_json, final_evaluation_commands,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_complete(run_dir: Path) -> bool:
    try:
        value = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        value.get("completed") is True
        and value.get("execution_scope") == "offline_to_online"
        and value.get("online_steps_completed") == FORMAL_ONLINE_TRANSITIONS
        and value.get("offline_updates_completed") in {0, FORMAL_OFFLINE_UPDATES}
    )


def archive_checkpoints(run_dir: Path) -> Path:
    archive = run_dir / "checkpoints.zip"
    manifest_path = run_dir / "checkpoints.archive.json"
    checkpoints = sorted(path for path in run_dir.rglob("*.pt") if path.is_file())
    if not checkpoints:
        if archive.is_file() and manifest_path.is_file():
            with zipfile.ZipFile(archive) as handle:
                if handle.testzip() is not None:
                    raise ValueError(f"Corrupt checkpoint archive: {archive}")
            return archive
        raise FileNotFoundError(f"No checkpoints to archive under {run_dir}")
    members = [{
        "name": str(path.relative_to(run_dir)), "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    } for path in checkpoints]
    temporary = archive.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
                         allowZip64=True) as handle:
        for path in checkpoints:
            handle.write(path, arcname=str(path.relative_to(run_dir)))
    with zipfile.ZipFile(temporary) as handle:
        if handle.testzip() is not None:
            raise ValueError(f"CRC verification failed for {temporary}")
        if sorted(handle.namelist()) != sorted(value["name"] for value in members):
            raise ValueError("Checkpoint archive member list differs from source files")
        for value in members:
            digest = hashlib.sha256(handle.read(value["name"])).hexdigest()
            if digest != value["sha256"]:
                raise ValueError(f"Checkpoint archive SHA mismatch: {value['name']}")
    os.replace(temporary, archive)
    _atomic_json(manifest_path, {
        "kind": "acmpc_checkpoint_zip_archive_v1", "archive": archive.name,
        "archive_sha256": _sha256(archive), "compression": "ZIP_DEFLATED level 9",
        "members": members, "source_bytes": sum(value["size_bytes"] for value in members),
        "archive_bytes": archive.stat().st_size,
        "sources_deleted_after_crc_and_sha256_verification": True,
    })
    for path in checkpoints:
        path.unlink()
    return archive


def process_method(run_dir: Path) -> bool:
    if not _training_complete(run_dir):
        return False
    for command in final_evaluation_commands(run_dir=run_dir):
        output = Path(command[command.index("--output") + 1])
        if output.is_file():
            continue
        log = output.with_suffix(".log")
        with log.open("a", encoding="utf-8") as handle:
            environment = dict(os.environ)
            environment["MUJOCO_GL"] = "egl"
            subprocess.run(
                command, check=True, stdout=handle, stderr=subprocess.STDOUT,
                env=environment,
            )
    archive_checkpoints(run_dir)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, choices=FORMAL_TRAINING_SEEDS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    seed_dir = (args.run_root / f"seed_{args.training_seed}").resolve()
    pending = set(FORMAL_METHODS)
    while pending:
        for method in tuple(sorted(pending)):
            if process_method(seed_dir / method):
                print(f"evaluated and archived: {method}", flush=True)
                pending.remove(method)
        if pending:
            print(f"waiting for methods: {','.join(sorted(pending))}", flush=True)
            time.sleep(args.poll_seconds)
    print("all seven methods evaluated and checkpoint-archived", flush=True)


if __name__ == "__main__":
    main()
