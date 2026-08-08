from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

from .model import DeepKoopman

# format_version 3 adds: atomic writes (temp + os.replace), a ``history``
# field, and full resume state (optimizer + rng + epoch) so training can be
# continued after an interruption. format_version 2 checkpoints still load.
FORMAT_VERSION = 3


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically: temp file in the same dir, then rename.

    Prevents a half-written/corrupt checkpoint if the process is killed
    mid-save (a crash leaves only the .tmp file, never a broken final one).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path: str | Path,
    model: DeepKoopman,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int,
    best_validation: float,
    config: dict[str, Any],
    normalizers: dict[str, Any],
    elapsed_seconds: float,
    rng_state: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "architecture": model.architecture(),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(epoch),
        "best_validation": float(best_validation),
        "config": config,
        "normalizers": normalizers,
        "elapsed_seconds": float(elapsed_seconds),
        "rng_state": rng_state,
        "history": history if history is not None else [],
    }
    _atomic_save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[DeepKoopman, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    architecture = dict(payload["architecture"])
    architecture.pop("architecture", None)
    model = DeepKoopman(**architecture)
    model.load_state_dict(payload["model"])
    return model, payload


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
