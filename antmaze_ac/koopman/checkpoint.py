from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .model import DeepKoopman
from .visual_model import VisualLinearKoopman


KoopmanModel = DeepKoopman | VisualLinearKoopman

# format_version 3 adds: atomic writes (temp + os.replace), a ``history``
# field, and full resume state (optimizer + rng + epoch, optional scheduler /
# training_state) so training can be continued after an interruption.
# format_version 2 checkpoints still load.
FORMAT_VERSION = 3


def save_checkpoint(
    path: str | Path,
    model: KoopmanModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int,
    best_validation: float,
    config: dict[str, Any],
    normalizers: dict[str, Any],
    elapsed_seconds: float,
    rng_state: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "architecture": model.architecture(),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": int(epoch),
        "best_validation": float(best_validation),
        "config": config,
        "normalizers": normalizers,
        "elapsed_seconds": float(elapsed_seconds),
        "rng_state": rng_state,
        "history": history if history is not None else [],
        "training_state": training_state,
    }
    # A process interruption must never leave a partially written checkpoint
    # at a path that a resume script would trust.  Replace atomically after
    # torch has fully serialized the payload and flushed the temporary file.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )[1]
    )
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[KoopmanModel, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    architecture = dict(payload["architecture"])
    architecture_name = architecture.pop("architecture", None)
    if architecture_name in {None, "fullA_history_v2_adapted"}:
        # ``None`` keeps checkpoints written before the architecture tag was
        # introduced readable.
        model: KoopmanModel = DeepKoopman(**architecture)
    elif architecture_name == VisualLinearKoopman.ARCHITECTURE:
        model = VisualLinearKoopman(**architecture)
    else:
        raise ValueError(f"Unsupported Koopman architecture {architecture_name!r}")
    model.load_state_dict(payload["model"])
    return model, payload


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
