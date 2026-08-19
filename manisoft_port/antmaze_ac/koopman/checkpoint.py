from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .model import DeepKoopman


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
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "architecture": model.architecture(),
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "epoch": int(epoch),
            "best_validation": float(best_validation),
            "config": config,
            "normalizers": normalizers,
            "elapsed_seconds": float(elapsed_seconds),
            "rng_state": rng_state,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[DeepKoopman, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    architecture = dict(payload["architecture"])
    architecture_name = architecture.pop("architecture", None)
    if architecture_name == "fullA_history_context_v1":
        from .history_model import HistoryDeepKoopman

        model = HistoryDeepKoopman(**architecture)
    else:
        model = DeepKoopman(**architecture)
    model.load_state_dict(payload["model"])
    return model, payload


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
