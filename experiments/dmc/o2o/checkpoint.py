"""Atomic, identity-bound checkpoints for offline-to-online runs."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_KIND = "acmpc_dmc_o2o_checkpoint_v1"


def rng_state(generator: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_generator": generator.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any], generator: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    generator.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError("Unsupported O2O checkpoint")
    return payload
