"""Frozen PyTorch inference for framework-neutral Koopman exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenKoopman(nn.Module):
    """A validated, immutable lift/linear/reconstruction module."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        path = path.resolve()
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("kind") != "playground_koopman_export_v1":
                raise ValueError("Unsupported Koopman export kind")
            architecture = metadata.get("architecture", {})
            if architecture.get("activation") != "silu":
                raise ValueError("O2O currently supports SiLU Koopman encoders only")
            encoder_count = int(metadata["encoder_layer_count"])
            arrays = {name: np.asarray(archive[name]) for name in archive.files}

        self.path = path
        self.sha256 = file_sha256(path)
        self.metadata = metadata
        self.state_dim = int(architecture["state_dim"])
        self.action_dim = int(architecture["action_dim"])
        self.lift_dim = int(architecture["lift_dim"])
        self.lifted_dim = self.state_dim + self.lift_dim
        self.register_buffer("A", torch.as_tensor(arrays["A"], dtype=torch.float32))
        self.register_buffer("B", torch.as_tensor(arrays["B"], dtype=torch.float32))
        self.register_buffer("C", torch.as_tensor(arrays["C"], dtype=torch.float32))
        self.register_buffer(
            "center", torch.as_tensor(arrays["center"], dtype=torch.float32)
        )
        self.register_buffer(
            "scale", torch.as_tensor(arrays["scale"], dtype=torch.float32)
        )
        self.encoder_weights = nn.ParameterList()
        self.encoder_biases = nn.ParameterList()
        for index in range(encoder_count):
            self.encoder_weights.append(
                nn.Parameter(
                    torch.as_tensor(
                        arrays[f"encoder_{index}_weight"], dtype=torch.float32
                    ),
                    requires_grad=False,
                )
            )
            self.encoder_biases.append(
                nn.Parameter(
                    torch.as_tensor(
                        arrays[f"encoder_{index}_bias"], dtype=torch.float32
                    ),
                    requires_grad=False,
                )
            )
        self._validate()
        self.requires_grad_(False)

    def _validate(self) -> None:
        if tuple(self.A.shape) != (self.lifted_dim, self.lifted_dim):
            raise ValueError("Koopman A has the wrong shape")
        if tuple(self.B.shape) != (self.lifted_dim, self.action_dim):
            raise ValueError("Koopman B has the wrong shape")
        if tuple(self.C.shape) != (self.state_dim, self.lifted_dim):
            raise ValueError("Koopman C has the wrong shape")
        if tuple(self.center.shape) != (self.state_dim,) or tuple(
            self.scale.shape
        ) != (self.state_dim,):
            raise ValueError("Koopman normalizer has the wrong shape")
        for tensor in self.parameters():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError("Koopman export contains NaN or Inf")
        for tensor in self.buffers():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError("Koopman export contains NaN or Inf")
        if torch.any(self.scale <= 0):
            raise ValueError("Koopman normalizer scale must be positive")

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self.center) / self.scale

    def denormalize(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return normalized_state * self.scale + self.center

    def lift_normalized(self, normalized_state: torch.Tensor) -> torch.Tensor:
        encoded = normalized_state
        for index, (weight, bias) in enumerate(
            zip(self.encoder_weights, self.encoder_biases, strict=True)
        ):
            encoded = F.linear(encoded, weight, bias)
            if index + 1 < len(self.encoder_weights):
                encoded = F.silu(encoded)
        return torch.cat((normalized_state, encoded), dim=-1)

    def lift(self, observation: torch.Tensor) -> torch.Tensor:
        return self.lift_normalized(self.normalize(observation))

    def step(self, lifted_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return F.linear(lifted_state, self.A) + F.linear(action, self.B)

    def reconstruct_normalized(self, lifted_state: torch.Tensor) -> torch.Tensor:
        return F.linear(lifted_state, self.C)

    def reconstruct(self, lifted_state: torch.Tensor) -> torch.Tensor:
        return self.denormalize(self.reconstruct_normalized(lifted_state))

    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "architecture": self.metadata["architecture"],
            "best_validation_rollout_normalized_mse": self.metadata.get(
                "best_validation_rollout_normalized_mse"
            ),
        }
