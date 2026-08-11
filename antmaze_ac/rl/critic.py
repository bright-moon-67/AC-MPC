from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (512, 512),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        activation_types = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation.lower() not in activation_types:
            raise ValueError(f"Unsupported critic activation {activation!r}")
        activation_type = activation_types[activation.lower()]
        layers: list[nn.Module] = []
        dimensions = [state_dim, *map(int, hidden_dims), 1]
        for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(dimensions) - 2:
                layers.append(activation_type())
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)
