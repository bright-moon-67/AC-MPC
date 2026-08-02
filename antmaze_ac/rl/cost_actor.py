from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    choices = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported actor activation {name!r}") from exc


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dimensions = [input_dim, *map(int, hidden_dims)]
    activation_type = _activation(activation)
    for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
        layers.extend((nn.Linear(in_dim, out_dim), activation_type()))
    layers.append(nn.Linear(dimensions[-1], output_dim))
    return nn.Sequential(*layers)


class CostActor(nn.Module):
    """Produce bounded local diagonal stage Hessian and linear cost."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        epsilon: float = 1e-4,
        q_max: float = 100.0,
        p_max: float = 10.0,
        previous_action_dim: int | None = None,
        previous_action_cost_scale: float = 1e-2,
        delta_action_cost_scale: float = 1e-3,
        activation: str = "gelu",
        observation_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.observation_dim = int(
            self.state_dim if observation_dim is None else observation_dim
        )
        if self.observation_dim < self.state_dim:
            raise ValueError("observation_dim must be at least state_dim")
        self.cost_dim = self.state_dim + self.action_dim
        self.epsilon = float(epsilon)
        self.q_max = float(q_max)
        self.p_max = float(p_max)
        self.network = _mlp(
            self.observation_dim,
            hidden_dims,
            2 * self.cost_dim,
            activation,
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        scale = torch.ones(self.cost_dim)
        if previous_action_dim:
            start = self.state_dim - int(previous_action_dim)
            scale[start : self.state_dim] = float(previous_action_cost_scale)
        scale[self.state_dim :] = float(delta_action_cost_scale)
        self.register_buffer("hessian_scale", scale)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_q, raw_p = self.network(state).chunk(2, dim=-1)
        q = (self.epsilon + (self.q_max - self.epsilon) * torch.sigmoid(raw_q)) * self.hessian_scale
        # The action Hessian must retain the strict positive lower bound even if
        # a custom state scale is supplied.
        q = torch.cat((q[..., : self.state_dim], q[..., self.state_dim :].clamp_min(self.epsilon)), dim=-1)
        # AC-MPC maps all raw cost outputs through a sigmoid. State/control
        # linear terms are then centered to allow both signs.
        p = (2.0 * torch.sigmoid(raw_p) - 1.0) * self.p_max
        return q, p
