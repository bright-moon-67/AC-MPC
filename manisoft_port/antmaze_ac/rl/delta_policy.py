from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Independent, Normal

from .critic import Critic


@dataclass
class DeltaPolicyOutput:
    distribution: Independent
    mean: torch.Tensor
    value: torch.Tensor


class DeltaPolicy(nn.Module):
    """Main neural baseline: augmented state -> incremental action."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_init: float = -1.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        dimensions = [state_dim, *hidden_dims, action_dim]
        activation_types = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
        if activation.lower() not in activation_types:
            raise ValueError(f"Unsupported activation {activation!r}")
        activation_type = activation_types[activation.lower()]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(dimensions) - 2:
                layers.append(activation_type())
        self.actor = nn.Sequential(*layers)
        self.critic = Critic(state_dim, hidden_dims, activation)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std.clamp_min(1e-6))

    def forward(self, observation: torch.Tensor) -> DeltaPolicyOutput:
        normalized = (observation - self.state_mean) / self.state_std
        mean = self.actor(normalized)
        distribution = Independent(Normal(mean, self.log_std.exp().expand_as(mean)), 1)
        return DeltaPolicyOutput(distribution, mean, self.critic(normalized))

    def act(self, observation: torch.Tensor, deterministic: bool = False):
        output = self(observation)
        action = output.mean if deterministic else output.distribution.sample()
        return action, output.distribution.log_prob(action), output.value

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        output = self(observations)
        return output.distribution.log_prob(actions), output.distribution.entropy(), output.value, output
