from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .model import DeepKoopman, _activation


class HistoryDeepKoopman(DeepKoopman):
    """DeepKoopman with a finite state/action-history lifting encoder.

    The physical lifted prefix, full trainable ``A``, ``B`` and ``C`` readout
    are inherited unchanged from :class:`DeepKoopman`. The physical state is
    ``s_t``, the transition input is absolute ``u_t``, and the nonlinear
    lifting input is ``[s[t-H+1:t+1], u[t-H:t]]``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lift_dim: int = 32,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "silu",
        history_steps: int = 10,
    ) -> None:
        if history_steps < 1:
            raise ValueError("history_steps must be positive")
        super().__init__(state_dim, action_dim, lift_dim, hidden_dims, activation)
        self.history_steps = int(history_steps)
        self.context_dim = self.history_steps * (self.state_dim + self.action_dim)

        act = _activation(activation)
        dimensions = [self.context_dim, *map(int, hidden_dims), self.lift_dim]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.append(act())
        self.encoder = nn.Sequential(*layers)

    def lift(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[-1] != self.state_dim:
            raise ValueError(f"Expected last state dimension {self.state_dim}, got {x.shape[-1]}")
        if context is None:
            raise ValueError(
                "HistoryDeepKoopman.lift requires an aligned history context; "
                "use [s[t-H+1:t+1], u[t-H:t]]"
            )
        if context.shape[:-1] != x.shape[:-1] or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Expected context shape {tuple(x.shape[:-1]) + (self.context_dim,)}, "
                f"got {tuple(context.shape)}"
            )
        return torch.cat((x, self.encoder(context)), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_z = self.linear_step(self.lift(x, context), action)
        return self.reconstruct(next_z), next_z

    def rollout(
        self,
        x0: torch.Tensor,
        actions: torch.Tensor,
        initial_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.shape[-1] != self.action_dim:
            raise ValueError("Wrong action dimension")
        z = self.lift(x0, initial_context)
        states = []
        lifts = []
        for index in range(actions.shape[-2]):
            z = self.linear_step(z, actions[..., index, :])
            states.append(self.reconstruct(z))
            lifts.append(z)
        return torch.stack(states, dim=-2), torch.stack(lifts, dim=-2)

    def architecture(self) -> dict[str, int | list[int] | str]:
        architecture = super().architecture()
        architecture["architecture"] = "fullA_history_context_v1"
        architecture["history_steps"] = self.history_steps
        return architecture
