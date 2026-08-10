from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    activations = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}") from exc


class DeepKoopman(nn.Module):
    """Identity-skip model with ``z_{t+1}=A z_t+B delta_u_t``."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lift_dim: int = 32,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if state_dim < 1 or action_dim < 1 or lift_dim < 1:
            raise ValueError("state_dim, action_dim and lift_dim must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.lift_dim = int(lift_dim)
        self.lifted_dim = self.state_dim + self.lift_dim
        act = _activation(activation)
        dimensions = [self.state_dim, *map(int, hidden_dims), self.lift_dim]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.append(act())
        self.encoder = nn.Sequential(*layers)

        # Adapted from fullA_history_v2: full trainable A initialized near
        # identity and small B. We use the prompt's column-vector convention
        # z_next=A z+B v, represented for batched row tensors below.
        self.A = nn.Parameter(
            torch.eye(self.lifted_dim) + 0.001 * torch.randn(self.lifted_dim, self.lifted_dim)
        )
        self.B = nn.Parameter(torch.empty(self.lifted_dim, self.action_dim))
        nn.init.normal_(self.B, std=0.01)
        self.register_buffer(
            "action_state_scale",
            torch.ones(self.action_dim),
            persistent=False,
        )

        reading = torch.zeros(self.state_dim, self.lifted_dim)
        reading[:, : self.state_dim] = torch.eye(self.state_dim)
        self.register_buffer("C", reading)

    def configure_action_integrator(
        self,
        action_state_std: torch.Tensor | Sequence[float],
    ) -> None:
        """Make the final state block obey ``u_t=u_{t-1}+delta_u_t``.

        States are normalized while delta actions remain in physical units,
        hence the exact B diagonal is ``1 / std(u)`` rather than one.
        """

        if self.state_dim < self.action_dim:
            raise ValueError(
                "state_dim must include the previous-action block"
            )
        scale = torch.as_tensor(
            action_state_std,
            dtype=self.A.dtype,
            device=self.A.device,
        )
        if scale.shape != (self.action_dim,):
            raise ValueError(
                "action_state_std must have shape "
                f"({self.action_dim},), got {tuple(scale.shape)}"
            )
        if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
            raise ValueError("action_state_std must be finite and positive")
        self.action_state_scale.copy_(scale)
        self.project_action_integrator()

    @torch.no_grad()
    def project_action_integrator(self) -> None:
        """Project the known previous-action rows of A and B exactly."""

        start = self.state_dim - self.action_dim
        rows = slice(start, self.state_dim)
        self.A[rows].zero_()
        self.B[rows].zero_()
        indices = torch.arange(self.action_dim, device=self.A.device)
        self.A[start + indices, start + indices] = 1.0
        self.B[start + indices, indices] = self.action_state_scale.reciprocal()

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.state_dim:
            raise ValueError(f"Expected last state dimension {self.state_dim}, got {x.shape[-1]}")
        return torch.cat((x, self.encoder(x)), dim=-1)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.C.mT

    def linear_step(self, z: torch.Tensor, delta_action: torch.Tensor) -> torch.Tensor:
        return z @ self.A.mT + delta_action @ self.B.mT

    def forward(self, x: torch.Tensor, delta_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_z = self.linear_step(self.lift(x), delta_action)
        return self.reconstruct(next_z), next_z

    def rollout(self, x0: torch.Tensor, delta_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out K transitions.

        Args:
            x0: ``[..., state_dim]``.
            delta_actions: ``[..., K, action_dim]``.
        Returns:
            Physical states ``[..., K, state_dim]`` and lifted states
            ``[..., K, lifted_dim]``.
        """

        if delta_actions.shape[-1] != self.action_dim:
            raise ValueError("Wrong action dimension")
        z = self.lift(x0)
        states = []
        lifts = []
        for index in range(delta_actions.shape[-2]):
            z = self.linear_step(z, delta_actions[..., index, :])
            states.append(self.reconstruct(z))
            lifts.append(z)
        return torch.stack(states, dim=-2), torch.stack(lifts, dim=-2)

    def freeze_dynamics(self) -> "DeepKoopman":
        self.requires_grad_(False)
        self.eval()
        return self

    def architecture(self) -> dict[str, int | list[int] | str]:
        hidden_dims = [
            layer.out_features
            for layer in self.encoder
            if isinstance(layer, nn.Linear)
        ][:-1]
        activation = next(
            (layer.__class__.__name__.lower() for layer in self.encoder if not isinstance(layer, nn.Linear)),
            "none",
        )
        return {
            "architecture": "fullA_history_v2_adapted",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lift_dim": self.lift_dim,
            "hidden_dims": hidden_dims,
            "activation": activation,
        }
