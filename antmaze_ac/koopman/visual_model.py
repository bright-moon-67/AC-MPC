from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    activations = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}") from exc


class VisualLinearKoopman(nn.Module):
    r"""Controlled linear dynamics for a robot/visual joint state.

    A frozen vision backbone is expected to run outside this module.  Its
    cached feature ``b`` is compressed to ``v=E(b)`` and concatenated with the
    normalized robot state ``r``:

    ``s=[r,v]``, ``z=T s``, ``z_next=A z+B u``, and ``s_hat=C z``.

    ``T`` is square, so the model does not increase state dimension.  The
    default identity mode is the minimal model ``z=s``.  ``learned`` retains
    the original independently-trained ``T``/``C`` ablation, while
    ``learned_inverse`` trains only ``T`` and reconstructs with an exact,
    differentiable solve ``C=T^-1``.  ``learned_orthogonal`` instead trains a
    generator ``S`` and derives ``T=exp(S-S^T)`` and ``C=T^T``.  This keeps the
    transform exactly invertible and well conditioned without a linear solve.
    A separate linear decoder reconstructs the frozen backbone feature from the
    visual part of ``s``.
    """

    ARCHITECTURE = "visual_linear_controlled_v1"

    def __init__(
        self,
        robot_dim: int,
        action_dim: int,
        visual_feature_dim: int = 512,
        visual_latent_dim: int = 16,
        encoder_hidden_dims: Sequence[int] = (256, 64),
        activation: str = "silu",
        transform_mode: str = "identity",
    ) -> None:
        super().__init__()
        dimensions_to_check = {
            "robot_dim": robot_dim,
            "action_dim": action_dim,
            "visual_feature_dim": visual_feature_dim,
            "visual_latent_dim": visual_latent_dim,
        }
        for name, value in dimensions_to_check.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        hidden_dims = tuple(map(int, encoder_hidden_dims))
        if any(dimension < 1 for dimension in hidden_dims):
            raise ValueError("encoder_hidden_dims must contain positive values")
        if transform_mode not in {
            "identity",
            "learned",
            "learned_inverse",
            "learned_orthogonal",
        }:
            raise ValueError(
                "transform_mode must be 'identity', 'learned', "
                "'learned_inverse', or 'learned_orthogonal'"
            )

        self.robot_dim = int(robot_dim)
        self.action_dim = int(action_dim)
        self.visual_feature_dim = int(visual_feature_dim)
        self.visual_latent_dim = int(visual_latent_dim)
        self.encoder_hidden_dims = hidden_dims
        self.activation_name = activation.lower()
        self.transform_mode = transform_mode

        # Keep the same public dimension vocabulary as DeepKoopman so that the
        # existing finite-horizon KoopmanMPCActor can consume A, B, the exported
        # readout matrix, and z.
        self.state_dim = self.robot_dim + self.visual_latent_dim
        self.lift_dim = 0
        self.lifted_dim = self.state_dim

        activation_type = _activation(activation)
        encoder_dimensions = [
            self.visual_feature_dim,
            *hidden_dims,
            self.visual_latent_dim,
        ]
        encoder_layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(encoder_dimensions[:-1], encoder_dimensions[1:])
        ):
            encoder_layers.append(nn.Linear(input_dim, output_dim))
            if index < len(encoder_dimensions) - 2:
                encoder_layers.append(activation_type())
        self.visual_encoder = nn.Sequential(*encoder_layers)

        # Cached ResNet features are centered before training, so a pure
        # matrix (without bias) is the intended reconstruction map.
        self.feature_decoder = nn.Linear(
            self.visual_latent_dim,
            self.visual_feature_dim,
            bias=False,
        )

        self.A = nn.Parameter(
            torch.eye(self.lifted_dim)
            + 0.001 * torch.randn(self.lifted_dim, self.lifted_dim)
        )
        self.B = nn.Parameter(torch.empty(self.lifted_dim, self.action_dim))
        nn.init.normal_(self.B, std=0.01)

        identity = torch.eye(self.state_dim)
        if transform_mode == "identity":
            self.register_buffer("T", identity.clone())
            self.register_buffer("C", identity.clone())
        elif transform_mode == "learned":
            self.T = nn.Parameter(identity.clone())
            self.C = nn.Parameter(identity.clone())
        elif transform_mode == "learned_inverse":
            # There is deliberately no independently stored C in this mode.
            # readout_matrix() always derives the current exact inverse, so it
            # cannot become stale after an optimizer update to T.
            self.T = nn.Parameter(identity.clone())
        else:
            # Only S is persistent.  T and C are differentiable derived
            # matrices, so a checkpoint can never contain a stale readout.
            self.S = nn.Parameter(torch.zeros_like(identity))

    def encode_visual(self, visual_feature: torch.Tensor) -> torch.Tensor:
        if visual_feature.shape[-1] != self.visual_feature_dim:
            raise ValueError(
                "Expected last visual-feature dimension "
                f"{self.visual_feature_dim}, got {visual_feature.shape[-1]}"
            )
        return self.visual_encoder(visual_feature)

    def decode_visual(self, visual_latent: torch.Tensor) -> torch.Tensor:
        if visual_latent.shape[-1] != self.visual_latent_dim:
            raise ValueError(
                "Expected last visual-latent dimension "
                f"{self.visual_latent_dim}, got {visual_latent.shape[-1]}"
            )
        return self.feature_decoder(visual_latent)

    def make_state(
        self,
        robot_state: torch.Tensor,
        visual_feature: torch.Tensor,
    ) -> torch.Tensor:
        if robot_state.shape[-1] != self.robot_dim:
            raise ValueError(
                f"Expected last robot dimension {self.robot_dim}, "
                f"got {robot_state.shape[-1]}"
            )
        if robot_state.shape[:-1] != visual_feature.shape[:-1]:
            raise ValueError("Robot state and visual feature batch shapes differ")
        visual_latent = self.encode_visual(visual_feature)
        return torch.cat((robot_state, visual_latent), dim=-1)

    def visual_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected last state dimension {self.state_dim}, "
                f"got {state.shape[-1]}"
            )
        return state[..., self.robot_dim :]

    def transform_matrix(self) -> torch.Tensor:
        """Return the current column-convention transform ``T``.

        The legacy modes return their historically stored ``T`` unchanged.
        Orthogonal mode materializes one matrix exponential; callers that use
        the transform repeatedly should pass the result through the optional
        ``transform`` arguments below.
        """

        if self.transform_mode == "learned_orthogonal":
            skew_generator = self.S - self.S.mT
            return torch.matrix_exp(skew_generator)
        return self.T

    def _resolve_transform(self, transform: torch.Tensor | None) -> torch.Tensor:
        transform = self.transform_matrix() if transform is None else transform
        if transform.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                "Transform must have shape "
                f"[{self.state_dim}, {self.state_dim}]"
            )
        return transform

    def lift(
        self,
        state: torch.Tensor,
        *,
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected last state dimension {self.state_dim}, "
                f"got {state.shape[-1]}"
            )
        transform = self._resolve_transform(transform)
        return state @ transform.mT

    def readout_matrix(
        self,
        *,
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the current matrix mapping lifted coordinates back to state.

        In ``learned_inverse`` mode this is computed with ``solve`` on every
        call rather than cached.  Call this *after* Koopman training when
        constructing an existing MPC module, which will clone and freeze it.
        The returned tensor remains differentiable with respect to ``T`` (or
        the orthogonal generator ``S``) when used inside a training computation.
        Orthogonal mode returns ``T^T``.
        """

        if self.transform_mode in {"identity", "learned"}:
            return self.C
        transform = self._resolve_transform(transform)
        if self.transform_mode == "learned_orthogonal":
            return transform.mT
        identity = torch.eye(
            self.state_dim,
            dtype=transform.dtype,
            device=transform.device,
        )
        return torch.linalg.solve(transform, identity)

    def reconstruct(
        self,
        lifted_state: torch.Tensor,
        *,
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lifted_state.shape[-1] != self.lifted_dim:
            raise ValueError(
                f"Expected last lifted dimension {self.lifted_dim}, "
                f"got {lifted_state.shape[-1]}"
            )
        if self.transform_mode == "learned_inverse":
            transform = self._resolve_transform(transform)
            # Column convention: z=T s, hence s=solve(T,z).  Adding a final
            # singleton dimension lets torch.linalg.solve broadcast over any
            # leading batch/time dimensions without materializing T^-1.
            return torch.linalg.solve(
                transform,
                lifted_state.unsqueeze(-1),
            ).squeeze(-1)
        if self.transform_mode == "learned_orthogonal":
            transform = self._resolve_transform(transform)
            # C=T^T, while row-batched reconstruction is s=z C^T=z T.
            return lifted_state @ transform
        return lifted_state @ self.C.mT

    def linear_step(
        self,
        lifted_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if lifted_state.shape[-1] != self.lifted_dim:
            raise ValueError("Wrong lifted-state dimension")
        if action.shape[-1] != self.action_dim:
            raise ValueError("Wrong action dimension")
        if lifted_state.shape[:-1] != action.shape[:-1]:
            raise ValueError("Lifted state and action batch shapes differ")
        return lifted_state @ self.A.mT + action @ self.B.mT

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transform = self.transform_matrix()
        following_lift = self.linear_step(
            self.lift(state, transform=transform),
            action,
        )
        return (
            self.reconstruct(following_lift, transform=transform),
            following_lift,
        )

    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        *,
        transform: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out an action-conditioned sequence without predicting actions."""

        if initial_state.shape[-1] != self.state_dim:
            raise ValueError("Wrong initial-state dimension")
        if actions.ndim < 2 or actions.shape[-1] != self.action_dim:
            raise ValueError("Actions must end in [horizon, action_dim]")
        if initial_state.shape[:-1] != actions.shape[:-2]:
            raise ValueError("Initial state and action-sequence batch shapes differ")
        if actions.shape[-2] < 1:
            raise ValueError("Rollout horizon must be positive")

        # Resolve once so learned_orthogonal performs one matrix exponential
        # for the entire horizon rather than one per reconstruction step.
        transform = self._resolve_transform(transform)
        lifted_state = self.lift(initial_state, transform=transform)
        predicted_states: list[torch.Tensor] = []
        predicted_lifts: list[torch.Tensor] = []
        for index in range(actions.shape[-2]):
            lifted_state = self.linear_step(
                lifted_state,
                actions[..., index, :],
            )
            predicted_lifts.append(lifted_state)
            predicted_states.append(
                self.reconstruct(lifted_state, transform=transform)
            )
        return (
            torch.stack(predicted_states, dim=-2),
            torch.stack(predicted_lifts, dim=-2),
        )

    def freeze_dynamics(self) -> "VisualLinearKoopman":
        self.requires_grad_(False)
        self.eval()
        return self

    def architecture(self) -> dict[str, int | list[int] | str]:
        return {
            "architecture": self.ARCHITECTURE,
            "robot_dim": self.robot_dim,
            "action_dim": self.action_dim,
            "visual_feature_dim": self.visual_feature_dim,
            "visual_latent_dim": self.visual_latent_dim,
            "encoder_hidden_dims": list(self.encoder_hidden_dims),
            "activation": self.activation_name,
            "transform_mode": self.transform_mode,
        }
