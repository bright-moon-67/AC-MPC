from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    choices = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
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
    dimensions = [int(input_dim), *map(int, hidden_dims)]
    layers: list[nn.Module] = []
    activation_type = _activation(activation)
    for input_size, output_size in zip(dimensions[:-1], dimensions[1:]):
        layers.extend((nn.Linear(input_size, output_size), activation_type()))
    layers.append(nn.Linear(dimensions[-1], int(output_dim)))
    return nn.Sequential(*layers)


@dataclass
class KoopmanMPCActorOutput:
    """Output of the differentiable finite-horizon Koopman MPC head."""

    action: torch.Tensor
    action_sequence: torch.Tensor
    quadratic_diagonal: torch.Tensor
    linear_term: torch.Tensor
    qp_hessian: torch.Tensor
    qp_linear: torch.Tensor
    projected_gradient_residual: torch.Tensor


class KoopmanMPCActor(nn.Module):
    r"""Cost-map actor followed by constrained finite-horizon Koopman MPC.

    The frozen lifted dynamics are ``z[k+1] = A z[k] + B u[k]`` and the
    normalized physical readout is ``x[k] = C z[k]``.  At every control step,
    the cost-map network emits stage-dependent diagonal quadratic and signed
    linear terms for ``w[k] = [x[k+1], u[k]]``:

    ``sum_k 0.5*w[k]' diag(q[k]) w[k] + p[k]' w[k]``.

    The dynamics are condensed into a dense QP.  Fixed unrolled projected
    FISTA iterations make the action differentiable with respect to the cost
    network.  The projection enforces both the absolute muscle activation
    bounds and the soft-robot change-rate bound ``|u[k]-u[k-1]| <= max_delta``.
    It is a deterministic sequential projection, not an exact Euclidean
    projection or an exact active-set QP solver; ``projected_gradient_residual``
    is exposed so approximation quality can be monitored during training.
    """

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        *,
        horizon: int = 10,
        context_dim: int = 0,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "gelu",
        action_low: float | torch.Tensor = -0.3,
        action_high: float | torch.Tensor = 0.3,
        max_delta: float | torch.Tensor = 0.002,
        quadratic_log_scale: float = 1.5,
        linear_scale: float = 10.0,
        smoothness_weight: float = 10.0,
        solver_iterations: int = 20,
        step_fraction: float = 0.95,
        solver_epsilon: float = 1e-6,
        straight_through_projection: bool = True,
    ) -> None:
        super().__init__()
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        lifted_dim = int(A.shape[0])
        if B.ndim != 2 or B.shape[0] != lifted_dim:
            raise ValueError("B must have shape [lifted_dim, action_dim]")
        action_dim = int(B.shape[1])
        if C.ndim != 2 or C.shape[1] != lifted_dim:
            raise ValueError("C must have shape [physical_dim, lifted_dim]")
        physical_dim = int(C.shape[0])
        if horizon < 1 or solver_iterations < 1:
            raise ValueError("horizon and solver_iterations must be positive")
        if context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        if quadratic_log_scale <= 0 or linear_scale <= 0:
            raise ValueError("Cost-map scales must be positive")
        if smoothness_weight < 0:
            raise ValueError("smoothness_weight must be non-negative")
        if not 0 < step_fraction <= 1:
            raise ValueError("step_fraction must lie in (0, 1]")
        if solver_epsilon <= 0:
            raise ValueError("solver_epsilon must be positive")

        self.lifted_dim = lifted_dim
        self.physical_dim = physical_dim
        self.action_dim = action_dim
        self.augmented_dim = physical_dim + action_dim
        self.horizon = int(horizon)
        self.context_dim = int(context_dim)
        self.quadratic_log_scale = float(quadratic_log_scale)
        self.linear_scale = float(linear_scale)
        self.smoothness_weight = float(smoothness_weight)
        self.solver_iterations = int(solver_iterations)
        self.step_fraction = float(step_fraction)
        self.solver_epsilon = float(solver_epsilon)
        self.straight_through_projection = bool(straight_through_projection)

        self.register_buffer("A", A.detach().clone())
        self.register_buffer("B", B.detach().clone())
        self.register_buffer("C", C.detach().clone())

        low = torch.as_tensor(action_low, dtype=A.dtype, device=A.device)
        high = torch.as_tensor(action_high, dtype=A.dtype, device=A.device)
        delta = torch.as_tensor(max_delta, dtype=A.dtype, device=A.device)
        low = torch.broadcast_to(low, (action_dim,)).clone()
        high = torch.broadcast_to(high, (action_dim,)).clone()
        delta = torch.broadcast_to(delta, (action_dim,)).clone()
        if bool((low >= high).any()):
            raise ValueError("Every action lower bound must be below its upper bound")
        if not torch.isfinite(delta).all() or bool((delta <= 0).any()):
            raise ValueError("max_delta must be finite and positive")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("max_delta", delta)

        state_map, action_map = self._condense_physical_dynamics()
        self.register_buffer("state_map", state_map)
        self.register_buffer("action_map", action_map)
        self.register_buffer("difference", self._difference_matrix())

        output_dim = 2 * self.horizon * self.augmented_dim
        self.network = _mlp(
            lifted_dim + self.context_dim,
            hidden_dims,
            output_dim,
            activation,
        ).to(dtype=A.dtype, device=A.device)
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected a linear final cost-map layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def quadratic_lower_bound(self) -> float:
        return math.exp(-2.0 * self.quadratic_log_scale)

    @property
    def quadratic_upper_bound(self) -> float:
        return math.exp(2.0 * self.quadratic_log_scale)

    def _condense_physical_dynamics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``X=S*z0+T*U`` for ``X=[x1,...,xN]``."""

        lifted_power = torch.eye(
            self.lifted_dim,
            dtype=self.A.dtype,
            device=self.A.device,
        )
        lifted_action = self.A.new_zeros(
            self.lifted_dim,
            self.horizon * self.action_dim,
        )
        state_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        for step in range(self.horizon):
            lifted_power = self.A @ lifted_power
            lifted_action = self.A @ lifted_action
            columns = slice(
                step * self.action_dim,
                (step + 1) * self.action_dim,
            )
            lifted_action[:, columns] += self.B
            state_rows.append(self.C @ lifted_power)
            action_rows.append((self.C @ lifted_action).clone())
        return torch.cat(state_rows, dim=0), torch.cat(action_rows, dim=0)

    def _difference_matrix(self) -> torch.Tensor:
        """Map a flattened action plan to consecutive action differences."""

        if self.horizon == 1:
            return self.A.new_zeros(0, self.action_dim)
        difference = self.A.new_zeros(
            (self.horizon - 1) * self.action_dim,
            self.horizon * self.action_dim,
        )
        identity = torch.eye(
            self.action_dim,
            dtype=self.A.dtype,
            device=self.A.device,
        )
        for step in range(self.horizon - 1):
            rows = slice(
                step * self.action_dim,
                (step + 1) * self.action_dim,
            )
            current = slice(
                step * self.action_dim,
                (step + 1) * self.action_dim,
            )
            following = slice(
                (step + 1) * self.action_dim,
                (step + 2) * self.action_dim,
            )
            difference[rows, current] = -identity
            difference[rows, following] = identity
        return difference

    def cost_terms(
        self,
        lifted_state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lifted_state.shape[-1] != self.lifted_dim:
            raise ValueError("Wrong lifted-state dimension")
        batch_shape = lifted_state.shape[:-1]
        network_input = lifted_state
        if self.context_dim:
            if context is None or context.shape[-1] != self.context_dim:
                raise ValueError("Wrong or missing actor context dimension")
            if context.shape[:-1] != batch_shape:
                raise ValueError("Actor context batch shape does not match lifted state")
            network_input = torch.cat((lifted_state, context), dim=-1)
        elif context is not None:
            raise ValueError("This actor was constructed without context")

        raw = self.network(network_input).reshape(
            *batch_shape,
            2,
            self.horizon,
            self.augmented_dim,
        )
        raw_quadratic = torch.tanh(raw[..., 0, :, :])
        centered_log_weights = raw_quadratic - raw_quadratic.mean(
            dim=-1,
            keepdim=True,
        )
        quadratic = torch.exp(
            self.quadratic_log_scale * centered_log_weights
        )
        linear = self.linear_scale * torch.tanh(raw[..., 1, :, :])
        return quadratic, linear

    def condensed_quadratic(
        self,
        lifted_state: torch.Tensor,
        quadratic: torch.Tensor,
        linear: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``0.5 U' H U + f' U`` for the learned stage cost."""

        batch_shape = lifted_state.shape[:-1]
        expected = (*batch_shape, self.horizon, self.augmented_dim)
        if quadratic.shape != expected or linear.shape != expected:
            raise ValueError("Cost terms have the wrong shape")
        free_physical = lifted_state @ self.state_map.mT
        state_stop = self.physical_dim
        q_state = quadratic[..., :state_stop].reshape(
            *batch_shape,
            self.horizon * self.physical_dim,
        )
        q_action = quadratic[..., state_stop:].reshape(
            *batch_shape,
            self.horizon * self.action_dim,
        )
        p_state = linear[..., :state_stop].reshape(
            *batch_shape,
            self.horizon * self.physical_dim,
        )
        p_action = linear[..., state_stop:].reshape(
            *batch_shape,
            self.horizon * self.action_dim,
        )
        weighted_action_map = q_state.unsqueeze(-1) * self.action_map
        hessian = (
            self.action_map.mT @ weighted_action_map
            + torch.diag_embed(q_action)
        )
        if self.smoothness_weight and self.difference.numel():
            hessian = hessian + self.smoothness_weight * (
                self.difference.mT @ self.difference
            )
        qp_linear = (
            (q_state * free_physical + p_state).unsqueeze(-2)
            @ self.action_map
        ).squeeze(-2) + p_action
        return hessian, qp_linear

    def project_actions(
        self,
        flat_actions: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        """Apply absolute and sequential change-rate constraints."""

        if flat_actions.shape[-1] != self.horizon * self.action_dim:
            raise ValueError("Flat action plan has the wrong dimension")
        if previous_action.shape != (*flat_actions.shape[:-1], self.action_dim):
            raise ValueError("previous_action has the wrong shape")
        candidates = flat_actions.reshape(
            *flat_actions.shape[:-1],
            self.horizon,
            self.action_dim,
        )
        projected: list[torch.Tensor] = []
        preceding = torch.clamp(
            previous_action,
            min=self.action_low,
            max=self.action_high,
        )
        for step in range(self.horizon):
            lower = torch.maximum(self.action_low, preceding - self.max_delta)
            upper = torch.minimum(self.action_high, preceding + self.max_delta)
            hard_projection = torch.maximum(
                torch.minimum(candidates[..., step, :], upper),
                lower,
            )
            # A hard active constraint has a zero derivative with respect to
            # the unconstrained candidate.  During BC this makes an action
            # initialized on the wrong rate boundary effectively impossible to
            # move to the opposite boundary.  Keep the exact constrained value
            # in the forward pass while using the identity derivative in the
            # backward pass.  Evaluation remains the ordinary hard projection.
            following = (
                candidates[..., step, :]
                + (hard_projection - candidates[..., step, :]).detach()
                if self.training and self.straight_through_projection
                else hard_projection
            )
            projected.append(following)
            preceding = following
        return torch.stack(projected, dim=-2).flatten(start_dim=-2)

    def _solve_box_qp(
        self,
        hessian: torch.Tensor,
        linear: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Approximately solve the condensed constrained QP with FISTA."""

        lipschitz = hessian.abs().sum(dim=-1).amax(dim=-1)
        step = self.step_fraction / (lipschitz + self.solver_epsilon)
        repeated_previous = previous_action.unsqueeze(-2).expand(
            *previous_action.shape[:-1],
            self.horizon,
            self.action_dim,
        )
        current = self.project_actions(
            repeated_previous.flatten(start_dim=-2),
            previous_action,
        )
        extrapolated = current
        momentum = 1.0
        for _ in range(self.solver_iterations):
            gradient = (
                hessian @ extrapolated.unsqueeze(-1)
            ).squeeze(-1) + linear
            following = self.project_actions(
                extrapolated - step.unsqueeze(-1) * gradient,
                previous_action,
            )
            next_momentum = 0.5 * (
                1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)
            )
            extrapolated = following + (
                (momentum - 1.0) / next_momentum
            ) * (following - current)
            current = following
            momentum = next_momentum

        gradient = (hessian @ current.unsqueeze(-1)).squeeze(-1) + linear
        projected = self.project_actions(
            current - gradient,
            previous_action,
        )
        residual = torch.linalg.vector_norm(current - projected, dim=-1)
        return current, residual

    def forward(
        self,
        lifted_state: torch.Tensor,
        previous_action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> KoopmanMPCActorOutput:
        quadratic, linear = self.cost_terms(lifted_state, context)
        hessian, qp_linear = self.condensed_quadratic(
            lifted_state,
            quadratic,
            linear,
        )
        flat_action, residual = self._solve_box_qp(
            hessian,
            qp_linear,
            previous_action,
        )
        sequence = flat_action.reshape(
            *lifted_state.shape[:-1],
            self.horizon,
            self.action_dim,
        )
        return KoopmanMPCActorOutput(
            action=sequence[..., 0, :],
            action_sequence=sequence,
            quadratic_diagonal=quadratic,
            linear_term=linear,
            qp_hessian=hessian,
            qp_linear=qp_linear,
            projected_gradient_residual=residual,
        )
