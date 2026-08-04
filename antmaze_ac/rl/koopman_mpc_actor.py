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
    """Result of the finite-horizon, box-constrained Koopman MPC head."""

    action: torch.Tensor
    action_sequence: torch.Tensor
    quadratic_diagonal: torch.Tensor
    linear_term: torch.Tensor
    qp_hessian: torch.Tensor
    qp_linear: torch.Tensor
    projected_gradient_residual: torch.Tensor


class KoopmanMPCActor(nn.Module):
    r"""Cost-map actor with a constrained finite-horizon Koopman MPC layer.

    The frozen Koopman dynamics and physical readout are

    ``z[t+1] = A z[t] + B u[t]`` and ``x[t] = C z[t]``.

    A shallow network conditioned directly on the current Koopman lift emits
    stage-varying diagonal quadratic and signed linear terms for the normalized
    physical vector ``w[t]=[x[t+1], u[t]]``:

    ``sum(t=0..N-1) .5*w[t]' diag(q[t]) w[t] + p[t]' w[t]``.

    Thus an ``N=2`` PandaReach actor emits
    ``2 * N * (17 physical + 7 action) = 96`` scalars.  Koopman latent
    coordinates are used for prediction but are not directly penalized.  The
    action block of ``q`` is the learned diagonal ``R``; no additional fixed
    ``R`` is used.  Strictly positive ``q`` makes the condensed Hessian
    positive definite.  Its geometric mean is fixed separately at every
    stage as an additional BC stabilization constraint; this removes
    per-stage quadratic scales but also restricts cross-stage cost weighting
    and is not the parameterization used by the original AC-MPC paper.  The
    signed ``p`` is bounded separately.

    The dynamics are condensed into a dense box QP.  Fixed unrolled projected
    FISTA iterations enforce ``action_low <= u <= action_high`` and are
    differentiable almost everywhere.  This is not an exact active-set or
    interior-point QP solve and not KKT implicit differentiation: finite
    iterations can leave a nonzero residual, gradients depend on iteration
    count, and projection has zero derivative outside active bounds.

    Cost is placed on predicted ``x[t+1]`` rather than the fixed current
    ``x[t]`` because the latter is constant with respect to the MPC decision.
    There is no extra terminal value in this first implementation; the second
    predicted stage is the terminal planning state when ``N=2``.
    """

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        *,
        horizon: int = 2,
        context_dim: int = 0,
        hidden_dims: Sequence[int] = (128,),
        activation: str = "gelu",
        action_low: float | torch.Tensor = -1.0,
        action_high: float | torch.Tensor = 1.0,
        quadratic_log_scale: float = 1.5,
        linear_scale: float = 10.0,
        solver_iterations: int = 20,
        step_fraction: float = 0.95,
        solver_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        lifted_dim = A.shape[0]
        if B.ndim != 2 or B.shape[0] != lifted_dim:
            raise ValueError("B must have shape [lifted_dim, action_dim]")
        action_dim = B.shape[1]
        if C.ndim != 2 or C.shape[1] != lifted_dim:
            raise ValueError("C must have shape [physical_dim, lifted_dim]")
        physical_dim = C.shape[0]
        if horizon < 1 or solver_iterations < 1:
            raise ValueError("horizon and solver_iterations must be positive")
        if context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        if quadratic_log_scale <= 0 or linear_scale <= 0:
            raise ValueError("Cost-map scales must be positive")
        if not 0 < step_fraction <= 1:
            raise ValueError("step_fraction must lie in (0, 1]")
        if solver_epsilon <= 0:
            raise ValueError("solver_epsilon must be positive")

        self.lifted_dim = int(lifted_dim)
        self.physical_dim = int(physical_dim)
        self.action_dim = int(action_dim)
        self.augmented_dim = self.physical_dim + self.action_dim
        self.horizon = int(horizon)
        self.context_dim = int(context_dim)
        self.quadratic_log_scale = float(quadratic_log_scale)
        self.linear_scale = float(linear_scale)
        self.solver_iterations = int(solver_iterations)
        self.step_fraction = float(step_fraction)
        self.solver_epsilon = float(solver_epsilon)

        self.register_buffer("A", A.detach().clone())
        self.register_buffer("B", B.detach().clone())
        self.register_buffer("C", C.detach().clone())

        low = torch.as_tensor(action_low, dtype=A.dtype, device=A.device)
        high = torch.as_tensor(action_high, dtype=A.dtype, device=A.device)
        low = torch.broadcast_to(low, (action_dim,)).clone()
        high = torch.broadcast_to(high, (action_dim,)).clone()
        if bool((low >= high).any()):
            raise ValueError("Every action lower bound must be below its upper bound")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

        state_map, action_map = self._condense_physical_dynamics()
        self.register_buffer("state_map", state_map)
        self.register_buffer("action_map", action_map)

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

    def _condense_physical_dynamics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``X=S*z0+T*U`` for ``X=[x1,...,xN]``."""

        n = self.lifted_dim
        m = self.action_dim
        lifted_power = torch.eye(n, dtype=self.A.dtype, device=self.A.device)
        lifted_action = self.A.new_zeros(n, self.horizon * m)
        state_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        for step in range(self.horizon):
            lifted_power = self.A @ lifted_power
            lifted_action = self.A @ lifted_action
            lifted_action[:, step * m : (step + 1) * m] += self.B
            state_rows.append(self.C @ lifted_power)
            action_rows.append((self.C @ lifted_action).clone())
        return (
            torch.cat(state_rows, dim=0),
            torch.cat(action_rows, dim=0),
        )

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
                raise ValueError("Wrong or missing context dimension")
            if context.shape[:-1] != batch_shape:
                raise ValueError("Context batch shape does not match lifted state")
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
        # Center log-weights per stage.  Their geometric mean is exactly one.
        # This stabilizes action-only BC, but is a deliberate expressivity
        # restriction: stages cannot learn independent overall curvature.
        centered_log_weights = raw_quadratic - raw_quadratic.mean(
            dim=-1, keepdim=True
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
        """Return ``.5 U' H U + f' U`` for the learned physical cost."""

        batch_shape = lifted_state.shape[:-1]
        expected = (*batch_shape, self.horizon, self.augmented_dim)
        if quadratic.shape != expected or linear.shape != expected:
            raise ValueError("Cost terms have the wrong shape")
        free_physical = lifted_state @ self.state_map.mT
        state_stop = self.physical_dim
        q_state = quadratic[..., :state_stop].reshape(
            *batch_shape, self.horizon * self.physical_dim
        )
        q_action = quadratic[..., state_stop:].reshape(
            *batch_shape, self.horizon * self.action_dim
        )
        p_state = linear[..., :state_stop].reshape(
            *batch_shape, self.horizon * self.physical_dim
        )
        p_action = linear[..., state_stop:].reshape(
            *batch_shape, self.horizon * self.action_dim
        )
        weighted_action_map = q_state.unsqueeze(-1) * self.action_map
        hessian = (
            self.action_map.mT @ weighted_action_map
            + torch.diag_embed(q_action)
        )
        qp_linear = (
            (q_state * free_physical + p_state).unsqueeze(-2)
            @ self.action_map
        ).squeeze(-2) + p_action
        return hessian, qp_linear

    def _solve_box_qp(
        self,
        hessian: torch.Tensor,
        linear: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Approximately solve the condensed QP with fixed unrolled FISTA."""

        flat_low = self.action_low.repeat(self.horizon)
        flat_high = self.action_high.repeat(self.horizon)
        # The induced infinity norm upper-bounds lambda_max for symmetric H.
        lipschitz = hessian.abs().sum(dim=-1).amax(dim=-1)
        step = self.step_fraction / (lipschitz + self.solver_epsilon)
        current = torch.zeros_like(linear)
        extrapolated = current
        momentum = 1.0
        for _ in range(self.solver_iterations):
            gradient = (
                hessian @ extrapolated.unsqueeze(-1)
            ).squeeze(-1) + linear
            following = torch.clamp(
                extrapolated - step.unsqueeze(-1) * gradient,
                min=flat_low,
                max=flat_high,
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
        projected = torch.clamp(
            current - gradient,
            min=flat_low,
            max=flat_high,
        )
        residual = torch.linalg.vector_norm(current - projected, dim=-1)
        return current, residual

    def forward(
        self,
        lifted_state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> KoopmanMPCActorOutput:
        quadratic, linear = self.cost_terms(lifted_state, context)
        hessian, qp_linear = self.condensed_quadratic(
            lifted_state,
            quadratic,
            linear,
        )
        flat_action, residual = self._solve_box_qp(hessian, qp_linear)
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
