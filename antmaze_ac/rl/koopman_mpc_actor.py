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
    normalized_delta: torch.Tensor | None = None
    normalized_delta_sequence: torch.Tensor | None = None
    previous_action: torch.Tensor | None = None


class KoopmanMPCActor(nn.Module):
    r"""Cost-map actor followed by box-constrained finite-horizon Koopman MPC.

    The frozen lifted dynamics are ``z[k+1] = A z[k] + B u[k]`` and the
    normalized physical readout is ``x[k] = C z[k]``.  At every control step,
    the cost-map network emits stage-dependent diagonal quadratic and signed
    linear terms for ``w[k] = [x[k+1], u[k]]``:

    ``sum_k 0.5*w[k]' diag(q[k]) w[k] + p[k]' w[k]``.

    By default the dynamics are condensed directly into a dense QP in absolute
    actions.  When ``max_delta`` is configured, the exact affine substitution

    ``U = 1*u_previous + max_delta*L*D``

    changes the decision variable to normalized action increments ``D``.  The
    lower-triangular integration matrix ``L`` makes every planned absolute
    action depend on the previous applied action and all preceding increments.
    Projected FISTA then keeps ``D`` in ``[-1, 1]`` and the reconstructed
    absolute sequence inside the physical action box.
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
        physical_quadratic_scale: float | torch.Tensor = 1.0,
        quadratic_log_scale: float = 1.5,
        linear_scale: float = 10.0,
        action_quadratic_scale: float = 1.0,
        max_delta: float | None = None,
        normalized_delta_curvature: float = 0.0,
        solver_iterations: int = 20,
        step_fraction: float = 0.95,
        solver_epsilon: float = 1e-6,
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
        if (
            quadratic_log_scale <= 0
            or linear_scale <= 0
            or action_quadratic_scale <= 0
        ):
            raise ValueError("Cost-map scales must be positive")
        if not 0 < step_fraction <= 1:
            raise ValueError("step_fraction must lie in (0, 1]")
        if solver_epsilon <= 0:
            raise ValueError("solver_epsilon must be positive")
        if max_delta is not None and max_delta <= 0:
            raise ValueError("max_delta must be positive when configured")
        if normalized_delta_curvature < 0:
            raise ValueError("normalized_delta_curvature must be non-negative")
        if normalized_delta_curvature > 0 and max_delta is None:
            raise ValueError(
                "normalized_delta_curvature requires max_delta"
            )

        self.lifted_dim = lifted_dim
        self.physical_dim = physical_dim
        self.action_dim = action_dim
        self.augmented_dim = physical_dim + action_dim
        self.horizon = int(horizon)
        self.context_dim = int(context_dim)
        self.quadratic_log_scale = float(quadratic_log_scale)
        self.linear_scale = float(linear_scale)
        self.action_quadratic_scale = float(action_quadratic_scale)
        self.max_delta = None if max_delta is None else float(max_delta)
        self.normalized_delta_curvature = float(normalized_delta_curvature)
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
        integration = torch.tril(
            torch.ones(self.horizon, self.horizon, dtype=A.dtype, device=A.device)
        )
        delta_to_action = torch.kron(
            integration,
            torch.eye(action_dim, dtype=A.dtype, device=A.device),
        )
        if self.max_delta is not None:
            delta_to_action = self.max_delta * delta_to_action
        # This is a deterministic runtime transform, not learned state.  Keep
        # old absolute-action checkpoints loadable without missing buffers.
        self.register_buffer(
            "delta_to_action",
            delta_to_action,
            persistent=False,
        )
        physical_scale = torch.as_tensor(
            physical_quadratic_scale,
            dtype=A.dtype,
            device=A.device,
        )
        physical_scale = torch.broadcast_to(
            physical_scale,
            (physical_dim,),
        ).clone()
        if not torch.isfinite(physical_scale).all() or bool(
            (physical_scale <= 0).any()
        ):
            raise ValueError("Physical quadratic scales must be finite and positive")
        # This scale is reconstructed from the runtime configuration and state
        # normalizer.  Keeping it non-persistent preserves compatibility with
        # actor checkpoints written before reference-cost initialization.
        self.register_buffer(
            "physical_quadratic_scale",
            physical_scale,
            persistent=False,
        )

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

    def cost_terms(
        self,
        lifted_state: torch.Tensor,
        context: torch.Tensor | None = None,
        physical_reference: torch.Tensor | None = None,
        action_reference: torch.Tensor | None = None,
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
        quadratic = torch.cat(
            (
                self.physical_quadratic_scale
                * quadratic[..., : self.physical_dim],
                self.action_quadratic_scale
                * quadratic[..., self.physical_dim :],
            ),
            dim=-1,
        )
        linear = self.linear_scale * torch.tanh(raw[..., 1, :, :])
        if physical_reference is not None:
            expected_reference = (*batch_shape, self.physical_dim)
            if physical_reference.shape != expected_reference:
                raise ValueError(
                    "physical_reference must have shape "
                    f"{expected_reference}, got {tuple(physical_reference.shape)}"
                )
            # Give from-scratch PPO a valid tracking controller before the
            # residual cost map has learned anything.  For
            #   .5 * q * (x - x_ref)^2
            # the decision-dependent linear term is -q*x_ref.  The omitted
            # .5*q*x_ref^2 constant cannot affect the MPC solution.  The
            # network still learns both q and an additive signed linear
            # residual, so this changes the initialization/inductive bias,
            # not the differentiable KMPC structure.
            state_quadratic = quadratic[..., : self.physical_dim]
            tracking_linear = -state_quadratic * physical_reference.unsqueeze(-2)
            linear = torch.cat(
                (
                    linear[..., : self.physical_dim] + tracking_linear,
                    linear[..., self.physical_dim :],
                ),
                dim=-1,
            )
        if action_reference is not None:
            expected_action_reference = (*batch_shape, self.action_dim)
            if action_reference.shape != expected_action_reference:
                raise ValueError(
                    "action_reference must have shape "
                    f"{expected_action_reference}, got {tuple(action_reference.shape)}"
                )
            action_quadratic = quadratic[..., self.physical_dim :]
            linear = torch.cat(
                (
                    linear[..., : self.physical_dim],
                    linear[..., self.physical_dim :]
                    - action_quadratic * action_reference.unsqueeze(-2),
                ),
                dim=-1,
            )
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
        qp_linear = (
            (q_state * free_physical + p_state).unsqueeze(-2)
            @ self.action_map
        ).squeeze(-2) + p_action
        return hessian, qp_linear

    def normalized_delta_quadratic(
        self,
        hessian: torch.Tensor,
        linear: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Substitute normalized increments into an absolute-action QP."""

        if self.max_delta is None:
            raise RuntimeError("normalized_delta_quadratic requires max_delta")
        expected_previous = (*linear.shape[:-1], self.action_dim)
        if previous_action.shape != expected_previous:
            raise ValueError(
                "previous_action must have shape "
                f"{expected_previous}, got {tuple(previous_action.shape)}"
            )
        offset = previous_action.unsqueeze(-2).expand(
            *previous_action.shape[:-1],
            self.horizon,
            self.action_dim,
        ).reshape(*previous_action.shape[:-1], -1)
        transform = self.delta_to_action
        transformed_hessian = transform.mT @ hessian @ transform
        absolute_gradient_at_offset = (
            hessian @ offset.unsqueeze(-1)
        ).squeeze(-1) + linear
        transformed_linear = (
            absolute_gradient_at_offset.unsqueeze(-2) @ transform
        ).squeeze(-2)
        if self.normalized_delta_curvature:
            identity = torch.eye(
                transformed_hessian.shape[-1],
                dtype=transformed_hessian.dtype,
                device=transformed_hessian.device,
            )
            transformed_hessian = transformed_hessian + (
                self.normalized_delta_curvature * identity
            )
        return transformed_hessian, transformed_linear

    def normalized_delta_bounds(
        self,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return feasible first-step normalized bounds at ``previous_action``."""

        if self.max_delta is None:
            raise RuntimeError("normalized_delta_bounds requires max_delta")
        previous = torch.clamp(
            previous_action,
            min=self.action_low,
            max=self.action_high,
        )
        lower = torch.maximum(
            previous.new_full(previous.shape, -1.0),
            (self.action_low - previous) / self.max_delta,
        )
        upper = torch.minimum(
            previous.new_full(previous.shape, 1.0),
            (self.action_high - previous) / self.max_delta,
        )
        return lower, upper

    def _integrate_normalized_delta(
        self,
        flat_delta: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and integrate a normalized increment plan."""

        if self.max_delta is None:
            raise RuntimeError("Normalized delta integration requires max_delta")
        sequence = flat_delta.reshape(
            *flat_delta.shape[:-1], self.horizon, self.action_dim
        )
        previous = torch.clamp(
            previous_action,
            min=self.action_low,
            max=self.action_high,
        )
        applied_deltas: list[torch.Tensor] = []
        absolute_actions: list[torch.Tensor] = []
        for step in range(self.horizon):
            requested_delta = torch.clamp(sequence[..., step, :], -1.0, 1.0)
            following = torch.clamp(
                previous + self.max_delta * requested_delta,
                min=self.action_low,
                max=self.action_high,
            )
            applied_deltas.append((following - previous) / self.max_delta)
            absolute_actions.append(following)
            previous = following
        return torch.stack(applied_deltas, dim=-2), torch.stack(
            absolute_actions, dim=-2
        )

    def _project_decision(
        self,
        candidate: torch.Tensor,
        previous_action: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.max_delta is None:
            flat_low = self.action_low.repeat(self.horizon)
            flat_high = self.action_high.repeat(self.horizon)
            return torch.clamp(candidate, min=flat_low, max=flat_high)
        if previous_action is None:
            raise ValueError("previous_action is required for normalized delta MPC")
        normalized_delta, _ = self._integrate_normalized_delta(
            candidate, previous_action
        )
        return normalized_delta.flatten(start_dim=-2)

    def _solve_box_qp(
        self,
        hessian: torch.Tensor,
        linear: torch.Tensor,
        *,
        iterations: int | None = None,
        previous_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Approximately solve the configured decision-space QP with FISTA."""

        iteration_count = self.solver_iterations if iterations is None else int(
            iterations
        )
        if iteration_count < 1:
            raise ValueError("iterations must be positive")

        lipschitz = hessian.abs().sum(dim=-1).amax(dim=-1)
        step = self.step_fraction / (lipschitz + self.solver_epsilon)
        current = torch.zeros_like(linear)
        extrapolated = current
        momentum = 1.0
        for _ in range(iteration_count):
            gradient = (
                hessian @ extrapolated.unsqueeze(-1)
            ).squeeze(-1) + linear
            following = self._project_decision(
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

        gradient = (
            hessian @ current.unsqueeze(-1)
        ).squeeze(-1) + linear
        projected = self._project_decision(current - gradient, previous_action)
        residual = torch.linalg.vector_norm(current - projected, dim=-1)
        return current, residual

    def solve_condensed_qp(
        self,
        hessian: torch.Tensor,
        linear: torch.Tensor,
        *,
        iterations: int | None = None,
        previous_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve already-condensed learned costs with an explicit budget.

        BC uses this entry point to verify that expert-like actions do not
        depend on stopping FISTA at one particular unroll depth.
        """

        return self._solve_box_qp(
            hessian,
            linear,
            iterations=iterations,
            previous_action=previous_action,
        )

    def projected_kkt_mapping(
        self,
        hessian: torch.Tensor,
        linear: torch.Tensor,
        flat_action: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the box-QP projected first-order optimality mapping.

        This mapping is zero exactly at a KKT point of the convex box QP.
        Keeping the vector (instead of only its norm) lets BC penalize every
        expert action component without rewarding a cost map merely for a
        lucky fixed number of FISTA iterations.
        """

        expected = (*linear.shape[:-1], self.horizon * self.action_dim)
        if flat_action.shape != expected:
            raise ValueError(
                f"flat_action must have shape {expected}, got {flat_action.shape}"
            )
        gradient = (
            hessian @ flat_action.unsqueeze(-1)
        ).squeeze(-1) + linear
        # Use the same stable scale as the projected solver.  A unit-step
        # mapping often clips every component for these condensed, poorly
        # conditioned QPs, producing an uninformative saturated BC loss.
        lipschitz = hessian.abs().sum(dim=-1).amax(dim=-1)
        step = self.step_fraction / (lipschitz + self.solver_epsilon)
        projected = self._project_decision(
            flat_action - step.unsqueeze(-1) * gradient,
            previous_action,
        )
        return flat_action - projected

    def forward(
        self,
        lifted_state: torch.Tensor,
        context: torch.Tensor | None = None,
        physical_reference: torch.Tensor | None = None,
        action_reference: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
    ) -> KoopmanMPCActorOutput:
        quadratic, linear = self.cost_terms(
            lifted_state,
            context,
            physical_reference,
            action_reference,
        )
        absolute_hessian, absolute_linear = self.condensed_quadratic(
            lifted_state,
            quadratic,
            linear,
        )
        normalized_delta_sequence = None
        previous_output = None
        if self.max_delta is None:
            hessian, qp_linear = absolute_hessian, absolute_linear
            flat_action, residual = self._solve_box_qp(hessian, qp_linear)
            sequence = flat_action.reshape(
                *lifted_state.shape[:-1],
                self.horizon,
                self.action_dim,
            )
        else:
            if previous_action is None:
                raise ValueError(
                    "previous_action is required when max_delta is configured"
                )
            previous_output = previous_action
            hessian, qp_linear = self.normalized_delta_quadratic(
                absolute_hessian,
                absolute_linear,
                previous_action,
            )
            flat_delta, residual = self._solve_box_qp(
                hessian,
                qp_linear,
                previous_action=previous_action,
            )
            normalized_delta_sequence, sequence = (
                self._integrate_normalized_delta(flat_delta, previous_action)
            )
        return KoopmanMPCActorOutput(
            action=sequence[..., 0, :],
            action_sequence=sequence,
            quadratic_diagonal=quadratic,
            linear_term=linear,
            qp_hessian=hessian,
            qp_linear=qp_linear,
            projected_gradient_residual=residual,
            normalized_delta=(
                None
                if normalized_delta_sequence is None
                else normalized_delta_sequence[..., 0, :]
            ),
            normalized_delta_sequence=normalized_delta_sequence,
            previous_action=previous_output,
        )
