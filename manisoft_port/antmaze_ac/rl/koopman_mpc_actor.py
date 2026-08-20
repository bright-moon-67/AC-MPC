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
        increment_identity = torch.eye(
            self.horizon,
            dtype=A.dtype,
            device=A.device,
        )
        cumulative = torch.tril(torch.ones_like(increment_identity))
        self.register_buffer(
            "normalized_delta_constraint_matrix",
            torch.cat(
                (
                    increment_identity,
                    -increment_identity,
                    cumulative,
                    -cumulative,
                ),
                dim=0,
            ),
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
        additional_curvature: torch.Tensor | None = None,
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
        if self.normalized_delta_curvature or additional_curvature is not None:
            identity = torch.eye(
                transformed_hessian.shape[-1],
                dtype=transformed_hessian.dtype,
                device=transformed_hessian.device,
            )
            transformed_hessian = transformed_hessian + (
                self.normalized_delta_curvature * identity
            )
            if additional_curvature is not None:
                expected_curvature = transformed_hessian.shape[:-2]
                if additional_curvature.shape != expected_curvature:
                    raise ValueError(
                        "additional_curvature must have shape "
                        f"{expected_curvature}, got "
                        f"{tuple(additional_curvature.shape)}"
                    )
                if bool((additional_curvature <= 0).any()):
                    raise ValueError("additional_curvature must be positive")
                transformed_hessian = transformed_hessian + (
                    additional_curvature[..., None, None] * identity
                )
        return transformed_hessian, transformed_linear

    def additional_normalized_delta_curvature(
        self,
        lifted_state: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return optional state-conditioned positive D-space curvature."""

        return None

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
        """Exactly project and integrate a normalized increment plan."""

        if self.max_delta is None:
            raise RuntimeError("Normalized delta integration requires max_delta")
        expected_previous = (*flat_delta.shape[:-1], self.action_dim)
        if previous_action.shape != expected_previous:
            raise ValueError(
                "previous_action must have shape "
                f"{expected_previous}, got {tuple(previous_action.shape)}"
            )
        sequence = flat_delta.reshape(
            *flat_delta.shape[:-1],
            self.horizon,
            self.action_dim,
        )
        previous = torch.clamp(
            previous_action,
            min=self.action_low,
            max=self.action_high,
        )
        projected = self._project_normalized_delta_sequence(
            sequence,
            previous,
        )
        absolute = previous.unsqueeze(-2) + self.max_delta * torch.cumsum(
            projected,
            dim=-2,
        )
        return projected, absolute

    def _project_normalized_delta_sequence(
        self,
        sequence: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        """Return the Euclidean projection onto rate and action constraints.

        The constraints decouple by actuator.  For one actuator the method
        solves

            min_d 0.5 * ||d-v||^2
            subject to |d_k| <= 1
                       action_low <= u_previous
                           + max_delta * cumsum(d)_k <= action_high.

        Box projection is an exact fast path whenever its cumulative action
        sequence is feasible.  Only sequences that approach an absolute
        action boundary enter the small active-set QP.
        """

        if self.max_delta is None:
            raise RuntimeError("Normalized delta projection requires max_delta")
        expected = (
            *previous_action.shape[:-1],
            self.horizon,
            self.action_dim,
        )
        if sequence.shape != expected:
            raise ValueError(
                f"sequence must have shape {expected}, got {tuple(sequence.shape)}"
            )

        clamped = torch.clamp(sequence, -1.0, 1.0)
        absolute = previous_action.unsqueeze(-2) + self.max_delta * torch.cumsum(
            clamped,
            dim=-2,
        )
        within_absolute_box = (
            (absolute >= self.action_low)
            & (absolute <= self.action_high)
        ).all(dim=-2)
        if bool(within_absolute_box.all()):
            return clamped

        candidate_by_actuator = sequence.movedim(-1, -2).reshape(
            -1,
            self.horizon,
        )
        projected_by_actuator = clamped.movedim(-1, -2).reshape(
            -1,
            self.horizon,
        ).clone()
        previous_by_actuator = previous_action.reshape(
            -1,
            self.action_dim,
        ).reshape(-1)
        low_by_actuator = self.action_low.expand_as(previous_action).reshape(-1)
        high_by_actuator = self.action_high.expand_as(previous_action).reshape(-1)
        requires_exact = ~within_absolute_box.reshape(-1)
        exact_indices = torch.nonzero(
            requires_exact,
            as_tuple=False,
        ).flatten()
        for flat_index_tensor in exact_indices.unbind():
            flat_index = int(flat_index_tensor.detach())
            previous = previous_by_actuator[flat_index]
            lower_cumulative = (
                low_by_actuator[flat_index] - previous
            ) / self.max_delta
            upper_cumulative = (
                high_by_actuator[flat_index] - previous
            ) / self.max_delta
            projected_by_actuator[flat_index] = (
                self._project_bounded_cumulative_single(
                    candidate_by_actuator[flat_index],
                    lower_cumulative,
                    upper_cumulative,
                )
            )
        return projected_by_actuator.reshape(
            *sequence.shape[:-2],
            self.action_dim,
            self.horizon,
        ).movedim(-1, -2)

    def _project_bounded_cumulative_single(
        self,
        candidate: torch.Tensor,
        lower_cumulative: torch.Tensor,
        upper_cumulative: torch.Tensor,
    ) -> torch.Tensor:
        """Solve one small Euclidean projection QP by a primal active set."""

        if candidate.shape != (self.horizon,):
            raise ValueError("candidate must contain one actuator horizon")
        original_dtype = candidate.dtype
        solve_dtype = (
            torch.float64
            if original_dtype in (torch.float16, torch.bfloat16, torch.float32)
            else original_dtype
        )
        value = candidate.to(dtype=solve_dtype)
        matrix = self.normalized_delta_constraint_matrix.to(dtype=solve_dtype)
        one = value.new_ones(self.horizon)
        lower = lower_cumulative.to(dtype=solve_dtype)
        upper = upper_cumulative.to(dtype=solve_dtype)
        bounds = torch.cat(
            (
                one,
                one,
                upper.expand(self.horizon),
                (-lower).expand(self.horizon),
            )
        )
        point = torch.zeros_like(value)
        working: list[int] = []
        tolerance = 1e-10
        maximum_iterations = 8 * self.horizon + 8

        for _ in range(maximum_iterations):
            gradient = point - value
            if working:
                active = matrix[working]
                gram = active @ active.mT
                correction = torch.linalg.solve(
                    gram,
                    active @ gradient,
                )
                direction = -gradient + active.mT @ correction
            else:
                active = None
                gram = None
                direction = -gradient

            if float(direction.abs().amax().detach()) <= tolerance:
                if not working:
                    return point.to(dtype=original_dtype)
                assert active is not None and gram is not None
                multipliers = torch.linalg.solve(
                    gram,
                    -(active @ gradient),
                )
                smallest, remove_at = torch.min(multipliers, dim=0)
                if float(smallest.detach()) >= -tolerance:
                    return point.to(dtype=original_dtype)
                del working[int(remove_at.detach())]
                continue

            slack = bounds - matrix @ point
            directional_change = matrix @ direction
            can_block = directional_change > tolerance
            if working:
                can_block[working] = False
            ratios = torch.where(
                can_block,
                slack / directional_change,
                torch.full_like(slack, torch.inf),
            )
            blocking_ratio, blocking_index = torch.min(ratios, dim=0)
            if float(blocking_ratio.detach()) >= 1.0:
                step = value.new_tensor(1.0)
                add_constraint = False
            else:
                step = torch.clamp(blocking_ratio, min=0.0, max=1.0)
                add_constraint = True
            point = point + step * direction

            if add_constraint:
                index = int(blocking_index.detach())
                proposed = [*working, index]
                proposed_rank = int(
                    torch.linalg.matrix_rank(
                        matrix[proposed],
                        tol=tolerance,
                    ).detach()
                )
                if proposed_rank > len(working):
                    working.append(index)

        violation = torch.clamp(matrix @ point - bounds, min=0.0).amax()
        raise RuntimeError(
            "Exact normalized-delta projection did not converge; "
            f"maximum constraint violation={float(violation.detach()):.3e}"
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
            additional_curvature = (
                self.additional_normalized_delta_curvature(
                    lifted_state,
                    context,
                )
            )
            hessian, qp_linear = self.normalized_delta_quadratic(
                absolute_hessian,
                absolute_linear,
                previous_action,
                additional_curvature=additional_curvature,
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


class StructuredKoopmanMPCActor(KoopmanMPCActor):
    """Low-dimensional reference-cost actor for PPO-KMPC.

    The policy learns only five bounded positive multipliers: three tip-axis
    weights, one shared action weight, and one terminal tip multiplier.  The
    default signed linear cost is fixed by the standard reference-tracking
    identities ``p_x=-Q*x_ref`` and ``p_u=-R*u_ref``.  The controlled
    ``reference_mode='implicit'`` ablation keeps those five q multipliers but
    learns a free upstream-style stage-wise p.  The terminal multiplier can be
    fixed to one without changing the output-head shape for a paired ablation.
    """

    STRUCTURED_OUTPUT_DIM = 5

    def __init__(
        self,
        *args,
        structured_log_scale: float = math.log(2.0),
        structured_tip_indices: tuple[int, int, int] = (30, 31, 32),
        reference_mode: str = "explicit",
        use_terminal_multiplier: bool = True,
        **kwargs,
    ) -> None:
        if structured_log_scale <= 0:
            raise ValueError("structured_log_scale must be positive")
        if reference_mode not in ("explicit", "implicit"):
            raise ValueError(
                "reference_mode must be 'explicit' or 'implicit'"
            )
        super().__init__(*args, **kwargs)
        tip_indices = torch.as_tensor(
            structured_tip_indices,
            dtype=torch.long,
            device=self.A.device,
        )
        if tip_indices.shape != (3,):
            raise ValueError("structured_tip_indices must contain three indices")
        if bool((tip_indices < 0).any()) or bool(
            (tip_indices >= self.physical_dim).any()
        ):
            raise ValueError("structured_tip_indices are outside physical state")
        self.structured_log_scale = float(structured_log_scale)
        self.reference_mode = str(reference_mode)
        self.use_terminal_multiplier = bool(use_terminal_multiplier)
        self.register_buffer(
            "structured_tip_indices",
            tip_indices,
            persistent=False,
        )
        # The implicit-reference ablation preserves the five structured
        # positive q multipliers and changes only the source of the signed
        # linear term: instead of injecting -Q*x_ref/-R*u_ref, it learns the
        # same stage-wise free p used by the upstream actor.  Keeping q
        # structured avoids confounding this reference ablation with the
        # separate full-vs-structured cost-map difference.
        linear_output_dim = (
            0
            if self.reference_mode == "explicit"
            else self.horizon * self.augmented_dim
        )
        self.network = _mlp(
            self.lifted_dim + self.context_dim,
            kwargs.get("hidden_dims", (256, 256)),
            self.STRUCTURED_OUTPUT_DIM + linear_output_dim,
            kwargs.get("activation", "gelu"),
        ).to(dtype=self.A.dtype, device=self.A.device)
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected a linear final structured cost layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.cost_parameterization = (
            "structured_reference_weights_v1"
            if self.reference_mode == "explicit"
            else "structured_q_implicit_stage_linear_v1"
        )

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

        raw = self.network(network_input)
        structured_raw = raw[..., : self.STRUCTURED_OUTPUT_DIM]
        log_multipliers = self.structured_log_scale * torch.tanh(
            structured_raw
        )
        tip_log = log_multipliers[..., :3]
        action_multiplier = torch.exp(log_multipliers[..., 3])
        terminal_log = (
            log_multipliers[..., 4]
            if self.use_terminal_multiplier
            else torch.zeros_like(log_multipliers[..., 4])
        )
        terminal_multiplier = torch.exp(terminal_log)

        state_log = lifted_state.new_zeros(
            *batch_shape,
            self.horizon,
            self.physical_dim,
        )
        state_log[..., :, self.structured_tip_indices] = tip_log.unsqueeze(-2)
        state_log[..., -1, self.structured_tip_indices] = (
            tip_log + terminal_log.unsqueeze(-1)
        )
        q_state = self.physical_quadratic_scale * torch.exp(state_log)
        q_action = self.action_quadratic_scale * action_multiplier[..., None, None]
        q_action = q_action.expand(
            *batch_shape,
            self.horizon,
            self.action_dim,
        )
        quadratic = torch.cat((q_state, q_action), dim=-1)

        if self.reference_mode == "implicit":
            if physical_reference is not None or action_reference is not None:
                raise ValueError(
                    "Implicit-reference structured cost must not receive "
                    "explicit physical/action references"
                )
            learned_linear = self.linear_scale * torch.tanh(
                raw[..., self.STRUCTURED_OUTPUT_DIM :]
            ).reshape(
                *batch_shape,
                self.horizon,
                self.augmented_dim,
            )
            linear_state = learned_linear[..., : self.physical_dim]
            linear_action = learned_linear[..., self.physical_dim :]
            if self.use_terminal_multiplier:
                # Treat the terminal scalar as a multiplier on the complete
                # final-tip stage cost.  Scaling q and p together preserves
                # the implicit target -p/q while keeping Q2 orthogonal to Q8.
                linear_state = linear_state.clone()
                linear_state[..., -1, self.structured_tip_indices] = (
                    linear_state[..., -1, self.structured_tip_indices]
                    * terminal_multiplier.unsqueeze(-1)
                )
            return quadratic, torch.cat(
                (linear_state, linear_action), dim=-1
            )

        linear_state = torch.zeros_like(q_state)
        linear_action = torch.zeros_like(q_action)
        if physical_reference is not None:
            expected_reference = (*batch_shape, self.physical_dim)
            if physical_reference.shape != expected_reference:
                raise ValueError(
                    "physical_reference must have shape "
                    f"{expected_reference}, got {tuple(physical_reference.shape)}"
                )
            linear_state = -q_state * physical_reference.unsqueeze(-2)
        if action_reference is not None:
            expected_action_reference = (*batch_shape, self.action_dim)
            if action_reference.shape != expected_action_reference:
                raise ValueError(
                    "action_reference must have shape "
                    f"{expected_action_reference}, got {tuple(action_reference.shape)}"
                )
            linear_action = -q_action * action_reference.unsqueeze(-2)
        return quadratic, torch.cat((linear_state, linear_action), dim=-1)


class StructuredKoopmanMPCActorV2(KoopmanMPCActor):
    """Eleven-output grouped reference cost for the 45-D ManiSoft state.

    The outputs are three tip-position multipliers, one shape/orientation
    multiplier, one linear-velocity multiplier, one angular-velocity
    multiplier, three activation-axis multipliers, and one final-tip
    multiplier, plus one positive normalized-delta curvature multiplier.
    The signed terms remain tied to explicit references.
    """

    STRUCTURED_OUTPUT_DIM = 11
    ACTION_AXIS_COUNT = 3

    def __init__(
        self,
        *args,
        structured_log_scale: float = math.log(2.0),
        structured_tip_indices: tuple[int, int, int] = (30, 31, 32),
        structured_shape_indices: Sequence[int] = (),
        structured_linear_velocity_indices: Sequence[int] = (),
        structured_angular_velocity_indices: Sequence[int] = (),
        structured_normalized_delta_weight: float = 1e-4,
        use_terminal_multiplier: bool = True,
        **kwargs,
    ) -> None:
        if structured_log_scale <= 0:
            raise ValueError("structured_log_scale must be positive")
        if structured_normalized_delta_weight <= 0:
            raise ValueError(
                "structured_normalized_delta_weight must be positive"
            )
        super().__init__(*args, **kwargs)
        if self.max_delta is None:
            raise ValueError("Structured-v2 requires normalized-delta MPC")
        groups = {
            "tip": tuple(map(int, structured_tip_indices)),
            "shape": tuple(map(int, structured_shape_indices)),
            "linear_velocity": tuple(
                map(int, structured_linear_velocity_indices)
            ),
            "angular_velocity": tuple(
                map(int, structured_angular_velocity_indices)
            ),
        }
        if len(groups["tip"]) != 3:
            raise ValueError("structured_tip_indices must contain three indices")
        if any(not indices for indices in groups.values()):
            raise ValueError("Every structured-v2 physical group must be non-empty")
        flattened = [index for indices in groups.values() for index in indices]
        if min(flattened) < 0 or max(flattened) >= self.physical_dim:
            raise ValueError("Structured-v2 indices are outside physical state")
        if len(set(flattened)) != len(flattened):
            raise ValueError("Structured-v2 physical groups must be disjoint")
        if set(flattened) != set(range(self.physical_dim)):
            raise ValueError(
                "Structured-v2 physical groups must partition the physical state"
            )
        if self.action_dim % self.ACTION_AXIS_COUNT:
            raise ValueError(
                "Structured-v2 action dimension must be divisible by three"
            )

        self.structured_log_scale = float(structured_log_scale)
        self.structured_normalized_delta_weight = float(
            structured_normalized_delta_weight
        )
        self.use_terminal_multiplier = bool(use_terminal_multiplier)
        self.reference_mode = "explicit"
        for name, indices in groups.items():
            self.register_buffer(
                f"structured_{name}_indices",
                torch.as_tensor(
                    indices,
                    dtype=torch.long,
                    device=self.A.device,
                ),
                persistent=False,
            )
        self.register_buffer(
            "structured_action_axis",
            torch.arange(
                self.action_dim,
                dtype=torch.long,
                device=self.A.device,
            )
            % self.ACTION_AXIS_COUNT,
            persistent=False,
        )
        self.register_buffer(
            "zero_physical_reference_indices",
            torch.cat(
                (
                    self.structured_linear_velocity_indices,
                    self.structured_angular_velocity_indices,
                )
            ),
            persistent=False,
        )

        self.network = _mlp(
            self.lifted_dim + self.context_dim,
            kwargs.get("hidden_dims", (256, 256)),
            self.STRUCTURED_OUTPUT_DIM,
            kwargs.get("activation", "gelu"),
        ).to(dtype=self.A.dtype, device=self.A.device)
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected a linear final structured-v2 cost layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.cost_parameterization = "structured_reference_groups_v2"

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

        raw = self.network(network_input)
        log_multipliers = self.structured_log_scale * torch.tanh(raw)
        tip_log = log_multipliers[..., 0:3]
        shape_log = log_multipliers[..., 3]
        linear_velocity_log = log_multipliers[..., 4]
        angular_velocity_log = log_multipliers[..., 5]
        action_axis_log = log_multipliers[..., 6:9]
        terminal_log = (
            log_multipliers[..., 9]
            if self.use_terminal_multiplier
            else torch.zeros_like(log_multipliers[..., 9])
        )

        state_log = lifted_state.new_zeros(
            *batch_shape,
            self.horizon,
            self.physical_dim,
        )
        state_log[..., :, self.structured_tip_indices] = tip_log.unsqueeze(-2)
        state_log[..., :, self.structured_shape_indices] = (
            shape_log[..., None, None]
        )
        state_log[..., :, self.structured_linear_velocity_indices] = (
            linear_velocity_log[..., None, None]
        )
        state_log[..., :, self.structured_angular_velocity_indices] = (
            angular_velocity_log[..., None, None]
        )
        state_log[..., -1, self.structured_tip_indices] = (
            tip_log + terminal_log.unsqueeze(-1)
        )
        q_state = self.physical_quadratic_scale * torch.exp(state_log)

        selected_action_log = action_axis_log.index_select(
            -1,
            self.structured_action_axis,
        )
        q_action = self.action_quadratic_scale * torch.exp(
            selected_action_log.unsqueeze(-2)
        )
        q_action = q_action.expand(
            *batch_shape,
            self.horizon,
            self.action_dim,
        )

        linear_state = torch.zeros_like(q_state)
        linear_action = torch.zeros_like(q_action)
        if physical_reference is not None:
            expected_reference = (*batch_shape, self.physical_dim)
            if physical_reference.shape != expected_reference:
                raise ValueError(
                    "physical_reference must have shape "
                    f"{expected_reference}, got {tuple(physical_reference.shape)}"
                )
            linear_state = -q_state * physical_reference.unsqueeze(-2)
        if action_reference is not None:
            expected_action_reference = (*batch_shape, self.action_dim)
            if action_reference.shape != expected_action_reference:
                raise ValueError(
                    "action_reference must have shape "
                    f"{expected_action_reference}, got "
                    f"{tuple(action_reference.shape)}"
                )
            linear_action = -q_action * action_reference.unsqueeze(-2)
        return torch.cat((q_state, q_action), dim=-1), torch.cat(
            (linear_state, linear_action),
            dim=-1,
        )

    def additional_normalized_delta_curvature(
        self,
        lifted_state: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        network_input = lifted_state
        if self.context_dim:
            if context is None or context.shape[-1] != self.context_dim:
                raise ValueError("Wrong or missing actor context dimension")
            if context.shape[:-1] != lifted_state.shape[:-1]:
                raise ValueError("Actor context batch shape does not match lifted state")
            network_input = torch.cat((lifted_state, context), dim=-1)
        elif context is not None:
            raise ValueError("This actor was constructed without context")
        raw_curvature = self.network(network_input)[..., 10]
        multiplier = torch.exp(
            self.structured_log_scale * torch.tanh(raw_curvature)
        )
        return self.structured_normalized_delta_weight * multiplier
