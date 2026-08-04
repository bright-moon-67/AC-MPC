from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from antmaze_ac.control.quadratic_greedy import (
    QuadraticGreedyResult,
    greedy_action_from_low_rank_value,
    greedy_action_from_quadratic,
    low_rank_quadratic_value,
)


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
    for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
        layers.extend((nn.Linear(in_dim, out_dim), activation_type()))
    layers.append(nn.Linear(dimensions[-1], int(output_dim)))
    return nn.Sequential(*layers)


def _zero_final_layer(network: nn.Sequential) -> None:
    final = network[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("Expected the final network layer to be linear")
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)


def _smooth_action_bound(action: torch.Tensor, limit: float | None) -> torch.Tensor:
    if limit is None:
        return action
    return float(limit) * torch.tanh(action / float(limit))


@dataclass
class DirectQuadraticActorOutput:
    action: torch.Tensor
    raw_action: torch.Tensor
    quadratic: QuadraticGreedyResult
    cholesky_factor: torch.Tensor


class DirectQuadraticActor(nn.Module):
    """State-conditioned quadratic-Q actor with an analytic greedy action.

    The cost-map network can be conditioned either on the raw observation
    (the legacy behavior) or on its externally computed Koopman lift. It
    predicts only ``L_uu``, ``H_uz`` and ``h_u``. It constructs
    ``H_uu=L_uu L_uu' + epsilon I`` and solves an ``action_dim`` system:

    ``u* = -H_uu^{-1}(H_uz z + h_u)``.

    This is a structured actor, similar in spirit to a normalized-advantage
    policy head. Unless it is trained with a Bellman or model-consistency
    objective, it must not be interpreted as an MPC solver or a globally
    consistent action-value function.
    """

    def __init__(
        self,
        observation_dim: int,
        lifted_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        conditioning: str = "observation",
        activation: str = "gelu",
        cholesky_epsilon: float = 1e-4,
        cholesky_off_diagonal_scale: float = 1.0,
        state_action_scale: float = 1.0,
        action_linear_scale: float = 1.0,
        max_action: float | None = None,
    ) -> None:
        super().__init__()
        if observation_dim < 1 or lifted_dim < 1 or action_dim < 1:
            raise ValueError("Actor dimensions must be positive")
        if cholesky_epsilon <= 0:
            raise ValueError("cholesky_epsilon must be positive")
        if cholesky_off_diagonal_scale <= 0:
            raise ValueError("cholesky_off_diagonal_scale must be positive")
        if state_action_scale <= 0 or action_linear_scale <= 0:
            raise ValueError("Quadratic output scales must be positive")
        if max_action is not None and max_action <= 0:
            raise ValueError("max_action must be positive when provided")
        if conditioning not in {"observation", "lifted"}:
            raise ValueError(
                "conditioning must be either 'observation' or 'lifted'"
            )

        self.observation_dim = int(observation_dim)
        self.lifted_dim = int(lifted_dim)
        self.action_dim = int(action_dim)
        self.conditioning = conditioning
        self.cholesky_epsilon = float(cholesky_epsilon)
        self.cholesky_off_diagonal_scale = float(
            cholesky_off_diagonal_scale
        )
        self.state_action_scale = float(state_action_scale)
        self.action_linear_scale = float(action_linear_scale)
        self.max_action = None if max_action is None else float(max_action)

        triangular_size = self.action_dim * (self.action_dim + 1) // 2
        output_dim = (
            triangular_size
            + self.action_dim * self.lifted_dim
            + self.action_dim
        )
        head_input_dim = (
            self.observation_dim
            if self.conditioning == "observation"
            else self.lifted_dim
        )
        self.network = _mlp(
            head_input_dim,
            hidden_dims,
            output_dim,
            activation,
        )
        _zero_final_layer(self.network)

        rows, columns = torch.tril_indices(self.action_dim, self.action_dim)
        self.register_buffer("triangular_rows", rows)
        self.register_buffer("triangular_columns", columns)
        self.register_buffer("triangular_diagonal", rows == columns)

    def forward(
        self,
        observation: torch.Tensor,
        lifted_state: torch.Tensor,
    ) -> DirectQuadraticActorOutput:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("Wrong observation dimension")
        if lifted_state.shape[-1] != self.lifted_dim:
            raise ValueError("Wrong lifted-state dimension")

        triangular_size = self.action_dim * (self.action_dim + 1) // 2
        head_input = (
            observation
            if self.conditioning == "observation"
            else lifted_state
        )
        raw = self.network(head_input)
        raw_triangular = raw[..., :triangular_size]
        cross_start = triangular_size
        cross_stop = cross_start + self.action_dim * self.lifted_dim
        raw_state_action = raw[..., cross_start:cross_stop]
        raw_action_linear = raw[..., cross_stop:]

        triangular = (
            self.cholesky_off_diagonal_scale * torch.tanh(raw_triangular)
        )
        diagonal = (
            F.softplus(raw_triangular[..., self.triangular_diagonal])
            + self.cholesky_epsilon
        )
        triangular = triangular.clone()
        triangular[..., self.triangular_diagonal] = diagonal
        cholesky = raw.new_zeros(
            *raw.shape[:-1],
            self.action_dim,
            self.action_dim,
        )
        cholesky[
            ...,
            self.triangular_rows,
            self.triangular_columns,
        ] = triangular
        identity = torch.eye(
            self.action_dim,
            dtype=raw.dtype,
            device=raw.device,
        )
        action_hessian = (
            cholesky @ cholesky.mT
            + self.cholesky_epsilon * identity
        )
        state_action = (
            self.state_action_scale
            * torch.tanh(raw_state_action).reshape(
                *raw.shape[:-1],
                self.action_dim,
                self.lifted_dim,
            )
        )
        action_linear = (
            self.action_linear_scale * torch.tanh(raw_action_linear)
        )
        quadratic = greedy_action_from_quadratic(
            lifted_state,
            action_hessian,
            state_action,
            action_linear,
        )
        action = _smooth_action_bound(quadratic.action, self.max_action)
        return DirectQuadraticActorOutput(
            action=action,
            raw_action=quadratic.action,
            quadratic=quadratic,
            cholesky_factor=cholesky,
        )


@dataclass
class MinimalDirectQuadraticActorOutput:
    """Output of the reduced action-quadratic policy head."""

    action: torch.Tensor
    raw_action: torch.Tensor
    action_hessian: torch.Tensor
    action_linear: torch.Tensor
    cholesky_factor: torch.Tensor

    @torch.no_grad()
    def condition_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return per-sample SPD/conditioning diagnostics on demand.

        Eigenvalues are deliberately not computed in ``forward`` because that
        would add a comparatively expensive operation to every BC update.
        """

        eigenvalues = torch.linalg.eigvalsh(self.action_hessian)
        return {
            "minimum_eigenvalue": eigenvalues[..., 0],
            "maximum_eigenvalue": eigenvalues[..., -1],
            "condition_number": eigenvalues[..., -1] / eigenvalues[..., 0],
            "trace": self.action_hessian.diagonal(dim1=-2, dim2=-1).sum(-1),
        }


class MinimalDirectQuadraticActor(nn.Module):
    """Minimal direct-``H`` actor with a unique global Hessian scale.

    A shallow head emits only the lower triangle of ``L_uu`` and the already
    combined action-linear term ``g_u``.  It therefore has

    ``action_dim * (action_dim + 1) / 2 + action_dim``

    outputs (35 for PandaReach), rather than emitting the unidentifiable full
    ``H_uz`` block.  The policy is

    ``H_uu = normalize_trace(L_uu L_uu' + epsilon I)`` and
    ``u* = -H_uu^{-1} g_u``.

    Trace normalization fixes the common scalar freedom
    ``(c H_uu, c g_u)`` while preserving positive definiteness.  It does not
    make the remaining matrix shape identifiable from action-only BC, nor
    does it supply Bellman consistency or closed-loop/OOD guarantees.

    The module uses the lifted tensor exactly as supplied.  The retained
    PandaReach ``H1-min35-lifted`` route passes direct ``psi(x)``; checkpoint
    metadata and the saved-run evaluator enforce that coordinate convention.
    """

    def __init__(
        self,
        observation_dim: int,
        lifted_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128,),
        *,
        context_dim: int = 0,
        conditioning: str = "lifted",
        activation: str = "gelu",
        cholesky_epsilon: float = 1e-4,
        cholesky_off_diagonal_scale: float = 1.0,
        action_linear_scale: float = 1.0,
        max_action: float | None = None,
    ) -> None:
        super().__init__()
        if observation_dim < 1 or lifted_dim < 1 or action_dim < 1:
            raise ValueError("Actor dimensions must be positive")
        if context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        if cholesky_epsilon <= 0:
            raise ValueError("cholesky_epsilon must be positive")
        if cholesky_off_diagonal_scale <= 0 or action_linear_scale <= 0:
            raise ValueError("Quadratic output scales must be positive")
        if max_action is not None and max_action <= 0:
            raise ValueError("max_action must be positive when provided")
        if conditioning not in {"observation", "lifted"}:
            raise ValueError(
                "conditioning must be either 'observation' or 'lifted'"
            )

        self.observation_dim = int(observation_dim)
        self.lifted_dim = int(lifted_dim)
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.conditioning = conditioning
        self.cholesky_epsilon = float(cholesky_epsilon)
        self.cholesky_off_diagonal_scale = float(
            cholesky_off_diagonal_scale
        )
        self.action_linear_scale = float(action_linear_scale)
        self.max_action = None if max_action is None else float(max_action)

        self.triangular_size = self.action_dim * (self.action_dim + 1) // 2
        self.output_dim = self.triangular_size + self.action_dim
        head_input_dim = self.context_dim + (
            self.observation_dim
            if self.conditioning == "observation"
            else self.lifted_dim
        )
        self.network = _mlp(
            head_input_dim,
            hidden_dims,
            self.output_dim,
            activation,
        )
        _zero_final_layer(self.network)

        rows, columns = torch.tril_indices(self.action_dim, self.action_dim)
        self.register_buffer("triangular_rows", rows)
        self.register_buffer("triangular_columns", columns)
        self.register_buffer("triangular_diagonal", rows == columns)

    def forward(
        self,
        observation: torch.Tensor,
        lifted_state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> MinimalDirectQuadraticActorOutput:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("Wrong observation dimension")
        if lifted_state.shape[-1] != self.lifted_dim:
            raise ValueError("Wrong lifted-state dimension")

        head_input = (
            observation
            if self.conditioning == "observation"
            else lifted_state
        )
        if self.context_dim:
            if context is None or context.shape[-1] != self.context_dim:
                raise ValueError("Wrong or missing context dimension")
            if context.shape[:-1] != head_input.shape[:-1]:
                raise ValueError("Context batch shape does not match actor input")
            head_input = torch.cat((head_input, context), dim=-1)
        elif context is not None:
            raise ValueError("This actor was constructed without context")
        raw = self.network(head_input)
        raw_triangular = raw[..., : self.triangular_size]
        raw_action_linear = raw[..., self.triangular_size :]

        triangular = (
            self.cholesky_off_diagonal_scale * torch.tanh(raw_triangular)
        )
        diagonal = (
            F.softplus(raw_triangular[..., self.triangular_diagonal])
            + self.cholesky_epsilon
        )
        triangular = triangular.clone()
        triangular[..., self.triangular_diagonal] = diagonal
        cholesky = raw.new_zeros(
            *raw.shape[:-1], self.action_dim, self.action_dim
        )
        cholesky[
            ..., self.triangular_rows, self.triangular_columns
        ] = triangular

        identity = torch.eye(
            self.action_dim, dtype=raw.dtype, device=raw.device
        )
        unnormalized_hessian = (
            cholesky @ cholesky.mT + self.cholesky_epsilon * identity
        )
        trace = unnormalized_hessian.diagonal(dim1=-2, dim2=-1).sum(-1)
        action_hessian = (
            float(self.action_dim)
            * unnormalized_hessian
            / trace[..., None, None]
        )
        action_linear = (
            self.action_linear_scale * torch.tanh(raw_action_linear)
        )
        raw_action = -torch.linalg.solve(
            action_hessian, action_linear.unsqueeze(-1)
        ).squeeze(-1)
        action = _smooth_action_bound(raw_action, self.max_action)
        return MinimalDirectQuadraticActorOutput(
            action=action,
            raw_action=raw_action,
            action_hessian=action_hessian,
            action_linear=action_linear,
            cholesky_factor=cholesky,
        )


@dataclass
class LowRankValueActorOutput:
    action: torch.Tensor
    raw_action: torch.Tensor
    diagonal: torch.Tensor
    factors: torch.Tensor
    value_linear: torch.Tensor
    quadratic: QuadraticGreedyResult


class LowRankValueActor(nn.Module):
    """Learn a local quadratic value and greedify it through frozen dynamics.

    The value Hessian is parameterized as
    ``P=P0+diag(softplus(d))+U U'``. The controller uses the frozen ``A,B``
    pair to construct the action-value blocks and solves only an
    ``action_dim x action_dim`` system. Positive semidefiniteness of the
    increment does not by itself guarantee closed-loop stability or Bellman
    consistency when the value parameters depend on the current state.
    """

    def __init__(
        self,
        observation_dim: int,
        A: torch.Tensor,
        B: torch.Tensor,
        R: torch.Tensor,
        base_hessian: torch.Tensor,
        *,
        rank: int = 4,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "gelu",
        diagonal_scale: float = 1.0,
        factor_scale: float = 0.1,
        value_linear_scale: float = 10.0,
        diagonal_initial_bias: float = -6.0,
        solve_jitter: float = 1e-6,
        max_action: float | None = None,
    ) -> None:
        super().__init__()
        if A.ndim != 2 or A.shape[-1] != A.shape[-2]:
            raise ValueError("A must be a square matrix")
        lifted_dim = A.shape[-1]
        if B.ndim != 2 or B.shape[0] != lifted_dim:
            raise ValueError("B must have shape [lifted_dim, action_dim]")
        action_dim = B.shape[-1]
        if R.shape != (action_dim, action_dim):
            raise ValueError("R must have shape [action_dim, action_dim]")
        if base_hessian.shape != (lifted_dim, lifted_dim):
            raise ValueError(
                "base_hessian must have shape [lifted_dim, lifted_dim]"
            )
        if observation_dim < 1 or rank < 1:
            raise ValueError("observation_dim and rank must be positive")
        if diagonal_scale <= 0 or factor_scale <= 0 or value_linear_scale <= 0:
            raise ValueError("Value output scales must be positive")
        if solve_jitter < 0:
            raise ValueError("solve_jitter must be non-negative")
        if max_action is not None and max_action <= 0:
            raise ValueError("max_action must be positive when provided")

        self.observation_dim = int(observation_dim)
        self.lifted_dim = int(lifted_dim)
        self.action_dim = int(action_dim)
        self.rank = int(rank)
        self.diagonal_scale = float(diagonal_scale)
        self.factor_scale = float(factor_scale)
        self.value_linear_scale = float(value_linear_scale)
        self.diagonal_initial_bias = float(diagonal_initial_bias)
        self.solve_jitter = float(solve_jitter)
        self.max_action = None if max_action is None else float(max_action)

        self.register_buffer("A", A.detach().clone())
        self.register_buffer("B", B.detach().clone())
        self.register_buffer("R", R.detach().clone())
        self.register_buffer(
            "base_hessian",
            base_hessian.detach().clone(),
        )

        output_dim = self.lifted_dim * (self.rank + 2)
        self.network = _mlp(
            self.observation_dim,
            hidden_dims,
            output_dim,
            activation,
        )
        _zero_final_layer(self.network)

    def value_terms(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("Wrong observation dimension")
        raw = self.network(observation)
        diagonal_end = self.lifted_dim
        factor_end = diagonal_end + self.lifted_dim * self.rank
        raw_diagonal = raw[..., :diagonal_end]
        raw_factors = raw[..., diagonal_end:factor_end]
        raw_linear = raw[..., factor_end:]
        diagonal = (
            self.diagonal_scale
            * F.softplus(raw_diagonal + self.diagonal_initial_bias)
        )
        factors = (
            self.factor_scale
            * torch.tanh(raw_factors).reshape(
                *raw.shape[:-1],
                self.lifted_dim,
                self.rank,
            )
        )
        value_linear = self.value_linear_scale * torch.tanh(raw_linear)
        return diagonal, factors, value_linear

    def forward(
        self,
        observation: torch.Tensor,
        lifted_state: torch.Tensor,
    ) -> LowRankValueActorOutput:
        if lifted_state.shape[-1] != self.lifted_dim:
            raise ValueError("Wrong lifted-state dimension")
        diagonal, factors, value_linear = self.value_terms(observation)
        quadratic = greedy_action_from_low_rank_value(
            self.A,
            self.B,
            self.R,
            lifted_state,
            self.base_hessian,
            diagonal,
            factors,
            value_linear,
            jitter=self.solve_jitter,
        )
        action = _smooth_action_bound(quadratic.action, self.max_action)
        return LowRankValueActorOutput(
            action=action,
            raw_action=quadratic.action,
            diagonal=diagonal,
            factors=factors,
            value_linear=value_linear,
            quadratic=quadratic,
        )

    def value(
        self,
        observation: torch.Tensor,
        lifted_state: torch.Tensor,
    ) -> torch.Tensor:
        diagonal, factors, value_linear = self.value_terms(observation)
        return low_rank_quadratic_value(
            lifted_state,
            self.base_hessian,
            diagonal,
            factors,
            value_linear,
        )
