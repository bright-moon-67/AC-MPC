from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuadraticGreedyResult:
    """Terms of a quadratic action-value function and its greedy action."""

    action: torch.Tensor
    action_hessian: torch.Tensor
    state_action: torch.Tensor
    action_linear: torch.Tensor


def greedy_action_from_quadratic(
    state: torch.Tensor,
    action_hessian: torch.Tensor,
    state_action: torch.Tensor,
    action_linear: torch.Tensor,
    *,
    jitter: float = 0.0,
) -> QuadraticGreedyResult:
    """Minimize a function that is quadratic in the action.

    The action-dependent terms use the convention

    ``0.5 * u' H_uu u + u' H_uz z + h_u' u``.

    Therefore the unique unconstrained minimizer is
    ``u = -H_uu^{-1}(H_uz z + h_u)``. Batch dimensions may be
    broadcast. ``H_uu`` must be positive definite; ``jitter`` is only a
    numerical regularizer and is not a substitute for an SPD parameterization.
    """

    if state.ndim < 1:
        raise ValueError("state must have at least one dimension")
    if action_hessian.ndim < 2 or action_hessian.shape[-1] != action_hessian.shape[-2]:
        raise ValueError("action_hessian must end in a square matrix")
    action_dim = action_hessian.shape[-1]
    if state_action.shape[-2:] != (action_dim, state.shape[-1]):
        raise ValueError("state_action must have shape [..., action_dim, state_dim]")
    if action_linear.shape[-1] != action_dim:
        raise ValueError("action_linear must have shape [..., action_dim]")
    if jitter < 0:
        raise ValueError("jitter must be non-negative")

    regularized = action_hessian
    if jitter:
        identity = torch.eye(
            action_dim,
            dtype=action_hessian.dtype,
            device=action_hessian.device,
        )
        regularized = action_hessian + float(jitter) * identity
    state_for_control = state.to(state_action.dtype)
    rhs = (
        state_action @ state_for_control.unsqueeze(-1)
    ).squeeze(-1) + action_linear
    action = -torch.linalg.solve(regularized, rhs.unsqueeze(-1)).squeeze(-1)
    return QuadraticGreedyResult(
        action=action,
        action_hessian=regularized,
        state_action=state_action,
        action_linear=action_linear,
    )


def greedy_action_from_value(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    state: torch.Tensor,
    value_hessian: torch.Tensor,
    value_linear: torch.Tensor,
    *,
    state_control: torch.Tensor | None = None,
    control_linear: torch.Tensor | None = None,
    dynamics_bias: torch.Tensor | None = None,
    jitter: float = 0.0,
) -> QuadraticGreedyResult:
    """Construct and minimize a one-step quadratic action value.

    Dynamics are ``z+ = A z + B u + b``. The stage and terminal value
    conventions are

    ``l(z,u) = 0.5 z'Qz + z'Nu + 0.5 u'Ru + q'z + r'u``

    and ``V(z) = 0.5 z'Pz + p'z``. Terms independent of ``u`` are omitted,
    yielding

    ``H_uu = R + B'PB``,
    ``H_uz = N' + B'PA``, and
    ``h_u = r + B'(Pb+p)``.

    This is a local greedy controller. A learned ``P`` is not automatically a
    Riccati solution or a globally Bellman-consistent value function.
    """

    if A.ndim != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")
    state_dim = A.shape[-1]
    if B.ndim != 2 or B.shape[0] != state_dim:
        raise ValueError("B must have shape [state_dim, action_dim]")
    action_dim = B.shape[-1]
    if R.shape != (action_dim, action_dim):
        raise ValueError("R must have shape [action_dim, action_dim]")
    if state.shape[-1] != state_dim:
        raise ValueError("state dimension does not match A")
    if value_hessian.shape[-2:] != (state_dim, state_dim):
        raise ValueError("value_hessian must have shape [..., state_dim, state_dim]")
    if value_linear.shape[-1] != state_dim:
        raise ValueError("value_linear must have shape [..., state_dim]")

    bt_p = B.mT @ value_hessian
    action_hessian = R + bt_p @ B
    state_action_term = bt_p @ A
    if state_control is not None:
        if state_control.shape[-2:] != (state_dim, action_dim):
            raise ValueError(
                "state_control must have shape [..., state_dim, action_dim]"
            )
        state_action_term = state_action_term + state_control.mT

    value_affine = value_linear
    if dynamics_bias is not None:
        if dynamics_bias.shape[-1] != state_dim:
            raise ValueError("dynamics_bias must have shape [..., state_dim]")
        value_affine = value_affine + (
            value_hessian @ dynamics_bias.unsqueeze(-1)
        ).squeeze(-1)
    action_linear_term = (
        B.mT @ value_affine.unsqueeze(-1)
    ).squeeze(-1)
    if control_linear is not None:
        if control_linear.shape[-1] != action_dim:
            raise ValueError("control_linear must have shape [..., action_dim]")
        action_linear_term = action_linear_term + control_linear

    return greedy_action_from_quadratic(
        state,
        action_hessian,
        state_action_term,
        action_linear_term,
        jitter=jitter,
    )


def greedy_action_from_low_rank_value(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    state: torch.Tensor,
    base_hessian: torch.Tensor,
    diagonal: torch.Tensor,
    factors: torch.Tensor,
    value_linear: torch.Tensor,
    *,
    state_control: torch.Tensor | None = None,
    control_linear: torch.Tensor | None = None,
    dynamics_bias: torch.Tensor | None = None,
    jitter: float = 0.0,
) -> QuadraticGreedyResult:
    """Greedy action for ``P=P0+diag(d)+U U'`` without materializing ``P``.

    The structured form reduces the learned output size and avoids a batched
    dense ``state_dim x state_dim`` value matrix. It still solves one
    ``action_dim x action_dim`` system per sample.
    """

    if A.ndim != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")
    state_dim = A.shape[-1]
    if B.ndim != 2 or B.shape[0] != state_dim:
        raise ValueError("B must have shape [state_dim, action_dim]")
    action_dim = B.shape[-1]
    if R.shape != (action_dim, action_dim):
        raise ValueError("R must have shape [action_dim, action_dim]")
    if base_hessian.shape != (state_dim, state_dim):
        raise ValueError("base_hessian must have shape [state_dim, state_dim]")
    if state.shape[-1] != state_dim or diagonal.shape[-1] != state_dim:
        raise ValueError("state and diagonal dimensions must match A")
    if factors.ndim < 2 or factors.shape[-2] != state_dim:
        raise ValueError("factors must have shape [..., state_dim, rank]")
    if value_linear.shape[-1] != state_dim:
        raise ValueError("value_linear must have shape [..., state_dim]")

    bt_p0 = B.mT @ base_hessian
    action_hessian = R + bt_p0 @ B
    state_action_term = bt_p0 @ A

    diagonal_b = diagonal.unsqueeze(-1) * B
    diagonal_a = diagonal.unsqueeze(-1) * A
    action_hessian = action_hessian + B.mT @ diagonal_b
    state_action_term = state_action_term + B.mT @ diagonal_a

    ut_b = factors.mT @ B
    ut_a = factors.mT @ A
    action_hessian = action_hessian + ut_b.mT @ ut_b
    state_action_term = state_action_term + ut_b.mT @ ut_a

    if state_control is not None:
        if state_control.shape[-2:] != (state_dim, action_dim):
            raise ValueError(
                "state_control must have shape [..., state_dim, action_dim]"
            )
        state_action_term = state_action_term + state_control.mT

    value_affine = value_linear
    if dynamics_bias is not None:
        if dynamics_bias.shape[-1] != state_dim:
            raise ValueError("dynamics_bias must have shape [..., state_dim]")
        p0_bias = base_hessian @ dynamics_bias.unsqueeze(-1)
        diagonal_bias = diagonal * dynamics_bias
        low_rank_bias = (
            factors
            @ (factors.mT @ dynamics_bias.unsqueeze(-1))
        ).squeeze(-1)
        value_affine = (
            value_affine
            + p0_bias.squeeze(-1)
            + diagonal_bias
            + low_rank_bias
        )
    action_linear_term = (
        B.mT @ value_affine.unsqueeze(-1)
    ).squeeze(-1)
    if control_linear is not None:
        if control_linear.shape[-1] != action_dim:
            raise ValueError("control_linear must have shape [..., action_dim]")
        action_linear_term = action_linear_term + control_linear

    return greedy_action_from_quadratic(
        state,
        action_hessian,
        state_action_term,
        action_linear_term,
        jitter=jitter,
    )


def low_rank_quadratic_value(
    state: torch.Tensor,
    base_hessian: torch.Tensor,
    diagonal: torch.Tensor,
    factors: torch.Tensor,
    value_linear: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``0.5*z'(P0+diag(d)+UU')z+p'z`` efficiently."""

    if state.shape[-1] != base_hessian.shape[-1]:
        raise ValueError("state dimension does not match base_hessian")
    base = 0.5 * torch.einsum("...i,ij,...j->...", state, base_hessian, state)
    diagonal_term = 0.5 * (diagonal * state.square()).sum(dim=-1)
    projected = (factors.mT @ state.unsqueeze(-1)).squeeze(-1)
    low_rank_term = 0.5 * projected.square().sum(dim=-1)
    linear_term = (value_linear * state).sum(dim=-1)
    return base + diagonal_term + low_rank_term + linear_term
