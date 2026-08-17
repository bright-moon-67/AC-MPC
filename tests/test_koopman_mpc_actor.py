import math

import numpy as np
import pytest
import torch
from scipy.optimize import Bounds, LinearConstraint, minimize

from antmaze_ac.rl.koopman_mpc_actor import (
    KoopmanMPCActor,
    StructuredKoopmanMPCActor,
    StructuredKoopmanMPCActorV2,
)


def _randomize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                0.08
                * torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )


def test_soft_koopman_mpc_enforces_absolute_limits_with_gradients():
    dtype = torch.float64
    actor = KoopmanMPCActor(
        torch.tensor([[0.95, 0.05], [0.0, 0.9]], dtype=dtype),
        torch.tensor([[0.25], [0.4]], dtype=dtype),
        torch.eye(2, dtype=dtype),
        horizon=4,
        context_dim=2,
        hidden_dims=(5,),
        action_low=-0.15,
        action_high=0.2,
        linear_scale=4.0,
        solver_iterations=30,
    )
    _randomize(actor, seed=211)
    state = torch.randn(6, 2, dtype=dtype, requires_grad=True)
    context = torch.randn(6, 2, dtype=dtype, requires_grad=True)
    output = actor(state, context)

    assert output.action.shape == (6, 1)
    assert output.action_sequence.shape == (6, 4, 1)
    assert bool((output.action_sequence >= -0.15).all())
    assert bool((output.action_sequence <= 0.2).all())
    assert torch.all(torch.linalg.eigvalsh(output.qp_hessian) > 0)

    loss = output.action.square().mean() + 0.03 * output.qp_linear.square().mean()
    gradients = torch.autograd.grad(
        loss,
        (state, context, *actor.parameters()),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_condensed_cost_matches_explicit_normalized_rollout():
    dtype = torch.float64
    A = torch.tensor([[0.9, 0.1], [0.0, 0.85]], dtype=dtype)
    B = torch.tensor([[0.2], [0.35]], dtype=dtype)
    C = torch.eye(2, dtype=dtype)
    actor = KoopmanMPCActor(
        A,
        B,
        C,
        horizon=3,
        hidden_dims=(),
        action_low=-10.0,
        action_high=10.0,
    )
    state = torch.tensor([[0.4, -0.3]], dtype=dtype)
    q = torch.tensor(
        [[[1.2, 0.7, 0.4], [0.8, 1.1, 0.6], [0.9, 0.5, 0.7]]],
        dtype=dtype,
    )
    p = torch.tensor(
        [[[0.2, -0.1, 0.1], [0.1, 0.2, -0.2], [-0.1, 0.1, 0.3]]],
        dtype=dtype,
    )
    actions = torch.tensor([[[0.1], [-0.25], [0.3]]], dtype=dtype)

    hessian, linear = actor.condensed_quadratic(state, q, p)
    flat = actions.flatten(start_dim=-2)
    condensed_delta = (
        0.5
        * (flat.unsqueeze(-2) @ hessian @ flat.unsqueeze(-1)).squeeze()
        + (linear * flat).sum()
    )

    def rollout(candidate: torch.Tensor) -> torch.Tensor:
        value = state[0]
        total = state.new_zeros(())
        for step in range(actor.horizon):
            control = candidate[step]
            value = A @ value + B @ control
            augmented = torch.cat((value, control))
            total = total + 0.5 * (q[0, step] * augmented.square()).sum()
            total = total + (p[0, step] * augmented).sum()
        return total

    explicit_delta = rollout(actions[0]) - rollout(torch.zeros_like(actions[0]))
    torch.testing.assert_close(condensed_delta, explicit_delta)


def test_absolute_box_projection_reaches_action_boundary():
    dtype = torch.float64
    actor = KoopmanMPCActor(
        torch.eye(1, dtype=dtype),
        torch.ones(1, 1, dtype=dtype),
        torch.eye(1, dtype=dtype),
        horizon=4,
        hidden_dims=(),
        action_low=-0.3,
        action_high=0.3,
        solver_iterations=30,
    )
    with torch.no_grad():
        actor.network[-1].bias[actor.horizon * actor.augmented_dim :] = -10.0
    output = actor(torch.zeros(1, 1, dtype=dtype))
    assert bool((output.action_sequence >= -0.3).all())
    assert bool((output.action_sequence <= 0.3).all())


def test_explicit_solver_budget_and_kkt_mapping_are_differentiable():
    dtype = torch.float64
    actor = KoopmanMPCActor(
        torch.tensor([[0.9]], dtype=dtype),
        torch.tensor([[0.25]], dtype=dtype),
        torch.eye(1, dtype=dtype),
        horizon=3,
        context_dim=1,
        hidden_dims=(4,),
        action_low=-1.0,
        action_high=1.0,
        linear_scale=0.1,
        solver_iterations=2,
    )
    _randomize(actor, seed=813)
    output = actor(
        torch.tensor([[0.2]], dtype=dtype),
        torch.tensor([[0.1]], dtype=dtype),
    )
    longer, longer_residual = actor.solve_condensed_qp(
        output.qp_hessian,
        output.qp_linear,
        iterations=20,
    )
    assert longer.shape == (1, 3)
    assert longer_residual.shape == (1,)
    assert not torch.equal(
        longer,
        output.action_sequence.flatten(start_dim=-2),
    )

    mapping = actor.projected_kkt_mapping(
        output.qp_hessian,
        output.qp_linear,
        torch.tensor([[0.04, -0.02, 0.01]], dtype=dtype),
    )
    loss = mapping.square().mean() + longer.square().mean()
    gradients = torch.autograd.grad(loss, tuple(actor.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_physical_reference_initializes_a_tracking_cost():
    dtype = torch.float64
    actor = KoopmanMPCActor(
        torch.eye(2, dtype=dtype),
        torch.ones(2, 1, dtype=dtype) * 0.1,
        torch.eye(2, dtype=dtype),
        horizon=2,
        action_low=-1.0,
        action_high=1.0,
        physical_quadratic_scale=torch.ones(2, dtype=dtype),
        action_quadratic_scale=3.0,
    )
    lifted = torch.tensor([[0.1, -0.2]], dtype=dtype)
    reference = torch.tensor([[0.4, 0.3]], dtype=dtype)
    quadratic, linear = actor.cost_terms(
        lifted,
        physical_reference=reference,
        action_reference=torch.tensor([[0.2]], dtype=dtype),
    )
    torch.testing.assert_close(
        quadratic[..., :2],
        torch.ones(1, 2, 2, dtype=dtype),
    )
    torch.testing.assert_close(
        quadratic[..., 2:],
        torch.full((1, 2, 1), 3.0, dtype=dtype),
    )
    torch.testing.assert_close(
        linear[..., :2],
        -reference.unsqueeze(-2).expand(1, 2, 2),
    )
    torch.testing.assert_close(
        linear[..., 2:],
        torch.full((1, 2, 1), -0.6, dtype=dtype),
    )


def test_normalized_delta_qp_matches_absolute_cost_and_rate_limits():
    dtype = torch.float64
    actor = KoopmanMPCActor(
        torch.tensor([[0.9]], dtype=dtype),
        torch.tensor([[0.2]], dtype=dtype),
        torch.eye(1, dtype=dtype),
        horizon=3,
        hidden_dims=(),
        action_low=-0.3,
        action_high=0.3,
        max_delta=0.001,
        solver_iterations=20,
    )
    state = torch.tensor([[0.2]], dtype=dtype)
    quadratic = torch.tensor(
        [[[1.2, 0.7], [0.8, 0.6], [1.1, 0.9]]], dtype=dtype
    )
    linear_terms = torch.tensor(
        [[[0.1, -0.2], [-0.1, 0.3], [0.2, -0.4]]], dtype=dtype
    )
    previous = torch.tensor([[0.2995]], dtype=dtype)
    absolute_hessian, absolute_linear = actor.condensed_quadratic(
        state, quadratic, linear_terms
    )
    delta_hessian, delta_linear = actor.normalized_delta_quadratic(
        absolute_hessian, absolute_linear, previous
    )
    normalized = torch.tensor([[[0.5], [-1.0], [0.25]]], dtype=dtype)
    flat_delta = normalized.flatten(start_dim=-2)
    _, absolute = actor._integrate_normalized_delta(flat_delta, previous)
    flat_absolute = absolute.flatten(start_dim=-2)
    offset = previous.unsqueeze(-2).expand_as(absolute).flatten(start_dim=-2)

    absolute_difference = (
        0.5
        * (
            flat_absolute.unsqueeze(-2)
            @ absolute_hessian
            @ flat_absolute.unsqueeze(-1)
        ).squeeze()
        + (absolute_linear * flat_absolute).sum()
        - 0.5
        * (offset.unsqueeze(-2) @ absolute_hessian @ offset.unsqueeze(-1)).squeeze()
        - (absolute_linear * offset).sum()
    )
    delta_cost = (
        0.5
        * (flat_delta.unsqueeze(-2) @ delta_hessian @ flat_delta.unsqueeze(-1)).squeeze()
        + (delta_linear * flat_delta).sum()
    )
    torch.testing.assert_close(delta_cost, absolute_difference)

    output = actor(state, previous_action=previous)
    assert output.normalized_delta_sequence is not None
    assert bool((output.normalized_delta_sequence.abs() <= 1.0 + 1e-12).all())
    assert bool((output.action_sequence <= 0.3 + 1e-12).all())
    physical_delta = torch.diff(
        torch.cat((previous.unsqueeze(-2), output.action_sequence), dim=-2),
        dim=-2,
    )
    assert bool((physical_delta.abs() <= 0.001 + 1e-12).all())


def _scipy_bounded_cumulative_projection(
    candidate: np.ndarray,
    previous_action: float,
    *,
    max_delta: float,
    action_low: float,
    action_high: float,
) -> np.ndarray:
    horizon = int(candidate.shape[0])
    cumulative = np.tril(np.ones((horizon, horizon), dtype=np.float64))
    lower = np.full(
        horizon,
        (action_low - previous_action) / max_delta,
        dtype=np.float64,
    )
    upper = np.full(
        horizon,
        (action_high - previous_action) / max_delta,
        dtype=np.float64,
    )
    result = minimize(
        lambda value: 0.5 * np.sum((value - candidate) ** 2),
        np.zeros(horizon, dtype=np.float64),
        jac=lambda value: value - candidate,
        bounds=Bounds(-np.ones(horizon), np.ones(horizon)),
        constraints=LinearConstraint(cumulative, lower, upper),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert result.success, result.message
    return np.asarray(result.x, dtype=np.float64)


def test_normalized_delta_projection_fixes_greedy_boundary_counterexample():
    dtype = torch.float64
    actor = KoopmanMPCActor(
        torch.eye(1, dtype=dtype),
        torch.ones(1, 1, dtype=dtype),
        torch.eye(1, dtype=dtype),
        horizon=2,
        hidden_dims=(),
        action_low=-0.3,
        action_high=0.3,
        max_delta=0.015,
    )
    candidate = torch.tensor([[2.0, 2.0]], dtype=dtype)
    previous = torch.tensor([[0.29]], dtype=dtype)
    projected = actor._project_decision(candidate, previous)
    expected = torch.full((1, 2), 1.0 / 3.0, dtype=dtype)
    torch.testing.assert_close(projected, expected, atol=1e-12, rtol=0.0)
    greedy = torch.tensor([[2.0 / 3.0, 0.0]], dtype=dtype)
    exact_distance = (projected - candidate).square().sum()
    greedy_distance = (greedy - candidate).square().sum()
    assert float(exact_distance) < float(greedy_distance)


def test_normalized_delta_projection_matches_random_scipy_oracle():
    dtype = torch.float64
    horizon = 6
    action_dim = 2
    actor = KoopmanMPCActor(
        torch.eye(1, dtype=dtype),
        torch.ones(1, action_dim, dtype=dtype),
        torch.eye(1, dtype=dtype),
        horizon=horizon,
        hidden_dims=(),
        action_low=-0.3,
        action_high=0.3,
        max_delta=0.015,
    )
    generator = torch.Generator().manual_seed(1907)
    candidate_sequence = 3.0 * torch.randn(
        12,
        horizon,
        action_dim,
        generator=generator,
        dtype=dtype,
    )
    previous = 0.6 * torch.rand(
        12,
        action_dim,
        generator=generator,
        dtype=dtype,
    ) - 0.3
    projected = actor._project_decision(
        candidate_sequence.flatten(start_dim=-2),
        previous,
    ).reshape_as(candidate_sequence)

    for batch in range(candidate_sequence.shape[0]):
        for actuator in range(action_dim):
            oracle = _scipy_bounded_cumulative_projection(
                candidate_sequence[batch, :, actuator].numpy(),
                float(previous[batch, actuator]),
                max_delta=0.015,
                action_low=-0.3,
                action_high=0.3,
            )
            np.testing.assert_allclose(
                projected[batch, :, actuator].detach().numpy(),
                oracle,
                atol=2e-7,
                rtol=0.0,
            )

    absolute = previous.unsqueeze(-2) + 0.015 * torch.cumsum(
        projected,
        dim=-2,
    )
    assert bool((projected.abs() <= 1.0 + 1e-10).all())
    assert bool((absolute >= -0.3 - 1e-10).all())
    assert bool((absolute <= 0.3 + 1e-10).all())


def test_exact_normalized_delta_projection_gradient_matches_finite_difference():
    dtype = torch.float64
    actor = KoopmanMPCActor(
        torch.eye(1, dtype=dtype),
        torch.ones(1, 1, dtype=dtype),
        torch.eye(1, dtype=dtype),
        horizon=2,
        hidden_dims=(),
        action_low=-0.3,
        action_high=0.3,
        max_delta=0.015,
    )
    candidate = torch.tensor(
        [[1.8, 1.4]],
        dtype=dtype,
        requires_grad=True,
    )
    previous = torch.tensor([[0.29]], dtype=dtype)
    weights = torch.tensor([[0.7, -0.2]], dtype=dtype)
    value = (actor._project_decision(candidate, previous) * weights).sum()
    analytic = torch.autograd.grad(value, candidate)[0]

    epsilon = 1e-6
    finite_difference = torch.zeros_like(candidate)
    for index in range(candidate.shape[-1]):
        positive = candidate.detach().clone()
        negative = candidate.detach().clone()
        positive[0, index] += epsilon
        negative[0, index] -= epsilon
        positive_value = (
            actor._project_decision(positive, previous) * weights
        ).sum()
        negative_value = (
            actor._project_decision(negative, previous) * weights
        ).sum()
        finite_difference[0, index] = (
            positive_value - negative_value
        ) / (2.0 * epsilon)
    torch.testing.assert_close(
        analytic,
        finite_difference,
        atol=2e-7,
        rtol=2e-6,
    )


def test_normalized_delta_curvature_shifts_only_decision_hessian():
    dtype = torch.float64
    common = dict(
        A=torch.tensor([[0.9]], dtype=dtype),
        B=torch.tensor([[0.2]], dtype=dtype),
        C=torch.eye(1, dtype=dtype),
        horizon=3,
        hidden_dims=(),
        max_delta=0.015,
    )
    base = KoopmanMPCActor(**common)
    regularized = KoopmanMPCActor(
        **common,
        normalized_delta_curvature=1e-4,
    )
    absolute_hessian = torch.tensor(
        [[[1.0, 0.1, 0.0], [0.1, 1.2, 0.2], [0.0, 0.2, 0.8]]],
        dtype=dtype,
    )
    absolute_linear = torch.tensor([[0.2, -0.1, 0.3]], dtype=dtype)
    previous = torch.tensor([[0.05]], dtype=dtype)
    base_hessian, base_linear = base.normalized_delta_quadratic(
        absolute_hessian,
        absolute_linear,
        previous,
    )
    shifted_hessian, shifted_linear = regularized.normalized_delta_quadratic(
        absolute_hessian,
        absolute_linear,
        previous,
    )
    torch.testing.assert_close(shifted_linear, base_linear)
    torch.testing.assert_close(
        shifted_hessian,
        base_hessian + 1e-4 * torch.eye(3, dtype=dtype),
    )


def test_structured_cost_starts_at_same_reference_tracking_objective():
    dtype = torch.float64
    common = dict(
        A=torch.eye(4, dtype=dtype) * 0.9,
        B=torch.ones(4, 2, dtype=dtype) * 0.05,
        C=torch.eye(4, dtype=dtype),
        horizon=3,
        context_dim=2,
        hidden_dims=(7,),
        physical_quadratic_scale=torch.tensor(
            [1e-8, 2.0, 3.0, 4.0], dtype=dtype
        ),
        action_quadratic_scale=5.0,
    )
    full = KoopmanMPCActor(**common)
    structured = StructuredKoopmanMPCActor(
        **common,
        structured_tip_indices=(1, 2, 3),
    )
    lifted = torch.randn(5, 4, dtype=dtype)
    context = torch.randn(5, 2, dtype=dtype)
    state_reference = torch.randn(5, 4, dtype=dtype)
    action_reference = torch.randn(5, 2, dtype=dtype)
    full_quadratic, full_linear = full.cost_terms(
        lifted,
        context,
        state_reference,
        action_reference,
    )
    structured_quadratic, structured_linear = structured.cost_terms(
        lifted,
        context,
        state_reference,
        action_reference,
    )
    torch.testing.assert_close(structured_quadratic, full_quadratic)
    torch.testing.assert_close(structured_linear, full_linear)
    assert structured.network[-1].out_features == 5
    assert sum(p.numel() for p in structured.parameters()) < sum(
        p.numel() for p in full.parameters()
    )


def test_structured_cost_has_bounded_multipliers_and_finite_gradients():
    dtype = torch.float64
    actor = StructuredKoopmanMPCActor(
        torch.eye(4, dtype=dtype) * 0.9,
        torch.ones(4, 2, dtype=dtype) * 0.05,
        torch.eye(4, dtype=dtype),
        horizon=3,
        context_dim=2,
        hidden_dims=(7,),
        physical_quadratic_scale=torch.ones(4, dtype=dtype),
        action_quadratic_scale=2.0,
        structured_log_scale=math.log(2.0),
        structured_tip_indices=(1, 2, 3),
        action_low=-0.3,
        action_high=0.3,
        max_delta=0.01,
        solver_iterations=10,
    )
    _randomize(actor, seed=1408)
    lifted = torch.randn(6, 4, dtype=dtype)
    context = torch.randn(6, 2, dtype=dtype)
    state_reference = torch.randn(6, 4, dtype=dtype)
    action_reference = torch.randn(6, 2, dtype=dtype)
    quadratic, linear = actor.cost_terms(
        lifted,
        context,
        state_reference,
        action_reference,
    )
    state_q = quadratic[..., :4]
    action_q = quadratic[..., 4:]
    assert float(state_q[..., 1:].min()) >= 0.25 - 1e-12
    assert float(state_q[..., 1:].max()) <= 4.0 + 1e-12
    assert float(action_q.min()) >= 1.0 - 1e-12
    assert float(action_q.max()) <= 4.0 + 1e-12
    torch.testing.assert_close(
        linear[..., :4],
        -state_q * state_reference.unsqueeze(-2),
    )
    torch.testing.assert_close(
        linear[..., 4:],
        -action_q * action_reference.unsqueeze(-2),
    )
    # The structured cost map itself remains differentiable even when a
    # particular box-QP test instance lands on a saturated active set.
    loss = quadratic.mean() + 0.01 * linear.square().mean()
    gradients = torch.autograd.grad(loss, tuple(actor.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_structured_implicit_reference_keeps_q_and_learns_free_stage_p():
    dtype = torch.float64
    actor = StructuredKoopmanMPCActor(
        torch.eye(4, dtype=dtype) * 0.9,
        torch.ones(4, 2, dtype=dtype) * 0.05,
        torch.eye(4, dtype=dtype),
        horizon=3,
        context_dim=2,
        hidden_dims=(7,),
        physical_quadratic_scale=torch.tensor(
            [1e-8, 2.0, 3.0, 4.0], dtype=dtype
        ),
        action_quadratic_scale=5.0,
        structured_tip_indices=(1, 2, 3),
        reference_mode="implicit",
    )
    lifted = torch.randn(5, 4, dtype=dtype)
    context = torch.randn(5, 2, dtype=dtype)
    quadratic, linear = actor.cost_terms(lifted, context)
    expected_q = torch.tensor(
        [1e-8, 2.0, 3.0, 4.0, 5.0, 5.0], dtype=dtype
    )
    torch.testing.assert_close(
        quadratic,
        expected_q.expand(5, 3, 6),
    )
    torch.testing.assert_close(linear, torch.zeros_like(linear))
    assert actor.network[-1].out_features == 5 + 3 * 6

    with torch.no_grad():
        actor.network[-1].bias[5:] = 0.2
    _, learned_linear = actor.cost_terms(lifted, context)
    assert bool((learned_linear.abs() > 0).all())
    with pytest.raises(ValueError, match="must not receive explicit"):
        actor.cost_terms(
            lifted,
            context,
            torch.zeros(5, 4, dtype=dtype),
            torch.zeros(5, 2, dtype=dtype),
        )


def test_structured_terminal_multiplier_can_be_disabled_without_shape_change():
    dtype = torch.float64
    common = dict(
        A=torch.eye(4, dtype=dtype) * 0.9,
        B=torch.ones(4, 2, dtype=dtype) * 0.05,
        C=torch.eye(4, dtype=dtype),
        horizon=3,
        context_dim=2,
        hidden_dims=(7,),
        physical_quadratic_scale=torch.ones(4, dtype=dtype),
        structured_tip_indices=(1, 2, 3),
    )
    terminal_on = StructuredKoopmanMPCActor(**common)
    terminal_off = StructuredKoopmanMPCActor(
        **common,
        use_terminal_multiplier=False,
    )
    terminal_off.load_state_dict(terminal_on.state_dict())
    with torch.no_grad():
        terminal_on.network[-1].bias[4] = 1.0
        terminal_off.network[-1].bias[4] = 1.0
    lifted = torch.randn(5, 4, dtype=dtype)
    context = torch.randn(5, 2, dtype=dtype)
    reference = torch.randn(5, 4, dtype=dtype)
    action_reference = torch.randn(5, 2, dtype=dtype)
    q_on, _ = terminal_on.cost_terms(
        lifted, context, reference, action_reference
    )
    q_off, p_off = terminal_off.cost_terms(
        lifted, context, reference, action_reference
    )
    torch.testing.assert_close(q_off[..., -1, 1:4], q_off[..., 0, 1:4])
    assert bool((q_on[..., -1, 1:4] > q_off[..., -1, 1:4]).all())
    torch.testing.assert_close(
        p_off[..., :4], -q_off[..., :4] * reference.unsqueeze(-2)
    )
    assert terminal_on.network[-1].out_features == 5
    assert terminal_off.network[-1].out_features == 5


def test_structured_v2_groups_all_physical_and_action_dimensions():
    dtype = torch.float64
    physical_scale = torch.tensor(
        [1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-2, 2e-2, 2e-2],
        dtype=dtype,
    )
    actor = StructuredKoopmanMPCActorV2(
        torch.eye(8, dtype=dtype) * 0.9,
        torch.ones(8, 6, dtype=dtype) * 0.02,
        torch.eye(8, dtype=dtype),
        horizon=3,
        context_dim=2,
        hidden_dims=(7,),
        physical_quadratic_scale=physical_scale,
        action_quadratic_scale=2.0,
        structured_log_scale=math.log(2.0),
        structured_tip_indices=(0, 1, 2),
        structured_shape_indices=(3, 4),
        structured_linear_velocity_indices=(5,),
        structured_angular_velocity_indices=(6, 7),
        structured_normalized_delta_weight=1e-4,
        max_delta=0.015,
    )
    assert actor.network[-1].out_features == 11
    assert actor.cost_parameterization == "structured_reference_groups_v2"
    torch.testing.assert_close(
        actor.zero_physical_reference_indices,
        torch.tensor([5, 6, 7]),
    )

    with torch.no_grad():
        actor.network[-1].bias.copy_(
            torch.tensor(
                [
                    0.2,
                    -0.2,
                    0.4,
                    -0.4,
                    0.3,
                    -0.3,
                    0.1,
                    0.2,
                    0.3,
                    0.5,
                    -0.25,
                ],
                dtype=dtype,
            )
        )
    lifted = torch.randn(4, 8, dtype=dtype)
    context = torch.randn(4, 2, dtype=dtype)
    reference = torch.randn(4, 8, dtype=dtype)
    action_reference = torch.randn(4, 6, dtype=dtype)
    quadratic, linear = actor.cost_terms(
        lifted,
        context,
        reference,
        action_reference,
    )
    logs = math.log(2.0) * torch.tanh(actor.network[-1].bias)
    expected_state = physical_scale.clone()
    expected_state[0:3] *= torch.exp(logs[0:3])
    expected_state[3:5] *= torch.exp(logs[3])
    expected_state[5] *= torch.exp(logs[4])
    expected_state[6:8] *= torch.exp(logs[5])
    torch.testing.assert_close(
        quadratic[:, 0, :8],
        expected_state.expand(4, 8),
    )
    torch.testing.assert_close(
        quadratic[:, 1, :8],
        expected_state.expand(4, 8),
    )
    terminal_state = expected_state.clone()
    terminal_state[0:3] *= torch.exp(logs[9])
    torch.testing.assert_close(
        quadratic[:, -1, :8],
        terminal_state.expand(4, 8),
    )
    expected_action = 2.0 * torch.exp(logs[6:9]).repeat(2)
    torch.testing.assert_close(
        quadratic[..., 8:],
        expected_action.expand(4, 3, 6),
    )
    torch.testing.assert_close(
        linear[..., :8],
        -quadratic[..., :8] * reference.unsqueeze(-2),
    )
    torch.testing.assert_close(
        linear[..., 8:],
        -quadratic[..., 8:] * action_reference.unsqueeze(-2),
    )
    expected_curvature = 1e-4 * torch.exp(logs[10])
    torch.testing.assert_close(
        actor.additional_normalized_delta_curvature(lifted, context),
        expected_curvature.expand(4),
    )
    output = actor(
        lifted,
        context,
        reference,
        action_reference,
        previous_action=torch.zeros(4, 6, dtype=dtype),
    )
    absolute_hessian, absolute_linear = actor.condensed_quadratic(
        lifted,
        quadratic,
        linear,
    )
    base_hessian, _ = actor.normalized_delta_quadratic(
        absolute_hessian,
        absolute_linear,
        torch.zeros(4, 6, dtype=dtype),
    )
    identity = torch.eye(18, dtype=dtype)
    torch.testing.assert_close(
        output.qp_hessian,
        base_hessian + expected_curvature * identity,
    )

    loss = quadratic.mean() + 0.01 * linear.square().mean()
    gradients = torch.autograd.grad(loss, tuple(actor.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)
