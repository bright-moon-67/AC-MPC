import torch

from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor


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
