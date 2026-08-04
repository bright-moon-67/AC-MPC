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


def test_koopman_mpc_condensing_matches_explicit_rollout_cost() -> None:
    dtype = torch.float64
    A = torch.tensor([[0.9, 0.1], [0.0, 0.85]], dtype=dtype)
    B = torch.tensor([[0.2], [0.35]], dtype=dtype)
    C = torch.eye(2, dtype=dtype)
    actor = KoopmanMPCActor(
        A,
        B,
        C,
        horizon=4,
        hidden_dims=(),
        solver_iterations=5,
    )
    state = torch.tensor([[0.4, -0.3], [-0.1, 0.2]], dtype=dtype)
    q = torch.tensor(
        [
            [[1.2, 0.7, 0.4], [0.8, 1.1, 0.6], [0.9, 0.5, 0.7], [1.4, 0.8, 0.3]],
            [[0.4, 1.6, 0.5], [1.3, 0.8, 0.9], [0.7, 1.2, 0.4], [0.6, 0.9, 1.1]],
        ],
        dtype=dtype,
    )
    p = torch.tensor(
        [
            [[0.2, -0.1, 0.1], [0.1, 0.2, -0.2], [-0.1, 0.1, 0.3], [0.2, 0.0, -0.1]],
            [[-0.3, 0.15, 0.2], [0.1, -0.2, 0.05], [0.3, 0.1, -0.1], [0.0, -0.1, 0.2]],
        ],
        dtype=dtype,
    )
    actions = torch.tensor(
        [[[0.1], [-0.25], [0.3], [0.05]], [[-0.2], [0.4], [0.1], [-0.3]]],
        dtype=dtype,
    )

    hessian, linear = actor.condensed_quadratic(state, q, p)
    flat_actions = actions.flatten(start_dim=-2)
    condensed_action_cost = (
        0.5
        * (
            flat_actions.unsqueeze(-2)
            @ hessian
            @ flat_actions.unsqueeze(-1)
        ).squeeze((-1, -2))
        + (linear * flat_actions).sum(dim=-1)
    )

    zero_action_cost = []
    explicit_action_cost = []
    for batch in range(len(state)):
        def rollout(candidate: torch.Tensor) -> torch.Tensor:
            value = state[batch]
            total = state.new_zeros(())
            for step in range(actor.horizon):
                control = candidate[step]
                value = A @ value + B @ control
                augmented = torch.cat((value, control))
                total = total + 0.5 * (
                    q[batch, step] * augmented.square()
                ).sum() + (p[batch, step] * augmented).sum()
            return total

        zero_action_cost.append(rollout(torch.zeros_like(actions[batch])))
        explicit_action_cost.append(rollout(actions[batch]))
    explicit_delta = torch.stack(explicit_action_cost) - torch.stack(zero_action_cost)
    torch.testing.assert_close(condensed_action_cost, explicit_delta)


def test_koopman_mpc_matches_interior_unconstrained_solution() -> None:
    dtype = torch.float64
    actor = KoopmanMPCActor(
        torch.tensor([[0.92]], dtype=dtype),
        torch.tensor([[0.25]], dtype=dtype),
        torch.tensor([[1.0]], dtype=dtype),
        horizon=3,
        hidden_dims=(),
        action_low=-10.0,
        action_high=10.0,
        linear_scale=0.5,
        solver_iterations=300,
    )
    state = torch.tensor([[0.3]], dtype=dtype)
    output = actor(state)
    exact = -torch.linalg.solve(
        output.qp_hessian,
        output.qp_linear.unsqueeze(-1),
    ).squeeze(-1)

    torch.testing.assert_close(
        output.action_sequence.flatten(start_dim=-2),
        exact,
        rtol=2e-5,
        atol=2e-6,
    )
    assert float(output.projected_gradient_residual.max()) < 2e-6


def test_koopman_mpc_enforces_box_and_has_finite_gradients() -> None:
    dtype = torch.float64
    actor = KoopmanMPCActor(
        torch.tensor([[0.95, 0.05], [0.0, 0.9]], dtype=dtype),
        torch.tensor([[0.25], [0.4]], dtype=dtype),
        torch.eye(2, dtype=dtype),
        horizon=4,
        hidden_dims=(5,),
        action_low=-0.15,
        action_high=0.2,
        linear_scale=4.0,
        solver_iterations=30,
    )
    _randomize(actor, seed=211)
    state = torch.randn(6, 2, dtype=dtype, requires_grad=True)
    output = actor(state)

    assert output.action.shape == (6, 1)
    assert output.action_sequence.shape == (6, 4, 1)
    assert bool((output.action_sequence >= -0.15).all())
    assert bool((output.action_sequence <= 0.2).all())
    assert bool((output.quadratic_diagonal >= actor.quadratic_lower_bound).all())
    assert bool((output.quadratic_diagonal <= actor.quadratic_upper_bound).all())
    assert torch.all(torch.linalg.eigvalsh(output.qp_hessian) > 0)

    loss = output.action.square().mean() + 0.03 * output.qp_linear.square().mean()
    gradients = torch.autograd.grad(loss, (state, *actor.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_pandareach_n2_cost_map_has_96_outputs() -> None:
    lifted_dim = 49
    physical_dim = 17
    action_dim = 7
    actor = KoopmanMPCActor(
        torch.eye(lifted_dim),
        torch.zeros(lifted_dim, action_dim),
        torch.eye(physical_dim, lifted_dim),
        horizon=2,
        hidden_dims=(16,),
        solver_iterations=2,
    )
    final = actor.network[-1]
    assert isinstance(final, torch.nn.Linear)
    assert final.out_features == 96
    output = actor(torch.zeros(3, lifted_dim))
    assert output.quadratic_diagonal.shape == (3, 2, 24)
    assert output.linear_term.shape == (3, 2, 24)
    assert output.action.shape == (3, 7)
    assert "dynamics_bias" not in actor.state_dict()
    assert "affine_offset" not in actor.state_dict()


def test_koopman_mpc_accepts_separate_goal_context_with_gradients() -> None:
    actor = KoopmanMPCActor(
        torch.eye(4),
        0.1 * torch.ones(4, 2),
        torch.eye(3, 4),
        horizon=3,
        context_dim=3,
        hidden_dims=(12,),
        solver_iterations=5,
    )
    _randomize(actor, seed=91)
    lifted = torch.randn(6, 4, requires_grad=True)
    goal = torch.randn(6, 3, requires_grad=True)
    output = actor(lifted, goal)
    lifted_gradient, goal_gradient = torch.autograd.grad(
        output.action.square().mean(), (lifted, goal)
    )
    assert output.action_sequence.shape == (6, 3, 2)
    assert torch.isfinite(lifted_gradient).all()
    assert torch.isfinite(goal_gradient).all()
    assert float(goal_gradient.abs().sum()) > 0
