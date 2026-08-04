import torch

from antmaze_ac.control.quadratic_greedy import greedy_action_from_value
from antmaze_ac.rl.quadratic_actors import (
    DirectQuadraticActor,
    LowRankValueActor,
    MinimalDirectQuadraticActor,
)


def _randomize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                0.15
                * torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )


def test_direct_quadratic_actor_is_spd_stationary_and_differentiable() -> None:
    dtype = torch.float64
    actor = DirectQuadraticActor(
        observation_dim=5,
        lifted_dim=4,
        action_dim=3,
        hidden_dims=(7,),
        cholesky_epsilon=2e-3,
        max_action=0.8,
    ).to(dtype=dtype)
    _randomize(actor, seed=101)
    observation = torch.randn(6, 5, dtype=dtype, requires_grad=True)
    lifted_state = torch.randn(6, 4, dtype=dtype, requires_grad=True)

    output = actor(observation, lifted_state)
    hessian = output.quadratic.action_hessian
    rhs = (
        output.quadratic.state_action @ lifted_state.unsqueeze(-1)
    ).squeeze(-1) + output.quadratic.action_linear
    stationarity = (
        hessian @ output.raw_action.unsqueeze(-1)
    ).squeeze(-1) + rhs

    assert output.action.shape == (6, 3)
    assert torch.all(torch.linalg.eigvalsh(hessian) > 0)
    torch.testing.assert_close(
        stationarity,
        torch.zeros_like(stationarity),
        rtol=2e-11,
        atol=2e-11,
    )

    gradients = torch.autograd.grad(
        output.action.square().mean() + 0.07 * output.raw_action.sum(),
        (observation, lifted_state, *actor.parameters()),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_direct_quadratic_actor_can_condition_cost_map_on_lifted_state() -> None:
    dtype = torch.float64
    actor = DirectQuadraticActor(
        observation_dim=5,
        lifted_dim=4,
        action_dim=3,
        hidden_dims=(8,),
        conditioning="lifted",
        max_action=0.8,
    ).to(dtype=dtype)
    _randomize(actor, seed=113)
    observation_a = torch.randn(6, 5, dtype=dtype)
    observation_b = torch.randn(6, 5, dtype=dtype)
    lifted_state = torch.randn(6, 4, dtype=dtype, requires_grad=True)

    output_a = actor(observation_a, lifted_state)
    output_b = actor(observation_b, lifted_state)

    # In lifted conditioning mode, raw x is accepted for the common actor
    # interface but both the cost-map head and the analytic solve depend on z.
    torch.testing.assert_close(output_a.action, output_b.action)
    torch.testing.assert_close(
        output_a.quadratic.action_hessian,
        output_b.quadratic.action_hessian,
    )
    first_layer = actor.network[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == 4

    gradients = torch.autograd.grad(
        output_a.action.square().mean(),
        (lifted_state, *actor.parameters()),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_minimal_direct_h_is_spd_trace_normalized_and_differentiable() -> None:
    dtype = torch.float64
    actor = MinimalDirectQuadraticActor(
        observation_dim=5,
        lifted_dim=4,
        action_dim=3,
        hidden_dims=(7,),
        conditioning="lifted",
        cholesky_epsilon=2e-3,
        max_action=0.8,
    ).to(dtype=dtype)
    _randomize(actor, seed=127)
    observation = torch.randn(6, 5, dtype=dtype)
    lifted_state = torch.randn(6, 4, dtype=dtype, requires_grad=True)

    output = actor(observation, lifted_state)
    stationarity = (
        output.action_hessian @ output.raw_action.unsqueeze(-1)
    ).squeeze(-1) + output.action_linear
    eigenvalues = torch.linalg.eigvalsh(output.action_hessian)
    traces = output.action_hessian.diagonal(dim1=-2, dim2=-1).sum(-1)

    assert output.action.shape == (6, 3)
    assert output.action_linear.shape == (6, 3)
    assert output.action_hessian.shape == (6, 3, 3)
    assert torch.all(eigenvalues > 0)
    torch.testing.assert_close(traces, torch.full_like(traces, 3.0))
    torch.testing.assert_close(
        stationarity,
        torch.zeros_like(stationarity),
        rtol=2e-11,
        atol=2e-11,
    )
    assert torch.all(output.action.abs() < 0.8)

    gradients = torch.autograd.grad(
        output.action.square().mean() + 0.07 * output.raw_action.sum(),
        (lifted_state, *actor.parameters()),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(float(gradient.abs().sum()) > 0 for gradient in gradients)

    diagnostics = output.condition_diagnostics()
    torch.testing.assert_close(diagnostics["minimum_eigenvalue"], eigenvalues[:, 0])
    torch.testing.assert_close(diagnostics["maximum_eigenvalue"], eigenvalues[:, -1])
    torch.testing.assert_close(
        diagnostics["condition_number"], eigenvalues[:, -1] / eigenvalues[:, 0]
    )
    torch.testing.assert_close(diagnostics["trace"], traces)


def test_minimal_direct_h_has_minimal_output_and_configurable_conditioning() -> None:
    lifted_actor = MinimalDirectQuadraticActor(
        observation_dim=5,
        lifted_dim=4,
        action_dim=3,
        hidden_dims=(7,),
        conditioning="lifted",
    )
    observation_actor = MinimalDirectQuadraticActor(
        observation_dim=5,
        lifted_dim=4,
        action_dim=3,
        hidden_dims=(),
        conditioning="observation",
    )

    # 3 * 4 / 2 Cholesky entries plus one 3-vector g_u.
    assert lifted_actor.output_dim == 9
    assert lifted_actor.triangular_size == 6
    assert isinstance(lifted_actor.network[0], torch.nn.Linear)
    assert lifted_actor.network[0].in_features == 4
    assert lifted_actor.network[-1].out_features == 9
    assert sum(parameter.numel() for parameter in lifted_actor.parameters()) == (
        4 * 7 + 7 + 7 * 9 + 9
    )
    assert isinstance(observation_actor.network[0], torch.nn.Linear)
    assert observation_actor.network[0].in_features == 5
    assert observation_actor.network[-1].out_features == 9

    _randomize(observation_actor, seed=137)
    observation = torch.randn(6, 5)
    lifted_a = torch.randn(6, 4)
    lifted_b = torch.randn(6, 4)
    # With observation conditioning, z is only a common interface argument;
    # the minimal head has already emitted the combined g_u term.
    torch.testing.assert_close(
        observation_actor(observation, lifted_a).action,
        observation_actor(observation, lifted_b).action,
    )


def test_low_rank_actor_matches_dense_value_and_gradients() -> None:
    dtype = torch.float64
    A = torch.tensor(
        [[0.8, 0.1, 0.0], [0.0, 0.85, 0.15], [0.05, 0.0, 0.75]],
        dtype=dtype,
    )
    B = torch.tensor(
        [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]],
        dtype=dtype,
    )
    R = torch.tensor([[1.0, 0.15], [0.15, 0.7]], dtype=dtype)
    base_factor = torch.tensor(
        [[0.8, 0.1, -0.2], [0.0, 0.7, 0.3], [0.2, -0.1, 0.6]],
        dtype=dtype,
    )
    base_hessian = (
        base_factor.mT @ base_factor + 0.2 * torch.eye(3, dtype=dtype)
    )
    actor = LowRankValueActor(
        observation_dim=4,
        A=A,
        B=B,
        R=R,
        base_hessian=base_hessian,
        rank=2,
        hidden_dims=(),
        diagonal_initial_bias=-0.3,
        solve_jitter=2e-5,
    ).to(dtype=dtype)
    _randomize(actor, seed=131)
    observation = torch.randn(6, 4, dtype=dtype, requires_grad=True)
    lifted_state = torch.randn(6, 3, dtype=dtype, requires_grad=True)

    output = actor(observation, lifted_state)
    dense_hessian = (
        actor.base_hessian
        + torch.diag_embed(output.diagonal)
        + output.factors @ output.factors.mT
    )
    dense = greedy_action_from_value(
        actor.A,
        actor.B,
        actor.R,
        lifted_state,
        dense_hessian,
        output.value_linear,
        jitter=actor.solve_jitter,
    )

    torch.testing.assert_close(output.raw_action, dense.action)
    torch.testing.assert_close(
        output.quadratic.action_hessian,
        dense.action_hessian,
    )
    gradients = torch.autograd.grad(
        output.raw_action.square().mean(),
        (observation, lifted_state, *actor.parameters()),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_minimal_direct_actor_accepts_separate_goal_context() -> None:
    actor = MinimalDirectQuadraticActor(
        observation_dim=17,
        lifted_dim=49,
        action_dim=7,
        hidden_dims=(16,),
        context_dim=3,
        conditioning="lifted",
    )
    observation = torch.randn(5, 17)
    lifted = torch.randn(5, 49, requires_grad=True)
    goal = torch.randn(5, 3, requires_grad=True)
    loss = actor(observation, lifted, goal).action.square().mean()
    lifted_gradient, goal_gradient = torch.autograd.grad(loss, (lifted, goal))
    assert torch.isfinite(lifted_gradient).all()
    assert torch.isfinite(goal_gradient).all()
