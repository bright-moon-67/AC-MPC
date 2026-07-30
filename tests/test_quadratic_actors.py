import torch

from antmaze_ac.control.quadratic_greedy import greedy_action_from_value
from antmaze_ac.rl.quadratic_actors import (
    DirectQuadraticActor,
    LowRankValueActor,
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
