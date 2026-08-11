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


def test_soft_koopman_mpc_enforces_absolute_and_rate_limits_with_gradients():
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
        max_delta=0.03,
        linear_scale=4.0,
        solver_iterations=30,
    )
    _randomize(actor, seed=211)
    state = torch.randn(6, 2, dtype=dtype, requires_grad=True)
    context = torch.randn(6, 2, dtype=dtype, requires_grad=True)
    previous = torch.linspace(-0.1, 0.1, 6, dtype=dtype).unsqueeze(-1)
    output = actor(state, previous, context)

    assert output.action.shape == (6, 1)
    assert output.action_sequence.shape == (6, 4, 1)
    assert bool((output.action_sequence >= -0.15).all())
    assert bool((output.action_sequence <= 0.2).all())
    all_actions = torch.cat((previous.unsqueeze(-2), output.action_sequence), dim=-2)
    assert bool((torch.diff(all_actions, dim=-2).abs() <= 0.0300001).all())
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
        max_delta=20.0,
        smoothness_weight=0.0,
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
