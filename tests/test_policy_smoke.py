import torch

from antmaze_ac.koopman.model import DeepKoopman
from antmaze_ac.rl.ac_koopman_policy import GainHoldController, KoopmanLQRPolicy
from antmaze_ac.rl.cost_actor import CostActor
from antmaze_ac.rl.critic import Critic


def test_policy_shapes_and_actor_gradient():
    torch.manual_seed(0)
    koopman = DeepKoopman(3, 1, lift_dim=1, hidden_dims=(4,))
    with torch.no_grad():
        koopman.A.copy_(torch.eye(4) * 0.7)
        koopman.B.fill_(0.2)
    actor = CostActor(
        3,
        1,
        hidden_dims=(8,),
        q_max=2.0,
        p_max=1.0,
        previous_action_dim=1,
        previous_action_cost_scale=0.001,
        delta_action_cost_scale=0.01,
    )
    critic = Critic(3, hidden_dims=(8,))
    policy = KoopmanLQRPolicy(
        koopman,
        actor,
        critic,
        torch.zeros(3),
        torch.ones(3),
        dare_tolerance=1e-8,
        dare_max_iterations=1000,
        dare_jitter=1e-9,
    )
    observations = torch.randn(4, 3)
    q_diag, p = actor(torch.zeros(1, 3))
    assert q_diag.shape == (1, 4) and p.shape == (1, 4)
    assert q_diag[0, 2] < q_diag[0, 0] * 0.01
    assert q_diag[0, 3] < q_diag[0, 0] * 0.02
    assert torch.allclose(p, torch.zeros_like(p))
    output = policy(observations)
    assert output.mean.shape == (4, 1)
    assert output.value.shape == (4,)
    actions = output.distribution.rsample()
    log_prob, entropy, values, _ = policy.evaluate_actions(observations, actions)
    loss = -(log_prob.mean() + 0.01 * entropy.mean()) + values.square().mean() + output.mean.square().mean()
    loss.backward()
    actor_gradients = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
    assert actor_gradients
    assert all(torch.isfinite(gradient).all() for gradient in actor_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in actor_gradients) > 0
    assert all(parameter.grad is None for parameter in koopman.parameters())
    controller = GainHoldController(policy, gain_update_interval=2)
    assert controller.act(observations[0]).shape == (1,)
    assert controller.last_gain_recomputed
    assert controller.act(observations[1]).shape == (1,)
    assert not controller.last_gain_recomputed


def test_policy_accepts_goal_features_without_changing_koopman_state():
    torch.manual_seed(1)
    koopman = DeepKoopman(3, 1, lift_dim=1, hidden_dims=(4,))
    with torch.no_grad():
        koopman.A.copy_(torch.eye(4) * 0.7)
        koopman.B.fill_(0.2)
    actor = CostActor(
        3,
        1,
        hidden_dims=(8,),
        q_max=2.0,
        p_max=1.0,
        observation_dim=6,
    )
    policy = KoopmanLQRPolicy(
        koopman,
        actor,
        Critic(6, hidden_dims=(8,)),
        torch.zeros(6),
        torch.ones(6),
        dare_tolerance=1e-8,
        dare_max_iterations=1000,
    )

    observations = torch.randn(4, 6)
    q_diag, p = actor(observations)
    assert q_diag.shape == (4, 4)
    assert p.shape == (4, 4)
    output = policy(observations)
    assert output.mean.shape == (4, 1)
    assert output.value.shape == (4,)
    assert GainHoldController(policy, gain_update_interval=2).act(
        observations[0]
    ).shape == (1,)


def test_policy_uses_reconstructable_fallback_after_failed_retry():
    torch.manual_seed(3)
    koopman = DeepKoopman(3, 1, lift_dim=1, hidden_dims=(4,))
    with torch.no_grad():
        koopman.A.copy_(torch.eye(4) * 0.7)
        koopman.B.fill_(0.2)
    policy = KoopmanLQRPolicy(
        koopman,
        CostActor(3, 1, hidden_dims=(8,), q_max=2.0, p_max=1.0),
        Critic(3, hidden_dims=(8,)),
        torch.zeros(3),
        torch.ones(3),
        dare_tolerance=1e-30,
        dare_max_iterations=1,
        dare_jitter=1e-9,
        retry_max_iterations=1,
        retry_jitter_multiplier=1.0,
        fallback_delta_limit=0.25,
    )
    observations = torch.randn(5, 3)
    first = policy(observations)
    second = policy(observations)
    assert torch.all(first.solver_retry_used)
    assert torch.all(first.solver_fallback_used)
    assert not torch.any(first.solver_valid)
    assert torch.isfinite(first.mean).all()
    assert torch.max(torch.abs(first.mean)) <= 0.25
    torch.testing.assert_close(first.mean, second.mean)


def test_policy_recovers_with_second_dare_attempt():
    torch.manual_seed(4)
    koopman = DeepKoopman(3, 1, lift_dim=1, hidden_dims=(4,))
    with torch.no_grad():
        koopman.A.copy_(torch.eye(4) * 0.7)
        koopman.B.fill_(0.2)
    policy = KoopmanLQRPolicy(
        koopman,
        CostActor(3, 1, hidden_dims=(8,), q_max=2.0, p_max=1.0),
        Critic(3, hidden_dims=(8,)),
        torch.zeros(3),
        torch.ones(3),
        dare_tolerance=1e-8,
        dare_max_iterations=1,
        retry_max_iterations=100,
    )
    output = policy(torch.randn(3, 3))
    assert torch.all(output.solver_retry_used)
    assert torch.all(output.solver_valid)
    assert not torch.any(output.solver_fallback_used)
    assert torch.isfinite(output.mean).all()
