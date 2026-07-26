import copy

import torch

from antmaze_ac.koopman.model import DeepKoopman
from antmaze_ac.rl.ac_koopman_policy import KoopmanLQRPolicy
from antmaze_ac.rl.cost_actor import CostActor
from antmaze_ac.rl.critic import Critic
from antmaze_ac.rl.td3_bc import (
    OfflineTransitionBatch,
    TD3BCTrainer,
    TwinActionValueCritic,
    offline_validation_metrics,
)


def make_system():
    torch.manual_seed(9)
    state_dim, action_dim = 3, 1
    koopman = DeepKoopman(state_dim, action_dim, lift_dim=1, hidden_dims=(4,))
    with torch.no_grad():
        koopman.A.copy_(torch.eye(4) * 0.7)
        koopman.B.fill_(0.2)
    policy = KoopmanLQRPolicy(
        koopman,
        CostActor(
            state_dim,
            action_dim,
            hidden_dims=(8,),
            q_max=2.0,
            p_max=1.0,
        ),
        Critic(state_dim, hidden_dims=(8,)),
        torch.zeros(state_dim),
        torch.ones(state_dim),
        dare_tolerance=1e-8,
        dare_max_iterations=100,
        mean_action_limit=2.0,
    )
    policy.requires_grad_(False)
    policy.actor.requires_grad_(True)
    target_policy = copy.deepcopy(policy)
    target_policy.requires_grad_(False)
    critic = TwinActionValueCritic(
        state_dim,
        action_dim,
        torch.zeros(state_dim),
        torch.ones(state_dim),
        hidden_dims=(8,),
        action_scale=2.0,
    )
    target_critic = copy.deepcopy(critic)
    target_critic.requires_grad_(False)
    trainer = TD3BCTrainer(
        policy,
        target_policy,
        critic,
        target_critic,
        torch.optim.Adam(policy.actor.parameters(), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        discount=0.99,
        tau=0.01,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_frequency=2,
        alpha=2.5,
        bc_weight=1.0,
        bc_warmup_steps=10,
        max_delta_action=2.0,
        reward_scale=1.0,
        reward_bias=0.0,
        max_grad_norm=1.0,
    )
    return policy, critic, trainer


def test_td3_bc_update_preserves_dare_actor_and_action_support():
    policy, critic, trainer = make_system()
    batch = OfflineTransitionBatch(
        state=torch.randn(8, 3),
        action=torch.empty(8, 1).uniform_(-2.0, 2.0),
        next_state=torch.randn(8, 3),
        reward=torch.tensor([0, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float32),
        done=torch.tensor([0, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float32),
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.actor.named_parameters()
    }
    warmup_metrics = trainer.update(batch, gradient_step=2)
    assert warmup_metrics["actor_updated"] == 1
    assert warmup_metrics["td3_bc_lambda"] == 0.0
    assert warmup_metrics["behavior_cloning_loss"] is not None
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in policy.actor.named_parameters()
    )
    output = policy(batch.state)
    assert torch.max(torch.abs(output.mean)) <= 2.0
    assert torch.isfinite(output.mean).all()
    assert torch.all(output.solver_valid)

    td3_metrics = trainer.update(batch, gradient_step=12)
    assert td3_metrics["actor_updated"] == 1
    assert float(td3_metrics["td3_bc_lambda"]) > 0.0
    validation = offline_validation_metrics(policy, critic, batch)
    assert validation["behavior_cloning_loss"] >= 0.0
    assert validation["dare_fallback_fraction"] == 0.0
