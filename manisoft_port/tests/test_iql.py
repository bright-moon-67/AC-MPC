import copy
import math

import numpy as np
import torch

from antmaze_ac.data.offline_transition_dataset import OfflineTransitionDataset
from antmaze_ac.koopman.checkpoint import save_checkpoint
from antmaze_ac.koopman.history_model import HistoryDeepKoopman
from antmaze_ac.rl.iql import (
    IQLActionValue,
    IQLTrainer,
    IQLTransitionBatch,
    IQLValue,
    expectile_loss,
    policy_features,
)
from antmaze_ac.rl.manisoft_ppo_policies import make_manisoft_ppo_policy


def _koopman_checkpoint(tmp_path):
    torch.manual_seed(13)
    model = HistoryDeepKoopman(
        state_dim=45,
        action_dim=3,
        lift_dim=2,
        hidden_dims=(8,),
        history_steps=2,
    )
    path = tmp_path / "koopman.pt"
    save_checkpoint(
        path,
        model,
        epoch=0,
        best_validation=0.0,
        config={},
        normalizers={
            "state": {"mean": torch.zeros(45), "std": torch.ones(45)}
        },
        elapsed_seconds=0.0,
    )
    return path


def _policy(tmp_path):
    policy, _ = make_manisoft_ppo_policy(
        "ppo_kmpc",
        _koopman_checkpoint(tmp_path),
        torch.device("cpu"),
        kmpc_hidden_dims=(12,),
        horizon=2,
        solver_iterations=4,
        max_delta=0.01,
        kmpc_cost_parameterization="structured",
        structured_log_scale=math.log(2.0),
    )
    policy.requires_grad_(False)
    policy.actor.requires_grad_(True)
    policy.log_std.requires_grad_(True)
    return policy


def test_expectile_loss_weights_positive_target_difference_more():
    differences = torch.tensor([-2.0, 2.0])
    loss = expectile_loss(differences, 0.9)
    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_iql_update_trains_kmpc_actor_q_and_value_but_not_koopman(tmp_path):
    policy = _policy(tmp_path)
    feature_dim = (
        policy.koopman.lifted_dim
        + policy.task_context_dim
        + policy.action_dim
    )
    qf1 = IQLActionValue(feature_dim, policy.action_dim, (16, 16))
    qf2 = IQLActionValue(feature_dim, policy.action_dim, (16, 16))
    target_qf1 = copy.deepcopy(qf1).requires_grad_(False)
    target_qf2 = copy.deepcopy(qf2).requires_grad_(False)
    vf = IQLValue(feature_dim, (16, 16))
    policy_parameters = list(policy.actor.parameters()) + [policy.log_std]
    trainer = IQLTrainer(
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        vf,
        torch.optim.Adam(policy_parameters, lr=1e-3),
        torch.optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()), lr=1e-3
        ),
        torch.optim.Adam(vf.parameters(), lr=1e-3),
        expectile=0.9,
        temperature=0.1,
        max_advantage_weight=100.0,
        minimum_log_std=math.log(0.001),
        maximum_log_std=math.log(0.2),
    )
    observations = torch.zeros(8, policy.observation_dim)
    observations[:, -3:] = torch.tensor([1.0, 0.0, 0.0])
    next_observations = observations.clone()
    next_observations[:, 30] = 0.001
    batch = IQLTransitionBatch(
        observation=observations,
        action=torch.empty(8, policy.action_dim).uniform_(-0.2, 0.2),
        reward=torch.linspace(-0.1, 1.0, 8),
        next_observation=next_observations,
        terminal=torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32),
    )
    actor_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.actor.named_parameters()
    }
    q_before = [parameter.detach().clone() for parameter in qf1.parameters()]
    value_before = [parameter.detach().clone() for parameter in vf.parameters()]
    metrics = trainer.update(batch)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(actor_before[name], parameter)
        for name, parameter in policy.actor.named_parameters()
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(q_before, qf1.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(value_before, vf.parameters(), strict=True)
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in policy.koopman.parameters()
    )
    assert float(policy.log_std.exp().max()) <= 0.2 + 1e-7


def test_iql_features_explicitly_append_normalized_previous_action(tmp_path):
    policy = _policy(tmp_path)
    observations = torch.zeros(2, policy.observation_dim)
    observations[:, -3:] = torch.tensor([1.0, 0.0, 0.0])
    action_history_start = policy.state_dim + policy.history_steps * policy.state_dim
    latest_start = action_history_start + (policy.history_steps - 1) * policy.action_dim
    observations[0, latest_start : latest_start + policy.action_dim] = 0.15
    observations[1, latest_start : latest_start + policy.action_dim] = -0.30
    features = policy_features(policy, observations)
    assert features.shape[-1] == (
        policy.koopman.lifted_dim
        + policy.task_context_dim
        + policy.action_dim
    )
    torch.testing.assert_close(
        features[:, -policy.action_dim :],
        torch.tensor([[0.5] * policy.action_dim, [-1.0] * policy.action_dim]),
    )


def test_critic_warmup_does_not_update_actor(tmp_path):
    policy = _policy(tmp_path)
    feature_dim = policy.koopman.lifted_dim + policy.task_context_dim + policy.action_dim
    qf1 = IQLActionValue(feature_dim, policy.action_dim, (8,))
    qf2 = IQLActionValue(feature_dim, policy.action_dim, (8,))
    vf = IQLValue(feature_dim, (8,))
    trainer = IQLTrainer(
        policy,
        qf1,
        qf2,
        copy.deepcopy(qf1).requires_grad_(False),
        copy.deepcopy(qf2).requires_grad_(False),
        vf,
        torch.optim.Adam(policy.actor.parameters(), lr=1e-3),
        torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=1e-3),
        torch.optim.Adam(vf.parameters(), lr=1e-3),
    )
    observations = torch.zeros(4, policy.observation_dim)
    observations[:, -3:] = torch.tensor([1.0, 0.0, 0.0])
    batch = IQLTransitionBatch(
        observation=observations,
        action=torch.zeros(4, policy.action_dim),
        reward=torch.ones(4),
        next_observation=observations.clone(),
        terminal=torch.zeros(4),
    )
    actor_before = [p.detach().clone() for p in policy.actor.parameters()]
    q_before = [p.detach().clone() for p in qf1.parameters()]
    metrics = trainer.update(batch, update_policy=False)
    assert float(metrics["policy_updated"]) == 0.0
    assert all(
        torch.equal(before, after)
        for before, after in zip(actor_before, policy.actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(q_before, qf1.parameters(), strict=True)
    )


def test_offline_dataset_memmap_split_and_timeout_semantics(tmp_path):
    count, observation_dim, action_dim = 8, 5, 2
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        observations=np.arange(
            count * observation_dim, dtype=np.float32
        ).reshape(count, observation_dim),
        actions=np.zeros((count, action_dim), dtype=np.float32),
        behavior_action_means=np.full(
            (count, action_dim), 0.25, dtype=np.float32
        ),
        rewards=np.linspace(-1, 1, count, dtype=np.float32),
        next_observations=np.ones(
            (count, observation_dim), dtype=np.float32
        ),
        terminals=np.asarray(
            [False, False, False, True, False, False, False, False]
        ),
        timeouts=np.asarray(
            [False, False, False, False, False, False, False, True]
        ),
        episode_ids=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    dataset = OfflineTransitionDataset(
        dataset_path,
        cache_dir=tmp_path / "cache",
        treat_timeouts_as_terminal=True,
    )
    assert dataset.observation_dim == observation_dim
    assert dataset.action_dim == action_dim
    assert isinstance(dataset.arrays["observations"], np.memmap)
    train, validation = dataset.split_by_episode(0.5, seed=4)
    assert len(train) == len(validation) == 4
    assert set(dataset.arrays["episode_ids"][train]).isdisjoint(
        set(dataset.arrays["episode_ids"][validation])
    )
    batch = dataset.sample_batch(
        4,
        np.random.default_rng(2),
        torch.device("cpu"),
        indices=np.asarray([7]),
    )
    assert torch.all(batch.terminal == 1)
    assert batch.behavior_action_mean is not None
    assert torch.all(batch.behavior_action_mean == 0.25)
