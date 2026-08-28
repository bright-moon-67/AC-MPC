from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mani_skill")

from experiments.maniskill_pick_visual.train_visual_pickcube_ppo import (
    PPOConfig,
    _build_actor,
    _predicted_state29,
    compute_mpve_td_k_targets,
    compute_timeout_correct_gae,
    ppo_value_loss,
)


def test_published_batch_arithmetic_is_preserved_at_128_envs() -> None:
    config = PPOConfig()
    assert config.num_envs == 128
    assert config.rollout_steps == 16
    assert config.collection_chunks == 8
    assert config.batch_size == 16_384
    assert config.effective_minibatch_size() == 512
    assert config.update_epochs == 8
    assert config.total_timesteps // config.batch_size == 3_051
    assert (config.total_timesteps // config.batch_size) * config.batch_size == 49_987_584


def test_timeout_bootstraps_final_observation_but_stops_trace() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros_like(rewards)
    following = torch.tensor([[0.0], [10.0], [4.0]])
    boundaries = torch.tensor([[0.0], [1.0], [0.0]])
    advantages, returns = compute_timeout_correct_gae(
        rewards,
        values,
        following,
        boundaries,
        gamma=0.5,
        gae_lambda=1.0,
    )
    # t=1 uses final V=10, but cannot inherit t=2's advantage.
    assert torch.allclose(advantages[:, 0], torch.tensor([4.5, 7.0, 5.0]))
    assert torch.equal(returns, advantages)


def test_official_value_loss_has_leading_half() -> None:
    current = torch.tensor([2.0, 4.0])
    target = torch.tensor([0.0, 0.0])
    behavior = torch.zeros(2)
    loss = ppo_value_loss(
        current,
        target,
        behavior,
        clip_value_loss=False,
        clip_ratio=0.2,
    )
    assert loss.item() == 5.0


def test_mpve_targets_are_detached_td_k_returns() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    terminal = torch.tensor([4.0], requires_grad=True)
    targets = compute_mpve_td_k_targets(rewards, terminal, 0.5)
    assert torch.allclose(targets, torch.tensor([[3.25, 4.5, 5.0]]))
    assert not targets.requires_grad


def test_mpve_and_kmpc_build_identical_actor() -> None:
    koopman = SimpleNamespace(
        A=torch.eye(6),
        B=torch.ones(6, 2),
        C=torch.eye(6),
        lifted_dim=6,
    )
    config = PPOConfig(kmpc_horizon=2, mpve_horizon=2)
    torch.manual_seed(7)
    kmpc = _build_actor("KMPC", koopman, config, torch.device("cpu"))
    torch.manual_seed(7)
    mpve = _build_actor("AC-MPC-MPVE", koopman, config, torch.device("cpu"))
    for key, value in kmpc.state_dict().items():
        assert torch.equal(value, mpve.state_dict()[key])


def test_predicted_robot_replaces_only_robot_fields() -> None:
    base = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    normalized = torch.arange(21, dtype=torch.float32).unsqueeze(0) + 100
    predicted = _predicted_state29(
        base,
        normalized,
        torch.zeros(21),
        torch.ones(21),
    )
    assert torch.equal(predicted[0, :18], normalized[0, :18])
    assert torch.equal(predicted[0, 19:22], normalized[0, 18:21])
    assert torch.equal(predicted[0, 18:19], base[0, 18:19])
    assert torch.equal(predicted[0, 22:], base[0, 22:])
