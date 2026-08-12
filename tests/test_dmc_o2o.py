from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.dmc.o2o.checkpoint import (
    CHECKPOINT_KIND,
    atomic_torch_save,
    load_checkpoint,
    restore_rng,
    rng_state,
)
from experiments.dmc.o2o.config import O2OConfig
from experiments.dmc.o2o.dataset import (
    DATASET_KIND,
    DATASET_KEYS,
    OfflineDataset,
    OnlineReplay,
    _cartpole_reward,
    convert_exorl_cartpole,
    mixed_batch,
)
from experiments.dmc.o2o.koopman import FrozenKoopman, file_sha256
from experiments.dmc.o2o.learner import O2OLearner, TensorBatch
from experiments.dmc.o2o.networks import (
    KMPCTanhGaussianActor,
    MLPActor,
    QEnsemble,
)
from experiments.dmc.o2o import train as train_module
from experiments.dmc.o2o.train import (
    _load_offline_fork,
    _truncate_metrics_to_checkpoint,
    _validate_resume,
)


def _write_synthetic_koopman(path: Path) -> Path:
    state_dim = 5
    action_dim = 1
    lift_dim = 2
    lifted_dim = state_dim + lift_dim
    encoder_weight = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    encoder_bias = np.asarray([0.25, -0.75], dtype=np.float32)
    matrix_a = np.eye(lifted_dim, dtype=np.float32)
    matrix_b = np.linspace(0.01, 0.07, lifted_dim, dtype=np.float32)[:, None]
    matrix_c = np.concatenate(
        (
            np.eye(state_dim, dtype=np.float32),
            np.zeros((state_dim, lift_dim), dtype=np.float32),
        ),
        axis=1,
    )
    metadata = {
        "kind": "playground_koopman_export_v1",
        "architecture": {
            "architecture": "fullA_history_v2_adapted",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "lift_dim": lift_dim,
            "hidden_dims": [],
            "activation": "silu",
        },
        "encoder_layer_count": 1,
        "reward_layer_count": 0,
        "best_validation_rollout_normalized_mse": 0.01,
    }
    np.savez(
        path,
        A=matrix_a,
        B=matrix_b,
        C=matrix_c,
        center=np.asarray([1.0, -2.0, 0.5, 4.0, -3.0], dtype=np.float32),
        scale=np.asarray([2.0, 4.0, 0.5, 8.0, 1.5], dtype=np.float32),
        encoder_0_weight=encoder_weight,
        encoder_0_bias=encoder_bias,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    return path


@pytest.fixture
def koopman_path(tmp_path: Path) -> Path:
    return _write_synthetic_koopman(tmp_path / "koopman.npz")


@pytest.fixture
def koopman(koopman_path: Path) -> FrozenKoopman:
    return FrozenKoopman(koopman_path)


def _synthetic_exorl_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # ExORL index zero is a reset/dummy record.  Deliberately make its action
    # invalid for a real transition so the test detects an off-by-one loader.
    observation = np.asarray(
        [
            [10.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    action = np.asarray([[99.0], [0.0], [0.5], [-1.0]], dtype=np.float32)
    reward = np.zeros(4, dtype=np.float32)
    reward[1:] = _cartpole_reward(observation[1:], action[1:])
    discount = np.asarray([0.0, 1.0, 0.5, 1.0], dtype=np.float32)
    np.savez(
        path,
        observation=observation,
        action=action,
        reward=reward,
        discount=discount,
    )
    return observation, action


def test_exorl_conversion_aligns_dummy_reward_discount_and_mc_return(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    observation, action = _synthetic_exorl_episode(
        source / "20200101T000000_0_3.npz"
    )
    output = tmp_path / "transitions.npz"

    metadata = convert_exorl_cartpole(
        source, output, max_transitions=3, gamma=0.9
    )
    dataset = OfflineDataset.load(output)

    assert metadata["kind"] == DATASET_KIND
    assert metadata["transitions"] == 3
    assert metadata["episodes"] == 1
    np.testing.assert_array_equal(dataset.arrays["observation"], observation[:-1])
    np.testing.assert_array_equal(
        dataset.arrays["next_observation"], observation[1:]
    )
    np.testing.assert_array_equal(dataset.arrays["action"], action[1:])
    np.testing.assert_array_equal(
        dataset.arrays["discount"], np.asarray([1.0, 0.5, 1.0])
    )
    expected_reward = np.asarray([0.5, 0.95, 0.0], dtype=np.float32)
    np.testing.assert_allclose(dataset.arrays["reward"], expected_reward, atol=1e-7)
    np.testing.assert_allclose(
        dataset.arrays["mc_return"],
        np.asarray([0.5 + 0.9 * 0.95, 0.95, 0.0], dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_array_equal(dataset.arrays["episode_step"], [0, 1, 2])
    assert dataset.sha256 == metadata["output_sha256"]


def test_exorl_conversion_rejects_recorded_reward_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    path = source / "20200101T000000_0_3.npz"
    _synthetic_exorl_episode(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["reward"] = arrays["reward"].copy()
    arrays["reward"][2] += 0.01
    np.savez(path, **arrays)

    with pytest.raises(AssertionError, match="reward parity"):
        convert_exorl_cartpole(source, tmp_path / "transitions.npz")


def test_frozen_koopman_normalizes_lifts_steps_and_reconstructs(
    koopman: FrozenKoopman, koopman_path: Path
) -> None:
    observation = torch.tensor(
        [[3.0, 2.0, 1.0, 12.0, 0.0]], dtype=torch.float32
    )
    normalized = koopman.normalize(observation)
    np.testing.assert_allclose(
        normalized.numpy(), [[1.0, 1.0, 1.0, 1.0, 2.0]], atol=1e-7
    )
    expected_encoded = torch.tensor([[1.25, -1.25]], dtype=torch.float32)
    lifted = koopman.lift(observation)
    torch.testing.assert_close(lifted[:, :5], normalized)
    torch.testing.assert_close(lifted[:, 5:], expected_encoded)

    action = torch.tensor([[0.5]], dtype=torch.float32)
    following = koopman.step(lifted, action)
    expected_following = lifted + 0.5 * koopman.B.T
    torch.testing.assert_close(following, expected_following)
    torch.testing.assert_close(koopman.reconstruct(lifted), observation)
    assert koopman.sha256 == file_sha256(koopman_path)
    assert koopman.identity()["architecture"]["lift_dim"] == 2
    assert all(not parameter.requires_grad for parameter in koopman.parameters())


def test_mlp_tanh_policy_has_finite_bounded_reparameterized_samples() -> None:
    torch.manual_seed(1)
    actor = MLPActor(lifted_dim=7, action_dim=1, hidden_dim=16)
    lifted = torch.randn(5, 7)

    deterministic, deterministic_log_prob, plan = actor.sample(
        lifted, deterministic=True
    )
    stochastic, stochastic_log_prob, _ = actor.sample(lifted, samples=3)

    assert deterministic.shape == (5, 1)
    assert deterministic_log_prob.shape == (5,)
    assert stochastic.shape == (3, 5, 1)
    assert stochastic_log_prob.shape == (3, 5)
    assert plan is None
    assert torch.isfinite(deterministic_log_prob).all()
    assert torch.isfinite(stochastic_log_prob).all()
    assert torch.all(deterministic.abs() < 1.0)
    assert torch.all(stochastic.abs() < 1.0)
    (stochastic.mean() + stochastic_log_prob.mean()).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )


def test_mlp_log_std_matches_rlpd_direct_clipping_parameterization() -> None:
    actor = MLPActor(lifted_dim=7, action_dim=3, hidden_dim=8)
    output = actor.net[-1]
    with torch.no_grad():
        output.weight.zero_()
        output.bias.zero_()
        output.bias[3:] = torch.tensor([-30.0, 0.0, 3.0])

    _location, log_std = actor.distribution(torch.zeros(2, 7))

    torch.testing.assert_close(
        log_std,
        torch.tensor([[-20.0, 0.0, 2.0], [-20.0, 0.0, 2.0]]),
    )


def test_kmpc_tanh_policy_returns_plan_and_actor_gradients(
    koopman: FrozenKoopman,
) -> None:
    torch.manual_seed(2)
    actor = KMPCTanhGaussianActor(
        koopman, horizon=3, solver_iterations=3, hidden_dim=12
    )
    observation = torch.tensor(
        [
            [3.0, 2.0, 1.0, 12.0, 0.0],
            [2.0, -1.0, 0.5, 5.0, -1.5],
        ],
        dtype=torch.float32,
    )
    lifted = koopman.lift(observation)
    action, log_prob, plan = actor.sample(
        lifted, deterministic=True, return_plan=True
    )

    assert plan is not None
    assert plan.shape == (2, 3, 1)
    assert action.shape == (2, 1)
    assert log_prob.shape == (2,)
    torch.testing.assert_close(action, plan[:, 0], atol=2e-6, rtol=0.0)
    assert torch.isfinite(plan).all()
    assert torch.all(plan.abs() <= 1.0)
    (action.mean() + 0.01 * log_prob.mean()).backward()
    final = actor.controller[-1]
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()
    assert torch.count_nonzero(final.weight.grad) > 0
    assert all(parameter.grad is None for parameter in koopman.parameters())


def test_q_ensemble_is_vectorized_and_each_head_receives_gradients() -> None:
    torch.manual_seed(3)
    critic = QEnsemble(
        lifted_dim=7,
        action_dim=1,
        ensemble_size=4,
        hidden_dim=12,
        hidden_layers=2,
    )
    lifted = torch.randn(6, 7)
    action = torch.rand(6, 1) * 2.0 - 1.0
    value = critic(lifted, action)

    assert value.shape == (4, 6)
    assert torch.isfinite(value).all()
    value.square().mean().backward()
    assert critic.output.weight.grad is not None
    assert critic.output.weight.grad.shape[0] == 4
    assert torch.all(critic.output.weight.grad.flatten(1).norm(dim=1) > 0)


def _offline_dataset_for_mixing(size: int = 32) -> OfflineDataset:
    arrays = {
        "observation": np.full((size, 5), -7.0, dtype=np.float32),
        "action": np.zeros((size, 1), dtype=np.float32),
        "reward": np.zeros(size, dtype=np.float32),
        "discount": np.ones(size, dtype=np.float32),
        "next_observation": np.full((size, 5), -6.0, dtype=np.float32),
        "episode_id": np.zeros(size, dtype=np.int64),
        "episode_step": np.arange(size, dtype=np.int32),
        "mc_return": np.arange(size, dtype=np.float32),
    }
    assert set(arrays) == set(DATASET_KEYS)
    return OfflineDataset(
        arrays=arrays,
        metadata={"kind": DATASET_KIND, "transitions": size},
        path=Path("synthetic.npz"),
        sha256="synthetic",
    )


def test_replay_mixer_produces_exact_per_update_fifty_fifty_batch() -> None:
    offline = _offline_dataset_for_mixing()
    online = OnlineReplay(capacity=16)
    for index in range(8):
        online.add(
            np.full(5, 7.0 + index, dtype=np.float32),
            np.asarray([0.25], dtype=np.float32),
            reward=1.0,
            discount=1.0,
            next_observation=np.full(5, 8.0 + index, dtype=np.float32),
        )
    batch = mixed_batch(
        offline,
        online,
        batch_size=8,
        utd=2,
        offline_ratio=0.5,
        generator=np.random.default_rng(5),
    )

    assert {key: value.shape[0] for key, value in batch.items()} == {
        "observation": 16,
        "action": 16,
        "reward": 16,
        "discount": 16,
        "next_observation": 16,
        "mc_return": 16,
        "offline_mask": 16,
    }
    assert float(batch["offline_mask"].sum()) == 8.0
    for update in range(2):
        update_mask = batch["offline_mask"][update * 8 : (update + 1) * 8]
        assert float(update_mask.sum()) == 4.0
    offline_rows = batch["offline_mask"] == 1.0
    np.testing.assert_array_equal(batch["observation"][offline_rows], -7.0)
    assert np.all(batch["observation"][~offline_rows] >= 7.0)


def test_replay_mixer_rejects_nonintegral_per_update_ratio() -> None:
    offline = _offline_dataset_for_mixing()
    online = OnlineReplay(capacity=4)
    online.add(
        np.zeros(5, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        reward=0.0,
        discount=1.0,
        next_observation=np.zeros(5, dtype=np.float32),
    )
    with pytest.raises(ValueError, match=r"offline_ratio \* batch_size"):
        mixed_batch(
            offline,
            online,
            batch_size=3,
            utd=2,
            offline_ratio=0.5,
            generator=np.random.default_rng(6),
        )


def test_online_replay_checkpoint_zeroes_unwritten_capacity_and_round_trips() -> None:
    replay = OnlineReplay(capacity=4)
    for index in range(2):
        replay.add(
            np.full(5, index + 1, dtype=np.float32),
            np.asarray([0.25 * index], dtype=np.float32),
            reward=0.5 + index,
            discount=1.0,
            next_observation=np.full(5, index + 2, dtype=np.float32),
        )
    state = replay.state_dict()
    for value in state["arrays"].values():
        assert np.count_nonzero(value[2:]) == 0

    restored = OnlineReplay(capacity=4)
    restored.load_state_dict(state)
    assert restored.size == replay.size == 2
    assert restored.cursor == replay.cursor == 2
    for key in replay.arrays:
        np.testing.assert_array_equal(restored.arrays[key], replay.arrays[key])


def _small_config(method: str) -> O2OConfig:
    return O2OConfig(
        method=method,
        device="cpu",
        batch_size=4,
        hidden_dim=12,
        critic_hidden_layers=1,
        critic_ensemble_size=2,
        target_critic_subset=2,
        offline_updates=1,
        cql_actions=2,
        online_steps=1,
        online_utd=1,
        online_warmup_steps=1,
        replay_capacity=8,
        num_envs=1,
        env_workers=1,
        kmpc_horizon=3,
        kmpc_solver_iterations=2,
        controller_hidden_dim=8,
        mpve_total_horizon=3,
        eval_interval_online_steps=1,
        eval_episodes=1,
        checkpoint_interval_updates=1,
        log_interval_updates=1,
    )


def _tensor_batch(size: int = 4) -> TensorBatch:
    observation = torch.zeros(size, 5)
    observation[:, 1] = 1.0
    next_observation = observation.clone()
    return TensorBatch(
        observation=observation,
        action=torch.zeros(size, 1),
        reward=torch.linspace(0.1, 0.4, size),
        discount=torch.ones(size),
        next_observation=next_observation,
        mc_return=torch.ones(size),
        offline_mask=torch.ones(size),
    )


def test_rng_substreams_pair_critic_initialization_across_actor_methods(
    koopman_path: Path,
) -> None:
    # Constructing either actor must not consume the caller's default stream.
    torch.manual_seed(123456)
    caller_rng = torch.get_rng_state().clone()
    mlp = O2OLearner(
        _small_config("Cal-RLPD-MLP"),
        FrozenKoopman(koopman_path),
        torch.device("cpu"),
    )
    torch.testing.assert_close(torch.get_rng_state(), caller_rng, rtol=0.0, atol=0.0)
    kmpc = O2OLearner(
        _small_config("Cal-RLPD-AC-KMPC"),
        FrozenKoopman(koopman_path),
        torch.device("cpu"),
    )
    torch.testing.assert_close(torch.get_rng_state(), caller_rng, rtol=0.0, atol=0.0)

    # Actor parameter shapes legitimately differ, but every shared critic and
    # target tensor must be bit-identical under the paired seed.
    for module_name in ("critic", "target_critic"):
        left = getattr(mlp, module_name).state_dict()
        right = getattr(kmpc, module_name).state_dict()
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key]), key

    mlp_rng = mlp.state_dict()["rng_substreams"]
    kmpc_rng = kmpc.state_dict()["rng_substreams"]
    assert mlp_rng["version"] == kmpc_rng["version"]
    assert mlp_rng["base_seed"] == kmpc_rng["base_seed"]
    assert mlp_rng["seeds"] == kmpc_rng["seeds"]
    assert torch.equal(
        mlp_rng["training_sampling_state"],
        kmpc_rng["training_sampling_state"],
    )


def test_mpve_target_is_detached_and_zero_discount_stops_expansion(
    koopman: FrozenKoopman,
) -> None:
    learner = O2OLearner(
        _small_config("Cal-RLPD-AC-KMPC-MPVE"), koopman, torch.device("cpu")
    )
    batch = _tensor_batch()
    batch.discount[0] = 0.0
    real_target = torch.randn(4, requires_grad=True)

    model_target = learner._mpve_target(batch, real_target)

    assert model_target.shape == (4,)
    assert torch.isfinite(model_target).all()
    assert not model_target.requires_grad
    assert model_target.grad_fn is None
    torch.testing.assert_close(model_target[0], batch.reward[0])
    assert all(parameter.grad is None for parameter in learner.actor.parameters())
    assert all(parameter.grad is None for parameter in learner.target_critic.parameters())


def test_calql_critic_and_sac_actor_updates_keep_gradients_separate(
    koopman: FrozenKoopman,
) -> None:
    learner = O2OLearner(
        _small_config("Cal-RLPD-MLP"), koopman, torch.device("cpu")
    )
    batch = _tensor_batch()

    learner.update_critic(batch, apply_mpve=False)
    assert all(parameter.grad is None for parameter in learner.actor.parameters())

    before = [
        parameter.detach().clone()
        for parameter in learner.actor.parameters()
        if parameter.requires_grad
    ]
    metrics = learner.update_actor_and_temperature(batch)
    after = [
        parameter.detach()
        for parameter in learner.actor.parameters()
        if parameter.requires_grad
    ]

    assert all(parameter.grad is None for parameter in learner.critic.parameters())
    assert any(not torch.equal(left, right) for left, right in zip(before, after))
    assert np.isfinite(list(metrics.values())).all()


@pytest.mark.parametrize("cql_actions", (1, 3))
def test_fused_calql_proposal_cache_shapes_and_offline_mask(
    koopman_path: Path, cql_actions: int
) -> None:
    config = dataclasses.replace(
        _small_config("Cal-RLPD-AC-KMPC"), cql_actions=cql_actions
    )
    learner = O2OLearner(
        config, FrozenKoopman(koopman_path), torch.device("cpu")
    )
    batch = _tensor_batch(size=8)
    batch.offline_mask[:] = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0])
    plan_calls = 0
    original_plan = learner.actor.plan  # type: ignore[attr-defined]

    def counted_plan(lifted_state: torch.Tensor) -> torch.Tensor:
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(lifted_state)

    learner.actor.plan = counted_plan  # type: ignore[attr-defined,method-assign]
    sampled: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_sample = learner.actor.sample

    def recorded_sample(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        sampled.append((result[0].detach().clone(), result[1].detach().clone()))
        return result

    learner.actor.sample = recorded_sample  # type: ignore[method-assign]
    cache = learner._prepare_critic_cache(batch)

    assert plan_calls == 2  # one full current batch and one full next batch
    assert cache.lifted.shape == cache.next_lifted.shape == (8, 7)
    assert cache.target_next_action.shape == (8, 1)
    assert cache.target_next_log_prob.shape == (8,)
    assert cache.cql_current_actions is not None
    assert cache.cql_current_log_prob is not None
    assert cache.cql_next_actions is not None
    assert cache.cql_next_log_prob is not None
    assert cache.cql_current_actions.shape == (cql_actions, 8, 1)
    assert cache.cql_current_log_prob.shape == (cql_actions, 8)
    assert cache.cql_next_actions.shape == (cql_actions, 8, 1)
    assert cache.cql_next_log_prob.shape == (cql_actions, 8)
    assert len(sampled) == 2
    torch.testing.assert_close(cache.target_next_action, sampled[1][0][0])
    torch.testing.assert_close(
        cache.target_next_log_prob, sampled[1][1][0]
    )
    torch.testing.assert_close(cache.cql_next_actions, sampled[1][0][1:])
    torch.testing.assert_close(cache.cql_next_log_prob, sampled[1][1][1:])
    assert all(
        not value.requires_grad
        for value in dataclasses.astuple(cache)
        if isinstance(value, torch.Tensor)
    )

    slices = (cache.slice(slice(0, 3)), cache.slice(slice(3, 8)))
    torch.testing.assert_close(
        torch.cat([part.lifted for part in slices], dim=0), cache.lifted
    )
    torch.testing.assert_close(
        torch.cat([part.target_next_action for part in slices], dim=0),
        cache.target_next_action,
    )
    torch.testing.assert_close(
        torch.cat([part.cql_current_actions for part in slices], dim=1),
        cache.cql_current_actions,
    )
    torch.testing.assert_close(
        torch.cat([part.cql_next_log_prob for part in slices], dim=1),
        cache.cql_next_log_prob,
    )

    # CQL must select only the three offline rows, while evaluating all three
    # proposal families with the current (not cached) critic parameters.
    seen_shapes: list[tuple[torch.Size, torch.Size]] = []

    def record_shapes(_module, inputs) -> None:
        seen_shapes.append((inputs[0].shape, inputs[1].shape))

    data_q = learner.critic(cache.lifted, batch.action)
    handle = learner.critic.register_forward_pre_hook(record_shapes)
    penalty, _metrics = learner._cql_calibrated_penalty(batch, cache, data_q)
    handle.remove()
    assert torch.isfinite(penalty)
    assert seen_shapes == [
        (torch.Size((cql_actions * 3, 7)), torch.Size((cql_actions * 3, 1)))
    ] * 3


def test_non_calql_cache_contains_only_one_target_policy_sample(
    koopman: FrozenKoopman,
) -> None:
    learner = O2OLearner(
        _small_config("RLPD-MLP"), koopman, torch.device("cpu")
    )
    cache = learner._prepare_critic_cache(_tensor_batch(size=6))

    assert cache.target_next_action.shape == (6, 1)
    assert cache.target_next_log_prob.shape == (6,)
    assert cache.cql_current_actions is None
    assert cache.cql_current_log_prob is None
    assert cache.cql_next_actions is None
    assert cache.cql_next_log_prob is None


def test_kmpc_fused_cache_reduces_plan_calls_but_recomputes_every_q(
    koopman_path: Path,
) -> None:
    config = dataclasses.replace(
        _small_config("Cal-RLPD-AC-KMPC-MPVE"),
        cql_actions=3,
        kmpc_horizon=10,
        mpve_total_horizon=10,
    )

    def run(phase: str) -> tuple[int, int, int, O2OLearner]:
        learner = O2OLearner(
            config, FrozenKoopman(koopman_path), torch.device("cpu")
        )
        plan_calls = 0
        target_q_calls = 0
        current_q_calls = 0
        original_plan = learner.actor.plan  # type: ignore[attr-defined]

        def counted_plan(lifted_state: torch.Tensor) -> torch.Tensor:
            nonlocal plan_calls
            plan_calls += 1
            return original_plan(lifted_state)

        def count_target(_module, _inputs, _output) -> None:
            nonlocal target_q_calls
            target_q_calls += 1

        def count_current(_module, _inputs, _output) -> None:
            nonlocal current_q_calls
            current_q_calls += 1

        learner.actor.plan = counted_plan  # type: ignore[attr-defined,method-assign]
        target_handle = learner.target_critic.register_forward_hook(count_target)
        current_handle = learner.critic.register_forward_hook(count_current)
        learner.update(_tensor_batch(size=80), utd=20, phase=phase)  # type: ignore[arg-type]
        target_handle.remove()
        current_handle.remove()
        return plan_calls, target_q_calls, current_q_calls, learner

    offline_plan, offline_target_q, offline_current_q, offline_learner = run(
        "offline"
    )
    # Two full-batch proposal plans plus one fresh differentiable actor plan;
    # this is independent of UTD=20 (the uncached implementation used 61).
    assert offline_plan == 3
    assert offline_target_q == 20
    # Per critic update: data Q + random/current/next CQL Q, then one actor Q.
    assert offline_current_q == 20 * 4 + 1
    assert offline_learner.gradient_updates == 20
    assert offline_learner.actor_updates == 1

    online_plan, online_target_q, online_current_q, online_learner = run("online")
    # Online MPVE adds nine rollout plans and one terminal-action plan exactly
    # once, while the final actor update remains a fresh eleventh plan.
    assert online_plan == 2 + 10 + 1
    assert online_target_q == 20 + 1  # twenty REDQ targets + MPVE bootstrap
    assert online_current_q == 20 * 4 + 1
    assert online_learner.gradient_updates == 20
    assert online_learner.actor_updates == 1


def test_mpve_runs_online_only_once_per_real_step_and_uses_nine_model_steps(
    koopman_path: Path,
) -> None:
    config = dataclasses.replace(
        _small_config("Cal-RLPD-AC-KMPC-MPVE"),
        kmpc_horizon=10,
        mpve_total_horizon=10,
    )
    learner = O2OLearner(
        config, FrozenKoopman(koopman_path), torch.device("cpu")
    )
    model_steps = 0
    original_step = learner.koopman.step

    def counted_step(lifted_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        nonlocal model_steps
        model_steps += 1
        return original_step(lifted_state, action)

    learner.koopman.step = counted_step  # type: ignore[method-assign]

    offline_metrics = learner.update(
        _tensor_batch(size=8), utd=2, phase="offline"
    )
    assert model_steps == 0
    assert offline_metrics["mpve_applied"] == 0.0

    online_metrics = learner.update(
        _tensor_batch(size=8), utd=2, phase="online"
    )
    assert model_steps == 9
    assert online_metrics["mpve_applied"] == 1.0
    assert online_metrics["mpve_loss"] > 0.0
    assert learner.gradient_updates == 4
    assert learner.actor_updates == 2

    with pytest.raises(ValueError, match="phase"):
        learner.update(_tensor_batch(), utd=1, phase="typo")  # type: ignore[arg-type]


def test_checkpoint_round_trip_restores_learner_replay_and_rng(
    tmp_path: Path, koopman_path: Path
) -> None:
    config = _small_config("Cal-RLPD-AC-KMPC")
    learner = O2OLearner(
        config, FrozenKoopman(koopman_path), torch.device("cpu")
    )
    torch.manual_seed(90)
    learner.update(_tensor_batch(), utd=1, phase="offline")
    replay = OnlineReplay(capacity=8)
    replay.add(
        np.arange(5, dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
        reward=0.75,
        discount=1.0,
        next_observation=np.arange(5, dtype=np.float32) + 1.0,
    )
    generator = np.random.default_rng(91)
    random.seed(92)
    np.random.seed(93)
    torch.manual_seed(94)
    captured_rng = rng_state(generator)
    payload = {
        "kind": CHECKPOINT_KIND,
        "learner": learner.state_dict(),
        "online_replay": replay.state_dict(),
        "rng": captured_rng,
        "dataset_sha256": "dataset",
        "koopman_sha256": learner.koopman.sha256,
        "config_fingerprint": config.fingerprint,
    }
    checkpoint_path = tmp_path / "latest.pt"
    atomic_torch_save(checkpoint_path, payload)

    loaded = load_checkpoint(checkpoint_path)
    restored_learner = O2OLearner(
        config, FrozenKoopman(koopman_path), torch.device("cpu")
    )
    restored_learner.load_state_dict(loaded["learner"])
    assert restored_learner.gradient_updates == learner.gradient_updates == 1
    assert restored_learner.actor_updates == learner.actor_updates == 1
    for optimizer in (
        restored_learner.actor_optimizer,
        restored_learner.critic_optimizer,
        restored_learner.temperature_optimizer,
    ):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                for key, value in optimizer.state.get(parameter, {}).items():
                    if isinstance(value, torch.Tensor):
                        assert value.device == parameter.device or (
                            key == "step" and value.device.type == "cpu"
                        )
    restored_replay = OnlineReplay(capacity=8)
    restored_replay.load_state_dict(loaded["online_replay"])
    restored_generator = np.random.default_rng(0)
    restore_rng(loaded["rng"], restored_generator)

    observation = np.asarray([1.0, -2.0, 0.5, 4.0, -3.0], dtype=np.float32)
    np.testing.assert_allclose(
        learner.act(observation, deterministic=True),
        restored_learner.act(observation, deterministic=True),
        rtol=0.0,
        atol=0.0,
    )
    assert restored_replay.size == replay.size == 1
    assert restored_replay.cursor == replay.cursor == 1
    for key in replay.arrays:
        np.testing.assert_array_equal(restored_replay.arrays[key], replay.arrays[key])

    expected = np.random.default_rng()
    expected.bit_generator.state = copy.deepcopy(captured_rng["numpy_generator"])
    assert restored_generator.random() == expected.random()
    assert loaded["dataset_sha256"] == "dataset"
    assert loaded["koopman_sha256"] == file_sha256(koopman_path)

    # Optimizer moments and Torch RNG are part of a real resume contract, not
    # just model inference.  The first post-resume stochastic update must be
    # bit-identical on CPU.
    # Learner sampling is a private checkpointed substream: deliberately use
    # different caller-global Torch states for the two continuation updates.
    torch.manual_seed(1234)
    learner.update(_tensor_batch(), utd=1, phase="offline")
    torch.manual_seed(9876)
    restored_learner.update(_tensor_batch(), utd=1, phase="offline")
    for name in ("actor", "critic", "target_critic"):
        expected_state = getattr(learner, name).state_dict()
        actual_state = getattr(restored_learner, name).state_dict()
        assert expected_state.keys() == actual_state.keys()
        for key in expected_state:
            torch.testing.assert_close(
                actual_state[key], expected_state[key], rtol=0.0, atol=0.0
            )
    torch.testing.assert_close(
        restored_learner.log_temperature,
        learner.log_temperature,
        rtol=0.0,
        atol=0.0,
    )


def test_deterministic_evaluation_can_skip_device_specific_sampling_rng(
    koopman_path: Path,
) -> None:
    config = _small_config("Cal-RLPD-MLP")
    source = O2OLearner(
        config, FrozenKoopman(koopman_path), torch.device("cpu")
    )
    state = copy.deepcopy(source.state_dict())
    # Stand in for a CUDA Philox byte layout, which a CPU MT19937 generator
    # cannot restore.  Learned parameters are nevertheless portable.
    state["rng_substreams"]["training_sampling_state"] = torch.zeros(
        3, dtype=torch.uint8
    )
    evaluated = O2OLearner(
        config, FrozenKoopman(koopman_path), torch.device("cpu")
    )
    evaluated.load_state_dict(state, restore_sampling_rng=False)
    for key, value in source.actor.state_dict().items():
        assert torch.equal(evaluated.actor.state_dict()[key], value)

    strict_resume = O2OLearner(
        config, FrozenKoopman(koopman_path), torch.device("cpu")
    )
    with pytest.raises(ValueError, match="incompatible with the restore device"):
        strict_resume.load_state_dict(state)


def test_checkpoint_rejects_an_unrelated_payload(tmp_path: Path) -> None:
    path = tmp_path / "not_o2o.pt"
    torch.save({"kind": "other"}, path)
    with pytest.raises(ValueError, match="Unsupported O2O checkpoint"):
        load_checkpoint(path)


def test_mpve_offline_fork_accepts_only_completed_paired_ac_kmpc_checkpoint(
    tmp_path: Path, koopman_path: Path
) -> None:
    source_config = _small_config("Cal-RLPD-AC-KMPC")
    target_config = _small_config("Cal-RLPD-AC-KMPC-MPVE")
    dataset = _offline_dataset_for_mixing()
    koopman = FrozenKoopman(koopman_path)
    environment_protocol = {
        "protocol_name": "synthetic_dmc_native_v1",
        "task": "cartpole_swingup",
        "obs_dim": 5,
        "action_dim": 1,
    }
    checkpoint = {
        "kind": CHECKPOINT_KIND,
        "config": source_config.to_dict(),
        "config_fingerprint": source_config.fingerprint,
        "dataset": {"sha256": dataset.sha256},
        "koopman": koopman.identity(),
        "environment_protocol": environment_protocol,
        "phase": "offline",
        "offline_update": source_config.offline_updates,
        "online_step": 0,
        "online_episode": 0,
    }
    path = tmp_path / "offline.pt"

    def write_and_load(payload: dict[str, object]):
        atomic_torch_save(path, payload)
        return _load_offline_fork(
            path,
            target_config=target_config,
            dataset=dataset,
            koopman=koopman,
            environment_protocol=environment_protocol,
        )

    loaded, identity = write_and_load(checkpoint)
    assert loaded["phase"] == "offline"
    assert identity["kind"] == "acmpc_o2o_offline_fork_v1"
    assert identity["source_method"] == "Cal-RLPD-AC-KMPC"
    assert identity["source_sha256"] == file_sha256(path)

    mpve_resume = copy.deepcopy(checkpoint)
    mpve_resume["config"] = target_config.to_dict()
    mpve_resume["config_fingerprint"] = target_config.fingerprint
    mpve_resume["initialization"] = identity
    _validate_resume(
        mpve_resume,
        target_config,
        dataset,
        koopman,
        environment_protocol,
    )
    missing_lineage = copy.deepcopy(mpve_resume)
    missing_lineage["initialization"] = None
    with pytest.raises(ValueError, match="missing its offline-fork lineage"):
        _validate_resume(
            missing_lineage,
            target_config,
            dataset,
            koopman,
            environment_protocol,
        )

    invalid_phase = copy.deepcopy(checkpoint)
    invalid_phase["phase"] = "online"
    with pytest.raises(ValueError, match="completed pre-online"):
        write_and_load(invalid_phase)

    invalid_count = copy.deepcopy(checkpoint)
    invalid_count["offline_update"] = source_config.offline_updates - 1
    with pytest.raises(ValueError, match="completed pre-online"):
        write_and_load(invalid_count)

    invalid_online_step = copy.deepcopy(checkpoint)
    invalid_online_step["online_step"] = 1
    with pytest.raises(ValueError, match="completed pre-online"):
        write_and_load(invalid_online_step)

    invalid_online_episode = copy.deepcopy(checkpoint)
    invalid_online_episode["online_episode"] = 1
    with pytest.raises(ValueError, match="completed pre-online"):
        write_and_load(invalid_online_episode)

    changed_seed = dataclasses.replace(target_config, seed=target_config.seed + 1)
    atomic_torch_save(path, checkpoint)
    with pytest.raises(ValueError, match="config differs"):
        _load_offline_fork(
            path,
            target_config=changed_seed,
            dataset=dataset,
            koopman=koopman,
            environment_protocol=environment_protocol,
        )


def test_metrics_resume_truncation_uses_latest_checkpoint_counters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    retained = [
        {
            "phase": "offline_evaluation",
            "offline_update": 1,
            "online_step": 0,
            "return_mean": 10.0,
        },
        {
            "phase": "online_episode",
            "offline_update": 1,
            "online_step": 6,
            "episode": 0,
            "episode_return": 20.0,
        },
        {
            "phase": "online_episode",
            "offline_update": 1,
            "online_step": 7,
            "episode": 1,
            "episode_return": 30.0,
        },
    ]
    stale = [
        {
            "phase": "online_evaluation",
            "offline_update": 1,
            "online_step": 10,
            "return_mean": 100.0,
        },
        {
            "phase": "online_episode",
            "offline_update": 1,
            "online_step": 11,
            "episode": 2,
            "episode_return": 40.0,
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in (*retained, *stale))
        + '{"phase":',
        encoding="utf-8",
    )
    checkpoint = {
        "offline_update": 1,
        "online_step": 7,
        "online_episode": 2,
    }

    _truncate_metrics_to_checkpoint(path, checkpoint)

    actual = [json.loads(line) for line in path.read_text().splitlines()]
    assert actual == retained
    assert not path.with_suffix(".jsonl.tmp").exists()
    # Repeated resume is an exact no-op, not a source of duplicate rows.
    _truncate_metrics_to_checkpoint(path, checkpoint)
    assert [json.loads(line) for line in path.read_text().splitlines()] == retained


class _FakeProtocolEnvironment:
    def __init__(self, protocol: dict[str, object]) -> None:
        self._protocol = protocol

    def protocol_metadata(self) -> dict[str, object]:
        return dict(self._protocol)

    def close(self) -> None:
        pass


class _FakeVectorEnvironment:
    def __init__(
        self, protocol: dict[str, object], *, num_envs: int, seed: int
    ) -> None:
        self.protocol = dict(protocol)
        self.num_envs = num_envs
        self.seed = seed
        self.steps = 0
        self.closed = False

    def reset(self) -> np.ndarray:
        self.steps = 0
        return np.zeros((self.num_envs, 5), dtype=np.float32)

    def step(self, action: np.ndarray):
        assert action.shape == (self.num_envs, 1)
        self.steps += 1
        boundary = self.steps == 2
        transition = np.full(
            (self.num_envs, 5), self.steps, dtype=np.float32
        )
        policy_observation = (
            np.zeros_like(transition) if boundary else transition.copy()
        )
        return type(
            "FakeVectorStep",
            (),
            {
                "observation": policy_observation,
                "transition_observation": transition,
                "reward": np.arange(1, self.num_envs + 1, dtype=np.float32),
                "discount": np.ones(self.num_envs, dtype=np.float32),
                "reset_boundary": np.full(
                    self.num_envs, boundary, dtype=np.bool_
                ),
                "reset_seed": np.arange(
                    self.seed, self.seed + self.num_envs, dtype=np.int64
                ),
                "applied_action": np.asarray(action, dtype=np.float32),
            },
        )()

    def close(self) -> None:
        self.closed = True


class _FakeLearner:
    instances: list["_FakeLearner"] = []

    def __init__(self, config: O2OConfig, koopman: FrozenKoopman, device: torch.device):
        del koopman, device
        self.config = config
        self.update_calls: list[tuple[int, str, int]] = []
        self.act_batch_shapes: list[tuple[int, ...]] = []
        self.__class__.instances.append(self)

    def act(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
        del deterministic
        value = np.asarray(observation)
        self.act_batch_shapes.append(tuple(value.shape))
        return np.zeros((*value.shape[:-1], 1), dtype=np.float32)

    def update(self, batch: TensorBatch, utd: int, *, phase: str) -> dict[str, float]:
        self.update_calls.append((int(batch.reward.shape[0]), phase, utd))
        return {"critic_loss": 0.0, "q_mean": 0.0}

    def state_dict(self) -> dict[str, object]:
        return {"update_calls": list(self.update_calls)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.update_calls = list(state["update_calls"])  # type: ignore[arg-type]


def test_training_evaluation_batches_fixed_seed_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol: dict[str, object] = {
        "protocol_name": "synthetic_dmc_native_v1",
        "task": "cartpole_swingup",
        "obs_dim": 5,
        "action_dim": 1,
        "step_limit": 2,
    }
    fake = _FakeVectorEnvironment(protocol, num_envs=5, seed=9_100_000)

    class Actor:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def act(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
            assert deterministic is True
            self.shapes.append(tuple(observation.shape))
            return np.zeros((observation.shape[0], 1), dtype=np.float32)

    def make_vector(task: str, num_envs: int, seed: int, *, workers: int):
        assert (task, num_envs, seed, workers) == (
            "cartpole_swingup",
            5,
            9_100_000,
            1,
        )
        return fake

    monkeypatch.setattr(train_module, "make_dmc_vector_env", make_vector)
    actor = Actor()
    result = train_module.evaluate(actor, episodes=5, seed_base=9_100_000)

    assert actor.shapes == [(5, 5), (5, 5)]
    assert result["returns"] == [2.0, 4.0, 6.0, 8.0, 10.0]
    assert result["return_mean"] == 6.0
    assert result["episode_length_mean"] == 2.0
    assert fake.closed


def test_online_collection_batches_five_envs_but_updates_once_per_transition(
    tmp_path: Path,
    koopman_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol: dict[str, object] = {
        "protocol_name": "synthetic_dmc_native_v1",
        "task": "cartpole_swingup",
        "obs_dim": 5,
        "action_dim": 1,
        "step_limit": 2,
    }
    dataset = _offline_dataset_for_mixing()
    vector_calls: list[tuple[int, int, int]] = []
    fake_vector: _FakeVectorEnvironment | None = None

    def make_vector(
        task: str, num_envs: int, seed: int, *, workers: int, **_kwargs
    ) -> _FakeVectorEnvironment:
        nonlocal fake_vector
        assert task == "cartpole_swingup"
        vector_calls.append((num_envs, workers, seed))
        fake_vector = _FakeVectorEnvironment(
            protocol, num_envs=num_envs, seed=seed
        )
        return fake_vector

    _FakeLearner.instances.clear()
    monkeypatch.setattr(train_module.OfflineDataset, "load", lambda _path: dataset)
    monkeypatch.setattr(
        train_module,
        "make_dmc_adapter",
        lambda task, seed: _FakeProtocolEnvironment(protocol),
    )
    monkeypatch.setattr(train_module, "make_dmc_vector_env", make_vector)
    monkeypatch.setattr(train_module, "O2OLearner", _FakeLearner)
    monkeypatch.setattr(
        train_module,
        "evaluate",
        lambda *_args, **_kwargs: {
            "return_mean": 100.0,
            "return_std_population": 0.0,
            "return_min": 100.0,
            "return_max": 100.0,
            "episode_length_mean": 2.0,
            "returns": [100.0] * 10,
        },
    )
    config = O2OConfig(
        method="Cal-RLPD-AC-KMPC",
        seed=7,
        device="cpu",
        batch_size=2,
        hidden_dim=8,
        critic_hidden_layers=1,
        critic_ensemble_size=2,
        target_critic_subset=2,
        offline_updates=1,
        cql_actions=2,
        online_steps=10,
        online_utd=20,
        online_warmup_steps=5,
        replay_capacity=32,
        num_envs=5,
        env_workers=5,
        kmpc_horizon=2,
        kmpc_solver_iterations=1,
        controller_hidden_dim=4,
        mpve_total_horizon=2,
        eval_interval_online_steps=10,
        eval_episodes=10,
        checkpoint_interval_updates=1,
        log_interval_updates=1,
    )

    train_module.run(config, Path("unused.npz"), koopman_path, tmp_path / "run")

    assert vector_calls == [(5, 5, 100_007)]
    assert fake_vector is not None and fake_vector.closed
    learner = _FakeLearner.instances[0]
    assert learner.act_batch_shapes == [(5, 5), (5, 5)]
    offline_calls = [call for call in learner.update_calls if call[1] == "offline"]
    online_calls = [call for call in learner.update_calls if call[1] == "online"]
    assert offline_calls == [(2, "offline", 1)]
    assert online_calls == [(40, "online", 20)] * 10

    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    episodes = [row for row in rows if row["phase"] == "online_episode"]
    assert [row["online_step"] for row in episodes] == [6, 7, 8, 9, 10]
    assert [row["episode"] for row in episodes] == [0, 1, 2, 3, 4]
    assert [row["episode_return"] for row in episodes] == [2.0, 4.0, 6.0, 8.0, 10.0]
    assert all(row["episode_length"] == 2 for row in episodes)
    assert len({row["online_step"] for row in episodes}) == 5
    checkpoint = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert checkpoint["online_step"] == 10
    assert checkpoint["online_episode"] == 5
    assert checkpoint["phase"] == "online"

    # A completed/resumed structured run must only validate its pre-online
    # fork snapshot.  It must never overwrite that file with the current
    # post-online learner state.
    offline_path = tmp_path / "run" / "offline.pt"
    offline_sha = file_sha256(offline_path)
    offline_checkpoint = load_checkpoint(offline_path)
    assert offline_checkpoint["phase"] == "offline"
    assert offline_checkpoint["online_step"] == 0
    train_module.run(config, Path("unused.npz"), koopman_path, tmp_path / "run")
    assert file_sha256(offline_path) == offline_sha
    assert vector_calls == [(5, 5, 100_007)]
