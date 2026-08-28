from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from antmaze_ac.koopman.model import DeepKoopman
from experiments.dmc.actors import ActorConfig, build_actor
from experiments.dmc.ppo.train_dmc_ppo import (
    _TEST_ONLY_AUTHORIZATION,
    _validate_koopman_authorization_lineage,
    OnPolicyEpisodeCollector,
    ObservationRunningMeanStd,
    AcmeReturnNormalizer,
    PPOConfig,
    ValueNetwork,
    collect_mpve_prediction,
    clip_ppo_gradients,
    compute_gae,
    compute_mpve_td_k_targets,
    crossed_diagnostic_milestones,
    main,
    parse_args,
    ppo_config_from_experiment,
    mpve_value_loss,
    prepare_acme_ppo_learner_values,
    tanh_normal_log_prob,
    tanh_normal_sample,
    train,
)
from experiments.dmc.config import load_experiment_config, resolve_execution_spec
from experiments.dmc.collect.build_dmc_datasets import BuildConfig, build
from experiments.dmc.ppo.vector_env import ProcessDMCVectorEnv, SyncDMCVectorEnv
from experiments.dmc.protocol import protocol_fingerprint
from experiments.dmc.source_identity import source_identity
from experiments.dmc.reward_model import (
    TransitionRewardModel,
    transition_reward_input_contract,
)
from experiments.dmc.reward_oracle import OFFICIAL_OBSERVATION_ORACLE


class FakeDMCEnv:
    """Tiny deterministic DMC-shaped env; every second step is a timeout."""

    obs_dim = 5
    action_dim = 1
    action_low = np.array([-1.0], dtype=np.float32)
    action_high = np.array([1.0], dtype=np.float32)

    def __init__(
        self,
        task_name: str,
        seed: int,
        control_timestep: float | None = None,
        time_limit: float | None = None,
    ) -> None:
        assert task_name == "cartpole_swingup"
        self.task_name = task_name
        self.seed = int(seed)
        self.control_dt = 0.01 if control_timestep is None else float(control_timestep)
        self.physics_dt = 0.01
        self.n_substeps = round(self.control_dt / self.physics_dt)
        self.step_limit = 2
        self._step = 0
        self.closed = False

    def _observation(self, action: float = 0.0) -> np.ndarray:
        return np.asarray(
            [self.seed % 97, self._step, action, self._step**2, 1.0],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
        self._step = 0
        return self._observation()

    def step(self, action: np.ndarray):
        applied = np.asarray(action, dtype=np.float32).reshape(1)
        assert -1.0 <= float(applied[0]) <= 1.0
        self._step += 1
        done = self._step == self.step_limit
        observation = self._observation(float(applied[0]))
        reward = 1.0 - 0.1 * float(applied[0]) ** 2
        return observation, reward, done, {
            "discount": 1.0,
            "terminated": False,
            "truncated": done,
            "applied_action": applied.copy(),
        }

    def metadata(self) -> dict:
        return {
            "protocol_name": "dmc_native_v1",
            "protocol_schema_version": 1,
            "task": self.task_name,
            "domain": "cartpole",
            "dmc_task": "cartpole:swingup",
            "dm_control_version": "test",
            "mujoco_version": "test",
            "seed": self.seed,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "control_dt": self.control_dt,
            "physics_dt": self.physics_dt,
            "n_substeps": self.n_substeps,
            "time_limit": self.step_limit * self.control_dt,
            "step_limit": self.step_limit,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "obs_layout": [["position", 3], ["velocity", 2]],
        }

    def protocol_metadata(self) -> dict:
        metadata = self.metadata()
        metadata.pop("seed")
        return metadata

    def close(self) -> None:
        self.closed = True


def fake_env_factory(
    task_name: str,
    seed: int,
    control_timestep: float | None = None,
    time_limit: float | None = None,
) -> FakeDMCEnv:
    return FakeDMCEnv(task_name, seed, control_timestep, time_limit)


def _cli_base(tmp_path: Path) -> list[str]:
    return [
        "--config",
        "experiments/dmc/configs/cartpole_swingup.yaml",
        "--profile",
        "development",
        "--train-seed-index",
        "0",
        "--preflight-file",
        str(tmp_path / "preflight.json"),
        "--actor",
        "PPO",
        "--output-dir",
        str(tmp_path / "run"),
    ]


def _write_preflight(
    path: Path,
    *,
    profile: str = "development",
    mutate=None,
) -> Path:
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    environment_protocol = FakeDMCEnv(
        "cartpole_swingup", 0
    ).protocol_metadata()
    payload = {
        "kind": "dmc_training_free_preflight",
        "ready_for_user_review": True,
        "training_approved": False,
        "task": experiment.task,
        "profile": profile,
        "config_fingerprint": experiment.fingerprint,
        "resolved_execution_spec": resolve_execution_spec(experiment, profile),
        "environment_protocol": environment_protocol,
        "protocol_fingerprint": protocol_fingerprint(environment_protocol),
        "source_identity": source_identity(),
    }
    if mutate is not None:
        mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_is_yaml_only_and_fails_closed_without_approval(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(_cli_base(tmp_path))
    parsed = parse_args([*_cli_base(tmp_path), "--dry-run"])
    assert parsed.dry_run is True
    with pytest.raises(SystemExit):
        parse_args(
            [
                *_cli_base(tmp_path),
                "--dry-run",
                "--total-timesteps",
                "4",
            ]
        )


def test_cli_resolves_ppo_and_actor_only_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_train(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"kind": "captured"}

    monkeypatch.setattr(
        "experiments.dmc.ppo.train_dmc_ppo.train", fake_train
    )
    main([*_cli_base(tmp_path), "--dry-run"])
    capsys.readouterr()
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    positional = captured["args"]
    keyword = captured["kwargs"]
    assert positional[0] == experiment.task
    assert positional[3] == ppo_config_from_experiment(
        experiment, "development", train_seed_index=0
    )
    assert keyword["actor_config"] == experiment.actor_config
    assert keyword["experiment_config"].fingerprint == experiment.fingerprint
    assert keyword["profile"] == "development"
    assert keyword["train_seed_index"] == 0
    assert keyword["dry_run"] is True
    assert keyword["collect_dir"] == Path(
        "runs/dmc/data/cartpole_swingup/development/seed_20260812"
    )

    captured.clear()
    main([*_cli_base(tmp_path), "--dry-run", "--no-collect"])
    capsys.readouterr()
    assert captured["kwargs"]["collect_dir"] is None

    with pytest.raises(SystemExit):
        parse_args(
            [
                *_cli_base(tmp_path),
                "--dry-run",
                "--no-collect",
                "--collect-dir",
                str(tmp_path / "collection"),
            ]
        )


def test_public_dry_run_never_resets_env_or_constructs_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = ppo_config_from_experiment(
        experiment, "development", train_seed_index=0
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run attempted training work")

    monkeypatch.setattr(FakeDMCEnv, "reset", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    preflight = _write_preflight(tmp_path / "preflight.json")
    result = train(
        experiment.task,
        "PPO",
        tmp_path / "manifest",
        config,
        actor_config=experiment.actor_config,
        device_name="cpu",
        resume=False,
        env_factory=fake_env_factory,
        experiment_config=experiment,
        profile="development",
        train_seed_index=0,
        collection_seed_index=0,
        preflight_file=preflight,
        dry_run=True,
    )
    assert result["kind"] == "dmc_ppo_training_free_run_manifest"
    assert result["environment_steps"] == 0
    assert result["optimization_steps"] == 0
    assert not (tmp_path / "manifest" / "latest.pt").exists()


def test_public_dry_run_rejects_stale_preflight_before_training_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = ppo_config_from_experiment(
        experiment, "development", train_seed_index=0
    )
    stale = _write_preflight(
        tmp_path / "stale.json",
        mutate=lambda payload: payload.__setitem__(
            "config_fingerprint", "sha256:" + "0" * 64
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("stale dry-run attempted training work")

    monkeypatch.setattr(FakeDMCEnv, "reset", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    with pytest.raises(ValueError, match="config_fingerprint"):
        train(
            experiment.task,
            "PPO",
            tmp_path / "stale_manifest",
            config,
            actor_config=experiment.actor_config,
            device_name="cpu",
            resume=False,
            env_factory=fake_env_factory,
            experiment_config=experiment,
            profile="development",
            train_seed_index=0,
            collection_seed_index=0,
            preflight_file=stale,
            dry_run=True,
        )
    assert not (tmp_path / "stale_manifest" / "run_manifest.json").exists()


def test_formal_collection_directory_is_profile_isolated(tmp_path: Path) -> None:
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = ppo_config_from_experiment(
        experiment, "development", train_seed_index=0
    )
    with pytest.raises(ValueError, match="isolate task/profile/seed"):
        train(
            experiment.task,
            "PPO",
            tmp_path / "manifest",
            config,
            actor_config=experiment.actor_config,
            collect_dir=tmp_path / "seed_20260811",
            device_name="cpu",
            resume=False,
            env_factory=fake_env_factory,
            experiment_config=experiment,
            profile="development",
            train_seed_index=0,
            collection_seed_index=0,
            preflight_file=tmp_path / "unused.json",
            dry_run=True,
        )


def test_shared_approval_is_checked_with_live_protocol_before_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = ppo_config_from_experiment(
        experiment, "development", train_seed_index=0
    )
    seen: dict[str, object] = {}

    def deny_approval(
        checked_config,
        profile,
        approval_file,
        preflight_file,
        *,
        runtime_protocol_fingerprint,
    ):
        seen.update(
            config=checked_config,
            profile=profile,
            approval_file=approval_file,
            preflight_file=preflight_file,
            runtime_protocol_fingerprint=runtime_protocol_fingerprint,
        )
        raise RuntimeError("approval denied for test")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("optimizer/env step occurred before approval")

    monkeypatch.setattr(
        "experiments.dmc.ppo.train_dmc_ppo.validate_training_approval",
        deny_approval,
    )
    monkeypatch.setattr(FakeDMCEnv, "reset", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    with pytest.raises(RuntimeError, match="approval denied"):
        train(
            experiment.task,
            "PPO",
            tmp_path / "denied",
            config,
            actor_config=experiment.actor_config,
            device_name="cpu",
            resume=False,
            env_factory=fake_env_factory,
            experiment_config=experiment,
            profile="development",
            train_seed_index=0,
            collection_seed_index=0,
            approval_file=tmp_path / "approval.json",
            preflight_file=tmp_path / "preflight.json",
        )
    assert seen["config"].fingerprint == experiment.fingerprint
    assert seen["profile"] == "development"
    assert len(seen["runtime_protocol_fingerprint"]) == 64


def test_vector_env_keeps_transition_observation_across_autoreset() -> None:
    env = SyncDMCVectorEnv(
        "cartpole_swingup", 2, 11, env_factory=fake_env_factory
    )
    initial = env.reset()
    first = env.step(np.asarray([[3.0], [-4.0]], dtype=np.float32))
    np.testing.assert_array_equal(first.applied_action[:, 0], [1.0, -1.0])
    assert not first.reset_boundary.any()
    np.testing.assert_allclose(first.observation, first.transition_observation)

    second = env.step(np.zeros((2, 1), dtype=np.float32))
    assert second.reset_boundary.all()
    assert second.truncated.all()
    assert not second.terminated.any()
    np.testing.assert_array_equal(second.discount, np.ones(2, dtype=np.float32))
    # Final transition observations are step 2; policy observations have
    # already reset to step 0 under new deterministic seeds.
    np.testing.assert_array_equal(second.transition_observation[:, 1], [2.0, 2.0])
    np.testing.assert_array_equal(second.observation[:, 1], [0.0, 0.0])
    assert not np.array_equal(second.observation, second.transition_observation)
    np.testing.assert_array_equal(first.reset_seed, [11, 12])
    np.testing.assert_array_equal(second.reset_seed, [11, 12])
    assert env.protocol["control_dt"] == pytest.approx(0.01)
    assert env.protocol["step_limit"] == 2
    assert "control_timestep" not in env.protocol
    assert "action_repeat" not in env.protocol
    assert env.protocol["n_substeps"] == 1
    env.close()


def test_vector_env_rejects_invalid_discount_and_inconsistent_done_flags() -> None:
    with pytest.raises(ValueError, match="Invalid environment discount"):
        SyncDMCVectorEnv._transition_flags(False, {"discount": 1.01})
    with pytest.raises(ValueError, match="done must equal"):
        SyncDMCVectorEnv._transition_flags(
            False,
            {"discount": 0.0, "terminated": True, "truncated": False},
        )
    with pytest.raises(ValueError, match="both terminated and truncated"):
        SyncDMCVectorEnv._transition_flags(
            True,
            {"discount": 0.0, "terminated": True, "truncated": True},
        )


def test_process_vector_env_matches_sync_real_dmc() -> None:
    pytest.importorskip("dm_control")
    sync = SyncDMCVectorEnv("cartpole_swingup", 4, 123)
    parallel = ProcessDMCVectorEnv(
        "cartpole_swingup", 4, 123, workers=2
    )
    try:
        np.testing.assert_array_equal(sync.reset(), parallel.reset())
        rng = np.random.default_rng(456)
        for _ in range(4):
            actions = rng.uniform(-1.0, 1.0, size=(4, 1)).astype(np.float32)
            expected = sync.step(actions)
            actual = parallel.step(actions)
            for name in (
                "observation",
                "transition_observation",
                "reward",
                "discount",
                "terminated",
                "truncated",
                "reset_boundary",
                "reset_seed",
                "applied_action",
            ):
                np.testing.assert_array_equal(
                    getattr(actual, name), getattr(expected, name)
                )
    finally:
        parallel.close()
        sync.close()


def test_gae_bootstraps_timeout_but_stops_trace_at_reset() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[2.0], [3.0]])
    bootstrap = torch.tensor([[10.0], [4.0]])
    discounts = torch.ones_like(rewards)
    boundaries = torch.tensor([[True], [False]])

    advantages, returns = compute_gae(
        rewards,
        values,
        bootstrap,
        discounts,
        boundaries,
        gamma=0.9,
        gae_lambda=0.8,
    )
    # t=0 timeout delta includes 0.9*V(final_obs), but not advantage from the
    # autoreset episode at t=1.
    torch.testing.assert_close(advantages[0], torch.tensor([8.0]))
    torch.testing.assert_close(returns[0], torch.tensor([10.0]))
    torch.testing.assert_close(advantages[1], torch.tensor([2.6]))

    terminal_advantage, _ = compute_gae(
        rewards[:1],
        values[:1],
        bootstrap[:1],
        torch.zeros_like(discounts[:1]),
        torch.ones_like(boundaries[:1]),
        gamma=0.9,
        gae_lambda=0.8,
    )
    torch.testing.assert_close(terminal_advantage[0], torch.tensor([-1.0]))


def test_acme_observation_and_return_normalizers_are_exact_and_resumable() -> None:
    observation = ObservationRunningMeanStd(2)
    observation.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    restored_observation = ObservationRunningMeanStd(2)
    restored_observation.load_state_dict(observation.state_dict())
    restored_observation.update(torch.tensor([[5.0, 6.0]]))
    torch.testing.assert_close(
        restored_observation.mean,
        torch.tensor([3.0, 4.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        restored_observation.std,
        torch.full((2,), math.sqrt(8.0 / 3.0), dtype=torch.float64),
    )
    assert restored_observation.count == 3

    returns = AcmeReturnNormalizer(0.5)
    returns.begin_update()
    returns.update_value(torch.tensor([1.0, 3.0]))
    returns.update_advantage(torch.tensor([-2.0, 4.0]))
    assert returns.value_mean == pytest.approx(2.0)
    assert returns.value_std == pytest.approx(1.0)
    assert returns.advantage_scale == pytest.approx(3.0)
    scaled = returns.normalize_advantage(torch.tensor([-2.0, 4.0]))
    torch.testing.assert_close(scaled, torch.tensor([-2.0 / 3.0, 4.0 / 3.0]))
    # Acme scales by the EMA mean absolute advantage; it does not z-score or
    # subtract the minibatch mean.
    assert float(scaled.mean()) != pytest.approx(0.0)
    restored_returns = AcmeReturnNormalizer(0.5)
    restored_returns.load_state_dict(returns.state_dict())
    assert restored_returns.state_dict() == returns.state_dict()


def test_acme_learner_updates_obs_stats_before_recomputing_behavior_values() -> None:
    class ScalarCritic(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.squeeze(-1)

    critic = ScalarCritic()
    observation = ObservationRunningMeanStd(1)
    states = torch.tensor([[[0.0]], [[2.0]]])
    transition_states = torch.tensor([[[2.0]], [[4.0]]])

    learner_states, learner_transitions, values, bootstraps = (
        prepare_acme_ppo_learner_values(
            critic,
            observation,
            states,
            transition_states,
            normalize_observation=True,
        )
    )
    expected_std = math.sqrt(8.0 / 3.0)
    torch.testing.assert_close(
        learner_states.squeeze(-1),
        torch.tensor([[-2.0 / expected_std], [0.0]]),
    )
    torch.testing.assert_close(
        learner_transitions.squeeze(-1),
        torch.tensor([[0.0], [2.0 / expected_std]]),
    )
    torch.testing.assert_close(values, learner_states.squeeze(-1))
    torch.testing.assert_close(bootstraps, learner_transitions.squeeze(-1))
    assert observation.count == 3
    # Rollout-time values under the empty/identity normalizer would have been
    # [0, 2]; these prove the learner did a fresh forward after updating stats.
    assert not torch.equal(values, states.squeeze(-1))


def test_acme_state_dependent_tanh_policy_and_three_layer_critic() -> None:
    actor_config = ActorConfig(ppo_hidden_dim=16)
    actor = build_actor(
        "PPO", "cartpole_swingup", torch.device("cpu"), config=actor_config
    )
    linear_layers = [
        module for module in actor.network if isinstance(module, torch.nn.Linear)
    ]
    assert len(linear_layers) == 3
    inputs = torch.stack((torch.zeros(5), torch.ones(5)))
    location, scale = actor.distribution_parameters(inputs)
    assert location.shape == scale.shape == (2, 1)
    assert bool((scale > 1e-3).all())
    assert not torch.equal(scale[0], scale[1])
    action = tanh_normal_sample(location, scale)
    assert bool((action.abs() <= 1.0).all())
    assert torch.isfinite(tanh_normal_log_prob(location, scale, action)).all()
    torch.testing.assert_close(actor(inputs), torch.tanh(location))

    critic = ValueNetwork(5, hidden_dim=16)
    critic_linears = [
        module
        for module in critic.network
        if isinstance(module, torch.nn.Linear)
    ]
    assert len(critic_linears) == 4  # three-layer torso plus scalar head
    for layer in critic_linears:
        stddev = 1.0 / math.sqrt(layer.in_features)
        assert float(layer.weight.abs().max()) <= 2.0 * stddev
        assert torch.equal(layer.bias, torch.zeros_like(layer.bias))


def test_diagnostic_milestones_cover_the_50k_acme_curve_plan() -> None:
    milestones: list[int] = []
    batch_size = 256 * 8
    previous = 0
    for update in range(1, 489):
        current = update * batch_size
        milestones.extend(
            crossed_diagnostic_milestones(previous, current, 50_000)
        )
        previous = current
    assert previous == 999_424
    assert milestones == list(range(50_000, 1_000_000, 50_000))
    assert crossed_diagnostic_milestones(0, 2048, None) == []


def test_mpve_critic_gradient_cannot_rescale_the_paired_actor_gradient() -> None:
    def clipped_actor_gradient(extra_critic_coefficient: float) -> torch.Tensor:
        policy = torch.nn.Parameter(torch.tensor([2.0]))
        critic = torch.nn.Parameter(torch.tensor([3.0]))
        standard_loss = policy.square().sum() + critic.square().sum()
        extra_critic_loss = extra_critic_coefficient * critic.square().sum()
        (standard_loss + extra_critic_loss).backward()
        clip_ppo_gradients(
            [policy],
            [critic],
            max_grad_norm=0.5,
            separate_policy_and_critic=True,
        )
        return policy.grad.detach().clone()

    base = clipped_actor_gradient(0.0)
    mpve = clipped_actor_gradient(10_000.0)
    torch.testing.assert_close(mpve, base, rtol=0.0, atol=0.0)


def test_mpve_td_k_formula_and_detached_critic_only_gradient() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    terminal = torch.tensor([4.0], requires_grad=True)
    targets = compute_mpve_td_k_targets(rewards, terminal, gamma=0.5)
    torch.testing.assert_close(targets, torch.tensor([[3.25, 4.5, 5.0]]))
    assert not targets.requires_grad

    koopman = DeepKoopman(5, 1, lift_dim=2, hidden_dims=(4,)).freeze_dynamics()
    config = ActorConfig(
        hidden_dim=4,
        kmpc_horizon=3,
        kmpc_solver_iterations=2,
    )
    actor = build_actor(
        "AC-MPC-MPVE",
        "cartpole_swingup",
        torch.device("cpu"),
        koopman=koopman,
        config=config,
    )
    critic = ValueNetwork(koopman.state_dim, hidden_dim=8)
    reward_model = TransitionRewardModel(5, 1, hidden_dims=(4,))
    reward_model.requires_grad_(False).eval()
    lifted = koopman.lift(torch.zeros(2, 5))
    prediction = collect_mpve_prediction(
        actor,
        critic,
        koopman,
        reward_model,
        lifted,
        horizon=3,
        gamma=0.9,
    )
    assert prediction.action_sequence.shape == (2, 3, 1)
    assert prediction.value_observations.shape == (2, 3, koopman.state_dim)
    assert prediction.td_k_targets.shape == (2, 3)
    assert all(
        not value.requires_grad
        for value in (
            prediction.action_sequence,
            prediction.value_observations,
            prediction.predicted_rewards,
            prediction.terminal_value,
            prediction.td_k_targets,
        )
    )

    loss = mpve_value_loss(
        critic,
        prediction.value_observations,
        prediction.td_k_targets,
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in critic.parameters())
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert all(parameter.grad is None for parameter in reward_model.parameters())


def test_episode_collector_writes_applied_actions_and_offline_fields(
    tmp_path: Path,
) -> None:
    env = SyncDMCVectorEnv(
        "cartpole_swingup", 1, 3, env_factory=fake_env_factory
    )
    collection_dir = tmp_path / "seed_test"
    collector = OnPolicyEpisodeCollector(
        collection_dir,
        1,
        5,
        1,
        flush_transitions=100,
        metadata={
            "task": "cartpole_swingup",
            "protocol": env.protocol,
            "training_seed": 3,
            "actor_type": "PPO",
            "training_approved": True,
            "config_fingerprint": "sha256:" + "1" * 64,
            "approval_profile": "development",
            "approval_file_sha256": "2" * 64,
            "preflight_report_sha256": "3" * 64,
            "authorization_kind": "dmc_training_approval_v1",
            "train_seed_index": 2,
        },
        seed_index=2,
        seed_dir="seed_test",
        max_transitions=6,
        total_updates=3,
    )
    observation = env.reset()
    updates = (1, 1, 1, 2, 2, 3)
    requested = (9.0, -9.0, 0.25, -0.25, 0.5, -0.5)
    for offset, (update, action) in enumerate(zip(updates, requested)):
        transition = env.step(np.asarray([[action]], dtype=np.float32))
        collector.record(
            observation,
            np.asarray([[action]], dtype=np.float32),
            transition,
            update=update,
            global_step_start=10 + offset,
        )
        observation = transition.observation
    path = collector.flush(force=True)
    assert path is not None

    with np.load(path, allow_pickle=False) as archive:
        assert set(
            (
                "state",
                "requested_action",
                "action",
                "next_state",
                "reward",
                "discount",
                "terminated",
                "truncated",
                "done",
                "collector_truncated",
                "episode_id",
                "step_index",
                "update",
                "global_step",
                "reset_seed",
                "collection_schema_version",
                "protocol_json",
                "environment_protocol_json",
                "protocol_fingerprint",
                "task",
                "seed_index",
                "seed_dir",
                "rng_state_after_json",
            )
        ).issubset(archive.files)
        np.testing.assert_array_equal(archive["requested_action"][:, 0], requested)
        np.testing.assert_array_equal(
            archive["action"][:, 0], [1.0, -1.0, 0.25, -0.25, 0.5, -0.5]
        )
        for episode_id in np.unique(archive["episode_id"]):
            indices = np.flatnonzero(archive["episode_id"] == episode_id)
            np.testing.assert_allclose(
                archive["next_state"][indices[:-1]],
                archive["state"][indices[1:]],
            )
        np.testing.assert_array_equal(archive["step_index"], [0, 1] * 3)
        np.testing.assert_array_equal(archive["global_step"], np.arange(10, 16))
        np.testing.assert_array_equal(archive["update"], updates)
        np.testing.assert_array_equal(archive["reset_seed"], [3, 3, 4, 4, 5, 5])
        np.testing.assert_array_equal(
            archive["episode_id"],
            [2_000_000] * 2 + [2_000_001] * 2 + [2_000_002] * 2,
        )
        np.testing.assert_array_equal(archive["discount"], [1.0] * 6)
        np.testing.assert_array_equal(archive["terminated"], [False] * 6)
        np.testing.assert_array_equal(archive["truncated"], [False, True] * 3)
        np.testing.assert_array_equal(archive["done"], [False, True] * 3)
        np.testing.assert_array_equal(
            archive["collection_stage"],
            ["early"] * 2 + ["mid"] * 2 + ["late"] * 2,
        )
        assert not archive["collector_truncated"].any()
        environment_protocol_json = archive["environment_protocol_json"].item()
        assert json.loads(environment_protocol_json) == env.protocol
        assert archive["protocol_fingerprint"].item() == hashlib.sha256(
            environment_protocol_json.encode()
        ).hexdigest()
        collection_protocol = json.loads(archive["protocol_json"].item())
        assert collection_protocol["collector_max_episode_steps"] == 2
        assert collection_protocol["collector_truncates_episodes"] is False
        assert json.loads(archive["rng_state_after_json"].item())["collector"][
            "next_global_step"
        ] == 16
        assert int(archive["collection_schema_version"]) == 4
        assert int(archive["seed_index"]) == 2
        assert archive["seed_dir"].item() == "seed_test"
        assert int(archive["collection_max_transitions"]) == 6
        assert (
            archive["collection_selection_strategy"].item()
            == "stage_quota_first_complete_episode_v2"
        )
        assert archive["actor_type"].item() == "PPO"
        assert archive["training_approved"].item() is True
        assert archive["config_fingerprint"].item() == "sha256:" + "1" * 64
    budget = collector.budget_report()
    assert budget["effective_durable_upper_bound"] == 6
    assert [stage["selected_episodes"] for stage in budget["stages"]] == [1, 1, 1]
    dataset = tmp_path / "primary_dataset.npz"
    build(
        BuildConfig(
            task_name="cartpole_swingup",
            collect_root=tmp_path,
            output=dataset,
            seed_dirs=("seed_test",),
            validation_every=3,
            test_offset=2,
            source="ppo_training_stages",
        )
    )
    with np.load(dataset, allow_pickle=False) as archive:
        assert int(archive["dataset_schema_version"]) == 4
        assert int(archive["collection_schema_version"]) == 4
        assert archive["data_source"].item() == "ppo_training_stages"
        assert archive["actor_type"].item() == "PPO"
        assert archive["config_fingerprint"].item() == "sha256:" + "1" * 64
    v3_dir = tmp_path / "v3" / "seed_test"
    v3_dir.mkdir(parents=True)
    with np.load(path, allow_pickle=False) as archive:
        v3_payload = {name: archive[name] for name in archive.files}
    v3_payload["collection_schema_version"] = np.asarray(3, dtype=np.int64)
    np.savez_compressed(v3_dir / "coverage_000000.npz", **v3_payload)
    with pytest.raises(ValueError, match="schema 3 != 4"):
        build(
            BuildConfig(
                task_name="cartpole_swingup",
                collect_root=tmp_path / "v3",
                output=tmp_path / "must_not_build.npz",
                seed_dirs=("seed_test",),
                validation_every=3,
                test_offset=2,
                source="ppo_training_stages",
            )
        )
    changed_lineage = {
        **collector.metadata,
        "config_fingerprint": "sha256:" + "9" * 64,
    }
    with pytest.raises(ValueError, match="metadata mismatch"):
        OnPolicyEpisodeCollector(
            collection_dir,
            1,
            5,
            1,
            flush_transitions=100,
            metadata=changed_lineage,
            seed_index=2,
            seed_dir="seed_test",
            max_transitions=6,
            total_updates=3,
        )
    env.close()


def test_cartpole_collector_stage_quota_survives_synchronous_completion_bursts(
    tmp_path: Path,
) -> None:
    protocol = FakeDMCEnv("cartpole_swingup", 1).protocol_metadata()
    protocol.update({"step_limit": 1000, "time_limit": 10.0})
    metadata = {
        "task": "cartpole_swingup",
        "protocol": protocol,
        "training_seed": 17,
        "actor_type": "PPO",
        "training_approved": True,
        "config_fingerprint": "sha256:" + "1" * 64,
        "approval_profile": "benchmark",
        "approval_file_sha256": "2" * 64,
        "preflight_report_sha256": "3" * 64,
        "authorization_kind": "dmc_training_approval_v1",
        "train_seed_index": 0,
    }

    def make(name: str) -> OnPolicyEpisodeCollector:
        return OnPolicyEpisodeCollector(
            tmp_path / name,
            256,
            5,
            1,
            metadata=metadata,
            seed_index=0,
            seed_dir=name,
            max_transitions=300_000,
            total_updates=488,
        )

    def apply_burst(
        collector: OnPolicyEpisodeCollector, completion_update: int
    ) -> None:
        stage_index = collector._stage_index(completion_update)
        for _ in range(256):
            if collector._select_episode(stage_index, completion_update, 1000):
                collector._stage_episode_counts[stage_index] += 1
                collector._stage_transition_counts[stage_index] += 1000
            else:
                collector._stage_skipped_episodes[stage_index] += 1

    uninterrupted = make("uninterrupted")
    for completion_update in (125, 250, 375):
        apply_burst(uninterrupted, completion_update)
    full_report = uninterrupted.budget_report()
    assert [
        stage["selected_episodes"] for stage in full_report["stages"]
    ] == [100, 100, 100]
    assert [
        stage["selected_transitions"] for stage in full_report["stages"]
    ] == [100_000, 100_000, 100_000]
    assert sum(
        stage["selected_transitions"] for stage in full_report["stages"]
    ) == 300_000

    # The selection decision depends only on checkpointed durable stage counts.
    # Rehydrate those counts after two bursts and verify the resumed final burst
    # is bit-for-bit the same as the uninterrupted quota trajectory.
    before_resume = make("before_resume")
    for completion_update in (125, 250):
        apply_burst(before_resume, completion_update)
    resumed = make("resumed")
    resumed._stage_episode_counts = list(before_resume._stage_episode_counts)
    resumed._stage_transition_counts = list(
        before_resume._stage_transition_counts
    )
    resumed._stage_skipped_episodes = list(
        before_resume._stage_skipped_episodes
    )
    apply_burst(resumed, 375)
    assert resumed.budget_report()["stages"] == full_report["stages"]


def _smoke_config(
    total_timesteps: int,
    *,
    collect_max_transitions: int | None = None,
    max_wall_time_seconds: float | None = None,
    checkpoint_interval_updates: int = 1,
) -> PPOConfig:
    return PPOConfig(
        num_envs=2,
        rollout_steps=2,
        minibatch_size=4,
        update_epochs=1,
        total_timesteps=total_timesteps,
        learning_rate=1e-3,
        target_kl=None,
        checkpoint_interval_updates=checkpoint_interval_updates,
        critic_hidden_dim=8,
        seed=17,
        collect_flush_transitions=100,
        collect_max_transitions=collect_max_transitions,
        max_wall_time_seconds=max_wall_time_seconds,
        mpve_horizon=2,
    )


def test_ppo_single_update_checkpoint_collection_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    collection = tmp_path / "collection"
    actor_config = ActorConfig(ppo_hidden_dim=8)
    training_config = _smoke_config(
        12,
        collect_max_transitions=12,
        max_wall_time_seconds=1e-9,
    )
    with pytest.raises(PermissionError, match="experiment config"):
        train(
            "cartpole_swingup",
            "PPO",
            output,
            training_config,
            actor_config=actor_config,
            device_name="cpu",
            resume=False,
            env_factory=fake_env_factory,
        )
    first = train(
        "cartpole_swingup",
        "PPO",
        output,
        training_config,
        actor_config=actor_config,
        collect_dir=collection,
        device_name="cpu",
        resume=False,
        env_factory=fake_env_factory,
        _test_authorization=_TEST_ONLY_AUTHORIZATION,
    )
    assert first["update"] == 1
    assert first["global_step"] == 4
    checkpoint = torch.load(output / "latest.pt", weights_only=False)
    assert checkpoint["kind"] == "dmc_ppo_actor"
    assert checkpoint["format_version"] == 3
    assert (
        checkpoint["training_spec_version"]
        == "dmc_ppo_v4_raw_observation_critic"
    )
    assert checkpoint["task"] == "cartpole_swingup"
    assert checkpoint["actor_type"] == checkpoint["actor_name"] == "PPO"
    assert checkpoint["training_seed"] == 17
    assert checkpoint["actor_config"] == actor_config.to_dict()
    assert checkpoint["koopman_path"] is None
    assert checkpoint["koopman_sha256"] is None
    assert checkpoint["normalizer"] is None
    assert checkpoint["config_fingerprint"] == "test-only"
    assert checkpoint["authorization_kind"] == "private_test_only"
    assert len(checkpoint["protocol_fingerprint"]) == 64
    assert "actor_state" in checkpoint and "optimizer_state" in checkpoint
    assert checkpoint["log_std"] is None
    assert checkpoint["ppo_config"]["adam_epsilon"] == pytest.approx(1e-7)
    assert checkpoint["optimizer_state"]["param_groups"][0]["eps"] == pytest.approx(
        1e-7
    )
    assert checkpoint["observation_normalizer_state"]["count"] == 6
    assert checkpoint["return_normalizer_state"]["ema_counter"] == 1
    assert checkpoint["network_initialization_contract"]["alignment"] == (
        "acme_aligned_pytorch_synchronous_reference"
    )
    assert checkpoint["network_initialization_contract"]["seed_semantics"] == (
        "same_seed_method_specific_initializers_not_equal_parameters_v1"
    )
    assert checkpoint["network_initialization_contract"]["critic_input"] == (
        "normalized_raw_task_observation_v1"
    )
    assert checkpoint["network_initialization_contract"]["critic_seed"] == 17
    assert checkpoint["network_initialization_contract"][
        "critic_seed_independent_of_actor_construction"
    ] is True
    assert checkpoint["value_state"]["network.0.weight"].shape[1] == 5
    assert (
        checkpoint["network_initialization_contract"][
            "strict_parameter_pairing_group"
        ]
        is None
    )
    assert checkpoint["last_report"][
        "online_stochastic_last_100_episode_return_mean"
    ] == pytest.approx(first["online_stochastic_last_100_episode_return_mean"])
    assert checkpoint["collector_state"]["kind"] == "dmc_on_policy_collector_v4"
    assert checkpoint["collector_state"]["total_transitions"] == 4
    assert len(checkpoint["recent_episode_returns"]) == 2
    assert len(checkpoint["recent_episode_lengths"]) == 2
    best_checkpoint = torch.load(output / "best.pt", weights_only=False)
    assert best_checkpoint["collector_state"]["total_transitions"] == 4
    assert best_checkpoint["recent_episode_returns"] == checkpoint[
        "recent_episode_returns"
    ]
    assert checkpoint["collector_state"][
        "incomplete_episode_resume_policy"
    ] == "discard_after_env_reset"
    with np.load(sorted(collection.glob("coverage_*.npz"))[0]) as archive:
        assert np.max(np.abs(archive["action"])) <= 1.0
        assert len(archive["state"]) == 4

    resumed = train(
        "cartpole_swingup",
        "PPO",
        output,
        training_config,
        actor_config=actor_config,
        collect_dir=collection,
        device_name="cpu",
        resume=True,
        env_factory=fake_env_factory,
        _test_authorization=_TEST_ONLY_AUTHORIZATION,
    )
    assert resumed["update"] == 2
    assert resumed["global_step"] == 8
    assert resumed["collected_transitions"] == 8
    resumed_checkpoint = torch.load(output / "latest.pt", weights_only=False)
    assert len(resumed_checkpoint["recent_episode_returns"]) == 4
    assert len(resumed_checkpoint["recent_episode_lengths"]) == 4
    assert [
        stage["selected_episodes"]
        for stage in resumed["collection_budget"]["stages"]
    ] == [2, 2, 0]
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    assert not list(output.glob("*.tmp"))


def test_collection_checkpoint_and_metrics_resume_at_noninterval_update(
    tmp_path: Path,
) -> None:
    output = tmp_path / "interrupted_run"
    collection = tmp_path / "interrupted_collection"
    config = _smoke_config(
        12,
        collect_max_transitions=12,
        checkpoint_interval_updates=10,
    )
    with pytest.raises(RuntimeError, match="test-only interruption"):
        train(
            "cartpole_swingup",
            "PPO",
            output,
            config,
            actor_config=ActorConfig(ppo_hidden_dim=8),
            collect_dir=collection,
            device_name="cpu",
            resume=False,
            env_factory=fake_env_factory,
            _test_authorization=_TEST_ONLY_AUTHORIZATION,
            _test_interrupt_after_update=1,
        )
    interrupted = torch.load(output / "latest.pt", weights_only=False)
    assert interrupted["update"] == 1
    assert interrupted["collector_state"]["total_transitions"] == 4
    assert len(interrupted["recent_episode_returns"]) == 2

    # Simulate a valid-but-uncommitted metric tail.  Resume must treat latest.pt
    # as authoritative and atomically remove it before writing update 2.
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"update": 2, "uncommitted": True}) + "\n")
    with pytest.raises(RuntimeError, match="test-only interruption"):
        train(
            "cartpole_swingup",
            "PPO",
            output,
            config,
            actor_config=ActorConfig(ppo_hidden_dim=8),
            collect_dir=collection,
            device_name="cpu",
            resume=True,
            env_factory=fake_env_factory,
            _test_authorization=_TEST_ONLY_AUTHORIZATION,
            _test_interrupt_after_update=2,
        )
    second_checkpoint = torch.load(output / "latest.pt", weights_only=False)
    assert second_checkpoint["update"] == 2
    assert second_checkpoint["collector_state"]["total_transitions"] == 8
    assert len(second_checkpoint["recent_episode_returns"]) == 4
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"update": 3, "uncommitted": True}) + "\n")

    result = train(
        "cartpole_swingup",
        "PPO",
        output,
        config,
        actor_config=ActorConfig(ppo_hidden_dim=8),
        collect_dir=collection,
        device_name="cpu",
        resume=True,
        env_factory=fake_env_factory,
        _test_authorization=_TEST_ONLY_AUTHORIZATION,
    )
    assert result["update"] == 3
    final_checkpoint = torch.load(output / "latest.pt", weights_only=False)
    assert final_checkpoint["collector_state"]["total_transitions"] == 12
    assert len(final_checkpoint["recent_episode_returns"]) == 6
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["update"] for row in metrics] == [1, 2, 3]
    assert all("uncommitted" not in row for row in metrics)


def _write_stable_koopman(path: Path) -> None:
    model = DeepKoopman(5, 1, lift_dim=2, hidden_dims=(4,))
    reward_model = TransitionRewardModel(5, 1, hidden_dims=(4,))
    with torch.no_grad():
        model.A.copy_(0.8 * torch.eye(model.lifted_dim))
    environment_protocol_json = json.dumps(
        FakeDMCEnv("cartpole_swingup", 0).protocol_metadata(),
        sort_keys=True,
        separators=(",", ":"),
    )
    torch.save(
        {
            "kind": "dmc_k_step_koopman",
            "architecture": model.architecture(),
            "model_state": model.state_dict(),
            "state_kind": "cartpole_swingup",
            "normalizer": {
                "center": torch.zeros(5),
                "scale": torch.ones(5),
            },
            "environment_protocol_json": environment_protocol_json,
            "protocol_fingerprint": hashlib.sha256(
                environment_protocol_json.encode()
            ).hexdigest(),
            "reward_model_architecture": reward_model.architecture(),
            "reward_model_input_contract": transition_reward_input_contract(),
            "reward_model_state": reward_model.state_dict(),
        },
        path,
    )


def test_structured_koopman_accepts_distinct_actor_and_model_lineages() -> None:
    actor_authorization = {
        "config_fingerprint": "sha256:" + "1" * 64,
        "approval_profile": "development",
        "approval_file_sha256": "2" * 64,
        "preflight_report_sha256": "3" * 64,
    }
    model_authorization = {
        "config_fingerprint": "sha256:" + "5" * 64,
        "approval_profile": "benchmark",
        "approval_file_sha256": "6" * 64,
        "preflight_report_sha256": "7" * 64,
    }
    checkpoint = {
        "state_kind": "cartpole_swingup",
        "training_approved": True,
        "authorization_kind": "dmc_training_approval_v1",
        **model_authorization,
        "dataset_sha256": "4" * 64,
    }
    lineage = _validate_koopman_authorization_lineage(
        checkpoint,
        task_name="cartpole_swingup",
        authorization_metadata=actor_authorization,
    )
    assert lineage["dataset_sha256"] == "4" * 64
    assert lineage["config_fingerprint"] == model_authorization[
        "config_fingerprint"
    ]
    assert lineage["approval_file_sha256"] != actor_authorization[
        "approval_file_sha256"
    ]

    wrong_config = {**checkpoint, "config_fingerprint": "not-a-fingerprint"}
    with pytest.raises(ValueError, match="authorization lineage mismatch"):
        _validate_koopman_authorization_lineage(
            wrong_config,
            task_name="cartpole_swingup",
            authorization_metadata=actor_authorization,
        )
    unapproved = {**checkpoint, "training_approved": False}
    with pytest.raises(ValueError, match="training_approved"):
        _validate_koopman_authorization_lineage(
            unapproved,
            task_name="cartpole_swingup",
            authorization_metadata=actor_authorization,
        )


@pytest.mark.parametrize(
    "actor_type", ["KLQR", "AB-PQ", "KMPC", "AC-MPC-MPVE"]
)
def test_structured_actor_requires_koopman_and_runs_one_update(
    tmp_path: Path, actor_type: str
) -> None:
    with pytest.raises(ValueError, match="requires --koopman"):
        train(
            "cartpole_swingup",
            actor_type,
            tmp_path / "missing",
            _smoke_config(4),
            device_name="cpu",
            resume=False,
            env_factory=fake_env_factory,
        )

    koopman = tmp_path / "koopman.pt"
    _write_stable_koopman(koopman)
    actor_config = ActorConfig(
        hidden_dim=4,
        kmpc_horizon=2,
        kmpc_solver_iterations=2,
    )
    structured_output = tmp_path / actor_type.lower()
    training_config = _smoke_config(4)
    if actor_type == "AC-MPC-MPVE":
        training_config = replace(
            training_config,
            mpve_reward_source=OFFICIAL_OBSERVATION_ORACLE,
        )
    result = train(
        "cartpole_swingup",
        actor_type,
        structured_output,
        training_config,
        actor_config=actor_config,
        koopman_path=koopman,
        device_name="cpu",
        resume=False,
        env_factory=fake_env_factory,
        _test_authorization=_TEST_ONLY_AUTHORIZATION,
    )
    assert result["global_step"] == 4
    payload = torch.load(
        structured_output / "latest.pt", weights_only=False
    )
    assert payload["actor_type"] == actor_type
    assert payload["normalizer"] == {
        "center": [0.0] * 5,
        "scale": [1.0] * 5,
    }
    assert payload["koopman_sha256"] is not None
    assert payload["value_state"]["network.0.weight"].shape[1] == 5
    assert payload["normalization_contract"]["critic_input"] == (
        "normalized_raw_task_observation_v1"
    )
    assert payload["normalization_contract"]["controller_input"] == (
        "frozen_koopman_lifted_state_v1"
    )
    assert payload["network_initialization_contract"][
        "critic_seed_independent_of_actor_construction"
    ] is True
    if actor_type == "AC-MPC-MPVE":
        assert payload["value_expansion"]["actor_shared_with"] == "KMPC"
        assert payload["value_expansion"]["horizon"] == 2
        assert (
            payload["value_expansion"]["reward"]["source"]
            == OFFICIAL_OBSERVATION_ORACLE
        )
        assert payload["last_report"]["mpve_value_loss"] is not None
        assert payload["last_report"]["mpve_predicted_reward_mean"] is not None


def test_kmpc_and_mpve_keep_identical_actor_updates_under_the_same_seed(
    tmp_path: Path,
) -> None:
    koopman = tmp_path / "paired_koopman.pt"
    _write_stable_koopman(koopman)
    actor_config = ActorConfig(
        hidden_dim=4,
        kmpc_horizon=2,
        kmpc_solver_iterations=2,
    )
    outputs = {}
    for actor_type in ("KMPC", "AC-MPC-MPVE"):
        training_config = _smoke_config(4)
        if actor_type == "AC-MPC-MPVE":
            training_config = replace(
                training_config,
                mpve_reward_source=OFFICIAL_OBSERVATION_ORACLE,
            )
        train(
            "cartpole_swingup",
            actor_type,
            tmp_path / actor_type,
            training_config,
            actor_config=actor_config,
            koopman_path=koopman,
            device_name="cpu",
            resume=False,
            env_factory=fake_env_factory,
            _test_authorization=_TEST_ONLY_AUTHORIZATION,
        )
        outputs[actor_type] = torch.load(
            tmp_path / actor_type / "latest.pt", weights_only=False
        )

    kmpc = outputs["KMPC"]
    mpve = outputs["AC-MPC-MPVE"]
    for payload in (kmpc, mpve):
        assert payload["network_initialization_contract"][
            "structured_actor"
        ] == "koopman_mpc_cost_map_module_defaults_v1"
        assert payload["network_initialization_contract"][
            "strict_parameter_pairing_group"
        ] == "KMPC_AC-MPC-MPVE"
    assert kmpc["gradient_clip_contract"] == mpve["gradient_clip_contract"] == (
        "separate_policy_log_std_and_critic_global_norm_v1"
    )
    assert kmpc["actor_state"].keys() == mpve["actor_state"].keys()
    for name in kmpc["actor_state"]:
        torch.testing.assert_close(
            kmpc["actor_state"][name],
            mpve["actor_state"][name],
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        kmpc["log_std"], mpve["log_std"], rtol=0.0, atol=0.0
    )
    # The critic is the only optimized component the MPVE ablation may change.
    assert any(
        not torch.equal(kmpc["value_state"][name], mpve["value_state"][name])
        for name in kmpc["value_state"]
    )
