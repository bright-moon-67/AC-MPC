import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from antmaze_ac.koopman.losses import koopman_loss
from antmaze_ac.koopman.model import DeepKoopman
from experiments.dmc.collect.build_dmc_datasets import (
    DATASET_SCHEMA_VERSION,
    ON_POLICY_COLLECTION_SCHEMA_VERSION,
)
from experiments.dmc.approval import TrainingApprovalError, write_training_approval
from experiments.dmc.config import (
    load_experiment_config,
    resolve_execution_spec,
    resolve_koopman_config,
)
from experiments.dmc.koopman import train_dmc_koopman as trainer_module
from experiments.dmc.koopman.train_dmc_koopman import (
    EpochRandomSampler,
    KoopmanWindowDataset,
    _make_loader,
    _normalize_resume_history,
    _set_loader_epoch,
    _synchronize_history_file,
    build_windows,
    fit_normalizer,
    load_dataset,
    main,
    parse_args,
    rollout_prediction_metrics,
    rollout_prediction_metrics_streaming,
    transition_reward_metrics,
    train,
    train_config_from_experiment,
)
from experiments.dmc.protocol import (
    canonical_json,
    protocol_fingerprint,
    protocol_fingerprint_from_json,
)
from experiments.dmc.source_identity import source_identity
from experiments.dmc.reward_model import (
    TransitionRewardModel,
    reward_model_from_checkpoint,
    transition_reward_input_contract,
)
from experiments.dmc.tasks.registry import TASK_SPECS


def _episodes(
    *, num_episodes: int = 3, steps_per_episode: int = 7
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(4)
    states = []
    actions = []
    next_states = []
    rewards = []
    episode_ids = []
    step_indices = []
    for episode in range(num_episodes):
        state = rng.normal(size=5).astype(np.float32)
        for step in range(steps_per_episode):
            action = rng.uniform(-1, 1, size=1).astype(np.float32)
            next_state = state + np.array(
                [action[0], 0.1, -0.1, 0.05, -0.05], dtype=np.float32
            )
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(np.float32((action[0] + 1.0) / 2.0))
            episode_ids.append(episode)
            step_indices.append(step)
            state = next_state
    return {
        "state": np.asarray(states),
        "action": np.asarray(actions),
        "next_state": np.asarray(next_states),
        "reward": np.asarray(rewards, dtype=np.float32),
        "episode_id": np.asarray(episode_ids),
        "step_index": np.asarray(step_indices),
    }


def _environment_protocol(*, control_dt: float = 0.01) -> dict[str, object]:
    return {
        "protocol_name": "dmc_native_v1",
        "task": "cartpole_swingup",
        "obs_dim": 5,
        "action_dim": 1,
        "control_dt": control_dt,
        "step_limit": 1000,
    }


def _dataset_archive(
    *,
    environment_protocol: dict[str, object] | None = None,
    authorization: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    payload = _episodes(num_episodes=4, steps_per_episode=51)
    authorization = authorization or {
        "data_source": "ppo_training_stages",
        "actor_type": "PPO",
        "training_approved": True,
        "config_fingerprint": "sha256:" + "0" * 64,
        "approval_profile": "development",
        "approval_file_sha256": "1" * 64,
        "preflight_report_sha256": "2" * 64,
        "authorization_kind": "dmc_training_approval_v1",
    }
    environment_protocol_json = canonical_json(
        environment_protocol or _environment_protocol()
    )
    collection_protocol_json = canonical_json(
        {"task": "cartpole_swingup", "obs_dim": 5, "action_dim": 1}
    )
    count = len(payload["state"])
    episode_ids = payload["episode_id"].astype(np.int64, copy=False)
    episode_updates = np.asarray([20, 200, 400, 450], dtype=np.int64)
    boundary = payload["step_index"] == 50
    return {
        **payload,
        "requested_action": payload["action"].copy(),
        "reward": np.zeros(count, dtype=np.float32),
        "discount": np.ones(count, dtype=np.float32),
        "done": boundary.copy(),
        "terminated": np.zeros(count, dtype=np.bool_),
        "truncated": boundary.copy(),
        "collector_truncated": np.zeros(count, dtype=np.bool_),
        "update": episode_updates[episode_ids],
        "global_step": np.arange(count, dtype=np.int64),
        "reset_seed": 10_000 + episode_ids,
        "source_seed_index": np.zeros(len(payload["state"]), dtype=np.int64),
        "source_train_seed_indices": np.asarray([0], dtype=np.int64),
        "source_training_seeds": np.asarray([20260812], dtype=np.int64),
        **{name: np.asarray(value) for name, value in authorization.items()},
        "train_episode_ids": np.asarray([0, 1], dtype=np.int64),
        "validation_episode_ids": np.asarray([2], dtype=np.int64),
        "test_episode_ids": np.asarray([3], dtype=np.int64),
        "dataset_schema_version": np.asarray(
            DATASET_SCHEMA_VERSION, dtype=np.int64
        ),
        "collection_schema_version": np.asarray(
            ON_POLICY_COLLECTION_SCHEMA_VERSION, dtype=np.int64
        ),
        "protocol_json": np.asarray(collection_protocol_json),
        "environment_protocol_json": np.asarray(environment_protocol_json),
        "protocol_fingerprint": np.asarray(
            protocol_fingerprint_from_json(environment_protocol_json)
        ),
        "state_kind": np.asarray("cartpole_swingup"),
        "state_dim": np.asarray(5, dtype=np.int64),
        "action_dim": np.asarray(1, dtype=np.int64),
    }


def _write_dataset(
    tmp_path: Path,
    *,
    environment_protocol: dict[str, object] | None = None,
    authorization: dict[str, object] | None = None,
) -> Path:
    path = tmp_path / "dataset.npz"
    np.savez(
        path,
        **_dataset_archive(
            environment_protocol=environment_protocol,
            authorization=authorization,
        ),
    )
    return path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_preflight(
    tmp_path: Path,
    experiment,
    *,
    environment_protocol: dict[str, object] | None = None,
) -> Path:
    environment_protocol = environment_protocol or _environment_protocol()
    path = tmp_path / "preflight.json"
    _write_json(
        path,
        {
            "kind": "dmc_training_free_preflight",
            "ready_for_user_review": True,
            "training_approved": False,
            "task": experiment.task,
            "profile": "development",
            "config_fingerprint": experiment.fingerprint,
            "resolved_execution_spec": resolve_execution_spec(
                experiment, "development"
            ),
            "environment_protocol": environment_protocol,
            "protocol_fingerprint": protocol_fingerprint(environment_protocol),
            "source_identity": source_identity(),
        },
    )
    return path


def _forbid_optimizer(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run/fail-closed path constructed an optimizer")

    monkeypatch.setattr(trainer_module.torch.optim, "Adam", forbidden)


def _authorization_for(
    experiment,
    preflight: Path,
    *,
    approval: Path | None = None,
) -> dict[str, object]:
    return {
        "data_source": "ppo_training_stages",
        "actor_type": "PPO",
        "training_approved": True,
        "config_fingerprint": experiment.fingerprint,
        "approval_profile": "development",
        "approval_file_sha256": (
            hashlib.sha256(approval.read_bytes()).hexdigest()
            if approval is not None
            else "1" * 64
        ),
        "preflight_report_sha256": hashlib.sha256(
            preflight.read_bytes()
        ).hexdigest(),
        "authorization_kind": "dmc_training_approval_v1",
    }


def test_lazy_windows_match_materialized_windows():
    data = _episodes()
    center = np.zeros(5, dtype=np.float32)
    scale = np.ones(5, dtype=np.float32)
    expected_states, expected_actions = build_windows(
        data,
        np.array([0, 1, 2]),
        center,
        scale,
        k_step=3,
        obs_dim=5,
        action_dim=1,
    )
    dataset = KoopmanWindowDataset(
        data,
        np.array([0, 1, 2]),
        center,
        scale,
        k_step=3,
        obs_dim=5,
        action_dim=1,
    )
    actual = [dataset[index] for index in range(len(dataset))]
    actual_states = torch.stack([item[0] for item in actual]).numpy()
    actual_actions = torch.stack([item[1] for item in actual]).numpy()
    actual_rewards = torch.stack([item[2] for item in actual]).numpy()
    actual_reward_next_states = torch.stack([item[3] for item in actual]).numpy()
    assert np.array_equal(actual_states, expected_states)
    assert np.array_equal(actual_actions, expected_actions)
    assert np.array_equal(actual_rewards, data["reward"][dataset.starts])
    assert np.array_equal(
        actual_reward_next_states,
        data["next_state"][dataset.starts],
    )
    assert len(actual_rewards) == len(np.unique(dataset.starts))


def test_vectorized_window_batch_matches_scalar_items():
    data = _episodes()
    center = np.linspace(-0.2, 0.2, 5, dtype=np.float32)
    scale = np.linspace(0.5, 1.5, 5, dtype=np.float32)
    dataset = KoopmanWindowDataset(
        data,
        np.array([0, 1, 2]),
        center,
        scale,
        k_step=3,
        obs_dim=5,
        action_dim=1,
    )
    indices = np.asarray([7, 0, 4, 7], dtype=np.int64)
    vectorized = dataset.get_batch(indices)
    scalar = [dataset[int(index)] for index in indices]

    for component, expected_items in zip(vectorized, zip(*scalar), strict=True):
        assert torch.equal(component, torch.stack(list(expected_items)))


def test_streaming_rollout_metrics_match_materialized_metrics():
    data = _episodes()
    center = np.zeros(5, dtype=np.float32)
    scale = np.ones(5, dtype=np.float32)
    states, actions = build_windows(
        data,
        np.array([0, 1, 2]),
        center,
        scale,
        k_step=3,
        obs_dim=5,
        action_dim=1,
    )
    dataset = KoopmanWindowDataset(
        data,
        np.array([0, 1, 2]),
        center,
        scale,
        k_step=3,
        obs_dim=5,
        action_dim=1,
    )
    loader = _make_loader(dataset, batch_size=4, shuffle=False, seed=1)
    model = DeepKoopman(5, 1, lift_dim=3, hidden_dims=(8,))
    groups = (("all", (0, 5)),)
    expected = rollout_prediction_metrics(
        model,
        states,
        actions,
        center,
        scale,
        torch.device("cpu"),
        4,
        groups,
    )
    actual = rollout_prediction_metrics_streaming(
        model,
        loader,
        center,
        scale,
        torch.device("cpu"),
        groups,
    )
    assert actual["normalized_mse_all_steps"] == pytest.approx(
        expected["normalized_mse_all_steps"], rel=1e-6
    )
    assert actual["all_steps"]["all"]["rmse"] == pytest.approx(
        expected["all_steps"]["all"]["rmse"], rel=1e-6
    )


def test_transition_reward_model_shape_bounds_backward_and_strict_round_trip():
    torch.manual_seed(8)
    model = TransitionRewardModel(5, 1, hidden_dims=(16, 8), activation="silu")
    state = torch.randn(7, 5)
    action = torch.rand(7, 1) * 2.0 - 1.0
    next_state = torch.randn(7, 5)
    prediction = model(state, action, next_state)

    assert prediction.shape == (7,)
    assert torch.all(prediction >= 0.0)
    assert torch.all(prediction <= 1.0)
    prediction.mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    payload = {
        "reward_model_architecture": model.architecture(),
        "reward_model_input_contract": transition_reward_input_contract(),
        "reward_model_state": model.state_dict(),
    }
    restored = reward_model_from_checkpoint(payload)
    torch.testing.assert_close(restored(state, action, next_state), prediction)

    malformed = {**payload, "reward_model_architecture": dict(model.architecture())}
    malformed["reward_model_architecture"]["unexpected"] = 1
    with pytest.raises(ValueError, match="architecture fields"):
        reward_model_from_checkpoint(malformed)

    wrong_contract = {
        **payload,
        "reward_model_input_contract": dict(transition_reward_input_contract()),
    }
    wrong_contract["reward_model_input_contract"]["action"] = "requested_action"
    with pytest.raises(ValueError, match="input contract"):
        reward_model_from_checkpoint(wrong_contract)

    nonfinite = {**payload, "reward_model_state": dict(model.state_dict())}
    state_key = next(iter(nonfinite["reward_model_state"]))
    nonfinite["reward_model_state"][state_key] = nonfinite[
        "reward_model_state"
    ][state_key].clone()
    nonfinite["reward_model_state"][state_key].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        reward_model_from_checkpoint(nonfinite)


@pytest.mark.parametrize("state_dim", [True, 1.5, "5", np.int64(5)])
def test_transition_reward_model_rejects_type_coercible_dimensions(state_dim):
    with pytest.raises(ValueError, match="state_dim"):
        TransitionRewardModel(state_dim, 1, hidden_dims=(8,))


@pytest.mark.parametrize("hidden_dims", [(8, 1.5), (8, True), "8"])
def test_transition_reward_model_rejects_non_integer_hidden_dims(hidden_dims):
    with pytest.raises(ValueError, match="hidden_dims"):
        TransitionRewardModel(5, 1, hidden_dims=hidden_dims)


def test_transition_reward_metrics_use_applied_actions_and_all_transitions():
    data = _episodes(num_episodes=2, steps_per_episode=4)
    mask = np.ones(len(data["state"]), dtype=np.bool_)
    model = TransitionRewardModel(5, 1, hidden_dims=(8,))
    metrics = transition_reward_metrics(
        model,
        data,
        mask,
        np.zeros(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
        torch.device("cpu"),
        batch_size=3,
    )
    assert metrics["transitions"] == 8
    assert metrics["mse"] >= 0.0
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))
    assert metrics["mae"] >= 0.0


def test_joint_optimizer_updates_koopman_and_reward_model():
    torch.manual_seed(12)
    koopman = DeepKoopman(5, 1, lift_dim=3, hidden_dims=(8,))
    reward_model = TransitionRewardModel(5, 1, hidden_dims=(8,))
    parameters = [*koopman.parameters(), *reward_model.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    states = torch.randn(6, 4, 5)
    actions = torch.rand(6, 3, 1) * 2.0 - 1.0
    rewards = torch.rand(6)
    koopman_before = [parameter.detach().clone() for parameter in koopman.parameters()]
    reward_before = [
        parameter.detach().clone() for parameter in reward_model.parameters()
    ]

    dynamics_loss = koopman_loss(koopman, states, actions).total
    reward_loss = (
        reward_model(states[:, 0], actions[:, 0], states[:, 1]) - rewards
    ).square().mean()
    (dynamics_loss + reward_loss).backward()
    optimizer.step()

    assert any(
        not torch.equal(before, after)
        for before, after in zip(koopman_before, koopman.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            reward_before, reward_model.parameters(), strict=True
        )
    )


def test_lazy_window_sampling_is_seeded_and_not_prefix_biased():
    data = _episodes()
    kwargs = dict(
        data=data,
        selected_episode_ids=np.array([0, 1, 2]),
        center=np.zeros(5, dtype=np.float32),
        scale=np.ones(5, dtype=np.float32),
        k_step=3,
        obs_dim=5,
        action_dim=1,
        max_windows=5,
    )
    first = KoopmanWindowDataset(**kwargs, seed=9)
    second = KoopmanWindowDataset(**kwargs, seed=9)
    assert np.array_equal(first.starts, second.starts)
    assert not np.array_equal(first.starts, np.arange(5))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("split_not_1d", "1D integer"),
        ("split_not_integer", "1D integer"),
        ("duplicate_within_split", "unique"),
        ("overlap_between_splits", "pairwise disjoint"),
        ("unknown_episode", "absent from episode_id"),
        ("omitted_episode", "union must exactly"),
    ],
)
def test_dataset_split_ids_are_a_strict_partition(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    payload = _dataset_archive()
    if mutation == "split_not_1d":
        payload["train_episode_ids"] = np.asarray([[0, 1]], dtype=np.int64)
    elif mutation == "split_not_integer":
        payload["train_episode_ids"] = np.asarray([0.0, 1.0])
    elif mutation == "duplicate_within_split":
        payload["train_episode_ids"] = np.asarray([0, 0, 1], dtype=np.int64)
    elif mutation == "overlap_between_splits":
        payload["validation_episode_ids"] = np.asarray([1, 2], dtype=np.int64)
    elif mutation == "unknown_episode":
        payload["validation_episode_ids"] = np.asarray([2, 99], dtype=np.int64)
    elif mutation == "omitted_episode":
        payload["train_episode_ids"] = np.asarray([0], dtype=np.int64)
    path = tmp_path / "dataset.npz"
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=message):
        load_dataset(path, "cartpole_swingup")


def test_valid_dataset_split_ids_produce_exact_transition_masks(tmp_path: Path):
    payload = _dataset_archive()
    path = tmp_path / "dataset.npz"
    np.savez(path, **payload)

    data, masks = load_dataset(path, "cartpole_swingup")

    assert masks["train"].sum() == 102
    assert masks["validation"].sum() == 51
    assert masks["test"].sum() == 51
    assert np.array_equal(
        sum((mask.astype(np.int8) for mask in masks.values())),
        np.ones(len(data["state"]), dtype=np.int8),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dataset_v3", "dataset schema version 4"),
        ("collection_v3", "collection schema version 4"),
        ("unapproved", "not formally approved"),
        ("cut_episode", "must not contain cut episodes"),
        ("incomplete_episode", "incomplete"),
        ("reward_wrong_shape", "reward must have shape"),
        ("reward_out_of_range", "reward targets must be in"),
    ],
)
def test_primary_loader_requires_v4_approved_complete_episode_contract(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    payload = _dataset_archive()
    if mutation == "dataset_v3":
        payload["dataset_schema_version"] = np.asarray(3, dtype=np.int64)
    elif mutation == "collection_v3":
        payload["collection_schema_version"] = np.asarray(3, dtype=np.int64)
    elif mutation == "unapproved":
        payload["training_approved"] = np.asarray(False)
    elif mutation == "cut_episode":
        payload["collector_truncated"][-1] = True
    elif mutation == "incomplete_episode":
        payload["done"][-1] = False
        payload["truncated"][-1] = False
    elif mutation == "reward_wrong_shape":
        payload["reward"] = payload["reward"][:, None]
    elif mutation == "reward_out_of_range":
        payload["reward"][0] = 1.01
    path = tmp_path / "dataset.npz"
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=message):
        load_dataset(path, "cartpole_swingup")


def test_stage_coverage_requires_complete_episodes_in_each_training_third():
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    execution_spec = resolve_execution_spec(experiment, "development")
    complete = _dataset_archive()
    report = trainer_module._validate_stage_coverage(complete, execution_spec)
    assert report["stage_ends"] == {"early": 162, "mid": 325, "late": 488}
    assert report["episode_counts_by_train_seed_index"]["0"] == {
        "early": 1,
        "mid": 1,
        "late": 2,
    }

    early_only = _dataset_archive()
    early_only["update"][:] = 20
    with pytest.raises(ValueError, match="early/mid/late"):
        trainer_module._validate_stage_coverage(early_only, execution_spec)


def test_window_index_scans_episode_boundaries_once(monkeypatch):
    data = _episodes(num_episodes=40, steps_per_episode=4)
    original = trainer_module.np.flatnonzero
    calls = 0

    def counting_flatnonzero(values):
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(trainer_module.np, "flatnonzero", counting_flatnonzero)
    dataset = KoopmanWindowDataset(
        data,
        np.arange(40, dtype=np.int64),
        np.zeros(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
        k_step=2,
        obs_dim=5,
        action_dim=1,
    )

    assert len(dataset) == 120
    assert calls == 1


def test_window_index_rejects_noncontiguous_episode_blocks():
    data = _episodes()
    permutation = np.concatenate(
        (
            np.arange(0, 3),
            np.arange(7, 14),
            np.arange(3, 7),
            np.arange(14, 21),
        )
    )
    broken = {name: values[permutation] for name, values in data.items()}

    with pytest.raises(ValueError, match="one contiguous block"):
        KoopmanWindowDataset(
            broken,
            np.asarray([0, 1, 2], dtype=np.int64),
            np.zeros(5, dtype=np.float32),
            np.ones(5, dtype=np.float32),
            k_step=2,
            obs_dim=5,
            action_dim=1,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"center": np.zeros(4)}, "center must have shape"),
        ({"center": np.asarray([0, 0, np.nan, 0, 0])}, "center.*finite"),
        ({"scale": np.ones(4)}, "scale must have shape"),
        ({"scale": np.asarray([1, 1, 0, 1, 1])}, "scale.*positive"),
        ({"scale": np.asarray([1, 1, np.inf, 1, 1])}, "scale.*finite"),
        ({"max_windows": 0}, "max_windows must be a positive integer"),
        ({"max_windows": -1}, "max_windows must be a positive integer"),
        ({"max_windows": 2.5}, "max_windows must be a positive integer"),
        ({"max_windows": True}, "max_windows must be a positive integer"),
    ],
)
def test_window_dataset_validates_normalizer_and_limit(override, message):
    kwargs = {
        "data": _episodes(),
        "selected_episode_ids": np.asarray([0, 1, 2], dtype=np.int64),
        "center": np.zeros(5, dtype=np.float32),
        "scale": np.ones(5, dtype=np.float32),
        "k_step": 3,
        "obs_dim": 5,
        "action_dim": 1,
        "max_windows": None,
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        KoopmanWindowDataset(**kwargs)


def test_fit_normalizer_is_chunked_and_matches_float64_reference(monkeypatch):
    rng = np.random.default_rng(1729)
    state = rng.normal(loc=1e5, scale=3.0, size=(37, 5)).astype(np.float32)
    next_state = rng.normal(loc=-1e5, scale=7.0, size=(37, 5)).astype(np.float32)
    reference = np.concatenate((state, next_state)).astype(np.float64)
    expected_center = reference.mean(axis=0).astype(np.float32)
    expected_scale = reference.std(axis=0).astype(np.float32)

    def concatenate_forbidden(*_args, **_kwargs):
        raise AssertionError("fit_normalizer must not concatenate full inputs")

    monkeypatch.setattr(trainer_module.np, "concatenate", concatenate_forbidden)
    center, scale = fit_normalizer(state, next_state, chunk_size=7)

    assert np.allclose(center, expected_center, rtol=1e-6, atol=1e-5)
    assert np.allclose(scale, expected_scale, rtol=1e-6, atol=1e-5)
    assert center.dtype == scale.dtype == np.float32


@pytest.mark.parametrize(
    ("state_value", "next_value", "message"),
    [
        (np.empty((0, 5)), np.empty((0, 5)), "must not be empty"),
        (np.zeros((2, 5)), np.zeros((2, 4)), "feature dimensions"),
        (
            np.asarray([[0, 0, np.nan, 0, 0]], dtype=np.float32),
            np.zeros((1, 5)),
            "NaN or Inf",
        ),
    ],
)
def test_fit_normalizer_rejects_invalid_inputs(state_value, next_value, message):
    with pytest.raises((ValueError, FloatingPointError), match=message):
        fit_normalizer(state_value, next_value, chunk_size=2)


def _loader_order(loader, epoch: int) -> list[int]:
    _set_loader_epoch(loader, epoch)
    return torch.cat([batch[0] for batch in loader]).tolist()


def test_epoch_sampler_is_repeatable_and_resume_equivalent():
    dataset = TensorDataset(torch.arange(31))
    uninterrupted = _make_loader(
        dataset, batch_size=6, shuffle=True, seed=1234
    )
    resumed = _make_loader(dataset, batch_size=6, shuffle=True, seed=1234)

    epoch_seven = _loader_order(uninterrupted, 7)
    assert isinstance(uninterrupted.sampler, EpochRandomSampler)
    assert epoch_seven == _loader_order(uninterrupted, 7)
    assert epoch_seven == _loader_order(resumed, 7)
    assert epoch_seven != _loader_order(uninterrupted, 8)
    assert sorted(epoch_seven) == list(range(31))


def test_resume_history_is_deduplicated_and_disk_tail_is_replaced(tmp_path: Path):
    first = {"epoch": 1, "train_total": 3.0, "elapsed_seconds": 1.0}
    repeated_first = {
        "epoch": 1,
        "train_total": 2.5,
        "elapsed_seconds": 1.5,
    }
    second = {"epoch": 2, "train_total": 2.0, "elapsed_seconds": 2.0}
    normalized = _normalize_resume_history(
        [first, repeated_first, second], completed_epoch=2
    )
    assert normalized == [repeated_first, second]

    path = tmp_path / "history.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in [first, first, second])
        + "\n",
        encoding="utf-8",
    )
    _synchronize_history_file(path, normalized)
    on_disk = [json.loads(line) for line in path.read_text().splitlines()]
    assert on_disk == normalized
    assert [record["epoch"] for record in on_disk] == [1, 2]


def test_resume_history_rejects_missing_or_future_epochs():
    with pytest.raises(ValueError, match="exactly one recoverable"):
        _normalize_resume_history([{"epoch": 2}], completed_epoch=2)
    with pytest.raises(ValueError, match="within the checkpoint"):
        _normalize_resume_history([{"epoch": 1}, {"epoch": 3}], completed_epoch=2)


def test_hopper_template_identity_skip_and_full_loss_contract_are_preserved():
    experiment = load_experiment_config("experiments/dmc/configs/hopper_hop.yaml")
    config = train_config_from_experiment(experiment)
    assert {
        "linear": config.linear_weight,
        "rollout": config.rollout_weight,
        "stability": config.stability_weight,
        "latent_std": config.latent_std_weight,
        "identity": config.identity_weight,
    } == {
        "linear": 10.0,
        "rollout": 1.0,
        "stability": 0.1,
        "latent_std": 0.1,
        "identity": 1e-4,
    }
    assert config.activation == "silu"
    assert config.rollout_discount == 0.99

    model = DeepKoopman(5, 1, lift_dim=3, hidden_dims=(8,))
    states = torch.randn(2, 4, 5)
    actions = torch.randn(2, 3, 1)
    lifted = model.lift(states)
    torch.testing.assert_close(lifted[..., :5], states, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        model.reconstruct(lifted), states, rtol=0.0, atol=0.0
    )
    losses = koopman_loss(
        model,
        states,
        actions,
        rollout_discount=config.rollout_discount,
        linear_weight=config.linear_weight,
        rollout_weight=config.rollout_weight,
        stability_weight=config.stability_weight,
        latent_std_weight=config.latent_std_weight,
        identity_weight=config.identity_weight,
        controllability_svd_weight=config.controllability_svd_weight,
        augmentation_weight=config.augmentation_weight,
        reconstruction_weight=config.reconstruction_weight,
        spectral_radius_limit=config.spectral_radius_limit,
        target_latent_std=config.target_latent_std,
        svd_min_singular_value=config.svd_min_singular_value,
    )
    for name in ("linear", "rollout", "stability", "latent_std", "identity"):
        assert torch.isfinite(getattr(losses, name))


def test_dmc_k_step_horizons_are_selected_in_physical_time():
    expected_steps = {
            "cartpole_swingup": 50,
        "reacher_hard": 20,
        "hopper_hop": 40,
        "walker_run": 30,
        "humanoid_run": 20,
        "humanoid_run_pure_state": 20,
    }
    expected_seconds = {
        "cartpole_swingup": 0.5,
        "reacher_hard": 0.4,
        "hopper_hop": 0.8,
        "walker_run": 0.75,
        "humanoid_run": 0.5,
        "humanoid_run_pure_state": 0.5,
    }
    for task, steps in expected_steps.items():
        spec = TASK_SPECS[task]
        assert spec.k_step == steps
        assert spec.k_step * spec.native_control_dt == pytest.approx(
            expected_seconds[task]
        )
    assert len(set(expected_steps.values())) > 1


def test_best_checkpoint_atomic_write_does_not_replace_on_serialization_failure(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "best.pt"
    path.write_bytes(b"stable-checkpoint")

    def interrupted_save(_payload, stream):
        stream.write(b"partial-checkpoint")
        raise OSError("simulated interruption")

    monkeypatch.setattr(trainer_module.torch, "save", interrupted_save)
    with pytest.raises(OSError, match="simulated interruption"):
        trainer_module._atomic_torch_save(path, {"value": 1})

    assert path.read_bytes() == b"stable-checkpoint"
    assert list(tmp_path.glob(".best.pt.*.tmp")) == []


def test_rng_capture_restore_replays_cpu_and_cuda_streams():
    original = trainer_module._capture_rng_state()

    def draw():
        values = {
            "python": trainer_module.random.random(),
            "numpy": float(trainer_module.np.random.random()),
            "torch": trainer_module.torch.rand(4),
        }
        if torch.cuda.is_available():
            values["cuda"] = torch.rand(4, device="cuda").cpu()
        return values

    expected = draw()
    trainer_module._restore_rng_state(original)
    actual = draw()
    trainer_module._restore_rng_state(original)

    assert actual["python"] == expected["python"]
    assert actual["numpy"] == expected["numpy"]
    torch.testing.assert_close(actual["torch"], expected["torch"])
    if "cuda" in expected:
        torch.testing.assert_close(actual["cuda"], expected["cuda"])


def test_latest_resume_contract_validates_optimizer_rng_history_and_identity():
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = train_config_from_experiment(experiment)
    expected_identity = {"task_name": experiment.task, "dataset_sha256": "a" * 64}
    architecture = {
        "architecture": "fullA_history_v2_adapted",
        "state_dim": 5,
        "action_dim": 1,
        "lift_dim": config.lift_dim,
        "hidden_dims": list(config.hidden_dims),
        "activation": config.activation,
    }
    reward_model = TransitionRewardModel(
        5,
        1,
        hidden_dims=config.reward_hidden_dims,
        activation=config.activation,
    )
    koopman_model = DeepKoopman(
        5,
        1,
        lift_dim=config.lift_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
    )
    optimizer = torch.optim.Adam(
        [*koopman_model.parameters(), *reward_model.parameters()],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        amsgrad=config.adam_amsgrad,
    )
    for parameter in [*koopman_model.parameters(), *reward_model.parameters()]:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    reward_architecture = reward_model.architecture()
    reward_metrics = {
        "transitions": 31,
        "mse": 0.1,
        "rmse": float(np.sqrt(0.1)),
        "mae": 0.2,
        "prediction_mean": 0.4,
        "target_mean": 0.5,
    }
    best_dynamics = 0.4
    best_joint = best_dynamics + config.reward_loss_weight * reward_metrics["mse"]
    payload = {
        "format_version": 3,
        "architecture": architecture,
        "model": koopman_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 1,
        "best_validation": best_joint,
        "best_validation_joint_objective": best_joint,
        "best_validation_rollout_normalized_mse": best_dynamics,
        "best_validation_reward_metrics": reward_metrics,
        "latest_validation_joint_objective": best_joint,
        "latest_validation_rollout_normalized_mse": best_dynamics,
        "latest_validation_reward_metrics": reward_metrics,
        "validation_selection_metric": (
            "rollout_normalized_mse_plus_weighted_reward_mse"
        ),
        "reward_model_architecture": reward_architecture,
        "reward_model_input_contract": transition_reward_input_contract(),
        "reward_model_state": reward_model.state_dict(),
        "config": asdict(config),
        "normalizers": {"center": [0.0] * 5, "scale": [1.0] * 5},
        "rng_state": trainer_module._capture_rng_state(),
        "history": [
            {
                "epoch": 1,
                "train_total": 1.0,
                "validation_joint_objective": best_joint,
                "validation_rollout_normalized_mse": best_dynamics,
                "validation_reward_mse": reward_metrics["mse"],
                "validation_reward_rmse": reward_metrics["rmse"],
                "validation_reward_mae": reward_metrics["mae"],
            }
        ],
        "training_state": {
            **expected_identity,
            "best_epoch": 1,
            "epochs_without_improvement": 0,
        },
    }

    history = trainer_module._validate_resume_checkpoint(
        payload,
        expected_training_state=expected_identity,
        expected_architecture=architecture,
        expected_reward_architecture=reward_architecture,
        config=config,
        state_dim=5,
    )
    assert history == payload["history"]

    stale = {**payload, "training_state": dict(payload["training_state"])}
    stale["training_state"]["dataset_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        trainer_module._validate_resume_checkpoint(
            stale,
            expected_training_state=expected_identity,
            expected_architecture=architecture,
            expected_reward_architecture=reward_architecture,
            config=config,
            state_dim=5,
        )

    incomplete_rng = {**payload, "rng_state": {"random": (), "numpy": ()}}
    with pytest.raises(ValueError, match="RNG state is incomplete"):
        trainer_module._validate_resume_checkpoint(
            incomplete_rng,
            expected_training_state=expected_identity,
            expected_architecture=architecture,
            expected_reward_architecture=reward_architecture,
            config=config,
            state_dim=5,
        )

    malformed_reward = {
        **payload,
        "reward_model_state": dict(payload["reward_model_state"]),
    }
    malformed_reward["reward_model_state"].pop(
        next(iter(malformed_reward["reward_model_state"]))
    )
    with pytest.raises(ValueError, match="model state is invalid"):
        trainer_module._validate_resume_checkpoint(
            malformed_reward,
            expected_training_state=expected_identity,
            expected_architecture=architecture,
            expected_reward_architecture=reward_architecture,
            config=config,
            state_dim=5,
        )

    incomplete_optimizer = {
        **payload,
        "optimizer": {
            **payload["optimizer"],
            "state": dict(payload["optimizer"]["state"]),
        },
    }
    incomplete_optimizer["optimizer"]["state"].pop(
        next(iter(incomplete_optimizer["optimizer"]["state"]))
    )
    with pytest.raises(ValueError, match="slots are incomplete"):
        trainer_module._validate_resume_checkpoint(
            incomplete_optimizer,
            expected_training_state=expected_identity,
            expected_architecture=architecture,
            expected_reward_architecture=reward_architecture,
            config=config,
            state_dim=5,
        )


def test_train_config_is_exactly_the_complete_yaml_resolution():
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = train_config_from_experiment(experiment)

    assert trainer_module._train_config_mapping(config) == resolve_koopman_config(
        experiment
    )


def test_train_fails_closed_before_reading_data_or_constructing_optimizer(
    tmp_path: Path,
    monkeypatch,
):
    _forbid_optimizer(monkeypatch)
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = train_config_from_experiment(experiment)
    missing_dataset = tmp_path / "must-not-be-read.npz"

    with pytest.raises(TypeError, match="dry_run"):
        train(
            experiment.task,
            missing_dataset,
            tmp_path / "truthy-dry-run",
            config,
            dry_run=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PermissionError, match="experiment config"):
        train(
            experiment.task,
            missing_dataset,
            tmp_path / "missing-config",
            config,
            profile="development",
            preflight_file=tmp_path / "missing-preflight.json",
            dry_run=True,
        )
    with pytest.raises(PermissionError, match="profile"):
        train(
            experiment.task,
            missing_dataset,
            tmp_path / "missing-profile",
            config,
            experiment_config=experiment,
            preflight_file=tmp_path / "missing-preflight.json",
            dry_run=True,
        )
    with pytest.raises(PermissionError, match="preflight"):
        train(
            experiment.task,
            missing_dataset,
            tmp_path / "missing-preflight",
            config,
            experiment_config=experiment,
            profile="development",
            dry_run=True,
        )
    with pytest.raises(PermissionError, match="approval"):
        train(
            experiment.task,
            missing_dataset,
            tmp_path / "missing-approval",
            config,
            experiment_config=experiment,
            profile="development",
            preflight_file=tmp_path / "not-read.json",
        )


def test_train_rejects_type_coercible_free_hyperparameter_override(
    tmp_path: Path,
    monkeypatch,
):
    _forbid_optimizer(monkeypatch)
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = train_config_from_experiment(experiment)
    # ``True == 1.0`` in Python.  The approval boundary deliberately compares
    # canonical JSON identities so this is still an unauthorized type change.
    overridden = replace(config, target_latent_std=True)

    with pytest.raises(ValueError, match="exactly match"):
        train(
            experiment.task,
            tmp_path / "must-not-be-read.npz",
            tmp_path / "output",
            overridden,
            experiment_config=experiment,
            profile="development",
            preflight_file=tmp_path / "must-not-be-read.json",
            dry_run=True,
        )


def test_dry_run_writes_bound_manifest_without_model_or_optimizer_steps(
    tmp_path: Path,
    monkeypatch,
):
    _forbid_optimizer(monkeypatch)
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = train_config_from_experiment(experiment)
    preflight = _write_preflight(tmp_path, experiment)
    dataset = _write_dataset(
        tmp_path,
        authorization=_authorization_for(experiment, preflight),
    )
    output_dir = tmp_path / "dry-run"

    result = train(
        experiment.task,
        dataset,
        output_dir,
        config,
        experiment_config=experiment,
        profile="development",
        preflight_file=preflight,
        dry_run=True,
    )

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert result["optimizer_steps"] == result["epochs_completed"] == 0
    assert manifest["training_approved"] is False
    assert manifest["optimizer_steps"] == manifest["epochs_completed"] == 0
    assert manifest["config"] == resolve_koopman_config(experiment)
    assert manifest["reward_model_input_contract"] == (
        transition_reward_input_contract()
    )
    assert manifest["reward_model_architecture"] == {
        "architecture": TransitionRewardModel.ARCHITECTURE,
        "state_dim": 5,
        "action_dim": 1,
        "hidden_dims": list(config.reward_hidden_dims),
        "activation": config.activation,
    }
    assert manifest["config_fingerprint"] == experiment.fingerprint
    assert manifest["protocol_fingerprint"] == protocol_fingerprint(
        _environment_protocol()
    )
    assert manifest["dataset_authorization"]["config_fingerprint"] == (
        experiment.fingerprint
    )
    assert manifest["dataset_window_counts"] == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    assert not (output_dir / "best.pt").exists()
    assert not (output_dir / "latest.pt").exists()
    assert not (output_dir / "history.jsonl").exists()


def test_formal_train_rejects_dataset_runtime_protocol_before_optimizer(
    tmp_path: Path,
    monkeypatch,
):
    _forbid_optimizer(monkeypatch)
    experiment = load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    config = train_config_from_experiment(experiment)
    reviewed_protocol = _environment_protocol(control_dt=0.01)
    preflight = _write_preflight(
        tmp_path,
        experiment,
        environment_protocol=reviewed_protocol,
    )
    approval = tmp_path / "approval.json"
    write_training_approval(
        experiment,
        "development",
        preflight,
        approval,
        approve=True,
    )
    dataset = _write_dataset(
        tmp_path,
        environment_protocol=_environment_protocol(control_dt=0.02),
        authorization=_authorization_for(
            experiment,
            preflight,
            approval=approval,
        ),
    )

    with pytest.raises(TrainingApprovalError, match="Runtime DMC protocol"):
        train(
            experiment.task,
            dataset,
            tmp_path / "formal",
            config,
            experiment_config=experiment,
            profile="development",
            preflight_file=preflight,
            approval_file=approval,
        )


def test_cli_has_no_free_hyperparameter_overrides():
    required = [
        "--config",
        "experiments/dmc/configs/cartpole_swingup.yaml",
        "--profile",
        "development",
        "--preflight-file",
        "preflight.json",
        "--dataset",
        "dataset.npz",
        "--output-dir",
        "output",
        "--dry-run",
    ]
    with pytest.raises(SystemExit):
        parse_args([*required, "--epochs", "1"])

    formal = [value for value in required if value != "--dry-run"]
    with pytest.raises(SystemExit):
        parse_args(formal)


def test_cli_dry_run_uses_only_resolved_yaml_and_never_constructs_optimizer(
    tmp_path: Path,
    monkeypatch,
):
    _forbid_optimizer(monkeypatch)
    config_path = Path("experiments/dmc/configs/cartpole_swingup.yaml")
    experiment = load_experiment_config(config_path)
    preflight = _write_preflight(tmp_path, experiment)
    dataset = _write_dataset(
        tmp_path,
        authorization=_authorization_for(experiment, preflight),
    )
    output = tmp_path / "cli-dry-run"

    main(
        [
            "--config",
            str(config_path),
            "--profile",
            "development",
            "--preflight-file",
            str(preflight),
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["training_approved"] is False
    assert manifest["optimizer_steps"] == 0
    assert manifest["config"] == resolve_koopman_config(experiment)
