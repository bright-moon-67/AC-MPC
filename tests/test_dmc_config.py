from __future__ import annotations

import copy
from dataclasses import fields
import json
from pathlib import Path

import pytest

from experiments.dmc.config import (
    DATA_SPLITS,
    load_experiment_config,
    resolve_execution_spec,
    resolve_koopman_config,
    resolve_ppo_config,
    validate_config,
)
from experiments.dmc.actors import ACTOR_TYPES
from experiments.dmc.koopman.train_dmc_koopman import TrainConfig
from experiments.dmc.ppo.train_dmc_ppo import PPOConfig
from experiments.dmc.tasks.registry import TASK_SPECS


CONFIG_ROOT = Path("experiments/dmc/configs")
CARTPOLE_CONFIG = CONFIG_ROOT / "cartpole_swingup.yaml"


def _cartpole_raw() -> dict:
    return copy.deepcopy(load_experiment_config(CARTPOLE_CONFIG).raw)


def test_all_dmc_task_configs_validate_and_have_exact_budgets():
    configs = {
        load_experiment_config(path).task: path for path in CONFIG_ROOT.glob("*.yaml")
    }
    assert set(configs) == set(TASK_SPECS)
    for path in configs.values():
        config = load_experiment_config(path)
        assert config.raw["evaluation"]["episodes_per_seed"] == 10
        assert config.raw["evaluation"]["reference_episodes_per_seed"] == 1
        assert config.raw["evaluation"]["diagnostic_every_steps"] == 50_000
        assert config.raw["evaluation"]["checkpoint"] == "latest"
        assert config.raw["actors"]["types"] == list(ACTOR_TYPES)
        assert config.raw["profiles"]["development"]["train_seed_count"] == 1
        assert config.raw["profiles"]["benchmark"]["train_seed_count"] == 3
        for profile_name, profile in config.raw["profiles"].items():
            rollout_batch = profile["num_envs"] * profile["rollout_steps"]
            assert profile["num_envs"] == 256
            assert profile["rollout_steps"] == 8
            assert profile["minibatch_size"] == 256
            assert profile["update_epochs"] == 2
            if config.task == "cartpole_swingup" and profile_name == "development":
                assert profile["total_timesteps"] == 9_998_336
            else:
                assert profile["total_timesteps"] == 999_424
            assert profile["total_timesteps"] % rollout_batch == 0


def test_native_protocol_rejects_silent_reacher_20hz_override():
    config = copy.deepcopy(
        load_experiment_config(CONFIG_ROOT / "reacher_hard.yaml").raw
    )
    config["protocol"]["control_timestep"] = 0.05
    with pytest.raises(ValueError, match="native"):
        validate_config(config)


def test_primary_evaluation_checkpoint_is_preregistered_as_latest():
    config = _cartpole_raw()
    config["evaluation"]["checkpoint"] = "best"
    with pytest.raises(ValueError, match="latest checkpoint"):
        validate_config(config)


def test_training_and_evaluation_seeds_are_disjoint_unique_and_nonnegative():
    overlap = _cartpole_raw()
    overlap["seeds"]["evaluation"][0] = overlap["seeds"]["train"][0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_config(overlap)

    duplicate = _cartpole_raw()
    duplicate["seeds"]["train"][1] = duplicate["seeds"]["train"][0]
    with pytest.raises(ValueError, match="unique"):
        validate_config(duplicate)

    negative = _cartpole_raw()
    negative["seeds"]["evaluation"][0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_config(negative)


def test_config_fingerprint_binds_seed_and_all_resolved_hyperparameters():
    config = load_experiment_config(CARTPOLE_CONFIG)
    assert config.fingerprint.startswith("sha256:")
    for mutate in (
        lambda raw: raw["data"].__setitem__(
            "max_transitions_per_train_seed",
            raw["data"]["max_transitions_per_train_seed"] + 1,
        ),
        lambda raw: raw["seeds"]["train"].__setitem__(0, 20260821),
        lambda raw: raw["ppo"].__setitem__("checkpoint_interval_updates", 11),
        lambda raw: raw["koopman"].__setitem__("epochs", 501),
    ):
        changed = copy.deepcopy(config.raw)
        mutate(changed)
        validate_config(changed, path=config.path)
        changed_config = type(config)(path=config.path, raw=changed)
        assert changed_config.fingerprint != config.fingerprint


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("profiles.development", "num_envs", 128),
        ("profiles.development", "rollout_steps", 4),
        ("profiles.development", "update_epochs", 8),
        ("ppo", "anneal_learning_rate", True),
        ("ppo", "initial_std", 0.5),
        ("ppo", "entropy_coefficient", 1e-3),
        ("ppo", "normalize_advantage", False),
        ("ppo", "adam_epsilon", 1e-5),
    ),
)
def test_primary_ppo_contract_rejects_non_acme_reference_settings(
    section: str, field: str, value: object
):
    raw = _cartpole_raw()
    target = raw
    for name in section.split("."):
        target = target[name]
    target[field] = value
    with pytest.raises(ValueError, match="Acme"):
        validate_config(raw)


def test_primary_ppo_contract_allows_batch_aligned_budget_extension():
    raw = _cartpole_raw()
    raw["profiles"]["development"]["total_timesteps"] = 4_882 * 2_048
    validate_config(raw)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw.__setitem__("unexpected", 1),
        lambda raw: raw["protocol"].__setitem__("unexpected", 1),
        lambda raw: raw["profiles"]["development"].__setitem__("unexpected", 1),
        lambda raw: raw["ppo"].__setitem__("unexpected", 1),
        lambda raw: raw["data"].__setitem__("unexpected", 1),
        lambda raw: raw["koopman"].__setitem__("unexpected", 1),
        lambda raw: raw["actors"]["architecture"].__setitem__("unexpected", 1),
        lambda raw: raw["evaluation"].__setitem__("unexpected", 1),
        lambda raw: raw["proposed_gates"].__setitem__("unexpected", 1),
    ),
)
def test_unknown_keys_are_rejected_at_every_level(mutate):
    raw = _cartpole_raw()
    mutate(raw)
    with pytest.raises(ValueError, match="unknown"):
        validate_config(raw)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw.pop("ppo"),
        lambda raw: raw["protocol"].pop("score"),
        lambda raw: raw["profiles"]["development"].pop("train_seed_count"),
        lambda raw: raw["koopman"].pop("hidden_dims"),
        lambda raw: raw["actors"]["architecture"].pop("action_limit"),
    ),
)
def test_missing_keys_are_rejected_at_every_level(mutate):
    raw = _cartpole_raw()
    mutate(raw)
    with pytest.raises(ValueError, match="missing"):
        validate_config(raw)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw["profiles"]["development"].__setitem__(
            "learning_rate", float("nan")
        ),
        lambda raw: raw["ppo"].__setitem__("discount", float("inf")),
        lambda raw: raw["koopman"].__setitem__("weight_decay", float("nan")),
        lambda raw: raw["proposed_gates"].__setitem__(
            "ppo_mean_return_min", float("nan")
        ),
    ),
)
def test_non_finite_values_are_rejected(mutate):
    raw = _cartpole_raw()
    mutate(raw)
    with pytest.raises(ValueError, match="finite"):
        validate_config(raw)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw["protocol"].__setitem__("action_repeat", True),
        lambda raw: raw["seeds"]["train"].__setitem__(0, False),
        lambda raw: raw["profiles"]["development"].__setitem__("num_envs", True),
        lambda raw: raw["actors"]["architecture"].__setitem__("hidden_dim", True),
    ),
)
def test_boolean_values_are_not_accepted_as_integers(mutate):
    raw = _cartpole_raw()
    mutate(raw)
    with pytest.raises(ValueError, match="integer"):
        validate_config(raw)


def test_primary_action_limit_split_and_batch_budget_are_strict():
    action_limit = _cartpole_raw()
    action_limit["actors"]["architecture"]["action_limit"] = 1.1
    with pytest.raises(ValueError, match="action_limit=1.0"):
        validate_config(action_limit)

    split = _cartpole_raw()
    split["data"]["split"] = "transition_random"
    with pytest.raises(ValueError, match="data.split"):
        validate_config(split)
    assert split["data"]["split"] not in DATA_SPLITS

    budget = _cartpole_raw()
    budget["profiles"]["benchmark"]["total_timesteps"] += 1
    with pytest.raises(ValueError, match="rollout batch"):
        validate_config(budget)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("activation", "relu", "activation"),
        ("rollout_discount", 1.01, "rollout_discount"),
        ("adam_beta1", 1.0, "adam_beta1"),
        ("adam_beta2", -0.1, "adam_beta2"),
        ("adam_epsilon", 0.0, "adam_epsilon"),
        ("adam_amsgrad", 0, "boolean"),
    ],
)
def test_koopman_implicit_execution_defaults_are_strictly_resolved(
    field: str,
    value: object,
    message: str,
):
    raw = _cartpole_raw()
    raw["koopman"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reward_hidden_dims", [], "reward_hidden_dims"),
        ("reward_hidden_dims", [256, True], "integer"),
        ("reward_loss_weight", 0.0, "reward_loss_weight"),
        ("reward_loss_weight", float("nan"), "finite"),
    ],
)
def test_reward_model_training_contract_is_strict(
    field: str, value: object, message: str
):
    raw = _cartpole_raw()
    raw["koopman"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(raw)


def test_resolved_ppo_configs_match_trainer_fields_and_profile_seed_scope():
    config = load_experiment_config(CARTPOLE_CONFIG)
    expected_fields = {field.name for field in fields(PPOConfig)}

    development = resolve_ppo_config(config, "development")
    assert set(development) == expected_fields
    assert development["seed"] == config.raw["seeds"]["train"][0]
    assert development["update_epochs"] == 2
    assert development["num_envs"] * development["rollout_steps"] == 2048
    assert development["total_timesteps"] == 9_998_336
    PPOConfig(**development).validate()
    with pytest.raises(IndexError, match="approves 1"):
        resolve_ppo_config(config, "development", train_seed_index=1)

    benchmark = [
        resolve_ppo_config(config, "benchmark", train_seed_index=index)
        for index in range(3)
    ]
    assert [run["seed"] for run in benchmark] == config.raw["seeds"]["train"]
    assert all(set(run) == expected_fields for run in benchmark)
    with pytest.raises(IndexError, match="approves 3"):
        resolve_ppo_config(config, "benchmark", train_seed_index=3)


def test_resolved_koopman_config_matches_trainer_fields():
    config = load_experiment_config(CARTPOLE_CONFIG)
    resolved = resolve_koopman_config(config)
    assert set(resolved) == {field.name for field in fields(TrainConfig)}
    assert resolved["hidden_dims"] == [256, 256]
    assert resolved["reward_hidden_dims"] == [256, 256]
    assert resolved["reward_loss_weight"] == 1.0
    assert resolved["k_step"] == TASK_SPECS[config.task].k_step
    # The trainer accepts any sequence for hidden_dims; constructing its config
    # verifies that the mapping has no missing or extra optimization fields.
    train_config = TrainConfig(**resolved)
    assert list(train_config.hidden_dims) == [256, 256]


def test_resolved_execution_spec_is_self_contained_and_json_serializable():
    config = load_experiment_config(CARTPOLE_CONFIG)
    development = resolve_execution_spec(config, "development")
    benchmark = resolve_execution_spec(config, "benchmark")
    assert development["config_fingerprint"] == config.fingerprint
    assert len(development["ppo_runs"]) == 1
    assert len(benchmark["ppo_runs"]) == 3
    assert benchmark["koopman"] == resolve_koopman_config(config)
    encoded = json.dumps(benchmark, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == benchmark
