"""Strict, versioned configuration contract for reproducible DMC experiments.

The YAML files are the source of truth for every optimization parameter.  This
module deliberately does not import either trainer: the resolver functions
return plain, JSON-serializable mappings that launchers can pass to their
trainer-specific dataclasses without creating an import cycle.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.dmc.actors import ACTOR_TYPES, ActorConfig
from experiments.dmc.reward_oracle import validate_mpve_reward_source
from experiments.dmc.tasks.registry import DMC_NATIVE_PROTOCOL, get_task_spec


CONFIG_SCHEMA_VERSION = "dmc_experiment_v2"
NATIVE_PROTOCOL = DMC_NATIVE_PROTOCOL
PROFILE_NAMES = ("development", "benchmark")
DATA_SPLITS = frozenset({"episode_interleaved_8_1_1"})
STATUS_VALUES = frozenset(
    {
        "proposed_before_first_training",
        "provisional_after_cartpole_gate",
        "provisional_after_reacher_gate",
        "provisional_decision_gate",
        "optional_after_walker_gate",
        "requested_optional_high_dimensional_stress_test",
    }
)

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "task",
        "status",
        "protocol",
        "seeds",
        "profiles",
        "ppo",
        "data",
        "koopman",
        "actors",
        "evaluation",
        "proposed_gates",
    }
)
PROTOCOL_FIELDS = frozenset(
    {
        "name",
        "control_timestep",
        "action_repeat",
        "time_limit_seconds",
        "episode_steps",
        "score",
    }
)
SEED_FIELDS = frozenset({"train", "evaluation"})
PROFILE_FIELDS = frozenset(
    {
        "train_seed_count",
        "num_envs",
        "rollout_steps",
        "minibatch_size",
        "update_epochs",
        "total_timesteps",
        "learning_rate",
    }
)
PPO_FIELDS = frozenset(
    {
        "anneal_learning_rate",
        "discount",
        "gae_lambda",
        "clip_ratio",
        "value_coefficient",
        "entropy_coefficient",
        "initial_std",
        "max_grad_norm",
        "target_kl",
        "checkpoint_interval_updates",
        "max_wall_time_seconds",
        "critic_hidden_dim",
        "collect_flush_transitions",
        "normalize_observation",
        "normalize_advantage",
        "normalize_value",
        "normalization_ema_tau",
        "value_clip",
        "value_clipping_epsilon",
        "max_abs_reward",
        "adam_epsilon",
        "mpve_horizon",
        "mpve_value_loss_coefficient",
        "mpve_reward_source",
    }
)
DATA_FIELDS = frozenset(
    {
        "source",
        "max_transitions_per_train_seed",
        "collection_total_updates",
        "split",
    }
)
KOOPMAN_FIELDS = frozenset(
    {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "lift_dim",
        "hidden_dims",
        "reward_hidden_dims",
        "reward_loss_weight",
        "seed",
        "k_step",
        "activation",
        "rollout_discount",
        "linear_weight",
        "rollout_weight",
        "stability_weight",
        "latent_std_weight",
        "identity_weight",
        "controllability_svd_weight",
        "augmentation_weight",
        "reconstruction_weight",
        "svd_min_singular_value",
        "spectral_radius_limit",
        "stability_reference_dt",
        "target_latent_std",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "adam_amsgrad",
        "checkpoint_every",
        "patience",
        "max_windows",
    }
)
ACTOR_FIELDS = frozenset({"types", "architecture"})
ACTOR_ARCHITECTURE_FIELDS = frozenset(ActorConfig.__dataclass_fields__)
EVALUATION_FIELDS = frozenset(
    {
        "deterministic",
        "episodes_per_seed",
        "reference_episodes_per_seed",
        "diagnostic_every_steps",
        "checkpoint",
    }
)
GATE_FIELDS = frozenset(
    {
        "ppo_mean_return_min",
        "koopman_rollout_nmse_max",
        "koopman_vs_hold_rmse_ratio_max",
        "kmpc_to_ppo_return_ratio_min",
        "action_bound_fraction_max",
        "reward_model_rmse_max",
        "ac_mpc_mpve_to_kmpc_return_ratio_min",
    }
)

PPO_PROFILE_OUTPUT_FIELDS = (
    "num_envs",
    "rollout_steps",
    "minibatch_size",
    "update_epochs",
    "total_timesteps",
    "learning_rate",
)
PPO_SHARED_OUTPUT_FIELDS = (
    "anneal_learning_rate",
    "discount",
    "gae_lambda",
    "clip_ratio",
    "value_coefficient",
    "entropy_coefficient",
    "initial_std",
    "max_grad_norm",
    "target_kl",
    "checkpoint_interval_updates",
    "max_wall_time_seconds",
    "critic_hidden_dim",
    "collect_flush_transitions",
    "normalize_observation",
    "normalize_advantage",
    "normalize_value",
    "normalization_ema_tau",
    "value_clip",
    "value_clipping_epsilon",
    "max_abs_reward",
    "adam_epsilon",
    "mpve_horizon",
    "mpve_value_loss_coefficient",
    "mpve_reward_source",
)
KOOPMAN_OUTPUT_FIELDS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "lift_dim",
    "hidden_dims",
    "reward_hidden_dims",
    "reward_loss_weight",
    "seed",
    "k_step",
    "activation",
    "rollout_discount",
    "linear_weight",
    "rollout_weight",
    "stability_weight",
    "latent_std_weight",
    "identity_weight",
    "controllability_svd_weight",
    "augmentation_weight",
    "reconstruction_weight",
    "svd_min_singular_value",
    "spectral_radius_limit",
    "stability_reference_dt",
    "target_latent_std",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "adam_amsgrad",
    "checkpoint_every",
    "patience",
    "max_windows",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated view over one task's complete experiment YAML."""

    path: Path
    raw: dict[str, Any]

    @property
    def task(self) -> str:
        return str(self.raw["task"])

    @property
    def protocol(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw["protocol"])

    @property
    def actor_config(self) -> ActorConfig:
        return ActorConfig.from_mapping(
            copy.deepcopy(self.raw["actors"]["architecture"])
        )

    @property
    def canonical_json(self) -> str:
        """Canonical serialization binding every declared configuration field."""

        return json.dumps(
            self.raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


def _require_exact_keys(
    mapping: dict[str, Any], expected: frozenset[str], *, section: str
) -> None:
    actual = set(mapping)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        raise ValueError(f"Config section {section!r} has " + ", ".join(details))


def _require_mapping(
    parent: dict[str, Any], key: str, *, section: str
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        prefix = f"{section}." if section else ""
        raise ValueError(f"Config field {prefix}{key!r} must be a mapping")
    return value


def _integer(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config field {field!r} must be an integer")
    if minimum is not None and value < minimum:
        comparison = "non-negative" if minimum == 0 else f">= {minimum}"
        raise ValueError(f"Config field {field!r} must be {comparison}")
    return int(value)


def _positive_int(mapping: dict[str, Any], key: str, *, section: str) -> int:
    return _integer(mapping.get(key), field=f"{section}.{key}", minimum=1)


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config field {field!r} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Config field {field!r} must be a finite number")
    return result


def _bounded_number(
    value: Any,
    *,
    field: str,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> float:
    result = _finite_number(value, field=field)
    if lower is not None:
        invalid = result < lower if lower_inclusive else result <= lower
        if invalid:
            bracket = "[" if lower_inclusive else "("
            raise ValueError(
                f"Config field {field!r} must lie in {bracket}{lower}, ..."
            )
    if upper is not None:
        invalid = result > upper if upper_inclusive else result >= upper
        if invalid:
            bracket = "]" if upper_inclusive else ")"
            raise ValueError(
                f"Config field {field!r} must lie in ..., {upper}{bracket}"
            )
    return result


def _require_bool(mapping: dict[str, Any], key: str, *, section: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Config field {section + '.' + key!r} must be a boolean")
    return value


def _validate_protocol(raw: dict[str, Any], task_name: str) -> None:
    spec = get_task_spec(task_name)
    protocol = _require_mapping(raw, "protocol", section="")
    _require_exact_keys(protocol, PROTOCOL_FIELDS, section="protocol")
    if protocol["name"] != NATIVE_PROTOCOL:
        raise ValueError(
            "The primary benchmark config must use the named native protocol; "
            "put action-repeat ablations in a separate config"
        )
    if protocol["control_timestep"] != "native":
        raise ValueError("dmc_native_v1 requires control_timestep: native")
    action_repeat = _integer(
        protocol["action_repeat"], field="protocol.action_repeat", minimum=1
    )
    if action_repeat != 1:
        raise ValueError("dmc_native_v1 requires action_repeat: 1")
    if (
        _positive_int(protocol, "episode_steps", section="protocol")
        != spec.native_step_limit
    ):
        raise ValueError(
            f"Protocol episode_steps does not match live registry for {spec.name}"
        )
    time_limit = _finite_number(
        protocol["time_limit_seconds"], field="protocol.time_limit_seconds"
    )
    if abs(time_limit - spec.native_time_limit) > 1e-12:
        raise ValueError("Protocol time_limit_seconds does not match the task registry")
    if protocol["score"] != "sum_official_reward":
        raise ValueError("Primary score must be the sum of official DMC rewards")


def _validate_seeds(raw: dict[str, Any]) -> None:
    seeds = _require_mapping(raw, "seeds", section="")
    _require_exact_keys(seeds, SEED_FIELDS, section="seeds")
    train_seeds = seeds["train"]
    eval_seeds = seeds["evaluation"]
    if not isinstance(train_seeds, list) or len(train_seeds) != 3:
        raise ValueError("seeds.train must contain exactly 3 seeds")
    if not isinstance(eval_seeds, list) or len(eval_seeds) != 10:
        raise ValueError("seeds.evaluation must contain exactly 10 seeds")
    for group_name, values in (("train", train_seeds), ("evaluation", eval_seeds)):
        normalized = [
            _integer(value, field=f"seeds.{group_name}[{index}]", minimum=0)
            for index, value in enumerate(values)
        ]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"seeds.{group_name} must contain unique seeds")
    if set(train_seeds) & set(eval_seeds):
        raise ValueError("Training and evaluation seeds must be disjoint")


def _validate_profiles(raw: dict[str, Any]) -> None:
    profiles = _require_mapping(raw, "profiles", section="")
    _require_exact_keys(profiles, frozenset(PROFILE_NAMES), section="profiles")
    expected_seed_counts = {"development": 1, "benchmark": 3}
    for profile_name in PROFILE_NAMES:
        profile = _require_mapping(profiles, profile_name, section="profiles")
        section = f"profiles.{profile_name}"
        _require_exact_keys(profile, PROFILE_FIELDS, section=section)
        train_seed_count = _positive_int(profile, "train_seed_count", section=section)
        if train_seed_count != expected_seed_counts[profile_name]:
            raise ValueError(
                f"{section}.train_seed_count must be "
                f"{expected_seed_counts[profile_name]}"
            )
        num_envs = _positive_int(profile, "num_envs", section=section)
        rollout = _positive_int(profile, "rollout_steps", section=section)
        minibatch = _positive_int(profile, "minibatch_size", section=section)
        _positive_int(profile, "update_epochs", section=section)
        total_timesteps = _positive_int(profile, "total_timesteps", section=section)
        rollout_batch = num_envs * rollout
        if minibatch > rollout_batch:
            raise ValueError(f"{section}.minibatch_size exceeds its rollout batch")
        if rollout_batch % minibatch:
            raise ValueError(
                f"{section} rollout batch must divide evenly by minibatch_size"
            )
        if total_timesteps % rollout_batch:
            raise ValueError(
                f"{section}.total_timesteps must divide evenly by its rollout batch"
            )
        _bounded_number(
            profile["learning_rate"],
            field=f"{section}.learning_rate",
            lower=0.0,
            lower_inclusive=False,
        )
        acme_reference = {
            "num_envs": 256,
            "rollout_steps": 8,
            "minibatch_size": 256,
            "update_epochs": 2,
            "learning_rate": 3e-4,
        }
        mismatches = {
            key: (profile[key], expected)
            for key, expected in acme_reference.items()
            if profile[key] != expected
        }
        if mismatches:
            raise ValueError(
                f"{section} must preserve the named Acme continuous-PPO "
                f"reference learner settings: {mismatches}"
            )


def _validate_ppo(raw: dict[str, Any]) -> None:
    ppo = _require_mapping(raw, "ppo", section="")
    _require_exact_keys(ppo, PPO_FIELDS, section="ppo")
    _require_bool(ppo, "anneal_learning_rate", section="ppo")
    for key in (
        "normalize_observation",
        "normalize_advantage",
        "normalize_value",
        "value_clip",
    ):
        _require_bool(ppo, key, section="ppo")
    _bounded_number(ppo["discount"], field="ppo.discount", lower=0.0, upper=1.0)
    _bounded_number(ppo["gae_lambda"], field="ppo.gae_lambda", lower=0.0, upper=1.0)
    _bounded_number(
        ppo["clip_ratio"],
        field="ppo.clip_ratio",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
        upper_inclusive=False,
    )
    for key in ("value_coefficient", "entropy_coefficient"):
        _bounded_number(ppo[key], field=f"ppo.{key}", lower=0.0)
    for key in ("initial_std", "max_grad_norm"):
        _bounded_number(
            ppo[key], field=f"ppo.{key}", lower=0.0, lower_inclusive=False
        )
    if not isinstance(ppo["mpve_reward_source"], str):
        raise ValueError("ppo.mpve_reward_source must be a string")
    validate_mpve_reward_source(raw["task"], ppo["mpve_reward_source"])
    if ppo["initial_std"] != 1.0:
        raise ValueError(
            "ppo.initial_std must remain 1.0 for Acme-aligned structured-actor "
            "exploration parity (plain PPO uses a state-dependent scale head)"
        )
    target_kl = ppo["target_kl"]
    if target_kl is not None:
        _bounded_number(
            target_kl,
            field="ppo.target_kl",
            lower=0.0,
            lower_inclusive=False,
        )
    max_wall_time = ppo["max_wall_time_seconds"]
    if max_wall_time is not None:
        _bounded_number(
            max_wall_time,
            field="ppo.max_wall_time_seconds",
            lower=0.0,
            lower_inclusive=False,
        )
    for key in (
        "checkpoint_interval_updates",
        "critic_hidden_dim",
        "collect_flush_transitions",
        "mpve_horizon",
    ):
        _positive_int(ppo, key, section="ppo")
    _bounded_number(
        ppo["normalization_ema_tau"],
        field="ppo.normalization_ema_tau",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
        upper_inclusive=False,
    )
    for key in (
        "value_clipping_epsilon",
        "adam_epsilon",
        "mpve_value_loss_coefficient",
    ):
        _bounded_number(
            ppo[key], field=f"ppo.{key}", lower=0.0, lower_inclusive=False
        )
    max_abs_reward = ppo["max_abs_reward"]
    if max_abs_reward is not None:
        _bounded_number(
            max_abs_reward,
            field="ppo.max_abs_reward",
            lower=0.0,
            lower_inclusive=False,
        )
    acme_reference = {
        "anneal_learning_rate": False,
        "discount": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coefficient": 1.0,
        "entropy_coefficient": 3e-4,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "critic_hidden_dim": 256,
        "normalize_observation": True,
        "normalize_advantage": True,
        "normalize_value": True,
        "normalization_ema_tau": 0.995,
        "value_clip": False,
        "value_clipping_epsilon": 0.2,
        "max_abs_reward": None,
        "adam_epsilon": 1e-7,
    }
    mismatches = {
        key: (ppo[key], expected)
        for key, expected in acme_reference.items()
        if ppo[key] != expected
    }
    if mismatches:
        raise ValueError(
            "Primary PPO settings must preserve the named Google DeepMind "
            f"Acme reference: {mismatches}"
        )


def _validate_data(raw: dict[str, Any]) -> None:
    data = _require_mapping(raw, "data", section="")
    _require_exact_keys(data, DATA_FIELDS, section="data")
    _positive_int(data, "max_transitions_per_train_seed", section="data")
    _positive_int(data, "collection_total_updates", section="data")
    if not isinstance(data["source"], str) or data["source"] != "ppo_training_stages":
        raise ValueError("Primary Koopman data must cover PPO training stages")
    if not isinstance(data["split"], str) or data["split"] not in DATA_SPLITS:
        raise ValueError(
            f"Unsupported data.split {data['split']!r}; expected one of "
            f"{sorted(DATA_SPLITS)}"
        )


def _validate_koopman(raw: dict[str, Any], task_name: str) -> None:
    spec = get_task_spec(task_name)
    koopman = _require_mapping(raw, "koopman", section="")
    _require_exact_keys(koopman, KOOPMAN_FIELDS, section="koopman")
    for key in (
        "epochs",
        "batch_size",
        "lift_dim",
        "seed",
        "k_step",
        "checkpoint_every",
        "patience",
        "max_windows",
    ):
        minimum = 0 if key == "seed" else 1
        _integer(koopman[key], field=f"koopman.{key}", minimum=minimum)
    if koopman["k_step"] != spec.k_step:
        raise ValueError("Config k_step does not match the native task registry")
    if koopman["activation"] != "silu":
        raise ValueError("Primary Koopman activation must be 'silu'")
    hidden_dims = koopman["hidden_dims"]
    if not isinstance(hidden_dims, list) or not hidden_dims:
        raise ValueError("koopman.hidden_dims must be a non-empty list")
    for index, width in enumerate(hidden_dims):
        _integer(width, field=f"koopman.hidden_dims[{index}]", minimum=1)
    reward_hidden_dims = koopman["reward_hidden_dims"]
    if not isinstance(reward_hidden_dims, list) or not reward_hidden_dims:
        raise ValueError("koopman.reward_hidden_dims must be a non-empty list")
    for index, width in enumerate(reward_hidden_dims):
        _integer(
            width,
            field=f"koopman.reward_hidden_dims[{index}]",
            minimum=1,
        )
    for key in (
        "learning_rate",
        "gradient_clip",
        "stability_reference_dt",
        "target_latent_std",
        "adam_epsilon",
    ):
        _bounded_number(
            koopman[key],
            field=f"koopman.{key}",
            lower=0.0,
            lower_inclusive=False,
        )
    for key in (
        "weight_decay",
        "linear_weight",
        "rollout_weight",
        "stability_weight",
        "latent_std_weight",
        "identity_weight",
        "controllability_svd_weight",
        "augmentation_weight",
        "reconstruction_weight",
        "svd_min_singular_value",
    ):
        _bounded_number(koopman[key], field=f"koopman.{key}", lower=0.0)
    _bounded_number(
        koopman["reward_loss_weight"],
        field="koopman.reward_loss_weight",
        lower=0.0,
        lower_inclusive=False,
    )
    _bounded_number(
        koopman["rollout_discount"],
        field="koopman.rollout_discount",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    for key in ("adam_beta1", "adam_beta2"):
        _bounded_number(
            koopman[key],
            field=f"koopman.{key}",
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
    _require_bool(koopman, "adam_amsgrad", section="koopman")
    _bounded_number(
        koopman["spectral_radius_limit"],
        field="koopman.spectral_radius_limit",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )


def _validate_actors(raw: dict[str, Any]) -> None:
    actors = _require_mapping(raw, "actors", section="")
    _require_exact_keys(actors, ACTOR_FIELDS, section="actors")
    actor_types = actors["types"]
    if actor_types != list(ACTOR_TYPES):
        raise ValueError(f"actors.types must be exactly {list(ACTOR_TYPES)}")
    architecture = _require_mapping(actors, "architecture", section="actors")
    _require_exact_keys(
        architecture, ACTOR_ARCHITECTURE_FIELDS, section="actors.architecture"
    )
    for key in (
        "hidden_dim",
        "ppo_hidden_dim",
        "ppo_hidden_layers",
        "ab_rank",
        "kmpc_horizon",
        "kmpc_solver_iterations",
    ):
        _positive_int(architecture, key, section="actors.architecture")
    if architecture["ppo_activation"] != "relu":
        raise ValueError("Primary Acme-reference PPO activation must be 'relu'")
    if (
        architecture["ppo_distribution"]
        != "tanh_squashed_state_dependent_gaussian"
    ):
        raise ValueError(
            "Primary Acme-reference PPO must use the state-dependent "
            "tanh-squashed diagonal Gaussian"
        )
    if architecture["ppo_hidden_dim"] != 256 or architecture[
        "ppo_hidden_layers"
    ] != 3:
        raise ValueError("Primary Acme-reference PPO must use 3x256 hidden layers")
    action_limit = _finite_number(
        architecture["action_limit"], field="actors.architecture.action_limit"
    )
    if action_limit != 1.0:
        raise ValueError(
            "Primary DMC actors require action_limit=1.0 to match the action spec"
        )
    ActorConfig.from_mapping(copy.deepcopy(architecture)).validate()
    if raw["ppo"]["mpve_horizon"] > architecture["kmpc_horizon"]:
        raise ValueError("ppo.mpve_horizon cannot exceed actors kmpc_horizon")


def _validate_evaluation_and_gates(raw: dict[str, Any]) -> None:
    evaluation = _require_mapping(raw, "evaluation", section="")
    _require_exact_keys(evaluation, EVALUATION_FIELDS, section="evaluation")
    if _positive_int(evaluation, "episodes_per_seed", section="evaluation") != 10:
        raise ValueError("evaluation.episodes_per_seed must be 10")
    reference_episodes = _positive_int(
        evaluation, "reference_episodes_per_seed", section="evaluation"
    )
    if reference_episodes != 1:
        raise ValueError(
            "evaluation.reference_episodes_per_seed must be 1 for the "
            "pre-registered ten-episode Acme-aligned summary"
        )
    if reference_episodes > evaluation["episodes_per_seed"]:
        raise ValueError("reference episodes cannot exceed robustness episodes")
    if (
        _positive_int(
            evaluation, "diagnostic_every_steps", section="evaluation"
        )
        != 50_000
    ):
        raise ValueError("evaluation.diagnostic_every_steps must be 50000")
    if _require_bool(evaluation, "deterministic", section="evaluation") is not True:
        raise ValueError("Primary evaluation must use deterministic actor means")
    if evaluation["checkpoint"] != "latest":
        raise ValueError(
            "Primary evaluation must use the fixed-budget latest checkpoint"
        )

    gates = _require_mapping(raw, "proposed_gates", section="")
    _require_exact_keys(gates, GATE_FIELDS, section="proposed_gates")
    for key in GATE_FIELDS:
        upper = 1.0 if key == "action_bound_fraction_max" else None
        _bounded_number(
            gates[key], field=f"proposed_gates.{key}", lower=0.0, upper=upper
        )


def validate_config(raw: dict[str, Any], *, path: Path | None = None) -> None:
    if not isinstance(raw, dict):
        raise ValueError("DMC experiment config must be a mapping")
    _require_exact_keys(raw, TOP_LEVEL_FIELDS, section="root")
    if raw["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config schema {raw['schema_version']!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )
    if not isinstance(raw["task"], str):
        raise ValueError("Config field 'task' must be a string")
    task_name = raw["task"]
    get_task_spec(task_name)
    if not isinstance(raw["status"], str) or raw["status"] not in STATUS_VALUES:
        raise ValueError(
            f"Unsupported status {raw['status']!r}; expected one of "
            f"{sorted(STATUS_VALUES)}"
        )

    _validate_protocol(raw, task_name)
    _validate_seeds(raw)
    _validate_profiles(raw)
    _validate_ppo(raw)
    _validate_data(raw)
    _validate_koopman(raw, task_name)
    _validate_actors(raw)
    _validate_evaluation_and_gates(raw)

    if path is not None and path.suffix not in {".yaml", ".yml"}:
        raise ValueError("DMC experiment configs must be YAML files")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    validate_config(raw, path=path)
    return ExperimentConfig(path=path, raw=raw)


def _validated_config(config: ExperimentConfig) -> ExperimentConfig:
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    validate_config(config.raw, path=config.path)
    return config


def resolve_ppo_config(
    config: ExperimentConfig,
    profile: str,
    *,
    train_seed_index: int = 0,
) -> dict[str, Any]:
    """Resolve one PPO run to the exact plain mapping consumed by PPOConfig.

    ``train_seed_index`` indexes the approved seed subset: development exposes
    only the first configured seed, while benchmark exposes all three.
    """

    config = _validated_config(config)
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile {profile!r}; expected {PROFILE_NAMES}")
    index = _integer(train_seed_index, field="train_seed_index", minimum=0)
    profile_mapping = config.raw["profiles"][profile]
    seed_count = int(profile_mapping["train_seed_count"])
    if index >= seed_count:
        raise IndexError(
            f"Profile {profile!r} approves {seed_count} training seed(s), "
            f"not index {index}"
        )
    resolved = {
        key: copy.deepcopy(profile_mapping[key]) for key in PPO_PROFILE_OUTPUT_FIELDS
    }
    resolved.update(
        {
            key: copy.deepcopy(config.raw["ppo"][key])
            for key in PPO_SHARED_OUTPUT_FIELDS
        }
    )
    resolved["seed"] = int(config.raw["seeds"]["train"][index])
    resolved["collect_max_transitions"] = int(
        config.raw["data"]["max_transitions_per_train_seed"]
    )
    return resolved


def resolve_koopman_config(config: ExperimentConfig) -> dict[str, Any]:
    """Resolve the exact plain mapping consumed by the DMC Koopman trainer."""

    config = _validated_config(config)
    return {
        key: copy.deepcopy(config.raw["koopman"][key])
        for key in KOOPMAN_OUTPUT_FIELDS
    }


def resolve_execution_spec(
    config: ExperimentConfig, profile: str
) -> dict[str, Any]:
    """Return a self-contained, JSON-serializable approval payload."""

    config = _validated_config(config)
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile {profile!r}; expected {PROFILE_NAMES}")
    count = int(config.raw["profiles"][profile]["train_seed_count"])
    return {
        "kind": "dmc_resolved_execution_spec",
        "schema_version": CONFIG_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "task": config.task,
        "status": config.raw["status"],
        "profile": profile,
        "protocol": copy.deepcopy(config.raw["protocol"]),
        "ppo_runs": [
            resolve_ppo_config(config, profile, train_seed_index=index)
            for index in range(count)
        ],
        "koopman": resolve_koopman_config(config),
        "data": copy.deepcopy(config.raw["data"]),
        "actors": copy.deepcopy(config.raw["actors"]),
        "evaluation": copy.deepcopy(config.raw["evaluation"]),
        "evaluation_seeds": copy.deepcopy(config.raw["seeds"]["evaluation"]),
        "proposed_gates": copy.deepcopy(config.raw["proposed_gates"]),
    }


def default_config_path(task_name: str) -> Path:
    get_task_spec(task_name)
    return Path(__file__).with_name("configs") / f"{task_name}.yaml"
