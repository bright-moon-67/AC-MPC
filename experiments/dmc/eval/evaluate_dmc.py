"""Deterministic, checkpoint-driven evaluation for DMC actors.

The actor checkpoint is the source of truth for task, actor type, actor
architecture, training seed, and environment protocol.  Evaluation never
reconstructs an architecture from ad-hoc CLI defaults.  Structured actors also
restore and validate the exact Koopman checkpoint referenced by the actor run.

Example::

    python -m experiments.dmc.eval.evaluate_dmc \
        --actor-checkpoint runs/dmc/ppo/cartpole_swingup/development/seed_20260811/PPO/latest.pt \
        --eval-seed 20260901 --episodes 10

Use :mod:`experiments.dmc.eval.aggregate_dmc` for the canonical ten-evaluation-
seed report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import numpy as np
import torch

from experiments.dmc.actors import (
    ACTOR_TYPES,
    ActorConfig,
    actor_config_from_checkpoint,
    actor_mean,
    build_actor,
    checkpoint_protocol_fingerprint,
    load_koopman,
    normalizer_arrays,
)
from experiments.dmc.reward_model import transition_reward_input_contract
from experiments.dmc.reward_oracle import (
    LEARNED_TRANSITION_REWARD,
    OFFICIAL_OBSERVATION_ORACLE,
    exact_reward_oracle_metadata,
    validate_mpve_reward_source,
)
from experiments.dmc.tasks.registry import (
    DMC_CUSTOM_PROTOCOL,
    DMC_NATIVE_PROTOCOL,
    get_task_spec,
)


EVALUATION_SCHEMA_VERSION = "dmc_evaluation_v1"
DEFAULT_EPISODES_PER_EVAL_SEED = 10
DEFAULT_EVAL_SEED = 20_260_901

# ``dmc_ppo_actor`` is the stable payload emitted for both best and latest
# snapshots by the generic DMC PPO trainer.
SUPPORTED_ACTOR_CHECKPOINT_KINDS = frozenset({"dmc_ppo_actor"})
ACTOR_CHECKPOINT_FORMAT_VERSION = 3
PPO_TRAINING_SPEC_VERSION = "dmc_ppo_v4_raw_observation_critic"
FORMAL_AUTHORIZATION_KIND = "dmc_training_approval_v1"
CONFIG_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PROTOCOL_REQUIRED_FIELDS = frozenset(
    {
        "protocol_name",
        "protocol_schema_version",
        "task",
        "dm_control_version",
        "mujoco_version",
        "obs_dim",
        "action_dim",
        "control_dt",
        "physics_dt",
        "n_substeps",
        "time_limit",
        "step_limit",
        "action_low",
        "action_high",
        "obs_layout",
    }
)
KOOPMAN_LINEAGE_FIELDS = frozenset(
    {
        "dataset_sha256",
        "config_fingerprint",
        "approval_profile",
        "approval_file_sha256",
        "preflight_report_sha256",
    }
)


@dataclass(frozen=True)
class ActorCheckpointMetadata:
    path: Path
    kind: str
    task: str
    actor_type: str
    actor_config: ActorConfig
    protocol: dict[str, Any]
    training_seed: int
    authorization_kind: str | None
    training_approved: bool | None
    config_fingerprint: str | None
    approval_profile: str | None
    approval_file_sha256: str | None
    preflight_report_sha256: str | None
    train_seed_index: int | None
    authorization_verified: bool
    authorization_errors: tuple[str, ...]
    payload: dict[str, Any]

    @property
    def authorization(self) -> dict[str, Any]:
        return {
            "authorization_kind": self.authorization_kind,
            "training_approved": self.training_approved,
            "config_fingerprint": self.config_fingerprint,
            "approval_profile": self.approval_profile,
            "approval_file_sha256": self.approval_file_sha256,
            "preflight_report_sha256": self.preflight_report_sha256,
            "train_seed_index": self.train_seed_index,
            "authorization_verified": self.authorization_verified,
            "authorization_errors": list(self.authorization_errors),
        }


def _device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer_metadata(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Actor checkpoint field {key!r} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"Actor checkpoint field {key!r} must be non-negative")
    return value


def _authorization_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Parse formal authorization identity without rejecting legacy tests."""

    errors: list[str] = []
    kind_value = payload.get("authorization_kind")
    authorization_kind = kind_value if isinstance(kind_value, str) else None
    if authorization_kind != FORMAL_AUTHORIZATION_KIND:
        errors.append(
            f"authorization_kind must be {FORMAL_AUTHORIZATION_KIND}"
        )

    approved_value = payload.get("training_approved")
    training_approved = approved_value if isinstance(approved_value, bool) else None
    if training_approved is not True:
        errors.append("training_approved must be true")

    config_value = payload.get("config_fingerprint")
    config_fingerprint = config_value if isinstance(config_value, str) else None
    if (
        config_fingerprint is None
        or CONFIG_FINGERPRINT_PATTERN.fullmatch(config_fingerprint) is None
    ):
        errors.append("config_fingerprint must be a full sha256 fingerprint")

    profile_value = payload.get("approval_profile")
    approval_profile = profile_value if isinstance(profile_value, str) else None
    if approval_profile not in {"development", "benchmark"}:
        errors.append("approval_profile must be development or benchmark")

    approval_hash_value = payload.get("approval_file_sha256")
    approval_file_sha256 = (
        approval_hash_value if isinstance(approval_hash_value, str) else None
    )
    if (
        approval_file_sha256 is None
        or SHA256_PATTERN.fullmatch(approval_file_sha256) is None
    ):
        errors.append("approval_file_sha256 must be 64 lowercase hex characters")

    preflight_hash_value = payload.get("preflight_report_sha256")
    preflight_report_sha256 = (
        preflight_hash_value if isinstance(preflight_hash_value, str) else None
    )
    if (
        preflight_report_sha256 is None
        or SHA256_PATTERN.fullmatch(preflight_report_sha256) is None
    ):
        errors.append(
            "preflight_report_sha256 must be 64 lowercase hex characters"
        )

    index_value = payload.get("train_seed_index")
    if (
        isinstance(index_value, bool)
        or not isinstance(index_value, (int, np.integer))
        or int(index_value) < 0
    ):
        train_seed_index = None
        errors.append("train_seed_index must be a non-negative integer")
    else:
        train_seed_index = int(index_value)

    return {
        "authorization_kind": authorization_kind,
        "training_approved": training_approved,
        "config_fingerprint": config_fingerprint,
        "approval_profile": approval_profile,
        "approval_file_sha256": approval_file_sha256,
        "preflight_report_sha256": preflight_report_sha256,
        "train_seed_index": train_seed_index,
        "authorization_verified": not errors,
        "authorization_errors": tuple(errors),
    }


def load_actor_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ActorCheckpointMetadata:
    """Load and strictly validate the architecture-independent actor metadata."""

    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Actor checkpoint {path} must contain a mapping")

    kind = payload.get("kind")
    if kind not in SUPPORTED_ACTOR_CHECKPOINT_KINDS:
        raise ValueError(
            f"Unsupported actor checkpoint kind {kind!r}; expected one of "
            f"{sorted(SUPPORTED_ACTOR_CHECKPOINT_KINDS)}"
        )
    if payload.get("format_version") != ACTOR_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported actor checkpoint format_version "
            f"{payload.get('format_version')!r}; expected "
            f"{ACTOR_CHECKPOINT_FORMAT_VERSION}"
        )
    if payload.get("training_spec_version") != PPO_TRAINING_SPEC_VERSION:
        raise ValueError(
            f"Unsupported training_spec_version "
            f"{payload.get('training_spec_version')!r}; expected "
            f"{PPO_TRAINING_SPEC_VERSION!r}"
        )
    task = payload.get("task")
    if not isinstance(task, str):
        raise ValueError("Actor checkpoint is missing string field 'task'")
    get_task_spec(task)  # reject unknown task names before constructing modules

    actor_type = payload.get("actor_type")
    actor_name = payload.get("actor_name")
    if actor_type is None:
        actor_type = actor_name
    elif actor_name is not None and actor_name != actor_type:
        raise ValueError(
            f"Checkpoint actor_type {actor_type!r} and actor_name "
            f"{actor_name!r} disagree"
        )
    if actor_type not in ACTOR_TYPES:
        raise ValueError(
            f"Actor checkpoint actor_type must be one of {ACTOR_TYPES}, got "
            f"{actor_type!r}"
        )

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Actor checkpoint field 'protocol' must be a mapping")
    protocol = dict(protocol)
    _validate_saved_protocol(protocol, task)
    expected_protocol_fingerprint = payload.get("protocol_fingerprint")
    if not isinstance(expected_protocol_fingerprint, str):
        raise ValueError("Actor checkpoint is missing protocol_fingerprint")
    actual_protocol_fingerprint = _mapping_fingerprint(protocol)
    if expected_protocol_fingerprint != actual_protocol_fingerprint:
        raise ValueError(
            "Actor checkpoint protocol_fingerprint does not match protocol metadata"
        )

    if "actor_state" not in payload or not isinstance(payload["actor_state"], dict):
        raise ValueError("Actor checkpoint is missing mapping field 'actor_state'")
    actor_config = actor_config_from_checkpoint(payload)
    if actor_type == "AC-MPC-MPVE":
        ppo_config = payload.get("ppo_config")
        if not isinstance(ppo_config, Mapping):
            raise ValueError("AC-MPC-MPVE checkpoint is missing ppo_config")
        horizon = ppo_config.get("mpve_horizon")
        coefficient = ppo_config.get("mpve_value_loss_coefficient")
        reward_source = ppo_config.get("mpve_reward_source")
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, (int, np.integer))
            or not 1 <= int(horizon) <= actor_config.kmpc_horizon
        ):
            raise ValueError("AC-MPC-MPVE checkpoint has an invalid MPVE horizon")
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float, np.integer, np.floating))
            or not math.isfinite(float(coefficient))
            or float(coefficient) <= 0
        ):
            raise ValueError(
                "AC-MPC-MPVE checkpoint has an invalid value-loss coefficient"
            )
        if not isinstance(reward_source, str):
            raise ValueError("AC-MPC-MPVE checkpoint has no reward source")
        validate_mpve_reward_source(task, reward_source)
        reward_metadata = (
            exact_reward_oracle_metadata(task)
            if reward_source == OFFICIAL_OBSERVATION_ORACLE
            else {
                "source": LEARNED_TRANSITION_REWARD,
                "model_input_contract": transition_reward_input_contract(),
                "checkpoint_field": "reward_model_state",
            }
        )
        expected_value_expansion = {
            "enabled": True,
            "kind": "mpve_td_k_tro25_eq8_eq9_v1",
            "actor_shared_with": "KMPC",
            "horizon": int(horizon),
            "value_loss_coefficient": float(coefficient),
            "prediction_gradient": "detached",
            "terminal_target_gradient": "detached",
            "standard_gae_value_loss_retained": True,
            "reward": reward_metadata,
        }
        if payload.get("value_expansion") != expected_value_expansion:
            raise ValueError(
                "AC-MPC-MPVE checkpoint value_expansion metadata is invalid"
            )
    training_seed = _integer_metadata(payload, "training_seed")
    authorization = _authorization_metadata(payload)
    return ActorCheckpointMetadata(
        path=path,
        kind=str(kind),
        task=task,
        actor_type=str(actor_type),
        actor_config=actor_config,
        protocol=protocol,
        training_seed=training_seed,
        **authorization,
        payload=payload,
    )


def _protocol_name(protocol: Mapping[str, Any]) -> str:
    name = protocol.get("protocol_name", protocol.get("name"))
    if name not in {DMC_NATIVE_PROTOCOL, DMC_CUSTOM_PROTOCOL}:
        raise ValueError(
            "Checkpoint protocol must name dmc_native_v1 or dmc_custom_v1, "
            f"got {name!r}"
        )
    return str(name)


def _validate_saved_protocol(protocol: Mapping[str, Any], task: str) -> None:
    name = _protocol_name(protocol)
    missing = PROTOCOL_REQUIRED_FIELDS - protocol.keys()
    if missing:
        raise ValueError(
            f"Actor checkpoint protocol is missing fields: {sorted(missing)}"
        )
    if "task" in protocol and protocol["task"] != task:
        raise ValueError(
            f"Checkpoint protocol task {protocol['task']!r} does not match "
            f"actor task {task!r}"
        )
    action_repeat = protocol.get("action_repeat", 1)
    if action_repeat != 1:
        raise ValueError(
            "DMCAdapter evaluates one action per control step; checkpoints with "
            f"action_repeat={action_repeat!r} require a dedicated wrapper"
        )
    if protocol.get("score", "sum_official_reward") != "sum_official_reward":
        raise ValueError("Only sum_official_reward is supported for primary evaluation")
    if name == DMC_CUSTOM_PROTOCOL:
        control = protocol.get("control_dt", protocol.get("control_timestep"))
        if control in (None, "native"):
            raise ValueError("dmc_custom_v1 must save a numeric control timestep")


def _protocol_env_kwargs(protocol: Mapping[str, Any]) -> dict[str, Optional[float]]:
    control = protocol.get("control_dt", protocol.get("control_timestep"))
    if control in (None, "native"):
        control_timestep = None
    else:
        control_timestep = float(control)

    time_limit = protocol.get("time_limit", protocol.get("time_limit_seconds"))
    return {
        "control_timestep": control_timestep,
        "time_limit": None if time_limit is None else float(time_limit),
    }


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (float, np.floating)) or isinstance(
        actual, (float, np.floating)
    ):
        try:
            return bool(np.isclose(float(expected), float(actual), rtol=0.0, atol=1e-10))
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        try:
            expected_array = np.asarray(expected)
            actual_array = np.asarray(actual)
            if expected_array.shape != actual_array.shape:
                return False
            if expected_array.dtype.kind in "fiu" and actual_array.dtype.kind in "fiu":
                return bool(
                    np.allclose(
                        expected_array.astype(np.float64),
                        actual_array.astype(np.float64),
                        rtol=0.0,
                        atol=1e-10,
                    )
                )
            return bool(np.array_equal(expected_array, actual_array))
        except (TypeError, ValueError):
            return expected == actual
    return expected == actual


def validate_runtime_protocol(
    saved: Mapping[str, Any], runtime: Mapping[str, Any], task: str
) -> None:
    """Reject environment drift against either config- or runtime-style metadata."""

    missing = PROTOCOL_REQUIRED_FIELDS - runtime.keys()
    if missing:
        raise RuntimeError(
            f"Live DMC protocol metadata is missing fields: {sorted(missing)}"
        )
    aliases = {
        "name": "protocol_name",
        "control_timestep": "control_dt",
        "physics_timestep": "physics_dt",
        "time_limit_seconds": "time_limit",
        "episode_steps": "step_limit",
    }
    mismatches: dict[str, tuple[Any, Any]] = {}
    for saved_key, expected in saved.items():
        if saved_key in {"action_repeat", "score"}:
            continue
        runtime_key = aliases.get(saved_key, saved_key)
        if saved_key == "control_timestep" and expected == "native":
            continue
        if runtime_key not in runtime:
            # Config-only annotations are valid but do not describe live physics.
            continue
        actual = runtime[runtime_key]
        if not _values_match(expected, actual):
            mismatches[saved_key] = (expected, actual)
    if runtime.get("task") != task:
        mismatches["task"] = (task, runtime.get("task"))
    if runtime.get("protocol_name") != _protocol_name(saved):
        mismatches["protocol_name"] = (
            _protocol_name(saved),
            runtime.get("protocol_name"),
        )
    if mismatches:
        raise RuntimeError(
            f"Live DMC protocol does not match actor checkpoint: {mismatches}"
        )


def _reference_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    reference = payload.get("koopman")
    if reference is None:
        reference = {}
    elif isinstance(reference, (str, Path)):
        reference = {"path": str(reference)}
    elif isinstance(reference, dict):
        reference = dict(reference)
    else:
        raise TypeError("Actor checkpoint 'koopman' field must be a path or mapping")
    if "path" not in reference:
        for key in ("koopman_path", "koopman_checkpoint"):
            if payload.get(key) is not None:
                reference["path"] = str(payload[key])
                break
    if "sha256" not in reference and payload.get("koopman_sha256") is not None:
        reference["sha256"] = str(payload["koopman_sha256"])
    if "task" not in reference and payload.get("koopman_task") is not None:
        reference["task"] = str(payload["koopman_task"])
    return reference


def _resolve_saved_path(value: str | Path, actor_checkpoint: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = (actor_checkpoint.parent / path, Path.cwd() / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def resolve_koopman_path(
    metadata: ActorCheckpointMetadata,
    override: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a structured actor's Koopman path and validate its saved identity."""

    reference = _reference_mapping(metadata.payload)
    if reference.get("task") not in (None, metadata.task):
        raise ValueError(
            f"Actor checkpoint references Koopman task {reference.get('task')!r}, "
            f"not {metadata.task!r}"
        )
    if override is None:
        if "path" not in reference:
            raise ValueError(
                f"{metadata.actor_type} checkpoint must save its Koopman path or "
                "evaluation must receive --koopman"
            )
        path = _resolve_saved_path(reference["path"], metadata.path)
    else:
        path = Path(override).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Koopman checkpoint not found: {path}")
    expected_hash = reference.get("sha256")
    if not isinstance(expected_hash, str):
        raise ValueError(
            f"{metadata.actor_type} checkpoint must save koopman_sha256"
        )
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Koopman SHA-256 mismatch: expected {expected_hash}, got "
            f"{actual_hash} for {path}"
        )
    return path, reference


def episode_seeds(eval_seed: int, episodes: int) -> list[int]:
    """Derive non-overlapping episode resets from one evaluation-replicate seed."""

    if isinstance(eval_seed, bool) or not isinstance(eval_seed, (int, np.integer)):
        raise TypeError("eval_seed must be an integer")
    if int(eval_seed) < 0:
        raise ValueError("eval_seed must be non-negative")
    if isinstance(episodes, bool) or not isinstance(episodes, (int, np.integer)):
        raise TypeError("episodes must be an integer")
    episodes = int(episodes)
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    sequence = np.random.SeedSequence(int(eval_seed))
    children = sequence.spawn(episodes)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def _discount_value(info: Mapping[str, Any]) -> Optional[float]:
    if "discount" not in info:
        raise RuntimeError("DMC adapter info is missing the original discount")
    discount = info["discount"]
    if discount is None:
        return None
    value = float(discount)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(f"Invalid DMC transition discount {discount!r}")
    return value


def _actor_normalizer_arrays(
    payload: Mapping[str, Any], task: str
) -> tuple[np.ndarray, np.ndarray]:
    normalizer = payload.get("normalizer")
    if not isinstance(normalizer, dict):
        raise ValueError("Structured actor checkpoint must save its normalizer")

    def array(name: str) -> np.ndarray:
        value = normalizer.get(name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        result = np.asarray(value, dtype=np.float32)
        expected = (get_task_spec(task).obs_dim,)
        if result.shape != expected:
            raise ValueError(
                f"Actor normalizer {name} shape {result.shape} does not match "
                f"task shape {expected}"
            )
        if not np.isfinite(result).all():
            raise FloatingPointError(f"Actor normalizer {name} contains NaN or Inf")
        return result

    center = array("center")
    scale = array("scale")
    if np.any(scale <= 0):
        raise ValueError("Actor normalizer scale must be strictly positive")
    return center, scale


def _ppo_observation_normalizer_arrays(
    payload: Mapping[str, Any], task: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Restore the frozen Acme Welford statistics used by plain PPO."""

    ppo_config = payload.get("ppo_config")
    if not isinstance(ppo_config, Mapping) or ppo_config.get(
        "normalize_observation"
    ) is not True:
        raise ValueError(
            "PPO checkpoint must declare normalize_observation=true"
        )
    normalization_contract = payload.get("normalization_contract")
    if not isinstance(normalization_contract, Mapping) or (
        normalization_contract.get("observation")
        != "acme_welford_running_mean_std_v1"
        or normalization_contract.get("observation_scope")
        != "plain_ppo_raw_task_state"
    ):
        raise ValueError("PPO checkpoint observation-normalization contract is invalid")
    state = payload.get("observation_normalizer_state")
    if not isinstance(state, Mapping) or state.get("kind") != (
        "acme_welford_observation_normalizer_v1"
    ):
        raise ValueError("PPO checkpoint has no Acme observation normalizer")
    expected_dimension = get_task_spec(task).obs_dim
    if state.get("dimension") != expected_dimension:
        raise ValueError("PPO observation-normalizer dimension is invalid")
    epsilon = state.get("epsilon")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float, np.integer, np.floating))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0
    ):
        raise ValueError("PPO observation-normalizer epsilon is invalid")
    count = state.get("count")
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or int(count) < 1
    ):
        raise ValueError("Evaluable PPO checkpoint has empty observation statistics")

    def array(name: str) -> np.ndarray:
        value = state.get(name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (expected_dimension,) or not np.isfinite(result).all():
            raise ValueError(f"PPO observation-normalizer {name} is invalid")
        return result

    mean = array("mean")
    std = array("std")
    summed_variance = array("summed_variance")
    if np.any(summed_variance < 0):
        raise ValueError("PPO observation-normalizer variance is negative")
    expected_std = np.clip(
        np.sqrt(summed_variance / int(count)),
        float(epsilon),
        1e6,
    )
    if np.any(std <= 0):
        raise ValueError("PPO observation-normalizer std must be positive")
    if not np.allclose(std, expected_std, rtol=1e-6, atol=1e-7):
        raise ValueError("PPO observation-normalizer std is inconsistent")
    return mean, std, int(count)


def _koopman_checkpoint_field(payload: Mapping[str, Any], name: str) -> Any:
    if name in payload:
        return payload[name]
    training_state = payload.get("training_state")
    if isinstance(training_state, Mapping):
        return training_state.get(name)
    return None


def _validate_koopman_lineage(
    metadata: ActorCheckpointMetadata,
    koopman_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind a structured actor's saved lineage to the loaded Koopman file."""

    saved = metadata.payload.get("koopman_lineage")
    if saved is None:
        if metadata.authorization_verified:
            raise ValueError(
                "Formal structured actor checkpoint is missing koopman_lineage"
            )
        return None
    if not isinstance(saved, Mapping) or set(saved) != KOOPMAN_LINEAGE_FIELDS:
        raise ValueError(
            "Actor koopman_lineage must contain the complete canonical fields"
        )
    saved = dict(saved)
    actual = {
        field: _koopman_checkpoint_field(koopman_payload, field)
        for field in KOOPMAN_LINEAGE_FIELDS
    }
    if saved != actual:
        raise ValueError(
            "Actor checkpoint Koopman lineage does not match the loaded model"
        )
    dataset_sha256 = saved["dataset_sha256"]
    if (
        not isinstance(dataset_sha256, str)
        or SHA256_PATTERN.fullmatch(dataset_sha256) is None
    ):
        raise ValueError("Koopman dataset_sha256 must be 64 lowercase hex characters")
    if metadata.payload.get("koopman_dataset_sha256") != dataset_sha256:
        raise ValueError("Actor koopman_dataset_sha256 disagrees with lineage")
    if (
        metadata.payload.get("koopman_config_fingerprint")
        != saved["config_fingerprint"]
    ):
        raise ValueError("Actor koopman_config_fingerprint disagrees with lineage")
    config_fingerprint = saved.get("config_fingerprint")
    if (
        not isinstance(config_fingerprint, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", config_fingerprint) is None
    ):
        raise ValueError("Koopman config_fingerprint is invalid")
    if saved.get("approval_profile") not in {"development", "benchmark"}:
        raise ValueError("Koopman approval_profile is invalid")
    for field in ("approval_file_sha256", "preflight_report_sha256"):
        value = saved.get(field)
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Koopman {field} must be 64 lowercase hex characters")
    return saved


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def evaluate(
    actor_checkpoint: str | Path,
    *,
    koopman_path: str | Path | None = None,
    episodes: int = DEFAULT_EPISODES_PER_EVAL_SEED,
    eval_seed: int = DEFAULT_EVAL_SEED,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate one trained actor under one independent evaluation seed."""

    from experiments.dmc.tasks.adapter import make_dmc_adapter

    if episodes < 1:
        raise ValueError("episodes must be positive")
    device = _device(device_name)
    # PPO checkpoints also contain optimizer/value state.  Keep the payload on
    # CPU and let ``load_state_dict`` copy only actor weights to the target
    # device instead of needlessly occupying GPU memory with resume-only data.
    metadata = load_actor_checkpoint(actor_checkpoint, map_location="cpu")
    spec = get_task_spec(metadata.task)

    koopman = None
    koopman_checkpoint: Optional[dict[str, Any]] = None
    resolved_koopman: Optional[Path] = None
    center_tensor = None
    scale_tensor = None
    ppo_observation_center_tensor = None
    ppo_observation_scale_tensor = None
    ppo_observation_normalizer_count: int | None = None
    koopman_lineage: dict[str, Any] | None = None
    if metadata.actor_type != "PPO":
        resolved_koopman, _reference = resolve_koopman_path(metadata, koopman_path)
        koopman, koopman_checkpoint = load_koopman(
            resolved_koopman, metadata.task, device
        )
        koopman_protocol_fingerprint = checkpoint_protocol_fingerprint(
            koopman_checkpoint
        )
        if koopman_protocol_fingerprint != metadata.payload["protocol_fingerprint"]:
            raise ValueError(
                "Actor and Koopman checkpoints use different DMC environment protocols"
            )
        koopman_lineage = _validate_koopman_lineage(
            metadata, koopman_checkpoint
        )
        koopman_center, koopman_scale = normalizer_arrays(
            koopman_checkpoint, metadata.task
        )
        center, scale = _actor_normalizer_arrays(metadata.payload, metadata.task)
        if not np.array_equal(center, koopman_center) or not np.array_equal(
            scale, koopman_scale
        ):
            raise ValueError(
                "Actor and Koopman checkpoints use different state normalizers"
            )
        center_tensor = torch.as_tensor(center, dtype=torch.float32, device=device)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)
    else:
        if koopman_path is not None:
            raise ValueError("PPO evaluation does not consume a Koopman checkpoint")
        ppo_center, ppo_scale, ppo_observation_normalizer_count = (
            _ppo_observation_normalizer_arrays(metadata.payload, metadata.task)
        )
        ppo_observation_center_tensor = torch.as_tensor(
            ppo_center, dtype=torch.float32, device=device
        )
        ppo_observation_scale_tensor = torch.as_tensor(
            ppo_scale, dtype=torch.float32, device=device
        )

    actor = build_actor(
        metadata.actor_type,
        metadata.task,
        device,
        koopman=koopman,
        config=metadata.actor_config,
    )
    actor.load_state_dict(metadata.payload["actor_state"], strict=True)
    actor.eval()

    resets = episode_seeds(eval_seed, episodes)
    env_kwargs = _protocol_env_kwargs(metadata.protocol)
    envs: list[Any] = []
    returns_array = np.zeros(episodes, dtype=np.float64)
    discounted_returns_array = np.zeros(episodes, dtype=np.float64)
    continuations = np.ones(episodes, dtype=np.float64)
    lengths_array = np.zeros(episodes, dtype=np.int64)
    final_discounts: list[Optional[float]] = [None] * episodes
    final_infos: list[Optional[dict[str, Any]]] = [None] * episodes
    terminated_episodes = 0
    truncated_episodes = 0
    requested_bound_total = 0
    applied_bound_total = 0
    clipped_action_total = 0
    action_total = 0
    requested_bound_by_episode = np.zeros(episodes, dtype=np.int64)
    applied_bound_by_episode = np.zeros(episodes, dtype=np.int64)
    clipped_action_by_episode = np.zeros(episodes, dtype=np.int64)
    action_components_by_episode = np.zeros(episodes, dtype=np.int64)
    component_sums: dict[str, float] = {}
    runtime_protocol: dict[str, Any]
    try:
        for reset_seed in resets:
            envs.append(
                make_dmc_adapter(
                    metadata.task,
                    seed=reset_seed,
                    control_timestep=env_kwargs["control_timestep"],
                    time_limit=env_kwargs["time_limit"],
                )
            )
        runtime_protocol = dict(envs[0].protocol_metadata())
        validate_runtime_protocol(metadata.protocol, runtime_protocol, metadata.task)
        step_limit = int(envs[0].step_limit)
        if step_limit < 1:
            raise RuntimeError(f"DMC adapter returned invalid step_limit={step_limit}")
        low = np.asarray(envs[0].action_low, dtype=np.float32)
        high = np.asarray(envs[0].action_high, dtype=np.float32)
        # DMC bounds are [-1, 1], so 0.5% of the full span reproduces the
        # validated Hopper convention ``abs(action) >= 0.99``.
        bound_tolerance = 0.005 * (high - low)
        for env in envs[1:]:
            candidate_protocol = dict(env.protocol_metadata())
            validate_runtime_protocol(
                metadata.protocol, candidate_protocol, metadata.task
            )
            if candidate_protocol != runtime_protocol:
                raise RuntimeError(
                    "Evaluation environments have inconsistent runtime protocols"
                )
            if int(env.step_limit) != step_limit:
                raise RuntimeError("Evaluation environments have different step limits")
            if not np.array_equal(np.asarray(env.action_low), low) or not np.array_equal(
                np.asarray(env.action_high), high
            ):
                raise RuntimeError("Evaluation environments have different action bounds")

        observations = np.stack(
            [env.reset(seed=reset_seed) for env, reset_seed in zip(envs, resets)]
        ).astype(np.float32)
        active = np.ones(episodes, dtype=np.bool_)
        for _ in range(step_limit):
            active_indices = np.flatnonzero(active)
            if not len(active_indices):
                break
            state = torch.as_tensor(
                observations[active_indices], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                if metadata.actor_type == "PPO":
                    assert ppo_observation_center_tensor is not None
                    assert ppo_observation_scale_tensor is not None
                    normalized_state = (
                        state - ppo_observation_center_tensor
                    ) / ppo_observation_scale_tensor
                    action_tensor = actor_mean(
                        metadata.actor_type, actor, normalized_state, None
                    )
                else:
                    assert koopman is not None
                    assert center_tensor is not None and scale_tensor is not None
                    lifted = koopman.lift((state - center_tensor) / scale_tensor)
                    action_tensor = actor_mean(
                        metadata.actor_type, actor, state, lifted
                    )
            requested_batch = action_tensor.detach().cpu().numpy()
            expected_shape = (len(active_indices), spec.action_dim)
            if requested_batch.shape != expected_shape:
                raise RuntimeError(
                    f"Actor emitted shape {requested_batch.shape}, expected "
                    f"{expected_shape}"
                )
            if not np.isfinite(requested_batch).all():
                raise FloatingPointError("Actor emitted NaN or Inf")

            for batch_index, episode_index in enumerate(active_indices):
                requested = requested_batch[batch_index]
                observation, reward, done, info = envs[episode_index].step(requested)
                observations[episode_index] = observation
                lengths_array[episode_index] += 1
                reward = float(reward)
                if not math.isfinite(reward):
                    raise FloatingPointError(
                        "DMC environment emitted non-finite reward"
                    )
                discount = _discount_value(info)
                transition_terminated = bool(info.get("terminated", False))
                transition_truncated = bool(info.get("truncated", False))
                if transition_terminated and transition_truncated:
                    raise RuntimeError(
                        "DMC transition cannot be both terminated and truncated"
                    )
                if bool(done) != (transition_terminated or transition_truncated):
                    raise RuntimeError(
                        "DMC done disagrees with terminated/truncated diagnostics"
                    )
                returns_array[episode_index] += reward
                discounted_returns_array[episode_index] += (
                    continuations[episode_index] * reward
                )
                if discount is not None:
                    continuations[episode_index] *= discount

                requested_info = np.asarray(
                    info.get("requested_action", requested), dtype=np.float32
                ).reshape(-1)
                if "applied_action" not in info:
                    raise RuntimeError("DMC adapter info is missing applied_action")
                applied = np.asarray(
                    info["applied_action"], dtype=np.float32
                ).reshape(-1)
                if requested_info.shape != low.shape or applied.shape != low.shape:
                    raise RuntimeError(
                        "Adapter action diagnostics have the wrong shape"
                    )
                requested_bound_count = int(
                    (
                        (requested_info <= low + bound_tolerance)
                        | (requested_info >= high - bound_tolerance)
                    ).sum()
                )
                applied_bound_count = int(
                    (
                        (applied <= low + bound_tolerance)
                        | (applied >= high - bound_tolerance)
                    ).sum()
                )
                clipped_action_count = int(
                    (~np.isclose(requested_info, applied, rtol=0.0, atol=1e-7)).sum()
                )
                requested_bound_total += requested_bound_count
                applied_bound_total += applied_bound_count
                clipped_action_total += clipped_action_count
                action_total += spec.action_dim
                requested_bound_by_episode[episode_index] += requested_bound_count
                applied_bound_by_episode[episode_index] += applied_bound_count
                clipped_action_by_episode[episode_index] += clipped_action_count
                action_components_by_episode[episode_index] += spec.action_dim

                for key, value in info.get("reward_components", {}).items():
                    component = float(value)
                    if not math.isfinite(component):
                        raise FloatingPointError(
                            f"Reward component {key!r} is non-finite"
                        )
                    component_sums[key] = component_sums.get(key, 0.0) + component

                if done:
                    terminated_episodes += int(transition_terminated)
                    truncated_episodes += int(transition_truncated)
                    final_infos[episode_index] = dict(info)
                    final_discounts[episode_index] = discount
                    active[episode_index] = False

        if bool(active.any()):
            unfinished = np.flatnonzero(active).tolist()
            raise RuntimeError(
                f"DMC episodes {unfinished} did not finish within "
                f"env.step_limit={step_limit}"
            )
        for episode_index, final_info in enumerate(final_infos):
            if final_info is None:
                raise RuntimeError(
                    f"DMC episode {episode_index} has no final diagnostics"
                )
            reported_steps = int(
                final_info.get("step_count", lengths_array[episode_index])
            )
            if reported_steps != int(lengths_array[episode_index]):
                raise RuntimeError(
                    f"Adapter reported step_count={reported_steps}, observed "
                    f"{int(lengths_array[episode_index])} steps"
                )
    finally:
        for env in envs:
            env.close()

    returns = returns_array.tolist()
    environment_discounted_returns = discounted_returns_array.tolist()
    lengths = lengths_array.tolist()
    total_steps = int(sum(lengths))
    reference_episodes_per_seed = int(
        metadata.payload.get("evaluation_reference_episodes_per_seed", 1)
    )
    if not 1 <= reference_episodes_per_seed <= episodes:
        raise ValueError(
            "Checkpoint reference evaluation episode count is invalid"
        )
    reference_returns = returns[:reference_episodes_per_seed]
    reference_action_total = int(
        action_components_by_episode[:reference_episodes_per_seed].sum()
    )
    reference_requested_bound_total = int(
        requested_bound_by_episode[:reference_episodes_per_seed].sum()
    )
    reference_applied_bound_total = int(
        applied_bound_by_episode[:reference_episodes_per_seed].sum()
    )
    reference_clipped_action_total = int(
        clipped_action_by_episode[:reference_episodes_per_seed].sum()
    )
    numeric_final_discounts = [value for value in final_discounts if value is not None]
    report: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "dmc_deterministic_evaluation",
        "task": metadata.task,
        "actor_type": metadata.actor_type,
        "actor_checkpoint_kind": metadata.kind,
        "actor_checkpoint": str(metadata.path.resolve()),
        "actor_config": metadata.actor_config.to_dict(),
        "value_expansion": metadata.payload.get("value_expansion"),
        "observation_normalizer_count": ppo_observation_normalizer_count,
        "normalization_contract": metadata.payload.get(
            "normalization_contract"
        ),
        "koopman_checkpoint": (
            None if resolved_koopman is None else str(resolved_koopman.resolve())
        ),
        "koopman_sha256": metadata.payload.get("koopman_sha256"),
        "koopman_lineage": koopman_lineage,
        "koopman_dataset_sha256": metadata.payload.get(
            "koopman_dataset_sha256"
        ),
        "koopman_config_fingerprint": metadata.payload.get(
            "koopman_config_fingerprint"
        ),
        "training_seed": metadata.training_seed,
        **metadata.authorization,
        "resolved_execution_spec": metadata.payload.get(
            "resolved_execution_spec"
        ),
        "evaluation_seeds": metadata.payload.get("evaluation_seeds"),
        "evaluation_episodes_per_seed": metadata.payload.get(
            "evaluation_episodes_per_seed"
        ),
        "evaluation_reference_episodes_per_seed": (
            reference_episodes_per_seed
        ),
        "diagnostic_every_steps": metadata.payload.get(
            "diagnostic_every_steps"
        ),
        "eval_seed": int(eval_seed),
        "episode_seeds": resets,
        "deterministic": True,
        "episodes": int(episodes),
        "evaluation_runner": "synchronous_episode_batch_v1",
        "evaluation_num_envs": int(episodes),
        "environment_step_limit": int(runtime_protocol["step_limit"]),
        "protocol": metadata.protocol,
        "runtime_protocol": runtime_protocol,
        "return_mean_across_episodes": float(np.mean(returns)),
        "return_std_across_episodes": float(np.std(returns, ddof=0)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "episode_returns": returns,
        "acme_reference_episode_returns": reference_returns,
        "acme_reference_episode_count": len(reference_returns),
        "acme_reference_return_mean": float(np.mean(reference_returns)),
        "acme_reference_action_component_count": reference_action_total,
        "acme_reference_requested_action_bound_count": (
            reference_requested_bound_total
        ),
        "acme_reference_requested_action_bound_fraction": float(
            reference_requested_bound_total / max(reference_action_total, 1)
        ),
        "acme_reference_applied_action_bound_count": (
            reference_applied_bound_total
        ),
        "acme_reference_applied_action_bound_fraction": float(
            reference_applied_bound_total / max(reference_action_total, 1)
        ),
        "acme_reference_action_clipped_count": reference_clipped_action_total,
        "acme_reference_action_clipped_fraction": float(
            reference_clipped_action_total / max(reference_action_total, 1)
        ),
        "robustness_episode_returns": returns,
        "robustness_episode_count": len(returns),
        "robustness_return_mean": float(np.mean(returns)),
        "robustness_return_population_std": float(
            np.std(returns, ddof=0)
        ),
        "robustness_action_component_count": int(action_total),
        "robustness_requested_action_bound_count": int(requested_bound_total),
        "robustness_requested_action_bound_fraction": float(
            requested_bound_total / max(action_total, 1)
        ),
        "robustness_applied_action_bound_count": int(applied_bound_total),
        "robustness_applied_action_bound_fraction": float(
            applied_bound_total / max(action_total, 1)
        ),
        "robustness_action_clipped_count": int(clipped_action_total),
        "robustness_action_clipped_fraction": float(
            clipped_action_total / max(action_total, 1)
        ),
        "environment_discounted_return_mean": float(
            np.mean(environment_discounted_returns)
        ),
        "episode_length_mean": float(np.mean(lengths)),
        "episode_lengths": lengths,
        "episode_action_component_counts": action_components_by_episode.tolist(),
        "episode_requested_action_bound_counts": (
            requested_bound_by_episode.tolist()
        ),
        "episode_applied_action_bound_counts": applied_bound_by_episode.tolist(),
        "episode_action_clipped_counts": clipped_action_by_episode.tolist(),
        "mean_step_reward": float(sum(returns) / max(total_steps, 1)),
        "terminated_episodes": int(terminated_episodes),
        "truncated_episodes": int(truncated_episodes),
        "mean_final_discount": (
            None
            if not numeric_final_discounts
            else float(np.mean(numeric_final_discounts))
        ),
        "requested_action_bound_fraction": float(
            requested_bound_total / max(action_total, 1)
        ),
        "applied_action_bound_fraction": float(
            applied_bound_total / max(action_total, 1)
        ),
        "action_clipped_fraction": float(clipped_action_total / max(action_total, 1)),
        "mean_reward_components": {
            key: float(value / max(total_steps, 1))
            for key, value in component_sums.items()
        },
    }
    # Narrow compatibility aliases for existing analysis notebooks.  Their
    # unambiguous replacements above should be used by all new code.
    report["mean_return"] = report["return_mean_across_episodes"]
    report["std_return"] = report["return_std_across_episodes"]
    report["mean_episode_length"] = report["episode_length_mean"]
    report["action_bound_fraction"] = report["applied_action_bound_fraction"]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--koopman",
        type=Path,
        default=None,
        help="relocation override; identity is still checked against checkpoint metadata",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES_PER_EVAL_SEED)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        args.actor_checkpoint,
        koopman_path=args.koopman,
        episodes=args.episodes,
        eval_seed=args.eval_seed,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        _write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
