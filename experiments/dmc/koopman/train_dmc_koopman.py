"""Train K-step Koopman dynamics and a transition reward model on DMC data.

Generic port of ``train_hopper_hop_koopman.py``: state/action dims and the
diagnostic report groups come from the task registry, and the checkpoint /
resume / early-stopping logic is identical:

  * ``best.pt``  — best joint validation, PandaReach-compatible payload
    (``model_state`` / ``architecture`` / ``state_kind`` / ``normalizer`` / ...)
    with ``state_kind = <task name>`` plus the reward architecture/state used
    by MPC value expansion.
  * ``latest.pt`` — periodic, full resume state (model + optimizer + rng +
    epoch + history), atomic writes.
  * ``report.json`` — one-step + K-step rollout metrics per report group.

Usage:

    python -m experiments.dmc.koopman.train_dmc_koopman \
        --config experiments/dmc/configs/hopper_hop.yaml \
        --profile development \
        --preflight-file runs/dmc/preflight/hopper_hop_development_v2.json \
        --approval-file runs/dmc/approvals/hopper_hop_development_v2.json \
        --dataset runs/dmc/data/hopper_hop/development/hopper_hop_koopman.npz \
        --output-dir runs/dmc/koopman/hopper_hop/development
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler

from antmaze_ac.koopman.losses import koopman_loss
from antmaze_ac.koopman.checkpoint import FORMAT_VERSION
from antmaze_ac.koopman.model import DeepKoopman
from experiments.dmc.approval import (
    APPROVAL_KIND,
    validate_training_approval,
    validate_training_preflight,
)
from experiments.dmc.collect.build_dmc_datasets import DATASET_SCHEMA_VERSION
from experiments.dmc.config import (
    ExperimentConfig,
    PROFILE_NAMES,
    load_experiment_config,
    resolve_execution_spec,
    resolve_koopman_config,
)
from experiments.dmc.protocol import (
    canonical_json,
    protocol_fingerprint_from_json,
)
from experiments.dmc.reward_model import (
    TransitionRewardModel,
    transition_reward_input_contract,
)
from experiments.dmc.tasks.registry import get_task_spec


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 500
    batch_size: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    lift_dim: int = 48
    hidden_dims: tuple[int, ...] = (256, 256)
    reward_hidden_dims: tuple[int, ...] = (256, 256)
    reward_loss_weight: float = 1.0
    seed: int = 43
    k_step: int = 20
    activation: str = "silu"
    rollout_discount: float = 0.99
    linear_weight: float = 10.0
    rollout_weight: float = 1.0
    stability_weight: float = 0.1
    latent_std_weight: float = 0.1
    identity_weight: float = 1e-4
    controllability_svd_weight: float = 0.0
    augmentation_weight: float = 0.0
    reconstruction_weight: float = 1.0
    svd_min_singular_value: float = 0.0
    spectral_radius_limit: float = 0.95
    # The validated Hopper penalty used 0.95 at a 0.04 s control step.  Convert
    # that decay to each DMC task's native dt instead of imposing 0.95 per step
    # at both 25 Hz and 100 Hz.
    stability_reference_dt: float = 0.04
    target_latent_std: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_amsgrad: bool = False
    checkpoint_every: int = 25
    patience: int = 40
    max_windows: int = 1_000_000


KOOPMAN_MANIFEST_KIND = "dmc_koopman_training_free_run_manifest"
SUPPORTED_COLLECTION_SCHEMA_VERSIONS = frozenset({4})
_LOWER_HEX = frozenset("0123456789abcdef")


def _train_config_mapping(config: TrainConfig) -> dict[str, Any]:
    """Return a JSON-shaped TrainConfig mapping for exact YAML comparison."""

    mapping = asdict(config)
    mapping["hidden_dims"] = list(mapping["hidden_dims"])
    mapping["reward_hidden_dims"] = list(mapping["reward_hidden_dims"])
    return mapping


def _reward_architecture_mapping(
    *, state_dim: int, action_dim: int, config: TrainConfig
) -> dict[str, Any]:
    return {
        "architecture": TransitionRewardModel.ARCHITECTURE,
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "hidden_dims": list(config.reward_hidden_dims),
        "activation": str(config.activation).lower(),
    }


def _json_identity(value: dict[str, Any]) -> str:
    """Serialize a config with JSON's type distinctions preserved."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def train_config_from_experiment(experiment: ExperimentConfig) -> TrainConfig:
    """Construct the only TrainConfig authorized by an experiment YAML."""

    resolved = resolve_koopman_config(experiment)
    constructor = dict(resolved)
    constructor["hidden_dims"] = tuple(constructor["hidden_dims"])
    constructor["reward_hidden_dims"] = tuple(constructor["reward_hidden_dims"])
    return TrainConfig(**constructor)


def _coerce_experiment_config(
    value: ExperimentConfig | Path | str | None,
) -> ExperimentConfig:
    if isinstance(value, ExperimentConfig):
        return value
    if value is None:
        raise PermissionError(
            "Approval-bound Koopman training requires an experiment config"
        )
    return load_experiment_config(value)


def _validate_yaml_binding(
    experiment: ExperimentConfig,
    profile: str | None,
    task_name: str,
    config: TrainConfig,
) -> tuple[str, dict[str, Any]]:
    if profile is None:
        raise PermissionError("Approval-bound Koopman training requires a profile")
    execution_spec = resolve_execution_spec(experiment, profile)
    if experiment.task != task_name:
        raise ValueError(
            f"Requested task {task_name!r} does not match YAML task "
            f"{experiment.task!r}"
        )
    expected = resolve_koopman_config(experiment)
    try:
        actual_identity = _json_identity(_train_config_mapping(config))
        expected_identity = _json_identity(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError("TrainConfig is not a finite JSON-shaped mapping") from exc
    if actual_identity != expected_identity:
        raise ValueError(
            "TrainConfig does not exactly match the resolved approval-bound YAML"
        )
    return profile, execution_spec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a checkpoint only after serialization is complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _episode_mask(episode_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    return np.isin(episode_ids, selected_ids)


def _integer_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D integer array")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a 1D integer array")
    converted = array.astype(np.int64, copy=False)
    if not np.array_equal(array, converted):
        raise ValueError(f"{name} contains an ID outside the int64 range")
    return converted


def _archive_scalar(data: dict[str, np.ndarray], name: str) -> Any:
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError(f"Dataset field {name!r} must be scalar")
    return value.item()


def _lower_sha256(value: Any, *, name: str, prefixed: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Dataset field {name!r} must be a SHA-256 string")
    digest = value.removeprefix("sha256:") if prefixed else value
    if prefixed and not value.startswith("sha256:"):
        raise ValueError(f"Dataset field {name!r} must start with 'sha256:'")
    if len(digest) != 64 or any(character not in _LOWER_HEX for character in digest):
        raise ValueError(
            f"Dataset field {name!r} must contain 64 lowercase hex characters"
        )
    return value


def _validate_dataset_provenance(
    data: dict[str, np.ndarray], transition_count: int
) -> dict[str, Any]:
    """Validate the mandatory on-policy authorization lineage in dataset v4."""

    provenance = {
        name: _archive_scalar(data, name)
        for name in (
            "data_source",
            "actor_type",
            "training_approved",
            "config_fingerprint",
            "approval_profile",
            "approval_file_sha256",
            "preflight_report_sha256",
            "authorization_kind",
        )
    }
    if provenance["data_source"] != "ppo_training_stages":
        raise ValueError("Primary Koopman data_source must be 'ppo_training_stages'")
    if provenance["actor_type"] != "PPO":
        raise ValueError("Primary Koopman datasets must be collected by PPO")
    if provenance["training_approved"] is not True:
        raise ValueError("Koopman dataset collection was not formally approved")
    _lower_sha256(
        provenance["config_fingerprint"],
        name="config_fingerprint",
        prefixed=True,
    )
    if provenance["approval_profile"] not in PROFILE_NAMES:
        raise ValueError("Dataset approval_profile is invalid")
    _lower_sha256(
        provenance["approval_file_sha256"], name="approval_file_sha256"
    )
    _lower_sha256(
        provenance["preflight_report_sha256"], name="preflight_report_sha256"
    )
    if provenance["authorization_kind"] != APPROVAL_KIND:
        raise ValueError("Dataset authorization_kind is not a training approval")

    source_indices = _integer_vector(
        data["source_train_seed_indices"], name="source_train_seed_indices"
    )
    source_seeds = _integer_vector(
        data["source_training_seeds"], name="source_training_seeds"
    )
    transition_indices = _integer_vector(
        data["source_seed_index"], name="source_seed_index"
    )
    if source_indices.size == 0 or source_indices.size != source_seeds.size:
        raise ValueError(
            "Dataset source train-seed indices/seeds must be non-empty and aligned"
        )
    if np.unique(source_indices).size != source_indices.size or np.any(
        source_indices < 0
    ):
        raise ValueError("source_train_seed_indices must be unique and non-negative")
    if np.unique(source_seeds).size != source_seeds.size or np.any(source_seeds < 0):
        raise ValueError("source_training_seeds must be unique and non-negative")
    if len(transition_indices) != transition_count:
        raise ValueError("source_seed_index transition count does not match state")
    if not np.array_equal(np.unique(transition_indices), np.sort(source_indices)):
        raise ValueError(
            "Transition source_seed_index values do not match source provenance"
        )
    provenance["source_train_seed_indices"] = source_indices.tolist()
    provenance["source_training_seeds"] = source_seeds.tolist()
    return provenance


def _validate_episode_splits(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    episode_ids = _integer_vector(data["episode_id"], name="episode_id")
    if episode_ids.size == 0:
        raise ValueError("Dataset contains no transitions")
    existing_ids = np.unique(episode_ids)
    split_ids: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        field = f"{split}_episode_ids"
        ids = _integer_vector(data[field], name=field)
        if ids.size == 0:
            raise ValueError(f"{field} must not be empty")
        if np.unique(ids).size != ids.size:
            raise ValueError(f"{field} must contain unique episode IDs")
        unknown = np.setdiff1d(ids, existing_ids, assume_unique=True)
        if unknown.size:
            raise ValueError(
                f"{field} contains IDs absent from episode_id: {unknown.tolist()}"
            )
        split_ids[split] = ids

    all_split_ids = np.concatenate(list(split_ids.values()))
    if np.unique(all_split_ids).size != all_split_ids.size:
        raise ValueError("Episode split ID arrays must be pairwise disjoint")
    split_union = np.unique(all_split_ids)
    if not np.array_equal(split_union, existing_ids):
        omitted = np.setdiff1d(existing_ids, split_union, assume_unique=True)
        raise ValueError(
            "Episode split ID union must exactly equal dataset episode IDs; "
            f"omitted={omitted.tolist()}"
        )

    masks = {
        split: _episode_mask(episode_ids, ids) for split, ids in split_ids.items()
    }
    if np.any(
        masks["train"].astype(np.int8)
        + masks["validation"].astype(np.int8)
        + masks["test"].astype(np.int8)
        != 1
    ):
        # This should be implied by the ID checks, but retain a transition-level
        # assertion so later representation changes remain fail-closed.
        raise ValueError("Episode splits overlap or omit transitions")
    return episode_ids, masks


def load_dataset(
    path: Path, task_name: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    spec = get_task_spec(task_name)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {
        "state",
        "requested_action",
        "action",
        "next_state",
        "reward",
        "discount",
        "done",
        "terminated",
        "truncated",
        "collector_truncated",
        "episode_id",
        "step_index",
        "update",
        "global_step",
        "reset_seed",
        "train_episode_ids",
        "validation_episode_ids",
        "test_episode_ids",
        "dataset_schema_version",
        "collection_schema_version",
        "protocol_json",
        "environment_protocol_json",
        "protocol_fingerprint",
        "state_kind",
        "state_dim",
        "action_dim",
        "data_source",
        "actor_type",
        "training_approved",
        "config_fingerprint",
        "approval_profile",
        "approval_file_sha256",
        "preflight_report_sha256",
        "authorization_kind",
        "source_train_seed_indices",
        "source_training_seeds",
        "source_seed_index",
    }
    missing = required - data.keys()
    if missing:
        raise KeyError(f"Dataset is missing fields: {sorted(missing)}")
    if data["state"].shape != data["next_state"].shape:
        raise ValueError("state and next_state shapes differ")
    if (
        data["state"].shape[1:] != (spec.obs_dim,)
        or data["action"].shape[1:] != (spec.action_dim,)
    ):
        raise ValueError(
            f"Expected state [N,{spec.obs_dim}] and action "
            f"[N,{spec.action_dim}] for {task_name}"
        )
    if not all(
        np.isfinite(data[name]).all()
        for name in (
            "state",
            "requested_action",
            "action",
            "next_state",
            "reward",
            "discount",
        )
    ):
        raise FloatingPointError("Dataset contains NaN or Inf")
    if data["requested_action"].shape != data["action"].shape:
        raise ValueError("requested_action and action shapes differ")
    if data["reward"].shape != (len(data["state"]),):
        raise ValueError("reward must have shape [N]")
    if np.any(data["reward"] < 0.0) or np.any(data["reward"] > 1.0):
        raise ValueError("DMC reward targets must be in [0, 1]")
    if "state_kind" in data and str(data["state_kind"].item()) != task_name:
        raise ValueError(
            f"Dataset state_kind {data['state_kind'].item()!r} does not match "
            f"task {task_name!r}"
        )
    dataset_schema_version = int(data["dataset_schema_version"].item())
    if dataset_schema_version != DATASET_SCHEMA_VERSION or dataset_schema_version != 4:
        raise ValueError(
            "The primary DMC Koopman trainer requires dataset schema version 4"
        )
    collection_schema_version = int(data["collection_schema_version"].item())
    if collection_schema_version not in SUPPORTED_COLLECTION_SCHEMA_VERSIONS:
        raise ValueError(
            "The primary DMC Koopman trainer requires collection schema version 4"
        )
    if int(data["state_dim"].item()) != spec.obs_dim or int(
        data["action_dim"].item()
    ) != spec.action_dim:
        raise ValueError("Dataset scalar dimensions do not match the task registry")
    environment_protocol_json = str(data["environment_protocol_json"].item())
    try:
        expected_protocol_fingerprint = protocol_fingerprint_from_json(
            environment_protocol_json
        )
        environment_protocol = json.loads(environment_protocol_json)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Dataset has invalid environment protocol metadata") from exc
    if str(data["protocol_fingerprint"].item()) != expected_protocol_fingerprint:
        raise ValueError("Dataset protocol fingerprint is invalid")
    if environment_protocol.get("task") != task_name:
        raise ValueError("Dataset environment protocol belongs to a different task")
    collection_protocol = json.loads(str(data["protocol_json"].item()))
    if canonical_json(collection_protocol) != str(data["protocol_json"].item()):
        raise ValueError("Dataset collection protocol JSON is not canonical")
    for key in ("task", "obs_dim", "action_dim"):
        if collection_protocol.get(key) != environment_protocol.get(key):
            raise ValueError(
                f"Dataset collection/environment protocol {key} mismatch"
            )

    transition_count = len(data["state"])
    if len(data["action"]) != transition_count:
        raise ValueError("action transition count does not match state")
    _validate_dataset_provenance(data, transition_count)
    episode_ids, masks = _validate_episode_splits(data)
    if len(episode_ids) != transition_count:
        raise ValueError("episode_id transition count does not match state")
    step_index = _integer_vector(data["step_index"], name="step_index")
    if len(step_index) != transition_count:
        raise ValueError("step_index transition count does not match state")
    _validate_complete_dataset_episodes(data, episode_ids)
    return data, masks


def _contiguous_episode_boundaries(
    episode_ids: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """Index every contiguous episode block with one linear pass over IDs."""

    ids = _integer_vector(episode_ids, name="episode_id")
    if ids.size == 0:
        return {}
    changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), changes))
    stops = np.concatenate((changes, np.asarray([len(ids)], dtype=np.int64)))
    block_ids = ids[starts]
    if np.unique(block_ids).size != block_ids.size:
        raise ValueError("Each episode must occupy exactly one contiguous block")
    return {
        int(episode_id): (int(start), int(stop))
        for episode_id, start, stop in zip(block_ids, starts, stops, strict=True)
    }


def _boolean_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be a 1D boolean array")
    return array.astype(np.bool_, copy=False)


def _validate_complete_dataset_episodes(
    data: dict[str, np.ndarray], episode_ids: np.ndarray
) -> None:
    """Recheck the v4 complete-episode contract at the trainer boundary."""

    transition_count = len(episode_ids)
    for name in (
        "requested_action",
        "reward",
        "discount",
        "done",
        "terminated",
        "truncated",
        "collector_truncated",
        "update",
        "global_step",
        "reset_seed",
    ):
        if len(data[name]) != transition_count:
            raise ValueError(f"Dataset field {name!r} has the wrong length")
    done = _boolean_vector(data["done"], name="done")
    terminated = _boolean_vector(data["terminated"], name="terminated")
    truncated = _boolean_vector(data["truncated"], name="truncated")
    collector_truncated = _boolean_vector(
        data["collector_truncated"], name="collector_truncated"
    )
    if np.any(np.logical_and(terminated, truncated)):
        raise ValueError("A transition cannot be both terminated and truncated")
    if not np.array_equal(done, np.logical_or(terminated, truncated)):
        raise ValueError("Dataset done flags do not match terminal causes")
    if np.any(collector_truncated):
        raise ValueError("Primary PPO datasets must not contain cut episodes")
    discount = np.asarray(data["discount"])
    if discount.ndim != 1 or np.any(discount < 0.0) or np.any(discount > 1.0):
        raise ValueError("discount must be a 1D array in [0, 1]")

    step_index = _integer_vector(data["step_index"], name="step_index")
    updates = _integer_vector(data["update"], name="update")
    global_steps = _integer_vector(data["global_step"], name="global_step")
    reset_seeds = _integer_vector(data["reset_seed"], name="reset_seed")
    source_indices = _integer_vector(
        data["source_seed_index"], name="source_seed_index"
    )
    boundaries = _contiguous_episode_boundaries(episode_ids)
    seen_source_steps: set[tuple[int, int]] = set()
    for episode_id, (start, stop) in boundaries.items():
        length = stop - start
        if not np.array_equal(
            step_index[start:stop], np.arange(length, dtype=np.int64)
        ):
            raise ValueError(f"Episode {episode_id} has non-consecutive steps")
        if not done[stop - 1] or np.any(done[start : stop - 1]):
            raise ValueError(f"Episode {episode_id} is incomplete or ends early")
        if np.any(np.diff(updates[start:stop]) < 0) or updates[start] < 1:
            raise ValueError(f"Episode {episode_id} has invalid PPO update tags")
        if length > 1 and np.any(np.diff(global_steps[start:stop]) <= 0):
            raise ValueError(f"Episode {episode_id} has non-increasing global steps")
        if np.unique(reset_seeds[start:stop]).size != 1:
            raise ValueError(f"Episode {episode_id} mixes reset seeds")
        if np.unique(source_indices[start:stop]).size != 1:
            raise ValueError(f"Episode {episode_id} mixes source training seeds")
        if length > 1:
            chain_error = float(
                np.max(
                    np.abs(
                        data["next_state"][start : stop - 1]
                        - data["state"][start + 1 : stop]
                    )
                )
            )
            if chain_error > 2e-5:
                raise ValueError(
                    f"Episode {episode_id} transition chain mismatch "
                    f"{chain_error:.3e}"
                )
        for global_step in global_steps[start:stop]:
            key = (int(source_indices[start]), int(global_step))
            if key in seen_source_steps:
                raise ValueError("Dataset repeats a source/global_step pair")
            seen_source_steps.add(key)


def _validate_stage_coverage(
    data: dict[str, np.ndarray], execution_spec: dict[str, Any]
) -> dict[str, Any]:
    """Require complete episodes from every early/mid/late PPO third."""

    total_updates = int(execution_spec["data"]["collection_total_updates"])
    if total_updates < 3:
        raise ValueError("PPO data collection must cover at least three updates")
    stage_ends = (total_updates // 3, 2 * total_updates // 3, total_updates)
    stage_names = ("early", "mid", "late")
    source_indices = _integer_vector(
        data["source_seed_index"], name="source_seed_index"
    )
    updates = _integer_vector(data["update"], name="update")
    episode_ids = _integer_vector(data["episode_id"], name="episode_id")
    counts = {
        int(source): {name: 0 for name in stage_names}
        for source in np.unique(source_indices)
    }
    for _episode_id, (start, stop) in _contiguous_episode_boundaries(
        episode_ids
    ).items():
        source = int(source_indices[start])
        completion_update = int(updates[stop - 1])
        if completion_update > total_updates:
            raise ValueError("Dataset contains an update beyond the approved PPO run")
        stage_index = next(
            index
            for index, end in enumerate(stage_ends)
            if completion_update <= end
        )
        counts[source][stage_names[stage_index]] += 1
    missing = {
        source: [name for name, count in stages.items() if count == 0]
        for source, stages in counts.items()
        if any(count == 0 for count in stages.values())
    }
    if missing:
        raise ValueError(
            "Dataset does not cover complete early/mid/late episodes for every "
            f"training seed: {missing}"
        )
    return {
        "strategy": "complete_episode_completion_update_thirds_v1",
        "total_updates": total_updates,
        "stage_ends": dict(zip(stage_names, stage_ends, strict=True)),
        "episode_counts_by_train_seed_index": {
            str(source): stages for source, stages in counts.items()
        },
    }


def _validated_window_counts(
    data: dict[str, np.ndarray], config: TrainConfig
) -> dict[str, int]:
    """Count episode-safe windows without allocating overlapping tensors."""

    boundaries = _contiguous_episode_boundaries(data["episode_id"])
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        episode_ids = _integer_vector(
            data[f"{split}_episode_ids"], name=f"{split}_episode_ids"
        )
        count = 0
        for episode_id in episode_ids:
            start, stop = boundaries[int(episode_id)]
            count += max(0, stop - start - config.k_step + 1)
        if split == "train":
            count = min(count, config.max_windows)
        if count < 1:
            raise ValueError(
                f"Dataset {split} split has no complete K={config.k_step} window"
            )
        counts[split] = int(count)
    return counts


def _positive_integer(value: int | None, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class KoopmanWindowDataset(Dataset):
    """Lazy episode-safe K-step windows over the merged transition arrays.

    Earlier code materialized every overlapping window.  A million Humanoid
    windows then occupied several GB before the first optimizer step.  This
    dataset stores only int64 start offsets and constructs one normalized
    window when the DataLoader asks for it.
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
        selected_episode_ids: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        *,
        k_step: int,
        obs_dim: int,
        action_dim: int,
        max_windows: int | None = None,
        seed: int = 0,
    ) -> None:
        k_step = _positive_integer(k_step, name="k_step")
        obs_dim = _positive_integer(obs_dim, name="obs_dim")
        action_dim = _positive_integer(action_dim, name="action_dim")
        if max_windows is not None:
            max_windows = _positive_integer(max_windows, name="max_windows")
        self.data = data
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.k_step = k_step
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        if self.center.shape != (obs_dim,):
            raise ValueError(f"center must have shape ({obs_dim},)")
        if self.scale.shape != (obs_dim,):
            raise ValueError(f"scale must have shape ({obs_dim},)")
        if not np.isfinite(self.center).all():
            raise ValueError("center must contain only finite values")
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("scale must contain only finite positive values")

        state = np.asarray(data["state"])
        next_state = np.asarray(data["next_state"])
        action = np.asarray(data["action"])
        if state.ndim != 2 or state.shape[1:] != (obs_dim,):
            raise ValueError(f"state must have shape [N,{obs_dim}]")
        if next_state.shape != state.shape:
            raise ValueError("next_state must have the same shape as state")
        if action.shape != (len(state), action_dim):
            raise ValueError(f"action must have shape [{len(state)},{action_dim}]")

        episode_id = _integer_vector(data["episode_id"], name="episode_id")
        if len(episode_id) != len(state):
            raise ValueError("episode_id transition count does not match state")
        selected = _integer_vector(
            selected_episode_ids, name="selected_episode_ids"
        )
        if np.unique(selected).size != selected.size:
            raise ValueError("selected_episode_ids must contain unique IDs")
        boundaries = _contiguous_episode_boundaries(episode_id)
        missing = [int(value) for value in selected if int(value) not in boundaries]
        if missing:
            raise ValueError(
                f"selected_episode_ids are absent from episode_id: {missing}"
            )

        step_index = data.get("step_index")
        if step_index is not None:
            step_index = _integer_vector(step_index, name="step_index")
            if len(step_index) != len(state):
                raise ValueError("step_index transition count does not match state")
        starts: list[np.ndarray] = []
        for selected_episode in selected:
            block_start, block_stop = boundaries[int(selected_episode)]
            length = block_stop - block_start
            if length < self.k_step:
                continue
            if step_index is not None:
                episode_steps = step_index[block_start:block_stop]
                if not np.array_equal(
                    episode_steps, np.arange(length, dtype=np.int64)
                ):
                    raise ValueError(
                        f"Episode {selected_episode} has non-consecutive steps"
                    )
            if length > 1:
                chain_error = np.max(
                    np.abs(
                        next_state[block_start : block_stop - 1]
                        - state[block_start + 1 : block_stop]
                    )
                )
                if chain_error > 2e-5:
                    raise ValueError(
                        f"Episode {selected_episode} transition chain mismatch "
                        f"{chain_error:.3e}"
                    )
            starts.append(
                np.arange(
                    block_start,
                    block_stop - self.k_step + 1,
                    dtype=np.int64,
                )
            )
        if not starts:
            raise ValueError("No episode is long enough for a K-step window")
        start_array = np.concatenate(starts)
        if max_windows is not None and len(start_array) > max_windows:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(start_array), size=max_windows, replace=False)
            start_array = start_array[chosen]
        self.starts = start_array

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.get_batch(np.asarray([index], dtype=np.int64))
        return tuple(value[0] for value in batch)

    def get_batch(
        self, indices: np.ndarray | Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct a complete minibatch with vectorized NumPy gathers.

        The former DataLoader called ``__getitem__`` once per window.  At
        1.5M K=50 windows that means 1.5M Python concatenations every epoch.
        Gathering all offsets for a minibatch at once keeps the exact window
        contract while moving that work into NumPy's compiled loops.
        """

        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("batch indices must be a 1D integer array")
        if len(indices) == 0:
            raise ValueError("batch indices must not be empty")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Koopman window batch index is out of range")
        starts = self.starts[indices.astype(np.int64, copy=False)]
        offsets = np.arange(self.k_step, dtype=np.int64)
        positions = starts[:, None] + offsets[None, :]
        state_windows = np.asarray(self.data["state"])[positions]
        final_next_states = np.asarray(self.data["next_state"])[
            starts + self.k_step - 1
        ][:, None, :]
        physical = np.concatenate((state_windows, final_next_states), axis=1)
        states = np.ascontiguousarray(
            ((physical - self.center) / self.scale).astype(np.float32, copy=False)
        )
        actions = np.ascontiguousarray(
            np.asarray(self.data["action"])[positions].astype(np.float32, copy=False)
        )
        # Each overlapping K-step window supervises exactly its first
        # transition.  Returning all K rewards would make middle transitions
        # appear up to K times and silently reweight the reward objective.
        reward = np.ascontiguousarray(
            np.asarray(self.data["reward"])[starts].astype(np.float32, copy=False)
        )
        reward_next_state = np.ascontiguousarray(
            (
                (np.asarray(self.data["next_state"])[starts] - self.center)
                / self.scale
            ).astype(np.float32, copy=False)
        )
        batch_size = len(starts)
        if states.shape != (batch_size, self.k_step + 1, self.obs_dim):
            raise RuntimeError("Invalid Koopman state-window shape")
        if actions.shape != (batch_size, self.k_step, self.action_dim):
            raise RuntimeError("Invalid Koopman action-window shape")
        if reward.shape != (batch_size,) or not np.isfinite(reward).all():
            raise RuntimeError("Invalid reward target")
        if reward_next_state.shape != (batch_size, self.obs_dim) or not np.isfinite(
            reward_next_state
        ).all():
            raise RuntimeError("Invalid reward next-state target")
        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(reward),
            torch.from_numpy(reward_next_state),
        )


def fit_normalizer(
    state: np.ndarray,
    next_state: np.ndarray,
    *,
    chunk_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-dimension moments with bounded-memory parallel Welford updates."""

    chunk_size = _positive_integer(chunk_size, name="chunk_size")
    state = np.asarray(state)
    next_state = np.asarray(next_state)
    if state.ndim != 2 or next_state.ndim != 2:
        raise ValueError("state and next_state must be 2D arrays")
    if state.shape[1:] != next_state.shape[1:]:
        raise ValueError("state and next_state feature dimensions differ")
    if len(state) == 0 or len(next_state) == 0:
        raise ValueError("Normalizer inputs must not be empty")

    dimensions = state.shape[1]
    count = 0
    center = np.zeros(dimensions, dtype=np.float64)
    squared_deviation = np.zeros(dimensions, dtype=np.float64)
    for source in (state, next_state):
        for start in range(0, len(source), chunk_size):
            batch = np.asarray(source[start : start + chunk_size], dtype=np.float64)
            if not np.isfinite(batch).all():
                raise FloatingPointError("Normalizer inputs contain NaN or Inf")
            batch_count = len(batch)
            batch_center = batch.mean(axis=0)
            batch_delta = batch - batch_center
            batch_squared_deviation = np.einsum(
                "ij,ij->j", batch_delta, batch_delta
            )
            total = count + batch_count
            delta = batch_center - center
            squared_deviation += (
                batch_squared_deviation
                + np.square(delta) * count * batch_count / total
            )
            center += delta * batch_count / total
            count = total

    scale = np.maximum(np.sqrt(squared_deviation / count), 1e-4)
    center32 = center.astype(np.float32)
    scale32 = scale.astype(np.float32)
    if not np.isfinite(center32).all() or not np.isfinite(scale32).all():
        raise FloatingPointError("Normalizer moments are not finite float32 values")
    return center32, scale32


def build_windows(
    data: dict[str, np.ndarray],
    selected_episode_ids: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    k_step: int,
    obs_dim: int,
    action_dim: int,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [batch, K+1, obs_dim] state windows and [batch, K, action_dim] action windows."""
    dataset = KoopmanWindowDataset(
        data,
        selected_episode_ids,
        center,
        scale,
        k_step=k_step,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_windows=max_windows,
    )
    windows = [dataset[index] for index in range(len(dataset))]
    states = torch.stack([window[0] for window in windows]).numpy()
    actions = torch.stack([window[1] for window in windows]).numpy()
    return states, actions


class EpochRandomSampler(Sampler[int]):
    """Permutation sampler whose order is a pure function of seed and epoch."""

    def __init__(self, dataset: Dataset, *, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if (
            isinstance(epoch, (bool, np.bool_))
            or not isinstance(epoch, (int, np.integer))
            or int(epoch) < 0
        ):
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        epoch_seed = (self.seed + self.epoch) % (2**63 - 1)
        generator.manual_seed(epoch_seed)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


class VectorizedKoopmanLoader:
    """Deterministic minibatch loader with bounded threaded prefetch.

    Threads share the immutable dataset arrays, unlike process DataLoader
    workers which would replicate a multi-million-transition archive.  The
    worker count is execution-only and may be changed when resuming.
    """

    def __init__(
        self,
        dataset: KoopmanWindowDataset,
        *,
        batch_size: int,
        sampler: Sampler[int],
        prefetch_workers: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = min(batch_size, len(dataset))
        self.sampler = sampler
        self.prefetch_workers = max(0, int(prefetch_workers))

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def _index_batches(self):
        batch: list[int] = []
        for index in self.sampler:
            batch.append(int(index))
            if len(batch) == self.batch_size:
                yield np.asarray(batch, dtype=np.int64)
                batch = []
        if batch:
            yield np.asarray(batch, dtype=np.int64)

    def __iter__(self):
        batches = iter(self._index_batches())
        if self.prefetch_workers == 0:
            for indices in batches:
                yield self.dataset.get_batch(indices)
            return

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.prefetch_workers,
            thread_name_prefix="koopman-batch",
        )
        pending: list[concurrent.futures.Future] = []
        try:
            for _ in range(self.prefetch_workers * 2):
                try:
                    indices = next(batches)
                except StopIteration:
                    break
                pending.append(executor.submit(self.dataset.get_batch, indices))
            while pending:
                future = pending.pop(0)
                yield future.result()
                try:
                    indices = next(batches)
                except StopIteration:
                    continue
                pending.append(executor.submit(self.dataset.get_batch, indices))
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    prefetch_workers: int | None = None,
) -> Any:
    batch_size = _positive_integer(batch_size, name="batch_size")
    if len(dataset) < 1:
        raise ValueError("Cannot create a DataLoader for an empty dataset")
    sampler = (
        EpochRandomSampler(dataset, seed=seed)
        if shuffle
        else SequentialSampler(dataset)
    )
    if isinstance(dataset, KoopmanWindowDataset):
        if prefetch_workers is None:
            raw_workers = os.environ.get("DMC_KOOPMAN_LOADER_WORKERS", "4")
            try:
                prefetch_workers = int(raw_workers)
            except ValueError as exc:
                raise ValueError(
                    "DMC_KOOPMAN_LOADER_WORKERS must be a non-negative integer"
                ) from exc
        if prefetch_workers < 0:
            raise ValueError("prefetch_workers must be non-negative")
        return VectorizedKoopmanLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            prefetch_workers=prefetch_workers,
        )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        sampler=sampler,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def _set_loader_epoch(loader: Any, epoch: int) -> None:
    sampler = loader.sampler
    if isinstance(sampler, EpochRandomSampler):
        sampler.set_epoch(epoch)


def _dynamics_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Read dynamics tensors from both legacy 2-item and reward 3-item loaders."""

    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Koopman loader batch must contain states and actions")
    states, actions = batch[:2]
    if not isinstance(states, torch.Tensor) or not isinstance(actions, torch.Tensor):
        raise TypeError("Koopman loader states/actions must be tensors")
    return states, actions


@torch.no_grad()
def rollout_normalized_mse(
    model: DeepKoopman, batches: DataLoader, device: torch.device
) -> float:
    squared_error = 0.0
    elements = 0
    model.eval()
    for batch in batches:
        states, actions = _dynamics_batch(batch)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        prediction, _ = model.rollout(states[:, 0], actions)
        target = states[:, 1:]
        squared_error += float((prediction - target).square().sum())
        elements += target.numel()
    return squared_error / elements


def _group_slices(report_groups):
    return {name: slice(lo, hi) for name, (lo, hi) in report_groups}


@torch.no_grad()
def prediction_metrics(
    model: DeepKoopman,
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
    report_groups,
) -> dict[str, Any]:
    state = data["state"][mask]
    action = data["action"][mask]
    target = data["next_state"][mask]
    predictions: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(state), batch_size):
        stop = start + batch_size
        normalized = torch.as_tensor(
            (state[start:stop] - center) / scale,
            dtype=torch.float32,
            device=device,
        )
        command = torch.as_tensor(action[start:stop], dtype=torch.float32, device=device)
        predicted_normalized, _ = model(normalized, command)
        predictions.append(predicted_normalized.cpu().numpy() * scale + center)
    predicted = np.concatenate(predictions, axis=0)
    residual = predicted - target
    hold_residual = state - target
    groups = _group_slices(report_groups)
    metrics: dict[str, Any] = {"transitions": int(len(state))}
    for name, indices in groups.items():
        group_residual = residual[:, indices]
        group_hold = hold_residual[:, indices]
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(np.square(group_residual)))),
            "mae": float(np.mean(np.abs(group_residual))),
            "hold_rmse": float(np.sqrt(np.mean(np.square(group_hold)))),
            "hold_mae": float(np.mean(np.abs(group_hold))),
            "residual_mean": np.mean(group_residual, axis=0).tolist(),
        }
    normalized_residual = residual / scale
    metrics["normalized_mse"] = float(np.mean(np.square(normalized_residual)))
    return metrics


@torch.no_grad()
def transition_reward_metrics(
    model: TransitionRewardModel,
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    """Evaluate reward prediction once per physical transition."""

    state = data["state"][mask]
    action = data["action"][mask]
    next_state = data["next_state"][mask]
    target = data["reward"][mask]
    if len(state) < 1:
        raise ValueError("Reward metric split contains no transitions")
    squared_error = 0.0
    absolute_error = 0.0
    prediction_sum = 0.0
    target_sum = 0.0
    model.eval()
    for start in range(0, len(state), batch_size):
        stop = start + batch_size
        normalized_state = torch.as_tensor(
            (state[start:stop] - center) / scale,
            dtype=torch.float32,
            device=device,
        )
        applied_action = torch.as_tensor(
            action[start:stop], dtype=torch.float32, device=device
        )
        normalized_next_state = torch.as_tensor(
            (next_state[start:stop] - center) / scale,
            dtype=torch.float32,
            device=device,
        )
        reward_target = torch.as_tensor(
            target[start:stop], dtype=torch.float32, device=device
        )
        prediction = model(
            normalized_state, applied_action, normalized_next_state
        )
        residual = prediction - reward_target
        squared_error += float(residual.double().square().sum())
        absolute_error += float(residual.double().abs().sum())
        prediction_sum += float(prediction.double().sum())
        target_sum += float(reward_target.double().sum())
    count = int(len(state))
    mse = squared_error / count
    metrics: dict[str, float | int] = {
        "transitions": count,
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(absolute_error / count),
        "prediction_mean": float(prediction_sum / count),
        "target_mean": float(target_sum / count),
    }
    if not all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if key != "transitions"
    ):
        raise FloatingPointError("Reward-model metrics are non-finite")
    return metrics


@torch.no_grad()
def rollout_prediction_metrics(
    model: DeepKoopman,
    states: np.ndarray,
    actions: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
    report_groups,
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(states), batch_size):
        stop = start + batch_size
        initial = torch.as_tensor(states[start:stop, 0], dtype=torch.float32, device=device)
        command = torch.as_tensor(actions[start:stop], dtype=torch.float32, device=device)
        predicted_normalized, _ = model.rollout(initial, command)
        predictions.append(predicted_normalized.cpu().numpy())
    predicted_normalized = np.concatenate(predictions, axis=0)
    target_normalized = states[:, 1:]
    predicted = predicted_normalized * scale + center
    target = target_normalized * scale + center
    initial = states[:, :1] * scale + center
    hold = np.broadcast_to(initial, target.shape)
    groups = _group_slices(report_groups)
    result: dict[str, Any] = {
        "windows": int(len(states)),
        "k_step": int(actions.shape[1]),
        "all_steps": {},
        "normalized_mse_all_steps": float(
            np.mean(np.square(predicted_normalized - target_normalized))
        ),
        "horizons": {},
    }
    for name, indices in groups.items():
        group_residual = (predicted - target)[..., indices]
        group_hold = (hold - target)[..., indices]
        result["all_steps"][name] = {
            "rmse": float(np.sqrt(np.mean(np.square(group_residual)))),
            "mae": float(np.mean(np.abs(group_residual))),
            "hold_rmse": float(np.sqrt(np.mean(np.square(group_hold)))),
            "hold_mae": float(np.mean(np.abs(group_hold))),
            "residual_mean": np.mean(
                group_residual, axis=tuple(range(group_residual.ndim - 1))
            ).tolist(),
        }
    requested_horizons = sorted({1, 5, 10, int(actions.shape[1])})
    for horizon in requested_horizons:
        if horizon > actions.shape[1]:
            continue
        result["horizons"][str(horizon)] = {}
        for name, indices in groups.items():
            group_residual = (predicted - target)[:, horizon - 1, indices]
            group_hold = (hold - target)[:, horizon - 1, indices]
            result["horizons"][str(horizon)][name] = {
                "rmse": float(np.sqrt(np.mean(np.square(group_residual)))),
                "mae": float(np.mean(np.abs(group_residual))),
                "hold_rmse": float(np.sqrt(np.mean(np.square(group_hold)))),
                "hold_mae": float(np.mean(np.abs(group_hold))),
                "residual_mean": np.mean(group_residual, axis=0).tolist(),
            }
    return result


@torch.no_grad()
def rollout_prediction_metrics_streaming(
    model: DeepKoopman,
    batches: DataLoader,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    report_groups,
) -> dict[str, Any]:
    """Compute the same rollout report without materializing all predictions."""

    groups = _group_slices(report_groups)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)
    k_step = int(batches.dataset.k_step)
    horizons = [h for h in sorted({1, 5, 10, k_step}) if h <= k_step]

    def blank_group(indices: slice) -> dict[str, Any]:
        width = int(indices.stop - indices.start)
        return {
            "squared": 0.0,
            "absolute": 0.0,
            "hold_squared": 0.0,
            "hold_absolute": 0.0,
            "elements": 0,
            "vectors": 0,
            "residual_sum": torch.zeros(width, dtype=torch.float64),
        }

    all_stats = {name: blank_group(indices) for name, indices in groups.items()}
    horizon_stats = {
        horizon: {name: blank_group(indices) for name, indices in groups.items()}
        for horizon in horizons
    }
    normalized_squared = 0.0
    normalized_elements = 0
    windows = 0
    model.eval()
    for batch in batches:
        states, actions = _dynamics_batch(batch)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        predicted_normalized, _ = model.rollout(states[:, 0], actions)
        target_normalized = states[:, 1:]
        normalized_residual = predicted_normalized - target_normalized
        normalized_squared += float(normalized_residual.square().sum())
        normalized_elements += normalized_residual.numel()
        residual = normalized_residual * scale_t
        hold = (states[:, :1] - target_normalized) * scale_t
        windows += len(states)

        def update(stats: dict[str, Any], error: torch.Tensor, held: torch.Tensor) -> None:
            stats["squared"] += float(error.square().sum())
            stats["absolute"] += float(error.abs().sum())
            stats["hold_squared"] += float(held.square().sum())
            stats["hold_absolute"] += float(held.abs().sum())
            stats["elements"] += error.numel()
            stats["vectors"] += error.numel() // error.shape[-1]
            stats["residual_sum"] += error.double().sum(
                dim=tuple(range(error.ndim - 1))
            ).cpu()

        for name, indices in groups.items():
            update(all_stats[name], residual[..., indices], hold[..., indices])
        for horizon in horizons:
            for name, indices in groups.items():
                update(
                    horizon_stats[horizon][name],
                    residual[:, horizon - 1, indices],
                    hold[:, horizon - 1, indices],
                )

    def finish(stats: dict[str, Any]) -> dict[str, Any]:
        elements = max(int(stats["elements"]), 1)
        vectors = max(int(stats["vectors"]), 1)
        return {
            "rmse": float(math.sqrt(stats["squared"] / elements)),
            "mae": float(stats["absolute"] / elements),
            "hold_rmse": float(math.sqrt(stats["hold_squared"] / elements)),
            "hold_mae": float(stats["hold_absolute"] / elements),
            "residual_mean": (stats["residual_sum"] / vectors).tolist(),
        }

    return {
        "windows": int(windows),
        "k_step": k_step,
        "all_steps": {name: finish(stats) for name, stats in all_stats.items()},
        "normalized_mse_all_steps": float(
            normalized_squared / max(normalized_elements, 1)
        ),
        "horizons": {
            str(horizon): {
                name: finish(stats)
                for name, stats in horizon_stats[horizon].items()
            }
            for horizon in horizons
        },
    }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().cpu(),
        "cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(torch.as_tensor(state["torch"]).cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(device_state).cpu() for device_state in state["cuda"]]
        )


def _normalize_resume_history(
    records: Any,
    *,
    completed_epoch: int,
) -> list[dict[str, float | int]]:
    """Validate checkpoint history and collapse repeated epoch records."""

    if not isinstance(records, list):
        raise ValueError("Resume checkpoint history must be a list")
    completed_epoch = int(completed_epoch)
    by_epoch: dict[int, dict[str, float | int]] = {}
    for value in records:
        if not isinstance(value, dict):
            raise ValueError("Every resume history record must be a mapping")
        record = dict(value)
        epoch = record.get("epoch")
        if (
            isinstance(epoch, (bool, np.bool_))
            or not isinstance(epoch, (int, np.integer))
            or not 1 <= int(epoch) <= completed_epoch
        ):
            raise ValueError(
                "Resume history epochs must be integers within the checkpoint"
            )
        by_epoch[int(epoch)] = record
    expected_epochs = list(range(1, completed_epoch + 1))
    if sorted(by_epoch) != expected_epochs:
        raise ValueError(
            "Resume checkpoint history must contain exactly one recoverable "
            "record for every completed epoch"
        )
    return [by_epoch[epoch] for epoch in expected_epochs]


def _validate_reward_metric_mapping(
    value: Any, *, context: str
) -> dict[str, float | int]:
    expected_fields = {
        "transitions",
        "mse",
        "rmse",
        "mae",
        "prediction_mean",
        "target_mean",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{context} reward metric fields are invalid")
    transition_count = value["transitions"]
    if (
        isinstance(transition_count, bool)
        or not isinstance(transition_count, (int, np.integer))
        or int(transition_count) < 1
    ):
        raise ValueError(f"{context} reward transition count is invalid")
    result: dict[str, float | int] = {"transitions": int(transition_count)}
    for name in ("mse", "rmse", "mae", "prediction_mean", "target_mean"):
        metric = value[name]
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            or not 0.0 <= float(metric) <= 1.0 + 1e-6
        ):
            raise ValueError(f"{context} reward {name} is invalid")
        result[name] = float(metric)
    if not math.isclose(
        float(result["rmse"]),
        math.sqrt(float(result["mse"])),
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError(f"{context} reward RMSE is inconsistent with MSE")
    return result


def _validate_resume_checkpoint(
    payload: Any,
    *,
    expected_training_state: dict[str, Any],
    expected_architecture: dict[str, Any],
    expected_reward_architecture: dict[str, Any],
    config: TrainConfig,
    state_dim: int,
) -> list[dict[str, float | int]]:
    """Validate all durable state needed for equivalent latest.pt recovery."""

    if not isinstance(payload, dict):
        raise ValueError("Resume checkpoint must contain a mapping")
    required = {
        "format_version",
        "architecture",
        "model",
        "optimizer",
        "epoch",
        "best_validation",
        "config",
        "normalizers",
        "rng_state",
        "history",
        "training_state",
        "reward_model_architecture",
        "reward_model_input_contract",
        "reward_model_state",
        "best_validation_joint_objective",
        "best_validation_rollout_normalized_mse",
        "best_validation_reward_metrics",
        "latest_validation_joint_objective",
        "latest_validation_rollout_normalized_mse",
        "latest_validation_reward_metrics",
        "validation_selection_metric",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Resume checkpoint is missing fields: {sorted(missing)}")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("Resume checkpoint format version is unsupported")
    if payload["architecture"] != expected_architecture:
        raise ValueError("Resume checkpoint architecture differs from resolved YAML")
    if payload["reward_model_architecture"] != expected_reward_architecture:
        raise ValueError(
            "Resume checkpoint reward architecture differs from resolved YAML"
        )
    if payload["reward_model_input_contract"] != transition_reward_input_contract():
        raise ValueError("Resume checkpoint reward input contract is invalid")
    if not isinstance(payload["model"], Mapping):
        raise ValueError("Resume checkpoint model state must be a mapping")
    if not isinstance(payload["reward_model_state"], Mapping):
        raise ValueError("Resume checkpoint reward-model state must be a mapping")
    try:
        koopman_probe = DeepKoopman(
            state_dim=expected_architecture["state_dim"],
            action_dim=expected_architecture["action_dim"],
            lift_dim=expected_architecture["lift_dim"],
            hidden_dims=expected_architecture["hidden_dims"],
            activation=expected_architecture["activation"],
        )
        koopman_probe.load_state_dict(payload["model"], strict=True)
        reward_probe = TransitionRewardModel.from_architecture(
            payload["reward_model_architecture"]
        )
        reward_probe.load_state_dict(payload["reward_model_state"], strict=True)
        if not all(
            torch.isfinite(value).all()
            for probe in (koopman_probe, reward_probe)
            for value in probe.state_dict().values()
        ):
            raise ValueError("Checkpoint model state contains NaN or Inf")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Resume checkpoint model state is invalid") from exc
    optimizer_state = payload["optimizer"]
    if not isinstance(optimizer_state, Mapping) or set(optimizer_state) != {
        "state",
        "param_groups",
    }:
        raise ValueError("Resume checkpoint optimizer state is invalid")
    param_groups = optimizer_state["param_groups"]
    if not isinstance(param_groups, list) or len(param_groups) != 1:
        raise ValueError("Resume checkpoint must contain one optimizer group")
    if not isinstance(param_groups[0], Mapping):
        raise ValueError("Resume checkpoint optimizer group must be a mapping")
    saved_parameter_ids = param_groups[0].get("params")
    expected_parameters = [
        *koopman_probe.parameters(),
        *reward_probe.parameters(),
    ]
    if (
        not isinstance(saved_parameter_ids, list)
        or len(saved_parameter_ids) != len(expected_parameters)
        or any(
            isinstance(identifier, bool) or not isinstance(identifier, int)
            for identifier in saved_parameter_ids
        )
        or len(set(saved_parameter_ids)) != len(saved_parameter_ids)
    ):
        raise ValueError("Resume checkpoint optimizer parameter list is invalid")
    saved_optimizer_slots = optimizer_state["state"]
    if not isinstance(saved_optimizer_slots, Mapping) or set(
        saved_optimizer_slots
    ) != set(saved_parameter_ids):
        raise ValueError("Resume checkpoint optimizer slots are incomplete")
    for identifier, parameter in zip(
        saved_parameter_ids, expected_parameters, strict=True
    ):
        slot = saved_optimizer_slots[identifier]
        expected_slot_keys = {"step", "exp_avg", "exp_avg_sq"}
        if config.adam_amsgrad:
            expected_slot_keys.add("max_exp_avg_sq")
        if not isinstance(slot, Mapping) or set(slot) != expected_slot_keys:
            raise ValueError("Resume checkpoint Adam slot fields are invalid")
        step = torch.as_tensor(slot["step"])
        if step.numel() != 1 or not torch.isfinite(step).all() or float(step) < 1:
            raise ValueError("Resume checkpoint Adam step is invalid")
        for name in expected_slot_keys - {"step"}:
            moment = slot[name]
            if (
                not isinstance(moment, torch.Tensor)
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
                or not torch.isfinite(moment).all()
            ):
                raise ValueError("Resume checkpoint Adam moment is invalid")
    optimizer_group = param_groups[0]
    expected_optimizer_values = {
        "lr": config.learning_rate,
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_epsilon,
        "weight_decay": config.weight_decay,
        "amsgrad": config.adam_amsgrad,
    }
    optimizer_mismatches = {
        key: (optimizer_group.get(key), expected)
        for key, expected in expected_optimizer_values.items()
        if optimizer_group.get(key) != expected
    }
    if optimizer_mismatches:
        raise ValueError(
            f"Resume checkpoint optimizer hyperparameters differ: "
            f"{optimizer_mismatches}"
        )
    if payload["validation_selection_metric"] != (
        "rollout_normalized_mse_plus_weighted_reward_mse"
    ):
        raise ValueError("Resume checkpoint validation selection metric is invalid")
    try:
        saved_config_identity = _json_identity(dict(payload["config"]))
        expected_config_identity = _json_identity(asdict(config))
    except (TypeError, ValueError) as exc:
        raise ValueError("Resume checkpoint config is not strict JSON data") from exc
    if saved_config_identity != expected_config_identity:
        raise ValueError(
            "Resume checkpoint config differs from the resolved YAML; "
            "use a new output directory"
        )

    epoch = payload["epoch"]
    if (
        isinstance(epoch, (bool, np.bool_))
        or not isinstance(epoch, (int, np.integer))
        or not 1 <= int(epoch) <= config.epochs
    ):
        raise ValueError("Resume checkpoint epoch is outside the resolved run")
    best_validation = payload["best_validation"]
    if (
        isinstance(best_validation, bool)
        or not isinstance(best_validation, (int, float))
        or not math.isfinite(float(best_validation))
    ):
        raise ValueError("Resume checkpoint best_validation must be finite")
    best_joint = payload["best_validation_joint_objective"]
    if (
        isinstance(best_joint, bool)
        or not isinstance(best_joint, (int, float))
        or not math.isfinite(float(best_joint))
        or float(best_joint) != float(best_validation)
    ):
        raise ValueError("Resume checkpoint joint validation objective is invalid")
    best_dynamics = payload["best_validation_rollout_normalized_mse"]
    if (
        isinstance(best_dynamics, bool)
        or not isinstance(best_dynamics, (int, float))
        or not math.isfinite(float(best_dynamics))
        or float(best_dynamics) < 0.0
    ):
        raise ValueError("Resume checkpoint dynamics validation metric is invalid")
    reward_metrics = _validate_reward_metric_mapping(
        payload["best_validation_reward_metrics"],
        context="Resume checkpoint best-validation",
    )
    expected_joint = float(best_dynamics) + config.reward_loss_weight * float(
        reward_metrics["mse"]
    )
    if not math.isclose(
        float(best_joint), expected_joint, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            "Resume checkpoint joint validation objective is inconsistent"
        )
    latest_dynamics = payload["latest_validation_rollout_normalized_mse"]
    latest_joint = payload["latest_validation_joint_objective"]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (latest_dynamics, latest_joint)
    ):
        raise ValueError("Resume checkpoint latest validation metrics are invalid")
    latest_reward_metrics = _validate_reward_metric_mapping(
        payload["latest_validation_reward_metrics"],
        context="Resume checkpoint latest-validation",
    )
    expected_latest_joint = float(
        latest_dynamics
    ) + config.reward_loss_weight * float(latest_reward_metrics["mse"])
    if not math.isclose(
        float(latest_joint),
        expected_latest_joint,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Resume checkpoint latest joint validation objective is inconsistent"
        )

    saved_state = payload["training_state"]
    if not isinstance(saved_state, dict):
        raise ValueError("Resume checkpoint is missing strict training_state")
    mismatches = {
        key: (saved_state.get(key), expected)
        for key, expected in expected_training_state.items()
        if saved_state.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Resume checkpoint identity mismatch: {mismatches}")
    best_epoch = saved_state.get("best_epoch")
    if (
        isinstance(best_epoch, (bool, np.bool_))
        or not isinstance(best_epoch, (int, np.integer))
        or not 1 <= int(best_epoch) <= int(epoch)
    ):
        raise ValueError("Resume checkpoint best_epoch is invalid")
    stale_epochs = saved_state.get("epochs_without_improvement")
    if (
        isinstance(stale_epochs, (bool, np.bool_))
        or not isinstance(stale_epochs, (int, np.integer))
        or int(stale_epochs) < 0
    ):
        raise ValueError("Resume checkpoint early-stopping state is invalid")

    normalizers = payload["normalizers"]
    if not isinstance(normalizers, dict):
        raise ValueError("Resume checkpoint normalizers must be a mapping")
    center = np.asarray(normalizers.get("center"), dtype=np.float64)
    scale = np.asarray(normalizers.get("scale"), dtype=np.float64)
    if center.shape != (state_dim,) or not np.isfinite(center).all():
        raise ValueError("Resume checkpoint center normalizer is invalid")
    if (
        scale.shape != (state_dim,)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Resume checkpoint scale normalizer is invalid")

    rng_state = payload["rng_state"]
    if not isinstance(rng_state, dict) or not {
        "random",
        "numpy",
        "torch",
        "cuda",
    }.issubset(rng_state):
        raise ValueError("Resume checkpoint RNG state is incomplete")
    if not isinstance(rng_state["cuda"], list):
        raise ValueError("Resume checkpoint CUDA RNG state must be a list")
    history = _normalize_resume_history(
        payload["history"], completed_epoch=int(epoch)
    )
    last_record = history[-1]
    history_metric_pairs = {
        "validation_joint_objective": latest_joint,
        "validation_rollout_normalized_mse": latest_dynamics,
        "validation_reward_mse": latest_reward_metrics["mse"],
        "validation_reward_rmse": latest_reward_metrics["rmse"],
        "validation_reward_mae": latest_reward_metrics["mae"],
    }
    for name, expected in history_metric_pairs.items():
        actual = last_record.get(name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isclose(
                float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            raise ValueError(
                f"Resume checkpoint latest {name} differs from history"
            )
    return history


def _synchronize_history_file(
    path: Path,
    history: list[dict[str, float | int]],
) -> None:
    """Atomically make JSONL history match the authoritative checkpoint."""

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in history:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(
    task_name: str,
    dataset_path: Path,
    output_dir: Path,
    config: TrainConfig,
    *,
    device_name: str = "auto",
    experiment_config: ExperimentConfig | Path | str | None = None,
    profile: str | None = None,
    preflight_file: Path | None = None,
    approval_file: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train an approved model or emit a strictly training-free manifest.

    Public callers cannot authorize arbitrary dataclass values: the supplied
    ``TrainConfig`` must exactly equal the complete YAML resolver.  Formal runs
    additionally require the reviewed preflight and its separate approval.
    All of those bindings, including the dataset's environment protocol, are
    validated before a model or optimizer is created.
    """

    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be a boolean")
    spec = get_task_spec(task_name)
    experiment = _coerce_experiment_config(experiment_config)
    bound_profile, execution_spec = _validate_yaml_binding(
        experiment, profile, task_name, config
    )
    if (
        isinstance(config.reward_loss_weight, bool)
        or not isinstance(config.reward_loss_weight, (int, float))
        or not math.isfinite(config.reward_loss_weight)
        or config.reward_loss_weight <= 0.0
    ):
        raise ValueError("reward_loss_weight must be finite and positive")
    reward_architecture = _reward_architecture_mapping(
        state_dim=spec.obs_dim, action_dim=spec.action_dim, config=config
    )
    if preflight_file is None:
        raise PermissionError(
            "Koopman training and dry-run require a reviewed preflight file"
        )
    if not dry_run and approval_file is None:
        raise PermissionError(
            "Formal Koopman training requires a separate approval file"
        )

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    data, masks = load_dataset(dataset_path, task_name)
    dataset_sha256 = _sha256(dataset_path)
    protocol_fingerprint = str(data["protocol_fingerprint"].item())
    environment_protocol_json = str(data["environment_protocol_json"].item())
    collection_protocol_json = str(data["protocol_json"].item())
    environment_protocol = json.loads(environment_protocol_json)

    if dry_run:
        validate_training_preflight(
            experiment,
            bound_profile,
            preflight_file,
            runtime_protocol_fingerprint=protocol_fingerprint,
        )
        approval_payload: dict[str, Any] | None = None
        if approval_file is not None:
            approval_payload = validate_training_approval(
                experiment,
                bound_profile,
                approval_file,
                preflight_file,
                runtime_protocol_fingerprint=protocol_fingerprint,
            )
    else:
        approval_payload = validate_training_approval(
            experiment,
            bound_profile,
            approval_file,
            preflight_file,
            runtime_protocol_fingerprint=protocol_fingerprint,
        )

    authorization_metadata: dict[str, Any] = {
        "authorization_kind": (
            KOOPMAN_MANIFEST_KIND
            if approval_payload is None or dry_run
            else approval_payload["kind"]
        ),
        "training_approved": bool(approval_payload is not None and not dry_run),
        "config_fingerprint": experiment.fingerprint,
        "approval_profile": bound_profile,
        "approval_file_sha256": (
            _sha256(Path(approval_file)) if approval_file is not None else None
        ),
        "preflight_report_sha256": _sha256(Path(preflight_file)),
    }
    if approval_payload is not None and authorization_metadata[
        "preflight_report_sha256"
    ] != approval_payload["preflight_report_sha256"]:
        # ``validate_training_approval`` already enforces the reviewed bytes.
        # Retain this assertion at the trainer boundary so a later refactor
        # cannot silently save a different digest into the checkpoint.
        raise AssertionError("Validated preflight digest changed unexpectedly")
    dataset_authorization = _validate_dataset_provenance(data, len(data["state"]))
    # Dataset collection and model fitting are independent stages. Validate and
    # preserve the dataset's own approval/config/seed provenance above, but do
    # not require it to equal the current Koopman training approval. Task and
    # environment protocol compatibility are enforced by load_dataset().
    dataset_stage_coverage = _validate_stage_coverage(data, execution_spec)
    dataset_window_counts = _validated_window_counts(data, config)

    latest_path = output_dir / "latest.pt"
    if dry_run:
        manifest_path = output_dir / "run_manifest.json"
        manifest = {
            "kind": KOOPMAN_MANIFEST_KIND,
            "training_approved": False,
            "task": task_name,
            "profile": bound_profile,
            "config": _train_config_mapping(config),
            "reward_model_architecture": reward_architecture,
            "reward_model_input_contract": transition_reward_input_contract(),
            "resolved_execution_spec": execution_spec,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": dataset_sha256,
            "dataset_schema_version": int(data["dataset_schema_version"].item()),
            "transition_count": int(len(data["state"])),
            "split_episode_counts": {
                split: int(len(data[f"{split}_episode_ids"]))
                for split in ("train", "validation", "test")
            },
            "protocol_fingerprint": protocol_fingerprint,
            "environment_protocol": environment_protocol,
            "environment_protocol_json": environment_protocol_json,
            "collection_protocol_json": collection_protocol_json,
            "dataset_authorization": dataset_authorization,
            "dataset_stage_coverage": dataset_stage_coverage,
            "dataset_window_counts": dataset_window_counts,
            "would_resume": latest_path.is_file(),
            "optimizer_steps": 0,
            "epochs_completed": 0,
            "device_requested": device_name,
            **authorization_metadata,
            "notice": (
                "This training-free manifest is not an approval artifact and "
                "does not authorize optimization."
            ),
        }
        _atomic_json(manifest_path, manifest)
        return {
            "kind": KOOPMAN_MANIFEST_KIND,
            "task": task_name,
            "profile": bound_profile,
            "config_fingerprint": experiment.fingerprint,
            "dataset_sha256": dataset_sha256,
            "protocol_fingerprint": protocol_fingerprint,
            "run_manifest": str(manifest_path.resolve()),
            "would_resume": latest_path.is_file(),
            "optimizer_steps": 0,
            "epochs_completed": 0,
        }

    if approval_payload is None or authorization_metadata["training_approved"] is not True:
        raise PermissionError("Koopman optimization is not approved")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.jsonl"
    resume_payload: dict[str, Any] | None = None
    resume_history: list[dict[str, float | int]] | None = None
    if latest_path.exists():
        resume_payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        expected_resume_identity = {
            "checkpoint_kind": "dmc_k_step_koopman_latest",
            "task_name": task_name,
            "state_kind": task_name,
            "dataset_sha256": dataset_sha256,
            "dataset_schema_version": int(
                data["dataset_schema_version"].item()
            ),
            "protocol_fingerprint": protocol_fingerprint,
            "environment_protocol_json": environment_protocol_json,
            "collection_protocol_json": collection_protocol_json,
            "authorization_kind": approval_payload["kind"],
            "training_approved": True,
            "config_fingerprint": experiment.fingerprint,
            "approval_profile": bound_profile,
            "approval_file_sha256": authorization_metadata[
                "approval_file_sha256"
            ],
            "preflight_report_sha256": authorization_metadata[
                "preflight_report_sha256"
            ],
            "resolved_execution_spec": execution_spec,
            "dataset_authorization": dataset_authorization,
            "dataset_stage_coverage": dataset_stage_coverage,
            "dataset_window_counts": dataset_window_counts,
        }
        expected_architecture = {
            "architecture": "fullA_history_v2_adapted",
            "state_dim": spec.obs_dim,
            "action_dim": spec.action_dim,
            "lift_dim": config.lift_dim,
            "hidden_dims": list(config.hidden_dims),
            "activation": config.activation,
        }
        resume_history = _validate_resume_checkpoint(
            resume_payload,
            expected_training_state=expected_resume_identity,
            expected_architecture=expected_architecture,
            expected_reward_architecture=reward_architecture,
            config=config,
            state_dim=spec.obs_dim,
        )

    control_dt = environment_protocol.get("control_dt")
    if (
        isinstance(control_dt, bool)
        or not isinstance(control_dt, (int, float))
        or not math.isfinite(float(control_dt))
        or float(control_dt) <= 0
    ):
        raise ValueError("Dataset environment protocol has invalid control_dt")
    effective_spectral_radius_limit = config.spectral_radius_limit ** (
        float(control_dt) / config.stability_reference_dt
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )

    center, scale = fit_normalizer(
        data["state"][masks["train"]], data["next_state"][masks["train"]]
    )
    if resume_payload is not None:
        saved_normalizers = resume_payload["normalizers"]
        saved_center = np.asarray(saved_normalizers["center"], dtype=np.float32)
        saved_scale = np.asarray(saved_normalizers["scale"], dtype=np.float32)
        if not np.array_equal(saved_center, center) or not np.array_equal(
            saved_scale, scale
        ):
            raise ValueError(
                "Resume checkpoint normalizer differs from the bound dataset"
            )
    window_datasets = {
        split: KoopmanWindowDataset(
            data,
            data[f"{split}_episode_ids"],
            center,
            scale,
            k_step=config.k_step,
            obs_dim=spec.obs_dim,
            action_dim=spec.action_dim,
            max_windows=(config.max_windows if split == "train" else None),
            seed=config.seed + {"train": 0, "validation": 1, "test": 2}[split],
        )
        for split in ("train", "validation", "test")
    }
    loaders = {
        split: _make_loader(
            window_datasets[split],
            batch_size=config.batch_size,
            shuffle=split == "train",
            seed=config.seed,
        )
        for split in ("train", "validation", "test")
    }

    model = DeepKoopman(
        state_dim=spec.obs_dim,
        action_dim=spec.action_dim,
        lift_dim=config.lift_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
    ).to(device)
    reward_model = TransitionRewardModel(
        state_dim=spec.obs_dim,
        action_dim=spec.action_dim,
        hidden_dims=config.reward_hidden_dims,
        activation=config.activation,
    ).to(device)
    if reward_model.architecture() != reward_architecture:
        raise AssertionError("Constructed reward architecture differs from config")
    optimized_parameters = [*model.parameters(), *reward_model.parameters()]
    optimizer = torch.optim.Adam(
        optimized_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        amsgrad=config.adam_amsgrad,
    )

    start_epoch = 1
    best_validation = float("inf")
    best_validation_dynamics = float("inf")
    best_validation_reward: dict[str, float | int] | None = None
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0
    if resume_payload is not None:
        saved_state = resume_payload["training_state"]
        model.load_state_dict(resume_payload["model"])
        reward_model.load_state_dict(
            resume_payload["reward_model_state"], strict=True
        )
        optimizer.load_state_dict(resume_payload["optimizer"])
        _restore_rng_state(resume_payload.get("rng_state"))
        start_epoch = int(resume_payload["epoch"]) + 1
        best_validation = float(resume_payload["best_validation"])
        best_validation_dynamics = float(
            resume_payload["best_validation_rollout_normalized_mse"]
        )
        best_validation_reward = dict(
            resume_payload["best_validation_reward_metrics"]
        )
        best_epoch = int(saved_state.get("best_epoch", 0))
        epochs_without_improvement = int(
            saved_state.get(
                "epochs_without_improvement", max(0, start_epoch - 1 - best_epoch)
            )
        )
        if resume_history is None:
            raise AssertionError("Validated resume history disappeared")
        history = resume_history
        print(
            f"resumed Koopman training from epoch {start_epoch} "
            f"(best_validation={best_validation:.6g})",
            flush=True,
        )

    # The checkpoint is the durable transaction boundary.  Remove stale or
    # duplicated JSONL tail records left by an interrupted epoch before append.
    _synchronize_history_file(history_path, history)
    start_time = time.perf_counter()
    elapsed_base = float(history[-1].get("elapsed_seconds", 0.0)) if history else 0.0
    for epoch in range(start_epoch, config.epochs + 1):
        _set_loader_epoch(loaders["train"], epoch)
        model.train()
        reward_model.train()
        weighted_loss = 0.0
        weighted_koopman_loss = 0.0
        weighted_reward_loss = 0.0
        sample_count = 0
        last_loss = None
        for states, actions, rewards, reward_next_states in loaders["train"]:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            rewards = rewards.to(device, non_blocking=True)
            reward_next_states = reward_next_states.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = koopman_loss(
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
                spectral_radius_limit=effective_spectral_radius_limit,
                target_latent_std=config.target_latent_std,
                svd_min_singular_value=config.svd_min_singular_value,
            )
            reward_prediction = reward_model(
                states[:, 0], actions[:, 0], reward_next_states
            )
            reward_loss = F.mse_loss(reward_prediction, rewards)
            joint_loss = loss.total + config.reward_loss_weight * reward_loss
            if not torch.isfinite(joint_loss):
                raise FloatingPointError("Non-finite joint Koopman/reward loss")
            joint_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                optimized_parameters, config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite Koopman/reward gradient")
            optimizer.step()
            count = len(states)
            weighted_loss += float(joint_loss.detach()) * count
            weighted_koopman_loss += float(loss.total.detach()) * count
            weighted_reward_loss += float(reward_loss.detach()) * count
            sample_count += count
            last_loss = loss

        validation_mse = rollout_normalized_mse(model, loaders["validation"], device)
        if not math.isfinite(validation_mse):
            raise FloatingPointError("Koopman validation rollout MSE is non-finite")
        validation_reward = transition_reward_metrics(
            reward_model,
            data,
            masks["validation"],
            center,
            scale,
            device,
            config.batch_size,
        )
        validation_joint = (
            validation_mse
            + config.reward_loss_weight * float(validation_reward["mse"])
        )
        if not math.isfinite(validation_joint):
            raise FloatingPointError("Joint validation objective is non-finite")
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_total": weighted_loss / sample_count,
            "train_koopman_total": weighted_koopman_loss / sample_count,
            "train_reward_mse": weighted_reward_loss / sample_count,
            "validation_joint_objective": validation_joint,
            "validation_rollout_normalized_mse": validation_mse,
            "validation_reward_mse": float(validation_reward["mse"]),
            "validation_reward_rmse": float(validation_reward["rmse"]),
            "validation_reward_mae": float(validation_reward["mae"]),
            "spectral_radius": float(last_loss.spectral_radius.detach())
            if last_loss is not None
            else float("nan"),
            "stability_penalty": float(last_loss.stability.detach())
            if last_loss is not None
            else float("nan"),
            "elapsed_seconds": elapsed_base + time.perf_counter() - start_time,
        }
        history.append(epoch_record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

        if validation_joint < best_validation:
            best_validation = validation_joint
            best_validation_dynamics = validation_mse
            best_validation_reward = dict(validation_reward)
            best_epoch = epoch
            epochs_without_improvement = 0
            if best_validation_reward is None:
                raise AssertionError("Best reward validation metrics disappeared")
            _atomic_torch_save(
                best_path,
                {
                    "kind": "dmc_k_step_koopman",
                    "checkpoint_kind": "dmc_k_step_koopman_best",
                    "model_state": model.state_dict(),
                    "architecture": model.architecture(),
                    "reward_model_state": reward_model.state_dict(),
                    "reward_model_architecture": reward_model.architecture(),
                    "reward_model_input_contract": (
                        transition_reward_input_contract()
                    ),
                    "normalizer": {
                        "center": torch.from_numpy(center),
                        "scale": torch.from_numpy(scale),
                        "fit": "train episodes only; per-dim mean/std",
                    },
                    "state_kind": task_name,
                    "task": task_name,
                    "config": asdict(config),
                    "resolved_execution_spec": execution_spec,
                    "dataset_path": str(dataset_path.resolve()),
                    "dataset_sha256": dataset_sha256,
                    "protocol_fingerprint": protocol_fingerprint,
                    "environment_protocol_json": environment_protocol_json,
                    "collection_protocol_json": collection_protocol_json,
                    "dataset_schema_version": int(
                        data["dataset_schema_version"].item()
                    ),
                    "dataset_authorization": dataset_authorization,
                    "dataset_stage_coverage": dataset_stage_coverage,
                    "dataset_window_counts": dataset_window_counts,
                    **authorization_metadata,
                    "best_epoch": best_epoch,
                    "best_validation_joint_objective": best_validation,
                    "best_validation_rollout_normalized_mse": (
                        best_validation_dynamics
                    ),
                    "best_validation_reward_metrics": best_validation_reward,
                    "validation_selection_metric": (
                        "rollout_normalized_mse_plus_weighted_reward_mse"
                    ),
                    "k_step": config.k_step,
                    "history": history,
                    "split_episode_ids": {
                        split: torch.from_numpy(data[f"{split}_episode_ids"])
                        for split in ("train", "validation", "test")
                    },
                },
            )
        else:
            epochs_without_improvement += 1

        if (
            epoch % config.checkpoint_every == 0
            or epoch == config.epochs
            or epochs_without_improvement == config.patience
        ):
            from antmaze_ac.koopman.checkpoint import save_checkpoint

            if best_validation_reward is None:
                raise AssertionError("Latest checkpoint has no best reward metrics")

            save_checkpoint(
                latest_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation=best_validation,
                config=asdict(config),
                normalizers={
                    "center": center.tolist(),
                    "scale": scale.tolist(),
                    "fit": "train episodes only; per-dim mean/std",
                },
                elapsed_seconds=epoch_record["elapsed_seconds"],
                rng_state=_capture_rng_state(),
                history=history,
                training_state={
                    "checkpoint_kind": "dmc_k_step_koopman_latest",
                    "task_name": task_name,
                    "state_kind": task_name,
                    "dataset_sha256": dataset_sha256,
                    "dataset_schema_version": int(
                        data["dataset_schema_version"].item()
                    ),
                    "protocol_fingerprint": protocol_fingerprint,
                    "environment_protocol_json": environment_protocol_json,
                    "collection_protocol_json": collection_protocol_json,
                    "resolved_execution_spec": execution_spec,
                    "dataset_authorization": dataset_authorization,
                    "dataset_stage_coverage": dataset_stage_coverage,
                    "dataset_window_counts": dataset_window_counts,
                    **authorization_metadata,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "effective_spectral_radius_limit": (
                        effective_spectral_radius_limit
                    ),
                },
                extra_payload={
                    "reward_model_architecture": reward_model.architecture(),
                    "reward_model_input_contract": (
                        transition_reward_input_contract()
                    ),
                    "reward_model_state": reward_model.state_dict(),
                    "best_validation_joint_objective": best_validation,
                    "best_validation_rollout_normalized_mse": (
                        best_validation_dynamics
                    ),
                    "best_validation_reward_metrics": best_validation_reward,
                    "latest_validation_joint_objective": validation_joint,
                    "latest_validation_rollout_normalized_mse": validation_mse,
                    "latest_validation_reward_metrics": validation_reward,
                    "validation_selection_metric": (
                        "rollout_normalized_mse_plus_weighted_reward_mse"
                    ),
                },
            )

        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:04d} train={epoch_record['train_total']:.6g} "
                f"val_joint={validation_joint:.6g} "
                f"val_nMSE={validation_mse:.6g} "
                f"val_reward_MSE={validation_reward['mse']:.6g} "
                f"rhoA={epoch_record['spectral_radius']:.6g} "
                f"elapsed={epoch_record['elapsed_seconds']:.1f}s",
                flush=True,
            )

        if epochs_without_improvement >= config.patience:
            print(
                f"early stopping at epoch {epoch} "
                f"(no improvement for {config.patience} epochs, "
                f"best={best_validation:.6g} @ epoch {best_epoch})",
                flush=True,
            )
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if checkpoint.get("reward_model_architecture") != reward_architecture:
        raise ValueError("Best checkpoint reward architecture is invalid")
    if checkpoint.get(
        "reward_model_input_contract"
    ) != transition_reward_input_contract():
        raise ValueError("Best checkpoint reward input contract is invalid")
    reward_model.load_state_dict(checkpoint["reward_model_state"], strict=True)
    split_metrics = {
        split: {
            "one_step": prediction_metrics(
                model,
                data,
                masks[split],
                center,
                scale,
                device,
                config.batch_size,
                spec.report_groups,
            ),
            "rollout": rollout_prediction_metrics_streaming(
                model,
                loaders[split],
                center,
                scale,
                device,
                spec.report_groups,
            ),
            "reward": transition_reward_metrics(
                reward_model,
                data,
                masks[split],
                center,
                scale,
                device,
                config.batch_size,
            ),
        }
        for split, mask in masks.items()
    }
    reported_best_dynamics = float(
        checkpoint["best_validation_rollout_normalized_mse"]
    )
    recomputed_best_dynamics = float(
        split_metrics["validation"]["rollout"]["normalized_mse_all_steps"]
    )
    if not math.isclose(
        reported_best_dynamics,
        recomputed_best_dynamics,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError("Best checkpoint dynamics metric does not match its state")
    reported_best_reward = checkpoint["best_validation_reward_metrics"]
    recomputed_best_reward = split_metrics["validation"]["reward"]
    for name in ("mse", "rmse", "mae", "prediction_mean", "target_mean"):
        if not math.isclose(
            float(reported_best_reward[name]),
            float(recomputed_best_reward[name]),
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise ValueError(
                f"Best checkpoint reward {name} does not match its state"
            )
    recomputed_joint = reported_best_dynamics + config.reward_loss_weight * float(
        reported_best_reward["mse"]
    )
    if not math.isclose(
        float(checkpoint["best_validation_joint_objective"]),
        recomputed_joint,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("Best checkpoint joint validation objective is inconsistent")
    with torch.no_grad():
        spectral_radius = float(torch.linalg.eigvals(model.A).abs().max().cpu())
    report = {
        "kind": "dmc_k_step_koopman_report",
        "task": task_name,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "protocol_fingerprint": protocol_fingerprint,
        "environment_protocol_json": environment_protocol_json,
        "collection_protocol_json": collection_protocol_json,
        "checkpoint_path": str(best_path.resolve()),
        "device": str(device),
        "config": asdict(config),
        "resolved_execution_spec": execution_spec,
        "dataset_authorization": dataset_authorization,
        "dataset_stage_coverage": dataset_stage_coverage,
        "dataset_window_counts": dataset_window_counts,
        **authorization_metadata,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_joint_objective": float(
            checkpoint["best_validation_joint_objective"]
        ),
        "best_validation_rollout_normalized_mse": float(
            checkpoint["best_validation_rollout_normalized_mse"]
        ),
        "best_validation_reward_metrics": dict(
            checkpoint["best_validation_reward_metrics"]
        ),
        "validation_selection_metric": checkpoint["validation_selection_metric"],
        "reward_model_architecture": reward_model.architecture(),
        "reward_model_input_contract": transition_reward_input_contract(),
        "elapsed_seconds": elapsed_base + time.perf_counter() - start_time,
        "spectral_radius_A": spectral_radius,
        "effective_spectral_radius_limit": effective_spectral_radius_limit,
        "state_kind": task_name,
        "normalizer": {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "fit": "train episodes only",
        },
        "metrics": split_metrics,
        "window_counts": {
            split: int(len(window_datasets[split]))
            for split in ("train", "validation", "test")
        },
        "history": history,
        "scope": (
            f"K-step={config.k_step} open-loop identification on DMC "
            f"{task_name} multi-stage data "
            f"({spec.obs_dim}-dim state, {spec.action_dim}-dim action)"
        ),
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    parser.add_argument("--preflight-file", type=Path, required=True)
    parser.add_argument("--approval-file", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write a non-authorizing manifest; never construct an optimizer",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if not args.dry_run and args.approval_file is None:
        parser.error("--approval-file is required unless --dry-run is explicit")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiment = load_experiment_config(args.config)
    config = train_config_from_experiment(experiment)
    result = train(
        experiment.task,
        args.dataset,
        args.output_dir,
        config,
        device_name=args.device,
        experiment_config=experiment,
        profile=args.profile,
        preflight_file=args.preflight_file,
        approval_file=args.approval_file,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
