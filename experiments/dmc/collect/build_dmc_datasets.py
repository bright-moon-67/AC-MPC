"""Build a strict, protocol-stamped DMC Koopman dataset.

All requested seed directories are required.  The builder rejects duplicate
episode ids, duplicate trajectories, repeated global steps within a seed,
mixed protocols, incomplete episodes, broken transition chains, and split
overlap.  Splits are episode-level, so no transition or trajectory can leak
between train, validation, and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dmc.collect.collect_dmc_data import (
    COLLECTION_SCHEMA_VERSION as STANDALONE_COLLECTION_SCHEMA_VERSION,
)
from experiments.dmc.config import (
    PROFILE_NAMES,
    load_experiment_config,
    resolve_execution_spec,
)
from experiments.dmc.protocol import (
    canonical_json,
    protocol_fingerprint_from_json,
)
from experiments.dmc.tasks.registry import get_task_spec


ON_POLICY_COLLECTION_SCHEMA_VERSION = 4
DATASET_SCHEMA_VERSION = 4
DATA_SOURCES = ("standalone_ablation", "ppo_training_stages")
PROVENANCE_FIELDS = (
    "training_seed",
    "actor_type",
    "training_approved",
    "config_fingerprint",
    "approval_profile",
    "approval_file_sha256",
    "preflight_report_sha256",
    "authorization_kind",
    "train_seed_index",
)
COLLECTION_STAGE_NAMES = ("early", "mid", "late")
COLLECTION_CONTRACT_FIELDS = (
    "collection_selection_strategy",
    "collection_max_transitions",
    "collection_total_updates",
)


@dataclass(frozen=True)
class BuildConfig:
    task_name: str
    collect_root: Path
    output: Path
    seed_dirs: tuple[str, ...]
    validation_every: int = 10
    test_offset: int = 9
    source: str = "ppo_training_stages"
    expected_config_fingerprint: str | None = None
    expected_approval_profile: str | None = None
    expected_training_seeds: tuple[int, ...] | None = None
    expected_collection_max_transitions: int | None = None
    expected_collection_total_updates: int | None = None


@dataclass(frozen=True)
class EpisodeRecord:
    original_episode_id: int
    update: int
    source_seed_index: int
    source_seed_dir: str
    source_file: str
    trajectory_sha256: str
    collection_stage: str | None
    arrays: dict[str, np.ndarray]


TRANSITION_FIELDS = (
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
)


def _scalar(chunk: dict[str, np.ndarray], name: str) -> Any:
    value = np.asarray(chunk[name])
    if value.shape != ():
        raise ValueError(f"Chunk field {name!r} must be scalar, got {value.shape}")
    return value.item()


def _positive_integer_scalar(
    chunk: dict[str, np.ndarray], name: str, *, minimum: int = 1
) -> int:
    value = _scalar(chunk, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Chunk field {name!r} must be an integer scalar")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"Chunk field {name!r} must be {qualifier}")
    return int(value)


def _collection_stage_ranges(total_updates: int) -> tuple[tuple[int, int], ...]:
    first_end = total_updates // 3
    second_end = 2 * total_updates // 3
    return (
        (1, first_end),
        (first_end + 1, second_end),
        (second_end + 1, total_updates),
    )


def _collection_stage_for_update(update: int, total_updates: int) -> str:
    if not 1 <= update <= total_updates:
        raise ValueError(
            f"Episode completion update {update} lies outside "
            f"[1, {total_updates}]"
        )
    for stage, (_start, end) in zip(
        COLLECTION_STAGE_NAMES, _collection_stage_ranges(total_updates)
    ):
        if update <= end:
            return stage
    raise AssertionError("collection stage ranges do not cover total_updates")


def _load_chunks(
    root: Path, seed_dir: str
) -> list[tuple[Path, dict[str, np.ndarray]]]:
    source = root / seed_dir
    if not source.is_dir():
        raise FileNotFoundError(f"Required seed directory does not exist: {source}")
    chunks: list[tuple[Path, dict[str, np.ndarray]]] = []
    for path in sorted(source.glob("coverage_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            chunks.append((path, {name: archive[name] for name in archive.files}))
    if not chunks:
        raise FileNotFoundError(f"No coverage_*.npz chunks under {source}")
    return chunks


def _trajectory_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("state", "action", "next_state"):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_chunk(
    path: Path,
    chunk: dict[str, np.ndarray],
    *,
    task_name: str,
    obs_dim: int,
    action_dim: int,
    source: str,
) -> tuple[
    str,
    str,
    int,
    str,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    required = set(TRANSITION_FIELDS) | {
        "collection_schema_version",
        "protocol_json",
        "environment_protocol_json",
        "protocol_fingerprint",
        "task",
        "seed_index",
        "seed_dir",
    }
    if source == "ppo_training_stages":
        required.update(PROVENANCE_FIELDS)
        required.update(COLLECTION_CONTRACT_FIELDS)
        required.add("collection_stage")
    missing = required - chunk.keys()
    if missing:
        raise KeyError(f"Chunk {path} is missing fields: {sorted(missing)}")
    schema_version = int(_scalar(chunk, "collection_schema_version"))
    expected_schema = (
        ON_POLICY_COLLECTION_SCHEMA_VERSION
        if source == "ppo_training_stages"
        else STANDALONE_COLLECTION_SCHEMA_VERSION
    )
    if schema_version != expected_schema:
        raise ValueError(
            f"Chunk {path} schema {schema_version} != {expected_schema} for "
            f"source={source}"
        )
    if str(_scalar(chunk, "task")) != task_name:
        raise ValueError(f"Chunk {path} belongs to a different task")
    protocol_json = str(_scalar(chunk, "protocol_json"))
    environment_protocol_json = str(
        _scalar(chunk, "environment_protocol_json")
    )
    protocol_fingerprint = str(_scalar(chunk, "protocol_fingerprint"))
    try:
        expected_fingerprint = protocol_fingerprint_from_json(
            environment_protocol_json
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Chunk {path} has invalid environment_protocol_json"
        ) from exc
    if protocol_fingerprint != expected_fingerprint:
        raise ValueError(f"Chunk {path} has an invalid protocol fingerprint")
    try:
        protocol = json.loads(protocol_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Chunk {path} has invalid protocol_json") from exc
    environment_protocol = json.loads(environment_protocol_json)
    if canonical_json(environment_protocol) != environment_protocol_json:
        raise ValueError(f"Chunk {path} environment protocol is not canonical")
    if protocol.get("task") != task_name:
        raise ValueError(f"Chunk {path} protocol task mismatch")
    if int(protocol.get("obs_dim", -1)) != obs_dim:
        raise ValueError(f"Chunk {path} protocol obs_dim mismatch")
    if int(protocol.get("action_dim", -1)) != action_dim:
        raise ValueError(f"Chunk {path} protocol action_dim mismatch")
    for key in ("task", "obs_dim", "action_dim"):
        if environment_protocol.get(key) != protocol.get(key):
            raise ValueError(
                f"Chunk {path} collection/environment protocol {key} mismatch"
            )

    count = len(chunk["state"])
    if count == 0:
        raise ValueError(f"Chunk {path} is empty")
    for name in TRANSITION_FIELDS:
        if len(chunk[name]) != count:
            raise ValueError(f"Chunk {path} field {name!r} has wrong length")
    if chunk["state"].shape != (count, obs_dim):
        raise ValueError(f"Chunk {path} has invalid state shape")
    if chunk["next_state"].shape != (count, obs_dim):
        raise ValueError(f"Chunk {path} has invalid next_state shape")
    for name in ("requested_action", "action"):
        if chunk[name].shape != (count, action_dim):
            raise ValueError(f"Chunk {path} has invalid {name} shape")
    for name in (
        "state",
        "requested_action",
        "action",
        "next_state",
        "reward",
        "discount",
    ):
        if not np.isfinite(chunk[name]).all():
            raise FloatingPointError(f"Chunk {path} field {name!r} is non-finite")
    provenance: dict[str, Any] | None = None
    collection_contract: dict[str, Any] | None = None
    if source == "ppo_training_stages":
        stages = np.asarray(chunk["collection_stage"])
        if stages.shape != (count,):
            raise ValueError(
                f"Chunk {path} field 'collection_stage' has invalid shape "
                f"{stages.shape}"
            )
        if stages.dtype.kind != "U":
            raise TypeError(
                f"Chunk {path} field 'collection_stage' must be a string array"
            )
        strategy = _scalar(chunk, "collection_selection_strategy")
        if (
            not isinstance(strategy, str)
            or not strategy
            or strategy.strip() != strategy
        ):
            raise ValueError(
                f"Chunk {path} has invalid collection_selection_strategy"
            )
        max_transitions = _positive_integer_scalar(
            chunk, "collection_max_transitions"
        )
        total_updates = _positive_integer_scalar(
            chunk, "collection_total_updates", minimum=3
        )
        collection_contract = {
            "selection_strategy": strategy,
            "max_transitions": max_transitions,
            "total_updates": total_updates,
        }
        provenance = {name: _scalar(chunk, name) for name in PROVENANCE_FIELDS}
        if provenance["actor_type"] != "PPO":
            raise ValueError(f"Chunk {path} is not reference PPO data")
        if provenance["training_approved"] is not True:
            raise ValueError(f"Chunk {path} is not approval-bound training data")
        if provenance["authorization_kind"] != "dmc_training_approval_v1":
            raise ValueError(f"Chunk {path} has a non-formal authorization kind")
        if int(provenance["train_seed_index"]) != int(
            _scalar(chunk, "seed_index")
        ):
            raise ValueError(f"Chunk {path} seed lineage is inconsistent")
        for name in ("approval_file_sha256", "preflight_report_sha256"):
            value = provenance[name]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"Chunk {path} has invalid {name}")
        fingerprint = provenance["config_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or not fingerprint.startswith("sha256:")
            or len(fingerprint) != 71
        ):
            raise ValueError(f"Chunk {path} has invalid config_fingerprint")
    return (
        protocol_json,
        environment_protocol_json,
        int(_scalar(chunk, "seed_index")),
        str(_scalar(chunk, "seed_dir")),
        provenance,
        collection_contract,
    )


def _episodes_from_chunk(
    path: Path,
    chunk: dict[str, np.ndarray],
    *,
    source_seed_index: int,
    source_seed_dir: str,
    collection_total_updates: int | None = None,
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    episode_ids = np.asarray(chunk["episode_id"], dtype=np.int64)
    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        fields = list(TRANSITION_FIELDS)
        if collection_total_updates is not None:
            fields.append("collection_stage")
        arrays = {name: np.asarray(chunk[name][indices]) for name in fields}
        steps = np.asarray(arrays["step_index"], dtype=np.int64)
        if not np.array_equal(steps, np.arange(len(steps), dtype=np.int64)):
            raise ValueError(f"Episode {int(episode_id)} has non-consecutive steps")
        global_steps = np.asarray(arrays["global_step"], dtype=np.int64)
        if len(steps) > 1 and not bool(np.all(np.diff(global_steps) > 0)):
            raise ValueError(
                f"Episode {int(episode_id)} has non-increasing global steps"
            )
        chain_error = (
            float(
                np.max(
                    np.abs(arrays["next_state"][:-1] - arrays["state"][1:])
                )
            )
            if len(steps) > 1
            else 0.0
        )
        if chain_error > 2e-5:
            raise ValueError(
                f"Episode {int(episode_id)} chain mismatch {chain_error:.3e}"
            )
        updates = np.asarray(arrays["update"], dtype=np.int64)
        reset_seeds = np.unique(np.asarray(arrays["reset_seed"], dtype=np.int64))
        if len(updates) > 1 and not bool(np.all(np.diff(updates) >= 0)):
            raise ValueError(
                f"Episode {int(episode_id)} has non-monotonic PPO update tags"
            )
        if len(reset_seeds) != 1:
            raise ValueError(
                f"Episode {int(episode_id)} mixes reset seeds"
            )
        done = np.asarray(arrays["done"], dtype=np.bool_)
        terminated = np.asarray(arrays["terminated"], dtype=np.bool_)
        truncated = np.asarray(arrays["truncated"], dtype=np.bool_)
        collector_truncated = np.asarray(
            arrays["collector_truncated"], dtype=np.bool_
        )
        if bool(np.logical_and(terminated, truncated).any()):
            raise ValueError(
                f"Episode {int(episode_id)} marks a transition both terminated "
                "and truncated"
            )
        boundary = np.logical_or(terminated, truncated)
        if not np.array_equal(done, boundary):
            raise ValueError(
                f"Episode {int(episode_id)} has inconsistent done/terminal flags"
            )
        if bool(np.logical_and(collector_truncated, ~truncated).any()):
            raise ValueError(
                f"Episode {int(episode_id)} has an invalid collector truncation"
            )
        if not boundary[-1] or bool(boundary[:-1].any()):
            raise ValueError(
                f"Episode {int(episode_id)} is incomplete or terminates early"
            )
        collection_stage: str | None = None
        if collection_total_updates is not None:
            labels = np.unique(np.asarray(arrays["collection_stage"]))
            if len(labels) != 1:
                raise ValueError(
                    f"Episode {int(episode_id)} mixes collection stages"
                )
            collection_stage = str(labels[0])
            if collection_stage not in COLLECTION_STAGE_NAMES:
                raise ValueError(
                    f"Episode {int(episode_id)} has invalid collection stage "
                    f"{collection_stage!r}"
                )
            expected_stage = _collection_stage_for_update(
                int(updates[-1]), collection_total_updates
            )
            if collection_stage != expected_stage:
                raise ValueError(
                    f"Episode {int(episode_id)} collection stage "
                    f"{collection_stage!r} does not match completion update "
                    f"{int(updates[-1])} ({expected_stage!r})"
                )
        records.append(
            EpisodeRecord(
                original_episode_id=int(episode_id),
                # A long DMC episode may cross several PPO rollout updates.
                # Sort/split it by the policy stage at which it completed while
                # retaining every per-transition update tag in the dataset.
                update=int(updates[-1]),
                source_seed_index=source_seed_index,
                source_seed_dir=source_seed_dir,
                source_file=str(path.resolve()),
                trajectory_sha256=_trajectory_digest(arrays),
                collection_stage=collection_stage,
                arrays=arrays,
            )
        )
    return records


def build(config: BuildConfig) -> Path:
    if config.source not in DATA_SOURCES:
        raise ValueError(f"Unsupported data source {config.source!r}")
    if not config.seed_dirs:
        raise ValueError("At least one seed directory is required")
    if len(set(config.seed_dirs)) != len(config.seed_dirs):
        raise ValueError("seed_dirs contains duplicates")
    if config.validation_every < 3:
        raise ValueError("validation_every must be at least 3")
    formal_expectations = (
        config.expected_config_fingerprint,
        config.expected_approval_profile,
        config.expected_training_seeds,
        config.expected_collection_max_transitions,
        config.expected_collection_total_updates,
    )
    if any(value is not None for value in formal_expectations) and not all(
        value is not None for value in formal_expectations
    ):
        raise ValueError("Formal primary identity expectations must be complete")
    if config.source != "ppo_training_stages" and any(
        value is not None for value in formal_expectations
    ):
        raise ValueError("Standalone ablation cannot use primary identity binding")

    spec = get_task_spec(config.task_name)
    episodes: list[EpisodeRecord] = []
    source_files: list[str] = []
    expected_protocol_json: str | None = None
    expected_environment_protocol_json: str | None = None
    seen_seed_indices: dict[int, str] = {}
    seed_index_for_dir: dict[str, int] = {}
    seen_episode_ids: dict[int, str] = {}
    seen_trajectory_hashes: dict[str, int] = {}
    seen_global_steps_by_seed: dict[int, set[int]] = {}
    provenance_by_seed: dict[int, dict[str, Any]] = {}
    shared_provenance: dict[str, Any] | None = None
    collection_contract_by_seed: dict[int, dict[str, Any]] = {}
    shared_collection_contract: dict[str, Any] | None = None
    collection_transition_counts_by_seed: dict[int, int] = {}
    collection_stage_transition_counts_by_seed: dict[int, dict[str, int]] = {}
    collection_stage_episode_counts_by_seed: dict[int, dict[str, int]] = {}

    for requested_seed_dir in config.seed_dirs:
        chunks = _load_chunks(config.collect_root, requested_seed_dir)
        for path, chunk in chunks:
            (
                protocol_json,
                environment_protocol_json,
                seed_index,
                embedded_seed_dir,
                provenance,
                collection_contract,
            ) = _validate_chunk(
                path,
                chunk,
                task_name=config.task_name,
                obs_dim=spec.obs_dim,
                action_dim=spec.action_dim,
                source=config.source,
            )
            if embedded_seed_dir != requested_seed_dir:
                raise ValueError(
                    f"Chunk {path} says seed_dir={embedded_seed_dir!r}, expected "
                    f"{requested_seed_dir!r}"
                )
            previous_index = seed_index_for_dir.setdefault(
                requested_seed_dir, seed_index
            )
            if previous_index != seed_index:
                raise ValueError(
                    f"Seed directory {requested_seed_dir!r} mixes seed indices "
                    f"{previous_index} and {seed_index}"
                )
            previous_seed_dir = seen_seed_indices.setdefault(
                seed_index, requested_seed_dir
            )
            if previous_seed_dir != requested_seed_dir:
                raise ValueError(
                    f"Seed index {seed_index} appears under multiple directories"
                )
            if provenance is not None:
                previous_provenance = provenance_by_seed.setdefault(
                    seed_index, provenance
                )
                if previous_provenance != provenance:
                    raise ValueError(
                        f"Seed {seed_index} mixes PPO provenance across chunks"
                    )
                shared = {
                    name: provenance[name]
                    for name in PROVENANCE_FIELDS
                    if name not in {"training_seed", "train_seed_index"}
                }
                if shared_provenance is None:
                    shared_provenance = shared
                elif shared_provenance != shared:
                    raise ValueError("PPO collection seeds have mixed approvals")
            if collection_contract is not None:
                previous_contract = collection_contract_by_seed.setdefault(
                    seed_index, collection_contract
                )
                if previous_contract != collection_contract:
                    raise ValueError(
                        f"Seed {seed_index} mixes collection selection contracts "
                        "across chunks"
                    )
                if shared_collection_contract is None:
                    shared_collection_contract = collection_contract
                elif shared_collection_contract != collection_contract:
                    raise ValueError(
                        "PPO collection seeds have mixed collection selection "
                        "contracts"
                    )
            if expected_protocol_json is None:
                expected_protocol_json = protocol_json
            elif expected_protocol_json != protocol_json:
                raise ValueError(f"Mixed DMC protocols; mismatch at {path}")
            if expected_environment_protocol_json is None:
                expected_environment_protocol_json = environment_protocol_json
            elif expected_environment_protocol_json != environment_protocol_json:
                raise ValueError(
                    f"Mixed DMC environment protocols; mismatch at {path}"
                )

            global_steps = np.asarray(chunk["global_step"], dtype=np.int64)
            seen_steps = seen_global_steps_by_seed.setdefault(seed_index, set())
            chunk_steps = {int(value) for value in global_steps}
            if len(chunk_steps) != len(global_steps):
                raise ValueError(f"Chunk {path} repeats global_step values")
            repeated_steps = seen_steps & chunk_steps
            if repeated_steps:
                raise ValueError(
                    f"Seed {seed_index} repeats global_step values across chunks"
                )
            seen_steps.update(chunk_steps)

            source_files.append(str(path.resolve()))
            chunk_records = _episodes_from_chunk(
                path,
                chunk,
                source_seed_index=seed_index,
                source_seed_dir=requested_seed_dir,
                collection_total_updates=(
                    int(collection_contract["total_updates"])
                    if collection_contract is not None
                    else None
                ),
            )
            if collection_contract is not None:
                collection_transition_counts_by_seed[seed_index] = (
                    collection_transition_counts_by_seed.get(seed_index, 0)
                    + len(chunk["state"])
                )
                transition_stage_counts = (
                    collection_stage_transition_counts_by_seed.setdefault(
                        seed_index,
                        {stage: 0 for stage in COLLECTION_STAGE_NAMES},
                    )
                )
                episode_stage_counts = (
                    collection_stage_episode_counts_by_seed.setdefault(
                        seed_index,
                        {stage: 0 for stage in COLLECTION_STAGE_NAMES},
                    )
                )
                for record in chunk_records:
                    if record.collection_stage is None:
                        raise AssertionError("primary episode stage disappeared")
                    transition_stage_counts[record.collection_stage] += len(
                        record.arrays["state"]
                    )
                    episode_stage_counts[record.collection_stage] += 1
            for record in chunk_records:
                namespace_start = 1_000_000 * seed_index
                if not (
                    namespace_start
                    <= record.original_episode_id
                    < namespace_start + 1_000_000
                ):
                    raise ValueError(
                        f"Episode {record.original_episode_id} is outside seed "
                        f"{seed_index}'s id namespace"
                    )
                if record.original_episode_id in seen_episode_ids:
                    raise ValueError(
                        f"Duplicate episode_id {record.original_episode_id} in "
                        f"{seen_episode_ids[record.original_episode_id]} and {path}"
                    )
                seen_episode_ids[record.original_episode_id] = str(path)
                duplicate_id = seen_trajectory_hashes.get(record.trajectory_sha256)
                if duplicate_id is not None:
                    raise ValueError(
                        "Duplicate trajectory would leak across episode splits: "
                        f"episodes {duplicate_id} and {record.original_episode_id}"
                    )
                seen_trajectory_hashes[
                    record.trajectory_sha256
                ] = record.original_episode_id
                episodes.append(record)

    if (
        expected_protocol_json is None
        or expected_environment_protocol_json is None
        or not episodes
    ):
        raise ValueError("No episodes were loaded")
    if config.source == "ppo_training_stages" and (
        shared_provenance is None
        or shared_collection_contract is None
        or len(provenance_by_seed) != len(config.seed_dirs)
        or len(collection_contract_by_seed) != len(config.seed_dirs)
    ):
        raise ValueError(
            "Primary PPO dataset is missing complete provenance or collection "
            "selection contract"
        )
    if shared_collection_contract is not None:
        max_transitions = int(shared_collection_contract["max_transitions"])
        quota_base, quota_remainder = divmod(max_transitions, 3)
        stage_quotas = tuple(
            quota_base + (1 if index < quota_remainder else 0)
            for index in range(3)
        )
        for seed_index in sorted(collection_contract_by_seed):
            transition_count = collection_transition_counts_by_seed[seed_index]
            if transition_count > max_transitions:
                raise ValueError(
                    f"Seed {seed_index} collection has {transition_count} "
                    f"transitions, exceeding cap {max_transitions}"
                )
            stage_counts = collection_stage_transition_counts_by_seed[seed_index]
            for stage, quota in zip(COLLECTION_STAGE_NAMES, stage_quotas):
                if stage_counts[stage] > quota:
                    raise ValueError(
                        f"Seed {seed_index} {stage} collection has "
                        f"{stage_counts[stage]} transitions, exceeding stage "
                        f"quota {quota}"
                    )
    if provenance_by_seed:
        training_seeds = [
            int(provenance_by_seed[index]["training_seed"])
            for index in sorted(provenance_by_seed)
        ]
        if len(set(training_seeds)) != len(training_seeds):
            raise ValueError("PPO collection seed indices reuse a training seed")
        if config.expected_training_seeds is not None:
            expected_indices = list(range(len(config.expected_training_seeds)))
            if sorted(provenance_by_seed) != expected_indices:
                raise ValueError(
                    "PPO collection seed indices do not match the YAML profile"
                )
            if training_seeds != list(config.expected_training_seeds):
                raise ValueError(
                    "PPO collection training seeds do not match the YAML profile"
                )
            if shared_provenance is None:
                raise AssertionError("formal primary provenance disappeared")
            if (
                shared_provenance["config_fingerprint"]
                != config.expected_config_fingerprint
            ):
                raise ValueError(
                    "PPO collection config fingerprint does not match --config"
                )
            if (
                shared_provenance["approval_profile"]
                != config.expected_approval_profile
            ):
                raise ValueError(
                    "PPO collection approval profile does not match --profile"
                )
            if shared_collection_contract is None:
                raise AssertionError("formal collection contract disappeared")
            if (
                shared_collection_contract["max_transitions"]
                != config.expected_collection_max_transitions
            ):
                raise ValueError(
                    "PPO collection transition cap does not match the YAML "
                    "profile"
                )
            if (
                shared_collection_contract["total_updates"]
                != config.expected_collection_total_updates
            ):
                raise ValueError(
                    "PPO collection total updates do not match the resolved "
                    "YAML profile"
                )

    episodes.sort(
        key=lambda record: (
            record.update,
            record.source_seed_index,
            record.original_episode_id,
        )
    )
    train_episode_ids: list[int] = []
    validation_episode_ids: list[int] = []
    test_episode_ids: list[int] = []
    test_bucket = config.test_offset % config.validation_every
    validation_bucket = (test_bucket - 1) % config.validation_every
    for index, _record in enumerate(episodes):
        bucket = index % config.validation_every
        if bucket == test_bucket:
            test_episode_ids.append(index)
        elif bucket == validation_bucket:
            validation_episode_ids.append(index)
        else:
            train_episode_ids.append(index)
    if not (train_episode_ids and validation_episode_ids and test_episode_ids):
        raise ValueError(
            "Every split must contain at least one episode; collect at least "
            f"{config.validation_every} complete episodes"
        )

    primary_source = config.source == "ppo_training_stages"
    concatenated_fields = [
        name for name in TRANSITION_FIELDS if name != "episode_id"
    ]
    if primary_source:
        concatenated_fields.append("collection_stage")
    concatenated: dict[str, list[np.ndarray]] = {
        name: [] for name in concatenated_fields
    }
    remapped_episode_ids: list[np.ndarray] = []
    original_episode_ids: list[np.ndarray] = []
    source_seed_indices: list[np.ndarray] = []
    for index, record in enumerate(episodes):
        count = len(record.arrays["state"])
        for name in concatenated:
            concatenated[name].append(record.arrays[name])
        remapped_episode_ids.append(np.full(count, index, dtype=np.int64))
        original_episode_ids.append(
            np.full(count, record.original_episode_id, dtype=np.int64)
        )
        source_seed_indices.append(
            np.full(count, record.source_seed_index, dtype=np.int64)
        )

    protocol = json.loads(expected_protocol_json)
    output_dataset_schema = DATASET_SCHEMA_VERSION if primary_source else 3
    output_collection_schema = (
        ON_POLICY_COLLECTION_SCHEMA_VERSION
        if primary_source
        else STANDALONE_COLLECTION_SCHEMA_VERSION
    )
    output: dict[str, np.ndarray] = {
        name: np.concatenate(values) for name, values in concatenated.items()
    }
    output.update(
        {
            "episode_id": np.concatenate(remapped_episode_ids),
            "original_episode_id": np.concatenate(original_episode_ids),
            "source_seed_index": np.concatenate(source_seed_indices),
            "train_episode_ids": np.asarray(train_episode_ids, dtype=np.int64),
            "validation_episode_ids": np.asarray(
                validation_episode_ids, dtype=np.int64
            ),
            "test_episode_ids": np.asarray(test_episode_ids, dtype=np.int64),
            "episode_table_ids": np.arange(len(episodes), dtype=np.int64),
            "episode_trajectory_sha256": np.asarray(
                [record.trajectory_sha256 for record in episodes]
            ),
            "episode_source_files": np.asarray(
                [record.source_file for record in episodes]
            ),
            "episode_source_seed_dirs": np.asarray(
                [record.source_seed_dir for record in episodes]
            ),
            "dataset_schema_version": np.asarray(
                output_dataset_schema, dtype=np.int64
            ),
            "collection_schema_version": np.asarray(
                output_collection_schema, dtype=np.int64
            ),
            "data_source": np.asarray(config.source),
            "protocol_json": np.asarray(expected_protocol_json),
            "environment_protocol_json": np.asarray(
                expected_environment_protocol_json
            ),
            "protocol_fingerprint": np.asarray(
                protocol_fingerprint_from_json(expected_environment_protocol_json)
            ),
            "protocol_name": np.asarray(protocol["protocol_name"]),
            "control_dt": np.asarray(protocol["control_dt"], dtype=np.float64),
            "physics_dt": np.asarray(protocol["physics_dt"], dtype=np.float64),
            "time_limit": np.asarray(protocol["time_limit"], dtype=np.float64),
            "step_limit": np.asarray(protocol["step_limit"], dtype=np.int64),
            "collector_max_episode_steps": np.asarray(
                protocol["collector_max_episode_steps"], dtype=np.int64
            ),
            "collector_truncates_episodes": np.asarray(
                protocol["collector_truncates_episodes"], dtype=np.bool_
            ),
            "dm_control_version": np.asarray(protocol["dm_control_version"]),
            "mujoco_version": np.asarray(protocol["mujoco_version"]),
            "split_strategy": np.asarray(
                f"episode_stage_seed_sorted_mod_{config.validation_every}_"
                f"val_{validation_bucket}_test_{test_bucket}"
            ),
            "state_kind": np.asarray(config.task_name),
            "state_dim": np.asarray(spec.obs_dim, dtype=np.int64),
            "action_dim": np.asarray(spec.action_dim, dtype=np.int64),
            "source_files": np.asarray(source_files),
            "source_seed_dirs": np.asarray(config.seed_dirs),
        }
    )
    if primary_source:
        if shared_provenance is None or shared_collection_contract is None:
            raise AssertionError("primary provenance or contract disappeared")
        ordered_seed_indices = sorted(provenance_by_seed)
        total_updates = int(shared_collection_contract["total_updates"])
        stage_ranges = _collection_stage_ranges(total_updates)
        collection_contract_json = canonical_json(
            {
                "kind": "dmc_on_policy_stage_selection_contract",
                "schema_version": 1,
                "selection_strategy": shared_collection_contract[
                    "selection_strategy"
                ],
                "max_transitions_per_train_seed": int(
                    shared_collection_contract["max_transitions"]
                ),
                "total_updates": total_updates,
                "episode_stage_basis": "completion_update",
                "stage_update_ranges": {
                    stage: [start, end]
                    for stage, (start, end) in zip(
                        COLLECTION_STAGE_NAMES, stage_ranges
                    )
                },
            }
        )
        output.update(
            {
                **{
                    name: np.asarray(value)
                    for name, value in shared_provenance.items()
                },
                "source_train_seed_indices": np.asarray(
                    ordered_seed_indices, dtype=np.int64
                ),
                "source_training_seeds": np.asarray(
                    [
                        int(provenance_by_seed[index]["training_seed"])
                        for index in ordered_seed_indices
                    ],
                    dtype=np.int64,
                ),
                "episode_collection_stage": np.asarray(
                    [record.collection_stage for record in episodes]
                ),
                "episode_completion_update": np.asarray(
                    [record.update for record in episodes], dtype=np.int64
                ),
                "collection_selection_strategy": np.asarray(
                    shared_collection_contract["selection_strategy"]
                ),
                "collection_max_transitions": np.asarray(
                    shared_collection_contract["max_transitions"],
                    dtype=np.int64,
                ),
                "collection_total_updates": np.asarray(
                    total_updates, dtype=np.int64
                ),
                "collection_stage_names": np.asarray(COLLECTION_STAGE_NAMES),
                "collection_stage_update_ranges": np.asarray(
                    stage_ranges, dtype=np.int64
                ),
                "collection_selection_contract_json": np.asarray(
                    collection_contract_json
                ),
                "source_collection_transition_counts": np.asarray(
                    [
                        collection_transition_counts_by_seed[index]
                        for index in ordered_seed_indices
                    ],
                    dtype=np.int64,
                ),
                "source_collection_stage_transition_counts": np.asarray(
                    [
                        [
                            collection_stage_transition_counts_by_seed[index][
                                stage
                            ]
                            for stage in COLLECTION_STAGE_NAMES
                        ]
                        for index in ordered_seed_indices
                    ],
                    dtype=np.int64,
                ),
                "source_collection_stage_episode_counts": np.asarray(
                    [
                        [
                            collection_stage_episode_counts_by_seed[index][stage]
                            for stage in COLLECTION_STAGE_NAMES
                        ]
                        for index in ordered_seed_indices
                    ],
                    dtype=np.int64,
                ),
            }
        )
    for name in (
        "state",
        "requested_action",
        "action",
        "next_state",
        "reward",
        "discount",
    ):
        if not np.isfinite(output[name]).all():
            raise FloatingPointError(f"Dataset field {name!r} is non-finite")

    split_sets = [
        set(train_episode_ids),
        set(validation_episode_ids),
        set(test_episode_ids),
    ]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Episode splits overlap")
    if set.union(*split_sets) != set(range(len(episodes))):
        raise RuntimeError("Episode splits omit episodes")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output.with_name(
        config.output.stem + f".{os.getpid()}.tmp.npz"
    )
    try:
        np.savez_compressed(temporary, **output)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, config.output)
    finally:
        temporary.unlink(missing_ok=True)
    return config.output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from experiments.dmc.tasks.registry import TASK_SPECS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "complete experiment YAML; required for the primary "
            "ppo_training_stages dataset"
        ),
    )
    parser.add_argument("--profile", choices=PROFILE_NAMES, default=None)
    parser.add_argument(
        "--task",
        choices=sorted(TASK_SPECS),
        default=None,
        help="standalone_ablation only; primary task comes from --config",
    )
    parser.add_argument(
        "--source",
        choices=DATA_SOURCES,
        default=None,
        help=(
            "standalone_ablation must be explicit; primary source comes from "
            "--config"
        ),
    )
    parser.add_argument(
        "--collect-root",
        type=Path,
        default=Path("runs/dmc/data"),
        help=(
            "data root; primary chunks live under <task>/<profile>/seed_*, "
            "standalone chunks under <task>/seed_*"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "defaults beside the selected seed directories (including the "
            "primary profile namespace)"
        ),
    )
    parser.add_argument(
        "--seed-dirs",
        default=None,
        help=(
            "comma-separated required seed directories for standalone_ablation; "
            "primary seed directories come from --config and --profile"
        ),
    )
    args = parser.parse_args(argv)
    formal_primary = args.config is not None or args.profile is not None
    if formal_primary:
        if args.config is None or args.profile is None:
            parser.error("primary dataset building requires --config and --profile")
        forbidden = [
            name
            for name, value in (
                ("--task", args.task),
                ("--source", args.source),
                ("--seed-dirs", args.seed_dirs),
            )
            if value is not None
        ]
        if forbidden:
            parser.error(
                "primary task/source/seeds are YAML-bound; remove "
                + ", ".join(forbidden)
            )
    else:
        if args.source != "standalone_ablation":
            parser.error(
                "primary ppo_training_stages requires --config and --profile; "
                "without them --source must be standalone_ablation"
            )
        if args.task is None or args.seed_dirs is None:
            parser.error(
                "standalone_ablation requires explicit --task and --seed-dirs"
            )
    return args


def build_config_from_args(args: argparse.Namespace) -> BuildConfig:
    """Resolve a CLI namespace without allowing primary identity overrides."""

    if args.config is not None:
        experiment = load_experiment_config(args.config)
        execution = resolve_execution_spec(experiment, args.profile)
        task_name = experiment.task
        source = str(execution["data"]["source"])
        if source != "ppo_training_stages":
            raise ValueError(
                "Primary dataset config must declare data.source="
                "ppo_training_stages"
            )
        seed_dirs = tuple(
            f"seed_{int(run['seed'])}" for run in execution["ppo_runs"]
        )
        resolved_caps = {
            int(run["collect_max_transitions"])
            for run in execution["ppo_runs"]
        }
        yaml_cap = int(execution["data"]["max_transitions_per_train_seed"])
        if resolved_caps != {yaml_cap}:
            raise ValueError(
                "Resolved PPO collection caps do not match data configuration"
            )
        collection_total_updates = int(
            execution["data"]["collection_total_updates"]
        )
    else:
        task_name = str(args.task)
        source = "standalone_ablation"
        seed_dirs = tuple(
            item.strip() for item in str(args.seed_dirs).split(",") if item.strip()
        )
        if not seed_dirs:
            raise ValueError("--seed-dirs must contain at least one directory")

    if args.config is not None:
        collect_root = args.collect_root / task_name / str(args.profile)
        output = args.output or (
            collect_root / f"{task_name}_koopman.npz"
        )
    else:
        collect_root = args.collect_root / task_name
        output = args.output or (
            collect_root / f"{task_name}_koopman.npz"
        )
    return BuildConfig(
        task_name=task_name,
        collect_root=collect_root,
        output=output,
        seed_dirs=seed_dirs,
        source=source,
        expected_config_fingerprint=(
            experiment.fingerprint if args.config is not None else None
        ),
        expected_approval_profile=args.profile if args.config is not None else None,
        expected_training_seeds=(
            tuple(int(run["seed"]) for run in execution["ppo_runs"])
            if args.config is not None
            else None
        ),
        expected_collection_max_transitions=(
            yaml_cap if args.config is not None else None
        ),
        expected_collection_total_updates=(
            collection_total_updates if args.config is not None else None
        ),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config_from_args(args)
    path = build(config)
    print(f"[build] wrote {path}")


if __name__ == "__main__":
    main()
