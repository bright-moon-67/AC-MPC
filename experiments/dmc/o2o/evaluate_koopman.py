"""Strict common-scale evaluation of DMC Koopman exports on ExORL Proto."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.dmc.o2o.dataset import (
    _cartpole_reward,
    _episode_index_identity,
    temporal_stratified_episode_indices,
)
from experiments.dmc.reward_oracle import walker_run_exact_reward_numpy


EVALUATION_KIND = "acmpc_exorl_cartpole_koopman_test_evaluation_v1"
WALKER_EVALUATION_KIND = "acmpc_exorl_walker_run_koopman_test_evaluation_v1"
DATA_MANIFEST_KIND = "exorl_cartpole_koopman_adapter_v1"
WALKER_DATA_MANIFEST_KIND = "exorl_walker_run_koopman_adapter_v1"
MODEL_KIND = "playground_koopman_export_v1"
TASK = "CartpoleSwingup"
REWARD_IDENTITY = "dm_control_cartpole_swingup_dense_observation_oracle_v1"
STATE_NAMES = (
    "cart_position",
    "pole_cosine",
    "pole_sine",
    "cart_velocity",
    "angular_velocity",
)
STANDARD_REPORT_HORIZONS = (1, 5, 10, 20, 50)
WALKER_STATE_NAMES = tuple(
    [f"orientation_{index}" for index in range(14)]
    + ["height"]
    + [f"velocity_{index}" for index in range(9)]
)
TASK_CONFIGS = {
    "CartpoleSwingup": {
        "state_dim": 5, "action_dim": 1, "control_timestep": 0.01,
        "horizon_steps": 50, "state_names": STATE_NAMES,
        "reward": REWARD_IDENTITY, "manifest_kind": DATA_MANIFEST_KIND,
        "report_horizons": STANDARD_REPORT_HORIZONS,
    },
    "WalkerRun": {
        "state_dim": 24, "action_dim": 6, "control_timestep": 0.025,
        "horizon_steps": 20, "state_names": WALKER_STATE_NAMES,
        "reward": "dm_control_walker_run_exact_observation_oracle_v1",
        "manifest_kind": WALKER_DATA_MANIFEST_KIND,
        # Includes the playbook K=1/5/10 gate, Walker MPVE H=4, KMPC H=8,
        # an additional contact-drift diagnostic at 12, and full train K=20.
        "report_horizons": (1, 4, 5, 8, 10, 12, 20),
    },
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class EvaluationProtocol:
    """Fixed dataset and window contract; injectable only for small tests."""

    stage_ranges: tuple[tuple[str, int, int], ...]
    episode_steps: int
    horizon_steps: int
    state_dim: int = 5
    action_dim: int = 1
    split_modulus: int = 10
    train_residues: tuple[int, ...] = tuple(range(8))
    validation_residue: int = 8
    test_residue: int = 9
    rollout_discount: float = 0.99
    control_timestep: float = 0.01
    task: str = TASK
    state_names: tuple[str, ...] = STATE_NAMES

    def validate(self) -> None:
        if not self.stage_ranges:
            raise ValueError("At least one stage is required")
        cursor = 0
        names: set[str] = set()
        for name, left, right in self.stage_ranges:
            if not name or name in names:
                raise ValueError("Stage names must be non-empty and unique")
            if left != cursor or right <= left:
                raise ValueError("Stage episode ranges must be contiguous and non-empty")
            names.add(name)
            cursor = right
        if self.episode_steps < 1 or not 1 <= self.horizon_steps <= self.episode_steps:
            raise ValueError("Invalid episode or rollout horizon")
        config = TASK_CONFIGS.get(self.task)
        if config is None:
            raise ValueError(f"Unsupported Koopman evaluation task: {self.task}")
        if (
            self.state_dim != config["state_dim"]
            or self.action_dim != config["action_dim"]
            or self.state_dim != len(self.state_names)
        ):
            raise ValueError(
                f"{self.task} evaluator requires state/action dimensions "
                f"{config['state_dim']}/{config['action_dim']}"
            )
        residues = (*self.train_residues, self.validation_residue, self.test_residue)
        if len(set(residues)) != len(residues) or any(
            residue < 0 or residue >= self.split_modulus for residue in residues
        ):
            raise ValueError("Train/validation/test residues are invalid")
        if not math.isfinite(self.rollout_discount) or not 0 < self.rollout_discount <= 1:
            raise ValueError("rollout_discount must lie in (0, 1]")
        if not math.isfinite(self.control_timestep) or self.control_timestep <= 0:
            raise ValueError("control_timestep must be finite and positive")

    @property
    def episodes(self) -> int:
        return self.stage_ranges[-1][2]

    @property
    def total_transitions(self) -> int:
        return self.episodes * self.episode_steps

    @property
    def stage_counts(self) -> dict[str, int]:
        return {name: right - left for name, left, right in self.stage_ranges}

    def split_counts(self) -> dict[str, int]:
        train = validation = test = 0
        for count in self.stage_counts.values():
            residue = np.arange(count) % self.split_modulus
            train += int(np.isin(residue, self.train_residues).sum())
            validation += int(np.sum(residue == self.validation_residue))
            test += int(np.sum(residue == self.test_residue))
        return {"train": train, "validation": validation, "test": test}


PROTO1M_PROTOCOL = EvaluationProtocol(
    stage_ranges=(
        ("early", 0, 330),
        ("mid", 330, 660),
        ("late", 660, 1000),
    ),
    episode_steps=1000,
    horizon_steps=50,
)


def protocol_from_data_manifest(data_dir: Path) -> EvaluationProtocol:
    """Derive the strict stage layout from a bound adapter manifest."""

    manifest = _read_json_object(data_dir.resolve() / "manifest.json")
    stage_mapping = _require_mapping(manifest.get("stages"), field="manifest.stages")
    rows: list[tuple[str, int, int]] = []
    for name, value in stage_mapping.items():
        metadata = _require_mapping(value, field=f"manifest.stages.{name}")
        left = metadata.get("episode_id_start_inclusive")
        right = metadata.get("episode_id_end_exclusive")
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
        ):
            raise ValueError(f"Manifest stage {name!r} has invalid episode bounds")
        rows.append((name, left, right))
    rows.sort(key=lambda row: row[1])
    order = manifest.get("stage_order")
    if order is not None and order != [name for name, _left, _right in rows]:
        raise ValueError("Manifest stage_order differs from episode-bound order")
    episode_steps = manifest.get("episode_steps")
    if isinstance(episode_steps, bool) or not isinstance(episode_steps, int):
        raise ValueError("Manifest episode_steps must be an integer")
    task = manifest.get("task")
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unsupported manifest task: {task!r}")
    config = TASK_CONFIGS[task]
    return EvaluationProtocol(
        stage_ranges=tuple(rows),
        episode_steps=episode_steps,
        horizon_steps=config["horizon_steps"],
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        control_timestep=config["control_timestep"],
        task=task,
        state_names=config["state_names"],
    )


@dataclass(frozen=True)
class StageData:
    name: str
    path: Path
    sha256: str
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


@dataclass(frozen=True)
class LoadedData:
    directory: Path
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    canonical_path: Path
    canonical_sha256: str
    stages: tuple[StageData, ...]
    reference_center: np.ndarray
    reference_scale: np.ndarray
    reference_samples: int
    reward_parity_max_abs_error: float | None


@dataclass(frozen=True)
class LoadedModel:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    source_manifest_path: Path
    source_manifest_sha256: str
    matrix_a: np.ndarray
    matrix_b: np.ndarray
    matrix_c: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    encoder_weights: tuple[np.ndarray, ...]
    encoder_biases: tuple[np.ndarray, ...]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _require_saved_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    return Path(value).resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_model(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    name = name.strip()
    path = path.strip()
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Models must use non-empty NAME=PATH syntax")
    return name, Path(path)


def _validate_manifest_header(
    manifest: Mapping[str, Any], protocol: EvaluationProtocol
) -> None:
    expected = {
        "kind": TASK_CONFIGS[protocol.task]["manifest_kind"],
        "task": protocol.task,
        "total_transitions": protocol.total_transitions,
        "episodes": protocol.episodes,
        "stage_episode_counts": protocol.stage_counts,
        "episode_steps": protocol.episode_steps,
        "observation_dim": protocol.state_dim,
        "action_dim": protocol.action_dim,
        "trainer_episode_split": "per_stage_modulo_10_8_1_1",
        "trainer_split_episode_counts": protocol.split_counts(),
        "reward": TASK_CONFIGS[protocol.task]["reward"],
    }
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Proto stage manifest contract mismatch: {mismatches}")
    _require_sha256(
        manifest.get("source_episode_identity_sha256"),
        field="manifest.source_episode_identity_sha256",
    )
    if "stage_order" in manifest and manifest.get("stage_order") != [
        name for name, _left, _right in protocol.stage_ranges
    ]:
        raise ValueError("Manifest stage_order differs from the formal protocol")
    selection = manifest.get("selection")
    if selection is not None:
        if not isinstance(selection, Mapping) or selection.get("kind") != (
            "temporal_block_microstratum_start_v1"
        ):
            raise ValueError("Unsupported stratified source selection contract")
        source_total_episodes = selection.get("source_total_episodes")
        temporal_blocks = selection.get("temporal_blocks")
        selected_per_block = selection.get("selected_episodes_per_block")
        episodes_per_block = selection.get("episodes_per_block")
        microstratum_width = selection.get("microstratum_width_episodes")
        microstratum_offset = selection.get("microstratum_offset")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                source_total_episodes,
                temporal_blocks,
                selected_per_block,
                episodes_per_block,
                microstratum_width,
            )
        ) or microstratum_offset != 0:
            raise ValueError("Invalid temporal-stratified selection metadata")
        if source_total_episodes != temporal_blocks * episodes_per_block:
            raise ValueError("Selection block sizes do not cover the source episodes")
        if episodes_per_block != selected_per_block * microstratum_width:
            raise ValueError("Selection micro-strata do not cover each temporal block")
        expected_indices = temporal_stratified_episode_indices(
            source_total_episodes=source_total_episodes,
            temporal_deciles=temporal_blocks,
            episodes_per_decile=selected_per_block,
        )
        actual_indices = manifest.get("source_episode_indices")
        if actual_indices != list(expected_indices):
            raise ValueError("Manifest episode IDs differ from its stratified selection")
        expected_identity = _episode_index_identity(expected_indices)
        if manifest.get("source_episode_indices_sha256") != expected_identity:
            raise ValueError("Manifest stratified episode-ID SHA256 differs")
        expected_stage_counts = {
            f"decile_{index:02d}": selected_per_block
            for index in range(temporal_blocks)
        }
        if protocol.stage_counts != expected_stage_counts:
            raise ValueError("Protocol stage counts differ from stratified selection")


def _validate_stage_metadata(
    metadata: Mapping[str, Any],
    *,
    path: Path,
    checksum: str,
    left: int,
    right: int,
    protocol: EvaluationProtocol,
    source_episode_indices: list[int] | None = None,
) -> None:
    episodes = right - left
    expected = {
        "path": str(path),
        "sha256": checksum,
        "episode_id_start_inclusive": left,
        "episode_id_end_exclusive": right,
        "episodes": episodes,
        "states_shape": [episodes, protocol.episode_steps + 1, protocol.state_dim],
        "actions_shape": [episodes, protocol.episode_steps, protocol.action_dim],
        "rewards_shape": [episodes, protocol.episode_steps],
    }
    if source_episode_indices is not None:
        expected.update(
            source_episode_indices=source_episode_indices,
            source_episode_index_first=source_episode_indices[0],
            source_episode_index_last=source_episode_indices[-1],
        )
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stage metadata mismatch for {path.name}: {mismatches}")


def _load_stage(
    path: Path,
    *,
    expected_episodes: int,
    protocol: EvaluationProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected_keys = {"states", "actions", "rewards"}
            if set(archive.files) != expected_keys:
                raise ValueError(
                    f"{path.name} keys differ: {archive.files}, expected {sorted(expected_keys)}"
                )
            states = np.asarray(archive["states"])
            actions = np.asarray(archive["actions"])
            rewards = np.asarray(archive["rewards"])
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(path.name):
            raise
        raise ValueError(f"Could not load strict stage archive {path}") from exc
    expected_shapes = (
        (expected_episodes, protocol.episode_steps + 1, protocol.state_dim),
        (expected_episodes, protocol.episode_steps, protocol.action_dim),
        (expected_episodes, protocol.episode_steps),
    )
    for name, array, shape in zip(
        ("states", "actions", "rewards"),
        (states, actions, rewards),
        expected_shapes,
        strict=True,
    ):
        if array.shape != shape:
            raise ValueError(f"{path.name} {name} shape {array.shape} != {shape}")
        if array.dtype != np.float32:
            raise ValueError(f"{path.name} {name} must have dtype float32")
        if not np.isfinite(array).all():
            raise FloatingPointError(f"{path.name} {name} contains NaN or Inf")
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        raise ValueError(f"{path.name} contains actions outside [-1, 1]")
    if np.any(rewards < 0.0) or np.any(rewards > 1.0):
        raise ValueError(f"{path.name} contains rewards outside [0, 1]")
    return states, actions, rewards


def load_proto_data(
    data_dir: Path,
    *,
    protocol: EvaluationProtocol = PROTO1M_PROTOCOL,
) -> LoadedData:
    """Load and fail-closed validate the canonical staged Proto dataset."""

    protocol.validate()
    data_dir = data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = _read_json_object(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    _validate_manifest_header(manifest, protocol)

    canonical_path = _require_saved_path(
        manifest.get("canonical_transitions_npz"),
        field="manifest.canonical_transitions_npz",
    )
    source_path = _require_saved_path(
        manifest.get("source_dataset"), field="manifest.source_dataset"
    )
    if canonical_path != source_path:
        raise ValueError("Manifest canonical and source dataset paths differ")
    if not canonical_path.is_file():
        raise FileNotFoundError(f"Canonical transition dataset is missing: {canonical_path}")
    canonical_sha256 = file_sha256(canonical_path)
    for field in ("canonical_transitions_npz_sha256", "source_dataset_sha256"):
        expected_sha = _require_sha256(manifest.get(field), field=f"manifest.{field}")
        if expected_sha != canonical_sha256:
            raise ValueError(f"Canonical transition SHA256 differs from manifest.{field}")

    manifest_stages = _require_mapping(manifest.get("stages"), field="manifest.stages")
    expected_stage_names = {name for name, _left, _right in protocol.stage_ranges}
    if set(manifest_stages) != expected_stage_names:
        raise ValueError("Manifest stage names differ from the formal protocol")
    stages: list[StageData] = []
    reward_parity_max: float | None = None
    reward_source = manifest.get("reward_source", "oracle")
    if reward_source not in {"oracle", "recorded", "zero"}:
        raise ValueError(f"Unsupported staged reward source: {reward_source!r}")
    training_states: list[np.ndarray] = []
    all_source_episode_indices = manifest.get("source_episode_indices")
    for name, left, right in protocol.stage_ranges:
        path = (data_dir / f"{name}.npz").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Stage archive is missing: {path}")
        checksum = file_sha256(path)
        metadata = _require_mapping(
            manifest_stages.get(name), field=f"manifest.stages.{name}"
        )
        _validate_stage_metadata(
            metadata,
            path=path,
            checksum=checksum,
            left=left,
            right=right,
            protocol=protocol,
            source_episode_indices=(
                all_source_episode_indices[left:right]
                if isinstance(all_source_episode_indices, list)
                else None
            ),
        )
        states, actions, rewards = _load_stage(
            path,
            expected_episodes=right - left,
            protocol=protocol,
        )
        if reward_source == "oracle":
            exact_reward = (
                _cartpole_reward(states[:, 1:], actions)
                if protocol.task == "CartpoleSwingup"
                else walker_run_exact_reward_numpy(states[:, 1:], actions)
            )
            stage_reward_error = float(
                np.max(
                    np.abs(
                        exact_reward.astype(np.float64)
                        - rewards.astype(np.float64)
                    )
                )
            )
            reward_parity_max = max(reward_parity_max or 0.0, stage_reward_error)
            if stage_reward_error > 2e-7:
                raise ValueError(
                    f"{name} stored reward differs from the exact {protocol.task} oracle: "
                    f"max_abs_error={stage_reward_error}"
                )
        local_episode = np.arange(right - left)
        train_mask = np.isin(local_episode % protocol.split_modulus, protocol.train_residues)
        training_states.append(states[train_mask, :-1].reshape(-1, protocol.state_dim))
        stages.append(
            StageData(
                name=name,
                path=path,
                sha256=checksum,
                states=states,
                actions=actions,
                rewards=rewards,
            )
        )

    reference = np.concatenate(training_states, axis=0)
    reference_center = np.mean(reference, axis=0, dtype=np.float64)
    reference_scale = np.std(reference, axis=0, dtype=np.float64)
    reference_scale = np.maximum(reference_scale, 1e-6)
    if not np.isfinite(reference_center).all() or not np.isfinite(reference_scale).all():
        raise FloatingPointError("Common train reference statistics are non-finite")
    expected_reference_samples = protocol.split_counts()["train"] * protocol.episode_steps
    if reference.shape != (expected_reference_samples, protocol.state_dim):
        raise AssertionError("Common train reference sample count drifted")
    return LoadedData(
        directory=data_dir,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        canonical_path=canonical_path,
        canonical_sha256=canonical_sha256,
        stages=tuple(stages),
        reference_center=reference_center,
        reference_scale=reference_scale,
        reference_samples=expected_reference_samples,
        reward_parity_max_abs_error=reward_parity_max,
    )


def _metadata_from_archive(archive: Any, path: Path) -> dict[str, Any]:
    if "metadata_json" not in archive.files:
        raise ValueError(f"{path} is missing metadata_json")
    encoded = np.asarray(archive["metadata_json"])
    if encoded.shape != ():
        raise ValueError(f"{path} metadata_json must be a scalar")
    value = encoded.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{path} metadata_json must encode text")
    try:
        metadata = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} metadata_json is invalid") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} metadata_json must encode an object")
    return metadata


def _archive_array(archive: Any, key: str, path: Path) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"{path} is missing array {key!r}")
    array = np.asarray(archive[key], dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{path} array {key!r} contains NaN or Inf")
    return array


def _validate_mlp_stack(
    archive: Any,
    *,
    prefix: str,
    layer_count: int,
    input_dim: int,
    final_dim: int,
    path: Path,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if layer_count < 1:
        raise ValueError(f"{path} {prefix}_layer_count must be positive")
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    preceding = input_dim
    for index in range(layer_count):
        weight = _archive_array(archive, f"{prefix}_{index}_weight", path)
        bias = _archive_array(archive, f"{prefix}_{index}_bias", path)
        if weight.ndim != 2 or weight.shape[1] != preceding:
            raise ValueError(f"{path} {prefix} layer {index} has an invalid weight shape")
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"{path} {prefix} layer {index} has an invalid bias shape")
        preceding = weight.shape[0]
        weights.append(weight)
        biases.append(bias)
    if preceding != final_dim:
        raise ValueError(f"{path} {prefix} final dimension {preceding} != {final_dim}")
    return tuple(weights), tuple(biases)


def load_model(
    path: Path,
    *,
    protocol: EvaluationProtocol = PROTO1M_PROTOCOL,
) -> LoadedModel:
    """Load a framework-neutral export and verify its source-manifest lineage."""

    protocol.validate()
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Koopman model does not exist: {path}")
    checksum = file_sha256(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = _metadata_from_archive(archive, path)
            if metadata.get("kind") != MODEL_KIND or metadata.get("task") != protocol.task:
                raise ValueError(f"{path} is not a {protocol.task} Playground Koopman export")
            architecture = _require_mapping(
                metadata.get("architecture"), field=f"{path}.metadata.architecture"
            )
            expected_architecture = {
                "architecture": "fullA_history_v2_adapted",
                "state_dim": protocol.state_dim,
                "action_dim": protocol.action_dim,
                "activation": "silu",
            }
            if any(architecture.get(key) != value for key, value in expected_architecture.items()):
                raise ValueError(
                    f"{path} architecture is incompatible with {protocol.task} O2O"
                )
            lift_dim = architecture.get("lift_dim")
            if isinstance(lift_dim, bool) or not isinstance(lift_dim, int) or lift_dim < 1:
                raise ValueError(f"{path} architecture lift_dim must be positive")
            if metadata.get("k_step") != protocol.horizon_steps:
                raise ValueError(
                    f"{path} was not trained with K={protocol.horizon_steps}"
                )
            encoder_count = metadata.get("encoder_layer_count")
            if isinstance(encoder_count, bool) or not isinstance(encoder_count, int):
                raise ValueError(f"{path} encoder_layer_count must be an integer")
            lifted_dim = protocol.state_dim + lift_dim
            matrix_a = _archive_array(archive, "A", path)
            matrix_b = _archive_array(archive, "B", path)
            matrix_c = _archive_array(archive, "C", path)
            center = _archive_array(archive, "center", path)
            scale = _archive_array(archive, "scale", path)
            expected_shapes = {
                "A": (lifted_dim, lifted_dim),
                "B": (lifted_dim, protocol.action_dim),
                "C": (protocol.state_dim, lifted_dim),
                "center": (protocol.state_dim,),
                "scale": (protocol.state_dim,),
            }
            actual_shapes = {
                "A": matrix_a.shape,
                "B": matrix_b.shape,
                "C": matrix_c.shape,
                "center": center.shape,
                "scale": scale.shape,
            }
            if actual_shapes != expected_shapes:
                raise ValueError(
                    f"{path} Koopman array shapes differ: {actual_shapes} != {expected_shapes}"
                )
            if np.any(scale <= 0):
                raise ValueError(f"{path} normalization scale must be positive")
            encoder_weights, encoder_biases = _validate_mlp_stack(
                archive,
                prefix="encoder",
                layer_count=encoder_count,
                input_dim=protocol.state_dim,
                final_dim=lift_dim,
                path=path,
            )
            reward_count = metadata.get("reward_layer_count", 0)
            if (
                isinstance(reward_count, bool)
                or not isinstance(reward_count, int)
                or reward_count < 0
            ):
                raise ValueError(f"{path} reward_layer_count must be non-negative")
            if reward_count:
                _validate_mlp_stack(
                    archive,
                    prefix="reward",
                    layer_count=reward_count,
                    input_dim=2 * protocol.state_dim + protocol.action_dim,
                    final_dim=1,
                    path=path,
                )
    except (OSError, ValueError) as exc:
        if isinstance(exc, (ValueError, FloatingPointError)):
            raise
        raise ValueError(f"Could not load Koopman export {path}") from exc

    source_directory = _require_saved_path(
        metadata.get("source_path"), field=f"{path}.metadata.source_path"
    )
    source_manifest_path = source_directory / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Model source manifest is unavailable: {source_manifest_path}"
        )
    source_manifest_sha256 = file_sha256(source_manifest_path)
    for field in ("source_sha256", "dataset_sha256", "data_manifest_sha256"):
        expected = _require_sha256(
            metadata.get(field), field=f"{path}.metadata.{field}"
        )
        if expected != source_manifest_sha256:
            raise ValueError(f"{path} source manifest SHA differs from metadata.{field}")
    return LoadedModel(
        path=path,
        sha256=checksum,
        metadata=metadata,
        source_manifest_path=source_manifest_path.resolve(),
        source_manifest_sha256=source_manifest_sha256,
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        matrix_c=matrix_c,
        center=center,
        scale=scale,
        encoder_weights=encoder_weights,
        encoder_biases=encoder_biases,
    )


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def _lift(model: LoadedModel, observation: np.ndarray) -> np.ndarray:
    normalized = (observation - model.center) / model.scale
    encoded = normalized
    for index, (weight, bias) in enumerate(
        zip(model.encoder_weights, model.encoder_biases, strict=True)
    ):
        encoded = encoded @ weight.T + bias
        if index + 1 < len(model.encoder_weights):
            encoded = _silu(encoded)
    lifted = np.concatenate((normalized, encoded), axis=-1)
    if not np.isfinite(lifted).all():
        raise FloatingPointError(f"{model.path} produced a non-finite lifted state")
    return lifted


def _empty_metrics(state_dim: int) -> dict[str, Any]:
    return {
        "windows": 0,
        "weighted_sse": 0.0,
        "weighted_hold_sse": 0.0,
        "one_step_sse": 0.0,
        "final_step_sse": 0.0,
        "physical_sse": np.zeros(state_dim, dtype=np.float64),
        "reward_weighted_sse": 0.0,
        "reward_sse": 0.0,
        "reward_absolute_error": 0.0,
        "reward_signed_error": 0.0,
        "reward_max_abs_error": 0.0,
        "reward_one_step_sse": 0.0,
        "reward_final_step_sse": 0.0,
    }


def _update_metrics(
    metrics: dict[str, Any],
    *,
    normalized_error_square: np.ndarray,
    normalized_hold_error_square: np.ndarray,
    physical_error_square: np.ndarray,
    reward_error: np.ndarray,
    weights: np.ndarray,
) -> None:
    batch_windows = normalized_error_square.shape[0]
    metrics["windows"] += batch_windows
    metrics["weighted_sse"] += float(
        np.sum(normalized_error_square * weights[None, :, None], dtype=np.float64)
    )
    metrics["weighted_hold_sse"] += float(
        np.sum(
            normalized_hold_error_square * weights[None, :, None],
            dtype=np.float64,
        )
    )
    metrics["one_step_sse"] += float(
        np.sum(normalized_error_square[:, 0], dtype=np.float64)
    )
    metrics["final_step_sse"] += float(
        np.sum(normalized_error_square[:, -1], dtype=np.float64)
    )
    metrics["physical_sse"] += np.sum(
        physical_error_square, axis=(0, 1), dtype=np.float64
    )
    reward_square = reward_error * reward_error
    metrics["reward_weighted_sse"] += float(
        np.sum(reward_square * weights[None, :], dtype=np.float64)
    )
    metrics["reward_sse"] += float(np.sum(reward_square, dtype=np.float64))
    metrics["reward_absolute_error"] += float(
        np.sum(np.abs(reward_error), dtype=np.float64)
    )
    metrics["reward_signed_error"] += float(np.sum(reward_error, dtype=np.float64))
    metrics["reward_max_abs_error"] = max(
        metrics["reward_max_abs_error"], float(np.max(np.abs(reward_error)))
    )
    metrics["reward_one_step_sse"] += float(
        np.sum(reward_square[:, 0], dtype=np.float64)
    )
    metrics["reward_final_step_sse"] += float(
        np.sum(reward_square[:, -1], dtype=np.float64)
    )


def _finalize_metrics(
    metrics: Mapping[str, Any],
    *,
    protocol: EvaluationProtocol,
    horizon_steps: int,
    weights: np.ndarray,
) -> dict[str, Any]:
    windows = int(metrics["windows"])
    if windows < 1:
        raise ValueError("Cannot finalize an empty evaluation")
    weighted_denominator = windows * protocol.state_dim * float(np.sum(weights))
    weighted_nmse = float(metrics["weighted_sse"]) / weighted_denominator
    hold_nmse = float(metrics["weighted_hold_sse"]) / weighted_denominator
    hold_ratio = weighted_nmse / hold_nmse if hold_nmse > 0 else None
    physical_rmse = np.sqrt(
        np.asarray(metrics["physical_sse"], dtype=np.float64)
        / (windows * horizon_steps)
    )
    reward_predictions = windows * horizon_steps
    reward_weighted_mse = float(metrics["reward_weighted_sse"]) / (
        windows * float(np.sum(weights))
    )
    reward_mse = float(metrics["reward_sse"]) / reward_predictions
    return {
        "windows": windows,
        "weighted_rollout_nmse": weighted_nmse,
        "weighted_rollout_nrmse": math.sqrt(weighted_nmse),
        "hold_weighted_rollout_nmse": hold_nmse,
        "model_to_hold_mse_ratio": hold_ratio,
        "one_step_nmse": float(metrics["one_step_sse"])
        / (windows * protocol.state_dim),
        f"step_{horizon_steps}_nmse": float(metrics["final_step_sse"])
        / (windows * protocol.state_dim),
        "physical_rmse_over_rollout": {
            "overall": math.sqrt(
                float(np.sum(metrics["physical_sse"]))
                / (windows * horizon_steps * protocol.state_dim)
            ),
            "by_dimension": {
                name: float(value)
        for name, value in zip(protocol.state_names, physical_rmse, strict=True)
            },
        },
        "exact_reward_prediction": {
            "contract": (
                "oracle(predicted_next_observation, dataset_action)"
                "_vs_oracle(true_next_observation, dataset_action)"
            ),
            "uses_learned_reward_model": False,
            "predictions": reward_predictions,
            "weighted_mse": reward_weighted_mse,
            "weighted_rmse": math.sqrt(reward_weighted_mse),
            "mse_over_rollout": reward_mse,
            "rmse_over_rollout": math.sqrt(reward_mse),
            "mae_over_rollout": float(metrics["reward_absolute_error"])
            / reward_predictions,
            "bias_over_rollout": float(metrics["reward_signed_error"])
            / reward_predictions,
            "max_abs_error": float(metrics["reward_max_abs_error"]),
            "one_step_rmse": math.sqrt(
                float(metrics["reward_one_step_sse"]) / windows
            ),
            f"step_{horizon_steps}_rmse": math.sqrt(
                float(metrics["reward_final_step_sse"]) / windows
            ),
        },
    }


def _evaluate_model(
    model: LoadedModel,
    data: LoadedData,
    *,
    protocol: EvaluationProtocol,
    batch_size: int,
) -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], dict[str, int], dict[str, int]
]:
    full_weights = protocol.rollout_discount ** np.arange(
        protocol.horizon_steps, dtype=np.float64
    )
    report_horizons = tuple(
        horizon
        for horizon in TASK_CONFIGS[protocol.task]["report_horizons"]
        if horizon <= protocol.horizon_steps
    )
    if protocol.horizon_steps not in report_horizons:
        report_horizons = (*report_horizons, protocol.horizon_steps)
    accumulated = {
        horizon: _empty_metrics(protocol.state_dim) for horizon in report_horizons
    }
    accumulated_by_stage = {
        stage.name: {
            horizon: _empty_metrics(protocol.state_dim) for horizon in report_horizons
        }
        for stage in data.stages
    }
    test_episodes_by_stage: dict[str, int] = {}
    windows_by_stage: dict[str, int] = {}
    offsets = np.arange(protocol.horizon_steps, dtype=np.int64)
    starts = np.arange(
        protocol.episode_steps - protocol.horizon_steps + 1, dtype=np.int64
    )
    for stage in data.stages:
        local_episode = np.arange(stage.states.shape[0], dtype=np.int64)
        test_episode = local_episode[
            local_episode % protocol.split_modulus == protocol.test_residue
        ]
        pairs = np.stack(
            np.meshgrid(test_episode, starts, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        test_episodes_by_stage[stage.name] = int(test_episode.size)
        windows_by_stage[stage.name] = int(pairs.shape[0])
        for lower in range(0, pairs.shape[0], batch_size):
            pair = pairs[lower : lower + batch_size]
            episode_index = pair[:, 0]
            start_index = pair[:, 1]
            time_index = start_index[:, None] + offsets[None, :]
            actions = stage.actions[episode_index[:, None], time_index].astype(
                np.float64, copy=False
            )
            true_state = stage.states[
                episode_index[:, None], time_index + 1
            ].astype(np.float64, copy=False)
            true_state_for_reward = true_state
            true_reward = (
                _cartpole_reward(true_state_for_reward, actions)
                if protocol.task == "CartpoleSwingup"
                else walker_run_exact_reward_numpy(true_state_for_reward, actions)
            ).astype(np.float64, copy=False)
            initial = stage.states[episode_index, start_index].astype(
                np.float64, copy=False
            )
            lifted = _lift(model, initial)
            predicted = np.empty_like(true_state)
            for offset in range(protocol.horizon_steps):
                lifted = lifted @ model.matrix_a.T + actions[:, offset] @ model.matrix_b.T
                normalized_state = lifted @ model.matrix_c.T
                predicted[:, offset] = normalized_state * model.scale + model.center
            if not np.isfinite(predicted).all():
                raise FloatingPointError(
                    f"{model.path} produced NaN or Inf in a {protocol.horizon_steps}-step rollout"
                )
            physical_error_square = (predicted - true_state) ** 2
            normalized_error_square = physical_error_square / (
                data.reference_scale[None, None, :] ** 2
            )
            normalized_hold_error_square = (
                (initial[:, None, :] - true_state)
                / data.reference_scale[None, None, :]
            ) ** 2
            predicted_reward = (
                _cartpole_reward(predicted, actions)
                if protocol.task == "CartpoleSwingup"
                else walker_run_exact_reward_numpy(predicted, actions)
            ).astype(np.float64, copy=False)
            reward_error = predicted_reward - true_reward
            for horizon in report_horizons:
                prefix = slice(0, horizon)
                _update_metrics(
                    accumulated[horizon],
                    normalized_error_square=normalized_error_square[:, prefix],
                    normalized_hold_error_square=normalized_hold_error_square[:, prefix],
                    physical_error_square=physical_error_square[:, prefix],
                    reward_error=reward_error[:, prefix],
                    weights=full_weights[prefix],
                )
                _update_metrics(
                    accumulated_by_stage[stage.name][horizon],
                    normalized_error_square=normalized_error_square[:, prefix],
                    normalized_hold_error_square=normalized_hold_error_square[:, prefix],
                    physical_error_square=physical_error_square[:, prefix],
                    reward_error=reward_error[:, prefix],
                    weights=full_weights[prefix],
                )
    return (
        {
            str(horizon): _finalize_metrics(
                accumulated[horizon],
                protocol=protocol,
                horizon_steps=horizon,
                weights=full_weights[:horizon],
            )
            for horizon in report_horizons
        },
        {
            stage.name: {
                str(horizon): _finalize_metrics(
                    accumulated_by_stage[stage.name][horizon],
                    protocol=protocol,
                    horizon_steps=horizon,
                    weights=full_weights[:horizon],
                )
                for horizon in report_horizons
            }
            for stage in data.stages
        },
        test_episodes_by_stage,
        windows_by_stage,
    )


def evaluate_models(
    data_dir: Path,
    named_models: Sequence[tuple[str, Path]],
    *,
    batch_size: int = 2048,
    protocol: EvaluationProtocol | None = None,
) -> dict[str, Any]:
    """Evaluate one or more exports on every legal test-split rollout window."""

    if protocol is None:
        protocol = protocol_from_data_manifest(data_dir)
    protocol.validate()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not named_models:
        raise ValueError("At least one NAME=MODEL pair is required")
    names = [name for name, _path in named_models]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Model names must be non-empty and unique")
    data = load_proto_data(data_dir, protocol=protocol)
    models = {
        name: load_model(path, protocol=protocol) for name, path in named_models
    }
    result_models: dict[str, Any] = {}
    common_test_counts: dict[str, int] | None = None
    common_window_counts: dict[str, int] | None = None
    for name, model in models.items():
        (
            metrics_by_horizon,
            metrics_by_stage_and_horizon,
            test_counts,
            window_counts,
        ) = _evaluate_model(
            model,
            data,
            protocol=protocol,
            batch_size=batch_size,
        )
        if common_test_counts is None:
            common_test_counts = test_counts
            common_window_counts = window_counts
        elif test_counts != common_test_counts or window_counts != common_window_counts:
            raise AssertionError("Models did not use an identical test window set")
        architecture = dict(model.metadata["architecture"])
        result_models[name] = {
            "path": str(model.path),
            "sha256": model.sha256,
            "architecture": architecture,
            "training": {
                "k_step": model.metadata["k_step"],
                "seed": model.metadata.get("seed"),
                "best_epoch": model.metadata.get("best_epoch"),
                "reported_best_validation_rollout_normalized_mse": model.metadata.get(
                    "best_validation_rollout_normalized_mse"
                ),
                "source_path": str(model.source_manifest_path.parent),
                "source_manifest_path": str(model.source_manifest_path),
                "source_manifest_sha256": model.source_manifest_sha256,
                "source_is_evaluation_proto1m": (
                    model.source_manifest_sha256 == data.manifest_sha256
                ),
            },
            "metrics": metrics_by_horizon[str(protocol.horizon_steps)],
            "metrics_by_horizon": metrics_by_horizon,
            "metrics_by_stage": {
                stage: values[str(protocol.horizon_steps)]
                for stage, values in metrics_by_stage_and_horizon.items()
            },
            "metrics_by_stage_and_horizon": metrics_by_stage_and_horizon,
        }
    assert common_test_counts is not None and common_window_counts is not None
    expected_test_episodes = protocol.split_counts()["test"]
    expected_windows = expected_test_episodes * (
        protocol.episode_steps - protocol.horizon_steps + 1
    )
    if sum(common_test_counts.values()) != expected_test_episodes:
        raise AssertionError("Test episode count drifted")
    if sum(common_window_counts.values()) != expected_windows:
        raise AssertionError("Episode-safe window count drifted")
    return {
        "kind": (
            EVALUATION_KIND
            if protocol.task == "CartpoleSwingup"
            else WALKER_EVALUATION_KIND
        ),
        "task": protocol.task,
        "device": "cpu_numpy_float64",
        "protocol": {
            "split": "per_stage_local_episode_index_mod_10_equals_9",
            "window_selection": "all_episode_safe_starts_inclusive",
            "horizon_steps": protocol.horizon_steps,
            "horizon_seconds": protocol.horizon_steps * protocol.control_timestep,
            "reported_prefix_horizons_steps": [
                int(horizon) for horizon in result_models[next(iter(result_models))][
                    "metrics_by_horizon"
                ]
            ],
            "prefix_horizon_window_contract": (
                "all prefixes use the identical episode-safe H=50 window starts"
                if protocol.horizon_steps == 50
                else "all prefixes use the identical maximum-horizon window starts"
            ),
            "rollout_discount": protocol.rollout_discount,
            "weighted_nmse_reference": "shared_Proto_train_split_per_dimension_std",
            "test_episodes": expected_test_episodes,
            "test_episodes_by_stage": common_test_counts,
            "windows": expected_windows,
            "windows_per_test_episode": (
                protocol.episode_steps - protocol.horizon_steps + 1
            ),
            "windows_by_stage": common_window_counts,
            "batch_size_execution_only": batch_size,
            "exact_reward_prediction": (
                "official_oracle(predicted_next_observation,dataset_action); "
                "learned reward head is not evaluated"
            ),
        },
        "data": {
            "directory": str(data.directory),
            "manifest_path": str(data.manifest_path),
            "manifest_sha256": data.manifest_sha256,
            "canonical_transitions_npz": str(data.canonical_path),
            "canonical_transitions_npz_sha256": data.canonical_sha256,
            "source_episode_identity_sha256": data.manifest[
                "source_episode_identity_sha256"
            ],
            "stage_sha256": {stage.name: stage.sha256 for stage in data.stages},
            "stored_reward_source": data.manifest.get("reward_source", "oracle"),
            "stored_reward_oracle_parity_max_abs_error": (
                data.reward_parity_max_abs_error
            ),
        },
        "common_train_reference": {
            "split": "per_stage_local_episode_index_mod_10_in_0_to_7",
            "samples": data.reference_samples,
            "center": data.reference_center.tolist(),
            "scale": data.reference_scale.tolist(),
            "minimum_scale": 1e-6,
        },
        "models": result_models,
        "finished_unix_seconds": time.time(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    model_paths = [path.resolve() for _name, path in args.model]
    if output in model_paths:
        raise ValueError("--output must not overwrite an input model")
    requested_task = getattr(args, "task", None)
    if requested_task is None:
        result = evaluate_models(
            args.data_dir, args.model, batch_size=args.batch_size
        )
    else:
        protocol = protocol_from_data_manifest(args.data_dir)
        if requested_task != protocol.task:
            raise ValueError(
                f"--task {requested_task} does not match manifest task {protocol.task}"
            )
        result = evaluate_models(
            args.data_dir, args.model, batch_size=args.batch_size, protocol=protocol
        )
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Prepared ExORL Proto1M stage directory",
    )
    parser.add_argument(
        "--model",
        type=parse_model,
        action="append",
        required=True,
        metavar="NAME=MODEL",
        help="Named Koopman export; repeat to compare models",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--task", choices=tuple(TASK_CONFIGS), default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
