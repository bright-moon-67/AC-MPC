"""Train SAC/RLPD/Cal-RLPD and AC-KMPC offline-to-online on DMC."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.dmc.o2o.checkpoint import (
    CHECKPOINT_KIND,
    atomic_torch_save,
    load_checkpoint,
    restore_rng,
    rng_state,
)
from experiments.dmc.o2o.config import O2OConfig, SUPPORTED_O2O_TASKS, TRAIN_METHODS
from experiments.dmc.o2o.dataset import (
    OfflineDataset,
    OnlineReplay,
    mark_offline,
    mixed_batch,
)
from experiments.dmc.o2o.koopman import FrozenKoopman, file_sha256
from experiments.dmc.o2o.learner import O2OLearner, TensorBatch
from experiments.dmc.o2o.networks import FrozenObservationNormalizer
from experiments.dmc.tasks.adapter import make_dmc_adapter
from experiments.dmc.tasks.registry import get_task_spec


RECORDED_REWARD_O2O_TASKS = frozenset({"hopper_stand", "hopper_hop"})


DIAGNOSTIC_EVAL_SEED_BASE = 9_000_000
from experiments.dmc.ppo.vector_env import make_dmc_vector_env


def _dataset_action_repeat(task: str, dataset: OfflineDataset) -> int | None:
    """Resolve an explicit outer-rate contract for recorded Hopper data."""

    if task != "hopper_hop":
        return None
    expected = {
        "action_repeat": 2,
        "control_dt": 0.04,
        "transitions_per_episode": 500,
    }
    mismatches = {
        key: {"dataset": dataset.metadata.get(key), "expected": value}
        for key, value in expected.items()
        if dataset.metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Hopper Hop O2O requires the TD-MPC2 AR2 dataset contract: "
            f"{mismatches}"
        )
    return 2


def _validate_dataset_environment_protocol(
    task: str,
    dataset: OfflineDataset,
    protocol: dict[str, Any],
) -> None:
    """Fail before training if recorded transitions and live timing differ."""

    if task != "hopper_hop":
        return
    expected = {
        "protocol_name": "tdmpc2_action_repeat2_v1",
        "action_repeat": 2,
        "control_dt": 0.04,
        "step_limit": 500,
    }
    mismatches = {
        key: {"runtime": protocol.get(key), "expected": value}
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if protocol.get("control_dt") != dataset.metadata.get("control_dt"):
        mismatches["dataset_control_dt"] = {
            "runtime": protocol.get("control_dt"),
            "dataset": dataset.metadata.get("control_dt"),
        }
    if protocol.get("action_repeat") != dataset.metadata.get("action_repeat"):
        mismatches["dataset_action_repeat"] = {
            "runtime": protocol.get("action_repeat"),
            "dataset": dataset.metadata.get("action_repeat"),
        }
    if mismatches:
        raise ValueError(
            "Hopper Hop dataset/environment protocol mismatch: "
            f"{mismatches}"
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


_PENDING_KEYS = (
    "observation", "action", "reward", "discount", "next_observation"
)


def _pending_trajectory_state(
    pending: dict[str, list[np.ndarray | float]] | None,
) -> dict[str, Any]:
    """Checkpoint a Cal-QL trajectory without inventing unfinished RTGs."""

    if pending is None:
        return {"kind": "calql_pending_trajectory_v1", "count": 0, "arrays": {}}
    lengths = {key: len(pending[key]) for key in _PENDING_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError("Pending Cal-QL trajectory field lengths disagree")
    count = next(iter(lengths.values()))
    if count == 0:
        return {"kind": "calql_pending_trajectory_v1", "count": 0, "arrays": {}}
    arrays = {
        "observation": np.asarray(pending["observation"], dtype=np.float32),
        "action": np.asarray(pending["action"], dtype=np.float32),
        "reward": np.asarray(pending["reward"], dtype=np.float32),
        "discount": np.asarray(pending["discount"], dtype=np.float32),
        "next_observation": np.asarray(
            pending["next_observation"], dtype=np.float32
        ),
    }
    if count and not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Pending Cal-QL trajectory is non-finite")
    return {
        "kind": "calql_pending_trajectory_v1",
        "count": count,
        "arrays": arrays,
    }


def _has_metric_row(path: Path, *, phase: str, offline_update: int, online_step: int) -> bool:
    if not path.is_file():
        return False
    matches = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in metrics.jsonl line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"metrics.jsonl line {line_number} is not an object")
        if (
            row.get("phase") == phase
            and row.get("offline_update") == offline_update
            and row.get("online_step") == online_step
        ):
            matches += 1
    if matches > 1:
        raise ValueError(
            f"metrics.jsonl repeats {phase!r} at offline={offline_update}, online={online_step}"
        )
    return matches == 1


def _read_metric_row(
    path: Path, *, phase: str, offline_update: int, online_step: int
) -> dict[str, Any]:
    """Return the unique durable metric row for a completed milestone."""

    rows = []
    if path.is_file():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in metrics.jsonl line {line_number}"
                ) from exc
            if (
                isinstance(row, dict)
                and row.get("phase") == phase
                and row.get("offline_update") == offline_update
                and row.get("online_step") == online_step
            ):
                rows.append(row)
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one {phase!r} metric at "
            f"offline={offline_update}, online={online_step}; got {len(rows)}"
        )
    return rows[0]


def _checkpoint_counter(payload: dict[str, Any], key: str) -> int:
    """Return a strict non-negative checkpoint counter (bools are not ints here)."""

    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Checkpoint counter {key!r} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"Checkpoint counter {key!r} must be non-negative")
    return result


def _truncate_metrics_to_checkpoint(
    path: Path, checkpoint: dict[str, Any]
) -> None:
    """Atomically discard rows newer than the authoritative latest checkpoint."""

    checkpoint_offline = _checkpoint_counter(checkpoint, "offline_update")
    checkpoint_online = _checkpoint_counter(checkpoint, "online_step")
    checkpoint_episodes = _checkpoint_counter(checkpoint, "online_episode")
    if not path.is_file():
        if checkpoint_offline or checkpoint_online or checkpoint_episodes:
            raise ValueError("Resume checkpoint exists but metrics.jsonl is missing")
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    retained: list[dict[str, Any]] = []
    online_episode_steps: set[int] = set()
    retained_episode_ids: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            # A killed writer can leave only the final line incomplete.
            if line_number == len(lines):
                break
            raise ValueError(
                f"metrics.jsonl line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"metrics.jsonl line {line_number} is not an object")
        offline_update = row.get("offline_update")
        online_step = row.get("online_step")
        if (
            isinstance(offline_update, bool)
            or not isinstance(offline_update, int)
            or offline_update < 0
            or isinstance(online_step, bool)
            or not isinstance(online_step, int)
            or online_step < 0
        ):
            raise ValueError(
                f"metrics.jsonl line {line_number} has invalid counters"
            )
        if offline_update > checkpoint_offline or online_step > checkpoint_online:
            continue
        if row.get("phase") == "online_episode":
            episode = row.get("episode")
            if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
                raise ValueError(
                    f"metrics.jsonl line {line_number} has an invalid episode"
                )
            if episode >= checkpoint_episodes:
                continue
            if episode in retained_episode_ids:
                raise ValueError("metrics.jsonl repeats an online episode id")
            if online_step in online_episode_steps:
                raise ValueError("online episode metric rows repeat online_step")
            retained_episode_ids.add(episode)
            online_episode_steps.add(online_step)
        retained.append(row)
    if checkpoint_episodes and retained_episode_ids != set(range(checkpoint_episodes)):
        raise ValueError("metrics.jsonl does not contain every checkpointed episode")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed + 17)


@torch.no_grad()
def evaluate(
    learner: O2OLearner,
    *,
    episodes: int,
    seed_base: int,
    action_repeat: int | None = None,
) -> dict[str, Any]:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("Evaluation episodes must be a positive integer")
    # Keep the ten canonical reset seeds and all 10,000 environment steps, but
    # evaluate the policy as one batch.  This matters for AC-KMPC: ten scalar
    # GPU planner calls per control step are pure launch overhead.  A single
    # in-process vector worker avoids process-spawn cost at every 5k checkpoint.
    task = getattr(getattr(learner, "config", None), "task", "cartpole_swingup")
    vector_kwargs: dict[str, Any] = {"workers": 1}
    if action_repeat is not None:
        vector_kwargs["action_repeat"] = action_repeat
    env = make_dmc_vector_env(
        task,
        episodes,
        seed=seed_base,
        **vector_kwargs,
    )
    try:
        observation = env.reset()
        totals = np.zeros(episodes, dtype=np.float64)
        lengths = np.zeros(episodes, dtype=np.int64)
        completed = np.zeros(episodes, dtype=np.bool_)
        step_limit = int(env.protocol["step_limit"])
        for _step in range(step_limit):
            action = learner.act(observation, deterministic=True)
            vector_step = env.step(action)
            totals += np.asarray(vector_step.reward, dtype=np.float64)
            lengths += ~completed
            boundary = np.asarray(vector_step.reset_boundary, dtype=np.bool_)
            if bool(boundary.any()) != bool(boundary.all()):
                raise RuntimeError("Canonical evaluation episodes lost synchronization")
            completed |= boundary
            observation = vector_step.observation
            if completed.all():
                break
        if not completed.all() or np.any(lengths != step_limit):
            raise RuntimeError("Canonical evaluation did not end synchronously")
        returns = totals.tolist()
    finally:
        env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std_population": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "episode_length_mean": float(np.mean(lengths)),
        "returns": returns,
    }


def _checkpoint_payload(
    *,
    config: O2OConfig,
    dataset: OfflineDataset,
    koopman: FrozenKoopman | None,
    observation_normalizer: FrozenObservationNormalizer | None,
    learner: O2OLearner,
    replay: OnlineReplay,
    generator: np.random.Generator,
    phase: str,
    offline_update: int,
    online_step: int,
    online_episode: int,
    environment_protocol: dict[str, Any],
    best_return: float,
    best_online_step: int,
    initialization: dict[str, Any] | None,
    pending_trajectory: dict[str, list[np.ndarray | float]] | None = None,
    online_extension: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": CHECKPOINT_KIND,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "method_spec": dataclasses.asdict(config.method_spec),
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "metadata": dataset.metadata,
        },
        "koopman": None if koopman is None else koopman.identity(),
        "raw_observation_normalizer": (
            None
            if observation_normalizer is None
            else observation_normalizer.identity()
        ),
        "environment_protocol": environment_protocol,
        "phase": phase,
        "offline_update": offline_update,
        "online_step": online_step,
        "online_episode": online_episode,
        "best_return": best_return,
        "best_online_step": best_online_step,
        "initialization": initialization,
        "online_pending_trajectory": _pending_trajectory_state(pending_trajectory),
        "online_extension": online_extension,
        "learner": learner.state_dict(),
        "online_replay": replay.state_dict(),
        "rng": rng_state(generator),
        "saved_unix_seconds": time.time(),
    }


def _validate_resume(
    payload: dict[str, Any],
    config: O2OConfig,
    dataset: OfflineDataset,
    koopman: FrozenKoopman | None,
    environment_protocol: dict[str, Any],
    *,
    observation_normalizer: FrozenObservationNormalizer | None = None,
    require_initialization: bool = True,
) -> None:
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("Resume config fingerprint differs")
    if (
        "method_spec" in payload
        and payload.get("method_spec") != dataclasses.asdict(config.method_spec)
    ):
        raise ValueError("Resume immutable method specification differs")
    if payload.get("dataset", {}).get("sha256") != dataset.sha256:
        raise ValueError("Resume offline dataset differs")
    expected_koopman = None if koopman is None else koopman.identity()
    if payload.get("koopman") != expected_koopman:
        raise ValueError("Resume Koopman identity differs")
    expected_normalizer = (
        None
        if observation_normalizer is None
        else observation_normalizer.identity()
    )
    if payload.get("raw_observation_normalizer") != expected_normalizer:
        raise ValueError("Resume raw observation normalizer differs")
    if payload.get("environment_protocol") != environment_protocol:
        raise ValueError("Resume DMC protocol differs")
    _checkpoint_counter(payload, "offline_update")
    _checkpoint_counter(payload, "online_step")
    _checkpoint_counter(payload, "online_episode")
    pending = payload.get("online_pending_trajectory")
    if not isinstance(pending, dict) or pending.get("kind") != "calql_pending_trajectory_v1":
        raise ValueError("Resume checkpoint is missing pending-trajectory state")
    # ``latest.pt`` is deliberately written only at synchronized reset
    # boundaries.  Restoring a partial trajectory without simulator state
    # would be invalid, so a resumable checkpoint must be empty here.
    pending_count = pending.get("count")
    pending_arrays = pending.get("arrays")
    legacy_empty_arrays = (
        pending_count == 0
        and isinstance(pending_arrays, dict)
        and set(pending_arrays) == set(_PENDING_KEYS)
        and all(np.asarray(value).size == 0 for value in pending_arrays.values())
    )
    if pending_count != 0 or (pending_arrays != {} and not legacy_empty_arrays):
        raise ValueError("Resumable checkpoint contains an unfinished trajectory")
    initialization = payload.get("initialization")
    if config.requires_offline_fork and require_initialization:
        if not isinstance(initialization, dict):
            raise ValueError("MPVE resume is missing its offline-fork lineage")
        source_path = initialization.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("MPVE offline-fork source path is invalid")
        _source, current_identity = _validated_offline_fork_source(
            Path(source_path),
            target_config=config,
            dataset=dataset,
            koopman=koopman,
            observation_normalizer=observation_normalizer,
            environment_protocol=environment_protocol,
        )
        if initialization != current_identity:
            raise ValueError(
                "MPVE offline-fork lineage no longer matches the immutable source"
            )
    elif initialization is not None:
        if (
            not isinstance(initialization, dict)
            or initialization.get("kind")
            != "acmpc_o2o_offline_continuation_v1"
        ):
            raise ValueError("Non-forking method contains invalid initialization lineage")
        source_path = initialization.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("Offline continuation source path is invalid")
        _source, current_identity = _validated_offline_continuation_source(
            Path(source_path),
            target_config=config,
            dataset=dataset,
            koopman=koopman,
            observation_normalizer=observation_normalizer,
            environment_protocol=environment_protocol,
        )
        if initialization != current_identity:
            raise ValueError(
                "Offline continuation lineage no longer matches the immutable source"
            )


def _validated_offline_fork_source(
    path: Path,
    *,
    target_config: O2OConfig,
    dataset: OfflineDataset,
    koopman: FrozenKoopman,
    environment_protocol: dict[str, Any],
    observation_normalizer: FrozenObservationNormalizer | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and identify the exact pre-online AC-KMPC fork source."""

    path = path.resolve()
    source = load_checkpoint(path)
    source_config = O2OConfig(**source.get("config", {}))
    if source_config.method != "Cal-RLPD-AC-KMPC":
        raise ValueError("MPVE must fork a Cal-RLPD-AC-KMPC offline checkpoint")
    if target_config.method != "Cal-RLPD-AC-KMPC-MPVE":
        raise ValueError("Offline checkpoint forking is reserved for the MPVE ablation")
    source_fields = source_config.to_dict()
    target_fields = target_config.to_dict()
    source_fields.pop("method")
    target_fields.pop("method")
    # MPVE may be uniformly extended after both branches completed their
    # original budget.  Its immutable fork source remains the pre-online
    # checkpoint from that original protocol; online_steps does not affect
    # actor/critic/optimizer state at the fork boundary.
    source_fields.pop("online_steps")
    target_fields.pop("online_steps")
    if source_fields != target_fields:
        raise ValueError("MPVE fork config differs from the AC-KMPC source")
    if source.get("config_fingerprint") != source_config.fingerprint:
        raise ValueError("Offline fork source config fingerprint is invalid")
    if source.get("dataset", {}).get("sha256") != dataset.sha256:
        raise ValueError("Offline fork dataset differs")
    if source.get("koopman", {}).get("sha256") != koopman.sha256:
        raise ValueError("Offline fork Koopman model differs")
    if observation_normalizer is not None or source.get("raw_observation_normalizer") is not None:
        raise ValueError("Structured offline fork unexpectedly has a raw normalizer")
    if source.get("environment_protocol") != environment_protocol:
        raise ValueError("Offline fork DMC protocol differs")
    if (
        source.get("phase") != "offline"
        or _checkpoint_counter(source, "offline_update")
        != target_config.offline_updates
        or _checkpoint_counter(source, "online_step") != 0
        or _checkpoint_counter(source, "online_episode") != 0
        or source.get("initialization") is not None
    ):
        raise ValueError("MPVE fork must be the completed pre-online checkpoint")
    identity = {
        "kind": "acmpc_o2o_offline_fork_v1",
        "source_path": str(path),
        "source_sha256": file_sha256(path),
        "source_method": source_config.method,
        "source_config_fingerprint": source_config.fingerprint,
        "shared_state": "actor_critic_target_temperature_optimizers_rng",
    }
    return source, identity


def _load_offline_fork(
    path: Path,
    *,
    target_config: O2OConfig,
    dataset: OfflineDataset,
    koopman: FrozenKoopman,
    environment_protocol: dict[str, Any],
    observation_normalizer: FrozenObservationNormalizer | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact pre-online AC-KMPC state for the MPVE ablation."""

    return _validated_offline_fork_source(
        path,
        target_config=target_config,
        dataset=dataset,
        koopman=koopman,
        observation_normalizer=observation_normalizer,
        environment_protocol=environment_protocol,
    )


def _validated_offline_continuation_source(
    path: Path,
    *,
    target_config: O2OConfig,
    dataset: OfflineDataset,
    koopman: FrozenKoopman | None,
    observation_normalizer: FrozenObservationNormalizer | None,
    environment_protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one completed offline-only ``latest.pt`` for online continuation."""

    path = path.resolve()
    if path.name != "latest.pt":
        raise ValueError("Online continuation must start from offline latest.pt")
    source = load_checkpoint(path)
    source_config = O2OConfig(**source.get("config", {}))
    if source_config.method != target_config.method:
        raise ValueError("Offline continuation method differs")
    source_fields = source_config.to_dict()
    target_fields = target_config.to_dict()
    for field in ("online_steps", "eval_interval_online_steps", "eval_episodes"):
        source_fields.pop(field)
        target_fields.pop(field)
    if source_fields != target_fields:
        raise ValueError("Offline continuation config differs outside online budget/eval cadence")
    if source.get("config_fingerprint") != source_config.fingerprint:
        raise ValueError("Offline continuation source config fingerprint is invalid")
    _validate_resume(
        source,
        source_config,
        dataset,
        koopman,
        environment_protocol,
        observation_normalizer=observation_normalizer,
        require_initialization=False,
    )
    expected = {
        "phase": "offline",
        "offline_update": target_config.offline_updates,
        "online_step": 0,
        "online_episode": 0,
        "initialization": None,
    }
    mismatches = {
        key: {"actual": source.get(key), "expected": value}
        for key, value in expected.items()
        if source.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Offline continuation source is not the final offline boundary: {mismatches}"
        )
    run_path = path.parent / "run.json"
    try:
        run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("Offline continuation source requires valid run.json") from exc
    if not isinstance(run_metadata, dict):
        raise ValueError("Offline continuation source run.json must contain an object")
    expected_run = {
        "kind": "acmpc_dmc_o2o_run_v1",
        "config_fingerprint": source_config.fingerprint,
        "execution_scope": "offline_only",
        "completed": True,
        "offline_updates_completed": target_config.offline_updates,
        "online_steps_completed": 0,
    }
    run_mismatches = {
        key: {"actual": run_metadata.get(key), "expected": value}
        for key, value in expected_run.items()
        if run_metadata.get(key) != value
    }
    if run_mismatches:
        raise ValueError(
            f"Offline continuation source run is not completed offline-only: {run_mismatches}"
        )
    identity = {
        "kind": "acmpc_o2o_offline_continuation_v1",
        "source_path": str(path),
        "source_sha256": file_sha256(path),
        "source_method": source_config.method,
        "source_config_fingerprint": source_config.fingerprint,
        "target_config_fingerprint": target_config.fingerprint,
        "source_offline_update": target_config.offline_updates,
        "shared_state": "actor_critic_target_temperature_optimizers_replay_rng",
    }
    return source, identity


def _validate_offline_snapshot(
    path: Path,
    *,
    config: O2OConfig,
    dataset: OfflineDataset,
    koopman: FrozenKoopman | None,
    observation_normalizer: FrozenObservationNormalizer | None,
    environment_protocol: dict[str, Any],
    allow_online_steps_difference: bool = False,
) -> None:
    payload = load_checkpoint(path)
    source_config = O2OConfig(**payload.get("config", {}))
    source_fields = source_config.to_dict()
    requested_fields = config.to_dict()
    if allow_online_steps_difference:
        source_fields.pop("online_steps")
        requested_fields.pop("online_steps")
    if source_fields != requested_fields:
        raise ValueError("AC-KMPC offline snapshot config differs")
    _validate_resume(
        payload,
        source_config,
        dataset,
        koopman,
        environment_protocol,
        observation_normalizer=observation_normalizer,
        require_initialization=False,
    )
    expected = {
        "phase": "offline",
        "offline_update": source_config.offline_updates,
        "online_step": 0,
        "online_episode": 0,
        "initialization": None,
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"AC-KMPC offline snapshot is not pre-online: {mismatches}")


def _config_from_extension_artifact(
    value: dict[str, Any], *, label: str
) -> O2OConfig:
    try:
        config = O2OConfig(**value.get("config", {}))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains an invalid extension config") from exc
    config.validate()
    if value.get("config_fingerprint") != config.fingerprint:
        raise ValueError(f"{label} config fingerprint is invalid")
    return config


def _extension_lineage_is_valid(
    value: Any,
    *,
    base_config: O2OConfig,
    target_config: O2OConfig,
) -> bool:
    timestamp = value.get("requested_unix_seconds") if isinstance(value, dict) else None
    return bool(
        isinstance(value, dict)
        and value.get("kind") == "acmpc_o2o_online_extension_v1"
        and value.get("previous_online_steps") == base_config.online_steps
        and value.get("extended_online_steps") == target_config.online_steps
        and value.get("previous_config_fingerprint") == base_config.fingerprint
        and value.get("extended_config_fingerprint") == target_config.fingerprint
        and isinstance(timestamp, (int, float))
        and not isinstance(timestamp, bool)
        and np.isfinite(float(timestamp))
    )


def _prepare_online_extension(
    *,
    base_config: O2OConfig,
    extended_online_steps: int,
    output: Path,
    dataset: OfflineDataset,
    koopman: FrozenKoopman | None,
    observation_normalizer: FrozenObservationNormalizer | None,
    environment_protocol: dict[str, Any],
) -> tuple[O2OConfig, dict[str, Any]]:
    """Idempotently migrate every run artifact to one larger online budget.

    The authoritative checkpoint, optional best checkpoint, and ``run.json``
    cannot be replaced in one filesystem transaction.  This routine accepts
    every prefix of the deliberate write order (latest -> best -> run), derives
    the original extension lineage from any already-migrated artifact, and
    finishes the remaining replacements without changing that lineage.
    """

    if extended_online_steps <= base_config.online_steps:
        raise ValueError("Extended online budget must exceed the completed budget")
    target_config = dataclasses.replace(
        base_config, online_steps=extended_online_steps
    )
    target_config.validate()
    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    run_path = output / "run.json"
    if not latest_path.is_file():
        raise ValueError("Online extension requires an existing completed run")
    try:
        run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("Online extension requires valid run.json") from exc
    if not isinstance(run_metadata, dict):
        raise ValueError("Online extension run.json must contain an object")

    checkpoint_paths = [latest_path]
    if best_path.is_file():
        checkpoint_paths.append(best_path)
    checkpoints = {path: load_checkpoint(path) for path in checkpoint_paths}

    artifacts: list[tuple[str, dict[str, Any], O2OConfig]] = []
    for path, payload in checkpoints.items():
        artifacts.append(
            (
                path.name,
                payload,
                _config_from_extension_artifact(payload, label=path.name),
            )
        )
    run_config = _config_from_extension_artifact(run_metadata, label="run.json")
    artifacts.append(("run.json", run_metadata, run_config))

    lineages: list[dict[str, Any]] = []
    for label, value, artifact_config in artifacts:
        if artifact_config.to_dict() == base_config.to_dict():
            if value.get("online_extension") is not None:
                raise ValueError(f"Base-config {label} unexpectedly has extension lineage")
        elif artifact_config.to_dict() == target_config.to_dict():
            lineage = value.get("online_extension")
            if not _extension_lineage_is_valid(
                lineage,
                base_config=base_config,
                target_config=target_config,
            ):
                raise ValueError(f"Target-config {label} has invalid extension lineage")
            lineages.append(dict(lineage))
        else:
            raise ValueError(
                f"{label} is neither the base nor requested extension config"
            )
    if lineages and any(value != lineages[0] for value in lineages[1:]):
        raise ValueError("Online extension artifacts contain different lineages")
    extension = (
        lineages[0]
        if lineages
        else {
            "kind": "acmpc_o2o_online_extension_v1",
            "previous_online_steps": base_config.online_steps,
            "extended_online_steps": target_config.online_steps,
            "previous_config_fingerprint": base_config.fingerprint,
            "extended_config_fingerprint": target_config.fingerprint,
            "requested_unix_seconds": time.time(),
        }
    )

    latest = checkpoints[latest_path]
    latest_config = _config_from_extension_artifact(latest, label="latest.pt")
    _validate_resume(
        latest,
        latest_config,
        dataset,
        koopman,
        environment_protocol,
        observation_normalizer=observation_normalizer,
    )
    latest_step = _checkpoint_counter(latest, "online_step")
    if latest_config.to_dict() == base_config.to_dict():
        if latest_step != base_config.online_steps:
            raise ValueError("Base extension checkpoint is not at its final online step")
    elif not base_config.online_steps <= latest_step <= target_config.online_steps:
        raise ValueError("Extended checkpoint online step lies outside its lineage")

    if run_config.to_dict() == base_config.to_dict():
        if (
            run_metadata.get("completed") is not True
            or run_metadata.get("offline_updates_completed")
            != (base_config.offline_updates if base_config.uses_calql else 0)
            or run_metadata.get("online_steps_completed")
            != base_config.online_steps
        ):
            raise ValueError("Online extension source run is not completed")
    elif run_metadata.get("completed") is True and (
        run_metadata.get("online_steps_completed") != target_config.online_steps
        or latest_step != target_config.online_steps
    ):
        raise ValueError("Completed target extension has inconsistent counters")

    # Complete any interrupted identity migration.  Existing target artifacts
    # are left byte-for-byte unchanged; base artifacts are replaced atomically.
    for checkpoint_path, payload in checkpoints.items():
        artifact_config = _config_from_extension_artifact(
            payload, label=checkpoint_path.name
        )
        if artifact_config.to_dict() == target_config.to_dict():
            continue
        migrated = dict(payload)
        migrated["config"] = target_config.to_dict()
        migrated["config_fingerprint"] = target_config.fingerprint
        migrated["online_extension"] = extension
        atomic_torch_save(checkpoint_path, migrated)

    if run_config.to_dict() == base_config.to_dict():
        run_metadata.update(
            config=target_config.to_dict(),
            config_fingerprint=target_config.fingerprint,
            completed=False,
            online_extension=extension,
        )
        for stale in (
            "offline_updates_completed",
            "online_steps_completed",
            "wall_time_seconds",
        ):
            run_metadata.pop(stale, None)
        _atomic_json(run_path, run_metadata)

    return target_config, extension


def run(
    config: O2OConfig,
    dataset_path: Path,
    koopman_path: Path | None,
    output: Path,
    *,
    initialize_from_offline: Path | None = None,
    initialize_from_offline_final: Path | None = None,
    extend_online_steps: int | None = None,
    stop_after_offline: bool = False,
) -> None:
    config.validate()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    dataset = OfflineDataset.load(dataset_path)
    environment_action_repeat = _dataset_action_repeat(config.task, dataset)
    dataset_task = dataset.metadata.get("task")
    if dataset_task is not None and dataset_task != config.task:
        raise ValueError(
            f"Dataset task {dataset_task!r} does not match "
            f"training task {config.task!r}"
        )
    reward_source = dataset.metadata.get("reward_source", "oracle")
    if reward_source != "oracle" and config.task not in RECORDED_REWARD_O2O_TASKS:
        raise ValueError(
            "Offline O2O training requires an oracle reward dataset for this task; "
            "reward-free Koopman data is not a valid offline-RL target"
        )
    if reward_source not in {"oracle", "recorded"}:
        raise ValueError("Offline O2O training does not accept zero-reward data")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    device = torch.device(config.device)
    generator = _seed_everything(config.seed)
    if config.requires_koopman:
        if koopman_path is None:
            raise ValueError(f"{config.method} requires --koopman")
        koopman: FrozenKoopman | None = FrozenKoopman(koopman_path)
        expected_task = get_task_spec(config.task)
        if (koopman.state_dim, koopman.action_dim) != (
            expected_task.obs_dim,
            expected_task.action_dim,
        ):
            raise ValueError(
                f"{config.task} O2O requires Koopman state/action dimensions "
                f"{expected_task.obs_dim}/{expected_task.action_dim}; got "
                f"{koopman.state_dim}/{koopman.action_dim}"
            )
        observation_normalizer: FrozenObservationNormalizer | None = None
    else:
        if koopman_path is not None:
            raise ValueError(f"{config.method} forbids --koopman")
        koopman = None
        observation_normalizer = FrozenObservationNormalizer.from_offline_observations(
            dataset.arrays["observation"], dataset_sha256=dataset.sha256
        )
    protocol_kwargs: dict[str, Any] = {"seed": config.seed}
    if environment_action_repeat is not None:
        protocol_kwargs["action_repeat"] = environment_action_repeat
    protocol_env = make_dmc_adapter(config.task, **protocol_kwargs)
    environment_protocol = protocol_env.protocol_metadata()
    protocol_env.close()
    _validate_dataset_environment_protocol(
        config.task, dataset, environment_protocol
    )

    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    extension_metadata: dict[str, Any] | None = None
    if extend_online_steps is not None:
        if isinstance(extend_online_steps, bool) or not isinstance(
            extend_online_steps, int
        ):
            raise ValueError("Extended online budget must be an integer")
        if extend_online_steps % 5_000:
            raise ValueError("Extended online budget must be a multiple of 5000")
        if initialize_from_offline is not None or initialize_from_offline_final is not None:
            raise ValueError("Cannot initialize an offline fork while extending")
        config, extension_metadata = _prepare_online_extension(
            base_config=config,
            extended_online_steps=extend_online_steps,
            output=output,
            dataset=dataset,
            koopman=koopman,
            observation_normalizer=observation_normalizer,
            environment_protocol=environment_protocol,
        )

    learner = O2OLearner(
        config,
        koopman,
        device,
        observation_normalizer=observation_normalizer,
    )
    task_spec = get_task_spec(config.task)
    replay = OnlineReplay(
        config.replay_capacity,
        obs_dim=task_spec.obs_dim,
        action_dim=task_spec.action_dim,
    )

    offline_update = 0
    online_step = 0
    online_episode = 0
    best_return = float("-inf")
    best_online_step = -1
    resumed = latest_path.is_file()
    if resumed and (
        initialize_from_offline is not None
        or initialize_from_offline_final is not None
    ):
        raise ValueError("Cannot combine offline initialization with resume")
    if (
        initialize_from_offline is not None
        and initialize_from_offline_final is not None
    ):
        raise ValueError("Offline initialization modes are mutually exclusive")
    # ``--stop-after-offline`` is useful for any method when the caller wants
    # to inspect/freeze the offline checkpoint before launching online
    # fine-tuning.  Methods without offline pretraining (plain RLPD) are
    # recorded as an explicit offline=N/A boundary and never enter the env.
    if config.requires_offline_fork and not resumed and initialize_from_offline is None:
        raise ValueError(
            "AC-KMPC-MPVE requires --initialize-from-offline from the paired "
            "Cal-RLPD-AC-KMPC offline.pt"
        )
    if not config.requires_offline_fork and initialize_from_offline is not None:
        raise ValueError("--initialize-from-offline is only valid for AC-KMPC-MPVE")
    initialization: dict[str, Any] | None = None
    if resumed:
        payload = load_checkpoint(latest_path)
        _validate_resume(
            payload,
            config,
            dataset,
            koopman,
            environment_protocol,
            observation_normalizer=observation_normalizer,
        )
        learner.load_state_dict(payload["learner"])
        replay.load_state_dict(payload["online_replay"])
        restore_rng(payload["rng"], generator)
        offline_update = int(payload["offline_update"])
        online_step = int(payload["online_step"])
        online_episode = int(payload["online_episode"])
        best_return = float(payload["best_return"])
        best_online_step = int(payload["best_online_step"])
        # A best checkpoint may have been written after the latest recovery
        # checkpoint and before a process interruption.  Restore its score so
        # the next save does not forget that improvement.
        if best_path.is_file():
            best_payload = load_checkpoint(best_path)
            if (
                best_payload.get("config_fingerprint") != config.fingerprint
                or best_payload.get("dataset", {}).get("sha256") != dataset.sha256
                or best_payload.get("environment_protocol") != environment_protocol
            ):
                raise ValueError("Best checkpoint identity differs on resume")
            expected_koopman = None if koopman is None else koopman.identity()
            if best_payload.get("koopman") != expected_koopman:
                raise ValueError("Best checkpoint Koopman identity differs on resume")
            saved_best_return = float(best_payload["best_return"])
            if saved_best_return > best_return:
                best_return = saved_best_return
                best_online_step = int(best_payload["best_online_step"])
        initialization = payload.get("initialization")
        saved_extension = payload.get("online_extension")
        if saved_extension is not None:
            if (
                not isinstance(saved_extension, dict)
                or saved_extension.get("kind")
                != "acmpc_o2o_online_extension_v1"
                or saved_extension.get("extended_config_fingerprint")
                != config.fingerprint
            ):
                raise ValueError("Resume checkpoint has invalid extension lineage")
            extension_metadata = saved_extension
        _truncate_metrics_to_checkpoint(metrics_path, payload)
    elif initialize_from_offline is not None:
        if koopman is None:
            raise ValueError("Offline fork initialization requires Koopman")
        payload, initialization = _load_offline_fork(
            initialize_from_offline,
            target_config=config,
            dataset=dataset,
            koopman=koopman,
            observation_normalizer=observation_normalizer,
            environment_protocol=environment_protocol,
        )
        learner.load_state_dict(payload["learner"])
        restore_rng(payload["rng"], generator)
        offline_update = int(payload["offline_update"])
    elif initialize_from_offline_final is not None:
        payload, initialization = _validated_offline_continuation_source(
            initialize_from_offline_final,
            target_config=config,
            dataset=dataset,
            koopman=koopman,
            observation_normalizer=observation_normalizer,
            environment_protocol=environment_protocol,
        )
        learner.load_state_dict(payload["learner"])
        replay.load_state_dict(payload["online_replay"])
        if replay.size != 0 or replay.cursor != 0:
            raise ValueError("Offline continuation source contains online replay data")
        restore_rng(payload["rng"], generator)
        offline_update = int(payload["offline_update"])

    run_metadata = {
        "kind": "acmpc_dmc_o2o_run_v1",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "method_spec": dataclasses.asdict(config.method_spec),
        "dataset": {"path": str(dataset.path), "sha256": dataset.sha256},
        "koopman": None if koopman is None else koopman.identity(),
        "raw_observation_normalizer": (
            None
            if observation_normalizer is None
            else observation_normalizer.identity()
        ),
        "environment_protocol": environment_protocol,
        "device": str(device),
        "checkpoint_artifacts": {
            "latest": "latest.pt",
            "best": "best.pt",
            "best_selection": "highest_recorded_evaluation_return",
            "milestones": {
                "offline": "offline_NNNNNN.pt at every offline diagnostic boundary",
                "online": "online_NNNNNN.pt at every online diagnostic boundary",
            },
        },
        "started_unix_seconds": time.time(),
        "resumed": resumed,
        "algorithm_label": config.method_spec.profile,
        "calibration_reference": "finite_horizon_discounted_episode_return_v1",
        "online_collection": {
            "runner": "ProcessDMCVectorEnv",
            "num_envs": config.num_envs,
            "env_workers": config.env_workers,
            "online_step_counter": "total_real_environment_transitions",
            "interaction_order": "batched_actor_then_batched_env_step",
            "learner_updates": (
                "one_UTD1_update_per_real_transition_deferred_to_completed_episode"
                if config.requires_completed_online_returns
                else "one_fused_UTD_update_per_real_transition"
            ),
            "online_mc_return": (
                "complete_episode_discounted_return_to_go"
                if config.requires_completed_online_returns
                else "not_used_for_online_calibration"
            ),
            "latest_checkpoint": "synchronized_all_env_reset_boundary_only",
        },
        "diagnostic_evaluation": {
            "deterministic": True,
            "episodes": config.eval_episodes,
            "seed_base": DIAGNOSTIC_EVAL_SEED_BASE,
            "disjoint_from_final_10x10_seed_base": 9_100_000,
        },
        "initialization": initialization,
        "online_extension": extension_metadata,
        "learning_rate_schedule": {
            "offline": {
                "actor": config.learning_rate_for_phase("actor", "offline"),
                "critic": config.learning_rate_for_phase("critic", "offline"),
            },
            "online": {
                "actor": config.learning_rate_for_phase("actor", "online"),
                "critic": config.learning_rate_for_phase("critic", "online"),
            },
            "transition": "preserve_adam_state_and_update_param_group_lr",
        },
        "mpve": {
            "enabled": config.uses_mpve,
            "scope": config.method_spec.mpve_scope,
            "updates_per_offline_gradient_step": (
                1 if config.uses_offline_mpve else 0
            ),
            "updates_per_real_environment_step": (
                1 if config.uses_online_mpve else 0
            ),
            "total_td_horizon": config.mpve_total_horizon if config.uses_mpve else None,
            "composition": (
                f"one_real_plus_{config.mpve_total_horizon - 1}_model"
                if config.uses_mpve
                else None
            ),
        },
        "execution_scope": "offline_only" if stop_after_offline else "offline_to_online",
    }
    _atomic_json(output / "run.json", run_metadata)

    def save_best_checkpoint(
        pending_trajectory: dict[str, list[np.ndarray | float]] | None = None,
    ) -> None:
        """Atomically publish the learner state selected by evaluation."""

        atomic_torch_save(
            best_path,
            _checkpoint_payload(
                config=config,
                dataset=dataset,
                koopman=koopman,
                observation_normalizer=observation_normalizer,
                learner=learner,
                replay=replay,
                generator=generator,
                phase=("offline" if online_step == 0 else "online"),
                offline_update=offline_update,
                online_step=online_step,
                online_episode=online_episode,
                environment_protocol=environment_protocol,
                best_return=best_return,
                best_online_step=best_online_step,
                initialization=initialization,
                pending_trajectory=pending_trajectory,
                online_extension=extension_metadata,
            ),
        )

    def save_milestone_checkpoint(
        stage: str,
        *,
        pending_trajectory: dict[str, list[np.ndarray | float]] | None = None,
    ) -> Path:
        """Save the exact learner selected for one scheduled diagnostic."""

        if stage not in {"offline", "online"}:
            raise ValueError("Milestone stage must be offline or online")
        counter = offline_update if stage == "offline" else online_step
        if counter > 999_999:
            raise ValueError("Milestone counter exceeds the six-digit artifact schema")
        path = output / f"{stage}_{counter:06d}.pt"
        atomic_torch_save(
            path,
            _checkpoint_payload(
                config=config,
                dataset=dataset,
                koopman=koopman,
                observation_normalizer=observation_normalizer,
                learner=learner,
                replay=replay,
                generator=generator,
                phase=stage,
                offline_update=offline_update,
                online_step=online_step,
                online_episode=online_episode,
                environment_protocol=environment_protocol,
                best_return=best_return,
                best_online_step=best_online_step,
                initialization=initialization,
                pending_trajectory=pending_trajectory,
                online_extension=extension_metadata,
            ),
        )
        return path

    def save_milestone_metrics(
        stage: str,
        evaluation: dict[str, Any],
    ) -> Path:
        if stage not in {"offline", "online"}:
            raise ValueError("Milestone stage must be offline or online")
        counter = offline_update if stage == "offline" else online_step
        path = output / f"evaluation_{stage}_{counter:06d}.json"
        checkpoint_path = output / f"{stage}_{counter:06d}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Milestone metrics require their checkpoint")
        _atomic_json(
            path,
            {
                "kind": "acmpc_dmc_o2o_training_diagnostic_v1",
                "task": config.task,
                "method": config.method,
                "training_seed": config.seed,
                "stage": stage,
                "offline_update": offline_update,
                "online_step": online_step,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "evaluation": evaluation,
            },
        )
        return path

    initial_evaluation_phase = (
        "offline_evaluation" if initialization is not None else "initial"
    )
    initial_evaluation_offline_update = (
        config.offline_updates if initialization is not None else 0
    )
    # Establish the recovery boundary before the relatively long fixed-seed
    # step-zero evaluation.  If the process dies during or immediately after
    # evaluation, the same checkpoint can safely reproduce a missing metric;
    # an already-flushed metric is detected and never duplicated.
    if not resumed:
        atomic_torch_save(
            latest_path,
            _checkpoint_payload(
                config=config,
                dataset=dataset,
                koopman=koopman,
                observation_normalizer=observation_normalizer,
                learner=learner,
                replay=replay,
                generator=generator,
                phase=("offline" if config.uses_offline_pretraining else "online"),
                offline_update=offline_update,
                online_step=online_step,
                online_episode=online_episode,
                environment_protocol=environment_protocol,
                best_return=best_return,
                best_online_step=best_online_step,
                initialization=initialization,
                online_extension=extension_metadata,
            ),
        )
        save_milestone_checkpoint(
            "offline" if config.uses_offline_pretraining else "online"
        )
    has_initial_evaluation = _has_metric_row(
        metrics_path,
        phase=initial_evaluation_phase,
        offline_update=initial_evaluation_offline_update,
        online_step=0,
    )
    if not has_initial_evaluation:
        if (
            online_step != 0
            or offline_update != initial_evaluation_offline_update
        ):
            raise ValueError(
                "Cannot reconstruct a missing step-zero evaluation after training"
            )
        initial_eval = evaluate(
            learner,
            episodes=config.eval_episodes,
            seed_base=DIAGNOSTIC_EVAL_SEED_BASE,
            action_repeat=environment_action_repeat,
        )
        if initial_eval["return_mean"] > best_return:
            best_return = float(initial_eval["return_mean"])
            best_online_step = 0
            save_best_checkpoint()
        _append_jsonl(
            metrics_path,
            {
                "phase": initial_evaluation_phase,
                "offline_update": offline_update,
                "online_step": online_step,
                **initial_eval,
            },
        )
        save_milestone_metrics(
            "offline" if config.uses_offline_pretraining else "online",
            initial_eval,
        )

    started = time.time()
    if config.requires_own_offline_pretraining:
        def evaluate_offline_diagnostic_if_due() -> None:
            """Persist an idempotent fixed-seed diagnostic at offline milestones."""

            nonlocal best_return, best_online_step

            if (
                offline_update <= 0
                or offline_update >= config.offline_updates
                or offline_update % config.offline_eval_interval_updates
            ):
                return
            if _has_metric_row(
                metrics_path,
                phase="offline_diagnostic",
                offline_update=offline_update,
                online_step=0,
            ):
                return
            # The checkpoint is authoritative before the relatively slow
            # environment evaluation, so a crash resumes and reconstructs
            # this exact missing diagnostic rather than skipping it.
            atomic_torch_save(
                latest_path,
                _checkpoint_payload(
                    config=config,
                    dataset=dataset,
                    koopman=koopman,
                    observation_normalizer=observation_normalizer,
                    learner=learner,
                    replay=replay,
                    generator=generator,
                    phase="offline",
                    offline_update=offline_update,
                    online_step=0,
                    online_episode=0,
                    environment_protocol=environment_protocol,
                    best_return=best_return,
                    best_online_step=best_online_step,
                    initialization=initialization,
                    online_extension=extension_metadata,
                ),
            )
            diagnostic = evaluate(
                learner,
                episodes=config.eval_episodes,
                seed_base=DIAGNOSTIC_EVAL_SEED_BASE,
                action_repeat=environment_action_repeat,
            )
            save_milestone_checkpoint("offline")
            if diagnostic["return_mean"] > best_return:
                best_return = float(diagnostic["return_mean"])
                best_online_step = 0
                save_best_checkpoint()
            _append_jsonl(
                metrics_path,
                {
                    "phase": "offline_diagnostic",
                    "offline_update": offline_update,
                    "online_step": 0,
                    **diagnostic,
                },
            )
            save_milestone_metrics("offline", diagnostic)

        # Covers a process interruption after the milestone checkpoint but
        # before its diagnostic row was durably flushed.
        evaluate_offline_diagnostic_if_due()
        while offline_update < config.offline_updates:
            batch_np = mark_offline(dataset.sample(config.batch_size, generator))
            metrics = learner.update(
                TensorBatch.from_numpy(batch_np, device),
                utd=1,
                phase="offline",
            )
            offline_update += 1
            if offline_update % config.log_interval_updates == 0:
                row = {
                    "phase": "offline",
                    "offline_update": offline_update,
                    "online_step": 0,
                    "elapsed_seconds": time.time() - started,
                    **metrics,
                }
                _append_jsonl(metrics_path, row)
                print(
                    f"method={config.method} offline={offline_update}/{config.offline_updates} "
                    f"q={metrics['q_mean']:.3f} loss={metrics['critic_loss']:.4g}",
                    flush=True,
                )
            if offline_update % config.checkpoint_interval_updates == 0:
                atomic_torch_save(
                    latest_path,
                    _checkpoint_payload(
                        config=config,
                        dataset=dataset,
                        koopman=koopman,
                        observation_normalizer=observation_normalizer,
                        learner=learner,
                        replay=replay,
                        generator=generator,
                        phase="offline",
                        offline_update=offline_update,
                        online_step=0,
                        online_episode=0,
                        environment_protocol=environment_protocol,
                        best_return=best_return,
                        best_online_step=best_online_step,
                        initialization=initialization,
                        online_extension=extension_metadata,
                    ),
                )
            evaluate_offline_diagnostic_if_due()
        # The configured budget need not be a multiple of the periodic
        # checkpoint interval.  Make the completed offline learner/RNG state
        # authoritative before evaluating or exposing the fork artifact.
        if online_step == 0:
            atomic_torch_save(
                latest_path,
                _checkpoint_payload(
                    config=config,
                    dataset=dataset,
                    koopman=koopman,
                    observation_normalizer=observation_normalizer,
                    learner=learner,
                    replay=replay,
                    generator=generator,
                    phase="offline",
                    offline_update=offline_update,
                    online_step=0,
                    online_episode=0,
                    environment_protocol=environment_protocol,
                    best_return=best_return,
                    best_online_step=best_online_step,
                    initialization=initialization,
                    online_extension=extension_metadata,
                ),
            )
        has_offline_evaluation = _has_metric_row(
            metrics_path,
            phase="offline_evaluation",
            offline_update=config.offline_updates,
            online_step=0,
        )
        if not has_offline_evaluation:
            if online_step != 0:
                raise ValueError(
                    "Cannot reconstruct a missing offline evaluation after online updates"
                )
            offline_eval = evaluate(
                learner,
                episodes=config.eval_episodes,
                seed_base=DIAGNOSTIC_EVAL_SEED_BASE,
                action_repeat=environment_action_repeat,
            )
            if offline_eval["return_mean"] > best_return:
                best_return = float(offline_eval["return_mean"])
                best_online_step = 0
                save_best_checkpoint()
            _append_jsonl(
                metrics_path,
                {
                    "phase": "offline_evaluation",
                    "offline_update": offline_update,
                    "online_step": 0,
                    **offline_eval,
                },
            )
        else:
            offline_eval = _read_metric_row(
                metrics_path,
                phase="offline_evaluation",
                offline_update=config.offline_updates,
                online_step=0,
            )
        save_milestone_checkpoint("offline")
        # Online step zero is the exact final offline learner. Keep a named
        # alias because it is one of the two paper-level 10x10 boundaries.
        save_milestone_checkpoint("online")
        save_milestone_metrics("offline", offline_eval)
        save_milestone_metrics("online", offline_eval)
        if config.method == "Cal-RLPD-AC-KMPC":
            # Preserve the exact common state before either structured online
            # branch sees a real transition.  MPVE forks this file.  Once
            # online learning starts it is immutable: a resume must never
            # replace it with a post-online learner state.
            offline_path = output / "offline.pt"
            if offline_path.is_file():
                _validate_offline_snapshot(
                    offline_path,
                    config=config,
                    dataset=dataset,
                    koopman=koopman,
                    observation_normalizer=observation_normalizer,
                    environment_protocol=environment_protocol,
                    allow_online_steps_difference=extension_metadata is not None,
                )
            elif online_step == 0:
                atomic_torch_save(
                    offline_path,
                    _checkpoint_payload(
                        config=config,
                        dataset=dataset,
                        koopman=koopman,
                        observation_normalizer=observation_normalizer,
                        learner=learner,
                        replay=replay,
                        generator=generator,
                        phase="offline",
                        offline_update=offline_update,
                        online_step=0,
                        online_episode=0,
                        environment_protocol=environment_protocol,
                        best_return=best_return,
                        best_online_step=best_online_step,
                        initialization=initialization,
                        online_extension=extension_metadata,
                    ),
                )
            else:
                raise ValueError(
                    "Online AC-KMPC resume is missing its immutable offline.pt"
                )

    if stop_after_offline:
        if config.uses_offline_pretraining:
            offline_result = _read_metric_row(
                metrics_path,
                phase="offline_evaluation",
                offline_update=config.offline_updates,
                online_step=0,
            )
        else:
            # RLPD starts from random initialization and has no offline
            # gradient phase.  Its step-zero evaluation is the only valid
            # offline boundary; do not fabricate a 60k offline checkpoint.
            offline_result = _read_metric_row(
                metrics_path,
                phase="initial",
                offline_update=0,
                online_step=0,
            )
        completion_return = float(offline_result["return_mean"])
        if not np.isfinite(completion_return):
            raise FloatingPointError("Offline completion return is non-finite")
        if completion_return > best_return:
            best_return = completion_return
            best_online_step = 0
            save_best_checkpoint()
        # The completion checkpoint and JSON must both be finite and agree.
        atomic_torch_save(
            latest_path,
            _checkpoint_payload(
                config=config,
                dataset=dataset,
                koopman=koopman,
                observation_normalizer=observation_normalizer,
                learner=learner,
                replay=replay,
                generator=generator,
                phase="offline",
                offline_update=offline_update,
                online_step=0,
                online_episode=0,
                environment_protocol=environment_protocol,
                best_return=best_return,
                best_online_step=best_online_step,
                initialization=initialization,
                online_extension=extension_metadata,
            ),
        )
        run_metadata.update(
            completed=True,
            completion_scope="offline_only",
            offline_updates_completed=offline_update,
            online_steps_completed=0,
            best_return=best_return,
            best_online_step=best_online_step,
            wall_time_seconds=time.time() - started,
        )
        _atomic_json(output / "run.json", run_metadata)
        return

    # Offline-only overrides end at the exact pre-online boundary.  Retain
    # learned parameters and Adam moments, but restore the method's original
    # online actor/critic rates before the first environment transition.
    learner.set_phase_learning_rates("online")

    if online_step % config.num_envs:
        raise ValueError("Resume online_step is not aligned to the vector width")
    if online_episode % config.num_envs:
        raise ValueError("Latest checkpoint is not at a synchronized reset boundary")
    # A completed checkpoint needs no simulator reconstruction.  In
    # particular, rerunning a finished detached command must be an idempotent
    # no-op rather than creating a fresh vector environment and then failing
    # the boundary assertion below.
    if online_step == config.online_steps:
        run_metadata.update(
            completed=True,
            offline_updates_completed=offline_update,
            online_steps_completed=online_step,
            best_return=best_return,
            best_online_step=best_online_step,
            wall_time_seconds=0.0,
        )
        _atomic_json(output / "run.json", run_metadata)
        return
    online_env_kwargs: dict[str, Any] = {"workers": config.env_workers}
    if environment_action_repeat is not None:
        online_env_kwargs["action_repeat"] = environment_action_repeat
    env = make_dmc_vector_env(
        config.task,
        config.num_envs,
        seed=config.seed + 100_000 + online_episode,
        **online_env_kwargs,
    )
    if env.protocol != environment_protocol:
        env.close()
        raise ValueError("Vector collector protocol differs from checkpoint protocol")
    observations = env.reset()
    episode_returns = np.zeros(config.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
    pending_trajectory: dict[str, list[np.ndarray | float]] | None = (
        {key: [] for key in _PENDING_KEYS}
        if config.requires_completed_online_returns
        else None
    )
    latest_checkpoint_online_step = online_step

    def online_update() -> dict[str, float]:
        ratio = (
            config.offline_replay_ratio
            if config.uses_offline_replay_online
            else 0.0
        )
        batch_np = mixed_batch(
            dataset,
            replay,
            batch_size=config.batch_size,
            utd=config.online_utd,
            offline_ratio=ratio,
            generator=generator,
        )
        return learner.update(
            TensorBatch.from_numpy(batch_np, device),
            utd=config.online_utd,
            phase="online",
        )

    try:
        while online_step < config.online_steps:
            if config.online_steps - online_step < config.num_envs:
                raise ValueError("Online budget ends inside a vector step")
            random_warmup = (
                not config.uses_offline_pretraining
                and online_step < config.online_warmup_steps
            )
            if random_warmup:
                actions = generator.uniform(
                    -1.0,
                    1.0,
                    size=(config.num_envs, learner.action_dim),
                ).astype(np.float32)
            else:
                actions = learner.act(observations, deterministic=False)
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (config.num_envs, learner.action_dim):
                raise RuntimeError("Batched policy emitted an invalid action shape")
            vector_step = env.step(actions)
            episode_returns += np.asarray(vector_step.reward, dtype=np.float64)
            episode_lengths += 1
            reset_boundary = np.asarray(vector_step.reset_boundary, dtype=np.bool_)
            if bool(reset_boundary.any()) != bool(reset_boundary.all()):
                raise RuntimeError(
                    "DMC vector environments lost synchronized episode boundaries"
                )
            completed_rows: list[dict[str, Any]] = []
            for environment_index in range(config.num_envs):
                transition = {
                    "observation": observations[environment_index].copy(),
                    "action": vector_step.applied_action[environment_index].copy(),
                    "reward": float(vector_step.reward[environment_index]),
                    "discount": float(vector_step.discount[environment_index]),
                    "next_observation": vector_step.transition_observation[
                        environment_index
                    ].copy(),
                }
                if pending_trajectory is None:
                    replay.add(
                        transition["observation"],
                        transition["action"],
                        transition["reward"],
                        transition["discount"],
                        transition["next_observation"],
                    )
                else:
                    for key in _PENDING_KEYS:
                        pending_trajectory[key].append(transition[key])
                online_step += 1
                ready = (
                    config.uses_offline_pretraining
                    or online_step >= config.online_warmup_steps
                )
                if ready and pending_trajectory is None:
                    metrics = online_update()
                else:
                    metrics = {}
                if reset_boundary[environment_index]:
                    if pending_trajectory is not None:
                        transition_count = len(pending_trajectory["reward"])
                        replay.add_episode(
                            np.asarray(pending_trajectory["observation"]),
                            np.asarray(pending_trajectory["action"]),
                            np.asarray(pending_trajectory["reward"]),
                            np.asarray(pending_trajectory["discount"]),
                            np.asarray(pending_trajectory["next_observation"]),
                            gamma=config.discount,
                        )
                        # Preserve one learner update per real transition while
                        # ensuring every online Cal-QL calibration target came
                        # from a completed trajectory.
                        for _ in range(transition_count):
                            metrics = online_update()
                        for key in _PENDING_KEYS:
                            pending_trajectory[key].clear()
                    completed_rows.append(
                        {
                            "phase": "online_episode",
                            "offline_update": offline_update,
                            # Processing one transition at a time gives each
                            # completed environment a unique real-step count.
                            "online_step": online_step,
                            "episode": online_episode,
                            "environment_index": environment_index,
                            "environment_episode": online_episode // config.num_envs,
                            "reset_seed": int(vector_step.reset_seed[environment_index]),
                            "episode_return": float(
                                episode_returns[environment_index]
                            ),
                            "episode_length": int(
                                episode_lengths[environment_index]
                            ),
                            **metrics,
                        }
                    )
                    online_episode += 1
            observations = vector_step.observation
            for row in completed_rows:
                _append_jsonl(metrics_path, row)
            at_reset_boundary = bool(reset_boundary.all())
            if at_reset_boundary:
                episode_returns.fill(0.0)
                episode_lengths.fill(0)

            should_evaluate = (
                online_step % config.eval_interval_online_steps == 0
                or online_step == config.online_steps
            )
            if should_evaluate:
                save_milestone_checkpoint(
                    "online", pending_trajectory=pending_trajectory
                )
                evaluation = evaluate(
                    learner,
                    episodes=config.eval_episodes,
                    seed_base=DIAGNOSTIC_EVAL_SEED_BASE,
                    action_repeat=environment_action_repeat,
                )
                row = {
                    "phase": "online_evaluation",
                    "offline_update": offline_update,
                    "online_step": online_step,
                    "elapsed_seconds": time.time() - started,
                    **evaluation,
                }
                _append_jsonl(metrics_path, row)
                save_milestone_metrics("online", evaluation)
                if evaluation["return_mean"] > best_return:
                    best_return = evaluation["return_mean"]
                    best_online_step = online_step
                    save_best_checkpoint(pending_trajectory)
                print(
                    f"method={config.method} online={online_step}/{config.online_steps} "
                    f"eval={evaluation['return_mean']:.2f} best={best_return:.2f}",
                    flush=True,
                )

            # Resume never depends on unsaved simulator state: only a point at
            # which all five environments have autoreset is authoritative.
            if at_reset_boundary:
                atomic_torch_save(
                    latest_path,
                    _checkpoint_payload(
                        config=config,
                        dataset=dataset,
                        koopman=koopman,
                        observation_normalizer=observation_normalizer,
                        learner=learner,
                        replay=replay,
                        generator=generator,
                        phase="online",
                        offline_update=offline_update,
                        online_step=online_step,
                        online_episode=online_episode,
                        environment_protocol=environment_protocol,
                        best_return=best_return,
                        best_online_step=best_online_step,
                        initialization=initialization,
                        pending_trajectory=pending_trajectory,
                        online_extension=extension_metadata,
                    ),
                )
                latest_checkpoint_online_step = online_step
    finally:
        env.close()

    if latest_checkpoint_online_step != online_step:
        raise RuntimeError(
            "Online training ended away from a checkpointable reset boundary"
        )

    run_metadata.update(
        completed=True,
        offline_updates_completed=offline_update,
        online_steps_completed=online_step,
        best_return=best_return,
        best_online_step=best_online_step,
        wall_time_seconds=time.time() - started,
    )
    _atomic_json(output / "run.json", run_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=TRAIN_METHODS, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--koopman", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--task",
        choices=sorted(SUPPORTED_O2O_TASKS),
        default="cartpole_swingup",
    )
    parser.add_argument("--kmpc-horizon", type=int, default=20)
    parser.add_argument("--mpve-total-horizon", type=int, default=10)
    parser.add_argument("--offline-updates", type=int, default=500_000)
    parser.add_argument("--online-steps", type=int, default=50_000)
    parser.add_argument("--online-utd", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--env-workers", type=int)
    parser.add_argument("--cql-weight", type=float, default=0.01)
    parser.add_argument("--offline-actor-learning-rate", type=float)
    parser.add_argument("--offline-critic-learning-rate", type=float)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--offline-eval-interval-updates", type=int, default=5_000
    )
    parser.add_argument(
        "--eval-interval-online-steps", type=int, default=2_500
    )
    parser.add_argument("--initialize-from-offline", type=Path)
    parser.add_argument(
        "--initialize-from-offline-final",
        type=Path,
        help="Start a fresh online run from a completed same-method offline latest.pt",
    )
    parser.add_argument("--extend-online-steps", type=int)
    parser.add_argument("--stop-after-offline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    spec = O2OConfig(method=args.method).method_spec
    if not args.smoke:
        requested_contract = {
            "--online-utd": (args.online_utd, spec.online_utd),
            "--num-envs": (args.num_envs, spec.num_envs),
            "--env-workers": (args.env_workers, spec.env_workers),
        }
        mismatches = {
            option: {"requested": requested, "method_default": expected}
            for option, (requested, expected) in requested_contract.items()
            if requested is not None and requested != expected
        }
        if mismatches:
            raise ValueError(
                f"Formal method execution contract cannot be overridden: {mismatches}"
            )
    num_envs = spec.num_envs if args.num_envs is None else args.num_envs
    env_workers = spec.env_workers if args.env_workers is None else args.env_workers
    online_utd = spec.online_utd if args.online_utd is None else args.online_utd
    # Cartpole only exposes a recoverable simulator boundary after its full
    # 1000-step time limit.  A smoke run therefore uses exactly one episode
    # per vector member rather than ending after an unrecoverable partial 10
    # transitions.
    smoke_online_steps = 1_000 * num_envs
    config = O2OConfig(
        method=args.method,
        seed=args.seed,
        device=args.device,
        task=args.task,
        kmpc_horizon=args.kmpc_horizon,
        mpve_total_horizon=args.mpve_total_horizon,
        offline_updates=20 if args.smoke else args.offline_updates,
        online_steps=smoke_online_steps if args.smoke else args.online_steps,
        online_utd=2 if args.smoke else online_utd,
        online_warmup_steps=(
            smoke_online_steps
            if args.smoke and not spec.offline_pretraining
            else spec.online_warmup_steps
        ),
        num_envs=num_envs,
        env_workers=env_workers,
        cql_weight=args.cql_weight,
        offline_actor_learning_rate=args.offline_actor_learning_rate,
        offline_critic_learning_rate=args.offline_critic_learning_rate,
        eval_interval_online_steps=(
            smoke_online_steps
            if args.smoke
            else args.eval_interval_online_steps
        ),
        eval_episodes=2 if args.smoke else args.eval_episodes,
        checkpoint_interval_updates=10 if args.smoke else 10_000,
        log_interval_updates=5 if args.smoke else 1_000,
        offline_eval_interval_updates=(
            10 if args.smoke else args.offline_eval_interval_updates
        ),
    )
    run(
        config,
        args.dataset,
        args.koopman,
        args.output_dir,
        initialize_from_offline=args.initialize_from_offline,
        initialize_from_offline_final=args.initialize_from_offline_final,
        extend_online_steps=args.extend_online_steps,
        stop_after_offline=args.stop_after_offline,
    )


if __name__ == "__main__":
    main()
