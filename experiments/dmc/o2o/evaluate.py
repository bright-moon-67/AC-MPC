"""Strict deterministic evaluation of an O2O ``latest`` or ``best`` checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments.dmc.o2o.checkpoint import load_checkpoint
from experiments.dmc.o2o.config import O2OConfig
from experiments.dmc.o2o.dataset import OfflineDataset
from experiments.dmc.o2o.koopman import FrozenKoopman, file_sha256
from experiments.dmc.o2o.learner import O2OLearner
from experiments.dmc.o2o.networks import FrozenObservationNormalizer
from experiments.dmc.tasks.adapter import make_dmc_adapter
from experiments.dmc.tasks.registry import get_task_spec


RUN_KIND = "acmpc_dmc_o2o_run_v1"
EVALUATION_KIND = "acmpc_dmc_o2o_checkpoint_evaluation_v1"
CHECKPOINT_NAMES = ("latest", "best")
_MILESTONE_CHECKPOINT = re.compile(r"^(offline|online)_\d{6}$")
EVALUATION_EPISODES = 10
EVALUATION_SEED_BASE = 9_100_000


@dataclass(frozen=True)
class ValidatedRun:
    run_dir: Path
    checkpoint_name: str
    checkpoint_path: Path
    checkpoint_sha256: str
    run_metadata: dict[str, Any]
    checkpoint: dict[str, Any]
    config: O2OConfig
    dataset: OfflineDataset | None
    dataset_path: Path
    dataset_sha256: str
    koopman: FrozenKoopman | None
    koopman_path: Path | None
    koopman_sha256: str | None
    observation_normalizer: FrozenObservationNormalizer | None


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required result file does not exist: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Identity field {key!r} must be a mapping")
    return dict(value)


def _resolve_saved_path(saved: Any, override: Path | None, *, field: str) -> Path:
    if not isinstance(saved, str) or not saved:
        raise ValueError(f"Saved {field} path is missing")
    return (override if override is not None else Path(saved)).resolve()


def _config_from_checkpoint(payload: Mapping[str, Any]) -> O2OConfig:
    mapping = _require_mapping(payload, "config")
    try:
        config = O2OConfig(**mapping)
    except TypeError as exc:
        raise ValueError("Checkpoint contains an invalid O2O config") from exc
    config.validate()
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("Checkpoint config fingerprint is invalid")
    return config


def _validate_counter(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Checkpoint field {key!r} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"Checkpoint field {key!r} must be non-negative")
    return result


def _restore_raw_normalizer(
    identity: Any, *, dataset_sha256: str
) -> FrozenObservationNormalizer:
    if not isinstance(identity, Mapping):
        raise ValueError("Raw checkpoint is missing observation normalizer identity")
    if identity.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Raw normalizer is bound to another offline dataset")
    try:
        center = np.asarray(identity["center"], dtype=np.float32)
        scale = np.asarray(identity["scale"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Raw normalizer center/scale is invalid") from exc
    normalizer = FrozenObservationNormalizer(
        center, scale, dataset_sha256=dataset_sha256
    )
    if normalizer.identity() != dict(identity):
        raise ValueError("Raw normalizer identity/hash is not canonical")
    return normalizer


def _validate_initialization_lineage(
    *,
    config: O2OConfig,
    initialization: Any,
    dataset_sha256: str,
    koopman_sha256: str | None,
    environment_protocol: Mapping[str, Any],
) -> None:
    if (
        isinstance(initialization, Mapping)
        and initialization.get("kind")
        == "acmpc_o2o_offline_continuation_v1"
    ):
        source_path_value = initialization.get("source_path")
        if not isinstance(source_path_value, str) or not source_path_value:
            raise ValueError("Offline continuation source path is invalid")
        source_path = Path(source_path_value).resolve()
        if source_path.name != "latest.pt" or not source_path.is_file():
            raise FileNotFoundError(
                f"Offline continuation latest.pt is missing: {source_path}"
            )
        source = load_checkpoint(source_path)
        source_config = _config_from_checkpoint(source)
        if source_config.method != config.method:
            raise ValueError("Offline continuation source method differs")
        source_fields = source_config.to_dict()
        target_fields = config.to_dict()
        for field in ("online_steps", "eval_interval_online_steps", "eval_episodes"):
            source_fields.pop(field)
            target_fields.pop(field)
        if source_fields != target_fields:
            raise ValueError("Offline continuation source config differs")
        if source.get("dataset", {}).get("sha256") != dataset_sha256:
            raise ValueError("Offline continuation source dataset differs")
        source_koopman = source.get("koopman")
        if koopman_sha256 is None:
            if source_koopman is not None:
                raise ValueError("Raw offline continuation source contains Koopman")
        elif (
            not isinstance(source_koopman, Mapping)
            or source_koopman.get("sha256") != koopman_sha256
        ):
            raise ValueError("Offline continuation source Koopman differs")
        if source.get("environment_protocol") != dict(environment_protocol):
            raise ValueError("Offline continuation source DMC protocol differs")
        if (
            source.get("phase") != "offline"
            or _validate_counter(source, "offline_update") != config.offline_updates
            or _validate_counter(source, "online_step") != 0
            or _validate_counter(source, "online_episode") != 0
            or source.get("initialization") is not None
        ):
            raise ValueError("Offline continuation source is not the final boundary")
        expected = {
            "kind": "acmpc_o2o_offline_continuation_v1",
            "source_path": str(source_path),
            "source_sha256": file_sha256(source_path),
            "source_method": source_config.method,
            "source_config_fingerprint": source_config.fingerprint,
            "target_config_fingerprint": config.fingerprint,
            "source_offline_update": config.offline_updates,
            "shared_state": "actor_critic_target_temperature_optimizers_replay_rng",
        }
        if dict(initialization) != expected:
            raise ValueError("Offline continuation lineage differs from its source")
        return
    if not config.requires_offline_fork:
        if initialization is not None:
            raise ValueError("Non-forking checkpoint contains fork lineage")
        return
    if not isinstance(initialization, Mapping):
        raise ValueError("MPVE checkpoint is missing offline-fork lineage")
    if koopman_sha256 is None:
        raise ValueError("MPVE checkpoint is missing its frozen Koopman identity")
    source_path_value = initialization.get("source_path")
    if not isinstance(source_path_value, str) or not source_path_value:
        raise ValueError("MPVE offline-fork source path is invalid")
    source_path = Path(source_path_value).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"MPVE offline-fork source is missing: {source_path}")
    source_sha256 = file_sha256(source_path)
    source = load_checkpoint(source_path)
    source_config = _config_from_checkpoint(source)
    if source_config.method != "Cal-RLPD-AC-KMPC":
        raise ValueError("MPVE source method is not Cal-RLPD-AC-KMPC")
    source_fields = source_config.to_dict()
    target_fields = config.to_dict()
    # The immutable offline snapshot is independent of the later online
    # interaction budget.  This permits a matrix-wide 50k -> N extension
    # without pretending that the offline initialization was retrained.
    for field in ("method", "online_steps"):
        source_fields.pop(field)
        target_fields.pop(field)
    if source_fields != target_fields:
        raise ValueError("MPVE source and target configs differ")
    if source.get("dataset", {}).get("sha256") != dataset_sha256:
        raise ValueError("MPVE source dataset differs")
    if source.get("koopman", {}).get("sha256") != koopman_sha256:
        raise ValueError("MPVE source Koopman model differs")
    if source.get("environment_protocol") != dict(environment_protocol):
        raise ValueError("MPVE source DMC protocol differs")
    if (
        source.get("phase") != "offline"
        or _validate_counter(source, "offline_update") != config.offline_updates
        or _validate_counter(source, "online_step") != 0
        or _validate_counter(source, "online_episode") != 0
        or source.get("initialization") is not None
    ):
        raise ValueError("MPVE source is not the immutable completed offline snapshot")
    expected = {
        "kind": "acmpc_o2o_offline_fork_v1",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_method": source_config.method,
        "source_config_fingerprint": source_config.fingerprint,
        "shared_state": "actor_critic_target_temperature_optimizers_rng",
    }
    if dict(initialization) != expected:
        raise ValueError("MPVE fork lineage differs from its current source artifact")


def validate_run_identity(
    run_dir: Path,
    *,
    checkpoint_name: str = "latest",
    dataset_override: Path | None = None,
    koopman_override: Path | None = None,
    load_artifacts: bool = True,
) -> ValidatedRun:
    """Cross-check run, checkpoint, config, dataset, and Koopman identities."""

    if checkpoint_name not in CHECKPOINT_NAMES and not _MILESTONE_CHECKPOINT.fullmatch(
        checkpoint_name
    ):
        raise ValueError(
            "checkpoint_name must be latest, best, or a stage milestone such as "
            "offline_050000/online_020000"
        )
    run_dir = run_dir.resolve()
    run_metadata = _read_mapping(run_dir / "run.json")
    if run_metadata.get("kind") != RUN_KIND:
        raise ValueError("Unsupported O2O run metadata kind")
    checkpoint_path = run_dir / f"{checkpoint_name}.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    config = _config_from_checkpoint(checkpoint)
    if run_metadata.get("config") != config.to_dict():
        raise ValueError("Run and checkpoint configs differ")
    if run_metadata.get("config_fingerprint") != config.fingerprint:
        raise ValueError("Run config fingerprint is invalid")
    expected_method_spec = dataclasses.asdict(config.method_spec)
    checkpoint_method_spec = checkpoint.get("method_spec")
    run_method_spec = run_metadata.get("method_spec")
    if checkpoint_method_spec is not None and checkpoint_method_spec != expected_method_spec:
        raise ValueError("Checkpoint immutable method specification differs")
    if run_method_spec is not None and run_method_spec != expected_method_spec:
        raise ValueError("Run immutable method specification differs")
    if run_metadata.get("initialization") != checkpoint.get("initialization"):
        raise ValueError("Run and checkpoint initialization lineage differ")

    checkpoint_dataset = _require_mapping(checkpoint, "dataset")
    run_dataset = _require_mapping(run_metadata, "dataset")
    if (
        checkpoint_dataset.get("path") != run_dataset.get("path")
        or checkpoint_dataset.get("sha256") != run_dataset.get("sha256")
    ):
        raise ValueError("Run and checkpoint dataset identities differ")
    expected_dataset_sha = checkpoint_dataset.get("sha256")
    if not isinstance(expected_dataset_sha, str) or len(expected_dataset_sha) != 64:
        raise ValueError("Checkpoint dataset SHA256 is invalid")
    dataset_path = _resolve_saved_path(
        checkpoint_dataset.get("path"), dataset_override, field="dataset"
    )

    checkpoint_koopman = checkpoint.get("koopman")
    run_koopman = run_metadata.get("koopman")
    if checkpoint_koopman != run_koopman:
        raise ValueError("Run and checkpoint Koopman identities differ")
    checkpoint_normalizer = checkpoint.get("raw_observation_normalizer")
    run_normalizer = run_metadata.get("raw_observation_normalizer")
    if checkpoint_normalizer != run_normalizer:
        raise ValueError("Run and checkpoint raw normalizer identities differ")
    observation_normalizer: FrozenObservationNormalizer | None = None
    if config.requires_koopman:
        if not isinstance(checkpoint_koopman, Mapping):
            raise ValueError("Structured checkpoint is missing Koopman identity")
        expected_koopman_sha = checkpoint_koopman.get("sha256")
        if not isinstance(expected_koopman_sha, str) or len(expected_koopman_sha) != 64:
            raise ValueError("Checkpoint Koopman SHA256 is invalid")
        koopman_path = _resolve_saved_path(
            checkpoint_koopman.get("path"), koopman_override, field="Koopman"
        )
        if checkpoint_normalizer is not None:
            raise ValueError("Structured checkpoint unexpectedly contains raw normalizer")
    else:
        if checkpoint_koopman is not None:
            raise ValueError("Raw baseline checkpoint unexpectedly contains Koopman")
        if koopman_override is not None:
            raise ValueError("Raw baseline evaluation forbids --koopman")
        expected_koopman_sha = None
        koopman_path = None
        observation_normalizer = _restore_raw_normalizer(
            checkpoint_normalizer, dataset_sha256=expected_dataset_sha
        )

    protocol = _require_mapping(checkpoint, "environment_protocol")
    if run_metadata.get("environment_protocol") != protocol:
        raise ValueError("Run and checkpoint environment protocols differ")
    if protocol.get("task") != config.task:
        raise ValueError("Saved environment protocol task differs from config")
    _validate_initialization_lineage(
        config=config,
        initialization=checkpoint.get("initialization"),
        dataset_sha256=expected_dataset_sha,
        koopman_sha256=expected_koopman_sha,
        environment_protocol=protocol,
    )
    if checkpoint.get("phase") not in {"offline", "online"}:
        raise ValueError("Checkpoint phase is invalid")
    _validate_counter(checkpoint, "offline_update")
    _validate_counter(checkpoint, "online_step")
    _validate_counter(checkpoint, "online_episode")
    learner_state = checkpoint.get("learner")
    if not isinstance(learner_state, Mapping):
        raise ValueError("Checkpoint is missing learner state")
    representation = learner_state.get("representation")
    if config.requires_koopman:
        if not isinstance(checkpoint_koopman, Mapping):
            raise AssertionError("Structured identity validation drifted")
        architecture = checkpoint_koopman.get("architecture")
        if not isinstance(architecture, Mapping):
            raise ValueError("Koopman architecture identity is missing")
        expected_representation = {
            "kind": "koopman_lifted_state_v1",
            "state_dim": architecture.get("state_dim"),
            "lift_dim": architecture.get("lift_dim"),
            "input_dim": (
                int(architecture.get("state_dim")) + int(architecture.get("lift_dim"))
                if isinstance(architecture.get("state_dim"), int)
                and isinstance(architecture.get("lift_dim"), int)
                else None
            ),
            "koopman_sha256": expected_koopman_sha,
        }
    else:
        assert observation_normalizer is not None
        expected_representation = {
            "kind": "normalized_raw_observation_v1",
            "input_dim": observation_normalizer.observation_dim,
            "normalizer": observation_normalizer.identity(),
        }
    if representation != expected_representation:
        raise ValueError("Learner representation identity differs from run artifacts")

    dataset: OfflineDataset | None = None
    koopman: FrozenKoopman | None = None
    if load_artifacts:
        dataset = OfflineDataset.load(dataset_path)
        if dataset.sha256 != expected_dataset_sha:
            raise ValueError("Offline dataset SHA256 differs from checkpoint")
        if checkpoint_dataset.get("metadata") != dataset.metadata:
            raise ValueError("Offline dataset metadata differs from checkpoint")
        if koopman_path is not None:
            koopman = FrozenKoopman(koopman_path)
            if koopman.sha256 != expected_koopman_sha:
                raise ValueError("Koopman SHA256 differs from checkpoint")
            actual_koopman = koopman.identity()
            assert isinstance(checkpoint_koopman, Mapping)
            for key in (
                "sha256",
                "architecture",
                "best_validation_rollout_normalized_mse",
            ):
                if checkpoint_koopman.get(key) != actual_koopman.get(key):
                    raise ValueError(f"Koopman identity field {key!r} differs")
            task_spec = get_task_spec(config.task)
            if (koopman.state_dim, koopman.action_dim) != (
                task_spec.obs_dim,
                task_spec.action_dim,
            ):
                raise ValueError(
                    f"{config.task} O2O evaluation requires Koopman dimensions "
                    f"{task_spec.obs_dim}/{task_spec.action_dim}"
                )
    else:
        if not dataset_path.is_file() or file_sha256(dataset_path) != expected_dataset_sha:
            raise ValueError("Offline dataset SHA256 differs from checkpoint")
        if koopman_path is not None and (
            not koopman_path.is_file()
            or file_sha256(koopman_path) != expected_koopman_sha
        ):
            raise ValueError("Koopman SHA256 differs from checkpoint")

    return ValidatedRun(
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=file_sha256(checkpoint_path),
        run_metadata=run_metadata,
        checkpoint=checkpoint,
        config=config,
        dataset=dataset,
        dataset_path=dataset_path,
        dataset_sha256=expected_dataset_sha,
        koopman=koopman,
        koopman_path=koopman_path,
        koopman_sha256=expected_koopman_sha,
        observation_normalizer=observation_normalizer,
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    return torch.device(name)


@torch.no_grad()
def evaluate_checkpoint(
    run_dir: Path,
    *,
    checkpoint_name: str = "latest",
    dataset_override: Path | None = None,
    koopman_override: Path | None = None,
    device_name: str = "cpu",
    episodes_per_seed: int = 1,
) -> dict[str, Any]:
    """Evaluate ten fixed reset seeds, optionally repeating each seed.

    ``episodes_per_seed=1`` is the inexpensive reference protocol used by
    training checkpoints.  ``episodes_per_seed=10`` is the robustness
    protocol (100 total episodes) and is deliberately kept out of training
    budgets.
    """
    if isinstance(episodes_per_seed, bool) or not isinstance(episodes_per_seed, int):
        raise TypeError("episodes_per_seed must be an integer")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive")

    validated = validate_run_identity(
        run_dir,
        checkpoint_name=checkpoint_name,
        dataset_override=dataset_override,
        koopman_override=koopman_override,
        load_artifacts=True,
    )
    device = _device(device_name)
    learner = O2OLearner(
        validated.config,
        validated.koopman,
        device,
        observation_normalizer=validated.observation_normalizer,
    )
    # CUDA Philox and CPU MT19937 generator states are intentionally not
    # interchangeable.  Evaluation is deterministic and consumes no sampling
    # noise, so load all learned/optimizer state while skipping only that
    # device-specific training RNG state.
    learner.load_state_dict(
        validated.checkpoint["learner"], restore_sampling_rng=False
    )

    expected_protocol = validated.checkpoint["environment_protocol"]
    action_repeat = expected_protocol.get("action_repeat")
    action_dim = get_task_spec(validated.config.task).action_dim
    returns: list[float] = []
    lengths: list[int] = []
    episode_seeds: list[int] = []
    for seed_index in range(EVALUATION_EPISODES):
        # Derive independent deterministic episode resets while retaining a
        # stable ten-seed grouping for reference/robustness summaries.
        for episode_index in range(episodes_per_seed):
            reset_seed = (
                EVALUATION_SEED_BASE
                + seed_index * episodes_per_seed
                + episode_index
            )
            episode_seeds.append(reset_seed)
            env_kwargs: dict[str, Any] = {"seed": reset_seed}
            if action_repeat is not None:
                env_kwargs["action_repeat"] = int(action_repeat)
            env = make_dmc_adapter(validated.config.task, **env_kwargs)
            try:
                runtime_protocol = env.protocol_metadata()
                if runtime_protocol != expected_protocol:
                    raise ValueError("Live DMC protocol differs from checkpoint")
                observation = env.reset(seed=reset_seed)
                episode_return = 0.0
                finished = False
                for step in range(int(env.step_limit)):
                    action = np.asarray(
                        learner.act(observation, deterministic=True)[0],
                        dtype=np.float32,
                    )
                    if action.shape != (action_dim,) or not np.isfinite(action).all():
                        raise RuntimeError(
                            f"Policy emitted an invalid {validated.config.task} action"
                        )
                    observation, reward, done, _info = env.step(action)
                    if not math.isfinite(float(reward)):
                        raise RuntimeError("DMC emitted a non-finite reward")
                    episode_return += float(reward)
                    if done:
                        lengths.append(step + 1)
                        finished = True
                        break
                if not finished:
                    raise RuntimeError(
                        "DMC episode did not finish at its saved step limit"
                    )
                returns.append(episode_return)
            finally:
                env.close()

    values = np.asarray(returns, dtype=np.float64)
    result = {
        "kind": EVALUATION_KIND,
        "task": validated.config.task,
        "method": validated.config.method,
        "training_seed": validated.config.seed,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(validated.checkpoint_path),
        "checkpoint_sha256": validated.checkpoint_sha256,
        "checkpoint_phase": validated.checkpoint["phase"],
        "offline_update": int(validated.checkpoint["offline_update"]),
        "online_step": int(validated.checkpoint["online_step"]),
        "config_fingerprint": validated.config.fingerprint,
        "dataset": {
            "path": str(validated.dataset_path),
            "sha256": validated.dataset_sha256,
        },
        "koopman": (
            {
                "path": str(validated.koopman_path),
                "sha256": validated.koopman_sha256,
            }
            if validated.koopman_path is not None
            else None
        ),
        "raw_observation_normalizer": (
            validated.observation_normalizer.identity()
            if validated.observation_normalizer is not None
            else None
        ),
        "environment_protocol": expected_protocol,
        "initialization": validated.checkpoint.get("initialization"),
        "evaluation_protocol": {
            "deterministic": True,
            "evaluation_seeds": EVALUATION_EPISODES,
            "episodes_per_seed": episodes_per_seed,
            "episodes": len(returns),
            "seed_base": EVALUATION_SEED_BASE,
            "episode_seeds": episode_seeds,
            "device": str(device),
        },
        "returns": returns,
        "return_mean": float(values.mean()),
        "return_std_population": float(values.std(ddof=0)),
        "return_min": float(values.min()),
        "return_max": float(values.max()),
        "return_median": float(np.median(values)),
        "episode_lengths": lengths,
        "episode_length_mean": float(np.mean(lengths)),
    }
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="latest, best, or a saved milestone such as offline_050000/online_020000",
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--koopman", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=1,
        help="repeat each of the 10 fixed evaluation seeds (10 gives 100 episodes)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.run_dir,
        checkpoint_name=args.checkpoint,
        dataset_override=args.dataset,
        koopman_override=args.koopman,
        device_name=args.device,
        episodes_per_seed=args.episodes_per_seed,
    )
    output = args.output or (
        args.run_dir / f"evaluation_{args.checkpoint}_{EVALUATION_EPISODES}.json"
    )
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
