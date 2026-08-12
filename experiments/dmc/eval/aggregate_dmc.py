"""Aggregate the three nested statistical levels of the DMC protocol.

Independent training seeds are the only inferential axis.  Evaluation-seed
means are repeated measurements nested within one trained policy, and episodes
are nested within an evaluation seed.  Neither lower level is used to inflate
the number of independent training replicates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.dmc.actors import ACTOR_TYPES, ActorConfig
from experiments.dmc.config import (
    ExperimentConfig,
    PROFILE_NAMES,
    default_config_path,
    load_experiment_config,
    resolve_execution_spec,
)
from experiments.dmc.eval.evaluate_dmc import (
    ActorCheckpointMetadata,
    KOOPMAN_LINEAGE_FIELDS,
    _authorization_metadata,
    _write_json_atomic,
    _validate_saved_protocol,
    evaluate,
    load_actor_checkpoint,
    validate_runtime_protocol,
)


AGGREGATE_SCHEMA_VERSION = "dmc_evaluation_aggregate_v1"
TRAINING_SEED_AGGREGATE_SCHEMA_VERSION = "dmc_training_seed_aggregate_v1"
CANONICAL_EVAL_SEED_COUNT = 10
STUDENT_T_95_CRITICAL_DF2 = 4.302652729911275


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return result


def _canonical_mapping(value: Any, *, field: str) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    mapping = dict(value)
    try:
        encoded = json.dumps(
            mapping,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain finite JSON data") from exc
    return mapping, encoded


def _validate_finite_json_tree(value: Any, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string mapping key")
            _validate_finite_json_tree(item, field=f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_json_tree(item, field=f"{field}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    if value is None or isinstance(
        value, (str, bool, int, float, np.bool_, np.integer, np.floating)
    ):
        return
    raise ValueError(f"{field} contains non-JSON value {type(value).__name__}")


def _integer_sequence(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence of integers")
    result = [
        _integer(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def _episode_returns(value: Any, *, field: str, expected_count: int) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence")
    returns = [
        _finite_float(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(returns) != expected_count:
        raise ValueError(
            f"{field} has {len(returns)} episodes, expected {expected_count}"
        )
    return returns


def _integer_count_sequence(
    value: Any,
    *,
    field: str,
    expected_count: int,
    minimum: int,
) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence of integer counts")
    counts = [
        _integer(item, field=f"{field}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    ]
    if len(counts) != expected_count:
        raise ValueError(
            f"{field} has {len(counts)} entries, expected {expected_count}"
        )
    return counts


def _validate_action_fraction_contract(
    row: Mapping[str, Any],
    *,
    prefix: str,
    expected_action_count: int,
    expected_applied_count: int,
    field_prefix: str,
) -> None:
    action_count = _integer(
        row.get(f"{prefix}_action_component_count"),
        field=f"{field_prefix}.{prefix}_action_component_count",
        minimum=1,
    )
    applied_count = _integer(
        row.get(f"{prefix}_applied_action_bound_count"),
        field=f"{field_prefix}.{prefix}_applied_action_bound_count",
        minimum=0,
    )
    if action_count != expected_action_count or applied_count != expected_applied_count:
        raise ValueError(f"{field_prefix} {prefix} action counts are inconsistent")
    fraction = _finite_float(
        row.get(f"{prefix}_applied_action_bound_fraction"),
        field=f"{field_prefix}.{prefix}_applied_action_bound_fraction",
    )
    if not 0.0 <= fraction <= 1.0 or not math.isclose(
        fraction,
        applied_count / action_count,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{field_prefix} {prefix} action fraction is inconsistent")


def _training_seed_stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) not in (1, 3) or not np.isfinite(array).all():
        raise ValueError("Training-seed statistics require one or three finite means")
    mean = float(np.mean(array))
    if len(array) == 1:
        return {
            "unit": "independently trained policy",
            "count": 1,
            "return_mean": mean,
            "return_sample_std": None,
            "return_standard_error": None,
            "return_student_t_95ci": None,
            "inference_estimable": False,
            "reason": (
                "One training seed cannot estimate between-training variability; "
                "eval seeds and episodes are nested observations, not replicates."
            ),
        }
    sample_std = float(np.std(array, ddof=1))
    standard_error = sample_std / math.sqrt(len(array))
    half_width = STUDENT_T_95_CRITICAL_DF2 * standard_error
    return {
        "unit": "independently trained policy",
        "count": 3,
        "return_mean": mean,
        "return_sample_std": sample_std,
        "return_standard_error": standard_error,
        "return_student_t_95ci": [mean - half_width, mean + half_width],
        "inference_estimable": True,
        "degrees_of_freedom": 2,
        "critical_value": STUDENT_T_95_CRITICAL_DF2,
    }


def _checkpoint_seed_plan(payload: Mapping[str, Any]) -> list[Any] | None:
    direct = payload.get("evaluation_seeds")
    if direct is not None:
        if not isinstance(direct, (list, tuple)):
            raise ValueError("Checkpoint evaluation_seeds must be a sequence")
        return list(direct)
    for key in ("experiment_config", "experiment"):
        experiment = payload.get(key)
        if isinstance(experiment, dict):
            seeds = experiment.get("seeds")
            if isinstance(seeds, dict) and seeds.get("evaluation") is not None:
                values = seeds["evaluation"]
                if not isinstance(values, (list, tuple)):
                    raise ValueError(
                        "Checkpoint experiment evaluation seeds must be a sequence"
                    )
                return list(values)
    return None


def _checkpoint_episode_budget(payload: Mapping[str, Any]) -> Any:
    direct = payload.get("evaluation_episodes_per_seed")
    if direct is not None:
        return direct
    for key in ("experiment_config", "experiment"):
        experiment = payload.get(key)
        if isinstance(experiment, dict):
            evaluation = experiment.get("evaluation")
            if isinstance(evaluation, dict) and evaluation.get("episodes_per_seed"):
                return evaluation["episodes_per_seed"]
    return None


def canonical_evaluation_plan(
    actor_checkpoint: str | Path,
    *,
    eval_seeds: Sequence[int] | None = None,
    episodes_per_seed: int | None = None,
) -> tuple[list[int], int, str]:
    """Resolve the saved or per-task canonical evaluation plan."""

    metadata = load_actor_checkpoint(actor_checkpoint, map_location="cpu")
    source = "explicit"
    if eval_seeds is None:
        resolved = _checkpoint_seed_plan(metadata.payload)
        source = "actor_checkpoint"
        if resolved is None:
            config = load_experiment_config(default_config_path(metadata.task))
            resolved = [int(value) for value in config.raw["seeds"]["evaluation"]]
            source = str(config.path.resolve())
        eval_seeds = resolved
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in eval_seeds
    ):
        raise ValueError("Every evaluation seed must be an integer")
    seeds = [int(value) for value in eval_seeds]
    if any(seed < 0 for seed in seeds):
        raise ValueError("Evaluation seeds must be non-negative")
    if len(seeds) != CANONICAL_EVAL_SEED_COUNT:
        raise ValueError(
            f"Canonical evaluation requires exactly {CANONICAL_EVAL_SEED_COUNT} "
            f"eval seeds, got {len(seeds)}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique")
    if metadata.training_seed in seeds:
        raise ValueError("Training seed must not appear in the evaluation seed set")

    if episodes_per_seed is None:
        episodes_per_seed = _checkpoint_episode_budget(metadata.payload)
        if episodes_per_seed is None:
            config = load_experiment_config(default_config_path(metadata.task))
            episodes_per_seed = int(config.raw["evaluation"]["episodes_per_seed"])
    if isinstance(episodes_per_seed, bool) or not isinstance(
        episodes_per_seed, (int, np.integer)
    ):
        raise ValueError("episodes_per_seed must be an integer")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive")
    return seeds, int(episodes_per_seed), source


def _require_verified_checkpoint_plan(
    metadata: ActorCheckpointMetadata,
    *,
    eval_seeds: Sequence[int],
    episodes_per_seed: int,
) -> None:
    """Prevent a verified checkpoint from silently evaluating another plan."""

    if not metadata.authorization_verified:
        return
    saved_seeds = _integer_sequence(
        metadata.payload.get("evaluation_seeds"),
        field="checkpoint evaluation_seeds",
    )
    if saved_seeds != list(eval_seeds):
        raise ValueError(
            "Formal checkpoint evaluation_seeds do not match the requested plan"
        )
    saved_episodes = _integer(
        metadata.payload.get("evaluation_episodes_per_seed"),
        field="checkpoint evaluation_episodes_per_seed",
        minimum=1,
    )
    if saved_episodes != episodes_per_seed:
        raise ValueError(
            "Formal checkpoint evaluation_episodes_per_seed does not match the "
            "requested plan"
        )
    execution_spec, _ = _canonical_mapping(
        metadata.payload.get("resolved_execution_spec"),
        field="checkpoint resolved_execution_spec",
    )
    if execution_spec.get("evaluation_seeds") != saved_seeds:
        raise ValueError(
            "Checkpoint evaluation_seeds disagree with resolved_execution_spec"
        )
    evaluation = execution_spec.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get(
        "episodes_per_seed"
    ) != saved_episodes:
        raise ValueError(
            "Checkpoint episode budget disagrees with resolved_execution_spec"
        )
    saved_reference_episodes = _integer(
        metadata.payload.get("evaluation_reference_episodes_per_seed"),
        field="checkpoint evaluation_reference_episodes_per_seed",
        minimum=1,
    )
    if saved_reference_episodes > saved_episodes or evaluation.get(
        "reference_episodes_per_seed"
    ) != saved_reference_episodes:
        raise ValueError(
            "Checkpoint reference episode budget disagrees with resolved_execution_spec"
        )
    saved_diagnostic_every = _integer(
        metadata.payload.get("diagnostic_every_steps"),
        field="checkpoint diagnostic_every_steps",
        minimum=1,
    )
    if evaluation.get("diagnostic_every_steps") != saved_diagnostic_every:
        raise ValueError(
            "Checkpoint diagnostic cadence disagrees with resolved_execution_spec"
        )


def _same_protocol(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    _, left_json = _canonical_mapping(left, field="left protocol")
    _, right_json = _canonical_mapping(right, field="right protocol")
    return left_json == right_json


def _validate_evaluation_rows(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    task: str,
    actor_type: str,
    training_seed: int,
    eval_seeds: Sequence[int],
    episodes_per_seed: int,
    reference_episodes_per_seed: int,
    protocol: Mapping[str, Any],
    actor_config: Mapping[str, Any] | None = None,
    authorization: Mapping[str, Any] | None = None,
    checkpoint_lineage: Mapping[str, Any] | None = None,
    runtime_protocol: Mapping[str, Any] | None = None,
) -> tuple[list[float], list[float], dict[str, Any]]:
    if not isinstance(per_seed, (list, tuple)) or len(per_seed) != len(eval_seeds):
        raise ValueError("per_eval_seed must contain exactly one row per eval seed")
    expected_runtime_json: str | None = None
    expected_runtime: dict[str, Any] | None = None
    expected_components: set[str] | None = None
    eval_means: list[float] = []
    pooled_returns: list[float] = []
    seen_eval_seeds: set[int] = set()
    if not 1 <= reference_episodes_per_seed <= episodes_per_seed:
        raise ValueError("Reference episode count must lie within each eval row")
    for index, (row_value, expected_eval_seed) in enumerate(
        zip(per_seed, eval_seeds, strict=True)
    ):
        if not isinstance(row_value, Mapping):
            raise ValueError(f"per_eval_seed[{index}] must be a mapping")
        row = dict(row_value)
        _validate_finite_json_tree(row, field=f"per_eval_seed[{index}]")
        if row.get("task") != task or row.get("actor_type") != actor_type:
            raise ValueError("Evaluation row task/actor metadata is inconsistent")
        row_training_seed = _integer(
            row.get("training_seed"),
            field=f"per_eval_seed[{index}].training_seed",
        )
        if row_training_seed != training_seed:
            raise ValueError("Evaluation row training_seed is inconsistent")
        row_eval_seed = _integer(
            row.get("eval_seed"), field=f"per_eval_seed[{index}].eval_seed"
        )
        if row_eval_seed != expected_eval_seed or row_eval_seed in seen_eval_seeds:
            raise ValueError("Evaluation rows do not match the eval seed plan")
        seen_eval_seeds.add(row_eval_seed)

        if "protocol" in row:
            _, row_protocol_json = _canonical_mapping(
                row["protocol"], field=f"per_eval_seed[{index}].protocol"
            )
            _, expected_protocol_json = _canonical_mapping(
                protocol, field="protocol"
            )
            if row_protocol_json != expected_protocol_json:
                raise ValueError("Evaluation row protocol is inconsistent")
        if actor_config is not None and "actor_config" in row:
            _, row_actor_config_json = _canonical_mapping(
                row["actor_config"], field=f"per_eval_seed[{index}].actor_config"
            )
            _, expected_actor_config_json = _canonical_mapping(
                actor_config, field="actor_config"
            )
            if row_actor_config_json != expected_actor_config_json:
                raise ValueError("Evaluation row actor_config is inconsistent")
        if authorization is not None and authorization.get(
            "authorization_verified"
        ) is True:
            row_authorization = _authorization_metadata(row)
            for field in (
                "authorization_kind",
                "training_approved",
                "config_fingerprint",
                "approval_profile",
                "approval_file_sha256",
                "preflight_report_sha256",
                "train_seed_index",
            ):
                if row_authorization[field] != authorization.get(field):
                    raise ValueError(
                        f"Evaluation row authorization field {field} is inconsistent"
                    )
            if row.get("authorization_verified") is not True or not row_authorization[
                "authorization_verified"
            ]:
                raise ValueError("Evaluation row authorization is unverified")
            if row.get("authorization_errors") not in ([], ()):
                raise ValueError(
                    "Verified evaluation row must have no authorization_errors"
                )
            if checkpoint_lineage is None:
                raise ValueError(
                    "Verified authorization requires checkpoint execution lineage"
                )
            for field in (
                "resolved_execution_spec",
                "evaluation_seeds",
                "evaluation_episodes_per_seed",
                "evaluation_reference_episodes_per_seed",
                "diagnostic_every_steps",
                "koopman_sha256",
                "koopman_lineage",
                "koopman_dataset_sha256",
                "koopman_config_fingerprint",
                "value_expansion",
            ):
                if row.get(field) != checkpoint_lineage.get(field):
                    raise ValueError(
                        f"Evaluation row checkpoint lineage field {field} is "
                        "inconsistent"
                    )

        row_runtime, row_runtime_json = _canonical_mapping(
            row.get("runtime_protocol"),
            field=f"per_eval_seed[{index}].runtime_protocol",
        )
        validate_runtime_protocol(protocol, row_runtime, task)
        if expected_runtime_json is None:
            expected_runtime = row_runtime
            expected_runtime_json = row_runtime_json
        elif row_runtime_json != expected_runtime_json:
            raise ValueError("Runtime protocol drifted between evaluation seeds")

        returns = _episode_returns(
            row.get("episode_returns"),
            field=f"per_eval_seed[{index}].episode_returns",
            expected_count=episodes_per_seed,
        )
        row_mean = _finite_float(
            row.get("return_mean_across_episodes"),
            field=f"per_eval_seed[{index}].return_mean_across_episodes",
        )
        if not math.isclose(
            row_mean,
            float(np.mean(returns)),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("Evaluation row mean does not match episode_returns")
        if "episodes" in row and _integer(
            row["episodes"], field=f"per_eval_seed[{index}].episodes", minimum=1
        ) != episodes_per_seed:
            raise ValueError("Evaluation row episode count is inconsistent")

        reference_returns = _episode_returns(
            row.get("acme_reference_episode_returns"),
            field=(
                f"per_eval_seed[{index}].acme_reference_episode_returns"
            ),
            expected_count=reference_episodes_per_seed,
        )
        if reference_returns != returns[:reference_episodes_per_seed]:
            raise ValueError(
                "Evaluation row reference returns are not the preregistered prefix"
            )
        if _integer(
            row.get("acme_reference_episode_count"),
            field=f"per_eval_seed[{index}].acme_reference_episode_count",
            minimum=1,
        ) != reference_episodes_per_seed:
            raise ValueError("Evaluation row reference episode count is inconsistent")
        reference_mean = _finite_float(
            row.get("acme_reference_return_mean"),
            field=f"per_eval_seed[{index}].acme_reference_return_mean",
        )
        if not math.isclose(
            reference_mean,
            float(np.mean(reference_returns)),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("Evaluation row reference mean is inconsistent")
        robustness_returns = _episode_returns(
            row.get("robustness_episode_returns"),
            field=f"per_eval_seed[{index}].robustness_episode_returns",
            expected_count=episodes_per_seed,
        )
        if robustness_returns != returns:
            raise ValueError("Evaluation row robustness returns are inconsistent")
        if _integer(
            row.get("robustness_episode_count"),
            field=f"per_eval_seed[{index}].robustness_episode_count",
            minimum=1,
        ) != episodes_per_seed:
            raise ValueError("Evaluation row robustness episode count is inconsistent")
        robustness_mean = _finite_float(
            row.get("robustness_return_mean"),
            field=f"per_eval_seed[{index}].robustness_return_mean",
        )
        if not math.isclose(
            robustness_mean,
            float(np.mean(returns)),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("Evaluation row robustness mean is inconsistent")

        episode_action_counts = _integer_count_sequence(
            row.get("episode_action_component_counts"),
            field=f"per_eval_seed[{index}].episode_action_component_counts",
            expected_count=episodes_per_seed,
            minimum=1,
        )
        episode_applied_bound_counts = _integer_count_sequence(
            row.get("episode_applied_action_bound_counts"),
            field=(
                f"per_eval_seed[{index}].episode_applied_action_bound_counts"
            ),
            expected_count=episodes_per_seed,
            minimum=0,
        )
        if any(
            bound > total
            for bound, total in zip(
                episode_applied_bound_counts,
                episode_action_counts,
                strict=True,
            )
        ):
            raise ValueError("Evaluation row applied-action count exceeds total")
        reference_action_count = sum(
            episode_action_counts[:reference_episodes_per_seed]
        )
        reference_applied_count = sum(
            episode_applied_bound_counts[:reference_episodes_per_seed]
        )
        robustness_action_count = sum(episode_action_counts)
        robustness_applied_count = sum(episode_applied_bound_counts)
        _validate_action_fraction_contract(
            row,
            prefix="acme_reference",
            expected_action_count=reference_action_count,
            expected_applied_count=reference_applied_count,
            field_prefix=f"per_eval_seed[{index}]",
        )
        _validate_action_fraction_contract(
            row,
            prefix="robustness",
            expected_action_count=robustness_action_count,
            expected_applied_count=robustness_applied_count,
            field_prefix=f"per_eval_seed[{index}]",
        )
        compatibility_fraction = _finite_float(
            row.get("applied_action_bound_fraction"),
            field=f"per_eval_seed[{index}].applied_action_bound_fraction",
        )
        if not math.isclose(
            compatibility_fraction,
            robustness_applied_count / robustness_action_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Compatibility action-bound fraction is inconsistent")

        components, _ = _canonical_mapping(
            row.get("mean_reward_components", {}),
            field=f"per_eval_seed[{index}].mean_reward_components",
        )
        for name, value in components.items():
            _finite_float(
                value,
                field=f"per_eval_seed[{index}].mean_reward_components.{name}",
            )
        component_keys = set(components)
        if expected_components is None:
            expected_components = component_keys
        elif component_keys != expected_components:
            raise ValueError("Reward component schema drifted between eval seeds")

        for field in (
            "episode_length_mean",
            "mean_step_reward",
            "requested_action_bound_fraction",
            "applied_action_bound_fraction",
            "action_clipped_fraction",
        ):
            if field in row:
                _finite_float(row[field], field=f"per_eval_seed[{index}].{field}")
        eval_means.append(row_mean)
        pooled_returns.extend(returns)

    if runtime_protocol is not None:
        _, declared_runtime_json = _canonical_mapping(
            runtime_protocol, field="runtime_protocol"
        )
        if declared_runtime_json != expected_runtime_json:
            raise ValueError("Aggregate runtime_protocol does not match evaluation rows")
    assert expected_runtime is not None
    return eval_means, pooled_returns, expected_runtime


def aggregate_evaluations(
    actor_checkpoint: str | Path,
    *,
    koopman_path: str | Path | None = None,
    eval_seeds: Sequence[int] | None = None,
    episodes_per_seed: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate one training seed across exactly ten independent eval seeds."""

    metadata = load_actor_checkpoint(actor_checkpoint, map_location="cpu")
    seeds, episodes, seed_plan_source = canonical_evaluation_plan(
        actor_checkpoint,
        eval_seeds=eval_seeds,
        episodes_per_seed=episodes_per_seed,
    )
    _require_verified_checkpoint_plan(
        metadata, eval_seeds=seeds, episodes_per_seed=episodes
    )
    per_seed = [
        evaluate(
            actor_checkpoint,
            koopman_path=koopman_path,
            episodes=episodes,
            eval_seed=seed,
            device_name=device_name,
        )
        for seed in seeds
    ]
    reference_episodes = _integer(
        metadata.payload.get("evaluation_reference_episodes_per_seed"),
        field="checkpoint evaluation_reference_episodes_per_seed",
        minimum=1,
    )
    if reference_episodes > episodes:
        raise ValueError("Reference episode count exceeds evaluation row budget")

    eval_means, episode_return_values, runtime_protocol = _validate_evaluation_rows(
        per_seed,
        task=metadata.task,
        actor_type=metadata.actor_type,
        training_seed=metadata.training_seed,
        eval_seeds=seeds,
        episodes_per_seed=episodes,
        reference_episodes_per_seed=reference_episodes,
        protocol=metadata.protocol,
        actor_config=metadata.actor_config.to_dict(),
        authorization=metadata.authorization,
        checkpoint_lineage={
            "resolved_execution_spec": metadata.payload.get(
                "resolved_execution_spec"
            ),
            "evaluation_seeds": metadata.payload.get("evaluation_seeds"),
            "evaluation_episodes_per_seed": metadata.payload.get(
                "evaluation_episodes_per_seed"
            ),
            "evaluation_reference_episodes_per_seed": metadata.payload.get(
                "evaluation_reference_episodes_per_seed"
            ),
            "diagnostic_every_steps": metadata.payload.get(
                "diagnostic_every_steps"
            ),
            "koopman_sha256": metadata.payload.get("koopman_sha256"),
            "koopman_lineage": metadata.payload.get("koopman_lineage"),
            "koopman_dataset_sha256": metadata.payload.get(
                "koopman_dataset_sha256"
            ),
            "koopman_config_fingerprint": metadata.payload.get(
                "koopman_config_fingerprint"
            ),
            "value_expansion": metadata.payload.get("value_expansion"),
        },
    )
    seed_means = np.asarray(eval_means, dtype=np.float64)
    pooled_returns = np.asarray(episode_return_values, dtype=np.float64)
    seed_std = float(np.std(seed_means, ddof=1))
    reference_return_values = [
        value
        for row in per_seed
        for value in row["acme_reference_episode_returns"]
    ]
    reference_returns = np.asarray(reference_return_values, dtype=np.float64)
    reference_action_count = sum(
        int(row["acme_reference_action_component_count"])
        for row in per_seed
    )
    reference_applied_count = sum(
        int(row["acme_reference_applied_action_bound_count"])
        for row in per_seed
    )
    robustness_action_count = sum(
        int(row["robustness_action_component_count"]) for row in per_seed
    )
    robustness_applied_count = sum(
        int(row["robustness_applied_action_bound_count"]) for row in per_seed
    )

    component_keys = sorted(per_seed[0]["mean_reward_components"])
    training_stats = _training_seed_stats(
        [float(np.mean(seed_means))]
    )
    report: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "kind": "dmc_ten_eval_seed_aggregate",
        "task": metadata.task,
        "actor_type": metadata.actor_type,
        "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
        "koopman_checkpoint": per_seed[0]["koopman_checkpoint"],
        "actor_config": metadata.actor_config.to_dict(),
        "training_seed": metadata.training_seed,
        **metadata.authorization,
        "resolved_execution_spec": metadata.payload.get(
            "resolved_execution_spec"
        ),
        "evaluation_seeds": metadata.payload.get("evaluation_seeds"),
        "evaluation_episodes_per_seed": metadata.payload.get(
            "evaluation_episodes_per_seed"
        ),
        "evaluation_reference_episodes_per_seed": metadata.payload.get(
            "evaluation_reference_episodes_per_seed"
        ),
        "diagnostic_every_steps": metadata.payload.get(
            "diagnostic_every_steps"
        ),
        "koopman_sha256": metadata.payload.get("koopman_sha256"),
        "koopman_lineage": metadata.payload.get("koopman_lineage"),
        "koopman_dataset_sha256": metadata.payload.get(
            "koopman_dataset_sha256"
        ),
        "koopman_config_fingerprint": metadata.payload.get(
            "koopman_config_fingerprint"
        ),
        "value_expansion": metadata.payload.get("value_expansion"),
        "training_seed_count": 1,
        "eval_seeds": seeds,
        "eval_seed_count": len(seeds),
        "eval_seed_plan_source": seed_plan_source,
        "episodes_per_eval_seed": episodes,
        "total_evaluation_episodes": int(len(seeds) * episodes),
        "protocol": metadata.protocol,
        "runtime_protocol": runtime_protocol,
        "return_mean_across_eval_seed_means": float(np.mean(seed_means)),
        "return_std_across_eval_seed_means": seed_std,
        # Eval-seed measurements are nested within this one trained policy.
        # Do not expose a pseudo-replicated inferential SE/CI at this level.
        "return_standard_error_across_eval_seed_means": None,
        "return_normal_95ci_across_eval_seed_means": None,
        "pooled_episode_return_mean": float(np.mean(pooled_returns)),
        "pooled_episode_return_std": float(np.std(pooled_returns, ddof=0)),
        "episode_length_mean_across_eval_seeds": float(
            np.mean([report["episode_length_mean"] for report in per_seed])
        ),
        "mean_step_reward_across_eval_seeds": float(
            np.mean([report["mean_step_reward"] for report in per_seed])
        ),
        "requested_action_bound_fraction_mean": float(
            np.mean(
                [report["requested_action_bound_fraction"] for report in per_seed]
            )
        ),
        "applied_action_bound_fraction_mean": float(
            np.mean(
                [report["applied_action_bound_fraction"] for report in per_seed]
            )
        ),
        "action_clipped_fraction_mean": float(
            np.mean([report["action_clipped_fraction"] for report in per_seed])
        ),
        "terminated_episodes": int(
            sum(report["terminated_episodes"] for report in per_seed)
        ),
        "truncated_episodes": int(
            sum(report["truncated_episodes"] for report in per_seed)
        ),
        "mean_reward_components_across_eval_seeds": {
            key: float(
                np.mean(
                    [
                        report["mean_reward_components"][key]
                        for report in per_seed
                    ]
                )
            )
            for key in component_keys
        },
        "inference_axis": "training_seed",
        "training_seed_statistics": training_stats,
        "eval_seed_statistics": {
            "unit": "evaluation-seed mean nested within one training seed",
            "count": len(seed_means),
            "return_mean": float(np.mean(seed_means)),
            "return_sample_std": seed_std,
            "descriptive_only": True,
            "return_standard_error": None,
            "return_95ci": None,
        },
        "episode_statistics": {
            "unit": "episode nested within an eval seed and training seed",
            "count": len(pooled_returns),
            "return_mean": float(np.mean(pooled_returns)),
            "return_population_std": float(np.std(pooled_returns, ddof=0)),
            "descriptive_only": True,
            "return_standard_error": None,
            "return_95ci": None,
        },
        "acme_reference_summary": {
            "kind": "acme_aligned_preregistered_reference_v1",
            "episode_selection": "first_episode_per_eval_seed_prefix_v1",
            "eval_seed_count": len(seeds),
            "episodes_per_eval_seed": reference_episodes,
            "episode_count": len(reference_returns),
            "episode_returns": reference_returns.tolist(),
            "return_mean": float(np.mean(reference_returns)),
            "return_population_std": float(np.std(reference_returns, ddof=0)),
            "action_component_count": reference_action_count,
            "applied_action_bound_count": reference_applied_count,
            "applied_action_bound_fraction": float(
                reference_applied_count / reference_action_count
            ),
            "descriptive_only": True,
            "return_standard_error": None,
            "return_95ci": None,
        },
        "robustness_summary": {
            "kind": "dmc_nested_episode_robustness_v1",
            "episode_selection": "all_episodes_per_eval_seed_v1",
            "eval_seed_count": len(seeds),
            "episodes_per_eval_seed": episodes,
            "episode_count": len(pooled_returns),
            "episode_returns": pooled_returns.tolist(),
            "return_mean": float(np.mean(pooled_returns)),
            "return_population_std": float(np.std(pooled_returns, ddof=0)),
            "action_component_count": robustness_action_count,
            "applied_action_bound_count": robustness_applied_count,
            "applied_action_bound_fraction": float(
                robustness_applied_count / robustness_action_count
            ),
            "descriptive_only": True,
            "return_standard_error": None,
            "return_95ci": None,
        },
        "per_eval_seed": per_seed,
    }
    return report


def _validated_single_training_cohort_summary(
    value: Any,
    *,
    field: str,
    expected_kind: str,
    expected_selection: str,
    eval_seed_count: int,
    episodes_per_eval_seed: int,
    expected_returns: Sequence[float],
    expected_action_count: int,
    expected_applied_count: int,
) -> dict[str, Any]:
    summary, _ = _canonical_mapping(value, field=field)
    if summary.get("kind") != expected_kind or summary.get(
        "episode_selection"
    ) != expected_selection:
        raise ValueError(f"{field} has the wrong cohort identity")
    if _integer(
        summary.get("eval_seed_count"),
        field=f"{field}.eval_seed_count",
        minimum=1,
    ) != eval_seed_count:
        raise ValueError(f"{field} eval seed count is inconsistent")
    if _integer(
        summary.get("episodes_per_eval_seed"),
        field=f"{field}.episodes_per_eval_seed",
        minimum=1,
    ) != episodes_per_eval_seed:
        raise ValueError(f"{field} episode budget is inconsistent")
    returns = _episode_returns(
        summary.get("episode_returns"),
        field=f"{field}.episode_returns",
        expected_count=len(expected_returns),
    )
    if returns != list(expected_returns):
        raise ValueError(f"{field} returns do not match evaluation rows")
    if _integer(
        summary.get("episode_count"),
        field=f"{field}.episode_count",
        minimum=1,
    ) != len(returns):
        raise ValueError(f"{field} episode count is inconsistent")
    mean = _finite_float(summary.get("return_mean"), field=f"{field}.return_mean")
    population_std = _finite_float(
        summary.get("return_population_std"),
        field=f"{field}.return_population_std",
    )
    if not math.isclose(mean, float(np.mean(returns)), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{field} return mean is inconsistent")
    if not math.isclose(
        population_std,
        float(np.std(returns, ddof=0)),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{field} return std is inconsistent")
    action_count = _integer(
        summary.get("action_component_count"),
        field=f"{field}.action_component_count",
        minimum=1,
    )
    applied_count = _integer(
        summary.get("applied_action_bound_count"),
        field=f"{field}.applied_action_bound_count",
        minimum=0,
    )
    applied_fraction = _finite_float(
        summary.get("applied_action_bound_fraction"),
        field=f"{field}.applied_action_bound_fraction",
    )
    if action_count != expected_action_count or applied_count != expected_applied_count:
        raise ValueError(f"{field} action counts do not match evaluation rows")
    if not 0.0 <= applied_fraction <= 1.0 or not math.isclose(
        applied_fraction,
        applied_count / action_count,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{field} action fraction is inconsistent")
    if summary.get("descriptive_only") is not True or summary.get(
        "return_standard_error"
    ) is not None or summary.get("return_95ci") is not None:
        raise ValueError(f"{field} must remain descriptive at one training seed")
    return summary


def _validated_training_seed_report(report_value: Any) -> dict[str, Any]:
    if not isinstance(report_value, Mapping):
        raise ValueError("Every training-seed aggregate must be a mapping")
    report = dict(report_value)
    _validate_finite_json_tree(report, field="training_seed_report")
    if report.get("schema_version") != AGGREGATE_SCHEMA_VERSION:
        raise ValueError("Unsupported single-training-seed aggregate schema")
    if report.get("kind") != "dmc_ten_eval_seed_aggregate":
        raise ValueError("Input is not a dmc_ten_eval_seed_aggregate")
    if _integer(
        report.get("training_seed_count"), field="training_seed_count", minimum=1
    ) != 1:
        raise ValueError("Each input must represent exactly one training seed")
    task = report.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError("Aggregate task must be a non-empty string")
    actor_type = report.get("actor_type")
    if actor_type not in ACTOR_TYPES:
        raise ValueError(f"Aggregate actor_type must be one of {ACTOR_TYPES}")
    training_seed = _integer(report.get("training_seed"), field="training_seed")
    authorization = _authorization_metadata(report)
    declared_authorization = report.get("authorization_verified", False)
    if not isinstance(declared_authorization, bool):
        raise ValueError("authorization_verified must be a boolean")
    if declared_authorization and not authorization["authorization_verified"]:
        raise ValueError(
            "Aggregate claims authorization_verified with incomplete identity"
        )
    if declared_authorization and report.get("authorization_errors") not in ([], ()):
        raise ValueError(
            "Verified aggregate must have no authorization_errors"
        )
    authorization["authorization_verified"] = bool(
        declared_authorization and authorization["authorization_verified"]
    )
    authorization_errors = list(authorization["authorization_errors"])
    if not declared_authorization:
        authorization_errors.append(
            "single-training-seed report did not verify checkpoint authorization"
        )
    authorization["authorization_errors"] = tuple(authorization_errors)

    actor_config, actor_config_json = _canonical_mapping(
        report.get("actor_config"), field="actor_config"
    )
    parsed_actor_config = ActorConfig.from_mapping(actor_config).to_dict()
    _, normalized_actor_config_json = _canonical_mapping(
        parsed_actor_config, field="actor_config"
    )
    if set(actor_config) != set(parsed_actor_config) or (
        actor_config_json != normalized_actor_config_json
    ):
        raise ValueError("actor_config must contain the complete canonical architecture")

    protocol, protocol_json = _canonical_mapping(
        report.get("protocol"), field="protocol"
    )
    _validate_saved_protocol(protocol, task)
    runtime_protocol, runtime_protocol_json = _canonical_mapping(
        report.get("runtime_protocol"), field="runtime_protocol"
    )
    validate_runtime_protocol(protocol, runtime_protocol, task)

    eval_seeds = _integer_sequence(report.get("eval_seeds"), field="eval_seeds")
    if len(eval_seeds) != CANONICAL_EVAL_SEED_COUNT:
        raise ValueError(
            f"Every training seed requires {CANONICAL_EVAL_SEED_COUNT} eval seeds"
        )
    if training_seed in eval_seeds:
        raise ValueError("Training seed must not appear in the eval seed plan")
    if _integer(
        report.get("eval_seed_count"), field="eval_seed_count", minimum=1
    ) != len(eval_seeds):
        raise ValueError("eval_seed_count does not match eval_seeds")
    episodes_per_seed = _integer(
        report.get("episodes_per_eval_seed"),
        field="episodes_per_eval_seed",
        minimum=1,
    )
    expected_episode_count = len(eval_seeds) * episodes_per_seed
    if _integer(
        report.get("total_evaluation_episodes"),
        field="total_evaluation_episodes",
        minimum=1,
    ) != expected_episode_count:
        raise ValueError("total_evaluation_episodes is inconsistent")
    reference_episodes_per_seed = _integer(
        report.get("evaluation_reference_episodes_per_seed"),
        field="evaluation_reference_episodes_per_seed",
        minimum=1,
    )
    if reference_episodes_per_seed > episodes_per_seed:
        raise ValueError("Reference episode budget exceeds robustness budget")

    checkpoint_lineage = {
        "resolved_execution_spec": report.get("resolved_execution_spec"),
        "evaluation_seeds": report.get("evaluation_seeds"),
        "evaluation_episodes_per_seed": report.get(
            "evaluation_episodes_per_seed"
        ),
        "evaluation_reference_episodes_per_seed": report.get(
            "evaluation_reference_episodes_per_seed"
        ),
        "diagnostic_every_steps": report.get("diagnostic_every_steps"),
        "koopman_sha256": report.get("koopman_sha256"),
        "koopman_lineage": report.get("koopman_lineage"),
        "koopman_dataset_sha256": report.get("koopman_dataset_sha256"),
        "koopman_config_fingerprint": report.get(
            "koopman_config_fingerprint"
        ),
        "value_expansion": report.get("value_expansion"),
    }
    resolved_execution_spec_json: str | None = None
    if authorization["authorization_verified"]:
        _, resolved_execution_spec_json = _canonical_mapping(
            checkpoint_lineage["resolved_execution_spec"],
            field="resolved_execution_spec",
        )
        saved_eval_seeds = _integer_sequence(
            checkpoint_lineage["evaluation_seeds"],
            field="evaluation_seeds",
        )
        if saved_eval_seeds != eval_seeds:
            raise ValueError(
                "Checkpoint evaluation_seeds do not match aggregate eval_seeds"
            )
        if _integer(
            checkpoint_lineage["evaluation_episodes_per_seed"],
            field="evaluation_episodes_per_seed",
            minimum=1,
        ) != episodes_per_seed:
            raise ValueError(
                "Checkpoint evaluation_episodes_per_seed does not match aggregate"
            )
        if _integer(
            checkpoint_lineage["evaluation_reference_episodes_per_seed"],
            field="evaluation_reference_episodes_per_seed",
            minimum=1,
        ) != reference_episodes_per_seed:
            raise ValueError(
                "Checkpoint reference episode budget does not match aggregate"
            )
        diagnostic_every_steps = _integer(
            checkpoint_lineage["diagnostic_every_steps"],
            field="diagnostic_every_steps",
            minimum=1,
        )
        execution_spec = checkpoint_lineage["resolved_execution_spec"]
        evaluation_spec = execution_spec.get("evaluation")
        if not isinstance(evaluation_spec, Mapping) or (
            evaluation_spec.get("reference_episodes_per_seed")
            != reference_episodes_per_seed
            or evaluation_spec.get("diagnostic_every_steps")
            != diagnostic_every_steps
        ):
            raise ValueError(
                "Checkpoint reference/diagnostic plan disagrees with execution spec"
            )

    eval_means, episode_returns, _ = _validate_evaluation_rows(
        report.get("per_eval_seed"),
        task=task,
        actor_type=actor_type,
        training_seed=training_seed,
        eval_seeds=eval_seeds,
        episodes_per_seed=episodes_per_seed,
        reference_episodes_per_seed=reference_episodes_per_seed,
        protocol=protocol,
        actor_config=parsed_actor_config,
        authorization=authorization,
        checkpoint_lineage=checkpoint_lineage,
        runtime_protocol=runtime_protocol,
    )
    evaluation_rows = list(report["per_eval_seed"])
    reference_returns = [
        value
        for row in evaluation_rows
        for value in row["acme_reference_episode_returns"]
    ]
    reference_action_count = sum(
        int(row["acme_reference_action_component_count"])
        for row in evaluation_rows
    )
    reference_applied_count = sum(
        int(row["acme_reference_applied_action_bound_count"])
        for row in evaluation_rows
    )
    robustness_action_count = sum(
        int(row["robustness_action_component_count"])
        for row in evaluation_rows
    )
    robustness_applied_count = sum(
        int(row["robustness_applied_action_bound_count"])
        for row in evaluation_rows
    )
    reference_summary = _validated_single_training_cohort_summary(
        report.get("acme_reference_summary"),
        field="acme_reference_summary",
        expected_kind="acme_aligned_preregistered_reference_v1",
        expected_selection="first_episode_per_eval_seed_prefix_v1",
        eval_seed_count=len(eval_seeds),
        episodes_per_eval_seed=reference_episodes_per_seed,
        expected_returns=reference_returns,
        expected_action_count=reference_action_count,
        expected_applied_count=reference_applied_count,
    )
    robustness_summary = _validated_single_training_cohort_summary(
        report.get("robustness_summary"),
        field="robustness_summary",
        expected_kind="dmc_nested_episode_robustness_v1",
        expected_selection="all_episodes_per_eval_seed_v1",
        eval_seed_count=len(eval_seeds),
        episodes_per_eval_seed=episodes_per_seed,
        expected_returns=episode_returns,
        expected_action_count=robustness_action_count,
        expected_applied_count=robustness_applied_count,
    )
    aggregate_mean = _finite_float(
        report.get("return_mean_across_eval_seed_means"),
        field="return_mean_across_eval_seed_means",
    )
    recomputed_mean = float(np.mean(eval_means))
    if not math.isclose(
        aggregate_mean, recomputed_mean, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError("Training-seed aggregate mean does not match eval rows")
    pooled_mean = _finite_float(
        report.get("pooled_episode_return_mean"),
        field="pooled_episode_return_mean",
    )
    if not math.isclose(
        pooled_mean,
        float(np.mean(episode_returns)),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError("Pooled episode mean does not match episode rows")
    return {
        "report": report,
        "task": task,
        "actor_type": actor_type,
        "training_seed": training_seed,
        "authorization": authorization,
        "checkpoint_lineage": checkpoint_lineage,
        "resolved_execution_spec_json": resolved_execution_spec_json,
        "actor_config": parsed_actor_config,
        "actor_config_json": normalized_actor_config_json,
        "protocol": protocol,
        "protocol_json": protocol_json,
        "runtime_protocol": runtime_protocol,
        "runtime_protocol_json": runtime_protocol_json,
        "eval_seeds": eval_seeds,
        "episodes_per_seed": episodes_per_seed,
        "reference_episodes_per_seed": reference_episodes_per_seed,
        "eval_means": eval_means,
        "episode_returns": episode_returns,
        "reference_returns": reference_returns,
        "reference_summary": reference_summary,
        "robustness_summary": robustness_summary,
        "reference_action_count": reference_action_count,
        "reference_applied_count": reference_applied_count,
        "robustness_action_count": robustness_action_count,
        "robustness_applied_count": robustness_applied_count,
        "reference_mean": float(np.mean(reference_returns)),
        "robustness_mean": float(np.mean(episode_returns)),
        "aggregate_mean": aggregate_mean,
    }


def aggregate_training_seed_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    expected_training_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Combine one or three single-policy reports without pseudo-replication."""

    if isinstance(reports, (str, bytes, Path)) or not isinstance(
        reports, Sequence
    ):
        raise TypeError("reports must be a sequence of aggregate mappings")
    if len(reports) not in (1, 3):
        raise ValueError("DMC aggregation requires 1 or 3 independent training seeds")
    validated = [_validated_training_seed_report(report) for report in reports]
    reference = validated[0]
    for current in validated[1:]:
        if current["task"] != reference["task"]:
            raise ValueError("Training-seed reports have different tasks")
        if current["actor_type"] != reference["actor_type"]:
            raise ValueError("Training-seed reports have different actor types")
        if current["actor_config_json"] != reference["actor_config_json"]:
            raise ValueError("Training-seed reports have different actor_config values")
        if current["protocol_json"] != reference["protocol_json"]:
            raise ValueError("Training-seed reports have different protocols")
        if current["runtime_protocol_json"] != reference["runtime_protocol_json"]:
            raise ValueError("Training-seed reports have different runtime protocols")
        if current["eval_seeds"] != reference["eval_seeds"] or current[
            "episodes_per_seed"
        ] != reference["episodes_per_seed"] or current[
            "reference_episodes_per_seed"
        ] != reference["reference_episodes_per_seed"]:
            raise ValueError("Training-seed reports use different eval seed plans")
        if current["checkpoint_lineage"] != reference["checkpoint_lineage"]:
            raise ValueError(
                "Training-seed reports have different checkpoint execution lineage"
            )

    actual_seeds = [item["training_seed"] for item in validated]
    if len(set(actual_seeds)) != len(actual_seeds):
        raise ValueError("Training seeds must be unique independent runs")
    if expected_training_seeds is not None:
        expected = _integer_sequence(
            list(expected_training_seeds), field="expected_training_seeds"
        )
        if len(expected) != len(validated) or set(expected) != set(actual_seeds):
            raise ValueError(
                "Training seeds do not match the task config/profile seed set"
            )
        order = {seed: index for index, seed in enumerate(expected)}
        validated.sort(key=lambda item: order[item["training_seed"]])
    else:
        validated.sort(key=lambda item: item["training_seed"])

    # Legacy aggregate_mean is the robustness mean over all episodes.  The
    # preregistered Acme-aligned comparison uses only the first episode of each
    # eval seed and therefore has its own training-seed inferential axis.
    training_means = [item["aggregate_mean"] for item in validated]
    training_stats = _training_seed_stats(training_means)
    reference_training_means = [item["reference_mean"] for item in validated]
    reference_training_stats = _training_seed_stats(reference_training_means)
    robustness_training_means = [item["robustness_mean"] for item in validated]
    robustness_training_stats = _training_seed_stats(robustness_training_means)
    authorizations = [item["authorization"] for item in validated]
    authorization_errors = [
        f"training_seed={item['training_seed']}: {error}"
        for item in validated
        for error in item["authorization"]["authorization_errors"]
    ]

    def common_authorization_value(field: str) -> Any:
        values = [authorization[field] for authorization in authorizations]
        if all(value == values[0] for value in values[1:]):
            return values[0]
        authorization_errors.append(
            f"training-seed reports have different {field} values"
        )
        return None

    authorization_kind = common_authorization_value("authorization_kind")
    config_fingerprint = common_authorization_value("config_fingerprint")
    approval_profile = common_authorization_value("approval_profile")
    approval_file_sha256 = common_authorization_value("approval_file_sha256")
    preflight_report_sha256 = common_authorization_value(
        "preflight_report_sha256"
    )
    authorization_verified = bool(
        all(
            authorization["authorization_verified"]
            and authorization["training_approved"] is True
            for authorization in authorizations
        )
        and not authorization_errors
    )
    all_eval_means = [
        value for item in validated for value in item["eval_means"]
    ]
    all_episode_returns = [
        value for item in validated for value in item["episode_returns"]
    ]
    all_reference_returns = [
        value for item in validated for value in item["reference_returns"]
    ]
    reference_action_count = sum(
        item["reference_action_count"] for item in validated
    )
    reference_applied_count = sum(
        item["reference_applied_count"] for item in validated
    )
    robustness_action_count = sum(
        item["robustness_action_count"] for item in validated
    )
    robustness_applied_count = sum(
        item["robustness_applied_count"] for item in validated
    )

    per_training_eval_stats = []
    for item in validated:
        values = np.asarray(item["eval_means"], dtype=np.float64)
        per_training_eval_stats.append(
            {
                "training_seed": item["training_seed"],
                "count": len(values),
                "return_mean": float(np.mean(values)),
                "return_sample_std": float(np.std(values, ddof=1)),
            }
        )
    per_eval_seed_stats = []
    for index, eval_seed in enumerate(reference["eval_seeds"]):
        values = np.asarray(
            [item["eval_means"][index] for item in validated], dtype=np.float64
        )
        per_eval_seed_stats.append(
            {
                "eval_seed": eval_seed,
                "training_seed_count": len(values),
                "return_mean_across_training_seeds": float(np.mean(values)),
                "return_sample_std_across_training_seeds": (
                    None if len(values) == 1 else float(np.std(values, ddof=1))
                ),
            }
        )
    per_training_episode_stats = []
    for item in validated:
        values = np.asarray(item["episode_returns"], dtype=np.float64)
        per_training_episode_stats.append(
            {
                "training_seed": item["training_seed"],
                "count": len(values),
                "return_mean": float(np.mean(values)),
                "return_population_std": float(np.std(values, ddof=0)),
            }
        )

    eval_array = np.asarray(all_eval_means, dtype=np.float64)
    episode_array = np.asarray(all_episode_returns, dtype=np.float64)
    reference_episode_array = np.asarray(
        all_reference_returns, dtype=np.float64
    )
    result: dict[str, Any] = {
        "schema_version": TRAINING_SEED_AGGREGATE_SCHEMA_VERSION,
        "kind": "dmc_training_seed_aggregate",
        "task": reference["task"],
        "actor_type": reference["actor_type"],
        "actor_config": reference["actor_config"],
        "protocol": reference["protocol"],
        "runtime_protocol": reference["runtime_protocol"],
        "training_seeds": [item["training_seed"] for item in validated],
        "training_seed_count": len(validated),
        "authorization_kind": authorization_kind,
        "training_approved": (
            True
            if all(
                authorization["training_approved"] is True
                for authorization in authorizations
            )
            else False
        ),
        "config_fingerprint": config_fingerprint,
        "approval_profile": approval_profile,
        "approval_file_sha256": approval_file_sha256,
        "preflight_report_sha256": preflight_report_sha256,
        "train_seed_indices": {
            str(item["training_seed"]): item["authorization"]["train_seed_index"]
            for item in validated
        },
        "authorization_verified": authorization_verified,
        "authorization_errors": authorization_errors,
        "resolved_execution_spec": reference["checkpoint_lineage"][
            "resolved_execution_spec"
        ],
        "evaluation_seeds": reference["checkpoint_lineage"][
            "evaluation_seeds"
        ],
        "evaluation_episodes_per_seed": reference["checkpoint_lineage"][
            "evaluation_episodes_per_seed"
        ],
        "evaluation_reference_episodes_per_seed": reference[
            "checkpoint_lineage"
        ]["evaluation_reference_episodes_per_seed"],
        "diagnostic_every_steps": reference["checkpoint_lineage"][
            "diagnostic_every_steps"
        ],
        "koopman_sha256": reference["checkpoint_lineage"]["koopman_sha256"],
        "koopman_lineage": reference["checkpoint_lineage"]["koopman_lineage"],
        "koopman_dataset_sha256": reference["checkpoint_lineage"][
            "koopman_dataset_sha256"
        ],
        "koopman_config_fingerprint": reference["checkpoint_lineage"][
            "koopman_config_fingerprint"
        ],
        "value_expansion": reference["checkpoint_lineage"][
            "value_expansion"
        ],
        "eval_seeds": reference["eval_seeds"],
        "eval_seed_count_per_training_seed": len(reference["eval_seeds"]),
        "episodes_per_eval_seed": reference["episodes_per_seed"],
        "total_evaluation_episodes": len(episode_array),
        "inference_axis": "training_seed",
        "axis_semantics": {
            "training_seed": (
                "independent policy-training replicate; the only inferential unit"
            ),
            "eval_seed": (
                "environment-seed mean nested within a training seed; descriptive"
            ),
            "episode": (
                "episode nested within eval seed and training seed; descriptive"
            ),
        },
        "training_seed_statistics": training_stats,
        "eval_seed_statistics": {
            "unit": "evaluation-seed mean nested within training seed",
            "count_per_training_seed": len(reference["eval_seeds"]),
            "total_nested_cells": len(eval_array),
            "return_mean_across_all_nested_cells": float(np.mean(eval_array)),
            "return_sample_std_across_all_nested_cells": float(
                np.std(eval_array, ddof=1)
            ),
            "descriptive_only": True,
            "return_standard_error": None,
            "return_95ci": None,
            "per_training_seed": per_training_eval_stats,
            "per_eval_seed": per_eval_seed_stats,
        },
        "episode_statistics": {
            "unit": "episode nested within eval seed and training seed",
            "total_nested_episodes": len(episode_array),
            "return_mean": float(np.mean(episode_array)),
            "return_population_std": float(np.std(episode_array, ddof=0)),
            "descriptive_only": True,
            "return_standard_error": None,
            "return_95ci": None,
            "per_training_seed": per_training_episode_stats,
        },
        "acme_reference_summary": {
            "kind": "acme_aligned_preregistered_reference_v1",
            "episode_selection": "first_episode_per_eval_seed_prefix_v1",
            "inference_axis": "training_seed",
            "eval_seed_count_per_training_seed": len(reference["eval_seeds"]),
            "episodes_per_eval_seed": reference[
                "reference_episodes_per_seed"
            ],
            "total_nested_episodes": len(reference_episode_array),
            "return_mean_across_all_nested_episodes": float(
                np.mean(reference_episode_array)
            ),
            "return_population_std_across_all_nested_episodes": float(
                np.std(reference_episode_array, ddof=0)
            ),
            "action_component_count": reference_action_count,
            "applied_action_bound_count": reference_applied_count,
            "applied_action_bound_fraction": float(
                reference_applied_count / reference_action_count
            ),
            "training_seed_statistics": reference_training_stats,
            "return_mean_across_training_seed_means": (
                reference_training_stats["return_mean"]
            ),
            "return_std_across_training_seed_means": (
                reference_training_stats["return_sample_std"]
            ),
            "return_standard_error_across_training_seed_means": (
                reference_training_stats["return_standard_error"]
            ),
            "return_student_t_95ci_across_training_seed_means": (
                reference_training_stats["return_student_t_95ci"]
            ),
            "per_training_seed": [
                {
                    "training_seed": item["training_seed"],
                    "episode_count": len(item["reference_returns"]),
                    "return_mean": item["reference_mean"],
                    "return_population_std": float(
                        np.std(item["reference_returns"], ddof=0)
                    ),
                    "action_component_count": item["reference_action_count"],
                    "applied_action_bound_count": item[
                        "reference_applied_count"
                    ],
                    "applied_action_bound_fraction": float(
                        item["reference_applied_count"]
                        / item["reference_action_count"]
                    ),
                }
                for item in validated
            ],
        },
        "robustness_summary": {
            "kind": "dmc_nested_episode_robustness_v1",
            "episode_selection": "all_episodes_per_eval_seed_v1",
            "inference_axis": "training_seed",
            "eval_seed_count_per_training_seed": len(reference["eval_seeds"]),
            "episodes_per_eval_seed": reference["episodes_per_seed"],
            "total_nested_episodes": len(episode_array),
            "return_mean_across_all_nested_episodes": float(
                np.mean(episode_array)
            ),
            "return_population_std_across_all_nested_episodes": float(
                np.std(episode_array, ddof=0)
            ),
            "action_component_count": robustness_action_count,
            "applied_action_bound_count": robustness_applied_count,
            "applied_action_bound_fraction": float(
                robustness_applied_count / robustness_action_count
            ),
            "training_seed_statistics": robustness_training_stats,
            "return_mean_across_training_seed_means": (
                robustness_training_stats["return_mean"]
            ),
            "return_std_across_training_seed_means": (
                robustness_training_stats["return_sample_std"]
            ),
            "return_standard_error_across_training_seed_means": (
                robustness_training_stats["return_standard_error"]
            ),
            "return_student_t_95ci_across_training_seed_means": (
                robustness_training_stats["return_student_t_95ci"]
            ),
            "per_training_seed": [
                {
                    "training_seed": item["training_seed"],
                    "episode_count": len(item["episode_returns"]),
                    "return_mean": item["robustness_mean"],
                    "return_population_std": float(
                        np.std(item["episode_returns"], ddof=0)
                    ),
                    "action_component_count": item["robustness_action_count"],
                    "applied_action_bound_count": item[
                        "robustness_applied_count"
                    ],
                    "applied_action_bound_fraction": float(
                        item["robustness_applied_count"]
                        / item["robustness_action_count"]
                    ),
                }
                for item in validated
            ],
        },
        "return_mean_across_training_seed_means": training_stats["return_mean"],
        "return_std_across_training_seed_means": training_stats[
            "return_sample_std"
        ],
        "return_standard_error_across_training_seed_means": training_stats[
            "return_standard_error"
        ],
        "return_student_t_95ci_across_training_seed_means": training_stats[
            "return_student_t_95ci"
        ],
        "per_training_seed": [item["report"] for item in validated],
    }
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is not allowed")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _read_aggregate_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read aggregate report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Aggregate report {path} must contain a JSON object")
    return payload


def _validate_formal_authorization(
    authorization: Mapping[str, Any],
    *,
    config: ExperimentConfig,
    profile: str,
    training_seed: int,
    expected_seed_indices: Mapping[int, int],
    label: str,
) -> None:
    """Fail closed on the approval identity before any environment is opened."""

    if authorization.get("authorization_verified") is not True:
        raise PermissionError(
            f"{label} has unverified training authorization: "
            f"{authorization.get('authorization_errors')}"
        )
    if authorization.get("training_approved") is not True:
        raise PermissionError(f"{label} requires training_approved=true")
    if authorization.get("config_fingerprint") != config.fingerprint:
        raise ValueError(f"{label} config_fingerprint does not match the config")
    if authorization.get("approval_profile") != profile:
        raise ValueError(f"{label} approval_profile does not match profile")
    if training_seed not in expected_seed_indices:
        raise ValueError(f"{label} training_seed is outside the config/profile")
    if authorization.get("train_seed_index") != expected_seed_indices[training_seed]:
        raise ValueError(
            f"{label} train_seed_index does not match the config seed ordering"
        )


def _validate_formal_execution_lineage(
    lineage: Mapping[str, Any],
    *,
    config: ExperimentConfig,
    profile: str,
    actor_type: str,
    authorization: Mapping[str, Any],
    eval_seeds: Sequence[int],
    episodes_per_seed: int,
    label: str,
) -> None:
    """Bind a formal artifact to its checkpoint-saved execution plan."""

    saved_spec, saved_spec_json = _canonical_mapping(
        lineage.get("resolved_execution_spec"),
        field=f"{label}.resolved_execution_spec",
    )
    expected_spec = resolve_execution_spec(config, profile)
    _, expected_spec_json = _canonical_mapping(
        expected_spec, field="expected resolved_execution_spec"
    )
    if saved_spec_json != expected_spec_json:
        raise ValueError(
            f"{label} resolved_execution_spec does not match config/profile"
        )
    # Keep the local binding explicit even though these values are also present
    # inside resolved_execution_spec.  PPO checkpoints deliberately save both.
    saved_seeds = _integer_sequence(
        lineage.get("evaluation_seeds"),
        field=f"{label}.evaluation_seeds",
    )
    if saved_seeds != list(eval_seeds):
        raise ValueError(f"{label} evaluation_seeds do not match the config")
    saved_episodes = _integer(
        lineage.get("evaluation_episodes_per_seed"),
        field=f"{label}.evaluation_episodes_per_seed",
        minimum=1,
    )
    if saved_episodes != episodes_per_seed:
        raise ValueError(
            f"{label} evaluation_episodes_per_seed does not match the config"
        )
    if saved_spec.get("evaluation_seeds") != saved_seeds or saved_spec.get(
        "evaluation"
    ) != config.raw["evaluation"]:
        raise ValueError(
            f"{label} direct evaluation plan disagrees with resolved_execution_spec"
        )

    koopman_fields = (
        "koopman_sha256",
        "koopman_lineage",
        "koopman_dataset_sha256",
        "koopman_config_fingerprint",
    )
    if actor_type == "PPO":
        unexpected = {
            field: lineage.get(field)
            for field in koopman_fields
            if lineage.get(field) is not None
        }
        if unexpected:
            raise ValueError(
                f"{label} PPO checkpoint unexpectedly declares Koopman lineage: "
                f"{unexpected}"
            )
        return

    koopman_sha256 = lineage.get("koopman_sha256")
    if (
        not isinstance(koopman_sha256, str)
        or len(koopman_sha256) != 64
        or any(character not in "0123456789abcdef" for character in koopman_sha256)
    ):
        raise ValueError(f"{label} koopman_sha256 must be 64 lowercase hex")
    koopman_lineage, _ = _canonical_mapping(
        lineage.get("koopman_lineage"), field=f"{label}.koopman_lineage"
    )
    if set(koopman_lineage) != KOOPMAN_LINEAGE_FIELDS:
        raise ValueError(f"{label} koopman_lineage has incomplete fields")
    dataset_sha256 = koopman_lineage.get("dataset_sha256")
    if (
        not isinstance(dataset_sha256, str)
        or len(dataset_sha256) != 64
        or any(character not in "0123456789abcdef" for character in dataset_sha256)
    ):
        raise ValueError(f"{label} Koopman dataset_sha256 is invalid")
    model_config_fingerprint = koopman_lineage.get("config_fingerprint")
    if (
        not isinstance(model_config_fingerprint, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", model_config_fingerprint) is None
    ):
        raise ValueError(f"{label} Koopman config_fingerprint is invalid")
    if koopman_lineage.get("approval_profile") not in {
        "development",
        "benchmark",
    }:
        raise ValueError(f"{label} Koopman approval_profile is invalid")
    for field in ("approval_file_sha256", "preflight_report_sha256"):
        value = koopman_lineage.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} Koopman {field} is invalid")
    if lineage.get("koopman_dataset_sha256") != dataset_sha256:
        raise ValueError(f"{label} koopman_dataset_sha256 disagrees with lineage")
    if lineage.get("koopman_config_fingerprint") != model_config_fingerprint:
        raise ValueError(
            f"{label} koopman_config_fingerprint disagrees with model lineage"
        )


def _checkpoint_source_identity(
    metadata: ActorCheckpointMetadata,
) -> dict[str, Any]:
    actor_config, actor_config_json = _canonical_mapping(
        metadata.actor_config.to_dict(), field="checkpoint actor_config"
    )
    protocol, protocol_json = _canonical_mapping(
        metadata.protocol, field="checkpoint protocol"
    )
    if metadata.payload.get("environment_protocol_json") != protocol_json:
        raise ValueError(
            "Formal checkpoint environment_protocol_json does not match protocol"
        )
    return {
        "task": metadata.task,
        "actor_type": metadata.actor_type,
        "training_seed": metadata.training_seed,
        "authorization": metadata.authorization,
        "actor_config": actor_config,
        "actor_config_json": actor_config_json,
        "protocol": protocol,
        "protocol_json": protocol_json,
        "checkpoint_lineage": {
            "resolved_execution_spec": metadata.payload.get(
                "resolved_execution_spec"
            ),
            "evaluation_seeds": metadata.payload.get("evaluation_seeds"),
            "evaluation_episodes_per_seed": metadata.payload.get(
                "evaluation_episodes_per_seed"
            ),
            "evaluation_reference_episodes_per_seed": metadata.payload.get(
                "evaluation_reference_episodes_per_seed"
            ),
            "diagnostic_every_steps": metadata.payload.get(
                "diagnostic_every_steps"
            ),
            "koopman_sha256": metadata.payload.get("koopman_sha256"),
            "koopman_lineage": metadata.payload.get("koopman_lineage"),
            "koopman_dataset_sha256": metadata.payload.get(
                "koopman_dataset_sha256"
            ),
            "koopman_config_fingerprint": metadata.payload.get(
                "koopman_config_fingerprint"
            ),
            "value_expansion": metadata.payload.get("value_expansion"),
        },
    }


def aggregate_training_seeds(
    sources: Sequence[str | Path | Mapping[str, Any]],
    *,
    profile: str | None = None,
    config_path: str | Path | None = None,
    device_name: str = "auto",
    koopman_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read/evaluate existing artifacts and aggregate independent training seeds."""

    if isinstance(sources, (str, bytes, Path)) or not isinstance(sources, Sequence):
        raise TypeError("sources must be a sequence")
    count = len(sources)
    if count not in (1, 3):
        raise ValueError("DMC profiles require 1 or 3 training-seed inputs")
    resolved_profile = profile or ("development" if count == 1 else "benchmark")
    if resolved_profile not in PROFILE_NAMES:
        raise ValueError(f"profile must be one of {PROFILE_NAMES}")
    expected_count = 1 if resolved_profile == "development" else 3
    if count != expected_count:
        raise ValueError(
            f"Profile {resolved_profile!r} requires {expected_count} inputs"
        )

    prepared: list[
        tuple[str, Any, str, ActorCheckpointMetadata | None]
    ] = []
    task_hints: list[str] = []
    for source in sources:
        if isinstance(source, Mapping):
            report = dict(source)
            prepared.append(("aggregate", report, "in-memory report", None))
            if isinstance(report.get("task"), str):
                task_hints.append(report["task"])
            continue
        path = Path(source)
        if path.suffix.lower() == ".json":
            report = _read_aggregate_json(path)
            prepared.append(("aggregate", report, str(path.resolve()), None))
            if isinstance(report.get("task"), str):
                task_hints.append(report["task"])
        else:
            metadata = load_actor_checkpoint(path, map_location="cpu")
            prepared.append(
                ("checkpoint", path, str(path.resolve()), metadata)
            )
            task_hints.append(metadata.task)
    if not task_hints or len(set(task_hints)) != 1:
        raise ValueError("All sources must declare one consistent DMC task")

    config = load_experiment_config(
        config_path or default_config_path(task_hints[0])
    )
    if config.task != task_hints[0]:
        raise ValueError("Experiment config task does not match input artifacts")
    expected_seeds = [
        int(value)
        for value in config.raw["seeds"]["train"][:expected_count]
    ]
    eval_seeds = [int(value) for value in config.raw["seeds"]["evaluation"]]
    episodes = int(config.raw["evaluation"]["episodes_per_seed"])
    checkpoint_name = f"{config.raw['evaluation']['checkpoint']}.pt"
    checkpoint_count = sum(kind == "checkpoint" for kind, _, _, _ in prepared)
    if koopman_path is not None and checkpoint_count != 1:
        raise ValueError("koopman_path override requires exactly one checkpoint input")

    expected_seed_indices = {
        seed: index for index, seed in enumerate(expected_seeds)
    }
    source_identities: list[dict[str, Any]] = []
    for kind, value, label, metadata in prepared:
        selected_checkpoint = (
            Path(value)
            if kind == "checkpoint"
            else Path(str(value.get("actor_checkpoint", "")))
        )
        if selected_checkpoint.name != checkpoint_name:
            raise ValueError(
                f"{label} does not use the approval-bound evaluation "
                f"checkpoint {checkpoint_name!r}"
            )
        if kind == "checkpoint":
            assert metadata is not None
            identity = _checkpoint_source_identity(metadata)
        else:
            identity = _validated_training_seed_report(value)
        if identity["task"] != config.task:
            raise ValueError(f"{label} task does not match the experiment config")
        _validate_formal_authorization(
            identity["authorization"],
            config=config,
            profile=resolved_profile,
            training_seed=identity["training_seed"],
            expected_seed_indices=expected_seed_indices,
            label=label,
        )
        _validate_formal_execution_lineage(
            identity["checkpoint_lineage"],
            config=config,
            profile=resolved_profile,
            actor_type=identity["actor_type"],
            authorization=identity["authorization"],
            eval_seeds=eval_seeds,
            episodes_per_seed=episodes,
            label=label,
        )
        if identity["actor_config"] != config.actor_config.to_dict():
            raise ValueError(f"{label} actor_config does not match the config")
        source_identities.append(identity)

    actual_source_seeds = [
        identity["training_seed"] for identity in source_identities
    ]
    if len(set(actual_source_seeds)) != len(actual_source_seeds):
        raise ValueError("Training seeds must be unique independent runs")
    if set(actual_source_seeds) != set(expected_seeds):
        raise ValueError("Training seeds do not match the config/profile seed set")
    reference_identity = source_identities[0]
    for identity in source_identities[1:]:
        if identity["actor_type"] != reference_identity["actor_type"]:
            raise ValueError("Input artifacts have different actor types")
        if identity["actor_config_json"] != reference_identity["actor_config_json"]:
            raise ValueError("Input artifacts have different actor_config values")
        if identity["protocol_json"] != reference_identity["protocol_json"]:
            raise ValueError("Input artifacts have different environment protocols")
        for field in (
            "authorization_kind",
            "config_fingerprint",
            "approval_profile",
            "approval_file_sha256",
            "preflight_report_sha256",
        ):
            if identity["authorization"][field] != reference_identity[
                "authorization"
            ][field]:
                raise ValueError(
                    f"Input artifacts have different {field} values"
                )

    reports: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    for (kind, value, label, _metadata), identity in zip(
        prepared, source_identities, strict=True
    ):
        if kind == "checkpoint":
            report = aggregate_evaluations(
                value,
                koopman_path=koopman_path,
                device_name=device_name,
            )
        else:
            report = value
        reports.append(report)
        source_metadata.append(
            {
                "kind": kind,
                "source": label,
                "training_seed": identity["training_seed"],
                "train_seed_index": identity["authorization"][
                    "train_seed_index"
                ],
            }
        )
    source_metadata.sort(key=lambda item: item["train_seed_index"])

    result = aggregate_training_seed_reports(
        reports, expected_training_seeds=expected_seeds
    )
    if result["eval_seeds"] != eval_seeds or (
        result["episodes_per_eval_seed"] != episodes
    ):
        raise ValueError(
            "Evaluation seed plan does not match the task experiment config"
        )
    expected_actor_config = config.actor_config.to_dict()
    if result["actor_config"] != expected_actor_config:
        raise ValueError("actor_config does not match the task experiment config")
    if result["training_approved"] is not True:
        raise PermissionError("Formal aggregate requires training_approved=true")
    if result["config_fingerprint"] != config.fingerprint:
        raise ValueError(
            "Actor checkpoint config_fingerprint does not match the full config"
        )
    if result["approval_profile"] != resolved_profile:
        raise ValueError("Actor checkpoint approval_profile does not match profile")
    if not isinstance(result["approval_file_sha256"], str) or not isinstance(
        result["preflight_report_sha256"], str
    ):
        raise ValueError("Approval and preflight identities must be shared SHA-256s")
    expected_indices = {
        str(seed): index for index, seed in enumerate(expected_seeds)
    }
    if result["train_seed_indices"] != expected_indices:
        raise ValueError(
            "train_seed_index does not match the config training-seed mapping"
        )
    if result["authorization_verified"] is not True:
        raise PermissionError(
            "Formal aggregate authorization identity is unverified: "
            f"{result['authorization_errors']}"
        )
    result.update(
        {
            "profile": resolved_profile,
            "config_path": str(config.path.resolve()),
            "config_fingerprint": config.fingerprint,
            "evaluation_checkpoint": config.raw["evaluation"]["checkpoint"],
            "sources": source_metadata,
        }
    )
    return result


def _parse_eval_seeds(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("eval seeds must be comma-separated integers") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="existing checkpoint; repeat once (development) or three times (benchmark)",
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        action="append",
        default=[],
        help="existing single-training-seed aggregate JSON; may be mixed with checkpoints",
    )
    parser.add_argument("--profile", choices=PROFILE_NAMES, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--koopman", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sources = [*args.actor_checkpoint, *args.aggregate_report]
    if not sources:
        raise SystemExit(
            "Provide --actor-checkpoint or --aggregate-report; this command "
            "never launches training."
        )
    report = aggregate_training_seeds(
        sources,
        profile=args.profile,
        config_path=args.config,
        device_name=args.device,
        koopman_path=args.koopman,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        _write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
