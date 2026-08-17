"""Strict, read-only comparison of the five pre-registered DMC methods.

This command consumes evaluation aggregate JSON files only.  It never opens a
DMC environment, restores a policy, or launches training.  The comparison is
fail-closed: all five methods must be present exactly once and every report
must be bound to the selected config, profile, approval, preflight, protocol,
evaluation plan, and training-seed set.

Example::

    python -m experiments.dmc.eval.compare_dmc \
        --config experiments/dmc/configs/cartpole_swingup.yaml \
        --profile development \
        --aggregate-report runs/dmc/eval/cartpole_swingup/development/PPO.json \
        --aggregate-report runs/dmc/eval/cartpole_swingup/development/KLQR.json \
        --aggregate-report runs/dmc/eval/cartpole_swingup/development/AB-PQ.json \
        --aggregate-report runs/dmc/eval/cartpole_swingup/development/KMPC.json \
        --aggregate-report runs/dmc/eval/cartpole_swingup/development/AC-MPC-MPVE.json \
        --output runs/dmc/eval/cartpole_swingup/development/comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.dmc.actors import ACTOR_TYPES
from experiments.dmc.config import (
    ExperimentConfig,
    PROFILE_NAMES,
    load_experiment_config,
    resolve_execution_spec,
)
from experiments.dmc.reward_model import transition_reward_input_contract
from experiments.dmc.reward_oracle import (
    LEARNED_TRANSITION_REWARD,
    OFFICIAL_OBSERVATION_ORACLE,
    exact_reward_oracle_metadata,
)
from experiments.dmc.eval.aggregate_dmc import (
    AGGREGATE_SCHEMA_VERSION,
    TRAINING_SEED_AGGREGATE_SCHEMA_VERSION,
    _read_aggregate_json,
    _validate_formal_authorization,
    _validate_formal_execution_lineage,
    _validated_training_seed_report,
    aggregate_training_seed_reports,
)
from experiments.dmc.eval.evaluate_dmc import _write_json_atomic


COMPARISON_SCHEMA_VERSION = "dmc_five_method_comparison_v1"
COMPARISON_KIND = "dmc_five_method_comparison"
EXPECTED_ACTOR_TYPES = ("PPO", "KLQR", "AB-PQ", "KMPC", "AC-MPC-MPVE")
REFERENCE_VIEW = "reference10"
ROBUSTNESS_VIEW = "robustness100"
FORMAL_AGGREGATE_EXTRA_FIELDS = frozenset(
    {"profile", "config_path", "evaluation_checkpoint", "sources"}
)
SHARED_AUTHORIZATION_FIELDS = (
    "authorization_kind",
    "config_fingerprint",
    "approval_profile",
    "approval_file_sha256",
    "preflight_report_sha256",
)


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


def _canonical_json(value: Any, *, field: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain finite JSON data") from exc


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    mapping = dict(value)
    _canonical_json(mapping, field=field)
    return mapping


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str] | frozenset[str], *, field: str
) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{field} fields differ from the declared schema; "
            f"missing={missing}, extra={extra}"
        )


def _float_sequence(
    value: Any, *, field: str, expected_count: int
) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence")
    values = [
        _finite_float(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(values) != expected_count:
        raise ValueError(
            f"{field} contains {len(values)} values, expected {expected_count}"
        )
    return values


def _positive_integer_sequence(
    value: Any, *, field: str, expected_count: int
) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence")
    values = [
        _integer(item, field=f"{field}[{index}]", minimum=1)
        for index, item in enumerate(value)
    ]
    if len(values) != expected_count:
        raise ValueError(
            f"{field} contains {len(values)} values, expected {expected_count}"
        )
    return values


def _same_number(actual: float, expected: float, *, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{field} is inconsistent with its underlying rows")


def _fraction(value: Any, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _expected_value_expansion(
    actor_type: str, config: ExperimentConfig
) -> dict[str, Any]:
    enabled = actor_type == "AC-MPC-MPVE"
    reward_source = str(config.raw["ppo"]["mpve_reward_source"])
    reward_metadata = (
        exact_reward_oracle_metadata(config.task)
        if enabled and reward_source == OFFICIAL_OBSERVATION_ORACLE
        else {
            "source": LEARNED_TRANSITION_REWARD,
            "model_input_contract": transition_reward_input_contract(),
            "checkpoint_field": "reward_model_state",
        }
        if enabled
        else None
    )
    return {
        "enabled": enabled,
        "kind": "mpve_td_k_tro25_eq8_eq9_v1" if enabled else None,
        "actor_shared_with": "KMPC" if enabled else None,
        "horizon": int(config.raw["ppo"]["mpve_horizon"]) if enabled else None,
        "value_loss_coefficient": (
            float(config.raw["ppo"]["mpve_value_loss_coefficient"])
            if enabled
            else None
        ),
        "prediction_gradient": "detached",
        "terminal_target_gradient": "detached",
        "standard_gae_value_loss_retained": True,
        "reward": reward_metadata,
    }


def _validate_formal_aggregate_wrapper(
    payload: Mapping[str, Any],
    *,
    recomputed: Mapping[str, Any],
    config: ExperimentConfig,
    profile: str,
    expected_training_seeds: Sequence[int],
    label: str,
) -> None:
    expected_keys = set(recomputed) | set(FORMAL_AGGREGATE_EXTRA_FIELDS)
    _require_exact_keys(payload, expected_keys, field=label)
    for key, expected_value in recomputed.items():
        if _canonical_json(payload[key], field=f"{label}.{key}") != _canonical_json(
            expected_value, field=f"recomputed.{key}"
        ):
            raise ValueError(f"{label}.{key} does not match the nested reports")
    if payload.get("profile") != profile:
        raise ValueError(f"{label}.profile does not match the selected profile")
    if payload.get("config_path") != str(config.path.resolve()):
        raise ValueError(f"{label}.config_path does not match the selected config")
    if payload.get("evaluation_checkpoint") != config.raw["evaluation"]["checkpoint"]:
        raise ValueError(f"{label}.evaluation_checkpoint does not match the config")

    sources = payload.get("sources")
    if not isinstance(sources, (list, tuple)) or len(sources) != len(
        expected_training_seeds
    ):
        raise ValueError(f"{label}.sources must contain one row per training seed")
    expected_index = {
        seed: index for index, seed in enumerate(expected_training_seeds)
    }
    seen: set[int] = set()
    for index, source_value in enumerate(sources):
        source = _require_mapping(source_value, field=f"{label}.sources[{index}]")
        _require_exact_keys(
            source,
            {"kind", "source", "training_seed", "train_seed_index"},
            field=f"{label}.sources[{index}]",
        )
        if source["kind"] not in {"aggregate", "checkpoint"}:
            raise ValueError(f"{label}.sources[{index}].kind is unsupported")
        if not isinstance(source["source"], str) or not source["source"]:
            raise ValueError(f"{label}.sources[{index}].source must be non-empty")
        seed = _integer(
            source["training_seed"],
            field=f"{label}.sources[{index}].training_seed",
        )
        if seed in seen or seed not in expected_index:
            raise ValueError(f"{label}.sources contains duplicate or unknown seeds")
        seen.add(seed)
        if seed != expected_training_seeds[index]:
            raise ValueError(
                f"{label}.sources does not follow the profile training-seed order"
            )
        if _integer(
            source["train_seed_index"],
            field=f"{label}.sources[{index}].train_seed_index",
        ) != expected_index[seed]:
            raise ValueError(f"{label}.sources has an invalid train_seed_index")


def _single_reports_from_payload(
    payload_value: Any,
    *,
    config: ExperimentConfig,
    profile: str,
    expected_training_seeds: Sequence[int],
    label: str,
) -> list[dict[str, Any]]:
    payload = _require_mapping(payload_value, field=label)
    schema = payload.get("schema_version")
    kind = payload.get("kind")
    if schema == AGGREGATE_SCHEMA_VERSION and kind == "dmc_ten_eval_seed_aggregate":
        if len(expected_training_seeds) != 1:
            raise ValueError(
                f"{label} is a single-training-seed report but profile {profile!r} "
                "requires three training seeds"
            )
        reports = [payload]
    elif (
        schema == TRAINING_SEED_AGGREGATE_SCHEMA_VERSION
        and kind == "dmc_training_seed_aggregate"
    ):
        nested = payload.get("per_training_seed")
        if not isinstance(nested, (list, tuple)):
            raise ValueError(f"{label}.per_training_seed must be a sequence")
        if len(nested) != len(expected_training_seeds):
            raise ValueError(
                f"{label} has {len(nested)} training seeds, expected "
                f"{len(expected_training_seeds)}"
            )
        reports = [
            _require_mapping(item, field=f"{label}.per_training_seed[{index}]")
            for index, item in enumerate(nested)
        ]
        recomputed = aggregate_training_seed_reports(
            reports, expected_training_seeds=expected_training_seeds
        )
        _validate_formal_aggregate_wrapper(
            payload,
            recomputed=recomputed,
            config=config,
            profile=profile,
            expected_training_seeds=expected_training_seeds,
            label=label,
        )
    else:
        raise ValueError(
            f"{label} uses a legacy or unsupported aggregate schema/kind"
        )
    return reports


def _validate_row_views(
    row: Mapping[str, Any],
    *,
    field: str,
    episodes_per_seed: int,
    reference_episodes_per_seed: int,
    action_dim: int,
) -> dict[str, Any]:
    if row.get("deterministic") is not True:
        raise ValueError(f"{field}.deterministic must be true")
    episode_returns = _float_sequence(
        row.get("episode_returns"),
        field=f"{field}.episode_returns",
        expected_count=episodes_per_seed,
    )
    episode_lengths = _positive_integer_sequence(
        row.get("episode_lengths"),
        field=f"{field}.episode_lengths",
        expected_count=episodes_per_seed,
    )
    episode_action_counts = _positive_integer_sequence(
        row.get("episode_action_component_counts"),
        field=f"{field}.episode_action_component_counts",
        expected_count=episodes_per_seed,
    )
    expected_action_counts = [length * action_dim for length in episode_lengths]
    if episode_action_counts != expected_action_counts:
        raise ValueError(
            f"{field}.episode_action_component_counts disagree with "
            "episode_lengths and protocol.action_dim"
        )

    reference_returns = _float_sequence(
        row.get("acme_reference_episode_returns"),
        field=f"{field}.acme_reference_episode_returns",
        expected_count=reference_episodes_per_seed,
    )
    if reference_returns != episode_returns[:reference_episodes_per_seed]:
        raise ValueError(
            f"{field}.acme_reference_episode_returns must be the pre-registered "
            "prefix of episode_returns"
        )
    if _integer(
        row.get("acme_reference_episode_count"),
        field=f"{field}.acme_reference_episode_count",
        minimum=1,
    ) != reference_episodes_per_seed:
        raise ValueError(f"{field}.acme_reference_episode_count is inconsistent")
    reference_mean = _finite_float(
        row.get("acme_reference_return_mean"),
        field=f"{field}.acme_reference_return_mean",
    )
    _same_number(
        reference_mean,
        float(np.mean(reference_returns)),
        field=f"{field}.acme_reference_return_mean",
    )

    robustness_returns = _float_sequence(
        row.get("robustness_episode_returns"),
        field=f"{field}.robustness_episode_returns",
        expected_count=episodes_per_seed,
    )
    if robustness_returns != episode_returns:
        raise ValueError(
            f"{field}.robustness_episode_returns must equal episode_returns"
        )
    if _integer(
        row.get("robustness_episode_count"),
        field=f"{field}.robustness_episode_count",
        minimum=1,
    ) != episodes_per_seed:
        raise ValueError(f"{field}.robustness_episode_count is inconsistent")
    robustness_mean = _finite_float(
        row.get("robustness_return_mean"),
        field=f"{field}.robustness_return_mean",
    )
    _same_number(
        robustness_mean,
        float(np.mean(robustness_returns)),
        field=f"{field}.robustness_return_mean",
    )
    robustness_std = _finite_float(
        row.get("robustness_return_population_std"),
        field=f"{field}.robustness_return_population_std",
    )
    _same_number(
        robustness_std,
        float(np.std(robustness_returns, ddof=0)),
        field=f"{field}.robustness_return_population_std",
    )

    reference_fraction = _fraction(
        row.get("acme_reference_applied_action_bound_fraction"),
        field=f"{field}.acme_reference_applied_action_bound_fraction",
    )
    robustness_fraction = _fraction(
        row.get("robustness_applied_action_bound_fraction"),
        field=f"{field}.robustness_applied_action_bound_fraction",
    )
    compatibility_fraction = _fraction(
        row.get("applied_action_bound_fraction"),
        field=f"{field}.applied_action_bound_fraction",
    )
    _same_number(
        robustness_fraction,
        compatibility_fraction,
        field=f"{field}.robustness_applied_action_bound_fraction",
    )

    reference_action_count = sum(
        episode_action_counts[:reference_episodes_per_seed]
    )
    robustness_action_count = sum(episode_action_counts)
    return {
        REFERENCE_VIEW: {
            "returns": reference_returns,
            "action_bound_fraction": reference_fraction,
            "action_count": reference_action_count,
        },
        ROBUSTNESS_VIEW: {
            "returns": robustness_returns,
            "action_bound_fraction": robustness_fraction,
            "action_count": robustness_action_count,
        },
    }


def _weighted_fraction(rows: Sequence[Mapping[str, Any]], *, view: str) -> float:
    denominator = sum(int(row[view]["action_count"]) for row in rows)
    if denominator <= 0:
        raise ValueError(f"{view} action count must be positive")
    numerator = sum(
        float(row[view]["action_bound_fraction"])
        * int(row[view]["action_count"])
        for row in rows
    )
    result = numerator / denominator
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{view} weighted action saturation is invalid")
    return float(result)


def _validate_declared_view_summary(
    report: Mapping[str, Any],
    *,
    key: str,
    field: str,
    expected_episode_count: int,
    expected_return_mean: float,
    expected_action_fraction: float,
) -> None:
    summary = _require_mapping(report.get(key), field=f"{field}.{key}")
    if _integer(
        summary.get("episode_count"),
        field=f"{field}.{key}.episode_count",
        minimum=1,
    ) != expected_episode_count:
        raise ValueError(f"{field}.{key}.episode_count is inconsistent")
    declared_mean = _finite_float(
        summary.get("return_mean"), field=f"{field}.{key}.return_mean"
    )
    _same_number(
        declared_mean, expected_return_mean, field=f"{field}.{key}.return_mean"
    )
    declared_fraction = _fraction(
        summary.get("applied_action_bound_fraction"),
        field=f"{field}.{key}.applied_action_bound_fraction",
    )
    _same_number(
        declared_fraction,
        expected_action_fraction,
        field=f"{field}.{key}.applied_action_bound_fraction",
    )


def _validate_method_payload(
    payload: Mapping[str, Any],
    *,
    config: ExperimentConfig,
    profile: str,
    expected_training_seeds: Sequence[int],
    label: str,
) -> dict[str, Any]:
    reports = _single_reports_from_payload(
        payload,
        config=config,
        profile=profile,
        expected_training_seeds=expected_training_seeds,
        label=label,
    )
    expected_indices = {
        seed: index for index, seed in enumerate(expected_training_seeds)
    }
    expected_eval_seeds = [int(seed) for seed in config.raw["seeds"]["evaluation"]]
    evaluation = config.raw["evaluation"]
    episodes_per_seed = int(evaluation["episodes_per_seed"])
    reference_episodes_per_seed = int(evaluation["reference_episodes_per_seed"])
    expected_checkpoint = f"{evaluation['checkpoint']}.pt"
    expected_execution_spec = resolve_execution_spec(config, profile)
    expected_execution_json = _canonical_json(
        expected_execution_spec, field="expected resolved_execution_spec"
    )

    validated_reports: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    actor_type: str | None = None
    for report_index, report in enumerate(reports):
        report_label = f"{label}.training_seed[{report_index}]"
        validated = _validated_training_seed_report(report)
        if validated["task"] != config.task:
            raise ValueError(f"{report_label}.task does not match the config")
        current_actor = validated["actor_type"]
        if actor_type is None:
            actor_type = current_actor
        elif current_actor != actor_type:
            raise ValueError(f"{label} contains multiple actor types")
        if validated["actor_config"] != config.actor_config.to_dict():
            raise ValueError(f"{report_label}.actor_config does not match the config")
        if Path(str(report.get("actor_checkpoint", ""))).name != expected_checkpoint:
            raise ValueError(
                f"{report_label} does not use approval-bound {expected_checkpoint}"
            )

        _validate_formal_authorization(
            validated["authorization"],
            config=config,
            profile=profile,
            training_seed=validated["training_seed"],
            expected_seed_indices=expected_indices,
            label=report_label,
        )
        _validate_formal_execution_lineage(
            validated["checkpoint_lineage"],
            config=config,
            profile=profile,
            actor_type=current_actor,
            authorization=validated["authorization"],
            eval_seeds=expected_eval_seeds,
            episodes_per_seed=episodes_per_seed,
            label=report_label,
        )
        lineage = validated["checkpoint_lineage"]
        if _canonical_json(
            lineage.get("resolved_execution_spec"),
            field=f"{report_label}.resolved_execution_spec",
        ) != expected_execution_json:
            raise ValueError(f"{report_label}.resolved_execution_spec is inconsistent")
        if lineage.get("evaluation_reference_episodes_per_seed") != (
            reference_episodes_per_seed
        ):
            raise ValueError(
                f"{report_label}.evaluation_reference_episodes_per_seed is invalid"
            )
        if lineage.get("diagnostic_every_steps") != int(
            evaluation["diagnostic_every_steps"]
        ):
            raise ValueError(f"{report_label}.diagnostic_every_steps is invalid")
        expected_value_expansion = _expected_value_expansion(current_actor, config)
        if lineage.get("value_expansion") != expected_value_expansion:
            raise ValueError(f"{report_label}.value_expansion is invalid or legacy")

        action_dim = _integer(
            validated["protocol"].get("action_dim"),
            field=f"{report_label}.protocol.action_dim",
            minimum=1,
        )
        row_views = [
            _validate_row_views(
                row,
                field=f"{report_label}.per_eval_seed[{row_index}]",
                episodes_per_seed=episodes_per_seed,
                reference_episodes_per_seed=reference_episodes_per_seed,
                action_dim=action_dim,
            )
            for row_index, row in enumerate(report["per_eval_seed"])
        ]
        reference_returns = [
            value for row in row_views for value in row[REFERENCE_VIEW]["returns"]
        ]
        robustness_returns = [
            value for row in row_views for value in row[ROBUSTNESS_VIEW]["returns"]
        ]
        reference_fraction = _weighted_fraction(row_views, view=REFERENCE_VIEW)
        robustness_fraction = _weighted_fraction(row_views, view=ROBUSTNESS_VIEW)
        reference_mean = float(np.mean(reference_returns))
        robustness_mean = float(np.mean(robustness_returns))
        _validate_declared_view_summary(
            report,
            key="acme_reference_summary",
            field=report_label,
            expected_episode_count=len(reference_returns),
            expected_return_mean=reference_mean,
            expected_action_fraction=reference_fraction,
        )
        _validate_declared_view_summary(
            report,
            key="robustness_summary",
            field=report_label,
            expected_episode_count=len(robustness_returns),
            expected_return_mean=robustness_mean,
            expected_action_fraction=robustness_fraction,
        )
        method_rows.append(
            {
                "training_seed": validated["training_seed"],
                REFERENCE_VIEW: {
                    "episode_count": len(reference_returns),
                    "return_mean": reference_mean,
                    "applied_action_bound_fraction": reference_fraction,
                },
                ROBUSTNESS_VIEW: {
                    "episode_count": len(robustness_returns),
                    "return_mean": robustness_mean,
                    "applied_action_bound_fraction": robustness_fraction,
                },
            }
        )
        validated_reports.append(validated)

    method_rows.sort(key=lambda row: expected_indices[row["training_seed"]])
    actual_seeds = [row["training_seed"] for row in method_rows]
    if actual_seeds != list(expected_training_seeds):
        raise ValueError(f"{label} training-seed set/order does not match the profile")
    assert actor_type is not None

    view_summaries: dict[str, Any] = {}
    for view in (REFERENCE_VIEW, ROBUSTNESS_VIEW):
        return_means = [float(row[view]["return_mean"]) for row in method_rows]
        action_fractions = [
            float(row[view]["applied_action_bound_fraction"])
            for row in method_rows
        ]
        per_seed_episode_count = int(method_rows[0][view]["episode_count"])
        if any(
            int(row[view]["episode_count"]) != per_seed_episode_count
            for row in method_rows
        ):
            raise ValueError(
                f"{label}.{view} episode counts drift across training seeds"
            )
        view_summaries[view] = {
            "training_seed_count": len(method_rows),
            "training_seeds": list(expected_training_seeds),
            "episodes_per_training_seed": per_seed_episode_count,
            "total_episode_count": per_seed_episode_count * len(method_rows),
            "return_mean_across_training_seed_means": float(np.mean(return_means)),
            "applied_action_bound_fraction_mean_across_training_seeds": float(
                np.mean(action_fractions)
            ),
            "per_training_seed": [
                {
                    "training_seed": row["training_seed"],
                    "episode_count": row[view]["episode_count"],
                    "return_mean": row[view]["return_mean"],
                    "applied_action_bound_fraction": row[view][
                        "applied_action_bound_fraction"
                    ],
                }
                for row in method_rows
            ],
        }

    authorization = validated_reports[0]["authorization"]
    lineage = validated_reports[0]["checkpoint_lineage"]
    return {
        "actor_type": actor_type,
        "views": view_summaries,
        "protocol": validated_reports[0]["protocol"],
        "runtime_protocol": validated_reports[0]["runtime_protocol"],
        "authorization_identity": {
            **{
                field: authorization[field]
                for field in SHARED_AUTHORIZATION_FIELDS
            },
            "training_approved": True,
            "authorization_verified": True,
        },
        "resolved_execution_spec": lineage["resolved_execution_spec"],
        "evaluation_plan": {
            "evaluation_seeds": lineage["evaluation_seeds"],
            "episodes_per_seed": lineage["evaluation_episodes_per_seed"],
            "reference_episodes_per_seed": lineage[
                "evaluation_reference_episodes_per_seed"
            ],
            "diagnostic_every_steps": lineage["diagnostic_every_steps"],
            "checkpoint": evaluation["checkpoint"],
            "deterministic": evaluation["deterministic"],
        },
        "koopman_identity": (
            None
            if actor_type == "PPO"
            else {
                "koopman_sha256": lineage["koopman_sha256"],
                "koopman_lineage": lineage["koopman_lineage"],
                "koopman_dataset_sha256": lineage["koopman_dataset_sha256"],
                "koopman_config_fingerprint": lineage[
                    "koopman_config_fingerprint"
                ],
            }
        ),
    }


def _comparison_metrics(methods: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    ppo = _finite_float(
        methods["PPO"]["return_mean_across_training_seed_means"],
        field="PPO return mean",
    )
    kmpc = _finite_float(
        methods["KMPC"]["return_mean_across_training_seed_means"],
        field="KMPC return mean",
    )
    mpve = _finite_float(
        methods["AC-MPC-MPVE"]["return_mean_across_training_seed_means"],
        field="AC-MPC-MPVE return mean",
    )
    if ppo <= 0.0:
        raise ValueError("PPO return mean must be positive for the KMPC/PPO ratio")
    if kmpc <= 0.0:
        raise ValueError(
            "KMPC return mean must be positive for the AC-MPC-MPVE/KMPC ratio"
        )
    result = {
        "kmpc_minus_ppo_return": kmpc - ppo,
        "kmpc_to_ppo_return_ratio": kmpc / ppo,
        "ac_mpc_mpve_minus_kmpc_return": mpve - kmpc,
        "ac_mpc_mpve_to_kmpc_return_ratio": mpve / kmpc,
    }
    for field, value in result.items():
        _finite_float(value, field=field)
    return result


def _view_gate_report(
    methods: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, float],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    ppo_value = float(methods["PPO"]["return_mean_across_training_seed_means"])
    kmpc_ratio = float(comparisons["kmpc_to_ppo_return_ratio"])
    mpve_ratio = float(comparisons["ac_mpc_mpve_to_kmpc_return_ratio"])
    saturation = {
        actor_type: {
            "value": float(
                methods[actor_type][
                    "applied_action_bound_fraction_mean_across_training_seeds"
                ]
            ),
            "operator": "<=",
            "threshold": float(thresholds["action_bound_fraction_max"]),
            "pass": float(
                methods[actor_type][
                    "applied_action_bound_fraction_mean_across_training_seeds"
                ]
            )
            <= float(thresholds["action_bound_fraction_max"]),
        }
        for actor_type in EXPECTED_ACTOR_TYPES
    }
    result: dict[str, Any] = {
        "ppo_mean_return": {
            "value": ppo_value,
            "operator": ">=",
            "threshold": float(thresholds["ppo_mean_return_min"]),
            "pass": ppo_value >= float(thresholds["ppo_mean_return_min"]),
        },
        "kmpc_to_ppo_return_ratio": {
            "value": kmpc_ratio,
            "operator": ">=",
            "threshold": float(thresholds["kmpc_to_ppo_return_ratio_min"]),
            "pass": kmpc_ratio
            >= float(thresholds["kmpc_to_ppo_return_ratio_min"]),
        },
        "ac_mpc_mpve_to_kmpc_return_ratio": {
            "value": mpve_ratio,
            "operator": ">=",
            "threshold": float(
                thresholds["ac_mpc_mpve_to_kmpc_return_ratio_min"]
            ),
            "pass": mpve_ratio
            >= float(thresholds["ac_mpc_mpve_to_kmpc_return_ratio_min"]),
        },
        "action_saturation_by_actor": saturation,
    }
    result["all_pass"] = bool(
        result["ppo_mean_return"]["pass"]
        and result["kmpc_to_ppo_return_ratio"]["pass"]
        and result["ac_mpc_mpve_to_kmpc_return_ratio"]["pass"]
        and all(item["pass"] for item in saturation.values())
    )
    return result


def compare_aggregate_reports(
    sources: Sequence[str | Path | Mapping[str, Any]],
    *,
    config_path: str | Path,
    profile: str,
) -> dict[str, Any]:
    """Validate and compare five existing aggregate JSON reports."""

    if isinstance(sources, (str, bytes, Path)) or not isinstance(sources, Sequence):
        raise TypeError("sources must be a sequence of five aggregate reports")
    if len(sources) != len(EXPECTED_ACTOR_TYPES):
        raise ValueError("Exactly five aggregate reports are required")
    if tuple(ACTOR_TYPES) != EXPECTED_ACTOR_TYPES:
        raise RuntimeError(
            "The runtime actor registry differs from the pre-registered five methods"
        )
    if profile not in PROFILE_NAMES:
        raise ValueError(f"profile must be one of {PROFILE_NAMES}")
    config = load_experiment_config(config_path)
    if tuple(config.raw["actors"]["types"]) != EXPECTED_ACTOR_TYPES:
        raise ValueError("Config actors.types is not the exact five-method set")

    expected_count = 1 if profile == "development" else 3
    expected_training_seeds = [
        int(seed) for seed in config.raw["seeds"]["train"][:expected_count]
    ]
    path_identities: set[Path] = set()
    payloads: list[tuple[dict[str, Any], str]] = []
    for index, source in enumerate(sources):
        if isinstance(source, Mapping):
            payload = _require_mapping(source, field=f"sources[{index}]")
            label = f"in-memory report {index}"
        else:
            path = Path(source).resolve()
            if path in path_identities:
                raise ValueError("Aggregate report paths must be unique")
            path_identities.add(path)
            payload = _read_aggregate_json(path)
            label = str(path)
        payloads.append((payload, label))

    methods: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for payload, label in payloads:
        validated = _validate_method_payload(
            payload,
            config=config,
            profile=profile,
            expected_training_seeds=expected_training_seeds,
            label=label,
        )
        actor_type = validated["actor_type"]
        if actor_type in methods:
            raise ValueError(f"Duplicate actor aggregate {actor_type!r}")
        methods[actor_type] = validated["views"]
        identities[actor_type] = validated

    if set(methods) != set(EXPECTED_ACTOR_TYPES):
        missing = sorted(set(EXPECTED_ACTOR_TYPES) - set(methods))
        extra = sorted(set(methods) - set(EXPECTED_ACTOR_TYPES))
        raise ValueError(
            f"Actor aggregates must be the exact five-method set; "
            f"missing={missing}, extra={extra}"
        )

    reference = identities["PPO"]
    for actor_type in EXPECTED_ACTOR_TYPES[1:]:
        current = identities[actor_type]
        for authorization_field in SHARED_AUTHORIZATION_FIELDS:
            if current["authorization_identity"][authorization_field] != (
                reference["authorization_identity"][authorization_field]
            ):
                raise ValueError(
                    f"{actor_type} and PPO have inconsistent "
                    f"{authorization_field} identities"
                )
        for field in (
            "protocol",
            "runtime_protocol",
            "resolved_execution_spec",
            "evaluation_plan",
        ):
            if _canonical_json(current[field], field=f"{actor_type}.{field}") != (
                _canonical_json(reference[field], field=f"PPO.{field}")
            ):
                raise ValueError(
                    f"{actor_type} and PPO have inconsistent {field} identities"
                )

    structured_identity = identities["KLQR"]["koopman_identity"]
    if structured_identity is None:
        raise ValueError("KLQR is missing strict Koopman lineage")
    for actor_type in EXPECTED_ACTOR_TYPES[2:]:
        if _canonical_json(
            identities[actor_type]["koopman_identity"],
            field=f"{actor_type}.koopman_identity",
        ) != _canonical_json(structured_identity, field="KLQR.koopman_identity"):
            raise ValueError(
                f"{actor_type} and KLQR do not share the same Koopman lineage"
            )

    evaluation_views: dict[str, Any] = {}
    for view, role in (
        (REFERENCE_VIEW, "primary_gate_binding"),
        (ROBUSTNESS_VIEW, "secondary_robustness_not_gate_binding"),
    ):
        view_methods = {
            actor_type: methods[actor_type][view]
            for actor_type in EXPECTED_ACTOR_TYPES
        }
        comparisons = _comparison_metrics(view_methods)
        evaluation_views[view] = {
            "role": role,
            "episodes_per_evaluation_seed": (
                int(config.raw["evaluation"]["reference_episodes_per_seed"])
                if view == REFERENCE_VIEW
                else int(config.raw["evaluation"]["episodes_per_seed"])
            ),
            "episodes_per_training_seed": view_methods["PPO"][
                "episodes_per_training_seed"
            ],
            "total_episodes_per_actor": view_methods["PPO"][
                "total_episode_count"
            ],
            "methods": view_methods,
            "comparisons": comparisons,
        }

    thresholds = {
        key: _finite_float(
            config.raw["proposed_gates"][key], field=f"proposed_gates.{key}"
        )
        for key in (
            "ppo_mean_return_min",
            "kmpc_to_ppo_return_ratio_min",
            "ac_mpc_mpve_to_kmpc_return_ratio_min",
            "action_bound_fraction_max",
        )
    }
    primary_gate = _view_gate_report(
        evaluation_views[REFERENCE_VIEW]["methods"],
        evaluation_views[REFERENCE_VIEW]["comparisons"],
        thresholds=thresholds,
    )
    secondary_diagnostic = _view_gate_report(
        evaluation_views[ROBUSTNESS_VIEW]["methods"],
        evaluation_views[ROBUSTNESS_VIEW]["comparisons"],
        thresholds=thresholds,
    )
    secondary_diagnostic.update(
        {
            "gate_binding": False,
            "affects_overall_control_primary_pass": False,
            "diagnostic_all_pass": secondary_diagnostic["all_pass"],
        }
    )

    result: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "kind": COMPARISON_KIND,
        "task": config.task,
        "profile": profile,
        "config_path": str(config.path.resolve()),
        "config_fingerprint": config.fingerprint,
        "actor_types": list(EXPECTED_ACTOR_TYPES),
        "training_seeds": expected_training_seeds,
        "training_seed_count": expected_count,
        "approval_identity": reference["authorization_identity"],
        "evaluation_plan": reference["evaluation_plan"],
        "protocol": reference["protocol"],
        "runtime_protocol": reference["runtime_protocol"],
        "shared_koopman_identity": structured_identity,
        "evaluation_views": evaluation_views,
        "gate_report": {
            "binding": "primary_reference10",
            "thresholds": thresholds,
            "primary_reference10": primary_gate,
            "secondary_robustness_not_gate_binding": secondary_diagnostic,
            "overall_control_primary_pass": primary_gate["all_pass"],
        },
    }
    _canonical_json(result, field="comparison result")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        action="append",
        required=True,
        help="existing aggregate JSON; repeat exactly five times",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths = {path.resolve() for path in args.aggregate_report}
    if args.output.resolve() in input_paths:
        raise ValueError("Comparison output must not overwrite an input aggregate")
    report = compare_aggregate_reports(
        args.aggregate_report,
        config_path=args.config,
        profile=args.profile,
    )
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
