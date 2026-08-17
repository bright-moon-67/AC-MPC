from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.dmc.config import (
    default_config_path,
    load_experiment_config,
    resolve_execution_spec,
)
from experiments.dmc.eval import aggregate_dmc
from experiments.dmc.eval.compare_dmc import (
    EXPECTED_ACTOR_TYPES,
    compare_aggregate_reports,
    main,
)
from experiments.dmc.reward_oracle import exact_reward_oracle_metadata


TASK = "cartpole_swingup"


def _protocol(*, dm_control_version: str = "test") -> dict:
    return {
        "protocol_name": "dmc_native_v1",
        "protocol_schema_version": 1,
        "task": TASK,
        "domain": "cartpole",
        "dmc_task": "cartpole:swingup",
        "dm_control_version": dm_control_version,
        "mujoco_version": "test",
        "obs_dim": 5,
        "action_dim": 1,
        "control_dt": 0.01,
        "physics_dt": 0.01,
        "n_substeps": 1,
        "time_limit": 10.0,
        "step_limit": 1000,
        "action_low": [-1.0],
        "action_high": [1.0],
        "obs_layout": [["position", 3], ["velocity", 2]],
    }


def _value_expansion(actor_type: str, config) -> dict:
    enabled = actor_type == "AC-MPC-MPVE"
    return {
        "enabled": enabled,
        "kind": "mpve_td_k_tro25_eq8_eq9_v1" if enabled else None,
        "actor_shared_with": "KMPC" if enabled else None,
        "horizon": config.raw["ppo"]["mpve_horizon"] if enabled else None,
        "value_loss_coefficient": (
            config.raw["ppo"]["mpve_value_loss_coefficient"]
            if enabled
            else None
        ),
        "prediction_gradient": "detached",
        "terminal_target_gradient": "detached",
        "standard_gae_value_loss_retained": True,
        "reward": exact_reward_oracle_metadata(TASK) if enabled else None,
    }


def _authorization(config, *, profile: str, index: int) -> dict:
    return {
        "authorization_kind": "dmc_training_approval_v1",
        "training_approved": True,
        "config_fingerprint": config.fingerprint,
        "approval_profile": profile,
        "approval_file_sha256": "2" * 64,
        "preflight_report_sha256": "3" * 64,
        "train_seed_index": index,
        "authorization_verified": True,
        "authorization_errors": [],
    }


def _lineage(config, *, profile: str, actor_type: str) -> dict:
    structured = actor_type != "PPO"
    koopman_lineage = (
        {
            "dataset_sha256": "4" * 64,
            "config_fingerprint": config.fingerprint,
            "approval_profile": profile,
            "approval_file_sha256": "2" * 64,
            "preflight_report_sha256": "3" * 64,
        }
        if structured
        else None
    )
    return {
        "resolved_execution_spec": resolve_execution_spec(config, profile),
        "evaluation_seeds": list(config.raw["seeds"]["evaluation"]),
        "evaluation_episodes_per_seed": config.raw["evaluation"][
            "episodes_per_seed"
        ],
        "evaluation_reference_episodes_per_seed": config.raw["evaluation"][
            "reference_episodes_per_seed"
        ],
        "diagnostic_every_steps": config.raw["evaluation"][
            "diagnostic_every_steps"
        ],
        "koopman_sha256": "5" * 64 if structured else None,
        "koopman_lineage": koopman_lineage,
        "koopman_dataset_sha256": "4" * 64 if structured else None,
        "koopman_config_fingerprint": config.fingerprint if structured else None,
        "value_expansion": _value_expansion(actor_type, config),
    }


def _report(
    config,
    actor_type: str,
    *,
    profile: str = "development",
    seed_index: int = 0,
    reference_return: float,
    robustness_other_return: float | None = None,
    reference_saturation: float = 0.1,
    robustness_saturation: float = 0.2,
    protocol: dict | None = None,
) -> dict:
    training_seed = config.raw["seeds"]["train"][seed_index]
    eval_seeds = list(config.raw["seeds"]["evaluation"])
    episodes_per_seed = config.raw["evaluation"]["episodes_per_seed"]
    reference_count = config.raw["evaluation"]["reference_episodes_per_seed"]
    assert reference_count == 1
    protocol = copy.deepcopy(protocol or _protocol())
    authorization = _authorization(config, profile=profile, index=seed_index)
    lineage = _lineage(config, profile=profile, actor_type=actor_type)
    other_return = (
        reference_return
        if robustness_other_return is None
        else robustness_other_return
    )
    rows = []
    for eval_seed in eval_seeds:
        returns = [reference_return] + [other_return] * (episodes_per_seed - 1)
        action_counts = [1000] * episodes_per_seed
        reference_applied_count = int(round(reference_saturation * 1000))
        robustness_applied_count = int(
            round(robustness_saturation * sum(action_counts))
        )
        remaining_applied = robustness_applied_count - reference_applied_count
        base_applied, remainder = divmod(
            remaining_applied, episodes_per_seed - reference_count
        )
        applied_counts = [reference_applied_count] + [
            base_applied + int(index < remainder)
            for index in range(episodes_per_seed - reference_count)
        ]
        assert all(0 <= count <= 1000 for count in applied_counts)
        row = {
            "schema_version": "dmc_evaluation_v1",
            "kind": "dmc_deterministic_evaluation",
            "task": TASK,
            "actor_type": actor_type,
            "actor_config": copy.deepcopy(config.actor_config.to_dict()),
            "value_expansion": copy.deepcopy(lineage["value_expansion"]),
            "training_seed": training_seed,
            **copy.deepcopy(authorization),
            **copy.deepcopy(lineage),
            "eval_seed": eval_seed,
            "deterministic": True,
            "episodes": episodes_per_seed,
            "protocol": copy.deepcopy(protocol),
            "runtime_protocol": copy.deepcopy(protocol),
            "return_mean_across_episodes": float(np.mean(returns)),
            "episode_returns": returns,
            "episode_lengths": [1000] * episodes_per_seed,
            "acme_reference_episode_returns": returns[:reference_count],
            "acme_reference_episode_count": reference_count,
            "acme_reference_return_mean": float(
                np.mean(returns[:reference_count])
            ),
            "robustness_episode_returns": list(returns),
            "robustness_episode_count": episodes_per_seed,
            "robustness_return_mean": float(np.mean(returns)),
            "robustness_return_population_std": float(np.std(returns, ddof=0)),
            "episode_action_component_counts": action_counts,
            "episode_applied_action_bound_counts": applied_counts,
            "acme_reference_action_component_count": sum(
                action_counts[:reference_count]
            ),
            "acme_reference_applied_action_bound_count": sum(
                applied_counts[:reference_count]
            ),
            "acme_reference_applied_action_bound_fraction": reference_saturation,
            "robustness_action_component_count": sum(action_counts),
            "robustness_applied_action_bound_count": sum(applied_counts),
            "robustness_applied_action_bound_fraction": robustness_saturation,
            "applied_action_bound_fraction": robustness_saturation,
            "requested_action_bound_fraction": robustness_saturation,
            "action_clipped_fraction": 0.0,
            "episode_length_mean": 1000.0,
            "mean_step_reward": float(np.mean(returns)) / 1000.0,
            "mean_reward_components": {},
        }
        rows.append(row)
    eval_means = [row["return_mean_across_episodes"] for row in rows]
    pooled = [value for row in rows for value in row["episode_returns"]]
    reference_returns = [
        value for row in rows for value in row["acme_reference_episode_returns"]
    ]
    reference_action_count = sum(
        row["acme_reference_action_component_count"] for row in rows
    )
    reference_applied_count = sum(
        row["acme_reference_applied_action_bound_count"] for row in rows
    )
    robustness_action_count = sum(
        row["robustness_action_component_count"] for row in rows
    )
    robustness_applied_count = sum(
        row["robustness_applied_action_bound_count"] for row in rows
    )
    report = {
        "schema_version": aggregate_dmc.AGGREGATE_SCHEMA_VERSION,
        "kind": "dmc_ten_eval_seed_aggregate",
        "task": TASK,
        "actor_type": actor_type,
        "actor_checkpoint": f"/fake/{training_seed}/{actor_type}/latest.pt",
        "actor_config": copy.deepcopy(config.actor_config.to_dict()),
        "training_seed": training_seed,
        **copy.deepcopy(authorization),
        **copy.deepcopy(lineage),
        "training_seed_count": 1,
        "eval_seeds": eval_seeds,
        "eval_seed_count": len(eval_seeds),
        "episodes_per_eval_seed": episodes_per_seed,
        "total_evaluation_episodes": len(pooled),
        "protocol": copy.deepcopy(protocol),
        "runtime_protocol": copy.deepcopy(protocol),
        "return_mean_across_eval_seed_means": float(np.mean(eval_means)),
        "pooled_episode_return_mean": float(np.mean(pooled)),
        "acme_reference_summary": {
            "kind": "acme_aligned_preregistered_reference_v1",
            "episode_selection": "first_episode_per_eval_seed_prefix_v1",
            "eval_seed_count": len(eval_seeds),
            "episodes_per_eval_seed": reference_count,
            "episode_count": len(reference_returns),
            "episode_returns": reference_returns,
            "return_mean": float(np.mean(reference_returns)),
            "return_population_std": float(np.std(reference_returns, ddof=0)),
            "action_component_count": reference_action_count,
            "applied_action_bound_count": reference_applied_count,
            "applied_action_bound_fraction": (
                reference_applied_count / reference_action_count
            ),
            "descriptive_only": True,
            "return_standard_error": None,
            "return_95ci": None,
        },
        "robustness_summary": {
            "kind": "dmc_nested_episode_robustness_v1",
            "episode_selection": "all_episodes_per_eval_seed_v1",
            "eval_seed_count": len(eval_seeds),
            "episodes_per_eval_seed": episodes_per_seed,
            "episode_count": len(pooled),
            "episode_returns": pooled,
            "return_mean": float(np.mean(pooled)),
            "return_population_std": float(np.std(pooled, ddof=0)),
            "action_component_count": robustness_action_count,
            "applied_action_bound_count": robustness_applied_count,
            "applied_action_bound_fraction": (
                robustness_applied_count / robustness_action_count
            ),
            "descriptive_only": True,
            "return_standard_error": None,
            "return_95ci": None,
        },
        "per_eval_seed": rows,
    }
    return report


def _development_reports(config=None) -> list[dict]:
    config = config or load_experiment_config(default_config_path(TASK))
    values = {
        "PPO": (800.0, 800.0, 0.10, 0.20),
        "KLQR": (780.0, 780.0, 0.11, 0.21),
        "AB-PQ": (790.0, 790.0, 0.12, 0.22),
        # Primary ratios pass; the robustness ratios deliberately fail.
        "KMPC": (760.0, 400.0, 0.13, 0.60),
        "AC-MPC-MPVE": (770.0, 390.0, 0.14, 0.61),
    }
    return [
        _report(
            config,
            actor_type,
            reference_return=values[actor_type][0],
            robustness_other_return=values[actor_type][1],
            reference_saturation=values[actor_type][2],
            robustness_saturation=values[actor_type][3],
        )
        for actor_type in EXPECTED_ACTOR_TYPES
    ]


def _benchmark_aggregate(config, actor_type: str, base_return: float) -> dict:
    reports = [
        _report(
            config,
            actor_type,
            profile="benchmark",
            seed_index=index,
            reference_return=base_return + index,
            robustness_other_return=base_return + index,
        )
        for index in range(3)
    ]
    result = aggregate_dmc.aggregate_training_seed_reports(
        reports,
        expected_training_seeds=config.raw["seeds"]["train"],
    )
    result.update(
        {
            "profile": "benchmark",
            "config_path": str(config.path.resolve()),
            "evaluation_checkpoint": config.raw["evaluation"]["checkpoint"],
            "sources": [
                {
                    "kind": "aggregate",
                    "source": f"/fake/{seed}/{actor_type}.json",
                    "training_seed": seed,
                    "train_seed_index": index,
                }
                for index, seed in enumerate(config.raw["seeds"]["train"])
            ],
        }
    )
    return result


def test_comparison_reports_both_views_and_only_primary_binds_gates():
    config = load_experiment_config(default_config_path(TASK))

    result = compare_aggregate_reports(
        list(reversed(_development_reports(config))),
        config_path=config.path,
        profile="development",
    )

    assert result["actor_types"] == list(EXPECTED_ACTOR_TYPES)
    reference = result["evaluation_views"]["reference10"]
    robustness = result["evaluation_views"]["robustness100"]
    assert reference["total_episodes_per_actor"] == 10
    assert robustness["total_episodes_per_actor"] == 100
    assert reference["methods"]["PPO"][
        "return_mean_across_training_seed_means"
    ] == pytest.approx(800.0)
    assert reference["comparisons"]["kmpc_minus_ppo_return"] == pytest.approx(
        -40.0
    )
    assert reference["comparisons"]["kmpc_to_ppo_return_ratio"] == pytest.approx(
        0.95
    )
    assert reference["comparisons"][
        "ac_mpc_mpve_minus_kmpc_return"
    ] == pytest.approx(10.0)
    assert reference["comparisons"][
        "ac_mpc_mpve_to_kmpc_return_ratio"
    ] == pytest.approx(770.0 / 760.0)

    gates = result["gate_report"]
    assert gates["primary_reference10"]["all_pass"] is True
    assert gates["secondary_robustness_not_gate_binding"]["all_pass"] is False
    assert gates["secondary_robustness_not_gate_binding"][
        "affects_overall_control_primary_pass"
    ] is False
    assert gates["overall_control_primary_pass"] is True
    assert gates["primary_reference10"]["action_saturation_by_actor"]["KMPC"][
        "value"
    ] == pytest.approx(0.13)


def test_benchmark_aggregate_uses_training_seed_axis_and_30_300_episodes():
    config = load_experiment_config(default_config_path(TASK))
    bases = {
        "PPO": 800.0,
        "KLQR": 780.0,
        "AB-PQ": 790.0,
        "KMPC": 760.0,
        "AC-MPC-MPVE": 770.0,
    }
    reports = [
        _benchmark_aggregate(config, actor_type, bases[actor_type])
        for actor_type in EXPECTED_ACTOR_TYPES
    ]

    result = compare_aggregate_reports(
        reports, config_path=config.path, profile="benchmark"
    )

    reference = result["evaluation_views"]["reference10"]
    robustness = result["evaluation_views"]["robustness100"]
    assert result["training_seed_count"] == 3
    assert reference["episodes_per_training_seed"] == 10
    assert reference["total_episodes_per_actor"] == 30
    assert robustness["episodes_per_training_seed"] == 100
    assert robustness["total_episodes_per_actor"] == 300
    assert reference["methods"]["PPO"][
        "return_mean_across_training_seed_means"
    ] == pytest.approx(801.0)
    assert len(reference["methods"]["PPO"]["per_training_seed"]) == 3

    reports[0]["sources"] = list(reversed(reports[0]["sources"]))
    with pytest.raises(ValueError, match="training-seed order"):
        compare_aggregate_reports(
            reports, config_path=config.path, profile="benchmark"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_actor", "Duplicate actor"),
        ("legacy", "legacy or unsupported"),
        ("unverified", "unverified training authorization"),
        ("config", "config_fingerprint"),
        ("profile", "approval_profile"),
        ("training_seed", "training_seed"),
        ("eval_plan", "evaluation_seeds"),
        ("reference_plan", "reference/diagnostic plan"),
    ],
)
def test_rejects_duplicate_legacy_unverified_and_lineage_drift(
    mutation: str, message: str
):
    config = load_experiment_config(default_config_path(TASK))
    reports = _development_reports(config)
    changed = reports[1]
    if mutation == "duplicate_actor":
        reports[1] = copy.deepcopy(reports[0])
    elif mutation == "legacy":
        changed["schema_version"] = "dmc_evaluation_aggregate_legacy"
    elif mutation == "unverified":
        changed["authorization_verified"] = False
        for row in changed["per_eval_seed"]:
            row["authorization_verified"] = False
    elif mutation == "config":
        changed["config_fingerprint"] = "sha256:" + "9" * 64
        for row in changed["per_eval_seed"]:
            row["config_fingerprint"] = changed["config_fingerprint"]
    elif mutation == "profile":
        changed["approval_profile"] = "benchmark"
        for row in changed["per_eval_seed"]:
            row["approval_profile"] = "benchmark"
    elif mutation == "training_seed":
        changed["training_seed"] += 999
        for row in changed["per_eval_seed"]:
            row["training_seed"] = changed["training_seed"]
    elif mutation == "eval_plan":
        changed["evaluation_seeds"] = list(reversed(changed["evaluation_seeds"]))
        changed["resolved_execution_spec"]["evaluation_seeds"] = list(
            changed["evaluation_seeds"]
        )
        for row in changed["per_eval_seed"]:
            row["evaluation_seeds"] = list(changed["evaluation_seeds"])
            row["resolved_execution_spec"] = copy.deepcopy(
                changed["resolved_execution_spec"]
            )
    elif mutation == "reference_plan":
        changed["evaluation_reference_episodes_per_seed"] = 2
        for row in changed["per_eval_seed"]:
            row["evaluation_reference_episodes_per_seed"] = 2

    with pytest.raises((PermissionError, ValueError), match=message):
        compare_aggregate_reports(
            reports, config_path=config.path, profile="development"
        )


@pytest.mark.parametrize(
    "identity", ["approval_file_sha256", "preflight_report_sha256"]
)
def test_rejects_cross_actor_approval_or_preflight_identity_drift(identity: str):
    config = load_experiment_config(default_config_path(TASK))
    reports = _development_reports(config)
    changed = reports[1]
    changed[identity] = "8" * 64
    changed["koopman_lineage"][identity.removesuffix("_sha256")] = "8" * 64
    # Koopman lineage uses the same full authorization field names.
    changed["koopman_lineage"][identity] = changed["koopman_lineage"].pop(
        identity.removesuffix("_sha256")
    )
    for row in changed["per_eval_seed"]:
        row[identity] = "8" * 64
        row["koopman_lineage"] = copy.deepcopy(changed["koopman_lineage"])

    with pytest.raises(ValueError, match=identity):
        compare_aggregate_reports(
            reports, config_path=config.path, profile="development"
        )


def test_rejects_protocol_drift_missing_reference_schema_and_nonfinite():
    config = load_experiment_config(default_config_path(TASK))

    protocol_reports = _development_reports(config)
    changed_protocol = _protocol(dm_control_version="different")
    protocol_reports[1]["protocol"] = copy.deepcopy(changed_protocol)
    protocol_reports[1]["runtime_protocol"] = copy.deepcopy(changed_protocol)
    for row in protocol_reports[1]["per_eval_seed"]:
        row["protocol"] = copy.deepcopy(changed_protocol)
        row["runtime_protocol"] = copy.deepcopy(changed_protocol)
    with pytest.raises(ValueError, match="protocol"):
        compare_aggregate_reports(
            protocol_reports, config_path=config.path, profile="development"
        )

    missing = _development_reports(config)
    del missing[0]["per_eval_seed"][0][
        "acme_reference_applied_action_bound_fraction"
    ]
    with pytest.raises(ValueError, match="acme_reference_applied"):
        compare_aggregate_reports(
            missing, config_path=config.path, profile="development"
        )

    nonfinite = _development_reports(config)
    nonfinite[0]["per_eval_seed"][0]["episode_returns"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compare_aggregate_reports(
            nonfinite, config_path=config.path, profile="development"
        )


def test_zero_ratio_denominator_is_rejected_instead_of_emitting_nonfinite():
    config = load_experiment_config(default_config_path(TASK))
    reports = _development_reports(config)
    ppo = reports[0]
    for row in ppo["per_eval_seed"]:
        row["episode_returns"] = [0.0] * 10
        row["return_mean_across_episodes"] = 0.0
        row["acme_reference_episode_returns"] = [0.0]
        row["acme_reference_return_mean"] = 0.0
        row["robustness_episode_returns"] = [0.0] * 10
        row["robustness_return_mean"] = 0.0
        row["robustness_return_population_std"] = 0.0
    ppo["return_mean_across_eval_seed_means"] = 0.0
    ppo["pooled_episode_return_mean"] = 0.0
    ppo["acme_reference_summary"]["episode_returns"] = [0.0] * 10
    ppo["acme_reference_summary"]["return_mean"] = 0.0
    ppo["acme_reference_summary"]["return_population_std"] = 0.0
    ppo["robustness_summary"]["episode_returns"] = [0.0] * 100
    ppo["robustness_summary"]["return_mean"] = 0.0
    ppo["robustness_summary"]["return_population_std"] = 0.0

    with pytest.raises(ValueError, match="PPO return mean must be positive"):
        compare_aggregate_reports(
            reports, config_path=config.path, profile="development"
        )


def test_cli_reads_only_json_and_writes_atomic_result(tmp_path: Path):
    config = load_experiment_config(default_config_path(TASK))
    input_paths = []
    for report in _development_reports(config):
        path = tmp_path / f"{report['actor_type']}.json"
        path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
        input_paths.append(path)
    output = tmp_path / "comparison.json"
    argv = ["--config", str(config.path), "--profile", "development"]
    for path in input_paths:
        argv.extend(["--aggregate-report", str(path)])
    argv.extend(["--output", str(output)])

    main(argv)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["kind"] == "dmc_five_method_comparison"
    assert payload["gate_report"]["overall_control_primary_pass"] is True
    assert not output.with_suffix(".json.tmp").exists()


def test_json_reader_rejects_duplicate_and_nonfinite_constants(tmp_path: Path):
    config = load_experiment_config(default_config_path(TASK))
    valid = _development_reports(config)
    valid_paths = []
    for index, report in enumerate(valid[1:], start=1):
        path = tmp_path / f"valid_{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        valid_paths.append(path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"x","schema_version":"y"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        compare_aggregate_reports(
            [duplicate, *valid_paths],
            config_path=config.path,
            profile="development",
        )

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite JSON constant"):
        compare_aggregate_reports(
            [nonfinite, *valid_paths],
            config_path=config.path,
            profile="development",
        )
