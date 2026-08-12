from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.dmc.actors import ActorConfig
from experiments.dmc.config import (
    default_config_path,
    load_experiment_config,
    resolve_execution_spec,
)
from experiments.dmc.eval import aggregate_dmc


TASK = "cartpole_swingup"


def _protocol(task: str = TASK, *, control_dt: float = 0.01) -> dict:
    return {
        "protocol_name": "dmc_native_v1",
        "protocol_schema_version": 1,
        "task": task,
        "domain": task.split("_", 1)[0],
        "dmc_task": task.replace("_", ":", 1),
        "dm_control_version": "test",
        "mujoco_version": "test",
        "obs_dim": 5,
        "action_dim": 1,
        "control_dt": control_dt,
        "physics_dt": 0.01,
        "n_substeps": max(1, int(round(control_dt / 0.01))),
        "time_limit": 10.0,
        "step_limit": 1000,
        "action_low": [-1.0],
        "action_high": [1.0],
        "obs_layout": [["position", 3], ["velocity", 2]],
    }


def _single_seed_report(
    training_seed: int,
    training_mean: float,
    *,
    task: str = TASK,
    actor_type: str = "PPO",
    actor_config: dict | None = None,
    protocol: dict | None = None,
    eval_seeds: list[int] | None = None,
    episodes_per_seed: int = 2,
    authorization: dict | None = None,
) -> dict:
    actor_config = actor_config or ActorConfig().to_dict()
    protocol = copy.deepcopy(protocol or _protocol(task))
    eval_seeds = list(eval_seeds or range(100, 110))
    resolved_execution_spec = None
    if authorization is not None:
        config = load_experiment_config(default_config_path(task))
        resolved_execution_spec = resolve_execution_spec(
            config, authorization["approval_profile"]
        )
    reference_episodes_per_seed = 1
    diagnostic_every_steps = 50_000
    if resolved_execution_spec is not None:
        reference_episodes_per_seed = int(
            resolved_execution_spec["evaluation"][
                "reference_episodes_per_seed"
            ]
        )
        diagnostic_every_steps = int(
            resolved_execution_spec["evaluation"]["diagnostic_every_steps"]
        )
    checkpoint_lineage = {
        "resolved_execution_spec": copy.deepcopy(resolved_execution_spec),
        "evaluation_seeds": (
            list(eval_seeds) if authorization is not None else None
        ),
        "evaluation_episodes_per_seed": (
            episodes_per_seed if authorization is not None else None
        ),
        "evaluation_reference_episodes_per_seed": (
            reference_episodes_per_seed
        ),
        "diagnostic_every_steps": diagnostic_every_steps,
        "koopman_sha256": None,
        "koopman_lineage": None,
        "koopman_dataset_sha256": None,
        "koopman_config_fingerprint": None,
        "value_expansion": None,
    }
    rows = []
    for index, eval_seed in enumerate(eval_seeds):
        eval_mean = training_mean + 0.1 * (index - 4.5)
        episode_offsets = [
            0.1 * (episode - (episodes_per_seed - 1) / 2.0)
            for episode in range(episodes_per_seed)
        ]
        episode_returns = [eval_mean + offset for offset in episode_offsets]
        reference_returns = episode_returns[:reference_episodes_per_seed]
        episode_action_counts = [3] * episodes_per_seed
        episode_applied_counts = [0] * episodes_per_seed
        row = {
            "task": task,
            "actor_type": actor_type,
            "training_seed": training_seed,
            "eval_seed": eval_seed,
            "runtime_protocol": copy.deepcopy(protocol),
            "episodes": episodes_per_seed,
            "return_mean_across_episodes": float(np.mean(episode_returns)),
            "episode_returns": episode_returns,
            "acme_reference_episode_returns": reference_returns,
            "acme_reference_episode_count": len(reference_returns),
            "acme_reference_return_mean": float(np.mean(reference_returns)),
            "robustness_episode_returns": episode_returns,
            "robustness_episode_count": len(episode_returns),
            "robustness_return_mean": float(np.mean(episode_returns)),
            "episode_action_component_counts": episode_action_counts,
            "episode_applied_action_bound_counts": episode_applied_counts,
            "acme_reference_action_component_count": (
                3 * reference_episodes_per_seed
            ),
            "acme_reference_applied_action_bound_count": 0,
            "acme_reference_applied_action_bound_fraction": 0.0,
            "robustness_action_component_count": 3 * episodes_per_seed,
            "robustness_applied_action_bound_count": 0,
            "robustness_applied_action_bound_fraction": 0.0,
            "applied_action_bound_fraction": 0.0,
            "mean_reward_components": {},
            **copy.deepcopy(checkpoint_lineage),
        }
        if authorization is None:
            row.update(
                {
                    "authorization_kind": None,
                    "training_approved": None,
                    "config_fingerprint": None,
                    "approval_profile": None,
                    "approval_file_sha256": None,
                    "preflight_report_sha256": None,
                    "train_seed_index": None,
                    "authorization_verified": False,
                    "authorization_errors": ["legacy synthetic report"],
                }
            )
        else:
            row.update(
                {
                    **authorization,
                    "authorization_verified": True,
                    "authorization_errors": [],
                }
            )
        rows.append(row)
    eval_means = [row["return_mean_across_episodes"] for row in rows]
    pooled = [value for row in rows for value in row["episode_returns"]]
    reference_pooled = [
        value
        for row in rows
        for value in row["acme_reference_episode_returns"]
    ]
    report = {
        "schema_version": aggregate_dmc.AGGREGATE_SCHEMA_VERSION,
        "kind": "dmc_ten_eval_seed_aggregate",
        "task": task,
        "actor_type": actor_type,
        "actor_checkpoint": f"/fake/seed_{training_seed}/latest.pt",
        "actor_config": copy.deepcopy(actor_config),
        "training_seed": training_seed,
        **copy.deepcopy(checkpoint_lineage),
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
            "episodes_per_eval_seed": reference_episodes_per_seed,
            "episode_count": len(reference_pooled),
            "episode_returns": reference_pooled,
            "return_mean": float(np.mean(reference_pooled)),
            "return_population_std": float(np.std(reference_pooled, ddof=0)),
            "action_component_count": (
                len(eval_seeds) * reference_episodes_per_seed * 3
            ),
            "applied_action_bound_count": 0,
            "applied_action_bound_fraction": 0.0,
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
            "action_component_count": len(eval_seeds) * episodes_per_seed * 3,
            "applied_action_bound_count": 0,
            "applied_action_bound_fraction": 0.0,
            "descriptive_only": True,
            "return_standard_error": None,
            "return_95ci": None,
        },
        "per_eval_seed": rows,
    }
    if authorization is None:
        report.update(
            {
                "authorization_kind": None,
                "training_approved": None,
                "config_fingerprint": None,
                "approval_profile": None,
                "approval_file_sha256": None,
                "preflight_report_sha256": None,
                "train_seed_index": None,
                "authorization_verified": False,
                "authorization_errors": ["legacy synthetic report"],
            }
        )
    else:
        report.update(
            {
                **authorization,
                "authorization_verified": True,
                "authorization_errors": [],
            }
        )
    return report


def _formal_authorization(
    config,
    *,
    profile: str,
    index: int,
    approval_hash: str = "2" * 64,
    preflight_hash: str = "3" * 64,
) -> dict:
    return {
        "authorization_kind": "dmc_training_approval_v1",
        "training_approved": True,
        "config_fingerprint": config.fingerprint,
        "approval_profile": profile,
        "approval_file_sha256": approval_hash,
        "preflight_report_sha256": preflight_hash,
        "train_seed_index": index,
    }


def _replace_authorization(report: dict, authorization: dict) -> None:
    report.update(
        {
            **authorization,
            "authorization_verified": True,
            "authorization_errors": [],
        }
    )
    for row in report["per_eval_seed"]:
        row.update(
            {
                **authorization,
                "authorization_verified": True,
                "authorization_errors": [],
            }
        )


def _checkpoint_metadata_from_report(config, report: dict) -> SimpleNamespace:
    authorization = aggregate_dmc._authorization_metadata(report)
    return SimpleNamespace(
        task=report["task"],
        actor_type=report["actor_type"],
        actor_config=config.actor_config,
        protocol=copy.deepcopy(report["protocol"]),
        training_seed=report["training_seed"],
        authorization=authorization,
        authorization_verified=authorization["authorization_verified"],
        payload={
            "environment_protocol_json": json.dumps(
                report["protocol"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            "resolved_execution_spec": copy.deepcopy(
                report["resolved_execution_spec"]
            ),
            "evaluation_seeds": list(report["evaluation_seeds"]),
            "evaluation_episodes_per_seed": report[
                "evaluation_episodes_per_seed"
            ],
        },
    )


def test_one_training_seed_makes_inference_explicitly_unestimable():
    report = aggregate_dmc.aggregate_training_seed_reports(
        [_single_seed_report(11, 25.0)]
    )

    training = report["training_seed_statistics"]
    assert report["inference_axis"] == "training_seed"
    assert report["authorization_verified"] is False
    assert report["training_approved"] is False
    assert training["count"] == 1
    assert training["return_mean"] == pytest.approx(25.0)
    assert training["return_sample_std"] is None
    assert training["return_standard_error"] is None
    assert training["return_student_t_95ci"] is None
    assert training["inference_estimable"] is False
    assert report["return_std_across_training_seed_means"] is None
    assert report["return_standard_error_across_training_seed_means"] is None
    assert report["return_student_t_95ci_across_training_seed_means"] is None
    assert report["eval_seed_statistics"]["descriptive_only"] is True
    assert report["eval_seed_statistics"]["return_standard_error"] is None
    assert report["episode_statistics"]["descriptive_only"] is True
    assert report["episode_statistics"]["return_95ci"] is None


def test_three_training_seeds_use_student_t_not_nested_sample_count():
    means = np.asarray([10.0, 20.0, 40.0])
    reports = [
        _single_seed_report(seed, mean)
        for seed, mean in zip((11, 12, 13), means, strict=True)
    ]

    report = aggregate_dmc.aggregate_training_seed_reports(reports)

    expected_mean = float(np.mean(means))
    expected_std = float(np.std(means, ddof=1))
    expected_se = expected_std / math.sqrt(3)
    half_width = aggregate_dmc.STUDENT_T_95_CRITICAL_DF2 * expected_se
    stats = report["training_seed_statistics"]
    assert stats["count"] == 3
    assert stats["return_mean"] == pytest.approx(expected_mean)
    assert stats["return_sample_std"] == pytest.approx(expected_std)
    assert stats["return_standard_error"] == pytest.approx(expected_se)
    assert stats["return_student_t_95ci"] == pytest.approx(
        [expected_mean - half_width, expected_mean + half_width]
    )
    assert stats["degrees_of_freedom"] == 2
    assert report["eval_seed_statistics"]["total_nested_cells"] == 30
    assert report["episode_statistics"]["total_nested_episodes"] == 60
    # Thirty eval means and sixty episodes remain descriptive; neither changes
    # the sqrt(3) denominator used for independent training replicates.
    assert report["eval_seed_statistics"]["return_standard_error"] is None
    assert report["episode_statistics"]["return_standard_error"] is None


def test_duplicate_training_seeds_are_rejected():
    reports = [
        _single_seed_report(11, 10.0),
        _single_seed_report(11, 20.0),
        _single_seed_report(13, 30.0),
    ]
    with pytest.raises(ValueError, match="unique independent"):
        aggregate_dmc.aggregate_training_seed_reports(reports)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("task", "different tasks"),
        ("actor", "different actor types"),
        ("actor_config", "different actor_config"),
        ("protocol", "different protocols"),
        ("eval_plan", "different eval seed plans"),
    ],
)
def test_cross_training_identity_drift_is_rejected(change: str, message: str):
    reports = [
        _single_seed_report(11, 10.0),
        _single_seed_report(12, 20.0),
        _single_seed_report(13, 30.0),
    ]
    if change == "task":
        reports[1] = _single_seed_report(12, 20.0, task="reacher_hard")
    elif change == "actor":
        reports[1] = _single_seed_report(12, 20.0, actor_type="KLQR")
    elif change == "actor_config":
        reports[1] = _single_seed_report(
            12, 20.0, actor_config=ActorConfig(hidden_dim=64).to_dict()
        )
    elif change == "protocol":
        reports[1] = _single_seed_report(
            12, 20.0, protocol=_protocol(control_dt=0.02)
        )
    elif change == "eval_plan":
        reports[1] = _single_seed_report(
            12, 20.0, eval_seeds=list(range(200, 210))
        )

    with pytest.raises(ValueError, match=message):
        aggregate_dmc.aggregate_training_seed_reports(reports)


@pytest.mark.parametrize("location", ["episode", "eval_mean", "aggregate_mean"])
def test_non_finite_data_at_every_nested_level_is_rejected(location: str):
    report = _single_seed_report(11, 10.0)
    if location == "episode":
        report["per_eval_seed"][0]["episode_returns"][0] = float("nan")
    elif location == "eval_mean":
        report["per_eval_seed"][0]["return_mean_across_episodes"] = float("inf")
    else:
        report["return_mean_across_eval_seed_means"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        aggregate_dmc.aggregate_training_seed_reports([report])


def test_eval_seed_rows_must_exactly_match_declared_plan():
    report = _single_seed_report(11, 10.0)
    report["per_eval_seed"][1]["eval_seed"] = report["per_eval_seed"][0]["eval_seed"]
    with pytest.raises(ValueError, match="eval seed plan"):
        aggregate_dmc.aggregate_training_seed_reports([report])


def test_profile_wrapper_binds_configured_training_seeds_and_orders_them():
    config = load_experiment_config(default_config_path(TASK))
    train_seeds = config.raw["seeds"]["train"]
    eval_seeds = config.raw["seeds"]["evaluation"]
    episodes = config.raw["evaluation"]["episodes_per_seed"]
    reports = [
        _single_seed_report(
            seed,
            float(index),
            actor_config=config.actor_config.to_dict(),
            eval_seeds=eval_seeds,
            episodes_per_seed=episodes,
            authorization=_formal_authorization(
                config, profile="benchmark", index=index
            ),
        )
        for index, seed in enumerate(train_seeds)
    ]

    result = aggregate_dmc.aggregate_training_seeds(
        list(reversed(reports)), profile="benchmark"
    )

    assert result["profile"] == "benchmark"
    assert result["training_seeds"] == train_seeds
    assert result["config_fingerprint"] == config.fingerprint
    assert result["training_approved"] is True
    assert result["authorization_verified"] is True
    assert result["authorization_kind"] == "dmc_training_approval_v1"
    assert result["approval_file_sha256"] == "2" * 64
    assert result["preflight_report_sha256"] == "3" * 64
    assert result["train_seed_indices"] == {
        str(seed): index for index, seed in enumerate(train_seeds)
    }
    assert result["resolved_execution_spec"] == resolve_execution_spec(
        config, "benchmark"
    )

    wrong = _single_seed_report(
        train_seeds[1],
        1.0,
        actor_config=config.actor_config.to_dict(),
        eval_seeds=eval_seeds,
        episodes_per_seed=episodes,
        authorization=_formal_authorization(
            config, profile="development", index=1
        ),
    )
    with pytest.raises(ValueError, match="config/profile"):
        aggregate_dmc.aggregate_training_seeds([wrong], profile="development")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unapproved", "training_approved"),
        ("config", "config_fingerprint"),
        ("profile", "approval_profile"),
        ("approval_hash", "approval_file_sha256"),
        ("preflight_hash", "preflight_report_sha256"),
        ("index", "train_seed_index"),
    ],
)
def test_profile_wrapper_fails_closed_on_authorization_drift(
    mutation: str, message: str
):
    config = load_experiment_config(default_config_path(TASK))
    reports = []
    for index, seed in enumerate(config.raw["seeds"]["train"]):
        authorization = _formal_authorization(
            config, profile="benchmark", index=index
        )
        reports.append(
            _single_seed_report(
                seed,
                float(index),
                actor_config=config.actor_config.to_dict(),
                eval_seeds=config.raw["seeds"]["evaluation"],
                episodes_per_seed=config.raw["evaluation"]["episodes_per_seed"],
                authorization=authorization,
            )
        )

    changed = dict(reports[1])
    changed["per_eval_seed"] = [dict(row) for row in reports[1]["per_eval_seed"]]
    authorization = _formal_authorization(config, profile="benchmark", index=1)
    if mutation == "unapproved":
        authorization["training_approved"] = False
    elif mutation == "config":
        authorization["config_fingerprint"] = "sha256:" + "9" * 64
    elif mutation == "profile":
        authorization["approval_profile"] = "development"
    elif mutation == "approval_hash":
        authorization["approval_file_sha256"] = "8" * 64
    elif mutation == "preflight_hash":
        authorization["preflight_report_sha256"] = "7" * 64
    elif mutation == "index":
        authorization["train_seed_index"] = 2
    _replace_authorization(changed, authorization)
    if mutation == "unapproved":
        changed["authorization_verified"] = False
        for row in changed["per_eval_seed"]:
            row["authorization_verified"] = False
    reports[1] = changed

    with pytest.raises((PermissionError, ValueError), match=message):
        aggregate_dmc.aggregate_training_seeds(reports, profile="benchmark")


def test_checkpoint_sources_only_invoke_evaluation_not_training(monkeypatch):
    config = load_experiment_config(default_config_path(TASK))
    reports_by_seed_dir = {
        f"seed_{seed}": _single_seed_report(
            seed,
            float(index),
            actor_config=config.actor_config.to_dict(),
            eval_seeds=config.raw["seeds"]["evaluation"],
            episodes_per_seed=config.raw["evaluation"]["episodes_per_seed"],
            authorization=_formal_authorization(
                config, profile="benchmark", index=index
            ),
        )
        for index, seed in enumerate(config.raw["seeds"]["train"])
    }
    calls: list[str] = []

    def fake_load(path, **_kwargs):
        report = reports_by_seed_dir[Path(path).parent.name]
        return _checkpoint_metadata_from_report(config, report)

    monkeypatch.setattr(aggregate_dmc, "load_actor_checkpoint", fake_load)

    def fake_aggregate(path, **_kwargs):
        calls.append(Path(path).parent.name)
        return reports_by_seed_dir[Path(path).parent.name]

    monkeypatch.setattr(aggregate_dmc, "aggregate_evaluations", fake_aggregate)
    sources = [Path(seed_dir) / "latest.pt" for seed_dir in reports_by_seed_dir]
    result = aggregate_dmc.aggregate_training_seeds(
        sources, profile="benchmark", device_name="cpu"
    )

    assert sorted(calls) == sorted(reports_by_seed_dir)
    assert result["training_seed_count"] == 3
    assert [item["train_seed_index"] for item in result["sources"]] == [0, 1, 2]


def test_formal_wrapper_rejects_posthoc_best_checkpoint_before_evaluation(
    monkeypatch,
):
    config = load_experiment_config(default_config_path(TASK))
    report = _single_seed_report(
        config.raw["seeds"]["train"][0],
        1.0,
        actor_config=config.actor_config.to_dict(),
        eval_seeds=config.raw["seeds"]["evaluation"],
        episodes_per_seed=config.raw["evaluation"]["episodes_per_seed"],
        authorization=_formal_authorization(
            config, profile="development", index=0
        ),
    )
    metadata = _checkpoint_metadata_from_report(config, report)
    evaluation_calls = 0

    def must_not_evaluate(*_args, **_kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        raise AssertionError("checkpoint selection must precede evaluation")

    monkeypatch.setattr(
        aggregate_dmc, "load_actor_checkpoint", lambda *_args, **_kwargs: metadata
    )
    monkeypatch.setattr(aggregate_dmc, "aggregate_evaluations", must_not_evaluate)

    with pytest.raises(ValueError, match="latest.pt"):
        aggregate_dmc.aggregate_training_seeds(
            [Path("best.pt")], profile="development", device_name="cpu"
        )
    assert evaluation_calls == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("authorization_kind", "unverified training authorization"),
        ("environment_protocol_json", "environment_protocol_json"),
        ("resolved_execution_spec", "resolved_execution_spec"),
        ("evaluation_seeds", "evaluation_seeds"),
        ("episodes", "evaluation_episodes_per_seed"),
    ],
)
def test_checkpoint_lineage_fails_closed_before_environment_evaluation(
    monkeypatch, mutation: str, message: str
):
    config = load_experiment_config(default_config_path(TASK))
    seed = config.raw["seeds"]["train"][0]
    report = _single_seed_report(
        seed,
        1.0,
        actor_config=config.actor_config.to_dict(),
        eval_seeds=config.raw["seeds"]["evaluation"],
        episodes_per_seed=config.raw["evaluation"]["episodes_per_seed"],
        authorization=_formal_authorization(
            config, profile="development", index=0
        ),
    )
    if mutation == "authorization_kind":
        report["authorization_kind"] = "private_test_only"
    metadata = _checkpoint_metadata_from_report(config, report)
    if mutation == "environment_protocol_json":
        metadata.payload["environment_protocol_json"] = "{}"
    elif mutation == "resolved_execution_spec":
        metadata.payload["resolved_execution_spec"]["profile"] = "benchmark"
    elif mutation == "evaluation_seeds":
        metadata.payload["evaluation_seeds"] = list(
            reversed(metadata.payload["evaluation_seeds"])
        )
    elif mutation == "episodes":
        metadata.payload["evaluation_episodes_per_seed"] -= 1

    evaluation_calls = 0

    def must_not_evaluate(*_args, **_kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        raise AssertionError("formal validation must precede evaluation")

    monkeypatch.setattr(
        aggregate_dmc, "load_actor_checkpoint", lambda *_args, **_kwargs: metadata
    )
    monkeypatch.setattr(aggregate_dmc, "aggregate_evaluations", must_not_evaluate)

    with pytest.raises((PermissionError, ValueError), match=message):
        aggregate_dmc.aggregate_training_seeds(
            [Path("latest.pt")], profile="development", device_name="cpu"
        )
    assert evaluation_calls == 0


@pytest.mark.parametrize("mutation", ["evaluation_seeds", "episodes"])
def test_single_seed_aggregate_cannot_override_verified_checkpoint_plan(
    monkeypatch, mutation: str
):
    config = load_experiment_config(default_config_path(TASK))
    seed = config.raw["seeds"]["train"][0]
    report = _single_seed_report(
        seed,
        1.0,
        actor_config=config.actor_config.to_dict(),
        eval_seeds=config.raw["seeds"]["evaluation"],
        episodes_per_seed=config.raw["evaluation"]["episodes_per_seed"],
        authorization=_formal_authorization(
            config, profile="development", index=0
        ),
    )
    metadata = _checkpoint_metadata_from_report(config, report)
    requested_seeds = list(config.raw["seeds"]["evaluation"])
    requested_episodes = int(config.raw["evaluation"]["episodes_per_seed"])
    if mutation == "evaluation_seeds":
        requested_seeds.reverse()
    else:
        requested_episodes -= 1
    evaluation_calls = 0

    def must_not_evaluate(*_args, **_kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        raise AssertionError("plan validation must precede evaluation")

    monkeypatch.setattr(
        aggregate_dmc, "load_actor_checkpoint", lambda *_args, **_kwargs: metadata
    )
    monkeypatch.setattr(aggregate_dmc, "evaluate", must_not_evaluate)

    with pytest.raises(ValueError, match=mutation):
        aggregate_dmc.aggregate_evaluations(
            Path("formal.pt"),
            eval_seeds=requested_seeds,
            episodes_per_seed=requested_episodes,
            device_name="cpu",
        )
    assert evaluation_calls == 0


def test_cli_reads_existing_report_and_writes_result_without_environment(
    tmp_path: Path,
):
    config = load_experiment_config(default_config_path(TASK))
    report = _single_seed_report(
        config.raw["seeds"]["train"][0],
        10.0,
        actor_config=config.actor_config.to_dict(),
        eval_seeds=config.raw["seeds"]["evaluation"],
        episodes_per_seed=config.raw["evaluation"]["episodes_per_seed"],
        authorization=_formal_authorization(
            config, profile="development", index=0
        ),
    )
    source = tmp_path / "single.json"
    output = tmp_path / "aggregate.json"
    source.write_text(json.dumps(report), encoding="utf-8")

    aggregate_dmc.main(
        [
            "--aggregate-report",
            str(source),
            "--profile",
            "development",
            "--output",
            str(output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["kind"] == "dmc_training_seed_aggregate"
    assert result["training_seed_statistics"]["inference_estimable"] is False
