"""Aggregate O2O learning curves with training seed as the inference axis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.dmc.o2o.evaluate import (
    EVALUATION_EPISODES,
    EVALUATION_KIND,
    EVALUATION_SEED_BASE,
    validate_run_identity,
)
from experiments.dmc.o2o.config import METHODS, METHOD_SPECS
from experiments.dmc.o2o.koopman import file_sha256


AGGREGATE_KIND = "acmpc_dmc_o2o_aggregate_v1"
MATRIX_MANIFEST_KIND = "acmpc_dmc_o2o_matrix_manifest_v1"
MAX_EPISODE_RETURN = 1000.0
RAW_METHODS = frozenset(
    method for method, spec in METHOD_SPECS.items() if not spec.requires_koopman
)
STRUCTURED_METHODS = frozenset(
    method for method, spec in METHOD_SPECS.items() if spec.requires_koopman
)
EARLY_AUC_STEP = 25_000


def _json_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _method_config_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the method recipe identity shared by all training seeds."""

    identity = dict(config)
    identity.pop("seed", None)
    return identity


# Two-sided 95% Student-t critical values.  Formal experiments use either
# three development seeds (df=2) or ten final seeds (df=9); the full table
# keeps partial and diagnostic aggregates honest as well.
_T95 = (
    12.7062047364,
    4.3026527299,
    3.1824463053,
    2.7764451052,
    2.5705818356,
    2.4469118488,
    2.3646242510,
    2.3060041352,
    2.2621571629,
    2.2281388520,
    2.2009851601,
    2.1788128297,
    2.1603686565,
    2.1447866879,
    2.1314495456,
    2.1199052992,
    2.1098155778,
    2.1009220402,
    2.0930240544,
    2.0859634473,
    2.0796138447,
    2.0738730679,
    2.0686576104,
    2.0638985616,
    2.0595385528,
    2.0555294386,
    2.0518305165,
    2.0484071418,
    2.0452296421,
    2.0422724563,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required result file does not exist: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Required metrics file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Metrics row {path}:{line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")
    return rows


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _validate_eval_row(row: Mapping[str, Any], *, field: str) -> tuple[int, float]:
    step = _nonnegative_int(row.get("online_step"), field=f"{field}.online_step")
    value = _finite_float(row.get("return_mean"), field=f"{field}.return_mean")
    if not -1e-5 <= value <= MAX_EPISODE_RETURN + 1e-5:
        raise ValueError(f"{field}.return_mean lies outside [0, 1000]")
    returns = row.get("returns")
    if not isinstance(returns, list) or len(returns) != EVALUATION_EPISODES:
        raise ValueError(f"{field}.returns must contain ten episodes")
    values = np.asarray(returns, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{field}.returns contain NaN or Inf")
    if not np.isclose(values.mean(), value, rtol=0.0, atol=1e-8):
        raise ValueError(f"{field}.return_mean disagrees with per-episode returns")
    return step, value


def _evaluation_curve(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    offline_zero: list[dict[str, Any]] = []
    initial_zero: list[dict[str, Any]] = []
    online: dict[int, dict[str, Any]] = {}
    for row in rows:
        phase = row.get("phase")
        step = row.get("online_step")
        if phase == "offline_evaluation" and step == 0:
            offline_zero.append(row)
        elif phase == "initial" and step == 0:
            initial_zero.append(row)
        elif phase == "online_evaluation":
            parsed_step = _nonnegative_int(step, field="online_evaluation.online_step")
            if parsed_step == 0:
                raise ValueError("online_evaluation rows must have a positive step")
            # A resumed run may safely repeat the same fixed-seed evaluation;
            # the latest durable row is authoritative.
            online[parsed_step] = row
    zero_candidates = offline_zero or initial_zero
    if not zero_candidates:
        raise ValueError("No step-zero offline/initial evaluation exists")
    zero_step, zero_return = _validate_eval_row(
        zero_candidates[-1], field="step_zero_evaluation"
    )
    if zero_step != 0:
        raise AssertionError("step-zero row selection drifted")
    curve: list[dict[str, float | int]] = [
        {"online_step": 0, "return_mean": zero_return}
    ]
    for step in sorted(online):
        parsed_step, value = _validate_eval_row(
            online[step], field=f"online_evaluation[{step}]"
        )
        curve.append({"online_step": parsed_step, "return_mean": value})
    if len(curve) < 2:
        raise ValueError("No online evaluation points exist")
    return curve


def formal_evaluation_grid(online_budget: int) -> tuple[int, ...]:
    """Canonical common grid for a 50k run or a matrix-wide extension."""

    if online_budget < 5_000 or online_budget % 5_000:
        raise ValueError("Formal online budget must be a multiple of 5k and at least 5k")
    return (0, 1_000, 2_500, *range(5_000, online_budget + 1, 5_000))


def _training_curve(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    curve: list[dict[str, float | int]] = []
    previous = -1
    for row in rows:
        if row.get("phase") != "online_episode":
            continue
        step = _nonnegative_int(row.get("online_step"), field="online_episode.online_step")
        value = _finite_float(
            row.get("episode_return"), field="online_episode.episode_return"
        )
        if step <= previous:
            raise ValueError("Online episode rows are not strictly increasing")
        if not -1e-5 <= value <= MAX_EPISODE_RETURN + 1e-5:
            raise ValueError("Training episode return lies outside [0, 1000]")
        curve.append({"online_step": step, "return": value})
        previous = step
    return curve


def curve_metrics(
    curve: list[Mapping[str, float | int]],
    *,
    target_step: int | None = None,
    require_target: bool = True,
) -> dict[str, Any]:
    """Trapezoidal AUC and regret for a step-zero-inclusive eval curve."""

    steps = np.asarray([int(point["online_step"]) for point in curve], dtype=np.int64)
    values = np.asarray([float(point["return_mean"]) for point in curve], dtype=np.float64)
    if len(steps) < 2 or steps[0] != 0 or np.any(np.diff(steps) <= 0):
        raise ValueError("Evaluation curve must start at zero and increase strictly")
    if not np.isfinite(values).all() or np.any(values < -1e-5) or np.any(
        values > MAX_EPISODE_RETURN + 1e-5
    ):
        raise ValueError("Evaluation curve returns must be finite and in [0, 1000]")
    if target_step is None:
        target_step = int(steps[-1])
    if target_step < 1:
        raise ValueError("target_step must be positive")
    complete = bool(steps[-1] >= target_step)
    if require_target and (not complete or target_step not in set(steps.tolist())):
        raise ValueError(f"Evaluation curve has no exact step {target_step} point")
    endpoint = target_step if complete else int(steps[-1])
    keep = steps < endpoint
    integration_steps = steps[keep].tolist()
    integration_values = values[keep].tolist()
    endpoint_value = float(np.interp(endpoint, steps, values))
    integration_steps.append(endpoint)
    integration_values.append(endpoint_value)
    x = np.asarray(integration_steps, dtype=np.float64)
    y = np.asarray(integration_values, dtype=np.float64)
    auc = float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5))
    regret = float(MAX_EPISODE_RETURN * endpoint - auc)
    return {
        "integration_end_step": endpoint,
        "target_online_step": target_step,
        "complete_to_budget": complete,
        "step0_return": float(values[0]),
        "auc_return_steps": auc,
        "auc_over_1000": auc / MAX_EPISODE_RETURN,
        "normalized_auc": auc / (MAX_EPISODE_RETURN * endpoint),
        "return_at_budget": endpoint_value if complete else None,
        "final_online_step": int(steps[-1]),
        "final_return": float(values[-1]),
        "cumulative_regret_return_steps": regret,
        "cumulative_regret_over_1000": regret / MAX_EPISODE_RETURN,
    }


def _checkpoint_evaluation(run_dir: Path, validated: Any) -> dict[str, Any] | None:
    path = run_dir / f"evaluation_latest_{EVALUATION_EPISODES}.json"
    if not path.is_file():
        return None
    report = _read_json(path)
    expected = {
        "kind": EVALUATION_KIND,
        "method": validated.config.method,
        "training_seed": validated.config.seed,
        "checkpoint_name": "latest",
        "checkpoint_sha256": validated.checkpoint_sha256,
        "online_step": int(validated.checkpoint["online_step"]),
        "config_fingerprint": validated.config.fingerprint,
        "environment_protocol": validated.checkpoint["environment_protocol"],
        "initialization": validated.checkpoint.get("initialization"),
        "raw_observation_normalizer": (
            validated.observation_normalizer.identity()
            if validated.observation_normalizer is not None
            else None
        ),
    }
    mismatches = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint evaluation identity mismatch: {mismatches}")
    if report.get("dataset", {}).get("sha256") != validated.dataset_sha256:
        raise ValueError("Checkpoint evaluation dataset identity differs")
    report_koopman = report.get("koopman")
    if validated.koopman_sha256 is None:
        if report_koopman is not None:
            raise ValueError("Raw checkpoint evaluation unexpectedly contains Koopman")
    elif (
        not isinstance(report_koopman, Mapping)
        or report_koopman.get("sha256") != validated.koopman_sha256
    ):
        raise ValueError("Checkpoint evaluation Koopman identity differs")
    protocol = report.get("evaluation_protocol")
    if not isinstance(protocol, dict) or protocol.get("episodes") != EVALUATION_EPISODES:
        raise ValueError("Checkpoint evaluation does not use ten episodes")
    if protocol.get("seed_base") != EVALUATION_SEED_BASE or protocol.get(
        "deterministic"
    ) is not True:
        raise ValueError("Checkpoint evaluation protocol is not canonical")
    returns = report.get("returns")
    if not isinstance(returns, list) or len(returns) != EVALUATION_EPISODES:
        raise ValueError("Checkpoint evaluation return vector is invalid")
    value = _finite_float(report.get("return_mean"), field="checkpoint return_mean")
    if not np.isclose(np.mean(returns), value, rtol=0.0, atol=1e-8):
        raise ValueError("Checkpoint evaluation mean disagrees with returns")
    return {
        "path": str(path.resolve()),
        "return_mean": value,
        "return_std_population": _finite_float(
            report.get("return_std_population"), field="checkpoint return std"
        ),
    }


def _run_result(run_dir: Path, *, require_complete: bool) -> dict[str, Any]:
    validated = validate_run_identity(
        run_dir, checkpoint_name="latest", load_artifacts=False
    )
    if validated.config.eval_episodes != EVALUATION_EPISODES:
        raise ValueError("Formal aggregation requires ten evaluation episodes")
    rows = _read_jsonl(run_dir / "metrics.jsonl")
    evaluation_curve = _evaluation_curve(rows)
    training_curve = _training_curve(rows)
    checkpoint_step = int(validated.checkpoint["online_step"])
    configured_budget = int(validated.config.online_steps)
    if int(evaluation_curve[-1]["online_step"]) != checkpoint_step:
        raise ValueError("Latest checkpoint and evaluation curve end at different steps")
    if require_complete:
        if validated.run_metadata.get("completed") is not True:
            raise ValueError(f"Run is not complete: {run_dir}")
        if checkpoint_step != configured_budget:
            raise ValueError("Latest checkpoint is not the configured final step")
        actual_grid = tuple(
            int(point["online_step"]) for point in evaluation_curve
        )
        expected_grid = formal_evaluation_grid(configured_budget)
        if actual_grid != expected_grid:
            raise ValueError(
                f"Formal evaluation grid differs: {actual_grid} != {expected_grid}"
            )
    metrics = curve_metrics(
        evaluation_curve,
        target_step=configured_budget,
        require_target=require_complete,
    )
    early_metrics = (
        curve_metrics(
            evaluation_curve,
            target_step=EARLY_AUC_STEP,
            require_target=require_complete,
        )
        if configured_budget >= EARLY_AUC_STEP
        else None
    )
    checkpoint_evaluation = _checkpoint_evaluation(run_dir, validated)
    return {
        "run_dir": str(run_dir.resolve()),
        "method": validated.config.method,
        "training_seed": validated.config.seed,
        "config": validated.config.to_dict(),
        "config_fingerprint": validated.config.fingerprint,
        "dataset_sha256": validated.dataset_sha256,
        "koopman_sha256": validated.koopman_sha256,
        "environment_protocol": validated.checkpoint["environment_protocol"],
        "initialization": validated.checkpoint.get("initialization"),
        "latest_checkpoint_sha256": validated.checkpoint_sha256,
        "evaluation_curve": evaluation_curve,
        "training_curve": training_curve,
        "curve_metrics": metrics,
        "early_25k_curve_metrics": early_metrics,
        "checkpoint_evaluation": checkpoint_evaluation,
    }


def _statistics(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    ordered = sorted((int(seed), float(value)) for seed, value in values_by_seed.items())
    values = np.asarray([value for _seed, value in ordered], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Training-seed statistic requires finite values")
    sample_std = float(values.std(ddof=1)) if len(values) > 1 else None
    sem = sample_std / math.sqrt(len(values)) if sample_std is not None else None
    if sem is None:
        critical = None
        distribution = "not_estimable_single_training_seed"
    else:
        degrees_of_freedom = len(values) - 1
        critical = (
            _T95[degrees_of_freedom - 1]
            if degrees_of_freedom <= len(_T95)
            else 1.9599639845
        )
        distribution = (
            f"student_t_df_{degrees_of_freedom}"
            if degrees_of_freedom <= len(_T95)
            else "asymptotic_normal_df_gt_30"
        )
    return {
        "inference_axis": "training_seed",
        "n_training_seeds": int(len(values)),
        "mean": float(values.mean()),
        "sample_std": sample_std,
        "sem": sem,
        "ci95_half_width": critical * sem if sem is not None else None,
        "ci95_distribution": distribution,
        "values_by_training_seed": {str(seed): value for seed, value in ordered},
    }


def _aggregate_point_curves(
    runs: list[dict[str, Any]], *, curve_key: str, value_key: str
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[int, float]] = defaultdict(dict)
    for run in runs:
        seed = int(run["training_seed"])
        for point in run[curve_key]:
            step = int(point["online_step"])
            if seed in by_step[step]:
                raise ValueError("A training seed repeats a curve step")
            by_step[step][seed] = float(point[value_key])
    return [
        {"online_step": step, "return": _statistics(by_step[step])}
        for step in sorted(by_step)
    ]


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _verify_source_snapshot(
    snapshot: Any, *, repo_root: Path, field: str
) -> dict[str, Any]:
    source = _require_mapping(snapshot, field=field)
    entries = source.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{field}.files must be a non-empty list")
    verified: list[dict[str, str]] = []
    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(raw_entry, field=f"{field}.files[{index}]")
        relative = entry.get("path")
        checksum = entry.get("sha256")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"{field}.files[{index}].path must be relative")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"{field}.files[{index}].sha256 is invalid")
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"{field}.files[{index}].path escapes repository") from exc
        if not path.is_file() or file_sha256(path) != checksum:
            raise ValueError(f"{field} source file differs: {relative}")
        verified.append({"path": relative, "sha256": checksum})
    if len({entry["path"] for entry in verified}) != len(verified):
        raise ValueError(f"{field}.files repeats a source path")
    expected_bundle = source.get("sha256")
    actual_bundle = _json_fingerprint({"files": verified})
    if expected_bundle != actual_bundle:
        raise ValueError(f"{field}.sha256 does not match its file identities")
    return dict(source)


def _resolve_manifest_output(value: Any, *, repo_root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a path string")
    path = Path(value)
    return (path if path.is_absolute() else repo_root / path).resolve()


def _matrix_authorization(
    manifest_path: Path | None,
    *,
    runs: list[dict[str, Any]],
    require_complete: bool,
    dataset_sha256: str,
    structured_koopman_sha256: str | None,
    training_seeds: tuple[int, ...],
    evaluation_grid: tuple[int, ...],
    online_budget: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Authorize an aggregate against the runner's immutable source manifest."""

    if manifest_path is None:
        return (
            {
                "kind": "acmpc_dmc_o2o_aggregate_authorization_v1",
                "source_verified": False,
                "formal_complete": False,
                "matrix_manifest": None,
                "reason": "matrix_manifest_not_provided",
            },
            None,
        )

    resolved_manifest = manifest_path.expanduser().resolve()
    manifest = _read_json(resolved_manifest)
    if manifest.get("kind") != MATRIX_MANIFEST_KIND:
        raise ValueError("Unsupported O2O matrix manifest kind")
    fingerprint = manifest.get("matrix_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Matrix manifest fingerprint is invalid")
    experiment = _require_mapping(
        manifest.get("experiment"), field="matrix_manifest.experiment"
    )
    if experiment.get("methods") != list(METHODS):
        raise ValueError("Matrix manifest method set/order differs")
    if experiment.get("seeds") != list(training_seeds):
        raise ValueError("Matrix manifest training seeds differ")
    manifest_budget = experiment.get("extend_online_steps") or experiment.get(
        "online_steps"
    )
    if manifest_budget != online_budget:
        raise ValueError("Matrix manifest online budget differs")
    if experiment.get("evaluation_grid_online_steps") != list(evaluation_grid):
        raise ValueError("Matrix manifest evaluation grid differs")
    if experiment.get("dataset", {}).get("sha256") != dataset_sha256:
        raise ValueError("Matrix manifest dataset identity differs")
    manifest_koopman = experiment.get("koopman")
    if not isinstance(manifest_koopman, Mapping):
        raise ValueError("Matrix manifest Koopman identity is missing")
    if (
        structured_koopman_sha256 is not None
        and manifest_koopman.get("sha256") != structured_koopman_sha256
    ):
        raise ValueError("Matrix manifest Koopman identity differs")

    resolved_configs = _require_mapping(
        experiment.get("resolved_method_configs"),
        field="matrix_manifest.experiment.resolved_method_configs",
    )
    if set(resolved_configs) != set(METHODS):
        raise ValueError("Matrix manifest resolved method configs differ")
    for run in runs:
        method = str(run["method"])
        manifest_config = _require_mapping(
            resolved_configs.get(method),
            field=f"matrix_manifest.resolved_method_configs[{method}]",
        )
        if _method_config_identity(manifest_config) != _method_config_identity(
            run["config"]
        ):
            raise ValueError(
                f"Run {method} config is not authorized by the matrix manifest"
            )

    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Matrix manifest jobs must be a list")
    train_jobs = [
        job
        for job in jobs
        if isinstance(job, Mapping) and job.get("stage") == "train"
    ]
    expected_pairs = {
        (str(run["method"]), int(run["training_seed"])): Path(run["run_dir"])
        for run in runs
    }
    if len(expected_pairs) != len(runs):
        raise ValueError("Aggregate repeats a method/training-seed pair")
    actual_pairs: dict[tuple[str, int], Path] = {}
    repo_roots: set[Path] = set()
    for index, job in enumerate(train_jobs):
        method = job.get("method")
        seed = job.get("seed")
        cwd = job.get("cwd")
        outputs = job.get("outputs")
        if (
            not isinstance(method, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(cwd, str)
            or not isinstance(outputs, list)
            or len(outputs) != 1
        ):
            raise ValueError(f"Matrix train job {index} identity is invalid")
        repo_root = Path(cwd).expanduser().resolve()
        repo_roots.add(repo_root)
        pair = (method, seed)
        if pair in actual_pairs:
            raise ValueError("Matrix manifest repeats a train method/seed pair")
        latest = _resolve_manifest_output(
            outputs[0], repo_root=repo_root, field=f"matrix jobs[{index}].outputs[0]"
        )
        if latest.name != "latest.pt":
            raise ValueError("Matrix train output is not latest.pt")
        actual_pairs[pair] = latest.parent
    if actual_pairs != expected_pairs:
        raise ValueError("Aggregate run directories are not the manifest train jobs")
    if len(repo_roots) != 1:
        raise ValueError("Matrix train jobs do not share one repository root")
    repo_root = next(iter(repo_roots))

    source_identity = _require_mapping(
        manifest.get("source_identity"), field="matrix_manifest.source_identity"
    )
    training_source = _verify_source_snapshot(
        source_identity.get("training_source"),
        repo_root=repo_root,
        field="matrix_manifest.source_identity.training_source",
    )
    result_source = _verify_source_snapshot(
        source_identity.get("result_source"),
        repo_root=repo_root,
        field="matrix_manifest.source_identity.result_source",
    )
    runner = _require_mapping(
        source_identity.get("runner"), field="matrix_manifest.source_identity.runner"
    )
    runner_path = _resolve_manifest_output(
        runner.get("path"), repo_root=repo_root, field="matrix runner.path"
    )
    if runner.get("sha256") != file_sha256(runner_path):
        raise ValueError("Matrix runner source differs")

    manifest_identity = {
        "path": str(resolved_manifest),
        "sha256": file_sha256(resolved_manifest),
        "matrix_fingerprint": fingerprint,
        "training_source": training_source,
        "result_source": result_source,
        "runner": dict(runner),
        "git": source_identity.get("git"),
    }
    return (
        {
            "kind": "acmpc_dmc_o2o_aggregate_authorization_v1",
            "source_verified": True,
            "formal_complete": bool(require_complete),
            "matrix_manifest": {
                "path": str(resolved_manifest),
                "sha256": manifest_identity["sha256"],
                "matrix_fingerprint": fingerprint,
            },
            "reason": None,
        },
        manifest_identity,
    )


def aggregate_runs(
    run_dirs: Iterable[Path],
    *,
    require_complete: bool = True,
    matrix_manifest: Path | None = None,
) -> dict[str, Any]:
    resolved = [Path(path).resolve() for path in run_dirs]
    if not resolved:
        raise ValueError("At least one run directory is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Run directories must be unique")
    runs = [_run_result(path, require_complete=require_complete) for path in resolved]

    dataset_hashes = {run["dataset_sha256"] for run in runs}
    structured_koopman_hashes = {
        run["koopman_sha256"]
        for run in runs
        if run["koopman_sha256"] is not None
    }
    protocols = {
        json.dumps(run["environment_protocol"], sort_keys=True) for run in runs
    }
    if len(dataset_hashes) != 1 or len(protocols) != 1:
        raise ValueError("Compared runs do not share dataset/DMC protocol")
    if len(structured_koopman_hashes) > 1:
        raise ValueError("Structured runs do not share one frozen Koopman model")

    # Algorithm-specific parameters are intentionally allowed to differ: the
    # raw baselines follow their own official/official-style recipes, while
    # the structured methods use AC-KMPC.  Only experiment-budget and
    # evaluation fields are cross-method invariants.
    shared_protocol_configs = []
    for run in runs:
        config = dict(run["config"])
        shared_protocol_configs.append(
            {
                key: config[key]
                for key in (
                    "task",
                    "online_steps",
                    "eval_episodes",
                    "eval_interval_online_steps",
                )
            }
        )
    if any(
        config != shared_protocol_configs[0]
        for config in shared_protocol_configs[1:]
    ):
        raise ValueError("Compared runs differ in shared online/evaluation protocol")
    primary_online_budget = int(shared_protocol_configs[0]["online_steps"])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["method"])].append(run)
    if require_complete and set(grouped) != set(METHODS):
        raise ValueError(
            "Formal aggregation requires exactly the complete five-method matrix"
        )
    for run in runs:
        method = str(run["method"])
        koopman_sha = run["koopman_sha256"]
        if method in RAW_METHODS and koopman_sha is not None:
            raise ValueError(f"Raw baseline {method} unexpectedly contains Koopman")
        if method in STRUCTURED_METHODS and koopman_sha is None:
            raise ValueError(f"Structured method {method} is missing Koopman")
    methods: dict[str, Any] = {}
    common_eval_grid: tuple[int, ...] | None = None
    common_training_seeds: tuple[int, ...] | None = None
    for method, method_runs in sorted(grouped.items()):
        method_runs.sort(key=lambda value: int(value["training_seed"]))
        method_config_identities = [
            _method_config_identity(run["config"]) for run in method_runs
        ]
        if any(
            identity != method_config_identities[0]
            for identity in method_config_identities[1:]
        ):
            raise ValueError(
                f"Method {method} training seeds use different method-specific configs"
            )
        seeds = [int(run["training_seed"]) for run in method_runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Method {method} repeats a training seed")
        seed_tuple = tuple(seeds)
        if common_training_seeds is None:
            common_training_seeds = seed_tuple
        elif seed_tuple != common_training_seeds:
            raise ValueError("Methods do not use the same ordered training-seed set")
        grids = [
            tuple(int(point["online_step"]) for point in run["evaluation_curve"])
            for run in method_runs
        ]
        if any(grid != grids[0] for grid in grids[1:]):
            raise ValueError(f"Method {method} training seeds use different eval grids")
        if common_eval_grid is None:
            common_eval_grid = grids[0]
        elif grids[0] != common_eval_grid:
            raise ValueError("Methods use different deterministic evaluation grids")

        metric_names = (
            "step0_return",
            "auc_return_steps",
            "auc_over_1000",
            "normalized_auc",
            "final_return",
            "cumulative_regret_return_steps",
            "cumulative_regret_over_1000",
        )
        metric_summary = {
            name: _statistics(
                {
                    int(run["training_seed"]): float(run["curve_metrics"][name])
                    for run in method_runs
                }
            )
            for name in metric_names
        }
        target_values = {
            int(run["training_seed"]): run["curve_metrics"]["return_at_budget"]
            for run in method_runs
        }
        metric_summary["return_at_budget"] = (
            _statistics({seed: float(value) for seed, value in target_values.items()})
            if all(value is not None for value in target_values.values())
            else None
        )
        early_runs = {
            int(run["training_seed"]): run["early_25k_curve_metrics"]
            for run in method_runs
        }
        if all(value is not None for value in early_runs.values()):
            metric_summary["return_at_25k"] = _statistics(
                {
                    seed: float(value["return_at_budget"])
                    for seed, value in early_runs.items()
                }
            )
            metric_summary["normalized_auc_at_25k"] = _statistics(
                {
                    seed: float(value["normalized_auc"])
                    for seed, value in early_runs.items()
                }
            )
        else:
            metric_summary["return_at_25k"] = None
            metric_summary["normalized_auc_at_25k"] = None
        checkpoint_values = {
            int(run["training_seed"]): run["checkpoint_evaluation"]["return_mean"]
            for run in method_runs
            if run["checkpoint_evaluation"] is not None
        }
        methods[method] = {
            "training_seeds": seeds,
            "n_training_seeds": len(seeds),
            "inference_axis": "training_seed",
            "method_config_identity_sha256": _json_fingerprint(
                method_config_identities[0]
            ),
            "method_config_without_seed": method_config_identities[0],
            "metrics": metric_summary,
            "checkpoint_latest_return": (
                _statistics(checkpoint_values)
                if len(checkpoint_values) == len(method_runs)
                else None
            ),
            "evaluation_curve": _aggregate_point_curves(
                method_runs, curve_key="evaluation_curve", value_key="return_mean"
            ),
            "training_curve": _aggregate_point_curves(
                method_runs, curve_key="training_curve", value_key="return"
            ),
            "per_seed": method_runs,
        }

    # Validate pairing after the common seed-set check so a malformed matrix
    # reports its primary design error before inspecting method-specific files.
    if require_complete:
        source_runs = {
            int(run["training_seed"]): run
            for run in grouped["Cal-RLPD-AC-KMPC"]
        }
        for mpve_run in grouped["Cal-RLPD-AC-KMPC-MPVE"]:
            seed = int(mpve_run["training_seed"])
            source_run = source_runs.get(seed)
            if source_run is None:
                raise ValueError("MPVE has no paired same-seed AC-KMPC run")
            source_path = (Path(source_run["run_dir"]) / "offline.pt").resolve()
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Paired AC-KMPC offline snapshot is missing: {source_path}"
                )
            initialization = mpve_run.get("initialization")
            if not isinstance(initialization, Mapping):
                raise ValueError("MPVE run is missing offline-fork lineage")
            if (
                initialization.get("source_path") != str(source_path)
                or initialization.get("source_sha256") != file_sha256(source_path)
            ):
                raise ValueError(
                    "MPVE lineage does not match the included same-seed AC-KMPC run"
                )

    if common_eval_grid is None or common_training_seeds is None:
        raise AssertionError("Aggregate common protocol identity was not resolved")
    structured_koopman_sha256 = (
        next(iter(structured_koopman_hashes))
        if structured_koopman_hashes
        else None
    )
    authorization, manifest_identity = _matrix_authorization(
        matrix_manifest,
        runs=runs,
        require_complete=require_complete,
        dataset_sha256=next(iter(dataset_hashes)),
        structured_koopman_sha256=structured_koopman_sha256,
        training_seeds=common_training_seeds,
        evaluation_grid=common_eval_grid,
        online_budget=primary_online_budget,
    )

    return {
        "kind": AGGREGATE_KIND,
        "formal_complete": authorization["formal_complete"],
        "authorization": authorization,
        "task": "cartpole_swingup",
        "protocol": {
            "inference_axis": "training_seed",
            "deterministic_evaluation_episodes": EVALUATION_EPISODES,
            "deterministic_evaluation_seed_base": EVALUATION_SEED_BASE,
            "primary_online_budget": primary_online_budget,
            "early_auc_diagnostic_step": EARLY_AUC_STEP,
            "maximum_episode_return": MAX_EPISODE_RETURN,
            "auc_rule": "trapezoidal_step0_and_online_evaluations",
            "auc_over_1000": "integral(return dstep) / 1000",
            "normalized_auc": "integral(return dstep) / (1000 * integration_steps)",
            "cumulative_regret": "integral((1000-return) dstep)",
            "require_complete": require_complete,
            "formal_complete": authorization["formal_complete"],
            "required_methods": list(METHODS) if require_complete else None,
            "common_training_seeds": list(common_training_seeds or ()),
        },
        "shared_identity": {
            "dataset_sha256": next(iter(dataset_hashes)),
            "structured_koopman_sha256": (
                structured_koopman_sha256
            ),
            "environment_protocol": runs[0]["environment_protocol"],
            "shared_online_evaluation_protocol": shared_protocol_configs[0],
            "matrix_source_identity": manifest_identity,
        },
        "methods": methods,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _discover(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("run.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--matrix-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    run_dirs = list(args.run_dir)
    if args.root is not None:
        run_dirs.extend(_discover(args.root))
    result = aggregate_runs(
        run_dirs,
        require_complete=not args.allow_incomplete,
        matrix_manifest=args.matrix_manifest,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
