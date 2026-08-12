"""Aggregate O2O learning curves with training seed as the inference axis."""

from __future__ import annotations

import argparse
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
from experiments.dmc.o2o.config import METHODS
from experiments.dmc.o2o.koopman import file_sha256


AGGREGATE_KIND = "acmpc_dmc_o2o_aggregate_v1"
MAX_EPISODE_RETURN = 1000.0
PRIMARY_ONLINE_BUDGET = 100_000

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
    target_step: int = PRIMARY_ONLINE_BUDGET,
    require_target: bool = True,
) -> dict[str, Any]:
    """Trapezoidal AUC and regret for a step-zero-inclusive eval curve."""

    if target_step < 1:
        raise ValueError("target_step must be positive")
    steps = np.asarray([int(point["online_step"]) for point in curve], dtype=np.int64)
    values = np.asarray([float(point["return_mean"]) for point in curve], dtype=np.float64)
    if len(steps) < 2 or steps[0] != 0 or np.any(np.diff(steps) <= 0):
        raise ValueError("Evaluation curve must start at zero and increase strictly")
    if not np.isfinite(values).all() or np.any(values < -1e-5) or np.any(
        values > MAX_EPISODE_RETURN + 1e-5
    ):
        raise ValueError("Evaluation curve returns must be finite and in [0, 1000]")
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
        "complete_to_100k": complete,
        "step0_return": float(values[0]),
        "auc_return_steps": auc,
        "auc_over_1000": auc / MAX_EPISODE_RETURN,
        "normalized_auc": auc / (MAX_EPISODE_RETURN * endpoint),
        "return_at_100k": endpoint_value if complete else None,
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
    if report.get("koopman", {}).get("sha256") != validated.koopman_sha256:
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
    if int(evaluation_curve[-1]["online_step"]) != checkpoint_step:
        raise ValueError("Latest checkpoint and evaluation curve end at different steps")
    if require_complete:
        if validated.run_metadata.get("completed") is not True:
            raise ValueError(f"Run is not complete: {run_dir}")
        if validated.config.online_steps < PRIMARY_ONLINE_BUDGET:
            raise ValueError("Formal run has an online budget below 100k")
        if checkpoint_step != validated.config.online_steps:
            raise ValueError("Latest checkpoint is not the configured final step")
    metrics = curve_metrics(
        evaluation_curve,
        target_step=PRIMARY_ONLINE_BUDGET,
        require_target=require_complete,
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


def aggregate_runs(
    run_dirs: Iterable[Path], *, require_complete: bool = True
) -> dict[str, Any]:
    resolved = [Path(path).resolve() for path in run_dirs]
    if not resolved:
        raise ValueError("At least one run directory is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Run directories must be unique")
    runs = [_run_result(path, require_complete=require_complete) for path in resolved]

    dataset_hashes = {run["dataset_sha256"] for run in runs}
    koopman_hashes = {run["koopman_sha256"] for run in runs}
    protocols = {
        json.dumps(run["environment_protocol"], sort_keys=True) for run in runs
    }
    if len(dataset_hashes) != 1 or len(koopman_hashes) != 1 or len(protocols) != 1:
        raise ValueError("Compared runs do not share dataset/Koopman/DMC protocol")
    shared_configs = []
    for run in runs:
        config = dict(run["config"])
        config.pop("method")
        config.pop("seed")
        shared_configs.append(config)
    if any(config != shared_configs[0] for config in shared_configs[1:]):
        raise ValueError("Compared runs differ in shared learner configuration")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["method"])].append(run)
    if require_complete and set(grouped) != set(METHODS):
        raise ValueError(
            "Formal aggregation requires exactly the complete five-method matrix"
        )
    methods: dict[str, Any] = {}
    common_eval_grid: tuple[int, ...] | None = None
    common_training_seeds: tuple[int, ...] | None = None
    for method, method_runs in sorted(grouped.items()):
        method_runs.sort(key=lambda value: int(value["training_seed"]))
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
            int(run["training_seed"]): run["curve_metrics"]["return_at_100k"]
            for run in method_runs
        }
        metric_summary["return_at_100k"] = (
            _statistics({seed: float(value) for seed, value in target_values.items()})
            if all(value is not None for value in target_values.values())
            else None
        )
        checkpoint_values = {
            int(run["training_seed"]): run["checkpoint_evaluation"]["return_mean"]
            for run in method_runs
            if run["checkpoint_evaluation"] is not None
        }
        methods[method] = {
            "training_seeds": seeds,
            "n_training_seeds": len(seeds),
            "inference_axis": "training_seed",
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

    return {
        "kind": AGGREGATE_KIND,
        "task": "cartpole_swingup",
        "protocol": {
            "inference_axis": "training_seed",
            "deterministic_evaluation_episodes": EVALUATION_EPISODES,
            "deterministic_evaluation_seed_base": EVALUATION_SEED_BASE,
            "primary_online_budget": PRIMARY_ONLINE_BUDGET,
            "maximum_episode_return": MAX_EPISODE_RETURN,
            "auc_rule": "trapezoidal_step0_and_online_evaluations",
            "auc_over_1000": "integral(return dstep) / 1000",
            "normalized_auc": "integral(return dstep) / (1000 * integration_steps)",
            "cumulative_regret": "integral((1000-return) dstep)",
            "require_complete": require_complete,
            "required_methods": list(METHODS) if require_complete else None,
            "common_training_seeds": list(common_training_seeds or ()),
        },
        "shared_identity": {
            "dataset_sha256": next(iter(dataset_hashes)),
            "koopman_sha256": next(iter(koopman_hashes)),
            "environment_protocol": runs[0]["environment_protocol"],
            "shared_config_excluding_method_and_seed": shared_configs[0],
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    run_dirs = list(args.run_dir)
    if args.root is not None:
        run_dirs.extend(_discover(args.root))
    result = aggregate_runs(run_dirs, require_complete=not args.allow_incomplete)
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
