"""Strictly validate and aggregate the formal Walker Run seed campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dmc.o2o.evaluate_10x10 import FINAL_EVAL_KIND
from experiments.dmc.o2o.formal_walker import (
    FORMAL_METHODS,
    FORMAL_TRAINING_SEEDS,
)
from experiments.dmc.o2o.koopman import file_sha256


AGGREGATE_KIND = "acmpc_walker_formal_training_seed_aggregate_v1"
# Student-t 97.5% two-sided critical values keyed by sample size n (df = n-1).
T_CRITICAL_95_BY_N = {
    3: 4.302652729911275,  # df=2
    4: 3.182446305283708,  # df=3
    5: 2.7764451051977987,  # df=4
}
OFFLINE_POINTS = tuple(range(0, 50_001, 5_000))
ONLINE_POINTS = tuple(range(0, 20_001, 2_500))


def _t_critical_95(n: int) -> float:
    if n not in T_CRITICAL_95_BY_N:
        raise ValueError(f"No t-critical table entry for {n} training seeds")
    return T_CRITICAL_95_BY_N[n]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _diagnostic(run_dir: Path, stage: str, counter: int) -> float:
    stem = f"{stage}_{counter:06d}"
    checkpoint_path = run_dir / f"{stem}.pt"
    payload = _read_json(run_dir / f"evaluation_{stem}.json")
    if not checkpoint_path.is_file():
        raise ValueError(f"Missing milestone checkpoint: {checkpoint_path}")
    expected = {
        "stage": stage,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Milestone metric/checkpoint identity differs: {stem}")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"Milestone evaluation payload is invalid: {stem}")
    returns = evaluation.get("returns")
    if not isinstance(returns, list) or len(returns) != 10:
        raise ValueError(f"Regular diagnostic must contain 10 episodes: {stem}")
    value = float(evaluation.get("return_mean"))
    if not math.isfinite(value):
        raise ValueError(f"Milestone return is non-finite: {stem}")
    return value


def _final_10x10(run_dir: Path, counter: int, seed: int, method: str) -> float:
    checkpoint = f"online_{counter:06d}"
    path = run_dir / f"evaluation_10x10_{checkpoint}.json"
    payload = _read_json(path)
    if (
        payload.get("kind") != FINAL_EVAL_KIND
        or payload.get("training_seed") != seed
        or payload.get("method") != method
        or payload.get("checkpoint_name") != checkpoint
        or payload.get("online_step") != counter
    ):
        raise ValueError(f"Final 10x10 identity differs: {path}")
    protocol = payload.get("evaluation_protocol")
    if not isinstance(protocol, dict) or (
        protocol.get("num_evaluation_seeds"),
        protocol.get("episodes_per_seed"),
        protocol.get("total_episodes"),
        protocol.get("parallel_workers"),
    ) != (10, 10, 100, 10):
        raise ValueError(f"Final evaluation is not parallel 10x10: {path}")
    returns = payload.get("returns")
    if not isinstance(returns, list) or len(returns) != 100:
        raise ValueError(f"Final evaluation lacks 100 returns: {path}")
    checkpoint_path = run_dir / f"{checkpoint}.pt"
    if (
        payload.get("checkpoint_path") != str(checkpoint_path.resolve())
        or payload.get("checkpoint_sha256") != file_sha256(checkpoint_path)
    ):
        raise ValueError(f"Final evaluation checkpoint identity differs: {path}")
    value = float(payload.get("return_mean"))
    if not math.isfinite(value):
        raise ValueError(f"Final return is non-finite: {path}")
    return value


def _training_seed_summary(
    values: list[float], n_seeds: int | None = None
) -> dict[str, Any]:
    # Preserve the original helper contract used by tests and downstream
    # notebooks while allowing partial formal aggregates to pass an explicit
    # expected seed count.
    if n_seeds is None:
        n_seeds = len(values)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_seeds,) or not np.isfinite(array).all():
        raise ValueError(
            f"Formal summary requires all {n_seeds} finite training seeds"
        )
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    sem = std / math.sqrt(len(array))
    half = _t_critical_95(len(array)) * sem
    return {
        "training_seed_values": values,
        "training_seed_count": len(values),
        "mean": mean,
        "sample_std": std,
        "standard_error": sem,
        "ci95_student_t": [mean - half, mean + half],
        "inference_unit": "independent_training_seed",
    }


def aggregate(
    run_root: Path, training_seeds: tuple[int, ...] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_root = run_root.resolve()
    training_seeds = (
        FORMAL_TRAINING_SEEDS
        if training_seeds is None
        else tuple(int(seed) for seed in training_seeds)
    )
    dataset_sha: str | None = None
    koopman_by_seed: dict[int, str] = {}
    curves: dict[str, dict[str, dict[int, list[float]]]] = {
        method: {"offline": {}, "online": {}} for method in FORMAL_METHODS
    }
    final: dict[str, dict[int, list[float]]] = {
        method: {0: [], 20_000: []} for method in FORMAL_METHODS
    }
    rows: list[dict[str, Any]] = []

    for seed in training_seeds:
        seed_dir = run_root / f"seed_{seed}"
        protocol = _read_json(seed_dir / "protocol.json")
        if (
            protocol.get("kind") != "acmpc_walker_formal_single_seed_protocol_v1"
            or protocol.get("training_seed") != seed
            or protocol.get("methods") != list(FORMAL_METHODS)
            or protocol.get("formal_training_seeds") != list(FORMAL_TRAINING_SEEDS)
        ):
            raise ValueError(f"Formal seed protocol differs: {seed_dir}")
        protocol_dataset = protocol.get("dataset")
        protocol_koopman = protocol.get("koopman")
        if not isinstance(protocol_dataset, dict) or not isinstance(protocol_koopman, dict):
            raise ValueError(f"Dataset/Koopman protocol is missing: {seed_dir}")
        current_dataset_sha = protocol_dataset.get("sha256")
        current_koopman_sha = protocol_koopman.get("sha256")
        if dataset_sha is None:
            dataset_sha = current_dataset_sha
        elif current_dataset_sha != dataset_sha:
            raise ValueError("Formal training seeds do not share one dataset SHA")
        if not isinstance(current_koopman_sha, str):
            raise ValueError("Formal seed Koopman SHA is invalid")
        koopman_by_seed[seed] = current_koopman_sha

        for method in FORMAL_METHODS:
            run_dir = seed_dir / method
            run = _read_json(run_dir / "run.json")
            config = run.get("config")
            if not isinstance(config, dict):
                raise ValueError(f"Run config is missing: {run_dir}")
            expected_offline_completed = 0 if method == "RLPD" else 50_000
            if (
                config.get("task") != "walker_run"
                or config.get("method") != method
                or config.get("seed") != seed
                or config.get("offline_updates") != 50_000
                or config.get("online_steps") != 20_000
                or config.get("offline_eval_interval_updates") != 5_000
                or config.get("eval_interval_online_steps") != 2_500
                or run.get("completed") is not True
                or run.get("offline_updates_completed") != expected_offline_completed
                or run.get("online_steps_completed") != 20_000
            ):
                raise ValueError(f"Formal completed-run contract differs: {run_dir}")
            if run.get("method_spec") != protocol["method_specs"][method]:
                raise ValueError(f"Method specification differs: {run_dir}")
            if run.get("dataset", {}).get("sha256") != dataset_sha:
                raise ValueError(f"Run dataset differs: {run_dir}")
            run_koopman = run.get("koopman")
            if method in {"Cal-RLPD-KMPC", "Cal-RLPD-Lift"}:
                if not isinstance(run_koopman, dict) or run_koopman.get(
                    "sha256"
                ) != current_koopman_sha:
                    raise ValueError(f"Seed-paired Koopman differs: {run_dir}")
            elif run_koopman is not None:
                raise ValueError(f"Baseline unexpectedly loads Koopman: {run_dir}")

            offline_points = (0,) if method == "RLPD" else OFFLINE_POINTS
            for counter in offline_points:
                # RLPD's random initialization is represented by online step 0.
                stage = "online" if method == "RLPD" else "offline"
                value = _diagnostic(run_dir, stage, counter)
                curves[method]["offline"].setdefault(counter, []).append(value)
                rows.append(
                    {
                        "training_seed": seed,
                        "method": method,
                        "stage": "offline",
                        "counter": counter,
                        "diagnostic_return_mean": value,
                    }
                )
            for counter in ONLINE_POINTS:
                value = _diagnostic(run_dir, "online", counter)
                curves[method]["online"].setdefault(counter, []).append(value)
                rows.append(
                    {
                        "training_seed": seed,
                        "method": method,
                        "stage": "online",
                        "counter": counter,
                        "diagnostic_return_mean": value,
                    }
                )
            for counter in (0, 20_000):
                final[method][counter].append(
                    _final_10x10(run_dir, counter, seed, method)
                )

    if len(set(koopman_by_seed.values())) != len(training_seeds):
        raise ValueError("Each formal training seed must have its own Koopman artifact")
    aggregate_curves = {
        method: {
            stage: {
                str(counter): _training_seed_summary(values, len(training_seeds))
                for counter, values in sorted(points.items())
            }
            for stage, points in stages.items()
        }
        for method, stages in curves.items()
    }
    result = {
        "kind": AGGREGATE_KIND,
        "task": "walker_run",
        "methods": list(FORMAL_METHODS),
        "training_seeds": list(training_seeds),
        "dataset_sha256": dataset_sha,
        "koopman_sha256_by_training_seed": koopman_by_seed,
        "curves": aggregate_curves,
        "final_10x10_checkpoint_training_seed_statistics": {
            method: {
                str(counter): _training_seed_summary(values, len(training_seeds))
                for counter, values in endpoints.items()
            }
            for method, endpoints in final.items()
        },
        "statistical_note": (
            "10x10 episodes characterize each checkpoint; confidence intervals "
            f"use the {len(training_seeds)} independent training-seed means only."
        ),
    }
    return result, rows


def _write_outputs(
    output_json: Path,
    output_csv: Path,
    result: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, output_json)
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=list(FORMAL_TRAINING_SEEDS),
        help="Training seeds to aggregate (default: the full formal seed set)",
    )
    args = parser.parse_args()
    result, rows = aggregate(
        args.run_root, training_seeds=tuple(args.training_seeds)
    )
    _write_outputs(args.output_json, args.output_csv, result, rows)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
