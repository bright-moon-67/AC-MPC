"""Validate, aggregate, tabulate, and plot the formal Cartpole campaign."""

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
from experiments.dmc.o2o.formal_cartpole import FORMAL_METHODS, FORMAL_TRAINING_SEEDS
from experiments.dmc.o2o.koopman import file_sha256


SUMMARY_KIND = "acmpc_cartpole_formal_training_seed_summary_v1"
OFFLINE_POINTS = tuple(range(0, 50_001, 5_000))
ONLINE_POINTS = tuple(range(0, 20_001, 2_500))
T_CRITICAL_95 = 2.7764451051977987  # df=4 for five training seeds
MAX_RETURN = 1000.0
COLORS = {
    "Cal-RLPD-KMPC": "#b279a2",
    "Cal-RLPD": "#54a24b",
    "Cal-RLPD-Lift": "#ff9da6",
    "Cal-QL": "#4c78a8",
    "RLPD": "#72b7b2",
    "AWAC": "#9c755f",
    "IQL": "#bab0ac",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def training_seed_statistics(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(FORMAL_TRAINING_SEEDS),) or not np.isfinite(array).all():
        raise ValueError("Formal statistics require five finite training-seed values")
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1))
    standard_error = sample_std / math.sqrt(len(array))
    half_width = T_CRITICAL_95 * standard_error
    return {
        "training_seed_values": [float(value) for value in array],
        "training_seed_count": len(array),
        "mean": mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_student_t": [mean - half_width, mean + half_width],
        "inference_unit": "independent_training_seed",
    }


def _archive_members(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = run_dir / "checkpoints.archive.json"
    manifest = _read_json(manifest_path)
    archive = run_dir / "checkpoints.zip"
    if (
        manifest.get("kind") != "acmpc_checkpoint_zip_archive_v1"
        or manifest.get("archive") != archive.name
        or not archive.is_file()
        or archive.stat().st_size != manifest.get("archive_bytes")
        or manifest.get("sources_deleted_after_crc_and_sha256_verification") is not True
    ):
        raise ValueError(f"Invalid verified checkpoint archive: {run_dir}")
    if list(run_dir.rglob("*.pt")):
        raise ValueError(f"Archived run still contains checkpoint sources: {run_dir}")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"Archive manifest has no members: {manifest_path}")
    indexed = {
        value["name"]: value
        for value in members
        if isinstance(value, dict)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("sha256"), str)
    }
    if len(indexed) != len(members):
        raise ValueError(f"Archive member identities are invalid: {manifest_path}")
    return manifest, indexed


def _diagnostic(
    run_dir: Path,
    stage: str,
    counter: int,
    *,
    seed: int,
    method: str,
    members: dict[str, dict[str, Any]],
) -> float:
    stem = f"{stage}_{counter:06d}"
    payload = _read_json(run_dir / f"evaluation_{stem}.json")
    member = members.get(f"{stem}.pt")
    expected_path = str((run_dir / f"{stem}.pt").resolve())
    if (
        member is None
        or payload.get("kind") != "acmpc_dmc_o2o_training_diagnostic_v1"
        or payload.get("task") != "cartpole_swingup"
        or payload.get("training_seed") != seed
        or payload.get("method") != method
        or payload.get("stage") != stage
        or payload.get(f"{stage}_update" if stage == "offline" else "online_step")
        != counter
        or payload.get("checkpoint_path") != expected_path
        or payload.get("checkpoint_sha256") != member["sha256"]
    ):
        raise ValueError(f"Diagnostic identity differs: {run_dir}/{stem}")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"Invalid diagnostic evaluation: {run_dir}/{stem}")
    returns = evaluation.get("returns")
    value = evaluation.get("return_mean")
    if (
        not isinstance(returns, list)
        or len(returns) != 10
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Diagnostic must contain ten finite returns: {run_dir}/{stem}")
    return float(value)


def _final_10x10(
    run_dir: Path,
    counter: int,
    *,
    seed: int,
    method: str,
    members: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = f"online_{counter:06d}"
    payload = _read_json(run_dir / f"evaluation_10x10_{checkpoint}.json")
    member = members.get(f"{checkpoint}.pt")
    protocol = payload.get("evaluation_protocol")
    returns = payload.get("returns")
    if (
        member is None
        or payload.get("kind") != FINAL_EVAL_KIND
        or payload.get("task") != "cartpole_swingup"
        or payload.get("training_seed") != seed
        or payload.get("method") != method
        or payload.get("checkpoint_name") != checkpoint
        or payload.get("online_step") != counter
        or payload.get("checkpoint_path")
        != str((run_dir / f"{checkpoint}.pt").resolve())
        or payload.get("checkpoint_sha256") != member["sha256"]
        or not isinstance(protocol, dict)
        or (
            protocol.get("num_evaluation_seeds"),
            protocol.get("episodes_per_seed"),
            protocol.get("total_episodes"),
            protocol.get("parallel_workers"),
        )
        != (10, 10, 100, 10)
        or not isinstance(returns, list)
        or len(returns) != 100
    ):
        raise ValueError(f"Final 10x10 identity differs: {run_dir}/{checkpoint}")
    value = float(payload.get("return_mean"))
    if not math.isfinite(value):
        raise ValueError(f"Final 10x10 return is non-finite: {run_dir}/{checkpoint}")
    return payload


def aggregate(run_root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    run_root = run_root.resolve()
    curve_values = {
        method: {"offline": {}, "online": {}} for method in FORMAL_METHODS
    }
    final_values = {method: {0: [], 20_000: []} for method in FORMAL_METHODS}
    auc_values = {method: [] for method in FORMAL_METHODS}
    regret_values = {method: [] for method in FORMAL_METHODS}
    gain_values = {method: [] for method in FORMAL_METHODS}
    curve_seed_rows: list[dict[str, Any]] = []
    endpoint_seed_rows: list[dict[str, Any]] = []
    dataset_sha: str | None = None
    koopman_sha_by_seed: dict[str, str] = {}
    archive_manifest_sha: dict[str, dict[str, str]] = {}

    for seed in FORMAL_TRAINING_SEEDS:
        seed_dir = run_root / f"seed_{seed}"
        protocol = _read_json(seed_dir / "protocol.json")
        if (
            protocol.get("kind") != "acmpc_cartpole_formal_single_seed_protocol_v1"
            or protocol.get("task") != "cartpole_swingup"
            or protocol.get("training_seed") != seed
            or protocol.get("formal_training_seeds") != list(FORMAL_TRAINING_SEEDS)
            or protocol.get("methods") != list(FORMAL_METHODS)
        ):
            raise ValueError(f"Formal protocol differs: {seed_dir}")
        current_dataset_sha = protocol.get("dataset", {}).get("sha256")
        if dataset_sha is None:
            dataset_sha = current_dataset_sha
            dataset_path = Path(protocol["dataset"]["path"])
            if file_sha256(dataset_path) != dataset_sha:
                raise ValueError("Formal dataset file SHA differs from protocol")
        elif current_dataset_sha != dataset_sha:
            raise ValueError("Formal seeds do not share one dataset SHA")
        koopman = protocol.get("koopman")
        if not isinstance(koopman, dict) or koopman.get("training_seed") != seed:
            raise ValueError(f"Seed-paired Koopman identity differs: {seed_dir}")
        koopman_path = Path(koopman["path"])
        if file_sha256(koopman_path) != koopman.get("sha256"):
            raise ValueError(f"Koopman file SHA differs: {koopman_path}")
        koopman_sha_by_seed[str(seed)] = koopman["sha256"]
        archive_manifest_sha[str(seed)] = {}

        for method in FORMAL_METHODS:
            run_dir = seed_dir / method
            run = _read_json(run_dir / "run.json")
            config = run.get("config")
            expected_offline = 0 if method == "RLPD" else 50_000
            if (
                not isinstance(config, dict)
                or config.get("task") != "cartpole_swingup"
                or config.get("method") != method
                or config.get("seed") != seed
                or config.get("offline_updates") != 50_000
                or config.get("online_steps") != 20_000
                or config.get("offline_eval_interval_updates") != 5_000
                or config.get("eval_interval_online_steps") != 2_500
                or run.get("completed") is not True
                or run.get("offline_updates_completed") != expected_offline
                or run.get("online_steps_completed") != 20_000
                or run.get("method_spec") != protocol["method_specs"][method]
                or run.get("dataset", {}).get("sha256") != dataset_sha
            ):
                raise ValueError(f"Completed-run contract differs: {run_dir}")
            run_koopman = run.get("koopman")
            if method in {"Cal-RLPD-KMPC", "Cal-RLPD-Lift"}:
                if not isinstance(run_koopman, dict) or run_koopman.get("sha256") != koopman["sha256"]:
                    raise ValueError(f"Structured run Koopman differs: {run_dir}")
            elif run_koopman is not None:
                raise ValueError(f"Baseline unexpectedly loaded Koopman: {run_dir}")

            _manifest, members = _archive_members(run_dir)
            archive_manifest_sha[str(seed)][method] = file_sha256(
                run_dir / "checkpoints.archive.json"
            )
            offline_grid = (0,) if method == "RLPD" else OFFLINE_POINTS
            for counter in offline_grid:
                source_stage = "online" if method == "RLPD" else "offline"
                value = _diagnostic(
                    run_dir, source_stage, counter, seed=seed, method=method,
                    members=members,
                )
                curve_values[method]["offline"].setdefault(counter, []).append(value)
                curve_seed_rows.append({
                    "training_seed": seed, "method": method, "stage": "offline",
                    "counter": counter, "diagnostic_return_mean": value,
                })
            online_seed_values = []
            for counter in ONLINE_POINTS:
                value = _diagnostic(
                    run_dir, "online", counter, seed=seed, method=method,
                    members=members,
                )
                online_seed_values.append(value)
                curve_values[method]["online"].setdefault(counter, []).append(value)
                curve_seed_rows.append({
                    "training_seed": seed, "method": method, "stage": "online",
                    "counter": counter, "diagnostic_return_mean": value,
                })
            area = float(np.trapezoid(online_seed_values, np.asarray(ONLINE_POINTS)))
            auc_values[method].append(area / ONLINE_POINTS[-1])
            regret_values[method].append(MAX_RETURN * ONLINE_POINTS[-1] - area)
            endpoints = {}
            for counter in (0, 20_000):
                payload = _final_10x10(
                    run_dir, counter, seed=seed, method=method, members=members,
                )
                value = float(payload["return_mean"])
                endpoints[counter] = value
                final_values[method][counter].append(value)
            gain = endpoints[20_000] - endpoints[0]
            gain_values[method].append(gain)
            endpoint_seed_rows.append({
                "method": method,
                "training_seed": seed,
                "online_000000_return_mean": endpoints[0],
                "online_020000_return_mean": endpoints[20_000],
                "absolute_gain": gain,
                "online_auc_mean_return": area / ONLINE_POINTS[-1],
                "cumulative_regret_return_steps": MAX_RETURN * ONLINE_POINTS[-1] - area,
            })

    if len(set(koopman_sha_by_seed.values())) != len(FORMAL_TRAINING_SEEDS):
        raise ValueError("Each formal training seed must own a distinct Koopman artifact")

    curves = {
        method: {
            stage: {
                str(counter): training_seed_statistics(values)
                for counter, values in sorted(points.items())
            }
            for stage, points in stages.items()
        }
        for method, stages in curve_values.items()
    }
    endpoints = {}
    for method in FORMAL_METHODS:
        initial = training_seed_statistics(final_values[method][0])
        final = training_seed_statistics(final_values[method][20_000])
        gain = training_seed_statistics(gain_values[method])
        endpoints[method] = {
            "online_000000": initial,
            "online_020000": final,
            "paired_absolute_gain": gain,
            "relative_gain_percent_of_endpoint_means": (
                (final["mean"] - initial["mean"]) / initial["mean"] * 100.0
            ),
            "online_auc_mean_return": training_seed_statistics(auc_values[method]),
            "cumulative_regret_return_steps": training_seed_statistics(
                regret_values[method]
            ),
        }
    result = {
        "kind": SUMMARY_KIND,
        "task": "cartpole_swingup",
        "run_root": str(run_root),
        "methods": list(FORMAL_METHODS),
        "training_seeds": list(FORMAL_TRAINING_SEEDS),
        "dataset_sha256": dataset_sha,
        "koopman_sha256_by_training_seed": koopman_sha_by_seed,
        "checkpoint_archive_manifest_sha256": archive_manifest_sha,
        "curves": curves,
        "final_10x10": endpoints,
        "curve_definitions": {
            "offline": "10 deterministic diagnostic episodes vs offline gradient updates",
            "online": "10 deterministic diagnostic episodes vs real online transitions",
            "online_auc_mean_return": "trapezoidal online diagnostic AUC divided by 20,000 transitions",
            "cumulative_regret_return_steps": "integral of (1000 - diagnostic return) over 0..20,000",
        },
        "statistical_note": (
            "Each 10x10 checkpoint contains 100 descriptive evaluation episodes. "
            "Sample std, SE, and Student-t 95% CI use only the five independent "
            "training-seed means (df=4)."
        ),
        "unavailable_metrics": [
            "action_saturation (actions were not stored by the evaluator)",
            "task success rate (Cartpole evaluator has no registered binary success metric)",
        ],
    }
    return result, {
        "curve_seed_rows": curve_seed_rows,
        "endpoint_seed_rows": endpoint_seed_rows,
    }


def _curve_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in FORMAL_METHODS:
        for stage in ("offline", "online"):
            for counter, stats in result["curves"][method][stage].items():
                rows.append({
                    "method": method, "stage": stage, "counter": int(counter),
                    "mean": stats["mean"], "sample_std": stats["sample_std"],
                    "standard_error": stats["standard_error"],
                    "ci95_low": stats["ci95_student_t"][0],
                    "ci95_high": stats["ci95_student_t"][1],
                })
    return rows


def _endpoint_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in FORMAL_METHODS:
        value = result["final_10x10"][method]
        initial, final, gain = (
            value["online_000000"], value["online_020000"],
            value["paired_absolute_gain"],
        )
        rows.append({
            "method": method,
            "training_seed_count": 5,
            "online_000000_mean": initial["mean"],
            "online_000000_sample_std": initial["sample_std"],
            "online_000000_se": initial["standard_error"],
            "online_000000_ci95_low": initial["ci95_student_t"][0],
            "online_000000_ci95_high": initial["ci95_student_t"][1],
            "online_020000_mean": final["mean"],
            "online_020000_sample_std": final["sample_std"],
            "online_020000_se": final["standard_error"],
            "online_020000_ci95_low": final["ci95_student_t"][0],
            "online_020000_ci95_high": final["ci95_student_t"][1],
            "paired_gain_mean": gain["mean"],
            "paired_gain_sample_std": gain["sample_std"],
            "paired_gain_ci95_low": gain["ci95_student_t"][0],
            "paired_gain_ci95_high": gain["ci95_student_t"][1],
            "relative_gain_percent_of_means": value[
                "relative_gain_percent_of_endpoint_means"
            ],
            "online_auc_mean_return": value["online_auc_mean_return"]["mean"],
            "cumulative_regret_return_steps": value[
                "cumulative_regret_return_steps"
            ]["mean"],
        })
    return rows


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Cartpole Swingup formal 5-seed 10×10 evaluation",
        "",
        "Statistics use five independent training-seed checkpoint means. Each checkpoint mean is based on 10 evaluation seeds × 10 episodes.",
        "",
        "| Method | Online 0 mean ± seed SD | Online 20k mean ± seed SD | Paired gain | 95% CI (20k) | Online AUC mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['online_000000_mean']:.2f} ± "
            f"{row['online_000000_sample_std']:.2f} | "
            f"{row['online_020000_mean']:.2f} ± {row['online_020000_sample_std']:.2f} | "
            f"{row['paired_gain_mean']:+.2f} | "
            f"[{row['online_020000_ci95_low']:.2f}, {row['online_020000_ci95_high']:.2f}] | "
            f"{row['online_auc_mean_return']:.2f} |"
        )
    lines.extend([
        "",
        "RLPD performs zero offline gradient updates, so its Online 0 score is an untrained-policy reference rather than an offline-RL result.",
        "",
    ])
    return "\n".join(lines)


def _plot(result: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)
    for method in FORMAL_METHODS:
        color = COLORS[method]
        for axis, stage in zip(axes, ("offline", "online"), strict=True):
            values = result["curves"][method][stage]
            x = np.asarray([int(counter) for counter in values]) / 1000.0
            mean = np.asarray([value["mean"] for value in values.values()])
            low = np.asarray([value["ci95_student_t"][0] for value in values.values()])
            high = np.asarray([value["ci95_student_t"][1] for value in values.values()])
            axis.plot(x, mean, marker="o", markersize=4, linewidth=2, color=color, label=method)
            if len(x) > 1:
                axis.fill_between(
                    x, np.clip(low, 0.0, MAX_RETURN), np.clip(high, 0.0, MAX_RETURN),
                    color=color, alpha=0.12,
                )
    axes[0].set_title("Offline policy evaluation")
    axes[0].set_xlabel("Offline gradient updates (thousands)")
    axes[1].set_title("Online policy evaluation")
    axes[1].set_xlabel("Real online transitions (thousands)")
    for axis in axes:
        axis.set_ylabel("Deterministic episode return")
        axis.set_ylim(0, 1000)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle("DMC Cartpole Swingup — formal 5-seed mean with Student-t 95% CI")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
    os.replace(temporary, output)
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    result: dict[str, Any],
    raw_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "formal_summary.json"
    curve_seed_csv = output_dir / "evaluation_curves_by_training_seed.csv"
    curve_summary_csv = output_dir / "evaluation_curves_summary.csv"
    endpoint_seed_csv = output_dir / "final_10x10_by_training_seed.csv"
    endpoint_summary_csv = output_dir / "final_10x10_summary.csv"
    endpoint_markdown = output_dir / "final_10x10_summary.md"
    plot_path = output_dir / "offline_online_evaluation_curves.png"
    endpoint_rows = _endpoint_summary_rows(result)
    _atomic_text(
        summary_json,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_csv(curve_seed_csv, raw_rows["curve_seed_rows"])
    _write_csv(curve_summary_csv, _curve_summary_rows(result))
    _write_csv(endpoint_seed_csv, raw_rows["endpoint_seed_rows"])
    _write_csv(endpoint_summary_csv, endpoint_rows)
    _atomic_text(endpoint_markdown, _markdown_table(endpoint_rows))
    _plot(result, plot_path)
    return {
        "summary_json": str(summary_json),
        "curve_by_training_seed_csv": str(curve_seed_csv),
        "curve_summary_csv": str(curve_summary_csv),
        "final_10x10_by_training_seed_csv": str(endpoint_seed_csv),
        "final_10x10_summary_csv": str(endpoint_summary_csv),
        "final_10x10_summary_markdown": str(endpoint_markdown),
        "offline_online_plot_png": str(plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_root / "summary"
    result, rows = aggregate(args.run_root)
    print(json.dumps(write_outputs(output_dir, result, rows), indent=2))


if __name__ == "__main__":
    main()
