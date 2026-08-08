"""Compare contact-dim Koopman predictability across contact configurations.

Reads the per-group metrics already computed by
``train_hopper_hop_koopman`` (report.json: qpos/qvel/contact groups, one_step
and rollout, rmse + hold_rmse) for every MuJoCo contact config plus the
original PhysX model (``runs/hopper_hop/koopman_v2``), and prints/aggregates a
comparison focused on the CONTACT dims (toe_touch / heel_touch).

Core hypothesis (NC-inspired): with compliant contact the contact forces are
smooth/continuous functions of state, so a linear Koopman model should predict
the contact dims with LOWER error than under near-rigid contact.

Output:
    runs/hopper_hop_mujoco/koopman/contact_dim_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

CONFIGS = ("mujoco_default", "mujoco_compliant", "mujoco_hard")
PHYSX_BASELINE = Path("runs/hopper_hop/koopman_v2/report.json")
GROUPS = ("qpos", "qvel", "contact", "all")


def _group_metrics(report: dict[str, Any], split: str, mode: str) -> dict[str, Any]:
    """Return {group: {rmse, hold_rmse}} for a split ('test') and mode (one_step/rollout)."""
    out: dict[str, Any] = {}
    metrics = report["metrics"][split][mode]
    for group in GROUPS:
        if group not in metrics:
            continue
        out[group] = {
            "rmse": float(metrics[group]["rmse"]),
            "hold_rmse": float(metrics[group]["hold_rmse"]),
        }
    return out


def compare(
    koopman_root: Path,
    splits: tuple[str, ...] = ("test",),
    modes: tuple[str, ...] = ("one_step", "rollout"),
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    # PhysX baseline first
    if PHYSX_BASELINE.exists():
        report = json.loads(PHYSX_BASELINE.read_text())
        rows["physx_hard (koopman_v2)"] = {
            "contact_rmse": {
                mode: _group_metrics(report, "test", mode).get("contact", {}).get("rmse")
                for mode in modes
            },
            "contact_hold_rmse": {
                mode: _group_metrics(report, "test", mode).get("contact", {}).get("hold_rmse")
                for mode in modes
            },
            "all_rmse": {
                mode: _group_metrics(report, "test", mode).get("all", {}).get("rmse")
                for mode in modes
            },
            "best_epoch": report.get("best_epoch"),
        }
    for config in CONFIGS:
        report_path = koopman_root / config / "report.json"
        if not report_path.exists():
            print(f"[skip] {report_path} not found", flush=True)
            continue
        report = json.loads(report_path.read_text())
        row: dict[str, Any] = {}
        for mode in modes:
            g = _group_metrics(report, "test", mode)
            row[f"contact_rmse.{mode}"] = g.get("contact", {}).get("rmse")
            row[f"contact_hold_rmse.{mode}"] = g.get("contact", {}).get("hold_rmse")
            row[f"all_rmse.{mode}"] = g.get("all", {}).get("rmse")
            row[f"qpos_rmse.{mode}"] = g.get("qpos", {}).get("rmse")
            row[f"qvel_rmse.{mode}"] = g.get("qvel", {}).get("rmse")
        row["best_epoch"] = report.get("best_epoch")
        row["best_val_nMSE"] = report.get("best_validation_rollout_normalized_mse")
        rows[config] = row

    # ---- print comparison table ----
    def _fmt(value: Any, width: int) -> str:
        return "n/a".rjust(width) if value is None else f"{value:.3f}".rjust(width)

    header = (
        f"{'config':<28} {'c1step':>7} {'c1step_hold':>11} {'croll':>7} "
        f"{'croll_hold':>11} {'all1step':>9} {'bestEp':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, row in rows.items():
        c1 = row.get("contact_rmse", {}).get("one_step") or row.get("contact_rmse.one_step")
        c1h = row.get("contact_hold_rmse", {}).get("one_step") or row.get("contact_hold_rmse.one_step")
        cr = row.get("contact_rmse", {}).get("rollout") or row.get("contact_rmse.rollout")
        crh = row.get("contact_hold_rmse", {}).get("rollout") or row.get("contact_hold_rmse.rollout")
        ar = row.get("all_rmse", {}).get("one_step") or row.get("all_rmse.one_step")
        print(
            f"{name:<28} {_fmt(c1, 7)} {_fmt(c1h, 11)} {_fmt(cr, 7)} "
            f"{_fmt(crh, 11)} {_fmt(ar, 9)} {row.get('best_epoch', '')!s:>6}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--koopman-root",
        type=Path,
        default=Path("runs/hopper_hop_mujoco/koopman"),
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare(args.koopman_root)
    out = args.out or (args.koopman_root / "contact_dim_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"\nwrote: {out}", flush=True)


if __name__ == "__main__":
    main()
