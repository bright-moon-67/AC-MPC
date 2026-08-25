#!/usr/bin/env python
"""Build a certified long-table path bank from full-state reachability probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from antmaze_ac.envs.table_entry_bank import load_table_entry_trajectory_bank


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reachability-report", required=True)
    parser.add_argument("--entry-bank", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--entry-index", type=int, required=True)
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-start", type=float, default=0.0)
    parser.add_argument("--rotation-stop", type=float, default=32.0)
    parser.add_argument("--rotation-step", type=float, default=8.0)
    parser.add_argument("--minimum-segment-length", type=float, default=0.09)
    parser.add_argument("--maximum-segment-length", type=float, default=0.11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.activation_scale <= 0
        or args.rotation_step <= 0
        or args.rotation_stop <= args.rotation_start
        or not 0 < args.minimum_segment_length <= args.maximum_segment_length
    ):
        raise ValueError("path-bank ranges are invalid")
    report_path = Path(args.reachability_report).expanduser().resolve()
    entry_path = Path(args.entry_bank).expanduser().resolve()
    scenario_path = Path(args.scenario).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (report_path, entry_path, scenario_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bank = load_table_entry_trajectory_bank(entry_path)
    if args.entry_index < 0 or args.entry_index >= bank.trajectory_count:
        raise ValueError("entry index is out of range")
    expected_angles = np.arange(
        args.rotation_start,
        args.rotation_stop + 0.5 * args.rotation_step,
        args.rotation_step,
        dtype=np.float64,
    )
    selected = []
    for angle in expected_angles:
        matches = [
            row
            for row in report["probes"]
            if int(row["entry_index"]) == args.entry_index
            and np.isclose(float(row["activation_scale"]), args.activation_scale)
            and np.isclose(float(row["rotation_degrees"]), angle)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one reachability row for angle {angle:g}")
        if not bool(matches[0]["passed"]):
            raise ValueError(f"reachability row at angle {angle:g} is not certified")
        selected.append(matches[0])
    tips = np.asarray([row["final_tip"] for row in selected], dtype=np.float64)
    lengths = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    if np.any(lengths < args.minimum_segment_length) or np.any(
        lengths > args.maximum_segment_length
    ):
        raise ValueError(
            "certified path segment lengths fall outside the requested range: "
            f"{lengths.tolist()}"
        )
    base_action = np.asarray(
        bank.actions[args.entry_index, -1], dtype=np.float64
    ).reshape(6, 3)
    physical_actions = []
    for angle in expected_angles:
        radians = np.deg2rad(angle)
        rotation = np.asarray(
            [
                [np.cos(radians), -np.sin(radians)],
                [np.sin(radians), np.cos(radians)],
            ],
            dtype=np.float64,
        )
        values = base_action.copy()
        values[:, :2] = args.activation_scale * values[:, :2] @ rotation.T
        physical_actions.append(np.clip(values.reshape(-1), -0.30, 0.30))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int64),
            kind=np.asarray("manisoft_table_equilibrium_path_bank"),
            entry_index=np.asarray(args.entry_index, dtype=np.int64),
            entry_name=np.asarray(bank.names[args.entry_index]),
            activation_scale=np.asarray(args.activation_scale, dtype=np.float64),
            rotation_degrees=expected_angles,
            tip_positions=tips,
            physical_actions=np.asarray(physical_actions, dtype=np.float32),
            segment_lengths=lengths,
            scenario_sha256=np.asarray(_sha256(scenario_path)),
            entry_bank_sha256=np.asarray(_sha256(entry_path)),
            reachability_report_sha256=np.asarray(_sha256(report_path)),
        )
    temporary.replace(output)
    summary = {
        "kind": "manisoft_table_equilibrium_path_bank_manifest",
        "path_bank": str(output),
        "path_bank_sha256": _sha256(output),
        "entry_index": args.entry_index,
        "entry_name": bank.names[args.entry_index],
        "point_count": len(tips),
        "rotation_degrees": expected_angles.tolist(),
        "segment_lengths_m": lengths.tolist(),
        "maximum_height_span_m": float(np.ptp(tips[:, 2])),
        "source_reachability_report": str(report_path),
    }
    manifest = output.with_name("manifest.json")
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
