"""Validate and launch one method for one formal Walker Run training seed.

This deliberately is not a matrix launcher.  A command names exactly one
training seed and one method, which keeps failures and later baseline additions
isolated.  The immutable seed protocol is written before any detached process
is started.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.dmc.o2o.config import O2OConfig
from experiments.dmc.o2o.dataset import OfflineDataset
from experiments.dmc.o2o.koopman import FrozenKoopman, file_sha256


FORMAL_METHODS = (
    "Cal-RLPD-KMPC",
    "Cal-RLPD",
    "Cal-RLPD-Lift",
    "Cal-QL",
    "RLPD",
    "AWAC",
    "IQL",
)
FORMAL_TRAINING_SEEDS = tuple(range(20260851, 20260856))
FORMAL_SEED_BUDGET_REVISION = {
    "revised_utc_date": "2026-08-19",
    "previous_training_seeds": list(range(20260851, 20260861)),
    "revised_training_seeds": list(FORMAL_TRAINING_SEEDS),
    "reason": "user-approved formal campaign budget reduction after seed 20260851",
}
FORMAL_OFFLINE_TRANSITIONS = 200_000
FORMAL_OFFLINE_UPDATES = 50_000
FORMAL_OFFLINE_EVAL_INTERVAL = 5_000
FORMAL_ONLINE_TRANSITIONS = 20_000
FORMAL_ONLINE_EVAL_INTERVAL = 2_500
FORMAL_DIAGNOSTIC_EPISODES = 10
FORMAL_KOOPMAN_HORIZON = 20
FORMAL_KMPC_HORIZON = 8
FORMAL_MPVE_HORIZON = 4
FORMAL_FINAL_EVALUATION = {
    "online_steps": [0, 20_000],
    "evaluation_seeds": 10,
    "episodes_per_seed": 10,
    "parallel_workers": 10,
}


def _resolve_prepared_data_path(path: str) -> Path:
    """Resolve a prepared-data path after the documented storage relocation."""

    candidate = Path(path).resolve()
    try:
        if candidate.exists():
            return candidate
    except OSError:
        # CI and unprivileged deployments may not be allowed to stat another
        # machine's recorded /root path.  Continue to the relocation rule.
        pass
    old_prefix = "/root/autodl-tmp/AC-MPC/runs/o2o/data/"
    new_prefix = "/root/acmpc-o2o-nonformal-20260819/data/"
    raw = str(path)
    if raw.startswith(old_prefix):
        relocated = Path(new_prefix + raw[len(old_prefix) :]).resolve()
        try:
            if relocated.exists():
                return relocated
        except OSError:
            pass
    return candidate


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_dataset(path: Path) -> OfflineDataset:
    dataset = OfflineDataset.load(path.resolve())
    metadata = dataset.metadata
    selection = metadata.get("selection")
    if metadata.get("task") != "walker_run" or len(dataset) != FORMAL_OFFLINE_TRANSITIONS:
        raise ValueError("Formal Walker dataset must contain exactly 200k transitions")
    if not isinstance(selection, dict) or selection.get("kind") != (
        "temporal_block_microstratum_start_v1"
    ):
        raise ValueError("Formal dataset is missing temporal microstratum identity")
    expected = {
        "source_total_episodes": 10_000,
        "temporal_blocks": 10,
        "episodes_per_block": 1_000,
        "selected_episodes_per_block": 20,
        "microstratum_width_episodes": 50,
        "microstratum_offset": 0,
    }
    actual = {key: selection.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Formal 10x20 temporal selection differs: {actual}")
    source_ids = metadata.get("source_episode_indices")
    expected_ids = [
        block * 1_000 + offset
        for block in range(10)
        for offset in range(0, 1_000, 50)
    ]
    if source_ids != expected_ids:
        raise ValueError("Formal dataset episode IDs are not the 10x20 fixed selection")
    return dataset


def validate_koopman(
    path: Path, *, dataset: OfflineDataset, training_seed: int
) -> FrozenKoopman:
    koopman = FrozenKoopman(path.resolve())
    if (koopman.state_dim, koopman.action_dim, koopman.lift_dim) != (24, 6, 48):
        raise ValueError("Formal Walker Koopman dimensions must be state24/action6/lift48")
    if koopman.metadata.get("k_step") != FORMAL_KOOPMAN_HORIZON:
        raise ValueError("Formal Walker Koopman rollout horizon must be 20")
    if koopman.metadata.get("reward_layer_count") != 0 or koopman.metadata.get(
        "reward_training"
    ) != "disabled; reward is outside the Koopman contract":
        raise ValueError("Formal Koopman must be completely reward-free")
    source_path = koopman.metadata.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("Koopman export is missing its prepared-data directory")
    adapter_manifest_path = _resolve_prepared_data_path(source_path) / "manifest.json"
    try:
        adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("Koopman prepared-data manifest is missing or invalid") from exc
    if file_sha256(adapter_manifest_path) != koopman.metadata.get("dataset_sha256"):
        raise ValueError("Koopman export does not match its prepared-data manifest")
    if adapter_manifest.get("canonical_transitions_npz_sha256") != dataset.sha256:
        raise ValueError("Koopman was not trained from this seed protocol dataset")
    if koopman.metadata.get("seed") != training_seed:
        raise ValueError("Koopman training seed differs from the RL training seed")
    return koopman


def protocol_manifest(
    *, training_seed: int, dataset: OfflineDataset, koopman: FrozenKoopman
) -> dict[str, Any]:
    return {
        "kind": "acmpc_walker_formal_single_seed_protocol_v1",
        "task": "walker_run",
        "training_seed": training_seed,
        "seed_label": f"seed_{training_seed}",
        "formal_training_seeds": list(FORMAL_TRAINING_SEEDS),
        "campaign_seed_budget_revision": FORMAL_SEED_BUDGET_REVISION,
        "methods": list(FORMAL_METHODS),
        "method_specs": {
            method: dataclasses.asdict(
                O2OConfig(method=method, task="walker_run").method_spec
            )
            for method in FORMAL_METHODS
        },
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "transitions": len(dataset),
            "selection": dataset.metadata["selection"],
        },
        "koopman": {
            "path": str(koopman.path),
            "sha256": koopman.sha256,
            "training_seed": training_seed,
            "shared_by_methods": ["Cal-RLPD-KMPC", "Cal-RLPD-Lift"],
        },
        "training": {
            "offline_updates": FORMAL_OFFLINE_UPDATES,
            "offline_eval_interval_updates": FORMAL_OFFLINE_EVAL_INTERVAL,
            "online_transitions": FORMAL_ONLINE_TRANSITIONS,
            "online_eval_interval_transitions": FORMAL_ONLINE_EVAL_INTERVAL,
            "diagnostic_episodes": FORMAL_DIAGNOSTIC_EPISODES,
            "rlpd_offline_updates": 0,
            "rlpd_uses_same_200k_prior_buffer_online": True,
            "koopman_horizon": FORMAL_KOOPMAN_HORIZON,
            "koopman_max_windows_per_epoch": 500_000,
            "kmpc_horizon": FORMAL_KMPC_HORIZON,
            "mpve_horizon": FORMAL_MPVE_HORIZON,
        },
        "final_evaluation": FORMAL_FINAL_EVALUATION,
        "evaluation_seed_pools": {
            "diagnostic_seed_base": 9_000_000,
            "final_10x10_seed_base": 9_100_000,
            "disjoint": True,
        },
        "checkpoint_schema": {
            "offline": "offline_NNNNNN.pt",
            "online": "online_NNNNNN.pt",
            "aliases": ["latest.pt", "best.pt"],
        },
    }


def training_command(
    *,
    method: str,
    training_seed: int,
    dataset: OfflineDataset,
    koopman: FrozenKoopman | None,
    output_dir: Path,
    device: str,
) -> list[str]:
    if training_seed not in FORMAL_TRAINING_SEEDS:
        raise ValueError(f"Training seed is outside {FORMAL_TRAINING_SEEDS}")
    if method not in FORMAL_METHODS:
        raise ValueError(f"Unknown formal method {method!r}")
    config = O2OConfig(method=method, task="walker_run")
    command = [
        sys.executable,
        "-m",
        "experiments.dmc.o2o.train",
        "--task",
        "walker_run",
        "--method",
        method,
        "--dataset",
        str(dataset.path),
        "--output-dir",
        str(output_dir.resolve()),
        "--seed",
        str(training_seed),
        "--device",
        device,
        "--kmpc-horizon",
        str(FORMAL_KMPC_HORIZON),
        "--mpve-total-horizon",
        str(FORMAL_MPVE_HORIZON),
        "--offline-updates",
        str(FORMAL_OFFLINE_UPDATES),
        "--offline-eval-interval-updates",
        str(FORMAL_OFFLINE_EVAL_INTERVAL),
        "--online-steps",
        str(FORMAL_ONLINE_TRANSITIONS),
        "--eval-interval-online-steps",
        str(FORMAL_ONLINE_EVAL_INTERVAL),
        "--eval-episodes",
        str(FORMAL_DIAGNOSTIC_EPISODES),
    ]
    if config.requires_koopman:
        if koopman is None:
            raise ValueError(f"{method} requires a completed validated Koopman")
        command.extend(("--koopman", str(koopman.path)))
    return command


def final_evaluation_commands(
    *, method: str, run_dir: Path, device: str = "cpu"
) -> list[list[str]]:
    if method not in FORMAL_METHODS:
        raise ValueError(f"Unknown formal method {method!r}")
    commands = []
    for checkpoint in ("online_000000", "online_020000"):
        commands.append(
            [
                sys.executable,
                "-m",
                "experiments.dmc.o2o.evaluate_10x10",
                "--run-dir",
                str(run_dir.resolve()),
                "--checkpoint",
                checkpoint,
                "--device",
                device,
                "--parallel-workers",
                "10",
                "--output",
                str(run_dir.resolve() / f"evaluation_10x10_{checkpoint}.json"),
            ]
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-seed", type=int, choices=FORMAL_TRAINING_SEEDS, required=True
    )
    parser.add_argument("--method", choices=FORMAL_METHODS, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--koopman", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="start one protected detached method process after validation",
    )
    parser.add_argument(
        "--show-final-evaluations",
        action="store_true",
        help="also print the online-0/20k parallel 10x10 commands",
    )
    args = parser.parse_args()

    dataset = validate_dataset(args.dataset)
    seed_dir = (args.run_root / f"seed_{args.training_seed}").resolve()
    requires_koopman = O2OConfig(
        method=args.method, task="walker_run"
    ).requires_koopman
    if requires_koopman and args.koopman is None:
        raise ValueError(f"{args.method} requires --koopman after Koopman training")
    koopman = (
        None
        if args.koopman is None
        else validate_koopman(
            args.koopman, dataset=dataset, training_seed=args.training_seed
        )
    )
    plan = {
        "kind": "acmpc_walker_formal_single_seed_plan_v1",
        "task": "walker_run",
        "training_seed": args.training_seed,
        "seed_label": f"seed_{args.training_seed}",
        "formal_training_seeds": list(FORMAL_TRAINING_SEEDS),
        "campaign_seed_budget_revision": FORMAL_SEED_BUDGET_REVISION,
        "methods": list(FORMAL_METHODS),
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "transitions": len(dataset),
            "selection": dataset.metadata["selection"],
        },
        "expected_koopman_path": str((seed_dir / "koopman" / "best.npz").resolve()),
    }
    plan_path = seed_dir / "seed_plan.json"
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing seed plan differs; refusing mixed identities")
    else:
        _atomic_json(plan_path, plan)

    manifest = None if koopman is None else protocol_manifest(
        training_seed=args.training_seed, dataset=dataset, koopman=koopman
    )
    protocol_path = seed_dir / "protocol.json"
    if protocol_path.is_file() and manifest is not None:
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing seed protocol differs; refusing mixed identities")
    elif manifest is not None:
        _atomic_json(protocol_path, manifest)

    output_dir = seed_dir / args.method
    command = training_command(
        method=args.method,
        training_seed=args.training_seed,
        dataset=dataset,
        koopman=koopman,
        output_dir=output_dir,
        device=args.device,
    )
    print(shlex.join(command), flush=True)
    if args.show_final_evaluations:
        for evaluation_command in final_evaluation_commands(
            method=args.method, run_dir=output_dir
        ):
            print(shlex.join(evaluation_command), flush=True)
    if not args.launch:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_path = output_dir / "launch.json"
    session_name = (
        f"walker_formal_{args.training_seed}_{args.method}"
        .lower()
        .replace("-", "_")
    )
    if subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0:
        raise RuntimeError(f"Protected tmux session already exists: {session_name}")
    log_path = output_dir / "train.log"
    repository = Path(__file__).resolve().parents[3]
    shell_command = (
        f"cd {shlex.quote(str(repository))} && "
        f"exec env PYTHONUNBUFFERED=1 {shlex.join(command)} "
        f">> {shlex.quote(str(log_path))} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, shell_command],
        check=True,
    )
    pane_pid = int(
        subprocess.check_output(
            ["tmux", "display-message", "-p", "-t", session_name, "#{pane_pid}"],
            text=True,
        ).strip()
    )
    _atomic_json(
        launch_path,
        {
            "kind": "acmpc_protected_single_method_tmux_launch_v1",
            "training_seed": args.training_seed,
            "method": args.method,
            "tmux_session": session_name,
            "pane_pid": pane_pid,
            "command": command,
            "log_path": str(log_path),
            "launched_unix_seconds": time.time(),
        },
    )
    print(
        f"launched tmux={session_name} pane_pid={pane_pid} log={log_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
