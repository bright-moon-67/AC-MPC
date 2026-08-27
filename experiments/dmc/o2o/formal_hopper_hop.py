"""Single-seed formal Hopper Hop O2O launcher (TD-MPC2 AR2 protocol)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.dmc.o2o.config import O2OConfig
from experiments.dmc.o2o.dataset import OfflineDataset
from experiments.dmc.o2o.formal_hopper import (
    FORMAL_DIAGNOSTIC_EPISODES,
    FORMAL_KMPC_HORIZON,
    FORMAL_KOOPMAN_HORIZON,
    FORMAL_METHODS,
    FORMAL_MPVE_HORIZON,
    FORMAL_OFFLINE_EVAL_INTERVAL,
    FORMAL_OFFLINE_TRANSITIONS,
    FORMAL_OFFLINE_UPDATES,
    FORMAL_ONLINE_EVAL_INTERVAL,
    FORMAL_ONLINE_TRANSITIONS,
    FORMAL_TRAINING_SEEDS,
)
from experiments.dmc.o2o.koopman import FrozenKoopman, file_sha256


TASK = "hopper_hop"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_dataset(path: Path) -> OfflineDataset:
    dataset = OfflineDataset.load(path.resolve())
    selection = dataset.metadata.get("selection")
    expected = {
        "kind": "hopper_temporal_block_v1",
        "source_total_episodes": 24_000,
        "temporal_blocks": 10,
        "episodes_per_block": 2_400,
        "selected_episodes_per_block": 40,
        "microstratum_width_episodes": 60,
        "microstratum_offset": 0,
    }
    if dataset.metadata.get("task") != TASK or len(dataset) != FORMAL_OFFLINE_TRANSITIONS:
        raise ValueError("Formal Hopper Hop dataset must contain exactly 200k transitions")
    if dataset.metadata.get("reward_source") != "recorded":
        raise ValueError("Formal Hopper Hop dataset must retain recorded TD-MPC2 rewards")
    timing = {
        "action_repeat": 2,
        "control_dt": 0.04,
        "transitions_per_episode": 500,
    }
    timing_mismatches = {
        key: {"dataset": dataset.metadata.get(key), "expected": value}
        for key, value in timing.items()
        if dataset.metadata.get(key) != value
    }
    if timing_mismatches:
        raise ValueError(
            "Formal Hopper Hop dataset is not TD-MPC2 AR2: "
            f"{timing_mismatches}"
        )
    if not isinstance(selection, dict) or any(selection.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Formal Hopper temporal selection differs: {selection}")
    ids = [
        block * 2_400 + offset
        for block in range(10)
        for offset in range(0, 2_400, 60)
    ]
    if dataset.metadata.get("source_episode_indices") != ids:
        raise ValueError("Formal dataset episode IDs are not the fixed 10x40 selection")
    return dataset


def validate_koopman(
    path: Path,
    *,
    dataset: OfflineDataset,
    seed: int,
    selection_kind: str = "hopper_mixed_quality_balanced_v1",
) -> FrozenKoopman:
    """Validate a seed-matched model, including relocated merged artifacts."""
    model = FrozenKoopman(path.resolve())
    if (model.state_dim, model.action_dim, model.lift_dim) != (15, 4, 48):
        raise ValueError("Formal Hopper Koopman dimensions must be state15/action4/lift48")
    if model.metadata.get("k_step") != FORMAL_KOOPMAN_HORIZON:
        raise ValueError("Formal Hopper Koopman rollout horizon must be 20")
    if (
        model.metadata.get("reward_layer_count") != 0
        or model.metadata.get("reward_training")
        != "disabled; reward is outside the Koopman contract"
    ):
        raise ValueError("Formal Hopper Koopman must be reward-free")

    source_path = model.metadata.get("source_path")
    if not isinstance(source_path, str):
        raise ValueError("Koopman export is missing source_path")
    manifest_path = Path(source_path).resolve() / "manifest.json"
    merged_root = model.path.parents[2]
    if not manifest_path.is_file():
        manifest_path = merged_root / "koopman_prepared" / "shared" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Koopman prepared-data manifest is unavailable")
    if file_sha256(manifest_path) != model.metadata.get("dataset_sha256"):
        raise ValueError("Koopman export does not match its prepared-data manifest")
    manifest = json.loads(manifest_path.read_text())

    corpus_value = manifest.get("canonical_transitions_npz")
    if not isinstance(corpus_value, str):
        raise ValueError("Koopman manifest is missing its canonical corpus path")
    corpus_path = Path(corpus_value).resolve()
    if not corpus_path.is_file():
        corpus_path = merged_root / "datasets" / "hopper_koopman_qualitymix_400k.npz"
    corpus = OfflineDataset.load(corpus_path)
    if manifest.get("canonical_transitions_npz_sha256") == dataset.sha256:
        raise ValueError("Koopman unexpectedly points at the 200k RL-only dataset")
    if len(corpus) != 400_000 or corpus.metadata.get("task") != "hopper_stand":
        raise ValueError("Koopman was not trained from the mixed 400k Hopper corpus")
    if manifest.get("canonical_transitions_npz_sha256") != corpus.sha256:
        raise ValueError("Koopman prepared-data manifest does not bind its corpus")
    if corpus.metadata.get("selection", {}).get("kind") != selection_kind:
        raise ValueError("Koopman corpus selection differs from the requested contract")
    if manifest.get("selection", {}).get("kind") != selection_kind:
        raise ValueError("Koopman prepared-data selection differs from its corpus")
    if model.metadata.get("seed") != seed:
        raise ValueError("Koopman seed differs from RL seed")
    return model


def training_command(
    *,
    method: str,
    seed: int,
    dataset: OfflineDataset,
    koopman: FrozenKoopman | None,
    output: Path,
    device: str,
    offline_actor_learning_rate: float | None = None,
    offline_critic_learning_rate: float | None = None,
    initialize_from_offline_final: Path | None = None,
) -> list[str]:
    config = O2OConfig(method=method, task=TASK)
    command = [
        sys.executable,
        "-m",
        "experiments.dmc.o2o.train",
        "--task",
        TASK,
        "--method",
        method,
        "--dataset",
        str(dataset.path),
        "--output-dir",
        str(output.resolve()),
        "--seed",
        str(seed),
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
            raise ValueError(f"{method} requires Koopman")
        command.extend(("--koopman", str(koopman.path)))
    if offline_actor_learning_rate is not None:
        command.extend(
            ("--offline-actor-learning-rate", str(offline_actor_learning_rate))
        )
    if offline_critic_learning_rate is not None:
        command.extend(
            ("--offline-critic-learning-rate", str(offline_critic_learning_rate))
        )
    if initialize_from_offline_final is not None:
        command.extend(
            (
                "--initialize-from-offline-final",
                str(initialize_from_offline_final.resolve()),
            )
        )
    return command


def training_session_name(run_root: Path, seed: int, method: str) -> str:
    root_tag = hashlib.sha256(str(run_root.resolve()).encode()).hexdigest()[:8]
    return f"hopper_hop_{root_tag}_{seed}_{method}".lower().replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-seed", type=int, choices=FORMAL_TRAINING_SEEDS, required=True
    )
    parser.add_argument("--method", choices=FORMAL_METHODS, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--koopman", type=Path)
    parser.add_argument(
        "--koopman-selection-kind",
        choices=(
            "hopper_mixed_temporal_block_v1",
            "hopper_mixed_quality_balanced_v1",
        ),
        default="hopper_mixed_quality_balanced_v1",
    )
    parser.add_argument("--offline-actor-learning-rate", type=float)
    parser.add_argument("--offline-critic-learning-rate", type=float)
    parser.add_argument("--initialize-from-offline-final", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    dataset = validate_dataset(args.dataset)
    seed_dir = (args.run_root / f"seed_{args.training_seed}").resolve()
    koopman = None if args.koopman is None else validate_koopman(
        args.koopman,
        dataset=dataset,
        seed=args.training_seed,
        selection_kind=args.koopman_selection_kind,
    )
    config = O2OConfig(method=args.method, task=TASK)
    if config.requires_koopman and koopman is None:
        raise ValueError(f"{args.method} requires --koopman")
    if not config.requires_koopman and koopman is not None:
        raise ValueError(f"{args.method} forbids --koopman")

    plan = {
        "kind": "acmpc_hopper_hop_formal_single_seed_protocol_v1",
        "task": TASK,
        "training_seed": args.training_seed,
        "formal_training_seeds": list(FORMAL_TRAINING_SEEDS),
        "methods": list(FORMAL_METHODS),
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "transitions": len(dataset),
            "selection": dataset.metadata["selection"],
        },
        "koopman_corpus": args.koopman_selection_kind,
        "training": {
            "offline_updates": FORMAL_OFFLINE_UPDATES,
            "online_transitions": FORMAL_ONLINE_TRANSITIONS,
            "offline_eval_interval_updates": FORMAL_OFFLINE_EVAL_INTERVAL,
            "online_eval_interval_transitions": FORMAL_ONLINE_EVAL_INTERVAL,
            "diagnostic_episodes": FORMAL_DIAGNOSTIC_EPISODES,
            "koopman_horizon": FORMAL_KOOPMAN_HORIZON,
            "kmpc_horizon": FORMAL_KMPC_HORIZON,
            "mpve_horizon": FORMAL_MPVE_HORIZON,
            "offline_actor_learning_rate": args.offline_actor_learning_rate,
            "offline_critic_learning_rate": args.offline_critic_learning_rate,
            "online_actor_learning_rate": 3e-4,
            "online_critic_learning_rate": 3e-4,
        },
        "protocol": "tdmpc2_action_repeat2_v1",
    }
    _atomic_json(seed_dir / "seed_plan.json", plan)
    if koopman is not None:
        _atomic_json(
            seed_dir / "protocol.json",
            {
                **plan,
                "koopman": {"path": str(koopman.path), "sha256": koopman.sha256},
            },
        )

    output = seed_dir / args.method
    command = training_command(
        method=args.method,
        seed=args.training_seed,
        dataset=dataset,
        koopman=koopman,
        output=output,
        device=args.device,
        offline_actor_learning_rate=args.offline_actor_learning_rate,
        offline_critic_learning_rate=args.offline_critic_learning_rate,
        initialize_from_offline_final=args.initialize_from_offline_final,
    )
    print(shlex.join(command), flush=True)
    if not args.launch:
        return
    output.mkdir(parents=True, exist_ok=True)
    session = training_session_name(args.run_root, args.training_seed, args.method)
    exists = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not exists:
        log_path = output / "training.log"
        shell_command = f"{shlex.join(command)} >> {shlex.quote(str(log_path))} 2>&1"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, shell_command],
            check=True,
        )
    print(f"tmux session: {session}")


if __name__ == "__main__":
    main()
