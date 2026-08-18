from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from experiments.dmc.o2o.aggregate import (
    AGGREGATE_KIND,
    MATRIX_MANIFEST_KIND,
    aggregate_runs,
    curve_metrics,
    formal_evaluation_grid,
)
from experiments.dmc.o2o.checkpoint import (
    CHECKPOINT_KIND,
    atomic_torch_save,
    load_checkpoint,
)
from experiments.dmc.o2o.config import METHODS, O2OConfig
from experiments.dmc.o2o.dataset import (
    DATASET_KIND,
    OfflineDataset,
    OnlineReplay,
    _cartpole_reward,
)
from experiments.dmc.o2o.evaluate import (
    EVALUATION_EPISODES,
    EVALUATION_KIND,
    evaluate_checkpoint,
    validate_run_identity,
)
from experiments.dmc.o2o.koopman import FrozenKoopman, file_sha256
from experiments.dmc.o2o.learner import O2OLearner
from experiments.dmc.o2o.networks import FrozenObservationNormalizer
from experiments.dmc.o2o.plot import plot_aggregate, plot_offline_curves


PROTOCOL = {
    "protocol_name": "dmc_native_v1",
    "task": "cartpole_swingup",
    "obs_dim": 5,
    "action_dim": 1,
    "step_limit": 2,
    "control_dt": 0.01,
}


def _write_koopman(path: Path) -> Path:
    state_dim, action_dim, lift_dim = 5, 1, 2
    lifted_dim = state_dim + lift_dim
    metadata = {
        "kind": "playground_koopman_export_v1",
        "architecture": {
            "architecture": "fullA_history_v2_adapted",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "lift_dim": lift_dim,
            "hidden_dims": [],
            "activation": "silu",
        },
        "encoder_layer_count": 1,
        "reward_layer_count": 0,
        "best_validation_rollout_normalized_mse": 0.02,
    }
    np.savez(
        path,
        A=np.eye(lifted_dim, dtype=np.float32),
        B=np.zeros((lifted_dim, action_dim), dtype=np.float32),
        C=np.concatenate(
            (
                np.eye(state_dim, dtype=np.float32),
                np.zeros((state_dim, lift_dim), dtype=np.float32),
            ),
            axis=1,
        ),
        center=np.zeros(state_dim, dtype=np.float32),
        scale=np.ones(state_dim, dtype=np.float32),
        encoder_0_weight=np.zeros((lift_dim, state_dim), dtype=np.float32),
        encoder_0_bias=np.zeros(lift_dim, dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    return path


def _write_dataset(path: Path) -> OfflineDataset:
    observation = np.asarray(
        [[0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    next_observation = np.asarray(
        [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    action = np.zeros((2, 1), dtype=np.float32)
    reward = _cartpole_reward(next_observation, action)
    metadata = {
        "kind": DATASET_KIND,
        "task": "cartpole_swingup",
        "transitions": 2,
        "episodes": 1,
        "gamma_for_mc_return": 0.99,
    }
    np.savez_compressed(
        path,
        observation=observation,
        action=action,
        reward=reward,
        discount=np.ones(2, dtype=np.float32),
        next_observation=next_observation,
        episode_id=np.zeros(2, dtype=np.int64),
        episode_step=np.asarray([0, 1], dtype=np.int32),
        mc_return=np.asarray([reward[0] + 0.99 * reward[1], reward[1]]),
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    return OfflineDataset.load(path)


def _config(method: str, seed: int) -> O2OConfig:
    return O2OConfig(
        method=method,
        seed=seed,
        device="cpu",
        hidden_dim=12,
        critic_hidden_layers=2 if method == "Cal-QL-Raw" else 1,
        critic_ensemble_size=2,
        target_critic_subset=2,
        offline_updates=1,
        cql_actions=2,
        online_steps=50_000,
        online_utd=1,
        online_warmup_steps=1,
        replay_capacity=8,
        num_envs=1,
        env_workers=1,
        kmpc_horizon=3,
        kmpc_solver_iterations=2,
        controller_hidden_dim=8,
        mpve_total_horizon=3,
        eval_interval_online_steps=5_000,
        eval_episodes=10,
        checkpoint_interval_updates=1,
        log_interval_updates=1,
    )


def _row(phase: str, step: int, value: float, **extra: Any) -> dict[str, Any]:
    return {
        "phase": phase,
        "offline_update": 1,
        "online_step": step,
        "return_mean": value,
        "return_std_population": 0.0,
        "returns": [value] * EVALUATION_EPISODES,
        **extra,
    }


def _write_run(
    root: Path,
    *,
    config: O2OConfig,
    dataset: OfflineDataset,
    koopman_path: Path,
    eval_values: tuple[float, float, float],
    include_learner: bool = False,
    initialization: dict[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True)
    koopman = FrozenKoopman(koopman_path) if config.requires_koopman else None
    normalizer = (
        None
        if config.requires_koopman
        else FrozenObservationNormalizer.from_offline_observations(
            dataset.arrays["observation"], dataset_sha256=dataset.sha256
        )
    )
    learner = O2OLearner(
        config,
        koopman,
        torch.device("cpu"),
        observation_normalizer=normalizer,
    )
    learner_state: dict[str, Any] = learner.state_dict()
    koopman_identity = koopman.identity() if koopman is not None else None
    normalizer_identity = normalizer.identity() if normalizer is not None else None
    checkpoint = {
        "kind": CHECKPOINT_KIND,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "metadata": dataset.metadata,
        },
        "koopman": koopman_identity,
        "raw_observation_normalizer": normalizer_identity,
        "environment_protocol": PROTOCOL,
        "phase": "online",
        "offline_update": 1,
        "online_step": config.online_steps,
        "online_episode": config.online_steps // 1_000,
        "best_return": eval_values[-1],
        "best_online_step": config.online_steps,
        "initialization": initialization,
        "learner": learner_state,
        "online_replay": OnlineReplay(8).state_dict(),
    }
    atomic_torch_save(root / "latest.pt", checkpoint)
    atomic_torch_save(root / "best.pt", checkpoint)
    run_metadata = {
        "kind": "acmpc_dmc_o2o_run_v1",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "dataset": {"path": str(dataset.path), "sha256": dataset.sha256},
        "koopman": koopman_identity,
        "raw_observation_normalizer": normalizer_identity,
        "environment_protocol": PROTOCOL,
        "initialization": initialization,
        "completed": True,
        "online_steps_completed": config.online_steps,
    }
    (root / "run.json").write_text(
        json.dumps(run_metadata, sort_keys=True), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = [
        _row("initial", 0, 1.0),
        _row("offline_evaluation", 0, eval_values[0]),
        {
            "phase": "online_episode",
            "offline_update": 1,
            "online_step": 1_000,
            "episode": 0,
            "episode_return": eval_values[0] + 20.0,
            "episode_length": 1_000,
        },
        {
            "phase": "online_episode",
            "offline_update": 1,
            "online_step": 2_000,
            "episode": 1,
            "episode_return": eval_values[1],
            "episode_length": 1_000,
        },
    ]
    grid = (1_000, 2_500, *range(5_000, config.online_steps + 1, 5_000))
    midpoint = config.online_steps // 2
    for step in grid:
        if step <= midpoint:
            value = float(
                np.interp(step, (0, midpoint), (eval_values[0], eval_values[1]))
            )
        else:
            value = float(
                np.interp(
                    step,
                    (midpoint, config.online_steps),
                    (eval_values[1], eval_values[2]),
                )
            )
        rows.append(_row("online_evaluation", step, value))
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def _write_offline_fork_source(
    path: Path,
    *,
    seed: int,
    dataset: OfflineDataset,
    koopman_path: Path,
) -> dict[str, Any]:
    """Create the immutable paired AC-KMPC snapshot expected by MPVE."""

    path.parent.mkdir(parents=True, exist_ok=True)
    config = _config("Cal-RLPD-AC-KMPC", seed)
    koopman = FrozenKoopman(koopman_path)
    checkpoint = {
        "kind": CHECKPOINT_KIND,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "metadata": dataset.metadata,
        },
        "koopman": koopman.identity(),
        "environment_protocol": PROTOCOL,
        "phase": "offline",
        "offline_update": config.offline_updates,
        "online_step": 0,
        "online_episode": 0,
        "initialization": None,
    }
    atomic_torch_save(path, checkpoint)
    resolved = path.resolve()
    return {
        "kind": "acmpc_o2o_offline_fork_v1",
        "source_path": str(resolved),
        "source_sha256": file_sha256(resolved),
        "source_method": config.method,
        "source_config_fingerprint": config.fingerprint,
        "shared_state": "actor_critic_target_temperature_optimizers_rng",
    }


def _replace_initialization(root: Path, initialization: dict[str, Any]) -> None:
    checkpoint = load_checkpoint(root / "latest.pt")
    checkpoint["initialization"] = initialization
    atomic_torch_save(root / "latest.pt", checkpoint)
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    metadata["initialization"] = initialization
    (root / "run.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def _write_formal_matrix(
    root: Path,
    *,
    dataset: OfflineDataset,
    koopman_path: Path,
    seeds: tuple[int, ...] = (11, 12),
) -> tuple[list[Path], dict[int, dict[str, Any]]]:
    from experiments.dmc.o2o.config import METHODS

    run_dirs: list[Path] = []
    initializations: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        for method in METHODS:
            if method == "Cal-RLPD-AC-KMPC-MPVE":
                continue
            run_dir = _write_run(
                root / method / f"seed_{seed}",
                config=_config(method, seed),
                dataset=dataset,
                koopman_path=koopman_path,
                eval_values=(100.0, 500.0, 900.0),
            )
            run_dirs.append(run_dir)
            if method == "Cal-RLPD-AC-KMPC":
                initializations[seed] = _write_offline_fork_source(
                    run_dir / "offline.pt",
                    seed=seed,
                    dataset=dataset,
                    koopman_path=koopman_path,
                )
        run_dirs.append(
            _write_run(
                root / "Cal-RLPD-AC-KMPC-MPVE" / f"seed_{seed}",
                config=_config("Cal-RLPD-AC-KMPC-MPVE", seed),
                dataset=dataset,
                koopman_path=koopman_path,
                eval_values=(100.0, 500.0, 900.0),
                initialization=initializations[seed],
            )
        )
    return run_dirs, initializations


def _write_matrix_manifest(
    path: Path,
    *,
    run_dirs: list[Path],
    dataset: OfflineDataset,
    koopman_path: Path,
    seeds: tuple[int, ...],
) -> Path:
    """Write the minimal immutable runner manifest consumed by aggregation."""

    repo_root = path.parent
    source = repo_root / "frozen_source.py"
    source.write_text("SOURCE_IDENTITY = 1\n", encoding="utf-8")
    source_sha = file_sha256(source)
    entries = [{"path": source.name, "sha256": source_sha}]
    snapshot_sha = hashlib.sha256(
        json.dumps(
            {"files": entries}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    snapshot = {"files": entries, "sha256": snapshot_sha}
    configs: dict[str, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        config = metadata["config"]
        method = config["method"]
        configs.setdefault(method, config)
        jobs.append(
            {
                "stage": "train",
                "method": method,
                "seed": config["seed"],
                "cwd": str(repo_root),
                "outputs": [str((run_dir / "latest.pt").resolve())],
            }
        )
    payload = {
        "kind": MATRIX_MANIFEST_KIND,
        "matrix_fingerprint": "1" * 64,
        "experiment": {
            "methods": list(METHODS),
            "seeds": list(seeds),
            "online_steps": 50_000,
            "extend_online_steps": None,
            "evaluation_grid_online_steps": list(formal_evaluation_grid(50_000)),
            "dataset": {"sha256": dataset.sha256},
            "koopman": {"sha256": file_sha256(koopman_path)},
            "resolved_method_configs": configs,
        },
        "source_identity": {
            "training_source": snapshot,
            "result_source": snapshot,
            "runner": {"path": source.name, "sha256": source_sha},
            "git": {"commit": "test"},
        },
        "jobs": jobs,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


class _FakeDMC:
    step_limit = 2
    action_low = np.asarray([-1.0], dtype=np.float32)
    action_high = np.asarray([1.0], dtype=np.float32)

    def __init__(self, _task: str, seed: int) -> None:
        self.seed = seed
        self.step_count = 0
        self.closed = False

    def protocol_metadata(self) -> dict[str, Any]:
        return dict(PROTOCOL)

    def reset(self, seed: int) -> np.ndarray:
        self.seed = seed
        self.step_count = 0
        return np.asarray([0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def step(self, action: np.ndarray):
        assert action.shape == (1,)
        self.step_count += 1
        done = self.step_count == self.step_limit
        observation = np.asarray(
            [0.0, float(self.step_count - 1), 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        return observation, 0.5, done, {"discount": 1.0}

    def close(self) -> None:
        self.closed = True


def test_evaluate_checkpoint_uses_fixed_ten_deterministic_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _write_dataset(tmp_path / "dataset.npz")
    koopman_path = _write_koopman(tmp_path / "koopman.npz")
    run_dir = _write_run(
        tmp_path / "run",
        config=_config("RLPD-Raw", 11),
        dataset=dataset,
        koopman_path=koopman_path,
        eval_values=(100.0, 500.0, 900.0),
        include_learner=True,
    )
    from experiments.dmc.o2o import evaluate as evaluation_module

    seeds: list[int] = []

    def factory(task: str, seed: int):
        seeds.append(seed)
        return _FakeDMC(task, seed)

    monkeypatch.setattr(evaluation_module, "make_dmc_adapter", factory)
    result = evaluate_checkpoint(run_dir, checkpoint_name="latest")

    assert result["kind"] == EVALUATION_KIND
    assert result["evaluation_protocol"]["deterministic"] is True
    assert result["evaluation_protocol"]["episodes"] == 10
    assert seeds == list(range(9_100_000, 9_100_010))
    assert result["returns"] == [1.0] * 10
    assert result["return_mean"] == pytest.approx(1.0)
    assert result["episode_lengths"] == [2] * 10


def test_run_identity_rejects_run_checkpoint_dataset_disagreement(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset.npz")
    koopman_path = _write_koopman(tmp_path / "koopman.npz")
    run_dir = _write_run(
        tmp_path / "run",
        config=_config("RLPD-Raw", 11),
        dataset=dataset,
        koopman_path=koopman_path,
        eval_values=(100.0, 500.0, 900.0),
    )
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    metadata["dataset"]["sha256"] = "0" * 64
    (run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset identities differ"):
        validate_run_identity(run_dir)


def test_curve_metrics_reports_auc_over_1000_and_cumulative_regret() -> None:
    metrics = curve_metrics(
        [
            {"online_step": 0, "return_mean": 100.0},
            {"online_step": 50_000, "return_mean": 500.0},
            {"online_step": 100_000, "return_mean": 900.0},
        ]
    )
    assert metrics["auc_return_steps"] == pytest.approx(50_000_000.0)
    assert metrics["auc_over_1000"] == pytest.approx(50_000.0)
    assert metrics["normalized_auc"] == pytest.approx(0.5)
    assert metrics["return_at_budget"] == pytest.approx(900.0)
    assert metrics["final_return"] == pytest.approx(900.0)
    assert metrics["cumulative_regret_return_steps"] == pytest.approx(50_000_000.0)
    assert metrics["cumulative_regret_over_1000"] == pytest.approx(50_000.0)


def test_formal_grid_includes_early_points_and_scales_with_common_budget() -> None:
    assert formal_evaluation_grid(50_000) == (
        0,
        1_000,
        2_500,
        5_000,
        10_000,
        15_000,
        20_000,
        25_000,
        30_000,
        35_000,
        40_000,
        45_000,
        50_000,
    )
    assert formal_evaluation_grid(60_000)[-1] == 60_000
    with pytest.raises(ValueError, match="multiple of 5k"):
        formal_evaluation_grid(52_500)


def test_aggregate_uses_training_seeds_and_plot_writes_png_pdf(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    dataset = _write_dataset(tmp_path / "dataset.npz")
    koopman_path = _write_koopman(tmp_path / "koopman.npz")
    specifications = (
        ("RLPD-Raw", 11, (100.0, 500.0, 900.0)),
        ("RLPD-Raw", 12, (200.0, 600.0, 1000.0)),
        ("Cal-RLPD-AC-KMPC", 11, (300.0, 700.0, 900.0)),
        ("Cal-RLPD-AC-KMPC", 12, (400.0, 800.0, 1000.0)),
    )
    run_dirs = [
        _write_run(
            tmp_path / method / f"seed_{seed}",
            config=_config(method, seed),
            dataset=dataset,
            koopman_path=koopman_path,
            eval_values=values,
        )
        for method, seed, values in specifications
    ]

    result = aggregate_runs(run_dirs, require_complete=False)

    assert result["kind"] == AGGREGATE_KIND
    assert result["formal_complete"] is False
    assert result["authorization"]["source_verified"] is False
    assert result["authorization"]["formal_complete"] is False
    rpld = result["methods"]["RLPD-Raw"]
    assert rpld["training_seeds"] == [11, 12]
    assert rpld["metrics"]["step0_return"]["mean"] == pytest.approx(150.0)
    assert rpld["metrics"]["auc_over_1000"]["mean"] == pytest.approx(27_500.0)
    assert rpld["metrics"]["normalized_auc"]["mean"] == pytest.approx(0.55)
    assert rpld["metrics"]["return_at_budget"]["mean"] == pytest.approx(950.0)
    assert rpld["metrics"]["final_return"]["mean"] == pytest.approx(950.0)
    assert rpld["metrics"]["cumulative_regret_over_1000"]["mean"] == pytest.approx(
        22_500.0
    )
    assert rpld["metrics"]["final_return"]["inference_axis"] == "training_seed"
    assert rpld["metrics"]["final_return"]["ci95_distribution"] == "student_t_df_1"
    assert rpld["metrics"]["final_return"]["ci95_half_width"] == pytest.approx(
        12.7062047364 * 50.0
    )
    assert len(rpld["evaluation_curve"]) == 13

    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(result), encoding="utf-8")
    png_path, pdf_path = plot_aggregate(aggregate_path, tmp_path / "curves")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF")


def _write_offline_run(root: Path, method: str) -> Path:
    """Write a minimal offline-only run (run.json + metrics.jsonl)."""

    root.mkdir(parents=True)
    (root / "run.json").write_text(
        json.dumps(
            {
                "kind": "acmpc_dmc_o2o_run_v1",
                "config": {"method": method},
                "completed": True,
                "execution_scope": "offline_only",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for step, value in ((10_000, 100.0), (20_000, 300.0), (30_000, 500.0)):
        rows.append(
            {
                "phase": "offline_diagnostic",
                "offline_update": step,
                "online_step": 0,
                "return_mean": value,
                "return_std_population": 1.0,
                "returns": [value] * EVALUATION_EPISODES,
            }
        )
    rows.append(
        {
            "phase": "offline_evaluation",
            "offline_update": 50_000,
            "online_step": 0,
            "return_mean": 700.0,
            "return_std_population": 2.0,
            "returns": [700.0] * EVALUATION_EPISODES,
        }
    )
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def test_plot_offline_curves_writes_png_and_csv(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    first = _write_offline_run(tmp_path / "Cal-QL-Raw", "Cal-QL-Raw")
    second = _write_offline_run(tmp_path / "kmpc", "Cal-QL-AC-KMPC")
    png_path, csv_path = plot_offline_curves(
        [first, second], tmp_path / "offline_curves"
    )
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    csv = csv_path.read_text(encoding="utf-8")
    assert (
        csv.splitlines()[0]
        == "method,offline_update,return_mean,return_std_population"
    )
    # The historical internal name is normalized for display and CSV.
    assert "Cal-QL-AC-KMPC" not in csv
    assert "Cal-QL-KMPC,10000,100.0,1.0" in csv
    assert "Cal-QL-Raw,50000,700.0,2.0" in csv


def test_aggregate_rejects_method_specific_config_drift_across_seeds(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset.npz")
    koopman_path = _write_koopman(tmp_path / "koopman.npz")
    run_dirs = [
        _write_run(
            tmp_path / "RLPD-Raw" / "seed_11",
            config=_config("RLPD-Raw", 11),
            dataset=dataset,
            koopman_path=koopman_path,
            eval_values=(100.0, 500.0, 900.0),
        ),
        _write_run(
            tmp_path / "RLPD-Raw" / "seed_12",
            config=dataclasses.replace(
                _config("RLPD-Raw", 12), actor_learning_rate=1e-4
            ),
            dataset=dataset,
            koopman_path=koopman_path,
            eval_values=(100.0, 500.0, 900.0),
        ),
    ]
    with pytest.raises(ValueError, match="method-specific configs"):
        aggregate_runs(run_dirs, require_complete=False)


def test_formal_aggregate_records_and_verifies_matrix_source_identity(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset.npz")
    koopman_path = _write_koopman(tmp_path / "koopman.npz")
    run_dirs, _initializations = _write_formal_matrix(
        tmp_path / "matrix_runs",
        dataset=dataset,
        koopman_path=koopman_path,
    )
    manifest = _write_matrix_manifest(
        tmp_path / "matrix_manifest.json",
        run_dirs=run_dirs,
        dataset=dataset,
        koopman_path=koopman_path,
        seeds=(11, 12),
    )

    result = aggregate_runs(run_dirs, matrix_manifest=manifest)
    assert result["formal_complete"] is True
    assert result["authorization"]["source_verified"] is True
    assert result["authorization"]["formal_complete"] is True
    identity = result["shared_identity"]["matrix_source_identity"]
    assert identity["matrix_fingerprint"] == "1" * 64
    assert identity["training_source"]["files"][0]["path"] == "frozen_source.py"

    (tmp_path / "frozen_source.py").write_text(
        "SOURCE_IDENTITY = 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="training_source source file differs"):
        aggregate_runs(run_dirs, matrix_manifest=manifest)


def test_formal_aggregate_requires_full_matrix_and_identical_seed_sets(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset.npz")
    koopman_path = _write_koopman(tmp_path / "koopman.npz")
    partial = [
        _write_run(
            tmp_path / "partial" / f"seed_{seed}",
            config=_config("RLPD-Raw", seed),
            dataset=dataset,
            koopman_path=koopman_path,
            eval_values=(100.0, 500.0, 900.0),
        )
        for seed in (11, 12)
    ]
    with pytest.raises(ValueError, match="complete five-method matrix"):
        aggregate_runs(partial)

    run_dirs = []
    from experiments.dmc.o2o.config import METHODS

    mpve_initializations = {
        seed: _write_offline_fork_source(
            tmp_path / "mpve_sources" / f"seed_{seed}" / "offline.pt",
            seed=seed,
            dataset=dataset,
            koopman_path=koopman_path,
        )
        for seed in (11, 13)
    }
    for method in METHODS:
        seeds = (11, 13) if method == METHODS[-1] else (11, 12)
        for seed in seeds:
            run_dirs.append(
                _write_run(
                    tmp_path / "mismatch" / method / f"seed_{seed}",
                    config=_config(method, seed),
                    dataset=dataset,
                    koopman_path=koopman_path,
                    eval_values=(100.0, 500.0, 900.0),
                    initialization=(
                        mpve_initializations[seed]
                        if method == "Cal-RLPD-AC-KMPC-MPVE"
                        else None
                    ),
                )
            )
    with pytest.raises(ValueError, match="same ordered training-seed set"):
        aggregate_runs(run_dirs)


def test_formal_aggregate_requires_each_mpve_fork_from_included_same_seed_ac_run(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset.npz")
    koopman_path = _write_koopman(tmp_path / "koopman.npz")
    run_dirs, initializations = _write_formal_matrix(
        tmp_path / "matrix",
        dataset=dataset,
        koopman_path=koopman_path,
    )

    result = aggregate_runs(run_dirs)
    assert result["protocol"]["common_training_seeds"] == [11, 12]

    mpve_run = tmp_path / "matrix" / "Cal-RLPD-AC-KMPC-MPVE" / "seed_12"
    tampered_sha = dict(initializations[12])
    tampered_sha["source_sha256"] = "0" * 64
    _replace_initialization(mpve_run, tampered_sha)
    with pytest.raises(ValueError, match="fork lineage"):
        aggregate_runs(run_dirs)

    missing_path = dict(initializations[12])
    missing_path["source_path"] = str(
        (tmp_path / "missing" / "seed_12" / "offline.pt").resolve()
    )
    _replace_initialization(mpve_run, missing_path)
    with pytest.raises(FileNotFoundError, match="fork source is missing"):
        aggregate_runs(run_dirs)

    alternate = _write_offline_fork_source(
        tmp_path / "alternate" / "seed_12" / "offline.pt",
        seed=12,
        dataset=dataset,
        koopman_path=koopman_path,
    )
    _replace_initialization(mpve_run, alternate)
    with pytest.raises(ValueError, match="included same-seed AC-KMPC run"):
        aggregate_runs(run_dirs)
