from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from experiments.dmc.o2o.aggregate import (
    AGGREGATE_KIND,
    aggregate_runs,
    curve_metrics,
)
from experiments.dmc.o2o.checkpoint import CHECKPOINT_KIND, atomic_torch_save
from experiments.dmc.o2o.config import O2OConfig
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
from experiments.dmc.o2o.koopman import FrozenKoopman
from experiments.dmc.o2o.learner import O2OLearner
from experiments.dmc.o2o.plot import plot_aggregate


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
        critic_hidden_layers=1,
        critic_ensemble_size=2,
        target_critic_subset=2,
        offline_updates=1,
        cql_actions=2,
        online_steps=100_000,
        online_utd=1,
        online_warmup_steps=1,
        replay_capacity=8,
        num_envs=1,
        env_workers=1,
        kmpc_horizon=3,
        kmpc_solver_iterations=2,
        controller_hidden_dim=8,
        mpve_total_horizon=3,
        eval_interval_online_steps=50_000,
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
) -> Path:
    root.mkdir(parents=True)
    koopman = FrozenKoopman(koopman_path)
    learner_state: dict[str, Any]
    if include_learner:
        learner = O2OLearner(config, koopman, torch.device("cpu"))
        learner_state = learner.state_dict()
    else:
        learner_state = {}
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
        "phase": "online",
        "offline_update": 1,
        "online_step": 100_000,
        "online_episode": 100,
        "best_return": eval_values[-1],
        "best_online_step": 100_000,
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
        "koopman": koopman.identity(),
        "environment_protocol": PROTOCOL,
        "completed": True,
        "online_steps_completed": 100_000,
    }
    (root / "run.json").write_text(
        json.dumps(run_metadata, sort_keys=True), encoding="utf-8"
    )
    rows = [
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
        _row("online_evaluation", 50_000, eval_values[1]),
        _row("online_evaluation", 100_000, eval_values[2]),
    ]
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


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
        config=_config("RLPD-MLP", 11),
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
        config=_config("RLPD-MLP", 11),
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
    assert metrics["return_at_100k"] == pytest.approx(900.0)
    assert metrics["final_return"] == pytest.approx(900.0)
    assert metrics["cumulative_regret_return_steps"] == pytest.approx(50_000_000.0)
    assert metrics["cumulative_regret_over_1000"] == pytest.approx(50_000.0)


def test_aggregate_uses_training_seeds_and_plot_writes_png_pdf(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    dataset = _write_dataset(tmp_path / "dataset.npz")
    koopman_path = _write_koopman(tmp_path / "koopman.npz")
    specifications = (
        ("RLPD-MLP", 11, (100.0, 500.0, 900.0)),
        ("RLPD-MLP", 12, (200.0, 600.0, 1000.0)),
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
    rpld = result["methods"]["RLPD-MLP"]
    assert rpld["training_seeds"] == [11, 12]
    assert rpld["metrics"]["step0_return"]["mean"] == pytest.approx(150.0)
    assert rpld["metrics"]["auc_over_1000"]["mean"] == pytest.approx(55_000.0)
    assert rpld["metrics"]["normalized_auc"]["mean"] == pytest.approx(0.55)
    assert rpld["metrics"]["return_at_100k"]["mean"] == pytest.approx(950.0)
    assert rpld["metrics"]["final_return"]["mean"] == pytest.approx(950.0)
    assert rpld["metrics"]["cumulative_regret_over_1000"]["mean"] == pytest.approx(
        45_000.0
    )
    assert rpld["metrics"]["final_return"]["inference_axis"] == "training_seed"
    assert rpld["metrics"]["final_return"]["ci95_distribution"] == "student_t_df_1"
    assert rpld["metrics"]["final_return"]["ci95_half_width"] == pytest.approx(
        12.7062047364 * 50.0
    )
    assert len(rpld["evaluation_curve"]) == 3

    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(result), encoding="utf-8")
    png_path, pdf_path = plot_aggregate(aggregate_path, tmp_path / "curves")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_formal_aggregate_requires_full_matrix_and_identical_seed_sets(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset.npz")
    koopman_path = _write_koopman(tmp_path / "koopman.npz")
    partial = [
        _write_run(
            tmp_path / "partial" / f"seed_{seed}",
            config=_config("RLPD-MLP", seed),
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
                )
            )
    with pytest.raises(ValueError, match="same ordered training-seed set"):
        aggregate_runs(run_dirs)
