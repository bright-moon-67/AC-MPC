from __future__ import annotations

import builtins
import json
import sys
from types import ModuleType

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256
import experiments.maniskill_pick_visual.train_visual_koopman as training_module
from experiments.maniskill_pick_visual.evaluate_visual_koopman import evaluate
from experiments.maniskill_pick_visual.train_visual_koopman import (
    TrainConfig,
    train,
)


def _write_synthetic_data(trajectory_path, feature_path) -> None:
    rng = np.random.default_rng(11)
    frozen: dict[str, np.ndarray] = {}
    with h5py.File(trajectory_path, "x") as trajectories:
        trajectories.attrs["causal_replay"] = True
        trajectories.attrs["goal_visible"] = True
        trajectories.attrs["control_mode"] = "pd_joint_delta_pos"
        trajectories.attrs["actions_are_applied"] = True
        trajectories.attrs["action_low"] = -np.ones(8, dtype=np.float32)
        trajectories.attrs["action_high"] = np.ones(8, dtype=np.float32)
        for episode in range(6):
            trajectory = trajectories.create_group(f"traj_{episode}")
            actions = rng.uniform(-1, 1, size=(5, 8)).astype(np.float32)
            robot = rng.normal(size=(6, 21)).astype(np.float32)
            frozen_features = rng.normal(size=(6, 512)).astype(np.float32)
            trajectory.create_dataset("robot", data=robot)
            trajectory.create_dataset("actions", data=actions)
            frozen[f"traj_{episode}"] = frozen_features
    with h5py.File(feature_path, "x") as features:
        features.attrs["complete"] = True
        features.attrs["source_sha256"] = sha256(trajectory_path)
        features.attrs["feature_dim"] = 512
        for name, frozen_features in frozen.items():
            feature_group = features.create_group(name)
            feature_group.create_dataset("resnet18", data=frozen_features)


def test_tiny_training_and_evaluation_round_trip(tmp_path, monkeypatch) -> None:
    trajectory_path = tmp_path / "trajectory.h5"
    feature_path = tmp_path / "features.h5"
    output_dir = tmp_path / "run"
    _write_synthetic_data(trajectory_path, feature_path)

    real_import = builtins.__import__

    def reject_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("disabled tracking must not import wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_wandb)

    report = train(
        TrainConfig(
            trajectory_h5=trajectory_path,
            feature_h5=feature_path,
            output_dir=output_dir,
            epochs=2,
            patience=2,
            batch_size=16,
            horizon=2,
            visual_latent_dim=4,
            encoder_hidden_dims=(16,),
            transform_mode="learned_inverse",
            seed=5,
            preload=True,
            wandb_mode="disabled",
        ),
        device_name="cpu",
    )

    checkpoint = output_dir / "best.pt"
    assert checkpoint.is_file()
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert len((output_dir / "metrics.jsonl").read_text().splitlines()) == 2
    assert report["architecture"]["visual_latent_dim"] == 4
    assert report["architecture"]["transform_mode"] == "learned_inverse"
    assert report["data"]["preloaded"] is True
    assert report["summary"]["test_metrics"] == report["test_metrics"]
    assert report["summary"]["test_loss"] == report["test_loss"]
    assert report["data"]["test_evaluation_window_counts"] == {
        "1": 5,
        "2": 4,
        "5": 1,
        "10": 0,
        "20": 0,
    }
    metrics = report["test_metrics"]
    assert metrics["windows"] == 4
    assert metrics["horizons"]["1"]["windows"] == 5
    assert metrics["horizons"]["2"]["windows"] == 4
    assert metrics["horizons"]["5"]["windows"] == 1
    assert metrics["horizons"]["10"] == {"windows": 0, "available": False}
    assert metrics["horizons"]["20"] == {"windows": 0, "available": False}
    one_step = metrics["one_step_action_ablation"]
    assert {
        "actual_robot_normalized_rmse",
        "random_robot_normalized_rmse",
        "zero_robot_normalized_rmse",
    } <= set(one_step)
    assert one_step["derangement"]["whole_action_sequence"] is True
    assert one_step["derangement"]["sequence_steps"] == 1
    assert one_step["derangement"]["pairs"] == 5
    horizon_one = metrics["horizons"]["1"]
    assert "arm_qpos_rmse_rad" in horizon_one
    assert "gripper_qpos_rmse_m" in horizon_one
    assert "arm_qvel_rmse_rad_s" in horizon_one
    assert "gripper_qvel_rmse_m_s" in horizon_one
    assert "qpos_rmse_rad" not in horizon_one
    assert len(metrics["B_column_norms"]) == 8
    assert len(metrics["B_column_norms_by_action"]) == 8
    assert 0 <= metrics["B_numerical_rank"] <= 8
    assert report["validation_selection_metric"] == "observable_validation"
    assert np.isfinite(report["best_validation_observable"])
    assert np.isfinite(report["best_validation_total_at_best"])
    assert (
        report["best_validation_total"]
        == report["best_validation_total_at_best"]
    )
    assert len(report["history"]) == 2
    for epoch in report["history"]:
        assert set(("train", "validation", "lr", "gradient_norm")) <= set(epoch)
        assert "total" in epoch["train"] and "total" in epoch["validation"]
        assert np.isfinite(epoch["validation_observable"])
        assert np.isfinite(epoch["gradient_norm"])
    _, checkpoint_payload = load_checkpoint(checkpoint)
    assert checkpoint_payload["best_validation"] == pytest.approx(
        report["best_validation_observable"]
    )
    result = evaluate(checkpoint, split="test", device_name="cpu")
    assert result["metrics"]["windows"] == 4
    assert {
        horizon: values["windows"]
        for horizon, values in result["metrics"]["horizons"].items()
    } == {"1": 5, "2": 4, "5": 1, "10": 0, "20": 0}
    assert "readout_frobenius_norm" in result["metrics"]


def test_default_horizon_is_twenty() -> None:
    assert TrainConfig.__dataclass_fields__["horizon"].default == 20


def test_observable_validation_uses_only_observable_errors(tmp_path) -> None:
    config = TrainConfig(
        trajectory_h5=tmp_path / "trajectory.h5",
        feature_h5=tmp_path / "features.h5",
        output_dir=tmp_path / "run",
        robot_rollout_weight=3.0,
        feature_reconstruction_weight=5.0,
        future_feature_reconstruction_weight=7.0,
    )
    values = {
        "total": 1234.0,
        "linear": 999.0,
        "robot_rollout": 2.0,
        "feature_reconstruction": 3.0,
        "future_feature_reconstruction": 4.0,
        "transform_condition": 888.0,
    }
    assert training_module.observable_validation(values, config) == pytest.approx(
        3.0 * 2.0 + 5.0 * 3.0 + 7.0 * 4.0
    )


def test_wandb_init_receives_explicit_group(tmp_path, monkeypatch) -> None:
    calls = []
    sentinel_run = object()
    fake_wandb = ModuleType("wandb")

    def fake_init(**kwargs):
        calls.append(kwargs)
        return sentinel_run

    fake_wandb.init = fake_init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = TrainConfig(
        trajectory_h5=tmp_path / "trajectory.h5",
        feature_h5=tmp_path / "features.h5",
        output_dir=tmp_path / "run",
        wandb_mode="offline",
        wandb_project="unit-project",
        wandb_name="unit-run",
        wandb_group="unit-group",
    )

    run = training_module._start_wandb(config, {"seed": 1})

    assert run is sentinel_run
    assert len(calls) == 1
    assert calls[0]["project"] == "unit-project"
    assert calls[0]["name"] == "unit-run"
    assert calls[0]["group"] == "unit-group"
    assert calls[0]["mode"] == "offline"


def test_plateau_scheduler_and_early_stopping(tmp_path, monkeypatch) -> None:
    trajectory_path = tmp_path / "trajectory.h5"
    feature_path = tmp_path / "features.h5"
    output_dir = tmp_path / "plateau"
    _write_synthetic_data(trajectory_path, feature_path)
    original_evaluate = training_module.evaluate_loss

    evaluation_calls = 0

    def constant_observable_with_decreasing_total(*args, **kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        values = original_evaluate(*args, **kwargs)
        values["robot_rollout"] = 0.5
        values["feature_reconstruction"] = 0.0
        values["future_feature_reconstruction"] = 0.0
        # The full objective improves, but the observable metric remains at
        # 1.0. Scheduler and early stopping must follow the latter.
        values["total"] = 10.0 - evaluation_calls
        return values

    monkeypatch.setattr(
        training_module,
        "evaluate_loss",
        constant_observable_with_decreasing_total,
    )
    report = train(
        TrainConfig(
            trajectory_h5=trajectory_path,
            feature_h5=feature_path,
            output_dir=output_dir,
            epochs=5,
            patience=1,
            batch_size=16,
            learning_rate=1e-3,
            lr_factor=0.5,
            lr_patience=0,
            min_lr=1e-5,
            horizon=2,
            visual_latent_dim=4,
            encoder_hidden_dims=(16,),
            seed=7,
            wandb_mode="disabled",
        ),
        device_name="cpu",
    )
    assert report["stopped_early"] is True
    assert report["summary"]["stop_reason"] == "early_stopping"
    assert report["summary"]["epochs_completed"] == 2
    assert report["best_validation_observable"] == pytest.approx(1.0)
    assert report["best_validation_total_at_best"] == pytest.approx(9.0)
    assert report["history"][1]["next_lr"] == pytest.approx(5e-4)
    lines = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(lines) == 2
    assert lines[-1]["next_lr"] == pytest.approx(5e-4)


def test_nonfinite_config_is_rejected_before_creating_run(tmp_path) -> None:
    output_dir = tmp_path / "invalid"
    with pytest.raises(ValueError, match="learning_rate"):
        train(
            TrainConfig(
                trajectory_h5=tmp_path / "missing-trajectory.h5",
                feature_h5=tmp_path / "missing-features.h5",
                output_dir=output_dir,
                learning_rate=float("nan"),
                wandb_mode="disabled",
            ),
            device_name="cpu",
        )
    assert not output_dir.exists()
