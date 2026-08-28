from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("h5py")

from experiments.maniskill_pick_visual.run_ablation import (
    ABLATION_SPECS,
    AblationConfig,
    _train_config,
    run_ablation,
)


def _inputs(tmp_path):
    trajectory = tmp_path / "trajectory.h5"
    features = tmp_path / "features.h5"
    trajectory.touch()
    features.touch()
    return trajectory, features


def _report(config, index):
    observable = (0.4, 0.1, 0.3, 0.2)[index % 4]
    horizons = {
        str(step): {
            "robot_rmse": index + step / 100.0,
            "tcp_rmse_mm": index + step / 10.0,
        }
        for step in (1, 5, 10, 20)
    }
    return {
        "best_epoch": index + 2,
        "best_validation_observable": observable,
        "best_validation_total_at_best": 10.0 - index,
        "best_validation_total": 10.0 - index,
        "test_metrics": {
            "horizons": horizons,
            "one_step_action_ablation": {
                "actual_robot_normalized_rmse": 0.1 + index,
                "shuffled_over_actual": 2.0 + index,
                "zero_over_actual": 3.0 + index,
            },
            "spectral_radius": 0.9 + index / 100.0,
        },
        "elapsed_seconds": 12.5 + index,
        "checkpoint": str(config.output_dir / "best.pt"),
    }


def test_runs_fixed_four_way_matrix_and_writes_summary(tmp_path, monkeypatch):
    trajectory, features = _inputs(tmp_path)
    output = tmp_path / "ablation"
    calls = []
    observed_wandb = []

    def fake_train(config, *, device_name="auto"):
        index = len(calls)
        calls.append((config, device_name))
        observed_wandb.append(
            (
                os.environ.get("WANDB_MODE"),
                os.environ.get("WANDB_PROJECT"),
                os.environ.get("WANDB_RUN_GROUP"),
                os.environ.get("WANDB_NAME"),
            )
        )
        config.output_dir.mkdir(parents=True)
        return _report(config, index)

    monkeypatch.setenv("WANDB_MODE", "online")
    summary = run_ablation(
        AblationConfig(
            trajectory_h5=trajectory,
            feature_h5=features,
            output_dir=output,
            horizon=20,
            epochs=7,
            patience=3,
            batch_size=11,
            seed=123,
            workers=2,
            device="cpu",
            preload=True,
            wandb_offline=True,
            wandb_project="unit-project",
            wandb_group="four-way",
        ),
        trainer=fake_train,
    )

    expected_names = [spec.name for spec in ABLATION_SPECS]
    assert summary["run_order"] == expected_names
    assert [run["name"] for run in summary["runs"]] == expected_names
    assert len(calls) == 4
    for (config, device_name), spec in zip(calls, ABLATION_SPECS, strict=True):
        assert config.trajectory_h5 == trajectory.resolve()
        assert config.feature_h5 == features.resolve()
        assert config.output_dir == (output / spec.name).resolve()
        assert config.visual_latent_dim == spec.visual_latent_dim
        assert config.transform_mode == spec.train_transform_mode
        assert config.horizon == 20
        assert config.epochs == 7
        assert config.patience == 3
        assert config.batch_size == 11
        assert config.seed == 123
        assert config.workers == 2
        assert config.wandb_mode == "offline"
        assert config.wandb_group == "four-way"
        assert device_name == "cpu"

    assert observed_wandb == [
        ("offline", "unit-project", "four-way", spec.name)
        for spec in ABLATION_SPECS
    ]
    assert os.environ["WANDB_MODE"] == "online"
    assert summary["best_run_by_observable"] == expected_names[1]
    assert summary["best_run_by_validation"] == expected_names[1]
    for index, run in enumerate(summary["runs"]):
        assert set(run["test_horizons"]) == {"1", "5", "10", "20"}
        assert run["action_ablation"]["shuffled_over_actual"] == 2.0 + index
        assert run["visual_latent_dim"] in {16, 32}
        assert run["spectral_radius"] == pytest.approx(0.9 + index / 100.0)
        assert run["elapsed_seconds"] == 12.5 + index
        assert run["best_validation_observable"] == pytest.approx(
            (0.4, 0.1, 0.3, 0.2)[index]
        )
        assert run["best_validation_total_at_best"] == pytest.approx(
            10.0 - index
        )
        assert run["best_validation_total"] == pytest.approx(10.0 - index)
        assert run["compatibility"]["validation_metric_source"] == (
            "best_validation_observable"
        )
        # The runner passes preload when the current TrainConfig exposes it and
        # records the effective value for compatibility with older revisions.
        assert run["compatibility"]["requested_preload"] is True
        assert run["compatibility"]["effective_preload"] is hasattr(
            calls[index][0], "preload"
        )

    summary_path = output / "ablation_summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_ablation_defaults_to_offline_wandb(tmp_path):
    trajectory, features = _inputs(tmp_path)
    config = AblationConfig(trajectory, features, tmp_path / "ablation")

    train_config, _ = _train_config(config.validated(), ABLATION_SPECS[0])

    assert config.wandb_offline is True
    assert train_config.wandb_mode == "offline"


def test_preflight_refuses_any_existing_run_without_calling_train(tmp_path):
    trajectory, features = _inputs(tmp_path)
    output = tmp_path / "ablation"
    (output / ABLATION_SPECS[2].name).mkdir(parents=True)
    calls = 0

    def fake_train(config, *, device_name="auto"):
        nonlocal calls
        calls += 1
        return _report(config, calls)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_ablation(
            AblationConfig(trajectory, features, output),
            trainer=fake_train,
        )
    assert calls == 0
    assert not (output / "ablation_summary.json").exists()


def test_failed_run_does_not_publish_complete_summary(tmp_path):
    trajectory, features = _inputs(tmp_path)
    output = tmp_path / "ablation"
    calls = 0

    def failing_train(config, *, device_name="auto"):
        nonlocal calls
        calls += 1
        config.output_dir.mkdir(parents=True)
        if calls == 2:
            raise RuntimeError("mock training failure")
        return _report(config, calls - 1)

    with pytest.raises(RuntimeError, match="mock training failure"):
        run_ablation(
            AblationConfig(trajectory, features, output),
            trainer=failing_train,
        )
    assert calls == 2
    assert not (output / "ablation_summary.json").exists()


def test_summary_uses_null_for_unavailable_requested_horizon(tmp_path):
    trajectory, features = _inputs(tmp_path)
    output = tmp_path / "ablation"

    def short_report(config, *, device_name="auto"):
        config.output_dir.mkdir(parents=True)
        report = _report(config, 0)
        del report["test_metrics"]["horizons"]["20"]
        return report

    summary = run_ablation(
        AblationConfig(trajectory, features, output),
        trainer=short_report,
    )
    assert all(run["test_horizons"]["20"] is None for run in summary["runs"])
