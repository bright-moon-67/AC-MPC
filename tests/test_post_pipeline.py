import json

import pytest

from scripts.post_koopman_pipeline import require_finite_koopman_report
from scripts.post_ppo_pipeline import (
    completed_training_seeds,
    evaluation_is_complete,
    sha256,
)


def test_post_pipeline_rejects_nonfinite_koopman_metrics(tmp_path):
    report = tmp_path / "evaluation.json"
    payload = {
        "horizons": {
            "1": {
                "samples": 2,
                "koopman_mse": {"all": 1.0},
                "naive_hold_mse": {"all": 2.0},
                "per_step_all_mse": [1.0],
            }
        }
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    require_finite_koopman_report(report)
    payload["horizons"]["1"]["koopman_mse"]["all"] = float("nan")
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FloatingPointError):
        require_finite_koopman_report(report)


def test_post_ppo_pipeline_validates_checkpoint_digest(tmp_path):
    seed_root = tmp_path / "seed_0"
    seed_root.mkdir()
    checkpoint = seed_root / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    (seed_root / "training_status.json").write_text(
        json.dumps(
            {
                "stop_reason": "total_timesteps",
                "last_checkpoint_sha256": sha256(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    assert completed_training_seeds(tmp_path, [0]) == ([0], [])
    checkpoint.write_bytes(b"changed")
    completed, failures = completed_training_seeds(tmp_path, [0])
    assert completed == []
    assert failures[0]["stop_reason"] == "last_checkpoint_sha256_mismatch"


def test_post_ppo_pipeline_requires_all_formal_evaluations(tmp_path):
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        seed_root = tmp_path / f"seed_{seed}"
        seed_root.mkdir()
        checkpoint = seed_root / "last.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        (seed_root / "evaluation_legacy_interval1_100ep.json").write_text(
            json.dumps(
                {
                    "episodes": 100,
                    "resolved_backend": "legacy",
                    "gain_update_interval": 1,
                    "checkpoint_sha256": sha256(checkpoint),
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "aggregate_legacy_interval1_100ep.json").write_text(
        json.dumps(
            {
                "seed_count": 5,
                "training_seeds": seeds,
                "episodes_per_seed": [100] * 5,
            }
        ),
        encoding="utf-8",
    )
    assert evaluation_is_complete(tmp_path, seeds, 100, 1)
    (tmp_path / "seed_4/evaluation_legacy_interval1_100ep.json").unlink()
    assert not evaluation_is_complete(tmp_path, seeds, 100, 1)
