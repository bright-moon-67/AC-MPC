from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.dmc.o2o.dataset import _cartpole_reward
from experiments.dmc.o2o import evaluate_koopman as evaluation


def _tiny_protocol() -> evaluation.EvaluationProtocol:
    return evaluation.EvaluationProtocol(
        stage_ranges=(
            ("early", 0, 10),
            ("mid", 10, 20),
            ("late", 20, 30),
        ),
        episode_steps=3,
        horizon_steps=2,
    )


def _write_proto_fixture(
    root: Path, protocol: evaluation.EvaluationProtocol
) -> Path:
    root.mkdir(parents=True)
    canonical = root / "transitions.npz"
    canonical.write_bytes(b"small canonical identity fixture")
    canonical_sha = evaluation.file_sha256(canonical)
    stage_metadata = {}
    for stage_number, (name, left, right) in enumerate(protocol.stage_ranges):
        episodes = right - left
        episode = np.arange(episodes, dtype=np.float32)[:, None]
        step = np.arange(protocol.episode_steps + 1, dtype=np.float32)[None, :]
        phase = 0.07 * (episode + left) + 0.11 * step + 0.03 * stage_number
        states = np.stack(
            (
                0.03 * (episode + left) + 0.02 * step,
                np.cos(phase),
                np.sin(phase),
                0.05 * (episode + 1.0) - 0.01 * step,
                0.08 * step - 0.02 * episode,
            ),
            axis=-1,
        ).astype(np.float32)
        action_step = np.arange(protocol.episode_steps, dtype=np.float32)[None, :]
        actions = np.tanh(
            0.04 * (episode + stage_number) - 0.12 * action_step
        )[..., None].astype(np.float32)
        rewards = _cartpole_reward(states[:, 1:], actions)
        path = (root / f"{name}.npz").resolve()
        np.savez_compressed(path, states=states, actions=actions, rewards=rewards)
        checksum = evaluation.file_sha256(path)
        stage_metadata[name] = {
            "path": str(path),
            "sha256": checksum,
            "episode_id_start_inclusive": left,
            "episode_id_end_exclusive": right,
            "episodes": episodes,
            "states_shape": list(states.shape),
            "actions_shape": list(actions.shape),
            "rewards_shape": list(rewards.shape),
        }
    manifest = {
        "kind": evaluation.DATA_MANIFEST_KIND,
        "task": evaluation.TASK,
        "policy": "synthetic Proto fixture",
        "source_dataset": str(canonical.resolve()),
        "source_dataset_sha256": canonical_sha,
        "canonical_transitions_npz": str(canonical.resolve()),
        "canonical_transitions_npz_sha256": canonical_sha,
        "total_transitions": protocol.total_transitions,
        "episodes": protocol.episodes,
        "stage_episode_counts": protocol.stage_counts,
        "stages": stage_metadata,
        "episode_steps": protocol.episode_steps,
        "observation_dim": protocol.state_dim,
        "action_dim": protocol.action_dim,
        "trainer_episode_split": "per_stage_modulo_10_8_1_1",
        "trainer_split_episode_counts": protocol.split_counts(),
        "reward": evaluation.REWARD_IDENTITY,
        "source_episode_identity_sha256": "0" * 64,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _write_hold_model(
    path: Path,
    source_directory: Path,
    protocol: evaluation.EvaluationProtocol,
) -> Path:
    lift_dim = 1
    lifted_dim = protocol.state_dim + lift_dim
    source_sha = evaluation.file_sha256(source_directory / "manifest.json")
    metadata = {
        "kind": evaluation.MODEL_KIND,
        "task": evaluation.TASK,
        "source_path": str(source_directory.resolve()),
        "source_sha256": source_sha,
        "dataset_sha256": source_sha,
        "data_manifest_sha256": source_sha,
        "k_step": protocol.horizon_steps,
        "seed": 7,
        "best_epoch": 1,
        "best_validation_rollout_normalized_mse": 0.5,
        "architecture": {
            "architecture": "fullA_history_v2_adapted",
            "state_dim": protocol.state_dim,
            "action_dim": protocol.action_dim,
            "lift_dim": lift_dim,
            "hidden_dims": [],
            "activation": "silu",
        },
        "encoder_layer_count": 1,
        "reward_layer_count": 0,
    }
    matrix_c = np.concatenate(
        (
            np.eye(protocol.state_dim, dtype=np.float32),
            np.zeros((protocol.state_dim, lift_dim), dtype=np.float32),
        ),
        axis=1,
    )
    np.savez(
        path,
        A=np.eye(lifted_dim, dtype=np.float32),
        B=np.zeros((lifted_dim, protocol.action_dim), dtype=np.float32),
        C=matrix_c,
        center=np.zeros(protocol.state_dim, dtype=np.float32),
        scale=np.ones(protocol.state_dim, dtype=np.float32),
        encoder_0_weight=np.zeros(
            (lift_dim, protocol.state_dim), dtype=np.float32
        ),
        encoder_0_bias=np.zeros(lift_dim, dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    return path


def test_common_scale_evaluation_uses_every_episode_safe_test_window(
    tmp_path: Path,
) -> None:
    protocol = _tiny_protocol()
    data_dir = _write_proto_fixture(tmp_path / "proto", protocol)
    model_path = _write_hold_model(tmp_path / "hold.npz", data_dir, protocol)

    result = evaluation.evaluate_models(
        data_dir,
        [("hold", model_path)],
        batch_size=2,
        protocol=protocol,
    )

    assert result["kind"] == evaluation.EVALUATION_KIND
    assert result["protocol"]["test_episodes"] == 3
    assert result["protocol"]["test_episodes_by_stage"] == {
        "early": 1,
        "mid": 1,
        "late": 1,
    }
    # Each three-step episode has starts 0 and 1 for an episode-safe H=2 window.
    assert result["protocol"]["windows"] == 6
    assert result["common_train_reference"]["samples"] == 72
    metrics = result["models"]["hold"]["metrics"]
    assert result["protocol"]["reported_prefix_horizons_steps"] == [1, 2]
    assert set(result["models"]["hold"]["metrics_by_horizon"]) == {"1", "2"}
    assert metrics["windows"] == 6
    assert metrics["model_to_hold_mse_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["weighted_rollout_nmse"] > 0
    assert metrics["one_step_nmse"] > 0
    assert metrics["step_2_nmse"] > metrics["one_step_nmse"]
    assert metrics["exact_reward_prediction"]["predictions"] == 12
    assert metrics["exact_reward_prediction"]["uses_learned_reward_model"] is False
    assert result["models"]["hold"]["training"]["source_is_evaluation_proto1m"]


def test_stage_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    protocol = _tiny_protocol()
    data_dir = _write_proto_fixture(tmp_path / "proto", protocol)
    stage_path = data_dir / "early.npz"
    with stage_path.open("ab") as handle:
        handle.write(b"changed after manifest")

    with pytest.raises(ValueError, match="Stage metadata mismatch"):
        evaluation.load_proto_data(data_dir, protocol=protocol)


def test_model_shape_and_source_manifest_lineage_are_strict(tmp_path: Path) -> None:
    protocol = _tiny_protocol()
    data_dir = _write_proto_fixture(tmp_path / "proto", protocol)
    model_path = _write_hold_model(tmp_path / "hold.npz", data_dir, protocol)
    valid = evaluation.load_model(model_path, protocol=protocol)
    assert valid.matrix_a.shape == (6, 6)

    with np.load(model_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["A"] = np.eye(5, dtype=np.float32)
    invalid_path = tmp_path / "invalid.npz"
    np.savez(invalid_path, **arrays)
    with pytest.raises(ValueError, match="array shapes differ"):
        evaluation.load_model(invalid_path, protocol=protocol)

    (data_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest SHA differs"):
        evaluation.load_model(model_path, protocol=protocol)


def test_run_writes_atomic_json_and_rejects_model_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.npz"
    model.write_bytes(b"model")
    expected = {"kind": evaluation.EVALUATION_KIND, "models": {}}
    monkeypatch.setattr(evaluation, "evaluate_models", lambda *args, **kwargs: expected)
    output = tmp_path / "nested" / "report.json"
    args = argparse.Namespace(
        data_dir=tmp_path,
        model=[("candidate", model)],
        batch_size=3,
        output=output,
    )

    assert evaluation.run(args) == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    args.output = model
    with pytest.raises(ValueError, match="must not overwrite"):
        evaluation.run(args)
