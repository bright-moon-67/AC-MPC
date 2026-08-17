from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from antmaze_ac.koopman.model import DeepKoopman
from experiments.dmc.actors import ActorConfig, build_actor
from experiments.dmc.eval import aggregate_dmc
from experiments.dmc.eval.evaluate_dmc import (
    episode_seeds,
    evaluate,
    load_actor_checkpoint,
)
from experiments.dmc.reward_oracle import exact_reward_oracle_metadata


TASK = "cartpole_swingup"


def _protocol(*, control_dt: float = 0.01) -> dict:
    return {
        "protocol_name": "dmc_custom_v1",
        "protocol_schema_version": 1,
        "task": TASK,
        "domain": "cartpole",
        "dmc_task": "cartpole:swingup",
        "dm_control_version": "test",
        "mujoco_version": "test",
        "obs_dim": 5,
        "action_dim": 1,
        "control_dt": control_dt,
        "control_timestep": control_dt,
        "physics_dt": 0.01,
        "physics_timestep": 0.01,
        "n_substeps": int(round(control_dt / 0.01)),
        "action_repeat": 1,
        "time_limit": 3 * control_dt,
        "time_limit_seconds": 3 * control_dt,
        "step_limit": 3,
        "episode_steps": 3,
        "action_low": [-1.0],
        "action_high": [1.0],
        "obs_layout": [["position", 3], ["velocity", 2]],
    }


class _FakeAdapter:
    def __init__(self, protocol: dict, reset_log: list[int]):
        self._protocol = dict(protocol)
        self._reset_log = reset_log
        self.step_limit = int(protocol["step_limit"])
        self.action_low = np.asarray(protocol["action_low"], dtype=np.float32)
        self.action_high = np.asarray(protocol["action_high"], dtype=np.float32)
        self._steps = 0

    def protocol_metadata(self) -> dict:
        return dict(self._protocol)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._steps = 0
        if seed is not None:
            self._reset_log.append(int(seed))
        return np.zeros(5, dtype=np.float32)

    def step(self, action: np.ndarray):
        requested = np.asarray(action, dtype=np.float32).reshape(1)
        applied = np.clip(requested, self.action_low, self.action_high)
        self._steps += 1
        done = self._steps == self.step_limit
        reward = 1.0
        info = {
            "step_count": self._steps,
            "step_type": "LAST" if done else "MID",
            "discount": 1.0,
            "terminated": False,
            "truncated": done,
            "requested_action": requested.copy(),
            "applied_action": applied.copy(),
            "reward_components": {"reward": reward},
        }
        return np.zeros(5, dtype=np.float32), reward, done, info

    def close(self) -> None:
        pass


def _install_fake_adapter(monkeypatch, protocol: dict):
    reset_log: list[int] = []
    factory_calls: list[dict] = []

    def factory(task_name, seed, control_timestep=None, time_limit=None):
        factory_calls.append(
            {
                "task": task_name,
                "seed": seed,
                "control_timestep": control_timestep,
                "time_limit": time_limit,
            }
        )
        return _FakeAdapter(protocol, reset_log)

    monkeypatch.setattr(
        "experiments.dmc.tasks.adapter.make_dmc_adapter", factory
    )
    return reset_log, factory_calls


def _ppo_checkpoint(
    path: Path,
    protocol: dict,
    *,
    training_seed: int = 123,
    authorization: dict | None = None,
) -> Path:
    config = ActorConfig(ppo_hidden_dim=8)
    actor = build_actor("PPO", TASK, torch.device("cpu"), config=config)
    for parameter in actor.parameters():
        parameter.data.zero_()
    payload = {
        "kind": "dmc_ppo_actor",
        "format_version": 3,
        "training_spec_version": "dmc_ppo_v4_raw_observation_critic",
        "task": TASK,
        "actor_type": "PPO",
        "actor_name": "PPO",
        "protocol": protocol,
        "protocol_fingerprint": hashlib.sha256(
            json.dumps(
                protocol,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
        "training_seed": training_seed,
        "actor_config": config.to_dict(),
        "actor_state": actor.state_dict(),
        "ppo_config": {"normalize_observation": True},
        "normalization_contract": {
            "observation": "acme_welford_running_mean_std_v1",
            "observation_scope": "plain_ppo_raw_task_state",
            "advantage": "zero_debiased_ema_mean_absolute_v1",
            "value": "zero_debiased_ema_first_second_moment_v1",
        },
        "evaluation_reference_episodes_per_seed": 1,
        "diagnostic_every_steps": 50_000,
        "observation_normalizer_state": {
            "kind": "acme_welford_observation_normalizer_v1",
            "dimension": 5,
            "epsilon": 1e-6,
            "count": 1,
            "mean": torch.zeros(5, dtype=torch.float64),
            "summed_variance": torch.ones(5, dtype=torch.float64),
            "std": torch.ones(5, dtype=torch.float64),
        },
        "koopman_path": None,
        "koopman_sha256": None,
    }
    if authorization is not None:
        payload.update(
            {
                **authorization,
                "resolved_execution_spec": {"kind": "synthetic_test_spec"},
                "evaluation_seeds": list(range(100, 110)),
                "evaluation_episodes_per_seed": 10,
            }
        )
    torch.save(payload, path)
    return path


def _formal_authorization(*, profile: str = "development", index: int = 0) -> dict:
    return {
        "authorization_kind": "dmc_training_approval_v1",
        "training_approved": True,
        "config_fingerprint": "sha256:" + "1" * 64,
        "approval_profile": profile,
        "approval_file_sha256": "2" * 64,
        "preflight_report_sha256": "3" * 64,
        "train_seed_index": index,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structured_checkpoints(tmp_path: Path, protocol: dict) -> tuple[Path, Path]:
    torch.manual_seed(3)
    koopman = DeepKoopman(
        state_dim=5, action_dim=1, lift_dim=2, hidden_dims=(4,), activation="silu"
    )
    koopman_path = tmp_path / "koopman.pt"
    torch.save(
        {
            "kind": "dmc_k_step_koopman",
            "model_state": koopman.state_dict(),
            "architecture": koopman.architecture(),
            "state_kind": TASK,
            "environment_protocol_json": json.dumps(
                protocol,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            "protocol_fingerprint": hashlib.sha256(
                json.dumps(
                    protocol,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
            "normalizer": {
                "center": torch.zeros(5),
                "scale": torch.ones(5),
            },
        },
        koopman_path,
    )
    config = ActorConfig(
        hidden_dim=4,
        ppo_hidden_dim=8,
        ab_rank=1,
        kmpc_horizon=2,
        kmpc_solver_iterations=2,
    )
    actor = build_actor(
        "KMPC", TASK, torch.device("cpu"), koopman=koopman, config=config
    )
    actor_path = tmp_path / "actor.pt"
    torch.save(
        {
            "kind": "dmc_ppo_actor",
            "format_version": 3,
            "training_spec_version": "dmc_ppo_v4_raw_observation_critic",
            "task": TASK,
            "actor_type": "KMPC",
            "actor_name": "KMPC",
            "protocol": protocol,
            "protocol_fingerprint": hashlib.sha256(
                json.dumps(
                    protocol,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
            "training_seed": 456,
            "actor_config": config.to_dict(),
            "actor_state": actor.state_dict(),
            "koopman_path": str(koopman_path),
            "koopman_sha256": _sha256(koopman_path),
            "koopman_task": TASK,
            "normalizer": {
                "center": torch.zeros(5),
                "scale": torch.ones(5),
            },
        },
        actor_path,
    )
    return actor_path, koopman_path


def test_evaluate_restores_ppo_metadata_and_uses_env_step_limit(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    reset_log, calls = _install_fake_adapter(monkeypatch, protocol)
    checkpoint = _ppo_checkpoint(tmp_path / "ppo.pt", protocol)

    report = evaluate(checkpoint, episodes=2, eval_seed=9001, device_name="cpu")

    assert report["task"] == TASK
    assert report["actor_type"] == "PPO"
    assert report["actor_config"]["ppo_hidden_dim"] == 8
    assert report["training_seed"] == 123
    assert report["authorization_verified"] is False
    assert report["training_approved"] is None
    assert report["eval_seed"] == 9001
    assert report["environment_step_limit"] == 3
    assert report["episode_lengths"] == [3, 3]
    assert report["episode_returns"] == [3.0, 3.0]
    assert report["acme_reference_episode_returns"] == [3.0]
    assert report["robustness_episode_returns"] == [3.0, 3.0]
    assert report["episode_action_component_counts"] == [3, 3]
    assert report["acme_reference_action_component_count"] == 3
    assert report["robustness_action_component_count"] == 6
    assert report["acme_reference_applied_action_bound_fraction"] == 0.0
    assert report["robustness_applied_action_bound_fraction"] == 0.0
    assert report["truncated_episodes"] == 2
    assert report["terminated_episodes"] == 0
    assert report["mean_final_discount"] == 1.0
    assert report["evaluation_runner"] == "synchronous_episode_batch_v1"
    assert report["evaluation_num_envs"] == 2
    assert len(reset_log) == 2 and len(set(reset_log)) == 2
    assert len(calls) == 2
    assert calls[0]["task"] == TASK
    assert calls[0]["control_timestep"] == protocol["control_dt"]
    assert calls[0]["time_limit"] == protocol["time_limit"]


def test_evaluate_propagates_complete_formal_authorization(tmp_path, monkeypatch):
    protocol = _protocol()
    _install_fake_adapter(monkeypatch, protocol)
    authorization = _formal_authorization(profile="benchmark", index=2)
    checkpoint = _ppo_checkpoint(
        tmp_path / "formal.pt",
        protocol,
        training_seed=20260813,
        authorization=authorization,
    )

    metadata = load_actor_checkpoint(checkpoint)
    report = evaluate(checkpoint, episodes=1, eval_seed=9001, device_name="cpu")

    assert metadata.authorization_verified is True
    assert report["authorization_verified"] is True
    for field, value in authorization.items():
        assert report[field] == value
    assert report["authorization_errors"] == []
    assert report["resolved_execution_spec"] == {"kind": "synthetic_test_spec"}
    assert report["evaluation_seeds"] == list(range(100, 110))
    assert report["evaluation_episodes_per_seed"] == 10


def test_malformed_authorization_remains_evaluable_but_is_not_verified(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    _install_fake_adapter(monkeypatch, protocol)
    malformed = {
        **_formal_authorization(),
        "config_fingerprint": "short",
        "train_seed_index": True,
    }
    checkpoint = _ppo_checkpoint(
        tmp_path / "malformed.pt", protocol, authorization=malformed
    )

    report = evaluate(checkpoint, episodes=1, eval_seed=1, device_name="cpu")

    assert report["authorization_verified"] is False
    assert report["config_fingerprint"] == "short"
    assert report["train_seed_index"] is None
    assert report["authorization_errors"]


def test_evaluate_rejects_runtime_protocol_drift(tmp_path, monkeypatch):
    saved = _protocol(control_dt=0.01)
    live = _protocol(control_dt=0.02)
    _install_fake_adapter(monkeypatch, live)
    checkpoint = _ppo_checkpoint(tmp_path / "ppo.pt", saved)

    with pytest.raises(RuntimeError, match="protocol does not match"):
        evaluate(checkpoint, episodes=1, eval_seed=1, device_name="cpu")


def test_structured_actor_restores_and_checks_koopman(tmp_path, monkeypatch):
    protocol = _protocol()
    _install_fake_adapter(monkeypatch, protocol)
    actor_path, koopman_path = _structured_checkpoints(tmp_path, protocol)

    report = evaluate(actor_path, episodes=1, eval_seed=2, device_name="cpu")
    assert report["actor_type"] == "KMPC"
    assert report["koopman_checkpoint"] == str(koopman_path.resolve())
    assert report["training_seed"] == 456

    payload = torch.load(actor_path, weights_only=False)
    payload["koopman_sha256"] = "0" * 64
    bad_path = tmp_path / "bad_actor.pt"
    torch.save(payload, bad_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate(bad_path, episodes=1, eval_seed=2, device_name="cpu")

    payload = torch.load(actor_path, weights_only=False)
    payload["normalizer"]["center"][0] = 1.0
    bad_normalizer_path = tmp_path / "bad_normalizer_actor.pt"
    torch.save(payload, bad_normalizer_path)
    with pytest.raises(ValueError, match="different state normalizers"):
        evaluate(
            bad_normalizer_path, episodes=1, eval_seed=2, device_name="cpu"
        )


def test_ac_mpc_mpve_checkpoint_evaluation_round_trip(tmp_path, monkeypatch):
    protocol = _protocol()
    _install_fake_adapter(monkeypatch, protocol)
    actor_path, koopman_path = _structured_checkpoints(tmp_path, protocol)
    payload = torch.load(actor_path, weights_only=False)
    payload["actor_type"] = payload["actor_name"] = "AC-MPC-MPVE"
    payload["ppo_config"] = {
        "mpve_horizon": 2,
        "mpve_value_loss_coefficient": 1.0,
        "mpve_reward_source": "dmc_official_observation_oracle_v1",
    }
    payload["value_expansion"] = {
        "enabled": True,
        "kind": "mpve_td_k_tro25_eq8_eq9_v1",
        "actor_shared_with": "KMPC",
        "horizon": 2,
        "value_loss_coefficient": 1.0,
        "prediction_gradient": "detached",
        "terminal_target_gradient": "detached",
        "standard_gae_value_loss_retained": True,
        "reward": exact_reward_oracle_metadata(TASK),
    }
    mpve_path = tmp_path / "ac_mpc_mpve.pt"
    torch.save(payload, mpve_path)

    metadata = load_actor_checkpoint(mpve_path)
    report = evaluate(mpve_path, episodes=1, eval_seed=3, device_name="cpu")
    assert metadata.actor_type == "AC-MPC-MPVE"
    assert report["actor_type"] == "AC-MPC-MPVE"
    assert report["koopman_checkpoint"] == str(koopman_path.resolve())
    assert report["value_expansion"] == payload["value_expansion"]

    payload["value_expansion"]["prediction_gradient"] = "enabled"
    malformed = tmp_path / "malformed_mpve.pt"
    torch.save(payload, malformed)
    with pytest.raises(ValueError, match="value_expansion"):
        load_actor_checkpoint(malformed)


def test_structured_actor_rejects_koopman_from_another_protocol(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    _install_fake_adapter(monkeypatch, protocol)
    actor_path, koopman_path = _structured_checkpoints(tmp_path, protocol)
    koopman_payload = torch.load(koopman_path, weights_only=False)
    different = _protocol(control_dt=0.02)
    environment_json = json.dumps(
        different, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    koopman_payload["environment_protocol_json"] = environment_json
    koopman_payload["protocol_fingerprint"] = hashlib.sha256(
        environment_json.encode("utf-8")
    ).hexdigest()
    different_koopman = tmp_path / "different_protocol_koopman.pt"
    torch.save(koopman_payload, different_koopman)

    actor_payload = torch.load(actor_path, weights_only=False)
    actor_payload["koopman_path"] = str(different_koopman)
    actor_payload["koopman_sha256"] = _sha256(different_koopman)
    mismatched_actor = tmp_path / "mismatched_actor.pt"
    torch.save(actor_payload, mismatched_actor)
    with pytest.raises(ValueError, match="different DMC environment protocols"):
        evaluate(mismatched_actor, episodes=1, eval_seed=2, device_name="cpu")


def test_structured_actor_propagates_and_validates_koopman_lineage(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    _install_fake_adapter(monkeypatch, protocol)
    actor_path, koopman_path = _structured_checkpoints(tmp_path, protocol)
    lineage = {
        "dataset_sha256": "4" * 64,
        "config_fingerprint": "sha256:" + "5" * 64,
        "approval_profile": "development",
        "approval_file_sha256": "6" * 64,
        "preflight_report_sha256": "7" * 64,
    }
    koopman_payload = torch.load(koopman_path, weights_only=False)
    koopman_payload.update(lineage)
    torch.save(koopman_payload, koopman_path)

    actor_payload = torch.load(actor_path, weights_only=False)
    actor_payload.update(
        {
            "koopman_sha256": _sha256(koopman_path),
            "koopman_lineage": dict(lineage),
            "koopman_dataset_sha256": lineage["dataset_sha256"],
            "koopman_config_fingerprint": lineage["config_fingerprint"],
        }
    )
    torch.save(actor_payload, actor_path)

    report = evaluate(actor_path, episodes=1, eval_seed=2, device_name="cpu")
    assert report["koopman_lineage"] == lineage
    assert report["koopman_dataset_sha256"] == lineage["dataset_sha256"]

    actor_payload["koopman_lineage"] = {
        **lineage,
        "dataset_sha256": "8" * 64,
    }
    bad_actor = tmp_path / "bad_lineage_actor.pt"
    torch.save(actor_payload, bad_actor)
    with pytest.raises(ValueError, match="lineage does not match"):
        evaluate(bad_actor, episodes=1, eval_seed=2, device_name="cpu")


def test_checkpoint_requires_explicit_architecture_metadata(tmp_path):
    path = _ppo_checkpoint(tmp_path / "ppo.pt", _protocol())
    payload = torch.load(path, weights_only=False)
    del payload["actor_config"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="actor_config"):
        load_actor_checkpoint(path)


def test_episode_seed_derivation_is_deterministic_and_replicate_disjoint():
    first = episode_seeds(100, 20)
    assert first == episode_seeds(100, 20)
    assert len(first) == len(set(first))
    assert set(first).isdisjoint(episode_seeds(101, 20))


def test_ten_seed_aggregate_keeps_training_and_eval_axes_separate(
    tmp_path, monkeypatch
):
    checkpoint = _ppo_checkpoint(tmp_path / "ppo.pt", _protocol(), training_seed=7)
    eval_seeds = list(range(100, 110))

    def fake_evaluate(actor_checkpoint, **kwargs):
        seed = kwargs["eval_seed"]
        value = float(seed - 100)
        return {
            "task": TASK,
            "actor_type": "PPO",
            "training_seed": 7,
            "eval_seed": seed,
            "koopman_checkpoint": None,
            "runtime_protocol": _protocol(),
            "return_mean_across_episodes": value,
            "episode_returns": [value],
            "acme_reference_episode_returns": [value],
            "acme_reference_episode_count": 1,
            "acme_reference_return_mean": value,
            "robustness_episode_returns": [value],
            "robustness_episode_count": 1,
            "robustness_return_mean": value,
            "episode_action_component_counts": [3],
            "episode_applied_action_bound_counts": [0],
            "acme_reference_action_component_count": 3,
            "acme_reference_applied_action_bound_count": 0,
            "acme_reference_applied_action_bound_fraction": 0.0,
            "robustness_action_component_count": 3,
            "robustness_applied_action_bound_count": 0,
            "robustness_applied_action_bound_fraction": 0.0,
            "episode_length_mean": 3.0,
            "mean_step_reward": value / 3.0,
            "requested_action_bound_fraction": 0.0,
            "applied_action_bound_fraction": 0.0,
            "action_clipped_fraction": 0.0,
            "terminated_episodes": 0,
            "truncated_episodes": 1,
            "mean_reward_components": {"reward": value / 3.0},
        }

    monkeypatch.setattr(aggregate_dmc, "evaluate", fake_evaluate)
    report = aggregate_dmc.aggregate_evaluations(
        checkpoint,
        eval_seeds=eval_seeds,
        episodes_per_seed=1,
        device_name="cpu",
    )

    assert report["training_seed"] == 7
    assert report["training_seed_count"] == 1
    assert report["eval_seeds"] == eval_seeds
    assert report["eval_seed_count"] == 10
    assert report["total_evaluation_episodes"] == 10
    assert report["return_mean_across_eval_seed_means"] == pytest.approx(4.5)
    assert report["acme_reference_summary"]["episode_count"] == 10
    assert report["acme_reference_summary"]["return_mean"] == pytest.approx(4.5)
    assert report["robustness_summary"]["episode_count"] == 10
    assert report["acme_reference_summary"][
        "applied_action_bound_fraction"
    ] == 0.0
    assert len(report["per_eval_seed"]) == 10


def test_canonical_plan_falls_back_to_task_config(tmp_path):
    checkpoint = _ppo_checkpoint(tmp_path / "ppo.pt", _protocol(), training_seed=7)
    seeds, episodes, source = aggregate_dmc.canonical_evaluation_plan(checkpoint)

    assert seeds == list(range(20260901, 20260911))
    assert episodes == 10
    assert source.endswith("experiments/dmc/configs/cartpole_swingup.yaml")
