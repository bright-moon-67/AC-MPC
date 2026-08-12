from pathlib import Path

import pytest

import experiments.dmc.preflight as preflight_module
from experiments.dmc.preflight import run_preflight
from experiments.dmc.source_identity import source_identity


CONFIG = Path("experiments/dmc/configs/cartpole_swingup.yaml")


@pytest.mark.parametrize(
    ("parity_steps", "throughput_steps", "message"),
    [
        (0, 1, "parity_steps"),
        (True, 1, "parity_steps"),
        (1, 0, "throughput_steps"),
        (1, False, "throughput_steps"),
        (1001, 1, "parity_steps"),
    ],
)
def test_preflight_rejects_vacuous_or_ambiguous_budgets(
    parity_steps, throughput_steps, message
):
    with pytest.raises(ValueError, match=message):
        run_preflight(
            CONFIG,
            parity_steps=parity_steps,
            throughput_steps=throughput_steps,
        )


def test_source_identity_is_content_sensitive(tmp_path: Path):
    source = tmp_path / "experiments" / "dmc" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_identity(tmp_path)
    assert first == source_identity(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = source_identity(tmp_path)
    assert first["fingerprint"] != second["fingerprint"]


def test_preflight_actor_probe_uses_the_selected_profile(monkeypatch):
    observed: dict[str, str] = {}

    class FakeAdapter:
        def protocol_metadata(self):
            return {
                "protocol_name": "dmc_native_v1",
                "task": "cartpole_swingup",
            }

        def close(self):
            pass

    monkeypatch.setattr(
        preflight_module, "ALL_TASK_ORDER", ("cartpole_swingup",)
    )
    monkeypatch.setattr(
        preflight_module,
        "verify_task_spec",
        lambda _task: {"passed": True},
    )
    monkeypatch.setattr(
        preflight_module,
        "_suite_parity",
        lambda _task, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        preflight_module,
        "_timeout_check",
        lambda _task: {"passed": True},
    )

    def actor_probe(_task, _config, *, profile):
        observed["profile"] = profile
        return {"PPO": {"passed": True}}

    monkeypatch.setattr(preflight_module, "_actor_forward_check", actor_probe)
    monkeypatch.setattr(
        preflight_module,
        "_mpve_forward_check",
        lambda _task, _config, *, profile, env_workers=None: {
            "passed": True,
            "profile": profile,
            "env_workers": env_workers,
            "optimizer_steps": 0,
            "environment_steps": 0,
        },
    )
    monkeypatch.setattr(
        preflight_module,
        "_mpve_reward_source_parity",
        lambda _task, **_kwargs: {
            "passed": True,
            "source": "dmc_official_observation_oracle_v1",
            "optimizer_steps": 0,
            "environment_steps": 1,
        },
    )
    monkeypatch.setattr(
        preflight_module,
        "_vector_rollout_probe",
            lambda _task, _config, *, profile, env_workers=None: {
                "passed": True,
                "profile": profile,
                "env_workers": env_workers,
            "protocol_fingerprint": preflight_module.protocol_fingerprint(
                {
                    "protocol_name": "dmc_native_v1",
                    "task": "cartpole_swingup",
                }
            ),
            "environment_transitions_per_step_second": 1.0,
            "optimizer_steps": 0,
        },
    )
    monkeypatch.setattr(
        preflight_module,
        "_throughput",
        lambda _task, steps, seed: {
            "steps": steps,
            "elapsed_seconds": 1.0,
            "environment_steps_per_second": float(steps),
        },
    )
    monkeypatch.setattr(
        "experiments.dmc.tasks.adapter.make_dmc_adapter",
        lambda *_args, **_kwargs: FakeAdapter(),
    )

    report = run_preflight(
        CONFIG,
        profile="benchmark",
        parity_steps=1,
        throughput_steps=1,
    )

    assert observed["profile"] == "benchmark"
    assert report["profile"] == "benchmark"
    assert report["mpve_critic_only_forward"]["optimizer_steps"] == 0
    assert report["configured_vector_rollout"]["optimizer_steps"] == 0
    assert report["configured_vector_rollout"][
        "protocol_matches_selected_environment"
    ] is True
    assert report["ready_for_user_review"] is True
