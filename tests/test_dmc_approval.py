import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.dmc.approval import (
    APPROVAL_KIND,
    TrainingApprovalError,
    build_training_approval,
    main,
    validate_training_approval,
    validate_training_preflight,
    write_training_approval,
)
from experiments.dmc.config import (
    ExperimentConfig,
    load_experiment_config,
    resolve_execution_spec,
)
from experiments.dmc.protocol import protocol_fingerprint
from experiments.dmc.source_identity import source_identity


@pytest.fixture
def config() -> ExperimentConfig:
    return load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _preflight_payload(config: ExperimentConfig) -> dict[str, object]:
    environment_protocol = {
        "protocol_name": "dmc_native_v1",
        "task": config.task,
        "obs_dim": 5,
        "action_dim": 1,
        "control_dt": 0.01,
        "step_limit": 1000,
    }
    return {
        "kind": "dmc_training_free_preflight",
        "ready_for_user_review": True,
        "training_approved": False,
        "task": config.task,
        "profile": "development",
        "config_fingerprint": config.fingerprint,
        "resolved_execution_spec": resolve_execution_spec(
            config, "development"
        ),
        "environment_protocol": environment_protocol,
        "protocol_fingerprint": protocol_fingerprint(environment_protocol),
        "source_identity": source_identity(),
        "checks": {"passed": True},
    }


def _write_preflight(tmp_path: Path, config: ExperimentConfig) -> Path:
    path = tmp_path / "preflight.json"
    _write_json(path, _preflight_payload(config))
    return path


def _write_valid_approval(
    tmp_path: Path,
    config: ExperimentConfig,
    *,
    profile: str = "development",
) -> tuple[Path, Path]:
    preflight = _write_preflight(tmp_path, config)
    approval = tmp_path / "approval.json"
    write_training_approval(
        config,
        profile,
        preflight,
        approval,
        approve=True,
    )
    return preflight, approval


def test_valid_approval_binds_exact_config_profile_and_preflight_bytes(
    tmp_path: Path,
    config: ExperimentConfig,
):
    preflight, approval_path = _write_valid_approval(tmp_path, config)

    approval = validate_training_approval(
        config, "development", approval_path, preflight
    )

    assert approval == {
        "kind": APPROVAL_KIND,
        "approved": True,
        "task": config.task,
        "profile": "development",
        "config_fingerprint": config.fingerprint,
        "preflight_report_sha256": hashlib.sha256(
            preflight.read_bytes()
        ).hexdigest(),
    }

    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    validate_training_approval(
        config,
        "development",
        approval_path,
        preflight,
        runtime_protocol_fingerprint=preflight_payload["protocol_fingerprint"],
    )
    with pytest.raises(TrainingApprovalError, match="Runtime DMC protocol"):
        validate_training_approval(
            config,
            "development",
            approval_path,
            preflight,
            runtime_protocol_fingerprint="0" * 64,
        )


def test_public_preflight_validation_binds_runtime_without_approving(
    tmp_path: Path,
    config: ExperimentConfig,
):
    preflight = _write_preflight(tmp_path, config)
    payload = json.loads(preflight.read_text(encoding="utf-8"))

    validated = validate_training_preflight(
        config,
        "development",
        preflight,
        runtime_protocol_fingerprint=payload["protocol_fingerprint"],
    )

    assert validated == payload
    assert validated["training_approved"] is False
    with pytest.raises(TrainingApprovalError, match="Runtime DMC protocol"):
        validate_training_preflight(
            config,
            "development",
            preflight,
            runtime_protocol_fingerprint="0" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ready_for_user_review", False),
        ("ready_for_user_review", 1),
        ("ready_for_user_review", None),
        ("training_approved", True),
        ("training_approved", 0),
        ("training_approved", None),
    ],
)
def test_preflight_review_flags_are_strict_booleans(
    tmp_path: Path,
    config: ExperimentConfig,
    field: str,
    value: object,
):
    preflight = tmp_path / "preflight.json"
    payload = _preflight_payload(config)
    payload[field] = value
    _write_json(preflight, payload)

    with pytest.raises(TrainingApprovalError):
        build_training_approval(
            config, "development", preflight, approve=True
        )


def test_missing_preflight_review_flag_is_denied(
    tmp_path: Path,
    config: ExperimentConfig,
):
    preflight = tmp_path / "preflight.json"
    payload = _preflight_payload(config)
    del payload["ready_for_user_review"]
    _write_json(preflight, payload)

    with pytest.raises(TrainingApprovalError, match="ready_for_user_review"):
        build_training_approval(
            config, "development", preflight, approve=True
        )


def test_preflight_must_bind_task_and_complete_config_fingerprint(
    tmp_path: Path,
    config: ExperimentConfig,
):
    for field, value in (
        ("task", "reacher_hard"),
        ("config_fingerprint", "sha256:" + "0" * 64),
    ):
        preflight = tmp_path / f"{field}.json"
        payload = _preflight_payload(config)
        payload[field] = value
        _write_json(preflight, payload)
        with pytest.raises(TrainingApprovalError, match=field):
            build_training_approval(
                config, "development", preflight, approve=True
            )


def test_any_preflight_byte_change_after_approval_is_denied(
    tmp_path: Path,
    config: ExperimentConfig,
):
    preflight, approval = _write_valid_approval(tmp_path, config)
    payload = json.loads(preflight.read_text(encoding="utf-8"))
    payload["review_note"] = "changed after approval"
    _write_json(preflight, payload)

    with pytest.raises(TrainingApprovalError, match="current preflight"):
        validate_training_approval(
            config, "development", approval, preflight
        )


def test_approval_is_invalid_for_another_profile_or_changed_config(
    tmp_path: Path,
    config: ExperimentConfig,
):
    preflight, approval = _write_valid_approval(tmp_path, config)

    with pytest.raises(TrainingApprovalError, match="profile"):
        validate_training_approval(config, "benchmark", approval, preflight)

    changed_raw = copy.deepcopy(config.raw)
    changed_raw["profiles"]["development"]["total_timesteps"] += 1
    changed = ExperimentConfig(path=config.path, raw=changed_raw)
    with pytest.raises(TrainingApprovalError, match="config_fingerprint"):
        validate_training_approval(changed, "development", approval, preflight)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "dmc_training_approval_v2", "kind"),
        ("approved", False, "approved"),
        ("approved", 1, "approved"),
        ("task", "reacher_hard", "task"),
        ("profile", "benchmark", "profile"),
        ("config_fingerprint", "sha256:" + "0" * 64, "config_fingerprint"),
        ("preflight_report_sha256", "0" * 64, "preflight"),
    ],
)
def test_each_approval_binding_is_fail_closed(
    tmp_path: Path,
    config: ExperimentConfig,
    field: str,
    value: object,
    message: str,
):
    preflight, approval_path = _write_valid_approval(tmp_path, config)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval[field] = value
    _write_json(approval_path, approval)

    with pytest.raises(TrainingApprovalError, match=message):
        validate_training_approval(
            config, "development", approval_path, preflight
        )


def test_missing_unknown_duplicate_and_non_object_approval_are_denied(
    tmp_path: Path,
    config: ExperimentConfig,
):
    preflight, approval_path = _write_valid_approval(tmp_path, config)
    valid = json.loads(approval_path.read_text(encoding="utf-8"))

    missing = dict(valid)
    del missing["task"]
    _write_json(approval_path, missing)
    with pytest.raises(TrainingApprovalError, match="missing"):
        validate_training_approval(
            config, "development", approval_path, preflight
        )

    unknown = {**valid, "reviewer": "implicit identities are not accepted"}
    _write_json(approval_path, unknown)
    with pytest.raises(TrainingApprovalError, match="unknown"):
        validate_training_approval(
            config, "development", approval_path, preflight
        )

    approval_path.write_text(
        '{"kind":"dmc_training_approval_v1","kind":"duplicate"}',
        encoding="utf-8",
    )
    with pytest.raises(TrainingApprovalError, match="Duplicate"):
        validate_training_approval(
            config, "development", approval_path, preflight
        )

    _write_json(approval_path, [valid])
    with pytest.raises(TrainingApprovalError, match="JSON object"):
        validate_training_approval(
            config, "development", approval_path, preflight
        )


def test_write_requires_literal_true_and_never_replaces_existing_file(
    tmp_path: Path,
    config: ExperimentConfig,
):
    preflight = _write_preflight(tmp_path, config)
    output = tmp_path / "nested" / "approval.json"

    for decision in (False, 1, "yes", None):
        with pytest.raises(TrainingApprovalError, match="explicit"):
            write_training_approval(
                config,
                "development",
                preflight,
                output,
                approve=decision,  # type: ignore[arg-type]
            )
        assert not output.exists()

    output.parent.mkdir(parents=True)
    output.write_text("do not replace", encoding="utf-8")
    with pytest.raises(TrainingApprovalError, match="Refusing to replace"):
        write_training_approval(
            config,
            "development",
            preflight,
            output,
            approve=True,
        )
    assert output.read_text(encoding="utf-8") == "do not replace"


@pytest.mark.parametrize("profile", ["", "dev", "Benchmark", 1, None])
def test_only_named_development_or_benchmark_profiles_are_accepted(
    tmp_path: Path,
    config: ExperimentConfig,
    profile: object,
):
    preflight = _write_preflight(tmp_path, config)
    with pytest.raises(TrainingApprovalError, match="profile"):
        build_training_approval(
            config,
            profile,  # type: ignore[arg-type]
            preflight,
            approve=True,
        )


def test_cli_without_approve_validates_but_writes_nothing(
    tmp_path: Path,
    config: ExperimentConfig,
):
    preflight = _write_preflight(tmp_path, config)
    output = tmp_path / "approval.json"

    with pytest.raises(SystemExit, match="No approval written"):
        main(
            [
                "--config",
                str(config.path),
                "--profile",
                "development",
                "--preflight",
                str(preflight),
                "--output",
                str(output),
            ]
        )

    assert not output.exists()
