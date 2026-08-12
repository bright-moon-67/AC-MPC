"""Fail-closed approval contract for starting a DMC training run.

The training-free preflight report only establishes that a configuration is
ready for human review.  It is deliberately not an authorization to train.
An authorization is a separate, versioned JSON document which binds the exact
configuration, requested profile, task, and bytes of the reviewed preflight
report.

Running this module without ``--approve`` never writes an approval artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from experiments.dmc.config import (
    ExperimentConfig,
    load_experiment_config,
    resolve_execution_spec,
)
from experiments.dmc.protocol import protocol_fingerprint


APPROVAL_KIND = "dmc_training_approval_v1"
PREFLIGHT_KIND = "dmc_training_free_preflight"
ALLOWED_PROFILES = ("development", "benchmark")

_APPROVAL_FIELDS = frozenset(
    {
        "kind",
        "approved",
        "task",
        "profile",
        "config_fingerprint",
        "preflight_report_sha256",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class TrainingApprovalError(ValueError):
    """Raised whenever an approval cannot be proven valid."""


def _reject_json_constant(value: str) -> None:
    raise TrainingApprovalError(f"Non-finite JSON value {value!r} is not allowed")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingApprovalError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _read_json_object(path: str | Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise TrainingApprovalError(f"Cannot read {label} {source}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingApprovalError(f"Invalid JSON in {label} {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingApprovalError(f"{label} {source} must contain a JSON object")
    return raw, payload


def _coerce_config(config: ExperimentConfig | str | Path) -> ExperimentConfig:
    if isinstance(config, ExperimentConfig):
        return config
    try:
        return load_experiment_config(config)
    except (OSError, TypeError, ValueError) as exc:
        raise TrainingApprovalError(f"Invalid experiment config: {exc}") from exc


def _validate_profile(config: ExperimentConfig, profile: str) -> None:
    if not isinstance(profile, str) or profile not in ALLOWED_PROFILES:
        choices = ", ".join(ALLOWED_PROFILES)
        raise TrainingApprovalError(
            f"profile must be one of ({choices}); received {profile!r}"
        )
    profiles = config.raw.get("profiles")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise TrainingApprovalError(f"Config does not define profile {profile!r}")


def _validate_preflight(
    config: ExperimentConfig,
    path: str | Path,
    *,
    profile: str,
) -> tuple[bytes, dict[str, Any]]:
    raw, report = _read_json_object(path, label="preflight report")
    if report.get("kind") != PREFLIGHT_KIND:
        raise TrainingApprovalError(
            f"Preflight kind must be {PREFLIGHT_KIND!r}"
        )
    if report.get("ready_for_user_review") is not True:
        raise TrainingApprovalError(
            "Preflight is not ready for user review "
            "(ready_for_user_review must be true)"
        )
    if report.get("training_approved") is not False:
        raise TrainingApprovalError(
            "Preflight must not itself approve training "
            "(training_approved must be false)"
        )
    if report.get("task") != config.task:
        raise TrainingApprovalError(
            "Preflight task does not match the experiment config"
        )
    if report.get("profile") != profile:
        raise TrainingApprovalError(
            "Preflight profile does not match the requested approval profile"
        )
    if report.get("config_fingerprint") != config.fingerprint:
        raise TrainingApprovalError(
            "Preflight config_fingerprint does not match the complete config"
        )
    if report.get("resolved_execution_spec") != resolve_execution_spec(
        config, profile
    ):
        raise TrainingApprovalError(
            "Preflight resolved_execution_spec does not match the config/profile"
        )
    environment_protocol = report.get("environment_protocol")
    if not isinstance(environment_protocol, dict):
        raise TrainingApprovalError(
            "Preflight is missing the live environment_protocol mapping"
        )
    try:
        expected_protocol_fingerprint = protocol_fingerprint(environment_protocol)
    except (TypeError, ValueError) as exc:
        raise TrainingApprovalError(
            "Preflight environment_protocol cannot be fingerprinted"
        ) from exc
    if report.get("protocol_fingerprint") != expected_protocol_fingerprint:
        raise TrainingApprovalError(
            "Preflight protocol_fingerprint does not match environment_protocol"
        )
    if environment_protocol.get("task") != config.task:
        raise TrainingApprovalError(
            "Preflight environment protocol task does not match the config"
        )
    if environment_protocol.get("protocol_name") != config.protocol["name"]:
        raise TrainingApprovalError(
            "Preflight environment protocol name does not match the config"
        )
    # Source identity is retained as immutable provenance, but is deliberately
    # not an execution lock. Protocol/config/artifact identities are the hard
    # boundaries; implementation-only changes such as CPU worker count must
    # not invalidate unrelated data and model lineages.
    reviewed_source = report.get("source_identity")
    if not isinstance(reviewed_source, dict) or not isinstance(
        reviewed_source.get("fingerprint"), str
    ):
        raise TrainingApprovalError("Preflight is missing source_identity")
    return raw, report


def _preflight_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validated_training_preflight(
    config: ExperimentConfig | str | Path,
    profile: str,
    preflight_path: str | Path,
    *,
    runtime_protocol_fingerprint: str | None = None,
) -> tuple[ExperimentConfig, bytes, dict[str, Any]]:
    """Return one fully validated config/preflight binding."""

    experiment = _coerce_config(config)
    _validate_profile(experiment, profile)
    preflight_raw, preflight = _validate_preflight(
        experiment, preflight_path, profile=profile
    )
    if runtime_protocol_fingerprint is not None:
        if not isinstance(runtime_protocol_fingerprint, str):
            raise TrainingApprovalError(
                "Runtime protocol fingerprint must be a string"
            )
        if runtime_protocol_fingerprint != preflight["protocol_fingerprint"]:
            raise TrainingApprovalError(
                "Runtime DMC protocol differs from the reviewed preflight"
            )
    return experiment, preflight_raw, preflight


def validate_training_preflight(
    config: ExperimentConfig | str | Path,
    profile: str,
    preflight_path: str | Path,
    *,
    runtime_protocol_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate a reviewed preflight without granting training authority.

    This is the public entry point for training-free run manifests.  A valid
    return value means only that the report matches the current config, profile
    and optional runtime protocol; it never implies approval to train. Source
    identity is recorded for reproducibility but is not a cross-stage lock.
    """

    _, _, preflight = _validated_training_preflight(
        config,
        profile,
        preflight_path,
        runtime_protocol_fingerprint=runtime_protocol_fingerprint,
    )
    return preflight


def build_training_approval(
    config: ExperimentConfig | str | Path,
    profile: str,
    preflight_path: str | Path,
    *,
    approve: bool = False,
) -> dict[str, Any]:
    """Build an approval payload only after an explicit approval decision.

    This function does not write a file.  ``approve`` is keyword-only and must
    be the boolean value ``True``; truthy substitutes such as ``1`` are rejected.
    """

    if approve is not True:
        raise TrainingApprovalError(
            "Training remains unapproved; explicit approve=True is required"
        )
    experiment, preflight_raw, preflight = _validated_training_preflight(
        config, profile, preflight_path
    )
    return {
        "kind": APPROVAL_KIND,
        "approved": True,
        "task": preflight["task"],
        "profile": profile,
        "config_fingerprint": preflight["config_fingerprint"],
        "preflight_report_sha256": _preflight_digest(preflight_raw),
    }


def validate_training_approval(
    config: ExperimentConfig | str | Path,
    profile: str,
    approval_path: str | Path,
    preflight_path: str | Path,
    *,
    runtime_protocol_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate all bindings and return the approval object.

    Any absent, malformed, stale, mismatched, or merely truthy value is denied.
    The digest is computed over the exact preflight file bytes, so editing or
    reformatting that report after review invalidates the approval.
    """

    experiment, preflight_raw, preflight = _validated_training_preflight(
        config,
        profile,
        preflight_path,
        runtime_protocol_fingerprint=runtime_protocol_fingerprint,
    )
    _, approval = _read_json_object(approval_path, label="training approval")

    fields = frozenset(approval)
    if fields != _APPROVAL_FIELDS:
        missing = sorted(_APPROVAL_FIELDS - fields)
        unknown = sorted(fields - _APPROVAL_FIELDS)
        raise TrainingApprovalError(
            f"Approval schema fields do not match {APPROVAL_KIND!r}; "
            f"missing={missing}, unknown={unknown}"
        )
    if approval["kind"] != APPROVAL_KIND:
        raise TrainingApprovalError(f"Approval kind must be {APPROVAL_KIND!r}")
    if approval["approved"] is not True:
        raise TrainingApprovalError("Approval field approved must be true")
    if approval["task"] != experiment.task or approval["task"] != preflight["task"]:
        raise TrainingApprovalError("Approval task binding does not match")
    if approval["profile"] != profile:
        raise TrainingApprovalError("Approval profile binding does not match")
    if (
        approval["config_fingerprint"] != experiment.fingerprint
        or approval["config_fingerprint"] != preflight["config_fingerprint"]
    ):
        raise TrainingApprovalError("Approval config_fingerprint binding does not match")

    expected_digest = _preflight_digest(preflight_raw)
    actual_digest = approval["preflight_report_sha256"]
    if not isinstance(actual_digest, str) or _SHA256_PATTERN.fullmatch(actual_digest) is None:
        raise TrainingApprovalError(
            "Approval preflight_report_sha256 must be 64 lowercase hex characters"
        )
    if actual_digest != expected_digest:
        raise TrainingApprovalError("Approval does not bind the current preflight report")
    return approval


def _write_new_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create, but never replace, an approval artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise TrainingApprovalError(
                f"Refusing to replace existing approval artifact {path}"
            ) from exc
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_training_approval(
    config: ExperimentConfig | str | Path,
    profile: str,
    preflight_path: str | Path,
    output_path: str | Path,
    *,
    approve: bool = False,
) -> dict[str, Any]:
    """Create a new approval artifact after explicit approval.

    Existing approval files are never overwritten.  Validation is repeated
    after creation before the payload is returned.
    """

    payload = build_training_approval(
        config,
        profile,
        preflight_path,
        approve=approve,
    )
    output = Path(output_path)
    _write_new_json_atomic(output, payload)
    return validate_training_approval(config, profile, output, preflight_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=ALLOWED_PROFILES, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Explicitly create the bound training approval artifact",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.approve:
        # Validate that this is at least a reviewable preflight, but never write.
        experiment = _coerce_config(args.config)
        _validate_profile(experiment, args.profile)
        _validate_preflight(experiment, args.preflight, profile=args.profile)
        print(
            json.dumps(
                {
                    "approved_artifact_written": False,
                    "config_fingerprint": experiment.fingerprint,
                    "profile": args.profile,
                    "task": experiment.task,
                    "training_approved": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(
            "No approval written. Re-run with --approve only after explicit user review."
        )

    payload = write_training_approval(
        args.config,
        args.profile,
        args.preflight,
        args.output,
        approve=True,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[approval] wrote {args.output}")


if __name__ == "__main__":
    main()
