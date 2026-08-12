"""Canonical serialization and identity helpers for DMC environment protocols."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize metadata deterministically for checkpoints and datasets."""

    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def protocol_fingerprint(protocol: Mapping[str, Any]) -> str:
    """SHA-256 identity of the live environment protocol, excluding collection policy."""

    return hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()


def protocol_fingerprint_from_json(protocol_json: str) -> str:
    """Hash a canonical environment-protocol JSON string after validating it."""

    decoded = json.loads(protocol_json)
    if not isinstance(decoded, dict):
        raise ValueError("environment_protocol_json must contain a mapping")
    canonical = canonical_json(decoded)
    if canonical != protocol_json:
        raise ValueError("environment_protocol_json is not canonical")
    return hashlib.sha256(protocol_json.encode("utf-8")).hexdigest()
