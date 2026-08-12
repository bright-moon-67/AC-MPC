"""Export a DMC PyTorch Koopman checkpoint to a framework-neutral NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
    if not np.isfinite(value).all():
        raise ValueError("Koopman checkpoint contains NaN or Inf")
    return value


def _linear_layers(
    state: Mapping[str, torch.Tensor], prefix: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = sorted(
        int(key.removeprefix(prefix).removesuffix(".weight"))
        for key in state
        if key.startswith(prefix) and key.endswith(".weight")
    )
    result = []
    for index in indices:
        weight_key = f"{prefix}{index}.weight"
        bias_key = f"{prefix}{index}.bias"
        if bias_key not in state:
            raise ValueError(f"Missing bias for {weight_key}")
        result.append((_numpy(state[weight_key]), _numpy(state[bias_key])))
    if not result:
        raise ValueError(f"No linear layers found under {prefix!r}")
    return result


def export_checkpoint(source: Path, output: Path) -> dict[str, Any]:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Koopman checkpoint must contain a mapping")
    if payload.get("kind") != "dmc_k_step_koopman":
        raise ValueError("Only formal DMC Koopman best checkpoints are supported")
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("Koopman checkpoint is missing architecture")
    if architecture.get("architecture") != "fullA_history_v2_adapted":
        raise ValueError("Unsupported Koopman architecture")
    model_state = payload.get("model_state")
    reward_state = payload.get("reward_model_state")
    if not isinstance(model_state, Mapping) or not isinstance(reward_state, Mapping):
        raise ValueError("Checkpoint is missing model or reward-model state")
    encoder = _linear_layers(model_state, "encoder.")
    reward = _linear_layers(reward_state, "network.")
    normalizer = payload.get("normalizer")
    if not isinstance(normalizer, Mapping):
        raise ValueError("Checkpoint is missing its state normalizer")

    metadata = {
        "kind": "playground_koopman_export_v1",
        "source_path": str(source.resolve()),
        "source_sha256": _sha256(source),
        "source_checkpoint_kind": payload.get("checkpoint_kind"),
        "task": payload.get("task"),
        "architecture": dict(architecture),
        "reward_model_architecture": payload.get("reward_model_architecture"),
        "reward_model_input_contract": payload.get("reward_model_input_contract"),
        "best_epoch": payload.get("best_epoch"),
        "best_validation_joint_objective": payload.get(
            "best_validation_joint_objective"
        ),
        "best_validation_rollout_normalized_mse": payload.get(
            "best_validation_rollout_normalized_mse"
        ),
        "best_validation_reward_metrics": payload.get(
            "best_validation_reward_metrics"
        ),
        "dataset_sha256": payload.get("dataset_sha256"),
        "source_protocol_fingerprint": payload.get("protocol_fingerprint"),
        "encoder_layer_count": len(encoder),
        "reward_layer_count": len(reward),
    }
    arrays: dict[str, np.ndarray] = {
        "A": _numpy(model_state["A"]),
        "B": _numpy(model_state["B"]),
        "C": _numpy(model_state["C"]),
        "center": _numpy(torch.as_tensor(normalizer["center"])),
        "scale": _numpy(torch.as_tensor(normalizer["scale"])),
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    }
    for layer, (weight, bias) in enumerate(encoder):
        arrays[f"encoder_{layer}_weight"] = weight
        arrays[f"encoder_{layer}_bias"] = bias
    for layer, (weight, bias) in enumerate(reward):
        arrays[f"reward_{layer}_weight"] = weight
        arrays[f"reward_{layer}_bias"] = bias

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
    temporary.replace(output)
    manifest = {**metadata, "export_path": str(output.resolve()), "export_sha256": _sha256(output)}
    manifest_path = output.with_suffix(output.suffix + ".json")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
