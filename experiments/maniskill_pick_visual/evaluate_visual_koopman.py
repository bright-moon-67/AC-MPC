"""Evaluate a trained visual controlled Koopman checkpoint on held-out episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.koopman.checkpoint import load_checkpoint
from antmaze_ac.koopman.visual_model import VisualLinearKoopman
from experiments.maniskill_pick_visual.dataset import VisualWindowDataset
from experiments.maniskill_pick_visual.train_visual_koopman import (
    TrainConfig,
    _loader,
    _resolve_device,
    evaluation_horizons,
    evaluate_loss,
    make_evaluation_datasets,
    preload_dataset,
    rollout_metrics,
    validate_train_config,
    validate_data_provenance,
)


def evaluate(
    checkpoint: str | Path,
    *,
    split: str = "test",
    device_name: str = "auto",
) -> dict[str, object]:
    device = _resolve_device(device_name)
    model, payload = load_checkpoint(checkpoint, map_location=device)
    if not isinstance(model, VisualLinearKoopman):
        raise TypeError("Checkpoint is not a VisualLinearKoopman model")
    saved = payload["config"]
    config_fields = TrainConfig.__dataclass_fields__
    kwargs = {name: saved[name] for name in config_fields if name in saved}
    kwargs["trajectory_h5"] = Path(saved["trajectory_h5"])
    kwargs["feature_h5"] = Path(saved["feature_h5"])
    kwargs["output_dir"] = Path(saved["output_dir"])
    if "encoder_hidden_dims" in kwargs:
        kwargs["encoder_hidden_dims"] = tuple(kwargs["encoder_hidden_dims"])
    config = TrainConfig(**kwargs)
    validate_train_config(config)
    trajectory_digest, feature_digest = validate_data_provenance(config)
    expected_trajectory = saved.get("trajectory_sha256")
    expected_features = saved.get("feature_sha256")
    if expected_trajectory is not None and trajectory_digest != expected_trajectory:
        raise RuntimeError("Trajectory SHA256 differs from the checkpoint provenance")
    if expected_features is not None and feature_digest != expected_features:
        raise RuntimeError("Feature-cache SHA256 differs from the checkpoint provenance")
    splits = payload.get("config", {}).get("episode_splits")
    if splits is None:
        report_path = Path(checkpoint).with_name("report.json")
        if not report_path.is_file():
            raise KeyError("Episode splits are absent and report.json was not found")
        splits = json.loads(report_path.read_text(encoding="utf-8"))["data"][
            "episode_splits"
        ]
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}; choose from {sorted(splits)}")
    normalizers = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in payload["normalizers"].items()
    }
    lazy_dataset = VisualWindowDataset(
        config.trajectory_h5,
        config.feature_h5,
        splits[split],
        config.horizon,
        normalizers,
        feature_key=config.feature_key,
    )
    dataset = preload_dataset(lazy_dataset) if config.preload else lazy_dataset
    evaluation_datasets = make_evaluation_datasets(
        config,
        splits[split],
        normalizers,
        reuse={config.horizon: dataset},
    )
    try:
        loader = _loader(dataset, config=config, shuffle=False)
        evaluation_loaders = {
            horizon: _loader(value, config=config, shuffle=False)
            for horizon, value in evaluation_datasets.items()
        }
        model.to(device).eval()
        return {
            "checkpoint": str(Path(checkpoint).resolve()),
            "split": split,
            "loss": evaluate_loss(model, loader, device, config),
            "metrics": rollout_metrics(
                model,
                evaluation_loaders,
                normalizers,
                device,
                primary_horizon=config.horizon,
                requested_horizons=evaluation_horizons(config.horizon),
            ),
        }
    finally:
        closed: set[int] = set()
        for value in (dataset, *evaluation_datasets.values()):
            if id(value) not in closed:
                value.close()
                closed.add(id(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.checkpoint, split=args.split, device_name=args.device),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
