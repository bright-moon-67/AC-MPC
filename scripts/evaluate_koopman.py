#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.data.build_sequences import Normalizer, valid_window_starts
from antmaze_ac.data.windows import load_npz_dataset
from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256


def grouped_mse(error: np.ndarray, observation_dim: int, action_dim: int) -> dict[str, float]:
    return {
        "xy": float(np.mean(error[..., :2] ** 2)),
        "other_state": float(np.mean(error[..., 2:observation_dim] ** 2)),
        "previous_action": float(np.mean(error[..., -action_dim:] ** 2)),
        "all": float(np.mean(error**2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/antmaze-umaze-v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--horizons", default="1,5,10,20,25")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to koopman.eval_batch_size stored in the checkpoint config",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-curves", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    model, payload = load_checkpoint(args.checkpoint)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model.to(device).freeze_dynamics()
    batch_size = int(args.batch_size or payload["config"]["koopman"]["eval_batch_size"])
    state_stats = payload["normalizers"]["state"]
    normalizer = Normalizer(np.asarray(state_stats["mean"]), np.asarray(state_stats["std"]))
    dataset = load_npz_dataset(Path(args.data) / f"{args.split}.npz")
    observation_dim = model.state_dim - model.action_dim
    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "split": args.split,
        "config": payload["config"],
        "normalizers": payload["normalizers"],
        "device": str(device),
        "batch_size": batch_size,
        "horizons": {},
    }
    curve_payload = None
    parsed_horizons = list(map(int, args.horizons.split(",")))
    for horizon in parsed_horizons:
        starts = valid_window_starts(dataset, horizon)
        squared_error_chunks = []
        naive_error_chunks = []
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            x0 = np.stack([dataset.x[index] for index in batch_starts])
            actions = np.stack(
                [dataset.delta_action[index : index + horizon] for index in batch_starts]
            )
            targets = np.stack([dataset.next_x[index : index + horizon] for index in batch_starts])
            x0_normalized = torch.as_tensor(normalizer.normalize(x0), dtype=torch.float32, device=device)
            action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
            with torch.no_grad():
                predictions, _ = model.rollout(x0_normalized, action_tensor)
            predictions = normalizer.denormalize(predictions.cpu().numpy())
            if args.save_curves and horizon == max(parsed_horizons) and curve_payload is None:
                curve_payload = {
                    "start_index": np.asarray([batch_starts[0]], dtype=np.int64),
                    "initial_state": x0[:1],
                    "target": targets[:1],
                    "prediction": predictions[:1],
                    "naive_hold": np.repeat(x0[:1, None, :], horizon, axis=1),
                    "delta_action": actions[:1],
                }
            squared_error_chunks.append(predictions - targets)
            naive_error_chunks.append(np.repeat(x0[:, None, :], horizon, axis=1) - targets)
        error = np.concatenate(squared_error_chunks)
        naive_error = np.concatenate(naive_error_chunks)
        metrics["horizons"][str(horizon)] = {
            "samples": len(starts),
            "koopman_mse": grouped_mse(error, observation_dim, model.action_dim),
            "naive_hold_mse": grouped_mse(naive_error, observation_dim, model.action_dim),
            "per_step_all_mse": np.mean(error**2, axis=(0, 2)).tolist(),
        }
    output = Path(args.output or Path(args.checkpoint).parent / f"evaluation_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if curve_payload is not None:
        curve_path = output.with_name(output.stem + "_curves.npz")
        np.savez_compressed(curve_path, **curve_payload)
        metrics["prediction_curves_npz"] = str(curve_path.resolve())
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            target = curve_payload["target"][0]
            prediction = curve_payload["prediction"][0]
            naive = curve_payload["naive_hold"][0]
            steps = np.arange(1, len(target) + 1)
            figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            for index, label in enumerate(("x", "y")):
                axes[index].plot(steps, target[:, index], label="target", linewidth=2)
                axes[index].plot(steps, prediction[:, index], label="Koopman")
                axes[index].plot(steps, naive[:, index], label="naive hold", linestyle="--")
                axes[index].set_ylabel(label)
                axes[index].grid(alpha=0.25)
            axes[0].legend()
            axes[-1].set_xlabel("prediction step")
            figure.tight_layout()
            plot_path = output.with_name(output.stem + "_xy_curve.png")
            figure.savefig(plot_path, dpi=160)
            plt.close(figure)
            metrics["prediction_curve_png"] = str(plot_path.resolve())
        except ImportError:
            metrics["prediction_curve_png"] = None
            metrics["prediction_curve_note"] = "Install the plots extra to render PNG."
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
