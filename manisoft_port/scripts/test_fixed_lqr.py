#!/usr/bin/env python
"""Run the fixed-cost Koopman-LQR closed loop for complete episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.data.build_sequences import Normalizer
from antmaze_ac.envs.factory import make_antmaze_env
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.ac_koopman_policy import GainHoldController
from antmaze_ac.rl.serialization import make_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--backend", choices=["auto", "legacy", "modern"], default="auto")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--gain-update-interval", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    policy, koopman_payload = make_policy(args.koopman_checkpoint, device)
    policy.eval()
    # Zero-initialized CostActor output is state-independent: fixed diagonal
    # stage Hessian and p=0. Freeze it to make that invariant explicit.
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(False)
    controller = GainHoldController(policy, args.gain_update_interval)
    stats = koopman_payload["normalizers"]["state"]
    normalizer = Normalizer(np.asarray(stats["mean"]), np.asarray(stats["std"]))
    env = make_antmaze_env(
        koopman_payload["config"]["experiment"]["env_id"],
        backend=args.backend,
    )

    episodes = []
    all_steps = []
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=episode)
        controller.reset()
        episode_rows = []
        total_reward = 0.0
        while True:
            current_observation = observation.copy()
            current_tensor = torch.as_tensor(
                current_observation,
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                requested_delta = controller.act(current_tensor)
            assert controller.last_lqr is not None
            lqr = controller.last_lqr
            requested_array = requested_delta.cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(requested_array)
            applied_delta = np.asarray(info["applied_delta_action"], dtype=np.float32)
            with torch.no_grad():
                predicted_normalized, _ = policy.koopman(
                    torch.as_tensor(
                        normalizer.normalize(current_observation),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(applied_delta, dtype=torch.float32, device=device),
                )
            predicted = normalizer.denormalize(predicted_normalized.cpu().numpy())
            prediction_error = predicted - observation
            row = {
                "reward": float(reward),
                "requested_delta_norm": float(np.linalg.norm(requested_array)),
                "applied_delta_energy": float(np.dot(applied_delta, applied_delta)),
                "action_energy": float(
                    np.dot(info["applied_action"], info["applied_action"])
                ),
                "saturation": float(info["action_saturation_ratio"]),
                "koopman_one_step_mse": float(np.mean(prediction_error**2)),
                "koopman_one_step_xy_mse": float(np.mean(prediction_error[:2] ** 2)),
                "dare_residual": float(torch.max(lqr.dare.residual)),
                "dare_relative_residual": float(
                    torch.max(lqr.dare.relative_residual)
                ),
                "condition_number": float(torch.max(lqr.dare.condition_number)),
                "spectral_radius": float(
                    torch.max(lqr.dare.closed_loop_spectral_radius)
                ),
                "dare_retry": float(controller.last_solver_retry),
                "dare_fallback": float(controller.last_solver_fallback),
            }
            episode_rows.append(row)
            all_steps.append(row)
            total_reward += float(reward)
            if terminated or truncated:
                break
        episodes.append(
            {
                "episode": episode,
                "return": total_reward,
                "success": float(total_reward > 0),
                "length": len(episode_rows),
                "saturation_rate": float(
                    np.mean([row["saturation"] for row in episode_rows])
                ),
            }
        )

    numeric_step_keys = tuple(all_steps[0])
    first_steps = all_steps[: min(10, len(all_steps))]
    summary = {
        "method": "fixed_cost_koopman_lqr",
        "koopman_checkpoint": str(Path(args.koopman_checkpoint).resolve()),
        "koopman_checkpoint_sha256": sha256(args.koopman_checkpoint),
        "episodes": args.episodes,
        "backend": args.backend,
        "gain_update_interval": args.gain_update_interval,
        "finite": bool(
            all(np.isfinite(list(row.values())).all() for row in all_steps)
        ),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "sparse_return_mean": float(np.mean([row["return"] for row in episodes])),
        "episode_length_mean": float(np.mean([row["length"] for row in episodes])),
        "initial_10_step_saturation_rate": float(
            np.mean([row["saturation"] for row in first_steps])
        ),
        **{
            f"{key}_mean": float(np.mean([row[key] for row in all_steps]))
            for key in numeric_step_keys
        },
        **{
            f"{key}_max": float(np.max([row[key] for row in all_steps]))
            for key in (
                "saturation",
                "koopman_one_step_mse",
                "dare_residual",
                "dare_relative_residual",
                "condition_number",
                "spectral_radius",
                "dare_retry",
                "dare_fallback",
            )
        },
        "episode_results": episodes,
    }
    output = Path(
        args.output
        or Path(args.koopman_checkpoint).parent
        / f"fixed_lqr_{args.backend}_interval{args.gain_update_interval}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()
    if not summary["finite"]:
        raise FloatingPointError("Fixed-cost loop produced NaN or Inf")
    if summary["initial_10_step_saturation_rate"] >= 1.0:
        raise RuntimeError("Fixed-cost controller immediately saturated every action dimension")


if __name__ == "__main__":
    main()
