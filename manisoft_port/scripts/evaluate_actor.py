#!/usr/bin/env python
"""Evaluate Actor-Critic Koopman-LQR or Delta-PPO on original sparse reward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.evaluation.path_plot import (
    antmaze_geometry,
    path_progress,
    save_path_diagnostics,
    target_goal,
)
from antmaze_ac.envs.factory import make_antmaze_env
from antmaze_ac.koopman.checkpoint import sha256
from antmaze_ac.rl.ac_koopman_policy import GainHoldController
from antmaze_ac.rl.serialization import (
    load_actor_checkpoint,
    load_delta_checkpoint,
    load_td3_bc_checkpoint,
)


def normalized_score(env_id: str, mean_return: float, backend: str) -> tuple[float, str]:
    if backend == "legacy":
        try:
            import d4rl
            import d4rl.infos  # noqa: F401 - populates d4rl.infos in D4RL 1.1

            return 100.0 * float(d4rl.get_normalized_score(env_id, mean_return)), "d4rl"
        except Exception:
            pass
    # D4RL AntMaze v2 reference min/max are 0 and 1.
    return 100.0 * mean_return, "antmaze_v2_reference_0_1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--method",
        choices=["auto", "actor", "td3_bc", "delta_ppo"],
        default="auto",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--backend", choices=["auto", "legacy", "modern"], default="auto")
    parser.add_argument("--gain-update-interval", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--plot-paths",
        type=int,
        default=0,
        help="Save XY trajectory subplots for the first N episodes.",
    )
    parser.add_argument("--path-plot-output", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    raw_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    method = args.method
    if method == "auto":
        checkpoint_method = raw_payload.get("method")
        if checkpoint_method == "delta_ppo":
            method = "delta_ppo"
        elif checkpoint_method == "td3_bc_koopman_lqr":
            method = "td3_bc"
        else:
            method = "actor"
    controller = None
    if method == "actor":
        policy, payload, koopman_payload = load_actor_checkpoint(args.checkpoint, device)
        controller = GainHoldController(policy, args.gain_update_interval)
    elif method == "td3_bc":
        policy, payload, koopman_payload = load_td3_bc_checkpoint(
            args.checkpoint,
            device,
        )
        controller = GainHoldController(policy, args.gain_update_interval)
    else:
        policy, payload, koopman_payload = load_delta_checkpoint(args.checkpoint, device)
        if args.gain_update_interval != 1:
            raise ValueError("gain_update_interval only applies to Actor-Critic Koopman-LQR")
    policy.eval()
    config = payload["config"]
    env_id = config["experiment"]["env_id"]
    env = make_antmaze_env(env_id, backend=args.backend)
    resolved_backend = (
        "legacy" if env.env.__class__.__name__ == "LegacyGymAdapter" else "modern"
    )
    rows = []
    plotted_paths = []
    dare_residuals = []
    dare_relative_residuals = []
    dare_conditions = []
    spectral_radii = []
    dare_failure_count = 0
    dare_retry_count = 0
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed_offset + episode)
        goal = target_goal(env)
        xy_path = [np.asarray(observation[:2], dtype=np.float64)]
        if controller is not None:
            controller.reset()
        total_reward = 0.0
        delta_energy = action_energy = saturation = 0.0
        length = 0
        while True:
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                if controller is not None:
                    delta = controller.act(observation_tensor)
                    assert controller.last_lqr is not None
                    lqr = controller.last_lqr
                    if controller.last_gain_recomputed:
                        dare_failure_count += int(controller.last_solver_fallback)
                        dare_retry_count += int(controller.last_solver_retry)
                        dare_residuals.append(float(torch.max(lqr.dare.residual)))
                        dare_relative_residuals.append(
                            float(torch.max(lqr.dare.relative_residual))
                        )
                        dare_conditions.append(
                            float(torch.max(lqr.dare.condition_number))
                        )
                        spectral_radii.append(
                            float(torch.max(lqr.dare.closed_loop_spectral_radius))
                        )
                else:
                    delta = policy(observation_tensor).mean
            delta_array = delta.cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(delta_array)
            xy_path.append(np.asarray(observation[:2], dtype=np.float64))
            total_reward += float(reward)
            delta_energy += float(np.dot(delta_array, delta_array))
            action_energy += float(np.dot(info["applied_action"], info["applied_action"]))
            saturation += float(info["action_saturation_ratio"])
            length += 1
            if terminated or truncated:
                break
        xy_array = np.stack(xy_path)
        progress = path_progress(xy_array, goal)
        row = {
            "episode": episode,
            "seed": args.seed_offset + episode,
            "return": total_reward,
            "success": float(total_reward > 0),
            "length": length,
            "delta_action_energy": delta_energy / length,
            "action_energy": action_energy / length,
            "saturation_rate": saturation / length,
            **progress,
        }
        rows.append(row)
        if len(plotted_paths) < args.plot_paths and goal is not None:
            plotted_paths.append(
                {
                    **row,
                    "xy": xy_array,
                    "goal": goal,
                }
            )
    mean_return = float(np.mean([row["return"] for row in rows]))
    score, score_source = normalized_score(env_id, mean_return, resolved_backend)
    numeric_keys = (
        "return",
        "success",
        "length",
        "delta_action_energy",
        "action_energy",
        "saturation_rate",
        "path_length_xy",
        "start_goal_distance",
        "final_goal_distance",
        "minimum_goal_distance",
        "goal_progress_fraction",
    )
    summary = {
        "method": method,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "koopman_checkpoint": payload.get("koopman_checkpoint"),
        "koopman_checkpoint_sha256": payload.get("koopman_checkpoint_sha256"),
        "training_seed": payload.get("seed"),
        "training_runtime": payload.get("runtime"),
        "episodes": args.episodes,
        "backend": args.backend,
        "resolved_backend": resolved_backend,
        "env_id": env_id,
        "gain_update_interval": args.gain_update_interval,
        **{
            f"{key}_mean": float(np.mean([row[key] for row in rows]))
            for key in numeric_keys
        },
        **{
            f"{key}_std": float(np.std([row[key] for row in rows]))
            for key in numeric_keys
        },
        "d4rl_normalized_score": score,
        "normalized_score_source": score_source,
        "dare_failure_count": dare_failure_count if controller is not None else None,
        "dare_retry_count": dare_retry_count if controller is not None else None,
        "dare_residual_max": max(dare_residuals) if dare_residuals else None,
        "dare_relative_residual_max": max(dare_relative_residuals)
        if dare_relative_residuals
        else None,
        "dare_condition_max": max(dare_conditions) if dare_conditions else None,
        "closed_loop_spectral_radius_max": max(spectral_radii)
        if spectral_radii
        else None,
        "config": config,
        "episodes_raw": rows,
    }
    output = Path(args.output or Path(args.checkpoint).parent / "evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if plotted_paths:
        geometry = antmaze_geometry(env)
        if geometry is None:
            raise RuntimeError("Could not extract AntMaze geometry for path plot")
        plot_output = Path(
            args.path_plot_output
            or output.with_name(f"{output.stem}_paths.png")
        )
        png_path, npz_path = save_path_diagnostics(
            plotted_paths,
            geometry,
            plot_output,
        )
        summary["path_plot_png"] = str(png_path.resolve())
        summary["path_data_npz"] = str(npz_path.resolve())
        summary["path_plot_episodes"] = len(plotted_paths)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
