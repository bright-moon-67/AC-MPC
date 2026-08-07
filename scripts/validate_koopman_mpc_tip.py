#!/usr/bin/env python
"""Minimal receding-horizon Koopman-MPC check for a small tip displacement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256


TIP_INDICES = (10, 21, 32)


def parse_vector(text: str, *, length: int, name: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} comma-separated values")
    return values


def clipped_action_sequence(
    requested_deltas: torch.Tensor,
    previous_action: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same cumulative absolute-action clipping as DeltaActionWrapper."""

    absolute_action = previous_action
    applied_deltas = []
    applied_actions = []
    for requested_delta in requested_deltas:
        next_action = torch.clamp(
            absolute_action + requested_delta,
            min=action_low,
            max=action_high,
        )
        applied_deltas.append(next_action - absolute_action)
        applied_actions.append(next_action)
        absolute_action = next_action
    return torch.stack(applied_deltas), torch.stack(applied_actions)


def optimize_mpc(
    *,
    model,
    state: np.ndarray,
    goal: np.ndarray,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    horizon: int,
    position_weight: float,
    control_weight: float,
    max_delta: float,
    iterations: int,
    learning_rate: float,
    initial_decision: torch.Tensor | None,
) -> dict[str, torch.Tensor | float]:
    device = state_mean.device
    dtype = state_mean.dtype
    state_tensor = torch.as_tensor(state, dtype=dtype, device=device)
    goal_tensor = torch.as_tensor(goal, dtype=dtype, device=device)
    previous_action = state_tensor[-model.action_dim :]
    if initial_decision is None:
        decision = torch.zeros(
            horizon,
            model.action_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
    else:
        if initial_decision.shape != (horizon, model.action_dim):
            raise ValueError("Warm-start decision has the wrong shape")
        decision = initial_decision.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([decision], lr=learning_rate)
    normalized_state = (state_tensor - state_mean) / state_std
    tip_indices = torch.as_tensor(TIP_INDICES, device=device)

    for _ in range(iterations):
        requested_deltas = max_delta * torch.tanh(decision)
        applied_deltas, _ = clipped_action_sequence(
            requested_deltas,
            previous_action,
            action_low,
            action_high,
        )
        predicted_normalized, _ = model.rollout(
            normalized_state.unsqueeze(0),
            applied_deltas.unsqueeze(0),
        )
        predicted_states = predicted_normalized[0] * state_std + state_mean
        predicted_tips = predicted_states.index_select(-1, tip_indices)
        position_cost = position_weight * (predicted_tips - goal_tensor).square().sum()
        control_cost = control_weight * applied_deltas.square().sum()
        loss = position_cost + control_cost
        if not torch.isfinite(loss):
            raise FloatingPointError("MPC objective became NaN or Inf")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        requested_deltas = max_delta * torch.tanh(decision)
        applied_deltas, applied_actions = clipped_action_sequence(
            requested_deltas,
            previous_action,
            action_low,
            action_high,
        )
        predicted_normalized, _ = model.rollout(
            normalized_state.unsqueeze(0),
            applied_deltas.unsqueeze(0),
        )
        predicted_states = predicted_normalized[0] * state_std + state_mean
        predicted_tips = predicted_states.index_select(-1, tip_indices)
        position_cost = position_weight * (predicted_tips - goal_tensor).square().sum()
        control_cost = control_weight * applied_deltas.square().sum()
        total_cost = float(position_cost + control_cost)

    return {
        "decision": decision.detach(),
        "requested_deltas": requested_deltas.detach(),
        "applied_deltas": applied_deltas.detach(),
        "applied_actions": applied_actions.detach(),
        "predicted_tips": predicted_tips.detach(),
        "cost": total_cost,
    }


def write_tip_csv(
    path: Path,
    tip_positions: np.ndarray,
    goal: np.ndarray,
    distances: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "tip_x", "tip_y", "tip_z", "goal_x", "goal_y", "goal_z", "distance"]
        )
        for step, (tip, distance) in enumerate(zip(tip_positions, distances)):
            writer.writerow([step, *map(float, tip), *map(float, goal), float(distance)])


def write_control_csv(
    path: Path,
    requested_deltas: np.ndarray,
    applied_deltas: np.ndarray,
    applied_actions: np.ndarray,
    saturation_ratios: np.ndarray,
    mpc_costs: np.ndarray,
) -> None:
    action_dim = requested_deltas.shape[1]
    header = ["step", "mpc_cost", "saturation_ratio"]
    header += [f"requested_delta_{index:02d}" for index in range(action_dim)]
    header += [f"applied_delta_{index:02d}" for index in range(action_dim)]
    header += [f"applied_action_{index:02d}" for index in range(action_dim)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for step in range(len(requested_deltas)):
            writer.writerow(
                [
                    step,
                    float(mpc_costs[step]),
                    float(saturation_ratios[step]),
                    *map(float, requested_deltas[step]),
                    *map(float, applied_deltas[step]),
                    *map(float, applied_actions[step]),
                ]
            )


def save_plot(
    path: Path,
    tip_positions: np.ndarray,
    goal: np.ndarray,
    distances: np.ndarray,
    applied_deltas: np.ndarray,
    applied_actions: np.ndarray,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    steps = np.arange(len(tip_positions))
    control_steps = np.arange(len(applied_deltas))
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    for dimension, label in enumerate(("x", "y", "z")):
        axes[0].plot(steps, tip_positions[:, dimension], label=f"tip_{label}")
        axes[0].axhline(goal[dimension], linestyle="--", label=f"goal_{label}")
    axes[0].set_ylabel("position (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, distances)
    axes[1].set_ylabel("distance (m)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(control_steps, np.linalg.norm(applied_deltas, axis=1), label="||applied delta||")
    axes[2].plot(control_steps, np.max(np.abs(applied_actions), axis=1), label="max |action|")
    axes[2].set_xlabel("control step")
    axes[2].set_ylabel("control")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal H=10 Koopman-MPC validation for a small ManiSoft tip displacement"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", default="runs/manisoft_koopman_mpc_tip_x20mm")
    parser.add_argument("--target-offset", default="0.02,0.0,0.0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--position-weight", type=float, default=100.0)
    parser.add_argument("--control-weight", type=float, default=0.1)
    parser.add_argument("--max-delta", type=float, default=0.02)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--optimizer-iterations", type=int, default=100)
    parser.add_argument("--optimizer-learning-rate", type=float, default=0.05)
    parser.add_argument("--success-threshold", type=float, default=0.002)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.steps < 1 or args.horizon < 1 or args.optimizer_iterations < 1:
        raise ValueError("steps, horizon and optimizer-iterations must be positive")
    if min(
        args.position_weight,
        args.control_weight,
        args.max_delta,
        args.absolute_action_limit,
        args.optimizer_learning_rate,
        args.success_threshold,
    ) <= 0:
        raise ValueError("All MPC weights, limits, learning rate and threshold must be positive")

    target_offset = parse_vector(args.target_offset, length=3, name="--target-offset")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not scenario.is_file():
        raise FileNotFoundError(f"Scenario not found: {scenario}")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model = model.to(device).freeze_dynamics()
    if (model.state_dim, model.action_dim) != (59, 18):
        raise ValueError(
            f"This ManiSoft check requires state/action dimensions (59,18), got "
            f"({model.state_dim},{model.action_dim})"
        )
    stats = payload["normalizers"]["state"]
    state_mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
    state_std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
    if state_mean.shape != (59,) or state_std.shape != (59,):
        raise ValueError("Checkpoint state normalizer must have 59 dimensions")

    # Keep the optimization core importable without a local ManiSoft install.
    from antmaze_ac.envs.manisoft_tracking_env import make_manisoft_tracking_env

    env = make_manisoft_tracking_env(
        scenario,
        target_offset=target_offset,
        episode_steps=args.steps,
        absolute_action_limit=args.absolute_action_limit,
    )
    try:
        observation, reset_info = env.reset()
        dynamics_state = np.asarray(observation[:59], dtype=np.float32)
        initial_tip = dynamics_state[np.asarray(TIP_INDICES)]
        goal = initial_tip + np.asarray(target_offset, dtype=np.float32)
        environment_goal = np.asarray(reset_info["target_tip"], dtype=np.float32)
        if not np.allclose(goal, environment_goal, atol=1e-7):
            raise RuntimeError("Environment target does not match the requested tip offset")

        base_action_space = env.unwrapped.action_space
        action_low = torch.as_tensor(base_action_space.low, dtype=torch.float32, device=device)
        action_high = torch.as_tensor(base_action_space.high, dtype=torch.float32, device=device)

        tip_positions = [initial_tip.copy()]
        distances = [float(np.linalg.norm(initial_tip - goal))]
        requested_history = []
        applied_delta_history = []
        applied_action_history = []
        saturation_history = []
        mpc_cost_history = []
        predicted_first_tip_history = []
        one_step_tip_error_history = []
        warm_start = None
        stop_reason = "max_steps"

        for step in range(args.steps):
            plan = optimize_mpc(
                model=model,
                state=dynamics_state,
                goal=goal,
                state_mean=state_mean,
                state_std=state_std,
                action_low=action_low,
                action_high=action_high,
                horizon=args.horizon,
                position_weight=args.position_weight,
                control_weight=args.control_weight,
                max_delta=args.max_delta,
                iterations=args.optimizer_iterations,
                learning_rate=args.optimizer_learning_rate,
                initial_decision=warm_start,
            )
            requested_plan = plan["requested_deltas"].cpu().numpy()
            applied_plan = plan["applied_deltas"].cpu().numpy()
            predicted_tips = plan["predicted_tips"].cpu().numpy()
            requested_delta = requested_plan[0]

            next_observation, _, terminated, truncated, info = env.step(requested_delta)
            actual_applied_delta = np.asarray(info["applied_delta_action"], dtype=np.float32)
            actual_applied_action = np.asarray(info["applied_action"], dtype=np.float32)
            if not np.allclose(actual_applied_delta, applied_plan[0], atol=1e-5):
                raise RuntimeError("Planner clipping and environment clipping disagree")

            dynamics_state = np.asarray(next_observation[:59], dtype=np.float32)
            tip = dynamics_state[np.asarray(TIP_INDICES)]
            distance = float(np.linalg.norm(tip - goal))
            predicted_first_tip = predicted_tips[0]

            tip_positions.append(tip.copy())
            distances.append(distance)
            requested_history.append(np.asarray(info["requested_delta_action"], dtype=np.float32))
            applied_delta_history.append(actual_applied_delta)
            applied_action_history.append(actual_applied_action)
            saturation_history.append(float(info["action_saturation_ratio"]))
            mpc_cost_history.append(float(plan["cost"]))
            predicted_first_tip_history.append(predicted_first_tip)
            one_step_tip_error_history.append(float(np.linalg.norm(predicted_first_tip - tip)))

            decision = plan["decision"]
            warm_start = torch.cat((decision[1:], torch.zeros_like(decision[:1])), dim=0)
            print(
                f"step={step + 1:03d} distance={distance:.6f} "
                f"mpc_cost={float(plan['cost']):.6f} "
                f"max_action={float(np.max(np.abs(actual_applied_action))):.4f}"
            )
            if terminated:
                stop_reason = "environment_success"
                break
            if truncated:
                stop_reason = "environment_time_limit"
                break
    finally:
        env.close()

    tip_array = np.asarray(tip_positions, dtype=np.float32)
    distance_array = np.asarray(distances, dtype=np.float32)
    requested_array = np.asarray(requested_history, dtype=np.float32)
    applied_delta_array = np.asarray(applied_delta_history, dtype=np.float32)
    applied_action_array = np.asarray(applied_action_history, dtype=np.float32)
    saturation_array = np.asarray(saturation_history, dtype=np.float32)
    cost_array = np.asarray(mpc_cost_history, dtype=np.float32)
    predicted_tip_array = np.asarray(predicted_first_tip_history, dtype=np.float32)
    one_step_error_array = np.asarray(one_step_tip_error_history, dtype=np.float32)
    success = bool(distance_array[-1] <= args.success_threshold)

    np.savez_compressed(
        output / "trajectory.npz",
        tip_position=tip_array,
        target_position=goal,
        distance=distance_array,
        requested_delta=requested_array,
        applied_delta=applied_delta_array,
        applied_action=applied_action_array,
        saturation_ratio=saturation_array,
        mpc_cost=cost_array,
        predicted_first_tip=predicted_tip_array,
        one_step_tip_prediction_error=one_step_error_array,
    )
    write_tip_csv(output / "tip_trajectory.csv", tip_array, goal, distance_array)
    write_control_csv(
        output / "controls.csv",
        requested_array,
        applied_delta_array,
        applied_action_array,
        saturation_array,
        cost_array,
    )
    plot_path = save_plot(
        output / "result.png",
        tip_array,
        goal,
        distance_array,
        applied_delta_array,
        applied_action_array,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "scenario": str(scenario),
        "device": str(device),
        "horizon": args.horizon,
        "position_weight": args.position_weight,
        "control_weight": args.control_weight,
        "max_delta": args.max_delta,
        "absolute_action_limit": args.absolute_action_limit,
        "target_offset": list(target_offset),
        "initial_tip": initial_tip.tolist(),
        "target_tip": goal.tolist(),
        "steps_executed": len(requested_array),
        "initial_distance": float(distance_array[0]),
        "final_distance": float(distance_array[-1]),
        "minimum_distance": float(distance_array.min()),
        "mean_one_step_tip_prediction_error": (
            float(one_step_error_array.mean()) if len(one_step_error_array) else None
        ),
        "maximum_absolute_action": (
            float(np.max(np.abs(applied_action_array))) if len(applied_action_array) else 0.0
        ),
        "mean_saturation_ratio": (
            float(saturation_array.mean()) if len(saturation_array) else 0.0
        ),
        "success_threshold": args.success_threshold,
        "success": success,
        "stop_reason": stop_reason,
        "plot": plot_path,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
