#!/usr/bin/env python
"""Validate fixed-reference tracking with condensed-QP Koopman-MPC."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sparse
import torch

from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256

TIP_INDICES = (30, 31, 32)
PHYSICAL_DIM = 45
ACTION_DIM = 18
DYNAMICS_STATE_DIM = PHYSICAL_DIM + ACTION_DIM


def compress_physical_to_model(
    physical: np.ndarray,
    model_physical_dim: int,
) -> np.ndarray:
    if model_physical_dim != PHYSICAL_DIM:
        raise ValueError(
            f"Unsupported model physical dimension {model_physical_dim}"
        )
    physical = np.asarray(physical, dtype=np.float32)
    if physical.ndim != 2 or physical.shape[1] != PHYSICAL_DIM:
        raise ValueError(
            f"physical state must have shape [T,{PHYSICAL_DIM}], got {physical.shape}"
        )
    return physical


def build_model_state(observation: np.ndarray, model_state_dim: int) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if model_state_dim != DYNAMICS_STATE_DIM:
        raise ValueError(f"Unsupported model state dimension {model_state_dim}")
    if observation.shape != (DYNAMICS_STATE_DIM + 3,):
        raise ValueError(
            f"Expected 66-D environment observation, got {observation.shape}"
        )
    return observation[:DYNAMICS_STATE_DIM]


class KoopmanMPCQP:
    """Condensed OSQP controller for ``z+ = A z + B delta_u``."""

    def __init__(
        self,
        *,
        model,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int,
        state_weight: float,
        action_weight: float,
        control_weight: float,
        smoothness_weight: float,
        track_tip_only: bool,
        action_tracking: bool,
        tip_state_scale: float,
        max_delta: float,
        qp_max_iterations: int,
        qp_absolute_tolerance: float,
        qp_relative_tolerance: float,
    ) -> None:
        setup_started = time.perf_counter()
        self.model = model
        self.device = state_mean.device
        self.dtype = state_mean.dtype
        self.state_mean_t = state_mean
        self.state_std_t = state_std
        self.state_mean = state_mean.detach().cpu().double().numpy()
        self.state_std = state_std.detach().cpu().double().numpy()
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        self.horizon = int(horizon)
        self.action_dim = int(model.action_dim)
        self.physical_dim = int(model.state_dim - model.action_dim)
        self.state_weight = float(state_weight)
        self.action_weight = float(action_weight)
        self.control_weight = float(control_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.action_tracking = bool(action_tracking)
        self.max_delta = float(max_delta)
        self.variable_dim = self.horizon * self.action_dim

        state_scales = np.ones(self.physical_dim, dtype=np.float64)
        if track_tip_only:
            state_scales.fill(0.0)
            state_scales[np.asarray(TIP_INDICES)] = 1.0
        else:
            state_scales[np.asarray(TIP_INDICES)] = float(tip_state_scale)
        self.state_scales = state_scales

        A = model.A.detach().cpu().double().numpy()
        B = model.B.detach().cpu().double().numpy()
        C_physical = model.C[: self.physical_dim].detach().cpu().double().numpy()
        powers = [np.eye(model.lifted_dim, dtype=np.float64)]
        for _ in range(self.horizon):
            powers.append(powers[-1] @ A)

        free_prediction = np.vstack(
            [C_physical @ powers[step + 1] for step in range(self.horizon)]
        )
        delta_prediction = np.zeros(
            (self.horizon * self.physical_dim, self.variable_dim),
            dtype=np.float64,
        )
        for step in range(self.horizon):
            rows = slice(step * self.physical_dim, (step + 1) * self.physical_dim)
            for control_step in range(step + 1):
                columns = slice(
                    control_step * self.action_dim,
                    (control_step + 1) * self.action_dim,
                )
                delta_prediction[rows, columns] = (
                    C_physical @ powers[step - control_step] @ B
                )

        physical_std = np.tile(self.state_std[: self.physical_dim], self.horizon)
        self.free_prediction = physical_std[:, None] * free_prediction
        self.delta_prediction = physical_std[:, None] * delta_prediction
        self.physical_mean = np.tile(
            self.state_mean[: self.physical_dim], self.horizon
        )
        self.cumulative = np.kron(
            np.tril(np.ones((self.horizon, self.horizon), dtype=np.float64)),
            np.eye(self.action_dim, dtype=np.float64),
        )

        if self.horizon > 1:
            difference = np.zeros(
                ((self.horizon - 1) * self.action_dim, self.variable_dim),
                dtype=np.float64,
            )
            identity = np.eye(self.action_dim, dtype=np.float64)
            for step in range(self.horizon - 1):
                rows = slice(step * self.action_dim, (step + 1) * self.action_dim)
                difference[
                    rows, step * self.action_dim : (step + 1) * self.action_dim
                ] = -identity
                difference[
                    rows, (step + 1) * self.action_dim : (step + 2) * self.action_dim
                ] = identity
        else:
            difference = np.zeros((0, self.variable_dim), dtype=np.float64)

        state_diagonal = np.tile(
            self.state_weight * np.square(self.state_scales), self.horizon
        )
        hessian = self.delta_prediction.T @ (
            state_diagonal[:, None] * self.delta_prediction
        )
        hessian += self.action_weight * (self.cumulative.T @ self.cumulative)
        hessian += self.control_weight * np.eye(self.variable_dim)
        hessian += self.smoothness_weight * (difference.T @ difference)
        hessian = 2.0 * ((hessian + hessian.T) * 0.5)

        constraint_matrix = sparse.vstack(
            (
                sparse.eye(self.variable_dim, format="csc"),
                sparse.csc_matrix(self.cumulative),
            ),
            format="csc",
        )
        initial_lower = np.concatenate(
            (
                np.full(self.variable_dim, -self.max_delta),
                np.tile(self.action_low, self.horizon),
            )
        )
        initial_upper = np.concatenate(
            (
                np.full(self.variable_dim, self.max_delta),
                np.tile(self.action_high, self.horizon),
            )
        )
        self.solver = osqp.OSQP()
        self.solver.setup(
            P=sparse.triu(sparse.csc_matrix(hessian), format="csc"),
            q=np.zeros(self.variable_dim, dtype=np.float64),
            A=constraint_matrix,
            l=initial_lower,
            u=initial_upper,
            verbose=False,
            polishing=True,
            warm_starting=True,
            max_iter=int(qp_max_iterations),
            eps_abs=float(qp_absolute_tolerance),
            eps_rel=float(qp_relative_tolerance),
        )
        self.setup_seconds = float(time.perf_counter() - setup_started)

    def _lift(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        normalized_state = (state_tensor - self.state_mean_t) / self.state_std_t
        with torch.no_grad():
            lifted = self.model.lift(normalized_state.unsqueeze(0))[0]
        return lifted.detach().cpu().double().numpy()

    def solve(
        self,
        *,
        state: np.ndarray,
        ref_state: np.ndarray,
        ref_action: np.ndarray,
        initial_deltas: np.ndarray | None,
    ) -> dict[str, np.ndarray | float | int | str]:
        started = time.perf_counter()
        state = np.asarray(state, dtype=np.float64)
        ref_state = np.asarray(ref_state, dtype=np.float64)
        ref_action = np.asarray(ref_action, dtype=np.float64)
        previous_action = state[-self.action_dim :]

        free_physical = self.free_prediction @ self._lift(state) + self.physical_mean
        state_error = free_physical - np.tile(
            ref_state[: self.physical_dim], self.horizon
        )
        state_diagonal = np.tile(
            self.state_weight * np.square(self.state_scales), self.horizon
        )
        linear = 2.0 * self.delta_prediction.T @ (state_diagonal * state_error)
        action_target = (
            ref_action if self.action_tracking else np.zeros_like(ref_action)
        )
        action_error = np.tile(previous_action - action_target, self.horizon)
        linear += 2.0 * self.action_weight * self.cumulative.T @ action_error

        lower = np.concatenate(
            (
                np.full(self.variable_dim, -self.max_delta),
                np.tile(self.action_low - previous_action, self.horizon),
            )
        )
        upper = np.concatenate(
            (
                np.full(self.variable_dim, self.max_delta),
                np.tile(self.action_high - previous_action, self.horizon),
            )
        )
        self.solver.update(q=linear, l=lower, u=upper)
        if initial_deltas is not None:
            initial_deltas = np.asarray(initial_deltas, dtype=np.float64)
            if initial_deltas.shape != (self.horizon, self.action_dim):
                raise ValueError("QP warm start has the wrong shape")
            self.solver.warm_start(x=initial_deltas.reshape(-1))
        result = self.solver.solve(raise_error=False)
        if result.info.status_val not in {1, 2} or result.x is None:
            raise RuntimeError(
                "Koopman MPC QP failed: "
                f"status={result.info.status!r}, iterations={result.info.iter}"
            )

        requested_deltas = np.asarray(result.x, dtype=np.float64).reshape(
            self.horizon, self.action_dim
        )
        requested_deltas = np.clip(requested_deltas, -self.max_delta, self.max_delta)
        applied_deltas = []
        applied_actions = []
        absolute_action = previous_action.copy()
        for requested_delta in requested_deltas:
            next_action = np.clip(
                absolute_action + requested_delta, self.action_low, self.action_high
            )
            applied_deltas.append(next_action - absolute_action)
            applied_actions.append(next_action)
            absolute_action = next_action
        applied_deltas = np.asarray(applied_deltas)
        applied_actions = np.asarray(applied_actions)

        flat_deltas = applied_deltas.reshape(-1)
        predicted_physical = (
            free_physical + self.delta_prediction @ flat_deltas
        ).reshape(self.horizon, self.physical_dim)
        weighted_state_error = (
            predicted_physical - ref_state[: self.physical_dim]
        ) * self.state_scales
        state_cost = self.state_weight * np.square(weighted_state_error).sum()
        action_cost = self.action_weight * np.square(
            applied_actions - action_target
        ).sum()
        control_cost = self.control_weight * np.square(applied_deltas).sum()
        smoothness_cost = self.smoothness_weight * np.square(
            np.diff(applied_deltas, axis=0)
        ).sum()

        return {
            "requested_deltas": requested_deltas.astype(np.float32),
            "applied_deltas": applied_deltas.astype(np.float32),
            "applied_actions": applied_actions.astype(np.float32),
            "cost": float(
                state_cost + action_cost + control_cost + smoothness_cost
            ),
            "qp_status": str(result.info.status),
            "qp_iterations": int(result.info.iter),
            "qp_solve_seconds": float(time.perf_counter() - started),
            "qp_internal_run_seconds": float(result.info.run_time),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Koopman-MPC tracking of a fixed-action reference equilibrium"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--reference", required=True, help="reference.npz")
    parser.add_argument("--output", default="runs/koopman_mpc_reference")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--state-weight", type=float, default=100.0)
    parser.add_argument("--tip-state-scale", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=30.0)
    parser.add_argument("--no-action-tracking", action="store_true")
    parser.add_argument("--track-tip-only", action="store_true")
    parser.add_argument("--control-weight", type=float, default=5.0)
    parser.add_argument("--smoothness-weight", type=float, default=10.0)
    parser.add_argument("--max-delta", type=float, default=0.002)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--qp-max-iterations", type=int, default=4000)
    parser.add_argument("--qp-absolute-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--qp-relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--success-threshold", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if min(args.steps, args.horizon, args.qp_max_iterations) < 1:
        raise ValueError("steps, horizon and qp-max-iterations must be positive")
    if args.action_weight < 0:
        raise ValueError("action-weight must be non-negative")
    if min(
        args.state_weight,
        args.control_weight,
        args.smoothness_weight,
        args.max_delta,
        args.absolute_action_limit,
        args.tip_state_scale,
        args.qp_absolute_tolerance,
        args.qp_relative_tolerance,
        args.success_threshold,
    ) <= 0:
        raise ValueError("All MPC weights, limits and thresholds must be positive")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    if not checkpoint.is_file() or not scenario.is_file() or not reference_path.is_file():
        raise FileNotFoundError("checkpoint / scenario / reference must exist")

    with np.load(reference_path, allow_pickle=False) as ref:
        ref_state = np.asarray(ref["reference_state"], dtype=np.float32)
        ref_action = np.asarray(ref["reference_action"], dtype=np.float32).reshape(-1)
    if ref_state.shape != (PHYSICAL_DIM,):
        raise ValueError(f"reference_state must be 45-D, got {ref_state.shape}")
    if ref_action.shape != (ACTION_DIM,):
        raise ValueError(f"reference_action must be 18-D, got {ref_action.shape}")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model = model.to(device).freeze_dynamics()
    if (model.state_dim, model.action_dim) != (DYNAMICS_STATE_DIM, ACTION_DIM):
        raise ValueError(
            "Requires state/action dimensions (63,18), "
            f"got ({model.state_dim},{model.action_dim})"
        )
    model_physical_dim = model.state_dim - model.action_dim
    stats = payload["normalizers"]["state"]
    state_mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
    state_std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
    if state_mean.shape != (DYNAMICS_STATE_DIM,) or state_std.shape != (
        DYNAMICS_STATE_DIM,
    ):
        raise ValueError("Checkpoint state normalizer must have 63 dimensions")

    ref_model_state = compress_physical_to_model(
        ref_state[None, :], model_physical_dim
    )[0]
    tip_idx = np.asarray(TIP_INDICES)
    ref_tip = ref_model_state[tip_idx].astype(np.float32, copy=False)

    from antmaze_ac.envs.manisoft_tracking_env import make_manisoft_tracking_env

    env = make_manisoft_tracking_env(
        scenario,
        target_offset=(0.01, 0.0, 0.0),
        episode_steps=args.steps,
        absolute_action_limit=args.absolute_action_limit,
    )
    try:
        validation_started = time.perf_counter()
        observation, _ = env.reset()
        dynamics_state = build_model_state(observation, model.state_dim)
        initial_tip = dynamics_state[tip_idx]
        initial_state_dist = float(np.linalg.norm(initial_tip - ref_tip))
        initial_action_dist = float(
            np.linalg.norm(dynamics_state[-ACTION_DIM:] - ref_action)
        )
        base_action_space = env.unwrapped.action_space
        action_low = np.asarray(base_action_space.low, dtype=np.float64)
        action_high = np.asarray(base_action_space.high, dtype=np.float64)
        planner = KoopmanMPCQP(
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            action_low=action_low,
            action_high=action_high,
            horizon=args.horizon,
            state_weight=args.state_weight,
            action_weight=args.action_weight,
            control_weight=args.control_weight,
            smoothness_weight=args.smoothness_weight,
            track_tip_only=args.track_tip_only,
            action_tracking=not args.no_action_tracking,
            tip_state_scale=args.tip_state_scale,
            max_delta=args.max_delta,
            qp_max_iterations=args.qp_max_iterations,
            qp_absolute_tolerance=args.qp_absolute_tolerance,
            qp_relative_tolerance=args.qp_relative_tolerance,
        )

        state_distances = [initial_state_dist]
        action_distances = [initial_action_dist]
        applied_delta_history = []
        applied_action_history = []
        mpc_cost_history = []
        qp_solve_time_history = []
        qp_internal_time_history = []
        qp_iteration_history = []
        simulation_time_history = []
        frame_time_history = []
        warm_start = None

        for step in range(args.steps):
            frame_start = time.perf_counter()
            plan = planner.solve(
                state=dynamics_state,
                ref_state=ref_model_state,
                ref_action=ref_action,
                initial_deltas=warm_start,
            )
            requested_delta = plan["requested_deltas"][0]
            applied_delta = plan["applied_deltas"][0]
            applied_action = plan["applied_actions"][0]

            simulation_start = time.perf_counter()
            observation, _, _, _, _ = env.step(requested_delta)
            simulation_seconds = time.perf_counter() - simulation_start
            dynamics_state = build_model_state(observation, model.state_dim)
            tip = dynamics_state[tip_idx]
            state_dist = float(np.linalg.norm(tip - ref_tip))
            action_dist = float(
                np.linalg.norm(dynamics_state[-ACTION_DIM:] - ref_action)
            )
            state_distances.append(state_dist)
            action_distances.append(action_dist)
            applied_delta_history.append(applied_delta)
            applied_action_history.append(applied_action)
            mpc_cost_history.append(float(plan["cost"]))
            qp_solve_time_history.append(float(plan["qp_solve_seconds"]))
            qp_internal_time_history.append(float(plan["qp_internal_run_seconds"]))
            qp_iteration_history.append(int(plan["qp_iterations"]))
            simulation_time_history.append(simulation_seconds)
            frame_time_history.append(time.perf_counter() - frame_start)
            warm_start = np.concatenate(
                (
                    plan["applied_deltas"][1:],
                    np.zeros_like(plan["applied_deltas"][:1]),
                ),
                axis=0,
            )
            if step % 25 == 0 or step == args.steps - 1:
                print(
                    f"step={step + 1:04d}/{args.steps} "
                    f"tip=({tip[0]:.4f},{tip[1]:.4f},{tip[2]:.4f}) "
                    f"ref=({ref_tip[0]:.4f},{ref_tip[1]:.4f},{ref_tip[2]:.4f}) "
                    f"state_dist={state_dist*1000:.1f}mm "
                    f"action_dist={action_dist*1000:.1f}mm "
                    f"mpc_cost={float(plan['cost']):.2f} "
                    f"qp={float(plan['qp_solve_seconds'])*1000:.2f}ms/"
                    f"{int(plan['qp_iterations'])}it",
                    flush=True,
                )

        state_array = np.asarray(state_distances, dtype=np.float32)
        action_array = np.asarray(action_distances, dtype=np.float32)
        qp_solve_times = np.asarray(qp_solve_time_history, dtype=np.float64)
        qp_internal_times = np.asarray(qp_internal_time_history, dtype=np.float64)
        qp_iterations = np.asarray(qp_iteration_history, dtype=np.int32)
        simulation_times = np.asarray(simulation_time_history, dtype=np.float64)
        frame_times = np.asarray(frame_time_history, dtype=np.float64)
        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "scenario": str(scenario),
            "reference": str(reference_path),
            "device": str(device),
            "horizon": args.horizon,
            "state_weight": args.state_weight,
            "tip_state_scale": args.tip_state_scale,
            "action_weight": args.action_weight,
            "action_tracking": not args.no_action_tracking,
            "control_weight": args.control_weight,
            "smoothness_weight": args.smoothness_weight,
            "max_delta": args.max_delta,
            "absolute_action_limit": args.absolute_action_limit,
            "mpc_solver": "osqp",
            "qp_setup_seconds": planner.setup_seconds,
            "qp_max_iterations": args.qp_max_iterations,
            "qp_absolute_tolerance": args.qp_absolute_tolerance,
            "qp_relative_tolerance": args.qp_relative_tolerance,
            "reference_tip": ref_tip.tolist(),
            "reference_action": ref_action.tolist(),
            "initial_tip": initial_tip.tolist(),
            "initial_state_distance": initial_state_dist,
            "final_state_distance": float(state_array[-1]),
            "minimum_state_distance": float(state_array.min()),
            "initial_action_distance": initial_action_dist,
            "final_action_distance": float(action_array[-1]),
            "success_threshold": args.success_threshold,
            "success": bool(state_array[-1] <= args.success_threshold),
            "stop_reason": "max_steps",
            "steps_executed": int(len(state_array) - 1),
            "mean_qp_solve_seconds": float(qp_solve_times.mean()),
            "p95_qp_solve_seconds": float(np.quantile(qp_solve_times, 0.95)),
            "max_qp_solve_seconds": float(qp_solve_times.max()),
            "mean_qp_internal_seconds": float(qp_internal_times.mean()),
            "mean_qp_iterations": float(qp_iterations.mean()),
            "max_qp_iterations_used": int(qp_iterations.max()),
            "mean_simulation_step_seconds": float(simulation_times.mean()),
            "mean_frame_seconds": float(frame_times.mean()),
            "p95_frame_seconds": float(np.quantile(frame_times, 0.95)),
            "realtime_at_50hz": bool(np.quantile(frame_times, 0.95) <= 0.02),
            "total_wall_seconds": float(time.perf_counter() - validation_started),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            output / "trajectory.npz",
            state_distance=state_array,
            action_distance=action_array,
            applied_delta=np.asarray(applied_delta_history, dtype=np.float32),
            applied_action=np.asarray(applied_action_history, dtype=np.float32),
            mpc_cost=np.asarray(mpc_cost_history, dtype=np.float32),
            qp_solve_seconds=qp_solve_times,
            qp_internal_seconds=qp_internal_times,
            qp_iterations=qp_iterations,
            simulation_step_seconds=simulation_times,
            frame_seconds=frame_times,
        )
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
