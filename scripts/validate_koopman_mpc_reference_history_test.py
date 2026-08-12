#!/usr/bin/env python
"""Validate fixed-reference tracking with condensed-QP Koopman-MPC
using the HISTORY-CONTEXT Koopman model (HistoryDeepKoopman).

Model: ``z_{t+1} = A z_t + B u_t`` (absolute action), but the nonlinear
lift uses a finite history context ``[s[t-H+1:t+1], u[t-H:t]]`` (H=10).
The physical state is 45-D, the action is the absolute 18-D muscle
activation.  The environment is the plain ManiSoftTipTrackingEnv (45-D
observation, absolute actions).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sparse
import torch

from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256

TIP_INDICES = (30, 31, 32)
PHYSICAL_DIM = 45
ACTION_DIM = 18
DYNAMICS_STATE_DIM = PHYSICAL_DIM


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
    """The history model state is the raw 45-D env observation."""

    observation = np.asarray(observation, dtype=np.float32)
    if model_state_dim != DYNAMICS_STATE_DIM:
        raise ValueError(f"Unsupported model state dimension {model_state_dim}")
    if observation.shape != (DYNAMICS_STATE_DIM,):
        raise ValueError(
            f"Expected 45-D environment observation, got {observation.shape}"
        )
    return observation


class HistoryBuffer:
    """Maintain the aligned history context ``[s, u]`` for the history model.

    Context at time ``t`` is ``[s[t-H+1:t+1], u[t-H:t]]``:
      - H normalized physical states (oldest .. current),
      - H raw absolute actions (u[t-H] .. u[t-1]; the current action is
        excluded because it is the MPC decision variable).
    """

    def __init__(
        self,
        history_steps: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
    ) -> None:
        self.history_steps = int(history_steps)
        self.state_mean = np.asarray(state_mean, dtype=np.float64)
        self.state_std = np.asarray(state_std, dtype=np.float64)
        self.state_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.action_history: deque[np.ndarray] = deque(maxlen=self.history_steps)

    def reset(self, initial_state: np.ndarray) -> None:
        """Warm up with the initial state repeated (matches training padding)."""

        self.state_history.clear()
        self.action_history.clear()
        initial = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        for _ in range(self.history_steps):
            self.state_history.append(initial.copy())
        for _ in range(self.history_steps):
            self.action_history.append(np.zeros(ACTION_DIM, dtype=np.float64))

    def append(self, state: np.ndarray, action: np.ndarray) -> None:
        self.state_history.append(np.asarray(state, dtype=np.float64).reshape(-1))
        self.action_history.append(np.asarray(action, dtype=np.float64).reshape(-1))

    def context(self) -> np.ndarray:
        """Return the 630-D context ``[normalized s (H*45), raw u (H*18)]``."""

        states = np.asarray(list(self.state_history), dtype=np.float64)
        normalized_states = (states - self.state_mean) / self.state_std
        actions = np.asarray(list(self.action_history), dtype=np.float64)
        return np.concatenate(
            (normalized_states.reshape(-1), actions.reshape(-1))
        ).astype(np.float32)


class KoopmanMPCQP:
    """Condensed OSQP controller for the history Koopman model.

    The linear dynamics are ``z+ = A z + B u`` (absolute action) in the
    lifted space; the history context only enters the initial lift.
    """

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
        self.physical_dim = int(model.state_dim)
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
        action_prediction = np.zeros(
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
                action_prediction[rows, columns] = (
                    C_physical @ powers[step - control_step] @ B
                )

        physical_std = np.tile(self.state_std[: self.physical_dim], self.horizon)
        self.free_prediction = physical_std[:, None] * free_prediction
        self.action_prediction = physical_std[:, None] * action_prediction
        self.physical_mean = np.tile(
            self.state_mean[: self.physical_dim], self.horizon
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
        hessian = self.action_prediction.T @ (
            state_diagonal[:, None] * self.action_prediction
        )
        # Action tracking ||u - u_ref||^2 -> identity in absolute coordinates.
        hessian += self.action_weight * np.eye(self.variable_dim)
        # Action-magnitude regularizer ||u||^2.
        hessian += self.control_weight * np.eye(self.variable_dim)
        # Smoothness ||u_k - u_{k-1}||^2.
        hessian += self.smoothness_weight * (difference.T @ difference)
        hessian = 2.0 * ((hessian + hessian.T) * 0.5)

        # First-step selector: u_0 - u_prev must stay within +/- max_delta.
        first_step_selector = np.zeros(
            (self.action_dim, self.variable_dim), dtype=np.float64
        )
        first_step_selector[:, : self.action_dim] = np.eye(
            self.action_dim, dtype=np.float64
        )
        self.first_step_selector = first_step_selector

        constraint_matrix = sparse.vstack(
            (
                sparse.eye(self.variable_dim, format="csc"),
                sparse.csc_matrix(difference),
                sparse.csc_matrix(first_step_selector),
            ),
            format="csc",
        )
        initial_lower = np.concatenate(
            (
                np.tile(self.action_low, self.horizon),
                np.full(
                    (self.horizon - 1) * self.action_dim, -self.max_delta
                ),
                np.full(self.action_dim, -self.max_delta),
            )
        )
        initial_upper = np.concatenate(
            (
                np.tile(self.action_high, self.horizon),
                np.full(
                    (self.horizon - 1) * self.action_dim, self.max_delta
                ),
                np.full(self.action_dim, self.max_delta),
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

    def _lift(self, state: np.ndarray, context: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        context_tensor = torch.as_tensor(context, dtype=self.dtype, device=self.device)
        normalized_state = (state_tensor - self.state_mean_t) / self.state_std_t
        with torch.no_grad():
            lifted = self.model.lift(
                normalized_state.unsqueeze(0), context_tensor.unsqueeze(0)
            )[0]
        return lifted.detach().cpu().double().numpy()

    def solve(
        self,
        *,
        state: np.ndarray,
        context: np.ndarray,
        ref_state: np.ndarray,
        ref_action: np.ndarray,
        previous_action: np.ndarray,
        initial_actions: np.ndarray | None,
    ) -> dict[str, np.ndarray | float | int | str]:
        started = time.perf_counter()
        state = np.asarray(state, dtype=np.float64)
        ref_state = np.asarray(ref_state, dtype=np.float64)
        ref_action = np.asarray(ref_action, dtype=np.float64)
        previous_action = np.asarray(previous_action, dtype=np.float64).reshape(-1)

        free_physical = self.free_prediction @ self._lift(state, context) + self.physical_mean
        state_error = free_physical - np.tile(
            ref_state[: self.physical_dim], self.horizon
        )
        state_diagonal = np.tile(
            self.state_weight * np.square(self.state_scales), self.horizon
        )
        linear = 2.0 * self.action_prediction.T @ (state_diagonal * state_error)
        action_target = (
            ref_action if self.action_tracking else np.zeros_like(ref_action)
        )
        # From 0.5*||u - u_target||^2 expanded: -u_target' u.
        linear -= 2.0 * self.action_weight * np.tile(action_target, self.horizon)

        first_lower = previous_action - self.max_delta
        first_upper = previous_action + self.max_delta
        lower = np.concatenate(
            (
                np.tile(self.action_low, self.horizon),
                np.full(
                    (self.horizon - 1) * self.action_dim, -self.max_delta
                ),
                first_lower,
            )
        )
        upper = np.concatenate(
            (
                np.tile(self.action_high, self.horizon),
                np.full(
                    (self.horizon - 1) * self.action_dim, self.max_delta
                ),
                first_upper,
            )
        )
        self.solver.update(q=linear, l=lower, u=upper)
        if initial_actions is not None:
            initial_actions = np.asarray(initial_actions, dtype=np.float64)
            if initial_actions.shape != (self.horizon, self.action_dim):
                raise ValueError("QP warm start has the wrong shape")
            self.solver.warm_start(x=initial_actions.reshape(-1))
        result = self.solver.solve(raise_error=False)
        if result.info.status_val not in {1, 2} or result.x is None:
            raise RuntimeError(
                "Koopman MPC QP failed: "
                f"status={result.info.status!r}, iterations={result.info.iter}"
            )

        requested_actions = np.asarray(result.x, dtype=np.float64).reshape(
            self.horizon, self.action_dim
        )
        applied_actions = np.clip(
            requested_actions, self.action_low, self.action_high
        )
        applied_deltas = np.diff(
            np.vstack((applied_actions[:1], applied_actions)), axis=0
        )

        flat_actions = applied_actions.reshape(-1)
        predicted_physical = (
            free_physical + self.action_prediction @ flat_actions
        ).reshape(self.horizon, self.physical_dim)
        weighted_state_error = (
            predicted_physical - ref_state[: self.physical_dim]
        ) * self.state_scales
        state_cost = self.state_weight * np.square(weighted_state_error).sum()
        action_cost = self.action_weight * np.square(
            applied_actions - action_target
        ).sum()
        control_cost = self.control_weight * np.square(applied_actions).sum()
        smoothness_cost = self.smoothness_weight * np.square(
            np.diff(applied_actions, axis=0)
        ).sum()

        return {
            "requested_actions": requested_actions.astype(np.float32),
            "applied_actions": applied_actions.astype(np.float32),
            "applied_deltas": applied_deltas.astype(np.float32),
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
        description="History-context Koopman-MPC tracking of a fixed-action "
        "reference equilibrium"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--reference", required=True, help="reference.npz")
    parser.add_argument("--output", default="runs/koopman_mpc_reference_history")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--state-weight", type=float, default=100.0)
    parser.add_argument("--tip-state-scale", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=30.0)
    parser.add_argument("--no-action-tracking", action="store_true")
    parser.add_argument("--track-tip-only", action="store_true")
    parser.add_argument("--control-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=10.0)
    parser.add_argument("--max-delta", type=float, default=0.002)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--qp-max-iterations", type=int, default=4000)
    parser.add_argument("--qp-absolute-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--qp-relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--success-threshold", type=float, default=0.02)
    parser.add_argument(
        "--ref-action-scale",
        type=float,
        default=1.0,
        help="[TEST] scale applied to the reference action (uref) to "
        "simulate an inaccurate feedforward action; 1.0 = exact uref",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if min(args.steps, args.horizon, args.qp_max_iterations) < 1:
        raise ValueError("steps, horizon and qp-max-iterations must be positive")
    if args.action_weight < 0:
        raise ValueError("action-weight must be non-negative")
    if args.ref_action_scale <= 0:
        raise ValueError("ref-action-scale must be positive")
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

    # [TEST ONLY] deliberately make uref inaccurate by scaling it, to probe
    # closed-loop robustness of the MPC.  The exact uref is kept in
    # ref_action_original for reference; the MPC sees the scaled one.
    ref_action_original = ref_action.copy()
    if args.ref_action_scale != 1.0:
        ref_action = ref_action * args.ref_action_scale
        print(
            f"[TEST] uref scaled by {args.ref_action_scale} "
            f"(||u_ref|| {np.linalg.norm(ref_action_original):.4f} -> "
            f"{np.linalg.norm(ref_action):.4f})",
            flush=True,
        )

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
            "Requires state/action dimensions (45,18), "
            f"got ({model.state_dim},{model.action_dim})"
        )
    if not hasattr(model, "history_steps"):
        raise ValueError("Checkpoint is not a HistoryDeepKoopman model")
    history_steps = int(model.history_steps)

    stats = payload["normalizers"]["state"]
    state_mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
    state_std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
    if state_mean.shape != (DYNAMICS_STATE_DIM,) or state_std.shape != (
        DYNAMICS_STATE_DIM,
    ):
        raise ValueError("Checkpoint state normalizer must have 45 dimensions")

    ref_model_state = compress_physical_to_model(
        ref_state[None, :], model.state_dim
    )[0]
    tip_idx = np.asarray(TIP_INDICES)
    ref_tip = ref_model_state[tip_idx].astype(np.float32, copy=False)

    from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv

    env = ManiSoftTipTrackingEnv(
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
        initial_action_dist = float(np.linalg.norm(ref_action))
        action_low = np.asarray(env.action_space.low, dtype=np.float64)
        action_high = np.asarray(env.action_space.high, dtype=np.float64)
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
        history_buffer = HistoryBuffer(
            history_steps=history_steps,
            state_mean=stats["mean"],
            state_std=stats["std"],
        )
        history_buffer.reset(dynamics_state)

        state_distances = [initial_state_dist]
        action_distances = [initial_action_dist]
        applied_action_history = []
        applied_delta_history = []
        mpc_cost_history = []
        qp_solve_time_history = []
        qp_internal_time_history = []
        qp_iteration_history = []
        simulation_time_history = []
        frame_time_history = []
        warm_start = None
        previous_action = np.zeros(ACTION_DIM, dtype=np.float64)

        for step in range(args.steps):
            frame_start = time.perf_counter()
            context = history_buffer.context()
            plan = planner.solve(
                state=dynamics_state,
                context=context,
                ref_state=ref_model_state,
                ref_action=ref_action,
                previous_action=previous_action,
                initial_actions=warm_start,
            )
            applied_action = plan["applied_actions"][0]
            applied_delta = plan["applied_deltas"][0]

            simulation_start = time.perf_counter()
            observation, _, _, _, _ = env.step(applied_action)
            simulation_seconds = time.perf_counter() - simulation_start
            previous_action = applied_action.copy()
            dynamics_state = build_model_state(observation, model.state_dim)
            history_buffer.append(dynamics_state, applied_action)
            tip = dynamics_state[tip_idx]
            state_dist = float(np.linalg.norm(tip - ref_tip))
            action_dist = float(np.linalg.norm(applied_action - ref_action))
            state_distances.append(state_dist)
            action_distances.append(action_dist)
            applied_action_history.append(applied_action)
            applied_delta_history.append(applied_delta)
            mpc_cost_history.append(float(plan["cost"]))
            qp_solve_time_history.append(float(plan["qp_solve_seconds"]))
            qp_internal_time_history.append(float(plan["qp_internal_run_seconds"]))
            qp_iteration_history.append(int(plan["qp_iterations"]))
            simulation_time_history.append(simulation_seconds)
            frame_time_history.append(time.perf_counter() - frame_start)
            warm_start = np.concatenate(
                (
                    plan["applied_actions"][1:],
                    plan["applied_actions"][-1:],
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
            "action_form": "absolute_history",
            "history_steps": history_steps,
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
            "ref_action_scale": args.ref_action_scale,
            "reference_action_original": ref_action_original.tolist(),
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
            applied_action=np.asarray(applied_action_history, dtype=np.float32),
            applied_delta=np.asarray(applied_delta_history, dtype=np.float32),
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
