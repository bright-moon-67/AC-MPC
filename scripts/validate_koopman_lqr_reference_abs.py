#!/usr/bin/env python
"""Validate reference tracking with steady-state Koopman-LQR using the
ABSOLUTE-ACTION Koopman model ``z_{t+1}=A z_t + B u_t``.

The model state is the raw 45-D physical state ``s_t`` (no previous-action
block) and the controller outputs the absolute 18-D muscle activation ``u_t``
directly.  The environment is the plain ManiSoftTipTrackingEnv (45-D
observation, absolute actions).  The LQR is solved once in reference error
coordinates; at the exact reference the controller commands ``u = u_ref``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.control.steady_state_lqr import affine_lqr
from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256

TIP_INDICES = (30, 31, 32)
PHYSICAL_DIM = 45
ACTION_DIM = 18
DYNAMICS_STATE_DIM = PHYSICAL_DIM


def compress_physical_to_model(
    physical: np.ndarray,
    model_physical_dim: int,
) -> np.ndarray:
    """Validate and return the model's 45-D physical state block."""

    if model_physical_dim != PHYSICAL_DIM:
        raise ValueError(
            f"Unsupported model physical dimension {model_physical_dim}"
        )
    physical = np.asarray(physical, dtype=np.float32)
    if physical.ndim != 2 or physical.shape[1] != PHYSICAL_DIM:
        raise ValueError(
            f"physical state must have shape [T,{PHYSICAL_DIM}], "
            f"got {physical.shape}"
        )
    return physical


def build_model_state(observation: np.ndarray, model_state_dim: int) -> np.ndarray:
    """The abs-action model state is the raw 45-D env observation."""

    observation = np.asarray(observation, dtype=np.float32)
    if model_state_dim != DYNAMICS_STATE_DIM:
        raise ValueError(f"Unsupported model state dimension {model_state_dim}")
    if observation.shape != (DYNAMICS_STATE_DIM,):
        raise ValueError(
            f"Expected 45-D environment observation, got {observation.shape}"
        )
    return observation


class KoopmanLQRController:
    """Infinite-horizon LQR feedback around a fixed lifted reference
    for the absolute-action model ``z+ = A z + B u``.

    The controller outputs the absolute action
    ``u = u_ref - feedback_scale * K (z - z_ref)`` (feedforward d is zero for
    the pure quadratic stage cost). Q is assembled in normalized lifted
    coordinates over the 45-D physical state only.
    """

    def __init__(
        self,
        *,
        model,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        ref_state: np.ndarray,
        ref_action: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        state_weight: float,
        action_weight: float,
        control_weight: float,
        track_tip_only: bool,
        action_tracking: bool,
        tip_state_scale: float,
        max_delta: float,
        feedback_scale: float,
        lqr_regularization: float,
        dare_tolerance: float,
        dare_max_iterations: int,
        dare_jitter: float,
    ) -> None:
        setup_started = time.perf_counter()
        self.model = model
        self.device = state_mean.device
        self.model_dtype = state_mean.dtype
        self.state_mean = state_mean
        self.state_std = state_std
        self.state_std_np = state_std.detach().cpu().double().numpy()
        self.action_dim = int(model.action_dim)
        self.physical_dim = int(model.state_dim)
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        self.ref_state = np.asarray(ref_state, dtype=np.float64)
        self.ref_action = np.asarray(ref_action, dtype=np.float64)
        self.state_weight = float(state_weight)
        self.action_weight = float(action_weight)
        self.control_weight = float(control_weight)
        self.action_tracking = bool(action_tracking)
        self.max_delta = float(max_delta)
        self.feedback_scale = float(feedback_scale)

        physical_diagonal_np = np.full(
            self.physical_dim, self.state_weight, dtype=np.float64
        )
        if track_tip_only:
            physical_diagonal_np.fill(0.0)
        physical_diagonal_np[np.asarray(TIP_INDICES)] = float(tip_state_scale)

        self.reference_lift = self._lift(self.ref_state)

        physical_diagonal = torch.as_tensor(
            physical_diagonal_np, dtype=torch.float64, device=self.device
        )
        output_diagonal = physical_diagonal
        self.output_diagonal = output_diagonal.detach().cpu().numpy()

        A = model.A.detach().to(dtype=torch.float64)
        B = model.B.detach().to(dtype=torch.float64)
        C = model.C.detach().to(dtype=torch.float64)
        Q = C.mT @ (output_diagonal[:, None] * C)
        Q = 0.5 * (Q + Q.mT)
        Q = Q + float(lqr_regularization) * torch.eye(
            model.lifted_dim, dtype=torch.float64, device=self.device
        )
        # Both terms penalize the absolute-action deviation from the fixed
        # reference in error coordinates.  Previously action_weight and
        # action_tracking were accepted by the CLI but never affected LQR.
        action_deviation_weight = self.control_weight
        if self.action_tracking:
            action_deviation_weight += self.action_weight
        R = action_deviation_weight * torch.eye(
            self.action_dim, dtype=torch.float64, device=self.device
        )
        q = torch.zeros(model.lifted_dim, dtype=torch.float64, device=self.device)
        r = torch.zeros(self.action_dim, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            result = affine_lqr(
                A,
                B,
                Q,
                R,
                q,
                r,
                tolerance=float(dare_tolerance),
                max_iterations=int(dare_max_iterations),
                jitter=float(dare_jitter),
                check_stabilizable=True,
                check_detectable=True,
                fail_on_nonconvergence=True,
                compute_closed_loop_spectral_radius=True,
                implicit_backward=False,
            )
        self.gain = result.gain.detach()
        self.value_hessian = result.value_hessian.detach()
        self.feedforward = result.feedforward.detach()
        self.dare_converged = bool(result.dare.converged.item())
        self.dare_iterations = int(result.dare.iterations)
        self.dare_residual = float(result.dare.residual.item())
        self.dare_relative_residual = float(result.dare.relative_residual.item())
        self.dare_condition_number = float(result.dare.condition_number.item())
        self.dare_closed_loop_spectral_radius = float(
            result.dare.closed_loop_spectral_radius.item()
        )
        effective_closed_loop = A - B @ (self.feedback_scale * self.gain)
        self.closed_loop_spectral_radius = float(
            torch.linalg.eigvals(effective_closed_loop).abs().max().item()
        )
        self.setup_seconds = float(time.perf_counter() - setup_started)

    def _lift(self, state: np.ndarray) -> torch.Tensor:
        state_tensor = torch.as_tensor(
            state, dtype=self.model_dtype, device=self.device
        )
        normalized_state = (state_tensor - self.state_mean) / self.state_std
        with torch.no_grad():
            lifted = self.model.lift(normalized_state.unsqueeze(0))[0]
        return lifted.detach().to(dtype=torch.float64)

    def solve(
        self, state: np.ndarray, previous_action: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        started = time.perf_counter()
        state = np.asarray(state, dtype=np.float64)
        previous_action = np.asarray(previous_action, dtype=np.float64).reshape(-1)
        lifted_error = self._lift(state) - self.reference_lift

        with torch.no_grad():
            raw_deviation_t = -(self.gain @ lifted_error) - self.feedforward
            value_t = lifted_error @ self.value_hessian @ lifted_error
        raw_deviation = raw_deviation_t.detach().cpu().numpy()
        requested_action = self.ref_action + self.feedback_scale * raw_deviation
        # Smooth per-step change-rate bound: move at most max_delta from the
        # previously applied action, avoiding bang-bang switching.
        normalized_step = (requested_action - previous_action) / self.max_delta
        smoothed_action = previous_action + self.max_delta * np.tanh(
            normalized_step
        )
        applied_action = np.clip(
            smoothed_action,
            self.action_low,
            self.action_high,
        )

        normalized_error = (state - self.ref_state) / self.state_std_np
        state_cost = float(
            np.sum(self.output_diagonal * np.square(normalized_error))
        )
        applied_deviation = applied_action - self.ref_action
        control_cost = self.control_weight * np.square(applied_deviation).sum()
        action_tracking_cost = (
            self.action_weight * np.square(applied_deviation).sum()
            if self.action_tracking
            else 0.0
        )

        return {
            "raw_deviation": raw_deviation.astype(np.float32),
            "requested_action": requested_action.astype(np.float32),
            "applied_action": applied_action.astype(np.float32),
            "stage_cost": float(
                state_cost + control_cost + action_tracking_cost
            ),
            "value": float(value_t.item()),
            "feedback_seconds": float(time.perf_counter() - started),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Absolute-action steady-state Koopman-LQR tracking of a "
        "fixed reference"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--reference", required=True, help="reference.npz")
    parser.add_argument("--output", default="runs/koopman_lqr_reference_abs")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--state-weight", type=float, default=0.001)
    parser.add_argument(
        "--tip-weight",
        "--tip-state-scale",
        dest="tip_state_scale",
        type=float,
        default=20.0,
        help=(
            "Direct diagonal Q weight for the three normalized tip-position "
            "components."
        ),
    )
    parser.add_argument("--action-weight", type=float, default=0.3)
    parser.add_argument(
        "--no-action-tracking",
        action="store_true",
        help="Disable the explicit reference-action error term.",
    )
    parser.add_argument(
        "--track-tip-only",
        action="store_true",
        help="Track only the three tip-position components of the 45-D state.",
    )
    parser.add_argument("--control-weight", type=float, default=100000.0)
    parser.add_argument("--max-delta", type=float, default=0.002)
    parser.add_argument(
        "--feedback-scale",
        type=float,
        default=0.03,
        help=(
            "Scale applied to the Koopman-LQR feedback deviation. Zero is "
            "the stable fixed-reference feedforward baseline."
        ),
    )
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument(
        "--lqr-regularization",
        type=float,
        default=None,
        help=(
            "Small positive lifted-state cost used for DARE detectability; "
            "defaults to checkpoint control.stage_cost_epsilon."
        ),
    )
    parser.add_argument("--dare-tolerance", type=float, default=None)
    parser.add_argument("--dare-max-iterations", type=int, default=None)
    parser.add_argument("--dare-jitter", type=float, default=None)
    parser.add_argument("--success-threshold", type=float, default=0.002)
    parser.add_argument(
        "--required-success-streak",
        type=int,
        default=100,
        help="Required terminal consecutive steps below success-threshold.",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=100,
        help="Number of final steps used for endpoint stability statistics.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("steps must be positive")
    if min(args.required_success_streak, args.stability_window) < 1:
        raise ValueError(
            "required-success-streak and stability-window must be positive"
        )
    if args.required_success_streak > args.steps:
        raise ValueError("required-success-streak cannot exceed steps")
    if args.stability_window > args.steps:
        raise ValueError("stability-window cannot exceed steps")
    if args.action_weight < 0:
        raise ValueError("action-weight must be non-negative")
    if min(
        args.state_weight,
        args.control_weight,
        args.max_delta,
        args.absolute_action_limit,
        args.tip_state_scale,
        args.success_threshold,
    ) <= 0:
        raise ValueError("LQR weights, limits and thresholds must be positive")
    if args.lqr_regularization is not None and args.lqr_regularization < 0:
        raise ValueError("lqr-regularization must be non-negative")
    if args.feedback_scale < 0:
        raise ValueError("feedback-scale must be non-negative")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    if (
        not checkpoint.is_file()
        or not scenario.is_file()
        or not reference_path.is_file()
    ):
        raise FileNotFoundError("checkpoint / scenario / reference must exist")

    with np.load(reference_path, allow_pickle=False) as ref:
        ref_state = np.asarray(ref["reference_state"], dtype=np.float32)
        ref_action = np.asarray(ref["reference_action"], dtype=np.float32).reshape(-1)
    if ref_state.shape != (PHYSICAL_DIM,):
        raise ValueError(f"reference_state must be 45-D, got {ref_state.shape}")
    if ref_action.shape != (ACTION_DIM,):
        raise ValueError(f"reference_action must be 18-D, got {ref_action.shape}")
    if not (np.isfinite(ref_state).all() and np.isfinite(ref_action).all()):
        raise ValueError("reference state/action contains NaN or Inf")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model = model.to(device).freeze_dynamics()
    if (model.state_dim, model.action_dim) != (
        DYNAMICS_STATE_DIM,
        ACTION_DIM,
    ):
        raise ValueError(
            "Requires state/action dimensions (45,18), "
            f"got ({model.state_dim},{model.action_dim})"
        )

    stats = payload["normalizers"]["state"]
    state_mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
    state_std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
    if state_mean.shape != (DYNAMICS_STATE_DIM,) or state_std.shape != (
        DYNAMICS_STATE_DIM,
    ):
        raise ValueError("Checkpoint state normalizer must have 45 dimensions")
    if not torch.isfinite(state_mean).all() or not torch.isfinite(state_std).all():
        raise ValueError("Checkpoint state normalizer contains NaN or Inf")
    if not torch.all(state_std > 0):
        raise ValueError("Checkpoint state normalizer std must be positive")
    if payload["normalizers"].get("action") != "absolute_physical_units":
        raise ValueError(
            "Checkpoint action normalizer must be absolute_physical_units"
        )

    control_config = payload.get("config", {}).get("control", {})
    lqr_regularization = (
        float(args.lqr_regularization)
        if args.lqr_regularization is not None
        else float(control_config.get("stage_cost_epsilon", 1.0e-4))
    )
    dare_tolerance = (
        float(args.dare_tolerance)
        if args.dare_tolerance is not None
        else float(control_config.get("dare_tolerance", 1.0e-7))
    )
    dare_max_iterations = (
        int(args.dare_max_iterations)
        if args.dare_max_iterations is not None
        else int(control_config.get("dare_max_iterations", 500))
    )
    dare_jitter = (
        float(args.dare_jitter)
        if args.dare_jitter is not None
        else float(control_config.get("dare_jitter", 1.0e-6))
    )
    if min(dare_tolerance, dare_max_iterations, dare_jitter) <= 0:
        raise ValueError("DARE tolerance, iterations and jitter must be positive")

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
        if np.any(ref_action < action_low) or np.any(ref_action > action_high):
            raise ValueError(
                "reference_action is outside the environment action bounds"
            )
        controller = KoopmanLQRController(
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            ref_state=ref_model_state,
            ref_action=ref_action,
            action_low=action_low,
            action_high=action_high,
            state_weight=args.state_weight,
            action_weight=args.action_weight,
            control_weight=args.control_weight,
            track_tip_only=args.track_tip_only,
            action_tracking=not args.no_action_tracking,
            tip_state_scale=args.tip_state_scale,
            max_delta=args.max_delta,
            feedback_scale=args.feedback_scale,
            lqr_regularization=lqr_regularization,
            dare_tolerance=dare_tolerance,
            dare_max_iterations=dare_max_iterations,
            dare_jitter=dare_jitter,
        )
        print(
            "LQR ready: "
            f"setup={controller.setup_seconds:.3f}s "
            f"DARE={controller.dare_iterations}it "
            f"relative_residual={controller.dare_relative_residual:.3e} "
            f"effective_rho={controller.closed_loop_spectral_radius:.6f}",
            flush=True,
        )

        state_distances = [initial_state_dist]
        tip_history = [initial_tip.copy()]
        action_distances = [initial_action_dist]
        raw_deviation_history = []
        requested_action_history = []
        applied_action_history = []
        lqr_stage_cost_history = []
        lqr_value_history = []
        feedback_time_history = []
        simulation_time_history = []
        frame_time_history = []
        stop_reason = "max_steps"
        previous_action = np.zeros(ACTION_DIM, dtype=np.float64)

        for step in range(args.steps):
            frame_start = time.perf_counter()
            control = controller.solve(dynamics_state, previous_action)
            previous_action = control["applied_action"].astype(np.float64)

            simulation_start = time.perf_counter()
            observation, _, _, _, _ = env.step(control["applied_action"])
            simulation_seconds = time.perf_counter() - simulation_start
            dynamics_state = build_model_state(observation, model.state_dim)
            tip = dynamics_state[tip_idx]
            state_dist = float(np.linalg.norm(tip - ref_tip))
            action_dist = float(
                np.linalg.norm(control["applied_action"] - ref_action)
            )

            state_distances.append(state_dist)
            tip_history.append(tip.copy())
            action_distances.append(action_dist)
            raw_deviation_history.append(control["raw_deviation"])
            requested_action_history.append(control["requested_action"])
            applied_action_history.append(control["applied_action"])
            lqr_stage_cost_history.append(float(control["stage_cost"]))
            lqr_value_history.append(float(control["value"]))
            feedback_time_history.append(float(control["feedback_seconds"]))
            simulation_time_history.append(simulation_seconds)
            frame_time_history.append(time.perf_counter() - frame_start)

            if step % 25 == 0 or step == args.steps - 1:
                print(
                    f"step={step + 1:04d}/{args.steps} "
                    f"tip=({tip[0]:.4f},{tip[1]:.4f},{tip[2]:.4f}) "
                    f"ref=({ref_tip[0]:.4f},{ref_tip[1]:.4f},{ref_tip[2]:.4f}) "
                    f"state_dist={state_dist*1000:.1f}mm "
                    f"action_error_norm={action_dist:.4f} "
                    f"lqr_cost={float(control['stage_cost']):.2f} "
                    f"feedback={float(control['feedback_seconds'])*1000:.2f}ms",
                    flush=True,
                )

        state_array = np.asarray(state_distances, dtype=np.float32)
        tip_array = np.asarray(tip_history, dtype=np.float32)
        action_array = np.asarray(action_distances, dtype=np.float32)
        raw_deviation_array = np.asarray(raw_deviation_history, dtype=np.float32)
        requested_action_array = np.asarray(requested_action_history, dtype=np.float32)
        applied_action_array = np.asarray(applied_action_history, dtype=np.float32)
        feedback_times = np.asarray(feedback_time_history, dtype=np.float64)
        simulation_times = np.asarray(simulation_time_history, dtype=np.float64)
        frame_times = np.asarray(frame_time_history, dtype=np.float64)

        final_state_dist = float(state_array[-1])
        min_state_dist = float(state_array.min())
        final_action_dist = float(action_array[-1])
        below_threshold = state_array <= args.success_threshold
        longest_success_streak = 0
        current_success_streak = 0
        first_success_streak_start = None
        current_streak_start = 0
        for index, is_below in enumerate(below_threshold):
            if is_below:
                if current_success_streak == 0:
                    current_streak_start = index
                current_success_streak += 1
                if current_success_streak > longest_success_streak:
                    longest_success_streak = current_success_streak
                if (
                    first_success_streak_start is None
                    and current_success_streak >= args.required_success_streak
                ):
                    first_success_streak_start = current_streak_start
            else:
                current_success_streak = 0
        terminal_success_streak = current_success_streak
        tail_state = state_array[-args.stability_window :]
        tail_tip = tip_array[-args.stability_window :]
        success = bool(
            terminal_success_streak >= args.required_success_streak
            and float(tail_state.max()) <= args.success_threshold
        )

        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "scenario": str(scenario),
            "reference": str(reference_path),
            "device": str(device),
            "controller": "steady_state_koopman_lqr_abs",
            "action_form": "absolute",
            "lqr_coordinates": "lifted_reference_error",
            "q_coordinate_system": "normalized_model_state",
            "state_weight": args.state_weight,
            "tip_weight": args.tip_state_scale,
            "tip_state_scale": args.tip_state_scale,
            "action_weight": args.action_weight,
            "action_tracking": not args.no_action_tracking,
            "control_weight": args.control_weight,
            "max_delta": args.max_delta,
            "feedback_scale": args.feedback_scale,
            "lqr_regularization": lqr_regularization,
            "absolute_action_limit": args.absolute_action_limit,
            "lqr_setup_seconds": controller.setup_seconds,
            "dare_tolerance": dare_tolerance,
            "dare_max_iterations": dare_max_iterations,
            "dare_jitter": dare_jitter,
            "dare_converged": controller.dare_converged,
            "dare_iterations": controller.dare_iterations,
            "dare_residual": controller.dare_residual,
            "dare_relative_residual": controller.dare_relative_residual,
            "dare_condition_number": controller.dare_condition_number,
            "dare_closed_loop_spectral_radius": (
                controller.dare_closed_loop_spectral_radius
            ),
            "closed_loop_spectral_radius": controller.closed_loop_spectral_radius,
            "reference_tip": ref_tip.tolist(),
            "reference_action": ref_action.tolist(),
            "initial_tip": initial_tip.tolist(),
            "initial_state_distance": initial_state_dist,
            "final_state_distance": final_state_dist,
            "minimum_state_distance": min_state_dist,
            "initial_action_distance": initial_action_dist,
            "final_action_distance": final_action_dist,
            "success_threshold": args.success_threshold,
            "required_success_streak": args.required_success_streak,
            "first_success_streak_start": first_success_streak_start,
            "longest_success_streak": longest_success_streak,
            "terminal_success_streak": terminal_success_streak,
            "stability_window": args.stability_window,
            "tail_state_distance_mean": float(tail_state.mean()),
            "tail_state_distance_std": float(tail_state.std()),
            "tail_state_distance_max": float(tail_state.max()),
            "tail_tip_axis_std": tail_tip.std(axis=0).tolist(),
            "success": success,
            "stop_reason": stop_reason,
            "steps_executed": int(len(state_array) - 1),
            "mean_feedback_seconds": float(feedback_times.mean()),
            "p95_feedback_seconds": float(np.quantile(feedback_times, 0.95)),
            "max_feedback_seconds": float(feedback_times.max()),
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
            tip_position=tip_array,
            action_distance=action_array,
            raw_deviation=raw_deviation_array,
            requested_action=requested_action_array,
            applied_action=applied_action_array,
            lqr_stage_cost=np.asarray(lqr_stage_cost_history, dtype=np.float32),
            lqr_value=np.asarray(lqr_value_history, dtype=np.float32),
            feedback_seconds=feedback_times,
            simulation_step_seconds=simulation_times,
            frame_seconds=frame_times,
        )
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
