"""Find and save a stable point-reference state/action pair.

The script samples a smooth random terminal muscle action, ramps from zero to
that action, and then holds it constant.  A result is saved only when the tip
has moved the requested distance and the final hold window satisfies the
position and speed stability thresholds.  Tip orientation is recorded as part
of the 45-D state but is never constrained.

This produces an open-loop equilibrium/feed-forward reference.  It is not a
disturbance-rejecting point-tracking controller by itself.
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from manisoft.backend.base import SoftRobotState
from manisoft.muscle import SplineMuscle

try:
    from collect_koopman_data import (
        compact_state,
        create_environment,
        has_ground_clearance,
        minimum_jerk,
        spatial_basis,
        state_layout,
        update_live_viewer,
    )
except ModuleNotFoundError:
    from scripts.collect_koopman_data import (
        compact_state,
        create_environment,
        has_ground_clearance,
        minimum_jerk,
        spatial_basis,
        state_layout,
        update_live_viewer,
    )


ACTION_SHAPE = (6, 3)
MODE_WEIGHTS = np.array([1.0, 0.8, 0.6, 0.35, 0.2, 0.12])
# Bending axes dominate so random trials normally create a tip displacement.
AXIS_WEIGHTS = np.array([1.0, 1.0, 0.25])


@dataclass
class AttemptResult:
    """All data needed to score and save one random hold attempt."""

    attempt_index: int
    initial_state: npt.NDArray[np.float32]
    initial_tip_position: npt.NDArray[np.float64]
    terminal_action: npt.NDArray[np.float32]
    trajectory_states: npt.NDArray[np.float32]
    trajectory_actions: npt.NDArray[np.float32]
    trajectory_phases: npt.NDArray[np.int8]
    tip_positions: npt.NDArray[np.float64]
    tip_speeds: npt.NDArray[np.float64]
    stable_window_start: int
    target_position: npt.NDArray[np.float64]
    achieved_offset: float
    position_rms: float
    position_max: float
    speed_mean: float
    speed_max: float
    reference_index: int
    reference_full_state: SoftRobotState
    reference_torque: npt.NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo_elastica_fast.yaml"),
        help="ManiSoft environment configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work_dirs/point_reference"),
    )
    parser.add_argument(
        "--offset-distance",
        type=float,
        default=0.20,
        help="Requested distance in metres from the settled initial tip.",
    )
    parser.add_argument(
        "--offset-tolerance",
        type=float,
        default=0.05,
        help="Allowed absolute error in the requested offset distance.",
    )
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--ramp-seconds",
        type=float,
        default=2.0,
        help="Minimum-jerk transition time from zero to the sampled action.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=8.0,
        help="Time for which the sampled terminal action remains unchanged.",
    )
    parser.add_argument(
        "--stability-window-seconds",
        type=float,
        default=2.0,
        help="Final part of the hold phase used to verify equilibrium.",
    )
    parser.add_argument(
        "--position-stability-tolerance",
        type=float,
        default=0.005,
        help="Maximum tip deviation from its window mean, in metres.",
    )
    parser.add_argument(
        "--speed-stability-tolerance",
        type=float,
        default=0.01,
        help="Maximum tip speed in the stable window, in metres/second.",
    )
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--muscle-torque-scale", type=float, default=30.0)
    parser.add_argument("--min-action-peak", type=float, default=0.15)
    parser.add_argument("--max-action-peak", type=float, default=0.70)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--min-tip-height", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Display each attempt in the interactive MuJoCo viewer.",
    )
    parser.add_argument("--viewer-fps", type=float, default=50.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace reference.npz and summary.json if they already exist.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "offset-distance": args.offset_distance,
        "offset-tolerance": args.offset_tolerance,
        "ramp-seconds": args.ramp_seconds,
        "hold-seconds": args.hold_seconds,
        "stability-window-seconds": args.stability_window_seconds,
        "position-stability-tolerance": (
            args.position_stability_tolerance
        ),
        "speed-stability-tolerance": args.speed_stability_tolerance,
        "control-hz": args.control_hz,
        "muscle-torque-scale": args.muscle_torque_scale,
        "min-action-peak": args.min_action_peak,
        "max-action-peak": args.max_action_peak,
        "viewer-fps": args.viewer_fps,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {invalid}")
    if args.settle_seconds < 0:
        raise ValueError("settle-seconds must be non-negative")
    if args.max_attempts <= 0:
        raise ValueError("max-attempts must be positive")
    if args.max_action_peak < args.min_action_peak:
        raise ValueError("max-action-peak must be >= min-action-peak")
    if args.stability_window_seconds > args.hold_seconds:
        raise ValueError(
            "stability-window-seconds cannot exceed hold-seconds"
        )


def sample_terminal_action(
    rng: np.random.Generator,
    min_peak: float,
    max_peak: float,
) -> npt.NDArray[np.float32]:
    """Sample a spatially smooth 6x3 activation with a known peak."""
    modal_coefficients = rng.normal(size=ACTION_SHAPE)
    modal_coefficients *= MODE_WEIGHTS[:, None]
    modal_coefficients *= AXIS_WEIGHTS[None, :]
    action = spatial_basis() @ modal_coefficients
    raw_peak = float(np.max(np.abs(action)))
    if raw_peak <= 0:
        raise RuntimeError("sampled a zero terminal action")
    requested_peak = rng.uniform(min_peak, max_peak)
    action *= requested_peak / raw_peak
    return action.astype(np.float32)


def stability_metrics(
    tip_positions: npt.NDArray[np.float64],
    tip_speeds: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float, float, float, float]:
    """Measure tip stationarity relative to the stable-window mean."""
    if tip_positions.ndim != 2 or tip_positions.shape[1] != 3:
        raise ValueError("tip_positions must have shape [T, 3]")
    if tip_speeds.shape != (tip_positions.shape[0],):
        raise ValueError("tip_speeds must have shape [T]")
    if tip_positions.shape[0] == 0:
        raise ValueError("the stability window cannot be empty")

    target_position = np.mean(tip_positions, axis=0)
    position_errors = np.linalg.norm(
        tip_positions - target_position[None, :], axis=1
    )
    position_rms = float(np.sqrt(np.mean(position_errors**2)))
    position_max = float(np.max(position_errors))
    speed_mean = float(np.mean(tip_speeds))
    speed_max = float(np.max(tip_speeds))
    return (
        target_position,
        position_rms,
        position_max,
        speed_mean,
        speed_max,
    )


def select_reference_index(
    tip_positions: npt.NDArray[np.float64],
    tip_speeds: npt.NDArray[np.float64],
    target_position: npt.NDArray[np.float64],
    position_scale: float,
    speed_scale: float,
) -> int:
    """Select one real sample representative of the stable equilibrium."""
    position_error = np.linalg.norm(
        tip_positions - target_position[None, :], axis=1
    )
    score = position_error / position_scale + tip_speeds / speed_scale
    return int(np.argmin(score))


def get_tip(state: SoftRobotState) -> tuple[np.ndarray, float]:
    position = np.asarray(state.element_positions[-1], dtype=np.float64)
    velocity = np.asarray(state.element_velocities[-1], dtype=np.float64)
    return position, float(np.linalg.norm(velocity))


def create_viewer(env, enabled: bool):
    if not enabled:
        return None
    from manisoft.backend import ManiSoftSimBackend
    from manisoft.visualize.mujoco_viewer import SoftRobotMujocoViewer

    if not isinstance(env.backend, ManiSoftSimBackend):
        raise TypeError("--live requires backend.type=ManiSoftSimBackend")
    return SoftRobotMujocoViewer(
        env.backend.mujoco_model,
        env.backend.mujoco_data,
        env.backend.softrobot_radius,
    ).launch()


def run_attempt(
    args: argparse.Namespace,
    rng: np.random.Generator,
    attempt_index: int,
) -> AttemptResult | None:
    """Run one independent random ramp-and-hold simulation."""
    env, configs = create_environment(args.config)
    viewer = None
    try:
        physics_dt = float(configs["backend"]["dt"])
        control_dt = 1.0 / args.control_hz
        actual_control_dt = physics_dt * env.update_interval
        if not np.isclose(actual_control_dt, control_dt, atol=1e-12, rtol=0):
            expected_interval = round(control_dt / physics_dt)
            raise ValueError(
                f"config gives control period {actual_control_dt:.8f} s, "
                f"but {args.control_hz:g} Hz requires {control_dt:.8f} s; "
                "set environment.update_interval to "
                f"{expected_interval}"
            )

        env.place()
        muscle = SplineMuscle(
            robot_length=float(configs["softrobot"]["length"]),
            robot_num_elements=int(configs["softrobot"]["num_elements"]),
            number_of_control_points=ACTION_SHAPE[0],
            muscle_torque_scale=args.muscle_torque_scale,
        )
        muscle.set_activation(np.zeros(ACTION_SHAPE, dtype=np.float64))

        def current_torque(element_lengths) -> np.ndarray:
            return muscle.evaluate(element_lengths)

        viewer = create_viewer(env, args.live)
        viewer_interval = max(1, round(args.control_hz / args.viewer_fps))
        _, soft_state = env.get_state(has_image=False)
        if viewer is not None:
            viewer.sync(soft_state.element_positions)

        settle_steps = round(args.settle_seconds * args.control_hz)
        for step_index in range(settle_steps):
            frame_start = time.perf_counter()
            env.step_with_torque_callback(current_torque)
            _, soft_state = env.get_state(has_image=False)
            if not has_ground_clearance(soft_state, args.min_tip_height):
                return None
            if viewer is not None:
                update_live_viewer(
                    viewer,
                    soft_state,
                    frame_start,
                    control_dt,
                    step_index % viewer_interval == 0,
                )

        initial_state = compact_state(soft_state)
        initial_tip_position, _ = get_tip(soft_state)
        terminal_action = sample_terminal_action(
            rng, args.min_action_peak, args.max_action_peak
        )
        ramp_steps = max(2, round(args.ramp_seconds * args.control_hz))
        hold_steps = max(1, round(args.hold_seconds * args.control_hz))
        total_steps = ramp_steps + hold_steps
        trajectory_states = np.empty(
            (total_steps, initial_state.size), dtype=np.float32
        )
        trajectory_actions = np.empty(
            (total_steps, int(np.prod(ACTION_SHAPE))), dtype=np.float32
        )
        trajectory_phases = np.empty(total_steps, dtype=np.int8)
        tip_positions = np.empty((total_steps, 3), dtype=np.float64)
        tip_speeds = np.empty(total_steps, dtype=np.float64)
        full_states: list[SoftRobotState] = []
        ramp_alpha = np.linspace(0.0, 1.0, ramp_steps)
        ramp_blend = minimum_jerk(ramp_alpha)
        for step_index in range(total_steps):
            frame_start = time.perf_counter()
            if step_index < ramp_steps:
                action = terminal_action * ramp_blend[step_index]
                phase = 0
            else:
                action = terminal_action
                phase = 1
            muscle.set_activation(action)
            env.step_with_torque_callback(current_torque)
            _, soft_state = env.get_state(has_image=False)
            if not has_ground_clearance(soft_state, args.min_tip_height):
                return None

            current_state = compact_state(soft_state)
            tip_position, tip_speed = get_tip(soft_state)
            trajectory_states[step_index] = current_state
            trajectory_actions[step_index] = action.reshape(-1)
            trajectory_phases[step_index] = phase
            tip_positions[step_index] = tip_position
            tip_speeds[step_index] = tip_speed
            full_states.append(soft_state)

            if viewer is not None:
                update_live_viewer(
                    viewer,
                    soft_state,
                    frame_start,
                    control_dt,
                    step_index % viewer_interval == 0,
                )

        window_steps = max(
            1, round(args.stability_window_seconds * args.control_hz)
        )
        stable_window_start = total_steps - window_steps
        stable_positions = tip_positions[stable_window_start:]
        stable_speeds = tip_speeds[stable_window_start:]
        (
            target_position,
            position_rms,
            position_max,
            speed_mean,
            speed_max,
        ) = stability_metrics(stable_positions, stable_speeds)
        achieved_offset = float(
            np.linalg.norm(target_position - initial_tip_position)
        )
        local_reference_index = select_reference_index(
            stable_positions,
            stable_speeds,
            target_position,
            args.position_stability_tolerance,
            args.speed_stability_tolerance,
        )
        reference_index = stable_window_start + local_reference_index
        reference_full_state = full_states[reference_index]
        reference_torque = muscle.evaluate(
            reference_full_state.element_lengths
        )

        return AttemptResult(
            attempt_index=attempt_index,
            initial_state=initial_state,
            initial_tip_position=initial_tip_position,
            terminal_action=terminal_action,
            trajectory_states=trajectory_states,
            trajectory_actions=trajectory_actions,
            trajectory_phases=trajectory_phases,
            tip_positions=tip_positions,
            tip_speeds=tip_speeds,
            stable_window_start=stable_window_start,
            target_position=target_position,
            achieved_offset=achieved_offset,
            position_rms=position_rms,
            position_max=position_max,
            speed_mean=speed_mean,
            speed_max=speed_max,
            reference_index=reference_index,
            reference_full_state=reference_full_state,
            reference_torque=reference_torque,
        )
    finally:
        if viewer is not None:
            viewer.close()
        del env
        gc.collect()


def result_is_valid(
    result: AttemptResult,
    args: argparse.Namespace,
) -> tuple[bool, bool]:
    offset_ok = (
        abs(result.achieved_offset - args.offset_distance)
        <= args.offset_tolerance
    )
    stable = (
        result.position_max <= args.position_stability_tolerance
        and result.speed_max <= args.speed_stability_tolerance
    )
    return offset_ok, stable


def save_result(
    result: AttemptResult,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "reference.npz"
    summary_path = args.output_dir / "summary.json"
    if not args.overwrite and (npz_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"{args.output_dir} already contains reference output; use a "
            "new --output-dir or pass --overwrite"
        )

    state = result.reference_full_state
    stable_slice = slice(result.stable_window_start, None)
    np.savez_compressed(
        npz_path,
        schema_version=np.array(1, dtype=np.int32),
        initial_state=result.initial_state,
        initial_tip_position=result.initial_tip_position,
        target_position=result.target_position,
        reference_state=result.trajectory_states[result.reference_index],
        reference_action=result.terminal_action.reshape(-1),
        reference_action_matrix=result.terminal_action,
        reference_torque=result.reference_torque,
        reference_tip_position=result.tip_positions[result.reference_index],
        reference_full_positions=np.asarray(state.element_positions),
        reference_full_directors=np.asarray(state.element_directors),
        reference_full_velocities=np.asarray(state.element_velocities),
        reference_full_angular_velocities=np.asarray(
            state.element_angular_velocities
        ),
        reference_element_lengths=np.asarray(state.element_lengths),
        stable_window_states=result.trajectory_states[stable_slice],
        stable_window_actions=result.trajectory_actions[stable_slice],
        stable_window_tip_positions=result.tip_positions[stable_slice],
        stable_window_tip_speeds=result.tip_speeds[stable_slice],
        trajectory_states=result.trajectory_states,
        trajectory_actions=result.trajectory_actions,
        trajectory_phases=result.trajectory_phases,
        trajectory_tip_positions=result.tip_positions,
        trajectory_tip_speeds=result.tip_speeds,
    )

    summary = {
        "schema_version": 1,
        "valid_reference": True,
        "reference_kind": "constant-action open-loop equilibrium",
        "orientation_constrained": False,
        "state_dim": 45,
        "state_layout": state_layout(),
        "action_dim": 18,
        "action_shape": list(ACTION_SHAPE),
        "action_semantics": (
            "6x3 SplineMuscle activation; replay this action through "
            "SplineMuscle, not reference_torque as a fixed torque array"
        ),
        "attempt_index": result.attempt_index,
        "seed": args.seed,
        "control_hz": args.control_hz,
        "muscle_torque_scale": args.muscle_torque_scale,
        "requested_offset_m": args.offset_distance,
        "offset_tolerance_m": args.offset_tolerance,
        "achieved_offset_m": result.achieved_offset,
        "initial_tip_position_m": result.initial_tip_position.tolist(),
        "target_position_m": result.target_position.tolist(),
        "reference_tip_position_m": (
            result.tip_positions[result.reference_index].tolist()
        ),
        "hold_seconds": args.hold_seconds,
        "stability_window_seconds": args.stability_window_seconds,
        "stable_window_position_rms_m": result.position_rms,
        "stable_window_position_max_m": result.position_max,
        "position_stability_tolerance_m": (
            args.position_stability_tolerance
        ),
        "stable_window_speed_mean_m_per_s": result.speed_mean,
        "stable_window_speed_max_m_per_s": result.speed_max,
        "speed_stability_tolerance_m_per_s": (
            args.speed_stability_tolerance
        ),
        "npz_fields": {
            "reference_state": "one real 45-D sample in the stable window",
            "reference_action": "18-D action that is held constant",
            "target_position": "mean tip position in the stable window",
            "reference_torque": (
                "20x3 distributed torque evaluated at the reference sample"
            ),
            "trajectory_phases": "0=ramp, 1=constant-action hold",
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return npz_path, summary_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    existing_outputs = [
        path
        for path in (
            args.output_dir / "reference.npz",
            args.output_dir / "summary.json",
        )
        if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"reference output already exists: {existing_outputs}; use a "
            "new --output-dir or pass --overwrite"
        )
    rng = np.random.default_rng(args.seed)
    best_result = None
    best_score = np.inf

    for attempt_index in range(1, args.max_attempts + 1):
        result = run_attempt(args, rng, attempt_index)
        if result is None:
            print(
                f"[{attempt_index}/{args.max_attempts}] rejected: "
                "ground-clearance constraint failed"
            )
            continue

        offset_ok, stable = result_is_valid(result, args)
        offset_error = abs(result.achieved_offset - args.offset_distance)
        score = (
            offset_error / args.offset_tolerance
            + result.position_max / args.position_stability_tolerance
            + result.speed_max / args.speed_stability_tolerance
        )
        if score < best_score:
            best_result = result
            best_score = score
        print(
            f"[{attempt_index}/{args.max_attempts}] "
            f"offset={result.achieved_offset:.4f} m "
            f"(ok={offset_ok}) | position_max={result.position_max:.6f} m "
            f"| speed_max={result.speed_max:.6f} m/s "
            f"(stable={stable})"
        )
        if offset_ok and stable:
            npz_path, summary_path = save_result(result, args)
            print(f"Valid point reference saved to {npz_path}")
            print(f"Verification summary saved to {summary_path}")
            print(
                "target_position="
                f"{np.array2string(result.target_position, precision=6)}"
            )
            print(
                "reference_action="
                + np.array2string(
                    result.terminal_action.reshape(-1), precision=6
                )
            )
            return

    if best_result is None:
        raise RuntimeError(
            "all attempts violated ground clearance; reduce action amplitude "
            "or min-tip-height"
        )
    raise RuntimeError(
        "no attempt satisfied both offset and stability requirements. "
        f"Best attempt: offset={best_result.achieved_offset:.4f} m, "
        f"position_max={best_result.position_max:.6f} m, "
        f"speed_max={best_result.speed_max:.6f} m/s. Increase "
        "--max-attempts/--hold-seconds or adjust action and tolerance limits."
    )


if __name__ == "__main__":
    main()
