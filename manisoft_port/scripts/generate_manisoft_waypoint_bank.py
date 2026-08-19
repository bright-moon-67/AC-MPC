#!/usr/bin/env python
"""Generate certified random-direction three-waypoint references for ManiSoft.

Each candidate starts from one spatially smooth 18-D muscle-action direction.
The three waypoints use increasing action scales 0.25/0.50/0.75.  Every
waypoint is simulated twice in fresh environments and is accepted only when
both runs hold the tip within the configured position/speed tolerances for the
entire final certification window.

By default the action directions carry a small vertical (z) weight (0.25 vs
1.0 on x/y), so the certified triplets lie almost entirely in the horizontal
plane.  Pass ``--z-variation-probability`` and ``--z-action-weight`` to make a
fraction of triplets rise or fall out of that plane along the z axis, which
adds vertical diversity while keeping the radial three-waypoint structure.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manisoft.envs import VisionLanguageManipulationEnvironment
from manisoft.muscle import SplineMuscle
from manisoft.utils import (
    koopman_section_state,
    load_yaml,
)


TIP_INDICES = np.asarray((30, 31, 32), dtype=np.int64)
ACTION_SHAPE = (6, 3)
MODE_WEIGHTS = np.asarray((1.0, 0.8, 0.6, 0.35, 0.2, 0.12))
AXIS_WEIGHTS = np.asarray((1.0, 1.0, 0.25))


def spatial_basis() -> np.ndarray:
    point = np.arange(1, ACTION_SHAPE[0] + 1)[:, None]
    mode = np.arange(1, ACTION_SHAPE[0] + 1)[None, :]
    return np.sqrt(2.0 / (ACTION_SHAPE[0] + 1)) * np.sin(
        np.pi * point * mode / (ACTION_SHAPE[0] + 1)
    )


def sample_terminal_action(
    rng: np.random.Generator, peak: float, *, z_weight: float = 0.25
) -> np.ndarray:
    coefficients = rng.normal(size=ACTION_SHAPE)
    coefficients *= MODE_WEIGHTS[:, None]
    axis_weights = np.asarray(
        (AXIS_WEIGHTS[0], AXIS_WEIGHTS[1], float(z_weight)),
        dtype=np.float64,
    )
    coefficients *= axis_weights[None, :]
    action = spatial_basis() @ coefficients
    raw_peak = float(np.max(np.abs(action)))
    if raw_peak <= 0:
        raise RuntimeError("sampled a zero terminal action")
    return (action * (peak / raw_peak)).astype(np.float32)


def create_environment(config_path: Path):
    configs = load_yaml(config_path)
    configs["renderer"] = None
    return VisionLanguageManipulationEnvironment.from_dict(configs), configs


def has_ground_clearance(state, min_tip_height: float) -> bool:
    positions = np.asarray(state.element_positions, dtype=np.float64)
    return bool(
        np.min(positions[1:, 2]) >= 0.0 and positions[-1, 2] >= min_tip_height
    )


def minimum_jerk(alpha: np.ndarray) -> np.ndarray:
    return 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5


def get_tip(state) -> tuple[np.ndarray, float]:
    position = np.asarray(state.element_positions[-1], dtype=np.float64)
    velocity = np.asarray(state.element_velocities[-1], dtype=np.float64)
    return position, float(np.linalg.norm(velocity))


@dataclass
class Certificate:
    reference_state: np.ndarray
    reference_action: np.ndarray
    initial_tip: np.ndarray
    reference_tip: np.ndarray
    stable_states: np.ndarray
    stable_positions: np.ndarray
    stable_speeds: np.ndarray
    position_max: float
    speed_max: float

    @property
    def distance_from_initial(self) -> float:
        return float(np.linalg.norm(self.reference_tip - self.initial_tip))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--triplets", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--muscle-torque-scale", type=float, default=30.0)
    parser.add_argument("--base-action-peak", type=float, default=0.30)
    parser.add_argument(
        "--scales", type=float, nargs=3, default=(0.25, 0.50, 0.75)
    )
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=8.0)
    parser.add_argument("--stable-steps", type=int, default=250)
    parser.add_argument("--position-tolerance", type=float, default=0.001)
    parser.add_argument("--speed-tolerance", type=float, default=0.005)
    parser.add_argument("--replay-tip-tolerance", type=float, default=0.001)
    parser.add_argument("--min-tip-height", type=float, default=0.15)
    parser.add_argument(
        "--distance-ranges-cm",
        type=float,
        nargs=6,
        default=(4.0, 8.0, 8.0, 14.0, 12.0, 20.0),
        metavar=("G1_MIN", "G1_MAX", "G2_MIN", "G2_MAX", "G3_MIN", "G3_MAX"),
    )
    parser.add_argument("--min-adjacent-distance-cm", type=float, default=2.5)
    parser.add_argument(
        "--z-variation-probability",
        type=float,
        default=0.5,
        help=(
            "Fraction of triplets whose sampled action direction gets an "
            "enhanced vertical (z) component, lifting the three waypoints "
            "out of the original horizontal plane."
        ),
    )
    parser.add_argument(
        "--z-action-weight",
        type=float,
        default=0.6,
        help=(
            "z-axis action coefficient weight used for z-variant triplets. "
            "0.25 reproduces the original near-horizontal plane; larger "
            "values add a stronger vertical tip displacement."
        ),
    )
    args = parser.parse_args()
    if min(args.triplets, args.max_attempts, args.stable_steps) < 1:
        parser.error("triplets, max-attempts and stable-steps must be positive")
    positive = (
        args.control_hz,
        args.muscle_torque_scale,
        args.base_action_peak,
        args.ramp_seconds,
        args.hold_seconds,
        args.position_tolerance,
        args.speed_tolerance,
        args.replay_tip_tolerance,
        args.min_adjacent_distance_cm,
    )
    if min(positive) <= 0 or args.settle_seconds < 0:
        parser.error("timing, action and tolerance values must be positive")
    if args.base_action_peak > 0.30:
        parser.error("base-action-peak cannot exceed the ManiSoft action limit 0.30")
    scales = np.asarray(args.scales, dtype=np.float64)
    if np.any(scales <= 0) or np.any(np.diff(scales) <= 0) or scales[-1] > 1:
        parser.error("scales must be strictly increasing values in (0,1]")
    ranges = np.asarray(args.distance_ranges_cm, dtype=np.float64).reshape(3, 2)
    if np.any(ranges <= 0) or np.any(ranges[:, 0] >= ranges[:, 1]):
        parser.error("every distance range must have 0 < minimum < maximum")
    if round(args.hold_seconds * args.control_hz) < args.stable_steps:
        parser.error("hold-seconds is shorter than the certification window")
    if not 0.0 <= args.z_variation_probability <= 1.0:
        parser.error("z-variation-probability must be in [0, 1]")
    if args.z_action_weight <= 0:
        parser.error("z-action-weight must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _simulate(args: argparse.Namespace, action: np.ndarray) -> Certificate | None:
    env, configs = create_environment(args.scenario)
    try:
        physics_dt = float(configs["backend"]["dt"])
        actual_control_dt = physics_dt * env.update_interval
        expected_control_dt = 1.0 / args.control_hz
        if not np.isclose(actual_control_dt, expected_control_dt, atol=1e-12, rtol=0):
            raise ValueError(
                f"scenario control period is {actual_control_dt}, expected "
                f"{expected_control_dt} for {args.control_hz:g} Hz"
            )
        env.place()
        muscle = SplineMuscle(
            robot_length=float(configs["softrobot"]["length"]),
            robot_num_elements=int(configs["softrobot"]["num_elements"]),
            number_of_control_points=ACTION_SHAPE[0],
            muscle_torque_scale=args.muscle_torque_scale,
        )
        muscle.set_activation(np.zeros(ACTION_SHAPE, dtype=np.float64))

        def current_torque(element_lengths):
            return muscle.evaluate(element_lengths)

        _, soft_state = env.get_state(has_image=False)
        for _ in range(round(args.settle_seconds * args.control_hz)):
            env.step_with_torque_callback(current_torque)
            _, soft_state = env.get_state(has_image=False)
            if not has_ground_clearance(soft_state, args.min_tip_height):
                return None
        initial_tip, _ = get_tip(soft_state)

        ramp_steps = max(2, round(args.ramp_seconds * args.control_hz))
        ramp = minimum_jerk(np.linspace(0.0, 1.0, ramp_steps))
        for alpha in ramp:
            muscle.set_activation(action.reshape(ACTION_SHAPE) * alpha)
            env.step_with_torque_callback(current_torque)
            _, soft_state = env.get_state(has_image=False)
            if not has_ground_clearance(soft_state, args.min_tip_height):
                return None

        muscle.set_activation(action.reshape(ACTION_SHAPE))
        hold_steps = round(args.hold_seconds * args.control_hz)
        states = np.empty((hold_steps, 45), dtype=np.float32)
        positions = np.empty((hold_steps, 3), dtype=np.float64)
        speeds = np.empty(hold_steps, dtype=np.float64)
        for index in range(hold_steps):
            env.step_with_torque_callback(current_torque)
            _, soft_state = env.get_state(has_image=False)
            if not has_ground_clearance(soft_state, args.min_tip_height):
                return None
            states[index] = koopman_section_state(soft_state)
            positions[index], speeds[index] = get_tip(soft_state)

        stable_states = states[-args.stable_steps :].copy()
        stable_positions = positions[-args.stable_steps :].copy()
        stable_speeds = speeds[-args.stable_steps :].copy()
        center = stable_positions.mean(axis=0)
        reference_index = int(
            np.argmin(np.linalg.norm(stable_positions - center[None, :], axis=1))
        )
        reference_state = stable_states[reference_index].copy()
        reference_tip = reference_state[TIP_INDICES].astype(np.float64)
        position_max = float(
            np.linalg.norm(stable_positions - reference_tip[None, :], axis=1).max()
        )
        speed_max = float(stable_speeds.max())
        if position_max > args.position_tolerance or speed_max > args.speed_tolerance:
            return None
        return Certificate(
            reference_state=reference_state,
            reference_action=action.astype(np.float32, copy=True),
            initial_tip=np.asarray(initial_tip, dtype=np.float64),
            reference_tip=reference_tip,
            stable_states=stable_states,
            stable_positions=stable_positions,
            stable_speeds=stable_speeds,
            position_max=position_max,
            speed_max=speed_max,
        )
    finally:
        del env
        gc.collect()


def _save_waypoint(
    path: Path,
    generated: Certificate,
    replayed: Certificate,
    *,
    scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int32),
            reference_state=generated.reference_state,
            reference_action=generated.reference_action,
            reference_tip_position=generated.reference_tip,
            initial_tip_position=generated.initial_tip,
            action_scale=np.asarray(scale, dtype=np.float32),
            stable_window_states=generated.stable_states,
            stable_window_tip_positions=generated.stable_positions,
            stable_window_tip_speeds=generated.stable_speeds,
            replay_reference_state=replayed.reference_state,
            replay_reference_tip_position=replayed.reference_tip,
            replay_stable_window_tip_positions=replayed.stable_positions,
            replay_stable_window_tip_speeds=replayed.stable_speeds,
        )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    scenario = args.scenario.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not scenario.is_file():
        raise FileNotFoundError(f"Missing scenario: {scenario}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite waypoint bank: {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    scales = np.asarray(args.scales, dtype=np.float64)
    distance_ranges = 0.01 * np.asarray(
        args.distance_ranges_cm, dtype=np.float64
    ).reshape(3, 2)
    min_adjacent = 0.01 * args.min_adjacent_distance_cm
    triplets: list[dict] = []

    for attempt in range(args.max_attempts):
        if len(triplets) >= args.triplets:
            break
        use_z_variant = (
            args.z_variation_probability > 0.0
            and rng.random() < args.z_variation_probability
        )
        z_weight = (
            args.z_action_weight if use_z_variant else float(AXIS_WEIGHTS[2])
        )
        base_action = sample_terminal_action(
            rng, args.base_action_peak, z_weight=z_weight
        ).reshape(-1)
        generated: list[Certificate] = []
        failed_reason = None
        for index, scale in enumerate(scales):
            certificate = _simulate(args, base_action * scale)
            if certificate is None:
                failed_reason = f"waypoint_{index + 1}_generation_stability"
                break
            distance = certificate.distance_from_initial
            if not distance_ranges[index, 0] <= distance <= distance_ranges[index, 1]:
                failed_reason = f"waypoint_{index + 1}_distance_{distance:.4f}"
                break
            generated.append(certificate)
        if failed_reason is None:
            radial = np.asarray([item.distance_from_initial for item in generated])
            adjacent = np.linalg.norm(
                np.diff(np.stack([item.reference_tip for item in generated]), axis=0),
                axis=1,
            )
            if np.any(np.diff(radial) <= 0):
                failed_reason = "non_monotonic_radial_distance"
            elif np.any(adjacent < min_adjacent):
                failed_reason = "adjacent_waypoints_too_close"

        replayed: list[Certificate] = []
        if failed_reason is None:
            for index, (scale, original) in enumerate(zip(scales, generated)):
                replay = _simulate(args, base_action * scale)
                if replay is None:
                    failed_reason = f"waypoint_{index + 1}_replay_stability"
                    break
                tip_error = float(
                    np.linalg.norm(replay.reference_tip - original.reference_tip)
                )
                if tip_error > args.replay_tip_tolerance:
                    failed_reason = f"waypoint_{index + 1}_replay_error_{tip_error:.6f}"
                    break
                replayed.append(replay)

        if failed_reason is not None:
            print(
                json.dumps(
                    {"attempt": attempt, "accepted": False, "reason": failed_reason},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        triplet_index = len(triplets)
        waypoint_rows = []
        for waypoint_index, (scale, first, replay) in enumerate(
            zip(scales, generated, replayed)
        ):
            relative = Path(f"triplet_{triplet_index:04d}") / f"waypoint_{waypoint_index + 1}.npz"
            path = output / relative
            _save_waypoint(path, first, replay, scale=float(scale))
            waypoint_rows.append(
                {
                    "index": waypoint_index,
                    "scale": float(scale),
                    "reference": relative.as_posix(),
                    "sha256": _sha256(path),
                    "tip_position_m": first.reference_tip.tolist(),
                    "distance_from_initial_m": first.distance_from_initial,
                    "generation_position_max_m": first.position_max,
                    "generation_speed_max_m_per_s": first.speed_max,
                    "replay_position_max_m": replay.position_max,
                    "replay_speed_max_m_per_s": replay.speed_max,
                    "replay_tip_error_m": float(
                        np.linalg.norm(replay.reference_tip - first.reference_tip)
                    ),
                }
            )
        triplets.append(
            {
                "index": triplet_index,
                "source_attempt": attempt,
                "base_action": base_action.tolist(),
                "z_variant": bool(use_z_variant),
                "z_action_weight": float(z_weight),
                "waypoints": waypoint_rows,
            }
        )
        print(
            json.dumps(
                {
                    "attempt": attempt,
                    "accepted": True,
                    "triplet": triplet_index,
                    "distances_cm": [
                        100.0 * item.distance_from_initial for item in generated
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(triplets) != args.triplets:
        raise RuntimeError(
            f"Generated {len(triplets)}/{args.triplets} certified triplets in "
            f"{args.max_attempts} attempts"
        )
    manifest = {
        "schema_version": 1,
        "kind": "manisoft_certified_three_waypoint_reference_bank",
        "scenario": str(scenario),
        "scenario_sha256": _sha256(scenario),
        "triplet_count": len(triplets),
        "z_variant_count": sum(1 for t in triplets if t["z_variant"]),
        "waypoint_count": 3,
        "state_dim": 45,
        "action_dim": 18,
        "action_limit": 0.30,
        "seed": args.seed,
        "certification": {
            "control_hz": args.control_hz,
            "stable_steps": args.stable_steps,
            "stable_seconds": args.stable_steps / args.control_hz,
            "position_tolerance_m": args.position_tolerance,
            "speed_tolerance_m_per_s": args.speed_tolerance,
            "independent_replay": True,
            "replay_tip_tolerance_m": args.replay_tip_tolerance,
            "distance_ranges_cm": np.asarray(args.distance_ranges_cm).reshape(3, 2).tolist(),
            "min_adjacent_distance_cm": args.min_adjacent_distance_cm,
            "z_variation": {
                "probability": args.z_variation_probability,
                "action_weight": args.z_action_weight,
            },
        },
        "generator_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "triplets": triplets,
    }
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "triplets": len(triplets)}))


if __name__ == "__main__":
    main()
