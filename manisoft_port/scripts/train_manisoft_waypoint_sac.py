#!/usr/bin/env python
"""Train or resume the obstacle-free ManiSoft waypoint SAC controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml

from antmaze_ac.envs.manisoft_waypoint_sac_env import ManiSoftWaypointSACEnv
from antmaze_ac.envs.waypoint_paths import CURRICULUM_STAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--config", default="configs/manisoft_waypoint_sac_physical.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--curriculum",
        choices=CURRICULUM_STAGES,
        default=None,
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument(
        "--eval-num-envs",
        type=int,
        default=1,
        help="Parallel environments used by periodic evaluation.",
    )
    parser.add_argument(
        "--eval-panel-seed",
        type=int,
        default=730000,
        help="First reset seed in the repeatable periodic evaluation panel.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--observation-mode",
        choices=("legacy70", "gate74"),
        default=None,
        help=(
            "Keep the original 45-D physical state fixed and choose either "
            "the compatible 70-D observation or four appended gate/prior features."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override SAC learning rate, including when resuming.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument(
        "--waypoint-segment-count-range",
        default=None,
        help="Override inclusive short-waypoint segment counts, e.g. 2,3.",
    )
    parser.add_argument(
        "--waypoint-segment-length-range",
        default=None,
        help="Override per-segment lengths in metres, e.g. 0.08,0.16.",
    )
    parser.add_argument(
        "--waypoint-segment-count-probabilities",
        default=None,
        help=(
            "Comma-separated probabilities for every integer in the segment "
            "count range, e.g. 0.25,0.75 for counts 2,3."
        ),
    )
    parser.add_argument(
        "--waypoint-minimum-turn-degrees",
        type=float,
        default=None,
        help="Override the smallest heading change between adjacent segments.",
    )
    parser.add_argument(
        "--waypoint-maximum-turn-degrees",
        type=float,
        default=None,
        help="Override the largest heading change between adjacent segments.",
    )
    parser.add_argument(
        "--waypoint-hard-turn-probability",
        type=float,
        default=None,
        help=(
            "Probability of drawing an entire training path from the configured "
            "hard-turn range while retaining broad-path rehearsal otherwise."
        ),
    )
    parser.add_argument(
        "--waypoint-maximum-extent",
        type=float,
        default=None,
        help="Override the maximum distance of any waypoint from the entry pose.",
    )
    parser.add_argument(
        "--entry-sampling-weights",
        default=None,
        help=(
            "Comma-separated reset weights for every certified entry posture; "
            "evaluation remains uniform unless its config explicitly overrides it."
        ),
    )
    parser.add_argument(
        "--cartesian-action-leak",
        type=float,
        default=None,
        help="Override the table Cartesian action integrator leak.",
    )
    parser.add_argument(
        "--cartesian-action-step-scale",
        type=float,
        default=None,
        help="Override the table Cartesian action integrator step scale.",
    )
    parser.add_argument(
        "--cartesian-prior-weight",
        type=float,
        default=None,
        help=(
            "Blend weight in [0, 1] for the calibrated Cartesian controller "
            "during both training and periodic evaluation."
        ),
    )
    parser.add_argument(
        "--cartesian-prior-proportional-gain",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--cartesian-prior-feedforward-scale",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--cartesian-prior-internal-waypoints-only",
        action="store_true",
        help=(
            "Use the Cartesian prior only until all ordered internal "
            "waypoints have been captured."
        ),
    )
    parser.add_argument(
        "--cartesian-prior-residual-scale",
        type=float,
        default=None,
        help="Bounded SAC residual authority added to the Cartesian prior.",
    )
    parser.add_argument(
        "--equilibrium-path-prior-weight",
        type=float,
        default=None,
        help=(
            "Blend weight for the certified long-range equilibrium-path "
            "reference controller."
        ),
    )
    parser.add_argument(
        "--equilibrium-path-residual-scale",
        type=float,
        default=None,
        help="SAC residual authority added after the equilibrium-path prior.",
    )
    parser.add_argument("--tracking-guard", type=float, default=None)
    parser.add_argument("--policy-action-penalty-scale", type=float, default=None)
    parser.add_argument("--target-lead-distance", type=float, default=None)
    parser.add_argument("--lookahead-distance", type=float, default=None)
    parser.add_argument("--min-desired-speed", type=float, default=None)
    parser.add_argument("--max-desired-speed", type=float, default=None)
    parser.add_argument("--terminal-precision-scale", type=float, default=None)
    parser.add_argument(
        "--terminal-distance-penalty-scale", type=float, default=None
    )
    parser.add_argument(
        "--waypoint-single-line-probability",
        type=float,
        default=None,
        help="Probability of replaying one short straight line in waypoint training.",
    )
    parser.add_argument(
        "--internal-waypoint-capture-radius",
        type=float,
        default=None,
        help="Training capture radius for ordered internal waypoints in metres.",
    )
    parser.add_argument(
        "--internal-waypoint-bonus",
        type=float,
        default=None,
        help="One-time reward for geometrically capturing an internal waypoint.",
    )
    parser.add_argument(
        "--internal-waypoint-progress-scale",
        type=float,
        default=None,
        help=(
            "Dense reward scale for reducing distance to the currently gated "
            "internal waypoint."
        ),
    )
    parser.add_argument(
        "--internal-waypoint-distance-penalty-scale",
        type=float,
        default=None,
        help=(
            "Persistent hinge penalty scale for distance remaining outside "
            "the active internal-waypoint capture ball."
        ),
    )
    parser.add_argument(
        "--waypoint-stall-steps",
        type=int,
        default=None,
        help=(
            "Truncate after this many steps without improving active-waypoint "
            "distance; zero disables the detector."
        ),
    )
    parser.add_argument(
        "--waypoint-stall-distance-epsilon",
        type=float,
        default=None,
        help="Minimum distance improvement that resets waypoint-stall timing.",
    )
    parser.add_argument(
        "--eval-waypoint-single-line-probability",
        type=float,
        default=None,
        help="Override rehearsal probability only in periodic evaluation.",
    )
    parser.add_argument(
        "--eval-waypoint-segment-count-range",
        default=None,
        help="Override segment counts only in periodic evaluation, e.g. 3,3.",
    )
    parser.add_argument(
        "--eval-waypoint-minimum-turn-degrees",
        type=float,
        default=None,
        help="Override minimum waypoint turn angle only in periodic evaluation.",
    )
    parser.add_argument(
        "--eval-waypoint-maximum-turn-degrees",
        type=float,
        default=None,
        help="Override maximum waypoint turn angle only in periodic evaluation.",
    )
    parser.add_argument(
        "--eval-waypoint-hard-turn-probability",
        type=float,
        default=None,
        help="Override hard-turn episode probability only in periodic evaluation.",
    )
    parser.add_argument(
        "--eval-internal-waypoint-capture-radius",
        type=float,
        default=None,
        help="Override the internal-waypoint radius only in periodic evaluation.",
    )
    parser.add_argument(
        "--eval-waypoint-stall-steps",
        type=int,
        default=None,
        help=(
            "Override active-waypoint stall truncation only in periodic "
            "evaluation; zero disables it for a comparable fixed panel."
        ),
    )
    parser.add_argument(
        "--actor-anchor-coef",
        type=float,
        default=0.0,
        help="MSE coefficient keeping the actor near the resumed source policy.",
    )
    parser.add_argument(
        "--frozen-base-model",
        default=None,
        help=(
            "Frozen SAC checkpoint whose deterministic action is the base of a "
            "residual SAC policy. Combine with --resume when continuing an "
            "already-trained residual policy on a new curriculum."
        ),
    )
    parser.add_argument(
        "--frozen-base-vec-normalize",
        default=None,
        help="VecNormalize state paired with --frozen-base-model.",
    )
    parser.add_argument(
        "--residual-action-scale",
        type=float,
        default=0.10,
        help="Largest residual correction per normalized action axis.",
    )
    parser.add_argument(
        "--residual-action-penalty-scale",
        type=float,
        default=0.0,
        help="Reward penalty coefficient on mean squared residual action.",
    )
    parser.add_argument(
        "--residual-stall-activation-steps",
        type=int,
        default=0,
        help=(
            "Keep the frozen base exact until this many controller steps pass "
            "without active-waypoint improvement; zero always enables residuals."
        ),
    )
    parser.add_argument(
        "--residual-stall-ramp-steps",
        type=int,
        default=1,
        help="Steps used to ramp a stall-activated residual from zero to full.",
    )
    parser.add_argument(
        "--ent-coef",
        default=None,
        help="SAC entropy coefficient (float, auto, or auto_INITIAL).",
    )
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument(
        "--net-arch",
        default=None,
        help="Comma-separated SAC actor/critic hidden widths, e.g. 512,512,256.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        default=None,
        help="SAC .zip checkpoint to continue with the new curriculum/environment.",
    )
    parser.add_argument(
        "--replay-buffer",
        default=None,
        help="Optional replay-buffer .pkl. Omit to start a fresh buffer on resume.",
    )
    parser.add_argument(
        "--source-policy-warmup",
        action="store_true",
        help=(
            "For anchored curriculum transfer, fill the new buffer with "
            "deterministic current-policy actions instead of unsafe uniform "
            "random residual actions."
        ),
    )
    parser.add_argument(
        "--actor-learning-delay-steps",
        type=int,
        default=0,
        help=(
            "Additional critic-only transitions after replay warm-up before "
            "an anchored resumed actor may update."
        ),
    )
    parser.add_argument(
        "--freeze-vec-normalize",
        action="store_true",
        help="Keep loaded observation-normalization statistics fixed on resume.",
    )
    parser.add_argument(
        "--vec-normalize",
        default=None,
        help="VecNormalize .pkl; defaults to OUTPUT/vecnormalize.pkl on resume.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow a nonempty output directory (normally only --resume does this).",
    )
    parser.add_argument(
        "--zero-init-actor",
        action="store_true",
        help=(
            "Initialize a fresh actor's deterministic mean at exactly zero; "
            "useful when zero residual reproduces a certified controller."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast wiring test: tiny buffer/batch and no periodic evaluation.",
    )
    return parser.parse_args()


def _positive_int(name: str, value: Any) -> int:
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be positive")
    return converted


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing SAC config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SAC config must contain a mapping")
    for section in ("environment", "sac", "training"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"SAC config is missing section: {section}")
    environment = dict(payload["environment"])
    if args.observation_mode is not None:
        environment["observation_mode"] = args.observation_mode
    training = dict(payload["training"])
    if args.curriculum is not None:
        environment["curriculum"] = args.curriculum
    if args.waypoint_segment_count_range is not None:
        counts = [
            int(value)
            for value in args.waypoint_segment_count_range.split(",")
            if value.strip()
        ]
        if len(counts) != 2 or counts[0] < 1 or counts[0] > counts[1]:
            raise ValueError(
                "waypoint-segment-count-range must be an increasing positive pair"
            )
        environment["waypoint_segment_count_range"] = counts
    if args.waypoint_segment_length_range is not None:
        lengths = [
            float(value)
            for value in args.waypoint_segment_length_range.split(",")
            if value.strip()
        ]
        if len(lengths) != 2 or lengths[0] <= 0 or lengths[0] > lengths[1]:
            raise ValueError(
                "waypoint-segment-length-range must be an increasing positive pair"
            )
        environment["waypoint_segment_length_range"] = lengths
    if args.waypoint_segment_count_probabilities is not None:
        probabilities = [
            float(value)
            for value in args.waypoint_segment_count_probabilities.split(",")
            if value.strip()
        ]
        count_low, count_high = [
            int(value) for value in environment["waypoint_segment_count_range"]
        ]
        expected = count_high - count_low + 1
        if (
            len(probabilities) != expected
            or any(value < 0 for value in probabilities)
            or sum(probabilities) <= 0
        ):
            raise ValueError(
                "waypoint-segment-count-probabilities must contain one "
                "non-negative value per segment count and have positive sum"
            )
        environment["waypoint_segment_count_probabilities"] = probabilities
    if args.waypoint_minimum_turn_degrees is not None:
        if not 0.0 <= args.waypoint_minimum_turn_degrees <= 180.0:
            raise ValueError(
                "waypoint-minimum-turn-degrees must lie in [0, 180]"
            )
        environment["waypoint_minimum_turn_degrees"] = float(
            args.waypoint_minimum_turn_degrees
        )
    if args.waypoint_maximum_turn_degrees is not None:
        if not 0.0 < args.waypoint_maximum_turn_degrees <= 180.0:
            raise ValueError(
                "waypoint-maximum-turn-degrees must lie in (0, 180]"
            )
        environment["waypoint_maximum_turn_degrees"] = float(
            args.waypoint_maximum_turn_degrees
        )
    if float(environment["waypoint_minimum_turn_degrees"]) > float(
        environment["waypoint_maximum_turn_degrees"]
    ):
        raise ValueError("minimum waypoint turn cannot exceed the maximum")
    if args.waypoint_hard_turn_probability is not None:
        if not 0.0 <= args.waypoint_hard_turn_probability <= 1.0:
            raise ValueError(
                "waypoint-hard-turn-probability must lie in [0, 1]"
            )
        environment["waypoint_hard_turn_probability"] = float(
            args.waypoint_hard_turn_probability
        )
    if args.waypoint_maximum_extent is not None:
        if args.waypoint_maximum_extent <= 0:
            raise ValueError("waypoint-maximum-extent must be positive")
        environment["waypoint_maximum_extent"] = float(
            args.waypoint_maximum_extent
        )
    if args.cartesian_action_leak is not None:
        if not 0.0 <= args.cartesian_action_leak < 1.0:
            raise ValueError("cartesian-action-leak must lie in [0, 1)")
        environment["cartesian_action_leak"] = float(args.cartesian_action_leak)
    if args.cartesian_action_step_scale is not None:
        if args.cartesian_action_step_scale <= 0:
            raise ValueError("cartesian-action-step-scale must be positive")
        environment["cartesian_action_step_scale"] = float(
            args.cartesian_action_step_scale
        )
    if args.cartesian_prior_weight is not None:
        if not 0.0 <= args.cartesian_prior_weight <= 1.0:
            raise ValueError("cartesian-prior-weight must lie in [0, 1]")
        environment["cartesian_prior_weight"] = float(
            args.cartesian_prior_weight
        )
    for key, value in (
        (
            "cartesian_prior_proportional_gain",
            args.cartesian_prior_proportional_gain,
        ),
        (
            "cartesian_prior_feedforward_scale",
            args.cartesian_prior_feedforward_scale,
        ),
    ):
        if value is not None:
            if value < 0:
                raise ValueError(f"{key.replace('_', '-')} must be non-negative")
            environment[key] = float(value)
    if args.cartesian_prior_internal_waypoints_only:
        environment["cartesian_prior_internal_waypoints_only"] = True
    if args.cartesian_prior_residual_scale is not None:
        if not 0.0 <= args.cartesian_prior_residual_scale <= 1.0:
            raise ValueError("cartesian-prior-residual-scale must lie in [0, 1]")
        environment["cartesian_prior_residual_scale"] = float(
            args.cartesian_prior_residual_scale
        )
    if args.equilibrium_path_prior_weight is not None:
        if not 0.0 <= args.equilibrium_path_prior_weight <= 1.0:
            raise ValueError("equilibrium-path-prior-weight must lie in [0, 1]")
        environment["equilibrium_path_prior_weight"] = float(
            args.equilibrium_path_prior_weight
        )
    if args.equilibrium_path_residual_scale is not None:
        if not 0.0 <= args.equilibrium_path_residual_scale <= 1.0:
            raise ValueError("equilibrium-path-residual-scale must lie in [0, 1]")
        environment["equilibrium_path_residual_scale"] = float(
            args.equilibrium_path_residual_scale
        )
    if args.entry_sampling_weights is not None:
        weights = [
            float(value)
            for value in args.entry_sampling_weights.split(",")
            if value.strip()
        ]
        if not weights or min(weights) < 0 or not np.isfinite(weights).all():
            raise ValueError(
                "entry-sampling-weights must contain finite non-negative values"
            )
        if sum(weights) <= 0:
            raise ValueError("entry-sampling-weights must have positive mass")
        environment["entry_sampling_weights"] = weights
    positive_environment_overrides = {
        "tracking_guard": args.tracking_guard,
        "target_lead_distance": args.target_lead_distance,
        "lookahead_distance": args.lookahead_distance,
        "min_desired_speed": args.min_desired_speed,
        "max_desired_speed": args.max_desired_speed,
    }
    for key, value in positive_environment_overrides.items():
        if value is not None:
            if value <= 0:
                raise ValueError(f"{key.replace('_', '-')} must be positive")
            environment[key] = float(value)
    nonnegative_environment_overrides = {
        "terminal_precision_scale": args.terminal_precision_scale,
        "terminal_distance_penalty_scale": args.terminal_distance_penalty_scale,
    }
    for key, value in nonnegative_environment_overrides.items():
        if value is not None:
            if value < 0:
                raise ValueError(f"{key.replace('_', '-')} must be non-negative")
            environment[key] = float(value)
    if float(environment["min_desired_speed"]) > float(
        environment["max_desired_speed"]
    ):
        raise ValueError("minimum desired speed cannot exceed maximum")
    if args.waypoint_single_line_probability is not None:
        if not 0.0 <= args.waypoint_single_line_probability <= 1.0:
            raise ValueError(
                "waypoint-single-line-probability must lie in [0, 1]"
            )
        environment["waypoint_single_line_probability"] = float(
            args.waypoint_single_line_probability
        )
    if args.internal_waypoint_capture_radius is not None:
        if args.internal_waypoint_capture_radius <= 0:
            raise ValueError("internal-waypoint-capture-radius must be positive")
        environment["internal_waypoint_capture_radius"] = float(
            args.internal_waypoint_capture_radius
        )
    if args.internal_waypoint_bonus is not None:
        if args.internal_waypoint_bonus < 0:
            raise ValueError("internal-waypoint-bonus must be non-negative")
        environment["internal_waypoint_bonus"] = float(
            args.internal_waypoint_bonus
        )
    if args.internal_waypoint_progress_scale is not None:
        if args.internal_waypoint_progress_scale < 0:
            raise ValueError(
                "internal-waypoint-progress-scale must be non-negative"
            )
        environment["internal_waypoint_progress_scale"] = float(
            args.internal_waypoint_progress_scale
        )
    if args.internal_waypoint_distance_penalty_scale is not None:
        if args.internal_waypoint_distance_penalty_scale < 0:
            raise ValueError(
                "internal-waypoint-distance-penalty-scale must be non-negative"
            )
        environment["internal_waypoint_distance_penalty_scale"] = float(
            args.internal_waypoint_distance_penalty_scale
        )
    if args.waypoint_stall_steps is not None:
        if args.waypoint_stall_steps < 0:
            raise ValueError("waypoint-stall-steps must be non-negative")
        environment["waypoint_stall_steps"] = int(args.waypoint_stall_steps)
    if args.waypoint_stall_distance_epsilon is not None:
        if args.waypoint_stall_distance_epsilon <= 0:
            raise ValueError(
                "waypoint-stall-distance-epsilon must be positive"
            )
        environment["waypoint_stall_distance_epsilon"] = float(
            args.waypoint_stall_distance_epsilon
        )
    if args.policy_action_penalty_scale is not None:
        if args.policy_action_penalty_scale < 0:
            raise ValueError("policy-action-penalty-scale must be non-negative")
        environment["policy_action_penalty_scale"] = float(
            args.policy_action_penalty_scale
        )
    if args.actor_anchor_coef < 0:
        raise ValueError("actor-anchor-coef must be non-negative")
    if (
        args.actor_anchor_coef > 0
        and args.resume is None
        and not args.zero_init_actor
    ):
        raise ValueError(
            "fresh actor anchoring requires --zero-init-actor; otherwise use "
            "--resume as the source policy"
        )
    if args.zero_init_actor and args.resume is not None:
        raise ValueError("zero-init-actor is only valid for a fresh model")
    if args.source_policy_warmup and (
        args.actor_anchor_coef <= 0
        or args.replay_buffer is not None
        or (args.resume is None and args.frozen_base_model is None)
    ):
        raise ValueError(
            "source-policy-warmup requires anchored --resume or a frozen-base "
            "residual transfer, without --replay-buffer"
        )
    if args.actor_learning_delay_steps < 0:
        raise ValueError("actor-learning-delay-steps must be non-negative")
    if args.actor_learning_delay_steps > 0 and (
        args.actor_anchor_coef <= 0
        or (args.resume is None and args.frozen_base_model is None)
    ):
        raise ValueError(
            "actor-learning-delay-steps requires an anchored curriculum transfer"
        )
    if (
        args.freeze_vec_normalize
        and args.resume is None
        and args.frozen_base_model is None
    ):
        raise ValueError(
            "freeze-vec-normalize requires --resume or --frozen-base-model"
        )
    residual_requested = args.frozen_base_model is not None or (
        args.frozen_base_vec_normalize is not None
    )
    if residual_requested and (
        args.frozen_base_model is None
        or args.frozen_base_vec_normalize is None
    ):
        raise ValueError(
            "frozen residual mode requires both --frozen-base-model and "
            "--frozen-base-vec-normalize"
        )
    if not 0.0 < args.residual_action_scale <= 1.0:
        raise ValueError("residual-action-scale must lie in (0, 1]")
    if args.residual_action_penalty_scale < 0:
        raise ValueError("residual-action-penalty-scale must be non-negative")
    if args.residual_stall_activation_steps < 0:
        raise ValueError(
            "residual-stall-activation-steps must be non-negative"
        )
    if args.residual_stall_ramp_steps < 1:
        raise ValueError("residual-stall-ramp-steps must be positive")
    if (
        args.eval_waypoint_single_line_probability is not None
        and not 0.0 <= args.eval_waypoint_single_line_probability <= 1.0
    ):
        raise ValueError(
            "eval-waypoint-single-line-probability must lie in [0, 1]"
        )
    if (
        args.eval_waypoint_hard_turn_probability is not None
        and not 0.0 <= args.eval_waypoint_hard_turn_probability <= 1.0
    ):
        raise ValueError(
            "eval-waypoint-hard-turn-probability must lie in [0, 1]"
        )
    if args.eval_waypoint_segment_count_range is not None:
        eval_counts = [
            int(value)
            for value in args.eval_waypoint_segment_count_range.split(",")
            if value.strip()
        ]
        if (
            len(eval_counts) != 2
            or eval_counts[0] < 1
            or eval_counts[0] > eval_counts[1]
        ):
            raise ValueError(
                "eval-waypoint-segment-count-range must be an increasing positive pair"
            )
    if (
        args.eval_waypoint_minimum_turn_degrees is not None
        and not 0.0 <= args.eval_waypoint_minimum_turn_degrees <= 180.0
    ):
        raise ValueError(
            "eval-waypoint-minimum-turn-degrees must lie in [0, 180]"
        )
    if (
        args.eval_waypoint_maximum_turn_degrees is not None
        and not 0.0 < args.eval_waypoint_maximum_turn_degrees <= 180.0
    ):
        raise ValueError(
            "eval-waypoint-maximum-turn-degrees must lie in (0, 180]"
        )
    if args.eval_panel_seed < 0:
        raise ValueError("eval-panel-seed must be non-negative")
    if args.eval_num_envs < 1:
        raise ValueError("eval-num-envs must be positive")
    if (
        args.eval_internal_waypoint_capture_radius is not None
        and args.eval_internal_waypoint_capture_radius <= 0
    ):
        raise ValueError(
            "eval-internal-waypoint-capture-radius must be positive"
        )
    if (
        args.eval_waypoint_stall_steps is not None
        and args.eval_waypoint_stall_steps < 0
    ):
        raise ValueError("eval-waypoint-stall-steps must be non-negative")
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("learning-rate must be positive")
        payload["sac"] = {**payload["sac"], "learning_rate": args.learning_rate}
    sac = dict(payload["sac"])
    if args.batch_size is not None:
        sac["batch_size"] = _positive_int("batch_size", args.batch_size)
    if args.gradient_steps is not None:
        sac["gradient_steps"] = _positive_int(
            "gradient_steps", args.gradient_steps
        )
    if args.learning_starts is not None:
        sac["learning_starts"] = _positive_int(
            "learning_starts", args.learning_starts
        )
    if args.ent_coef is not None:
        try:
            ent_coef: str | float = float(args.ent_coef)
        except ValueError:
            ent_coef = args.ent_coef
        sac["ent_coef"] = ent_coef
    if args.target_entropy is not None:
        sac["target_entropy"] = float(args.target_entropy)
    if args.net_arch is not None:
        widths = [int(value) for value in args.net_arch.split(",") if value.strip()]
        if not widths or min(widths) < 1:
            raise ValueError("net-arch must contain positive comma-separated widths")
        sac["net_arch"] = widths
    payload["sac"] = sac
    for key in (
        "total_timesteps",
        "num_envs",
        "checkpoint_freq",
        "eval_freq",
        "eval_episodes",
    ):
        value = getattr(args, key)
        if value is not None:
            training[key] = value
    if args.smoke:
        training.update(
            {
                "total_timesteps": 16,
                "num_envs": 1,
                "checkpoint_freq": 16,
                "eval_freq": 0,
                "eval_episodes": 1,
            }
        )
        payload["sac"] = {
            **payload["sac"],
            "buffer_size": 256,
            "learning_starts": 4,
            "batch_size": 4,
            "net_arch": [32, 32],
        }
        environment["episode_steps"] = min(int(environment["episode_steps"]), 20)
    payload["environment"] = environment
    payload["training"] = training
    for path_key in (
        "entry_bank_path",
        "table_action_calibration_path",
        "table_equilibrium_path_bank_path",
        "table_pose_path_bank_path",
        "table_pose_map_path",
    ):
        configured_path = environment.get(path_key)
        if configured_path is not None:
            resolved_path = Path(configured_path).expanduser()
            if not resolved_path.is_absolute():
                resolved_path = (path.parent.parent / resolved_path).resolve()
            environment[path_key] = str(resolved_path)
    for name in ("total_timesteps", "num_envs", "checkpoint_freq", "eval_episodes"):
        training[name] = _positive_int(name, training[name])
    training["eval_freq"] = int(training["eval_freq"])
    if training["eval_freq"] < 0:
        raise ValueError("eval_freq cannot be negative")
    return payload


def _make_env_factory(
    scenario: Path,
    environment_config: dict[str, Any],
    monitor_path: Path,
    rank: int,
    seed: int,
    fixed_panel_seed: int | None = None,
    residual_config: dict[str, Any] | None = None,
):
    def initialize():
        from stable_baselines3.common.monitor import Monitor
        from antmaze_ac.envs.fixed_seed_panel import FixedSeedPanelWrapper
        from antmaze_ac.envs.frozen_base_residual import (
            FrozenBaseResidualActionWrapper,
        )

        env = ManiSoftWaypointSACEnv(scenario, **environment_config)
        if residual_config is not None:
            env = FrozenBaseResidualActionWrapper(env, **residual_config)
        if fixed_panel_seed is not None:
            env = FixedSeedPanelWrapper(env, fixed_panel_seed)
        return Monitor(
            env,
            filename=str(monitor_path / f"worker_{rank}.csv"),
            info_keywords=(
                "is_success",
                "path_length",
                "waypoint_count",
                "waypoints_completed",
                "internal_waypoints_completed",
                "path_progress",
                "distance",
                "final_distance",
                "whole_arm_table_clearance",
                "action_rate_clipped_ratio",
                "table_violation",
                "terminal_timeout",
                "stalled",
                "path_progress_stalled",
                "waypoint_stalled",
                "dynamics_violation",
                "path_family",
                "path_generation_mode",
            )
            + (
                (
                    "residual_action_penalty",
                    "residual_activation_factor",
                )
                if residual_config is not None
                else ()
            ),
        )

    return initialize


def _make_vector_env(
    scenario: Path,
    environment_config: dict[str, Any],
    monitor_path: Path,
    num_envs: int,
    seed: int,
    fixed_panel_seed: int | None = None,
    fixed_panel_seed_stride: int = 0,
    residual_config: dict[str, Any] | None = None,
):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    monitor_path.mkdir(parents=True, exist_ok=True)
    factories = [
        _make_env_factory(
            scenario,
            environment_config,
            monitor_path,
            rank,
            seed,
            (
                None
                if fixed_panel_seed is None
                else fixed_panel_seed + rank * fixed_panel_seed_stride
            ),
            residual_config,
        )
        for rank in range(num_envs)
    ]
    if num_envs == 1:
        vector_env = DummyVecEnv(factories)
    else:
        vector_env = SubprocVecEnv(factories, start_method="spawn")
    vector_env.seed(seed)
    return vector_env


def _resolved_vecnormalize_path(args: argparse.Namespace, output: Path) -> Path | None:
    if args.vec_normalize is not None:
        return Path(args.vec_normalize).expanduser().resolve()
    if args.frozen_base_vec_normalize is not None:
        return Path(args.frozen_base_vec_normalize).expanduser().resolve()
    candidate = output / "vecnormalize.pkl"
    return candidate if args.resume is not None and candidate.is_file() else None


def main() -> None:
    args = parse_args()
    config = _load_config(args)
    scenario = Path(args.scenario).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not scenario.is_file():
        raise FileNotFoundError(f"missing ManiSoft scenario: {scenario}")
    if output.exists() and any(output.iterdir()):
        if args.resume is None and not args.allow_existing_output:
            raise FileExistsError(
                f"output is nonempty: {output}; use a fresh path or --allow-existing-output"
            )
    output.mkdir(parents=True, exist_ok=True)
    monitor_path = output / "monitor"
    monitor_path.mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device != "cpu":
        # Ampere/Ada TF32 materially speeds the larger SAC variants while the
        # simulator and replay data remain float32.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    environment_config = dict(config["environment"])
    sac_config = dict(config["sac"])
    training = dict(config["training"])
    num_envs = int(training["num_envs"])
    residual_config = None
    if args.frozen_base_model is not None:
        frozen_model_path = Path(args.frozen_base_model).expanduser().resolve()
        frozen_normalizer_path = (
            Path(args.frozen_base_vec_normalize).expanduser().resolve()
        )
        if not frozen_model_path.is_file():
            raise FileNotFoundError(f"missing frozen SAC model: {frozen_model_path}")
        if not frozen_normalizer_path.is_file():
            raise FileNotFoundError(
                f"missing frozen VecNormalize state: {frozen_normalizer_path}"
            )
        residual_config = {
            "frozen_model_path": str(frozen_model_path),
            "frozen_vecnormalize_path": str(frozen_normalizer_path),
            "residual_action_scale": float(args.residual_action_scale),
            "residual_action_penalty_scale": float(
                args.residual_action_penalty_scale
            ),
            "residual_stall_activation_steps": int(
                args.residual_stall_activation_steps
            ),
            "residual_stall_ramp_steps": int(
                args.residual_stall_ramp_steps
            ),
        }

    runtime = {
        "kind": "manisoft_waypoint_sac_training",
        "learning_mode": (
            "frozen_base_residual_sac"
            if residual_config is not None
            else (
                "equilibrium_reference_residual_sac"
                if float(
                    environment_config.get("equilibrium_path_residual_scale", 0.0)
                )
                > 0
                else (
                    "anchored_online_sac"
                    if args.actor_anchor_coef > 0
                    else "pure_online_sac"
                )
            )
        ),
        "reference_actions_used_for_learning": bool(
            float(environment_config.get("equilibrium_path_prior_weight", 0.0))
            > 0
        ),
        "source_policy_anchor_coefficient": float(args.actor_anchor_coef),
        "zero_initialized_actor": bool(args.zero_init_actor),
        "source_policy_warmup": bool(args.source_policy_warmup),
        "actor_learning_delay_steps": int(args.actor_learning_delay_steps),
        "freeze_vec_normalize": bool(args.freeze_vec_normalize),
        "eval_waypoint_single_line_probability": (
            None
            if args.eval_waypoint_single_line_probability is None
            else float(args.eval_waypoint_single_line_probability)
        ),
        "eval_waypoint_hard_turn_probability": (
            None
            if args.eval_waypoint_hard_turn_probability is None
            else float(args.eval_waypoint_hard_turn_probability)
        ),
        "eval_waypoint_segment_count_range": args.eval_waypoint_segment_count_range,
        "eval_waypoint_maximum_turn_degrees": (
            None
            if args.eval_waypoint_maximum_turn_degrees is None
            else float(args.eval_waypoint_maximum_turn_degrees)
        ),
        "eval_waypoint_minimum_turn_degrees": (
            None
            if args.eval_waypoint_minimum_turn_degrees is None
            else float(args.eval_waypoint_minimum_turn_degrees)
        ),
        "eval_internal_waypoint_capture_radius": (
            None
            if args.eval_internal_waypoint_capture_radius is None
            else float(args.eval_internal_waypoint_capture_radius)
        ),
        "eval_waypoint_stall_steps": args.eval_waypoint_stall_steps,
        "eval_panel_seed": int(args.eval_panel_seed),
        "eval_num_envs": int(args.eval_num_envs),
        "scenario": str(scenario),
        "config_source": str(Path(args.config).expanduser().resolve()),
        "seed": args.seed,
        "device": args.device,
        "resume": args.resume,
        "frozen_base_residual": residual_config,
        "resolved": config,
    }
    (output / "run_config.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8"
    )

    from stable_baselines3 import SAC
    from antmaze_ac.rl.anchored_sac import AnchoredSAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import VecNormalize
    from antmaze_ac.rl.fixed_panel_eval import FixedPanelEvalCallback

    vector_env = _make_vector_env(
        scenario,
        environment_config,
        monitor_path,
        num_envs,
        args.seed,
        residual_config=residual_config,
    )
    vecnormalize_path = _resolved_vecnormalize_path(args, output)
    if vecnormalize_path is None:
        env = VecNormalize(vector_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        if not vecnormalize_path.is_file():
            raise FileNotFoundError(f"missing VecNormalize state: {vecnormalize_path}")
        env = VecNormalize.load(str(vecnormalize_path), vector_env)
        env.training = not bool(args.freeze_vec_normalize)
        env.norm_reward = False

    algorithm_class = AnchoredSAC if args.actor_anchor_coef > 0 else SAC
    if args.resume is None:
        model = algorithm_class(
            "MlpPolicy",
            env,
            learning_rate=float(sac_config["learning_rate"]),
            buffer_size=int(sac_config["buffer_size"]),
            learning_starts=int(sac_config["learning_starts"]),
            batch_size=int(sac_config["batch_size"]),
            tau=float(sac_config["tau"]),
            gamma=float(sac_config["gamma"]),
            train_freq=int(sac_config["train_freq"]),
            gradient_steps=int(sac_config["gradient_steps"]),
            ent_coef=sac_config.get("ent_coef", "auto"),
            target_entropy=sac_config.get("target_entropy", "auto"),
            policy_kwargs={"net_arch": list(sac_config["net_arch"])},
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
        if residual_config is not None or args.zero_init_actor:
            # tanh(0) is exactly the zero residual, making deterministic
            # deployment identical to the frozen base before any update.
            torch.nn.init.zeros_(model.actor.mu.weight)
            torch.nn.init.zeros_(model.actor.mu.bias)
        if isinstance(model, AnchoredSAC):
            model.enable_actor_anchor(float(args.actor_anchor_coef))
            if args.source_policy_warmup:
                model.enable_source_policy_warmup()
            if args.actor_learning_delay_steps > 0:
                model.delay_actor_updates_until(
                    int(model.learning_starts)
                    + int(args.actor_learning_delay_steps)
                )
    else:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"missing SAC checkpoint: {resume_path}")
        model = algorithm_class.load(
            str(resume_path), env=env, device=args.device, print_system_info=True
        )
        # Stable-Baselines3 restores the source checkpoint's seed during load.
        # Re-apply the requested transfer seed so independent curriculum runs
        # do not silently collect identical path and exploration sequences.
        model.seed = int(args.seed)
        model.set_random_seed(int(args.seed))
        learning_rate = float(sac_config["learning_rate"])
        model.learning_rate = learning_rate
        model.lr_schedule = lambda _: learning_rate
        # SAC.load restores optimization settings from the checkpoint. Apply
        # explicit current-stage settings so CPU/GPU scaling changes are not
        # silently ignored during curriculum transfer.
        model.batch_size = int(sac_config["batch_size"])
        model.gradient_steps = int(sac_config["gradient_steps"])
        model.gamma = float(sac_config["gamma"])
        model.tau = float(sac_config["tau"])
        model.train_freq = int(sac_config["train_freq"])
        model._convert_train_freq()
        optimizers = [model.actor.optimizer, model.critic.optimizer]
        if model.ent_coef_optimizer is not None:
            optimizers.append(model.ent_coef_optimizer)
        for optimizer in optimizers:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
        if args.replay_buffer is not None:
            replay_path = Path(args.replay_buffer).expanduser().resolve()
            if not replay_path.is_file():
                raise FileNotFoundError(f"missing replay buffer: {replay_path}")
            model.load_replay_buffer(str(replay_path))
        else:
            # A transferred policy with a fresh curriculum buffer must collect
            # enough transitions before gradient updates. ``num_timesteps`` is
            # retained for provenance/checkpoint numbering, so express the
            # warm-up threshold in that same global coordinate system.
            model.learning_starts = int(model.num_timesteps) + int(
                sac_config["learning_starts"]
            )
        # A curriculum transfer keeps global timestep provenance, but its
        # rollout summaries must not begin with the previous curriculum's last
        # 100 episodes. These buffers are diagnostic only and do not affect SAC
        # updates or the replay buffer.
        model.ep_info_buffer = None
        model.ep_success_buffer = None
        model._episode_num = 0
        if isinstance(model, AnchoredSAC):
            # Copy only after loading and before the first optimizer update.
            # The frozen actor is deliberately omitted from checkpoints; it is
            # a training constraint, not part of the deployable policy.
            model.enable_actor_anchor(float(args.actor_anchor_coef))
            if args.source_policy_warmup:
                model.enable_source_policy_warmup()
            if args.actor_learning_delay_steps > 0:
                model.delay_actor_updates_until(
                    int(model.learning_starts)
                    + int(args.actor_learning_delay_steps)
                )

    callbacks = []
    save_freq = max(int(training["checkpoint_freq"]) // num_envs, 1)
    callbacks.append(
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(output / "checkpoints"),
            name_prefix=f"sac_{environment_config['curriculum']}",
            save_replay_buffer=bool(training.get("save_replay_buffer", False)),
            save_vecnormalize=True,
            verbose=2,
        )
    )

    eval_env = None
    if int(training["eval_freq"]) > 0:
        eval_environment_config = dict(environment_config)
        # Entry reweighting is normally a training intervention. A certified
        # long path bank, however, currently covers one explicit entry, so its
        # acceptance panel must retain that required entry selection.
        if environment_config["curriculum"] != "table_long_waypoints":
            eval_environment_config.pop("entry_sampling_weights", None)
        if args.eval_waypoint_single_line_probability is not None:
            eval_environment_config["waypoint_single_line_probability"] = float(
                args.eval_waypoint_single_line_probability
            )
        if args.eval_waypoint_segment_count_range is not None:
            eval_environment_config["waypoint_segment_count_range"] = [
                int(value)
                for value in args.eval_waypoint_segment_count_range.split(",")
                if value.strip()
            ]
            # A training-only non-uniform distribution no longer matches the
            # overridden fixed evaluation count range.
            eval_environment_config.pop(
                "waypoint_segment_count_probabilities", None
            )
        if args.eval_waypoint_maximum_turn_degrees is not None:
            eval_environment_config["waypoint_maximum_turn_degrees"] = float(
                args.eval_waypoint_maximum_turn_degrees
            )
        if args.eval_waypoint_minimum_turn_degrees is not None:
            eval_environment_config["waypoint_minimum_turn_degrees"] = float(
                args.eval_waypoint_minimum_turn_degrees
            )
        if args.eval_waypoint_hard_turn_probability is not None:
            eval_environment_config["waypoint_hard_turn_probability"] = float(
                args.eval_waypoint_hard_turn_probability
            )
        if float(eval_environment_config["waypoint_minimum_turn_degrees"]) > float(
            eval_environment_config["waypoint_maximum_turn_degrees"]
        ):
            raise ValueError(
                "evaluation minimum waypoint turn cannot exceed the maximum"
            )
        if args.eval_internal_waypoint_capture_radius is not None:
            eval_environment_config["internal_waypoint_capture_radius"] = float(
                args.eval_internal_waypoint_capture_radius
            )
        if args.eval_waypoint_stall_steps is not None:
            eval_environment_config["waypoint_stall_steps"] = int(
                args.eval_waypoint_stall_steps
            )
        eval_episodes = int(training["eval_episodes"])
        eval_num_envs = min(int(args.eval_num_envs), eval_episodes)
        episodes_per_eval_env = (
            eval_episodes + eval_num_envs - 1
        ) // eval_num_envs
        eval_vector = _make_vector_env(
            scenario,
            eval_environment_config,
            output / "eval_monitor",
            eval_num_envs,
            args.seed + 100_000,
            fixed_panel_seed=args.eval_panel_seed,
            fixed_panel_seed_stride=episodes_per_eval_env,
            residual_config=residual_config,
        )
        eval_env = VecNormalize(
            eval_vector, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0
        )
        callbacks.append(
            FixedPanelEvalCallback(
                eval_env,
                best_model_save_path=str(output / "best"),
                log_path=str(output / "evaluation"),
                eval_freq=max(int(training["eval_freq"]) // num_envs, 1),
                n_eval_episodes=eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
            )
        )

    print(json.dumps(runtime, indent=2, sort_keys=True), flush=True)
    try:
        model.learn(
            total_timesteps=int(training["total_timesteps"]),
            callback=CallbackList(callbacks),
            reset_num_timesteps=args.resume is None,
            progress_bar=False,
        )
        model.save(str(output / "final_model"))
        replay_path = None
        if bool(training.get("save_replay_buffer", False)):
            replay_path = output / "replay_buffer.pkl"
            model.save_replay_buffer(str(replay_path))
        env.save(str(output / "vecnormalize.pkl"))
        completion = {
            **runtime,
            "status": "complete",
            "model_timesteps": int(model.num_timesteps),
            "final_model": str(output / "final_model.zip"),
            "replay_buffer": None if replay_path is None else str(replay_path),
            "vecnormalize": str(output / "vecnormalize.pkl"),
        }
        (output / "training_complete.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
        )
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
