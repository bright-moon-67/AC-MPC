#!/usr/bin/env python
"""Collect MPC closed-loop trajectories for DAgger-style retraining.

The Koopman model is used inside a closed-loop MPC (exactly the same
optimizer as ``validate_koopman_mpc_reference.py``) to drive the soft robot
from the gravitational equilibrium towards randomly generated reference
equilibria.  Each control step records ``(state, action, next_state)`` in the
canonical 41-D/18-D episode format, so the resulting dataset matches the
state-action distribution the controller actually visits -- unlike the
broad-bandwidth random-excitation data used for the original model.

Saved episodes are consumed directly by ``build_manisoft_tip_sequences.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make both script directories importable without a package.
_AC_MPC_SCRIPTS = Path(__file__).resolve().parent
_MANISOFT_SCRIPTS = Path("/root/autodl-tmp/ManiSoft/scripts")
for _path in (_AC_MPC_SCRIPTS, _MANISOFT_SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from antmaze_ac.koopman.checkpoint import load_checkpoint
from validate_koopman_mpc_reference import (
    compress_physical_to_model,
    optimize_mpc_reference,
)
from collect_koopman_data import (
    compact_state,
    create_environment,
    has_ground_clearance,
    state_layout,
)
from collect_point_reference import sample_random_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Koopman best_validation.pt")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/dagger_mpc"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps-per-episode", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel collection processes.")
    parser.add_argument(
        "--success-stop-threshold",
        type=float,
        default=None,
        help=(
            "Stop the episode early when the tip gets within this many metres of "
            "the reference (collect 'near-success' trajectories). None disables it."
        ),
    )
    # Reference generation
    parser.add_argument("--ref-peak", type=float, default=0.25, help="Reference action peak.")
    parser.add_argument("--ref-settle-steps", type=int, default=300)
    # MPC closed-loop
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--state-weight", type=float, default=100.0)
    parser.add_argument("--action-weight", type=float, default=100.0)
    parser.add_argument("--control-weight", type=float, default=5.0)
    parser.add_argument("--smoothness-weight", type=float, default=10.0)
    parser.add_argument("--max-delta", type=float, default=0.002)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--optimizer-iterations", type=int, default=50)
    parser.add_argument("--optimizer-learning-rate", type=float, default=0.05)
    parser.add_argument("--settle-steps", type=int, default=50, help="Zero-activation settle before MPC.")
    parser.add_argument("--muscle-torque-scale", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-tip-height", type=float, default=0.15)
    return parser.parse_args()


def _torque_callback(muscle):
    def callback(element_lengths):
        return muscle.evaluate(element_lengths)
    return callback


def _clip(u: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(u, -limit, limit)


def generate_reference(
    env,
    configs: dict,
    muscle,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Settle a random action in a fresh env and return its 41-D reference state."""
    u_ref = sample_random_action(rng, args.ref_peak).reshape(-1)
    muscle.set_activation(u_ref.reshape(6, 3))
    torque = _torque_callback(muscle)
    previous_quaternion = None
    for _ in range(args.ref_settle_steps):
        env.step_with_torque_callback(torque)
        _, soft_state = env.get_state(has_image=False)
    ref_state = compact_state(
        soft_state, previous_quaternion=previous_quaternion
    )
    return ref_state.astype(np.float32), u_ref.astype(np.float32)


def collect_episode_worker(
    args: argparse.Namespace,
    worker_id: int,
    episode_ids: list[int],
    out_dir: Path,
) -> int:
    """Collect the given episodes in one independent process (one env at a time)."""
    from manisoft.muscle import SplineMuscle

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, payload = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device).freeze_dynamics()
    if model.action_dim != 18:
        raise ValueError(f"Expected 18-D action, got {model.action_dim}")
    model_physical_dim = model.state_dim - model.action_dim
    if model_physical_dim not in (11, 41):
        raise ValueError(f"Unsupported model physical dim {model_physical_dim}")
    stats = payload["normalizers"]["state"]
    state_mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
    state_std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
    action_low = torch.full((18,), -args.absolute_action_limit, dtype=torch.float32, device=device)
    action_high = torch.full((18,), args.absolute_action_limit, dtype=torch.float32, device=device)

    rng = np.random.default_rng(args.seed + worker_id * 100003)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for episode_id in episode_ids:
        # --- Reference environment (fresh, settle random action) ---
        ref_env, ref_configs = create_environment(args.config)
        ref_muscle = SplineMuscle(
            robot_length=float(ref_configs["softrobot"]["length"]),
            robot_num_elements=int(ref_configs["softrobot"]["num_elements"]),
            number_of_control_points=6,
            muscle_torque_scale=args.muscle_torque_scale,
        )
        ref_env.place()
        try:
            ref_state, u_ref = generate_reference(
                ref_env, ref_configs, ref_muscle, args, rng
            )
        finally:
            del ref_env, ref_muscle
            gc.collect()
        ref_model = compress_physical_to_model(ref_state[None, :], model_physical_dim)[0]
        ref_tip = ref_model[:3]

        # --- MPC environment (zero-activation settle -> gravity equilibrium) ---
        env, configs = create_environment(args.config)
        env.place()
        muscle = SplineMuscle(
            robot_length=float(configs["softrobot"]["length"]),
            robot_num_elements=int(configs["softrobot"]["num_elements"]),
            number_of_control_points=6,
            muscle_torque_scale=args.muscle_torque_scale,
        )
        muscle.set_activation(np.zeros((6, 3), dtype=np.float64))
        torque = _torque_callback(muscle)
        try:
            for _ in range(args.settle_steps):
                env.step_with_torque_callback(torque)
            _, soft_state = env.get_state(has_image=False)
            current_state = compact_state(soft_state).astype(np.float32)
            previous_action = np.zeros(18, dtype=np.float32)

            states = np.empty((args.steps_per_episode, 41), dtype=np.float32)
            actions = np.empty((args.steps_per_episode, 18), dtype=np.float32)
            next_states = np.empty((args.steps_per_episode, 41), dtype=np.float32)
            warm_start = None
            valid = True
            executed = 0

            for step in range(args.steps_per_episode):
                tip11 = compress_physical_to_model(
                    current_state[None, :], model_physical_dim
                )[0]
                dynamics_state = np.concatenate((tip11, previous_action)).astype(np.float32)
                plan = optimize_mpc_reference(
                    model=model,
                    state=dynamics_state,
                    ref_state=ref_model,
                    ref_action=u_ref,
                    state_mean=state_mean,
                    state_std=state_std,
                    action_low=action_low,
                    action_high=action_high,
                    horizon=args.horizon,
                    state_weight=args.state_weight,
                    action_weight=args.action_weight,
                    control_weight=args.control_weight,
                    smoothness_weight=args.smoothness_weight,
                    track_tip_only=False,
                    action_tracking=True,
                    max_delta=args.max_delta,
                    iterations=args.optimizer_iterations,
                    learning_rate=args.optimizer_learning_rate,
                    initial_decision=warm_start,
                )
                requested_delta = plan["requested_deltas"][0].cpu().numpy()
                next_action = _clip(previous_action + requested_delta, args.absolute_action_limit)
                muscle.set_activation(next_action.reshape(6, 3))

                states[step] = current_state
                actions[step] = next_action
                env.step_with_torque_callback(torque)
                _, soft_state = env.get_state(has_image=False)
                if not has_ground_clearance(soft_state, args.min_tip_height):
                    valid = False
                    break
                current_state = compact_state(
                    soft_state, previous_quaternion=current_state[-4:]
                ).astype(np.float32)
                next_states[step] = current_state
                previous_action = next_action
                executed += 1
                warm_start = torch.cat(
                    (plan["decision"][1:], torch.zeros_like(plan["decision"][:1])), dim=0
                )
                if (
                    args.success_stop_threshold is not None
                    and executed >= 10
                ):
                    tip_now = compress_physical_to_model(
                        current_state[None, :], model_physical_dim
                    )[0][:3]
                    if np.linalg.norm(tip_now - ref_tip) <= args.success_stop_threshold:
                        break

            if not valid:
                continue
            if not np.isfinite(states[:executed]).all() or not np.isfinite(next_states[:executed]).all():
                raise ValueError(f"worker {worker_id} episode {episode_id} contains NaN or Inf")

            episode_path = out_dir / f"episode_{saved:04d}.npz"
            np.savez_compressed(
                episode_path,
                state=states[:executed],
                action=actions[:executed],
                next_state=next_states[:executed],
            )
            saved += 1
            print(
                f"[w{worker_id} ep{episode_id}] {episode_path.name} transitions={executed}",
                flush=True,
            )
        finally:
            del env, muscle
            gc.collect()
    return saved


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.steps_per_episode < 1:
        raise ValueError("episodes and steps-per-episode must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.success_stop_threshold is not None and args.success_stop_threshold <= 0:
        raise ValueError("--success-stop-threshold must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("episode_*.npz"))
    existing += list(args.output_dir.glob("w*/episode_*.npz"))
    if existing:
        raise FileExistsError(
            f"{args.output_dir} already has episodes; use a fresh --output-dir or clean it first"
        )

    episode_ids = list(range(args.episodes))
    if args.workers == 1:
        total = collect_episode_worker(args, 0, episode_ids, args.output_dir)
    else:
        import multiprocessing as mp

        chunks = [episode_ids[i::args.workers] for i in range(args.workers)]
        with mp.Pool(args.workers) as pool:
            results = pool.starmap(
                collect_episode_worker,
                [
                    (args, i, chunks[i], args.output_dir / f"w{i}")
                    for i in range(args.workers)
                ],
            )
        total = int(sum(results))

    metadata = {
        "schema_version": 2,
        "transition_fields": ["state", "action", "next_state"],
        "state_dim": 41,
        "state_layout": state_layout(),
        "action_dim": 18,
        "collection": "dagger_mpc_closed_loop",
        "episodes": total,
        "steps_per_episode": args.steps_per_episode,
        "seed": args.seed,
        "workers": args.workers,
        "ref_peak": args.ref_peak,
        "success_stop_threshold": args.success_stop_threshold,
        "mpc": {
            "horizon": args.horizon,
            "state_weight": args.state_weight,
            "action_weight": args.action_weight,
            "max_delta": args.max_delta,
        },
        "checkpoint": str(args.checkpoint),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Done: {total} episodes -> {args.output_dir}")


if __name__ == "__main__":
    main()
