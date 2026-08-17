"""Collect PPO and optional exploration trajectories for Koopman training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from experiments.playground.tasks import PLAYGROUND_COMMIT, TASKS, load_task
from experiments.playground.train_ppo import _atomic_json


STAGES = ("early", "mid", "late")
BEHAVIORS = ("ppo", "uniform_iid", "uniform_held")


def _behavior_fractions(
    ppo_fraction: float,
    uniform_iid_fraction: float,
    uniform_held_fraction: float,
) -> np.ndarray:
    fractions = np.asarray(
        (ppo_fraction, uniform_iid_fraction, uniform_held_fraction),
        dtype=np.float64,
    )
    if not np.isfinite(fractions).all() or np.any(fractions < 0):
        raise ValueError("Behavior fractions must be finite and non-negative")
    if not np.isclose(float(np.sum(fractions)), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("Behavior fractions must sum to one")
    return fractions


def _allocate_behavior_modes(
    num_envs: int, fractions: np.ndarray, seed: int
) -> np.ndarray:
    """Allocate exact near-proportional behavior counts, then shuffle episodes."""

    if num_envs < 1 or fractions.shape != (len(BEHAVIORS),):
        raise ValueError("Invalid behavior allocation inputs")
    expected = fractions * num_envs
    counts = np.floor(expected).astype(np.int64)
    remainder = num_envs - int(np.sum(counts))
    if remainder:
        priorities = np.argsort(-(expected - counts), kind="stable")
        counts[priorities[:remainder]] += 1
    modes = np.repeat(np.arange(len(BEHAVIORS), dtype=np.int32), counts)
    np.random.default_rng(seed).shuffle(modes)
    return modes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks as ppo_networks

    if len(args.checkpoint) != 3:
        raise ValueError("Exactly three --checkpoint values are required")
    checkpoints = [path.resolve() for path in args.checkpoint]
    steps = [int(path.name) for path in checkpoints]
    if steps != sorted(steps) or len(set(steps)) != 3:
        raise ValueError("Checkpoints must be distinct and ordered early to late")
    if args.num_envs < 1 or args.episode_steps != TASKS[args.task].episode_steps:
        raise ValueError(
            "Collector requires positive envs and the official episode length"
        )
    if args.action_hold_steps < 1:
        raise ValueError("--action-hold-steps must be positive")
    behavior_fractions = _behavior_fractions(
        args.ppo_fraction,
        args.uniform_iid_fraction,
        args.uniform_held_fraction,
    )

    environment = load_task(args.task)
    reset_many = jax.vmap(environment.reset)
    step_many = jax.vmap(environment.step)
    networks = ppo_networks.make_ppo_networks(
        (TASKS[args.task].observation_dim,),
        TASKS[args.task].action_dim,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(networks)

    @jax.jit
    def collect_stage(
        params: Any, reset_key: Any, action_key: Any, behavior_mode: Any
    ):
        state = reset_many(jax.random.split(reset_key, args.num_envs))
        policy = make_policy(params, deterministic=False)
        behavior_mode = behavior_mode[:, None]

        def one_step(carry: Any, step_index: Any):
            current, key, held_action = carry
            key, sample_key, exploration_key = jax.random.split(key, 3)
            ppo_action, _extras = policy(current.obs, sample_key)
            uniform_action = jax.random.uniform(
                exploration_key,
                ppo_action.shape,
                minval=-1.0,
                maxval=1.0,
            )
            refresh = step_index % args.action_hold_steps == 0
            held_action = jp.where(refresh, uniform_action, held_action)
            action = jp.where(
                behavior_mode == 0,
                ppo_action,
                jp.where(behavior_mode == 1, uniform_action, held_action),
            )
            following = step_many(current, action)
            output = (
                current.obs,
                action,
                following.obs,
                following.reward,
                following.done,
            )
            return (following, key, held_action), output

        initial_action = jp.zeros(
            (args.num_envs, TASKS[args.task].action_dim), dtype=state.obs.dtype
        )
        (_final_state, _final_key, _final_held), trajectory = jax.lax.scan(
            one_step,
            (state, action_key, initial_action),
            jp.arange(args.episode_steps),
        )
        return trajectory

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": "mujoco_playground_koopman_collection_v1",
        "task": args.task,
        "playground_commit": PLAYGROUND_COMMIT,
        "seed": args.seed,
        "num_envs_per_stage": args.num_envs,
        "episode_steps": args.episode_steps,
        "transitions_per_stage": args.num_envs * args.episode_steps,
        "total_transitions": 3 * args.num_envs * args.episode_steps,
        "policy": "episode_level_behavior_mixture",
        "behaviors": {
            name: float(fraction)
            for name, fraction in zip(BEHAVIORS, behavior_fractions, strict=True)
        },
        "uniform_held_action_steps": args.action_hold_steps,
        "stages": {},
        "started_unix_seconds": time.time(),
    }
    stages_and_checkpoints = zip(STAGES, checkpoints, strict=True)
    for index, (stage, checkpoint_path) in enumerate(stages_and_checkpoints):
        params = checkpoint.load(checkpoint_path)
        behavior_mode = _allocate_behavior_modes(
            args.num_envs,
            behavior_fractions,
            args.seed + 1000 + index,
        )
        trajectory = collect_stage(
            params,
            jax.random.PRNGKey(args.seed + 2 * index),
            jax.random.PRNGKey(args.seed + 2 * index + 1),
            jp.asarray(behavior_mode),
        )
        observation, action, next_observation, reward, done = jax.tree.map(
            np.asarray, trajectory
        )
        # Store contiguous complete episodes: [episode, time, feature].
        states = np.concatenate(
            (
                np.swapaxes(observation, 0, 1),
                np.swapaxes(next_observation[-1:], 0, 1),
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        actions = np.swapaxes(action, 0, 1).astype(np.float32, copy=False)
        rewards = np.swapaxes(reward, 0, 1).astype(np.float32, copy=False)
        dones = np.swapaxes(done, 0, 1).astype(np.float32, copy=False)
        if not all(np.isfinite(value).all() for value in (states, actions, rewards)):
            raise FloatingPointError(f"{stage} collection contains NaN or Inf")
        output_path = args.output_dir / f"{stage}.npz"
        _atomic_npz(
            output_path,
            {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "behavior_mode": behavior_mode,
                "stage": np.asarray(stage),
                "checkpoint_step": np.asarray(
                    int(checkpoint_path.name), dtype=np.int64
                ),
            },
        )
        manifest["stages"][stage] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint_path.name),
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "episodes": args.num_envs,
            "transitions": args.num_envs * args.episode_steps,
            "reward_mean": float(np.mean(rewards)),
            "done_count": int(np.sum(dones)),
            "behavior_episode_counts": {
                name: int(np.sum(behavior_mode == mode))
                for mode, name in enumerate(BEHAVIORS)
            },
            "behavior_reward_means": {
                name: float(np.mean(rewards[behavior_mode == mode]))
                for mode, name in enumerate(BEHAVIORS)
                if np.any(behavior_mode == mode)
            },
        }
        print(
            f"stage={stage} transitions={args.num_envs * args.episode_steps} "
            f"reward_mean={np.mean(rewards):.6g}",
            flush=True,
        )
    manifest["finished_unix_seconds"] = time.time()
    manifest["wall_time_seconds"] = (
        manifest["finished_unix_seconds"] - manifest["started_unix_seconds"]
    )
    _atomic_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1000)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--ppo-fraction", type=float, default=1.0)
    parser.add_argument("--uniform-iid-fraction", type=float, default=0.0)
    parser.add_argument("--uniform-held-fraction", type=float, default=0.0)
    parser.add_argument("--action-hold-steps", type=int, default=10)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
