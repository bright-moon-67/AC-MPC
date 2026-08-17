"""Deterministically evaluate an existing Playground PPO checkpoint."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from experiments.playground.jax_compat import install_single_device_brax_compatibility
from experiments.playground.structured_networks import (
    STRUCTURED_METHODS,
    make_structured_ppo_networks,
)
from experiments.playground.tasks import TASKS, load_task
from experiments.playground.train_ppo import _atomic_json, _json_value


METHODS = ("PPO", *STRUCTURED_METHODS)


def _latest_checkpoint(path: Path) -> Path:
    if (path / "_CHECKPOINT_METADATA").is_file():
        return path
    candidates = sorted(
        (child for child in path.iterdir() if child.is_dir() and child.name.isdigit()),
        key=lambda child: int(child.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No numeric checkpoint found under {path}")
    return candidates[-1]


def run(args: argparse.Namespace) -> dict:
    import jax
    import numpy as np
    from brax.training import acting
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks as ppo_networks
    from mujoco_playground._src import wrapper

    install_single_device_brax_compatibility()
    checkpoint_path = _latest_checkpoint(args.checkpoint.resolve())
    params = checkpoint.load(checkpoint_path)
    environment = load_task(args.task)
    task = TASKS[args.task]
    kmpc_horizon = args.kmpc_horizon or task.kmpc_horizon_steps
    if args.method == "PPO":
        networks = ppo_networks.make_ppo_networks(
            (TASKS[args.task].observation_dim,),
            TASKS[args.task].action_dim,
            preprocess_observations_fn=running_statistics.normalize,
        )
    else:
        if args.koopman is None:
            raise ValueError(f"{args.method} evaluation requires --koopman")
        networks = make_structured_ppo_networks(
            (TASKS[args.task].observation_dim,),
            TASKS[args.task].action_dim,
            running_statistics.normalize,
            method=args.method,
            koopman_path=str(args.koopman.resolve()),
            hidden_dim=128,
            ab_rank=4,
            kmpc_horizon=kmpc_horizon,
            kmpc_solver_iterations=20,
            critic_input=args.critic_input,
        )
    make_policy = ppo_networks.make_inference_fn(networks)
    eval_environment = wrapper.wrap_for_brax_training(
        environment,
        episode_length=TASKS[args.task].episode_steps,
        action_repeat=1,
    )
    evaluator = acting.Evaluator(
        eval_environment,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.episodes,
        episode_length=TASKS[args.task].episode_steps,
        action_repeat=1,
        key=jax.random.PRNGKey(args.seed),
    )
    started = time.time()
    raw_metrics = evaluator.run_evaluation(params, {}, aggregate_episodes=False)
    returns = np.asarray(raw_metrics["eval/episode_reward"], dtype=np.float64)
    if returns.shape != (args.episodes,) or not np.isfinite(returns).all():
        raise RuntimeError(f"Evaluation returned invalid scores with shape {returns.shape}")
    result = {
        "kind": "mujoco_playground_deterministic_evaluation_v1",
        "task": args.task,
        "method": args.method,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_path.name),
        "koopman": str(args.koopman.resolve()) if args.koopman else None,
        "seed": args.seed,
        "episodes": args.episodes,
        "episode_steps": TASKS[args.task].episode_steps,
        "returns": returns.tolist(),
        "return_mean": float(np.mean(returns)),
        "return_std_population": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "return_median": float(statistics.median(returns.tolist())),
        "wall_time_seconds": time.time() - started,
        "raw_metrics": _json_value(raw_metrics),
    }
    _atomic_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--koopman", type=Path)
    parser.add_argument("--kmpc-horizon", type=int)
    parser.add_argument(
        "--critic-input",
        choices=("raw_observation", "lifted_state"),
        default="raw_observation",
    )
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.kmpc_horizon is not None and args.kmpc_horizon < 1:
        parser.error("--kmpc-horizon must be positive")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
