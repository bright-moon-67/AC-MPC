"""Train AB-PQ, KMPC, or AC-MPC-MPVE with GPU-native Playground PPO."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from experiments.playground.jax_compat import install_single_device_brax_compatibility
from experiments.playground.mpve import make_mpve_inference_fn, make_mpve_ppo_loss
from experiments.playground.structured_networks import (
    STRUCTURED_METHODS,
    make_structured_ppo_networks,
)
from experiments.playground.tasks import PLAYGROUND_COMMIT, TASKS, load_task
from experiments.playground.train_ppo import (
    _atomic_json,
    _json_value,
    official_ppo_config,
    smoke_ppo_config,
)


def run(args: argparse.Namespace) -> None:
    import jax

    compatibility_shim = install_single_device_brax_compatibility()
    from brax.training.agents.ppo import losses as ppo_losses
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground._src import wrapper

    if not any(device.platform in {"cuda", "gpu"} for device in jax.devices()):
        raise RuntimeError(f"CUDA JAX device required, found {jax.devices()}")
    koopman_path = args.koopman.resolve()
    if not koopman_path.is_file():
        raise FileNotFoundError(koopman_path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    config = smoke_ppo_config(args.task) if args.smoke else official_ppo_config(args.task)
    if args.timesteps is not None:
        if args.timesteps < 1:
            raise ValueError("--timesteps must be positive")
        config["num_timesteps"] = args.timesteps
    config["seed"] = args.seed
    config["max_devices_per_host"] = 1
    config["deterministic_eval"] = True
    task = TASKS[args.task]
    kmpc_horizon = args.kmpc_horizon or task.kmpc_horizon_steps
    mpve_horizon = args.mpve_horizon or task.mpve_horizon_steps
    if mpve_horizon > kmpc_horizon:
        raise ValueError("MPVE horizon cannot exceed the KMPC planning horizon")
    network_factory = functools.partial(
        make_structured_ppo_networks,
        method=args.method,
        koopman_path=str(koopman_path),
        hidden_dim=128,
        ab_rank=4,
        kmpc_horizon=kmpc_horizon,
        kmpc_solver_iterations=20,
        critic_input=args.critic_input,
    )
    if args.method == "AC-MPC-MPVE":
        original_loss = ppo_losses.compute_ppo_loss
        original_inference = ppo_networks.make_inference_fn
        ppo_losses.compute_ppo_loss = make_mpve_ppo_loss(
            original_loss,
            koopman_path=str(koopman_path),
            action_size=TASKS[args.task].action_dim,
            horizon=mpve_horizon,
            coefficient=1.0,
            reward_source=task.mpve_reward_source,
        )
        ppo_networks.make_inference_fn = make_mpve_inference_fn(
            original_inference,
            koopman_path=str(koopman_path),
            action_size=TASKS[args.task].action_dim,
            horizon=mpve_horizon,
        )
    metadata = {
        "kind": "mujoco_playground_structured_ppo_run_v1",
        "task": args.task,
        "method": args.method,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "playground_commit": PLAYGROUND_COMMIT,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "jax_devices": [str(device) for device in jax.devices()],
        "single_device_brax_compatibility_shim": compatibility_shim,
        "environment": TASKS[args.task].to_dict(),
        "koopman_export": str(koopman_path),
        "controller": {
            "hidden_dim": 128,
            "ab_rank": 4,
            "kmpc_horizon": kmpc_horizon,
            "kmpc_horizon_seconds": kmpc_horizon * task.control_timestep,
            "kmpc_solver_iterations": 20,
        },
        "mpve": {
            "enabled": args.method == "AC-MPC-MPVE",
            "horizon": mpve_horizon if args.method == "AC-MPC-MPVE" else None,
            "horizon_seconds": (
                mpve_horizon * task.control_timestep
                if args.method == "AC-MPC-MPVE"
                else None
            ),
            "coefficient": 1.0 if args.method == "AC-MPC-MPVE" else None,
            "detached_targets": args.method == "AC-MPC-MPVE",
            "reward_source": (
                task.mpve_reward_source
                if args.method == "AC-MPC-MPVE"
                else None
            ),
        },
        "critic_input": (
            "frozen_koopman_normalized_lifted_state"
            if args.critic_input == "lifted_state"
            else "running_standardized_raw_observation"
        ),
        "ppo": _json_value(config),
        "started_unix_seconds": time.time(),
    }
    _atomic_json(output / "run.json", metadata)

    def progress(step: int, metrics: dict[str, Any]) -> None:
        row = {
            "step": int(step),
            "wall_time_unix_seconds": time.time(),
            **_json_value(metrics),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(
            f"method={args.method} step={step} "
            f"eval_return={row.get('eval/episode_reward')}",
            flush=True,
        )

    environment = load_task(args.task)
    eval_environment = load_task(args.task)
    ppo.train(
        environment=environment,
        eval_env=eval_environment,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        save_checkpoint_path=str(output / "checkpoints"),
        **config,
    )
    metadata["finished_unix_seconds"] = time.time()
    metadata["completed"] = True
    _atomic_json(output / "run.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--method", choices=STRUCTURED_METHODS, required=True)
    parser.add_argument("--koopman", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--kmpc-horizon", type=int)
    parser.add_argument("--mpve-horizon", type=int)
    parser.add_argument(
        "--critic-input",
        choices=("raw_observation", "lifted_state"),
        default="raw_observation",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.kmpc_horizon is not None and args.kmpc_horizon < 1:
        parser.error("--kmpc-horizon must be positive")
    if args.mpve_horizon is not None and args.mpve_horizon < 1:
        parser.error("--mpve-horizon must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
