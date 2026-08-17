"""Train standard PPO on official MuJoCo Playground DMC environments.

The default path uses Playground's tuned Brax PPO configuration without
copying its hyperparameters into this repository.  ``--smoke`` is a small,
explicitly non-benchmark configuration used only to validate compilation,
metrics, and checkpoint output.
"""

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

from experiments.playground.tasks import PLAYGROUND_COMMIT, TASKS, load_task
from experiments.playground.jax_compat import install_single_device_brax_compatibility


def _json_value(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def official_ppo_config(task: str) -> dict[str, Any]:
    from mujoco_playground.config import dm_control_suite_params

    return dict(dm_control_suite_params.brax_ppo_config(task))


def smoke_ppo_config(task: str) -> dict[str, Any]:
    config = official_ppo_config(task)
    config.update(
        num_timesteps=32_768,
        num_evals=2,
        num_envs=256,
        num_eval_envs=10,
        num_resets_per_eval=0,
        unroll_length=8,
        batch_size=32,
        num_minibatches=8,
        num_updates_per_batch=1,
    )
    return config


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    import jax
    compatibility_shim = install_single_device_brax_compatibility()
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground._src import wrapper

    if not any(device.platform in {"cuda", "gpu"} for device in jax.devices()):
        raise RuntimeError(f"CUDA JAX device required, found {jax.devices()}")
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
    # Formal comparison evaluates deterministic policy modes.  Playground's
    # tuned learner settings remain unchanged.
    config["deterministic_eval"] = True

    metadata = {
        "kind": "mujoco_playground_ppo_run_v1",
        "task": args.task,
        "method": "PPO",
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "playground_commit": PLAYGROUND_COMMIT,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "jax_devices": [str(device) for device in jax.devices()],
        "single_device_brax_compatibility_shim": compatibility_shim,
        "environment": TASKS[args.task].to_dict(),
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
        reward = row.get("eval/episode_reward")
        print(f"step={step} eval_return={reward} metrics={len(row) - 2}", flush=True)

    environment = load_task(args.task)
    eval_environment = load_task(args.task)
    checkpoint_dir = output / "checkpoints"
    ppo.train(
        environment=environment,
        eval_env=eval_environment,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        save_checkpoint_path=str(checkpoint_dir),
        **config,
    )
    metadata["finished_unix_seconds"] = time.time()
    metadata["completed"] = True
    _atomic_json(output / "run.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
