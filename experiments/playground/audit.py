"""Training-free GPU smoke and throughput audit for Playground tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from experiments.playground.tasks import PLAYGROUND_COMMIT, TASKS, load_task


def _block(value: Any) -> Any:
    import jax

    return jax.tree.map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready")
        else leaf,
        value,
    )


def smoke_task(name: str, *, seed: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jp

    environment = load_task(name)
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    state = _block(reset(jax.random.PRNGKey(seed)))
    action = jp.zeros((environment.action_size,), dtype=jp.float32)
    state = _block(step(state, action))
    observation = state.obs
    if not bool(jp.all(jp.isfinite(observation))):
        raise RuntimeError(f"{name} emitted a non-finite observation")
    if not bool(jp.isfinite(state.reward)):
        raise RuntimeError(f"{name} emitted a non-finite reward")
    return {
        **TASKS[name].to_dict(),
        "reward_after_zero_action": float(state.reward),
        "observation_finite": True,
    }


def benchmark_task(
    name: str,
    *,
    seed: int,
    num_envs: int,
    steps: int,
) -> dict[str, Any]:
    if num_envs < 1 or steps < 1:
        raise ValueError("num_envs and steps must be positive")
    import jax
    import jax.numpy as jp

    environment = load_task(name)
    reset_many = jax.jit(jax.vmap(environment.reset))
    step_many = jax.jit(jax.vmap(environment.step))
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    state = _block(reset_many(keys))
    action = jp.zeros((num_envs, environment.action_size), dtype=jp.float32)
    state = _block(step_many(state, action))  # Compile outside the timer.
    start = time.perf_counter()
    for _ in range(steps):
        state = step_many(state, action)
    _block(state)
    elapsed = time.perf_counter() - start
    transitions = num_envs * steps
    return {
        "task": name,
        "num_envs": num_envs,
        "steps": steps,
        "transitions": transitions,
        "elapsed_seconds": elapsed,
        "transitions_per_second": transitions / elapsed,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import mujoco
    import mujoco_playground

    jax_devices = jax.devices()
    devices = [str(device) for device in jax_devices]
    platforms = [str(device.platform).lower() for device in jax_devices]
    if args.require_gpu and not any(platform in {"cuda", "gpu"} for platform in platforms):
        raise RuntimeError(f"A CUDA JAX device is required; found {devices}")
    smokes = [smoke_task(name, seed=args.seed) for name in TASKS]
    throughput = benchmark_task(
        args.benchmark_task,
        seed=args.seed,
        num_envs=args.num_envs,
        steps=args.steps,
    )
    return {
        "kind": "mujoco_playground_gpu_audit_v1",
        "playground_commit": PLAYGROUND_COMMIT,
        "playground_version": getattr(mujoco_playground, "__version__", None),
        "mujoco_version": mujoco.__version__,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "jax_devices": devices,
        "jax_device_platforms": platforms,
        "xla_preallocate": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "tasks": smokes,
        "throughput": throughput,
        "training_steps": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-task", choices=tuple(TASKS), default="CartpoleSwingup")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
