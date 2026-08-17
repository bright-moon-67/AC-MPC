"""Benchmark DMC vector stepping without running an optimizer."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from experiments.dmc.ppo.vector_env import make_dmc_vector_env


def benchmark(task: str, num_envs: int, workers: int, steps: int) -> dict[str, object]:
    started = time.perf_counter()
    env = make_dmc_vector_env(task, num_envs, 9182, workers=workers)
    construction_seconds = time.perf_counter() - started
    rng = np.random.default_rng(9183)
    try:
        reset_started = time.perf_counter()
        observation = env.reset()
        reset_seconds = time.perf_counter() - reset_started
        step_started = time.perf_counter()
        for _ in range(steps):
            action = rng.uniform(
                env.action_low,
                env.action_high,
                size=(num_envs, env.action_dim),
            ).astype(np.float32)
            transition = env.step(action)
        step_seconds = time.perf_counter() - step_started
        if not np.isfinite(observation).all() or not np.isfinite(
            transition.observation
        ).all():
            raise FloatingPointError("Vector benchmark produced non-finite states")
        return {
            "task": task,
            "num_envs": num_envs,
            "workers": workers,
            "steps_per_env": steps,
            "transitions": num_envs * steps,
            "construction_seconds": construction_seconds,
            "reset_seconds": reset_seconds,
            "step_seconds": step_seconds,
            "transitions_per_second": num_envs * steps / step_seconds,
            "protocol": env.protocol,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="cartpole_swingup")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16, 32])
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    reports = [
        benchmark(args.task, args.num_envs, workers, args.steps)
        for workers in args.workers
    ]
    protocol = reports[0]["protocol"]
    if any(report["protocol"] != protocol for report in reports[1:]):
        raise RuntimeError("Worker counts changed the DMC protocol")
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
