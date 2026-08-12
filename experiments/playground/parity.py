"""Numerical Cartpole audit against the original dm_control environment.

MuJoCo Playground uses float32 GPU dynamics, so bitwise identity is neither
expected nor required.  This audit starts both implementations from identical
qpos/qvel, applies identical actions, and records short-horizon observation
and reward drift without performing any training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _dmc_observation(time_step: Any) -> np.ndarray:
    observation = time_step.observation
    return np.concatenate(
        [
            np.asarray(observation["position"]).reshape(-1),
            np.asarray(observation["velocity"]).reshape(-1),
        ]
    ).astype(np.float64)


def run_parity(*, seed: int, trajectories: int, steps: int) -> dict[str, Any]:
    if trajectories < 1 or steps < 1:
        raise ValueError("trajectories and steps must be positive")
    import jax
    import jax.numpy as jp
    from dm_control import suite
    from mujoco import mjx
    from mujoco_playground._src import mjx_env

    from experiments.playground.tasks import load_task

    dmc = suite.load("cartpole", "swingup")
    playground = load_task("CartpoleSwingup")
    playground_step = jax.jit(playground.step)
    rng = np.random.default_rng(seed)
    observation_errors: list[float] = []
    reward_errors: list[float] = []
    initial_observation_errors: list[float] = []

    for trajectory in range(trajectories):
        # Match the official Swingup reset distribution.  Wider arbitrary
        # states can magnify float32/solver drift but are not the training
        # protocol being compared here.
        qpos = np.asarray(
            [0.01 * rng.normal(), np.pi + 0.01 * rng.normal()],
            dtype=np.float64,
        )
        qvel = (0.01 * rng.normal(size=2)).astype(np.float64)
        actions = rng.uniform(-1.0, 1.0, size=(steps, 1)).astype(np.float32)

        dmc_time_step = dmc.reset()
        del dmc_time_step
        dmc.physics.data.qpos[:] = qpos
        dmc.physics.data.qvel[:] = qvel
        dmc.physics.forward()

        pg_state = playground.reset(jax.random.PRNGKey(seed + trajectory))
        pg_data = mjx_env.make_data(
            playground.mj_model,
            qpos=jp.asarray(qpos, dtype=jp.float32),
            qvel=jp.asarray(qvel, dtype=jp.float32),
            impl=playground.mjx_model.impl.value,
            naconmax=playground._config.naconmax,
            njmax=playground._config.njmax,
        )
        pg_data = mjx.forward(playground.mjx_model, pg_data)
        pg_obs = playground._get_obs(pg_data, pg_state.info)
        pg_state = pg_state.replace(data=pg_data, obs=pg_obs)

        dmc_initial = np.concatenate(
            [
                qpos[:1],
                [np.cos(qpos[1]), np.sin(qpos[1])],
                qvel,
            ]
        )
        initial_observation_errors.append(
            float(np.max(np.abs(np.asarray(pg_obs) - dmc_initial)))
        )

        for action in actions:
            dmc_time_step = dmc.step(action)
            pg_state = playground_step(pg_state, jp.asarray(action))
            pg_state.reward.block_until_ready()
            observation_errors.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(pg_state.obs, dtype=np.float64)
                            - _dmc_observation(dmc_time_step)
                        )
                    )
                )
            )
            reward_errors.append(
                abs(float(pg_state.reward) - float(dmc_time_step.reward))
            )

    return {
        "kind": "cartpole_dmc_playground_parity_v1",
        "seed": seed,
        "trajectories": trajectories,
        "steps_per_trajectory": steps,
        "samples": trajectories * steps,
        "initial_observation_max_abs_error": max(initial_observation_errors),
        "observation_max_abs_error": max(observation_errors),
        "observation_mean_abs_max_error": float(np.mean(observation_errors)),
        "reward_max_abs_error": max(reward_errors),
        "reward_mean_abs_error": float(np.mean(reward_errors)),
        "comparison": "same_qpos_qvel_same_action_short_horizon",
        "initial_state_distribution": "official_cartpole_swingup_reset",
        "expected_precision": "float64_cpu_dmc_vs_float32_gpu_warp",
        "training_steps": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    result = run_parity(
        seed=args.seed, trajectories=args.trajectories, steps=args.steps
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
