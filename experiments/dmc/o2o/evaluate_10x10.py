"""Run the final deterministic 10-evaluation-seed x 10-episode protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.dmc.o2o.evaluate import (
    EVALUATION_KIND,
    _atomic_json,
    _device,
    validate_run_identity,
)
from experiments.dmc.o2o.learner import O2OLearner
from experiments.dmc.tasks.adapter import make_dmc_adapter
from experiments.dmc.tasks.registry import get_task_spec


FINAL_EVAL_KIND = "acmpc_dmc_o2o_final_evaluation_10x10_v1"
DEFAULT_SEED_BASE = 9_100_000
DEFAULT_NUM_SEEDS = 10
DEFAULT_EPISODES_PER_SEED = 10
DEFAULT_SEED_GROUP_STRIDE = 1_000


@torch.no_grad()
def evaluate_10x10(
    run_dir: Path,
    *,
    checkpoint_name: str = "latest",
    dataset_override: Path | None = None,
    koopman_override: Path | None = None,
    device_name: str = "cpu",
    seed_base: int = DEFAULT_SEED_BASE,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    episodes_per_seed: int = DEFAULT_EPISODES_PER_SEED,
    seed_group_stride: int = DEFAULT_SEED_GROUP_STRIDE,
) -> dict[str, Any]:
    if num_seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("num_seeds and episodes_per_seed must be positive")
    if seed_group_stride < episodes_per_seed:
        raise ValueError("seed_group_stride must be at least episodes_per_seed")
    validated = validate_run_identity(
        run_dir,
        checkpoint_name=checkpoint_name,
        dataset_override=dataset_override,
        koopman_override=koopman_override,
        load_artifacts=True,
    )
    device = _device(device_name)
    learner = O2OLearner(
        validated.config,
        validated.koopman,
        device,
        observation_normalizer=validated.observation_normalizer,
    )
    learner.load_state_dict(validated.checkpoint["learner"], restore_sampling_rng=False)
    expected_protocol = validated.checkpoint["environment_protocol"]
    action_dim = get_task_spec(validated.config.task).action_dim

    per_seed: list[dict[str, Any]] = []
    all_returns: list[float] = []
    all_lengths: list[int] = []
    for seed_index in range(num_seeds):
        seed_returns: list[float] = []
        seed_lengths: list[int] = []
        # Every episode has a unique reset seed while retaining a stable
        # ten-episode grouping for the reported evaluation seed.
        evaluation_seed = int(seed_base + seed_index * seed_group_stride)
        for episode_index in range(episodes_per_seed):
            reset_seed = int(evaluation_seed + episode_index)
            env = make_dmc_adapter(validated.config.task, seed=reset_seed)
            try:
                if env.protocol_metadata() != expected_protocol:
                    raise ValueError("Live DMC protocol differs from checkpoint")
                observation = env.reset(seed=reset_seed)
                episode_return = 0.0
                finished = False
                for step in range(int(env.step_limit)):
                    action = np.asarray(
                        learner.act(observation, deterministic=True)[0],
                        dtype=np.float32,
                    )
                    if action.shape != (action_dim,) or not np.isfinite(action).all():
                        raise RuntimeError(
                            f"Policy emitted an invalid {validated.config.task} action"
                        )
                    observation, reward, done, _info = env.step(action)
                    if not math.isfinite(float(reward)):
                        raise RuntimeError("DMC emitted a non-finite reward")
                    episode_return += float(reward)
                    if done:
                        seed_lengths.append(step + 1)
                        finished = True
                        break
                if not finished:
                    raise RuntimeError("DMC episode did not finish at its saved step limit")
            finally:
                env.close()
            seed_returns.append(episode_return)
            all_returns.append(episode_return)
            all_lengths.append(seed_lengths[-1])
        values = np.asarray(seed_returns, dtype=np.float64)
        per_seed.append(
            {
                "evaluation_seed": evaluation_seed,
                "episode_reset_seeds": [
                    int(evaluation_seed + i)
                    for i in range(episodes_per_seed)
                ],
                "returns": seed_returns,
                "episode_lengths": seed_lengths,
                "return_mean": float(values.mean()),
                "return_std_population": float(values.std(ddof=0)),
                "return_min": float(values.min()),
                "return_max": float(values.max()),
            }
        )

    values = np.asarray(all_returns, dtype=np.float64)
    return {
        "kind": FINAL_EVAL_KIND,
        "task": validated.config.task,
        "method": validated.config.method,
        "training_seed": validated.config.seed,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(validated.checkpoint_path),
        "checkpoint_sha256": validated.checkpoint_sha256,
        "checkpoint_phase": validated.checkpoint["phase"],
        "offline_update": int(validated.checkpoint["offline_update"]),
        "online_step": int(validated.checkpoint["online_step"]),
        "config_fingerprint": validated.config.fingerprint,
        "dataset": {"path": str(validated.dataset_path), "sha256": validated.dataset_sha256},
        "koopman": (
            {"path": str(validated.koopman_path), "sha256": validated.koopman_sha256}
            if validated.koopman_path is not None
            else None
        ),
        "raw_observation_normalizer": (
            validated.observation_normalizer.identity()
            if validated.observation_normalizer is not None
            else None
        ),
        "environment_protocol": expected_protocol,
        "initialization": validated.checkpoint.get("initialization"),
        "evaluation_protocol": {
            "deterministic": True,
            "evaluation_seeds": [
                int(seed_base + i * seed_group_stride) for i in range(num_seeds)
            ],
            "num_evaluation_seeds": num_seeds,
            "episodes_per_seed": episodes_per_seed,
            "total_episodes": num_seeds * episodes_per_seed,
            "seed_base": seed_base,
            "seed_group_stride": seed_group_stride,
            "device": str(device),
        },
        "per_evaluation_seed": per_seed,
        "returns": all_returns,
        "return_mean": float(values.mean()),
        "return_std_population": float(values.std(ddof=0)),
        "return_min": float(values.min()),
        "return_max": float(values.max()),
        "return_median": float(np.median(values)),
        "episode_lengths": all_lengths,
        "episode_length_mean": float(np.mean(all_lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("latest", "best"), default="latest")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--koopman", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--episodes-per-seed", type=int, default=DEFAULT_EPISODES_PER_SEED)
    parser.add_argument(
        "--seed-group-stride", type=int, default=DEFAULT_SEED_GROUP_STRIDE
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_10x10(
        args.run_dir,
        checkpoint_name=args.checkpoint,
        dataset_override=args.dataset,
        koopman_override=args.koopman,
        device_name=args.device,
        seed_base=args.seed_base,
        num_seeds=args.num_seeds,
        episodes_per_seed=args.episodes_per_seed,
        seed_group_stride=args.seed_group_stride,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
