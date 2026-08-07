"""Evaluate PPO-trained PandaReach3 checkpoints with the batched closed-loop eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.state_only_feasibility.train_pandareach_threewaypoint_bc import (
    BCConfig,
    closed_loop_evaluation,
    load_koopman,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_ppo import (
    _build_actor,
)


def evaluate_ppo_checkpoint(
    checkpoint_path: Path,
    evaluation_episodes: int,
    evaluation_num_envs: int,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_name = str(payload["actor_name"])
    actor_config = BCConfig(**payload["actor_config"])
    koopman, koopman_payload = load_koopman(
        Path(payload["koopman_path"]), device
    )
    actor = _build_actor(actor_name, koopman, actor_config, device)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()

    def _arr(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    center = _arr(koopman_payload["normalizer"]["center"])
    scale = _arr(koopman_payload["normalizer"]["scale"])
    context_center = _arr(payload["context_center"])
    context_scale = _arr(payload["context_scale"])
    config = BCConfig(
        evaluation_episodes=evaluation_episodes,
        evaluation_num_envs=evaluation_num_envs,
    )
    result = closed_loop_evaluation(
        actor_name,
        actor,
        koopman,
        center,
        scale,
        context_center,
        context_scale,
        config,
        device,
    )
    result.update(
        {
            "checkpoint": str(checkpoint_path),
            "global_step": int(payload["global_step"]),
        }
    )
    print(json.dumps({actor_name: result}, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-num-envs", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_ppo_checkpoint(
        args.checkpoint,
        args.evaluation_episodes,
        args.evaluation_num_envs,
        args.device,
    )


if __name__ == "__main__":
    main()
