#!/usr/bin/env python
"""Evaluate a unified teacher-residual SAC policy from the upright state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from train_manisoft_teacher_tracking_sac import _evaluate_upright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--teacher-episode", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20280865)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in (
            ("scenario", args.scenario),
            ("task_config", args.task_config),
            ("teacher_episode", args.teacher_episode),
            ("config", args.config),
            ("model", args.model),
            ("vecnormalize", args.vecnormalize),
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    from stable_baselines3 import SAC
    from stable_baselines3.common.save_util import load_from_pkl

    model = SAC.load(paths["model"], device=args.device, print_system_info=False)
    normalizer = load_from_pkl(paths["vecnormalize"])
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    result = _evaluate_upright(
        model,
        normalizer,
        scenario=paths["scenario"],
        task_config=paths["task_config"],
        teacher_episode=paths["teacher_episode"],
        environment=dict(payload["environment"]),
        seed=args.seed,
    )
    result.update(
        {
            "model": str(paths["model"]),
            "vecnormalize": str(paths["vecnormalize"]),
            "seed": int(args.seed),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
