"""Multi-seed closed-loop evaluation of a saved HopperHop actor checkpoint.

Loads an actor from either:
  * a PPO-actors ``latest.pt`` (``train_hopper_hop_ppo_actors.py``), or
  * a BC ``{actor}.pt`` (``train_hopper_hop_bc.py``),
and runs the vectorized closed-loop evaluation in MS-HopperHop over several
evaluation seeds, reporting per-seed and aggregate (mean +/- std) metrics so
the comparison is robust to episode randomness.

Example:
  python -m experiments.hopper_hop.eval_hopperhop_multiseed \
      --actor BC-KMPC \
      --checkpoint runs/hopper_hop/ppo_fair/BC-KMPC/latest.pt \
      --koopman runs/hopper_hop/koopman_v2/best.pt \
      --eval-seeds 10 --episodes 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.hopper_hop.train_hopper_hop_bc import (
    BC_ACTOR_ORDER,
    BCConfig,
    closed_loop_evaluation,
    load_koopman,
    _make_builders,
)
from experiments.hopper_hop.train_hopper_hop_ppo import Actor as PPOBaselineActor


def _load_actor(
    actor_name: str,
    checkpoint_path: Path,
    koopman_path: Path,
    device: torch.device,
):
    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    # PPO-actors latest.pt: keys actor_state / actor_name / center / scale
    if "actor_state" in payload and "actor_name" in payload:
        if payload["actor_name"] != actor_name:
            raise ValueError(
                f"Checkpoint actor {payload['actor_name']!r} != requested {actor_name!r}"
            )
        center = np.asarray(payload["center"].cpu(), dtype=np.float32)
        scale = np.asarray(payload["scale"].cpu(), dtype=np.float32)
        source = "ppo"
    # BC {actor}.pt: keys actor_state / name / normalizer
    elif "actor_state" in payload and payload.get("kind") == "hopperhop_bc_actor":
        if payload.get("name") != actor_name:
            raise ValueError(f"Checkpoint actor name mismatch")
        normalizer = payload["normalizer"]
        center = np.asarray(normalizer["state_center"], dtype=np.float32)
        scale = np.asarray(normalizer["state_scale"], dtype=np.float32)
        source = "bc"
    else:
        raise ValueError(
            f"Unrecognized checkpoint format: {checkpoint_path}"
        )

    koopman, _ = load_koopman(koopman_path, device)
    actor_config = BCConfig(seed=0)
    builders = _make_builders(koopman, actor_config, device)
    if actor_name == "PPO":
        builders["PPO"] = lambda: PPOBaselineActor(koopman.state_dim, 4)
    actor = builders[actor_name]().to(device)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()
    return actor, koopman, center, scale, source


def evaluate_multiseed(
    actor_name: str,
    checkpoint_path: Path,
    koopman_path: Path,
    *,
    eval_seeds: int = 10,
    episodes: int = 64,
    seed_start: int = 20_270_804,
    device_name: str = "auto",
) -> dict[str, object]:
    if actor_name not in BC_ACTOR_ORDER:
        raise ValueError(f"Unknown actor {actor_name!r}")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    actor, koopman, center, scale, source = _load_actor(
        actor_name, checkpoint_path, koopman_path, device
    )
    config = BCConfig(
        evaluation_episodes=episodes,
        evaluation_num_envs=episodes,
    )
    per_seed: list[dict[str, float]] = []
    for index in range(eval_seeds):
        config = BCConfig(
            evaluation_episodes=episodes,
            evaluation_num_envs=episodes,
            evaluation_seed_start=seed_start + index * 1000,
        )
        result = closed_loop_evaluation(
            actor_name, actor, koopman, center, scale, config, device
        )
        per_seed.append(
            {
                "seed": seed_start + index * 1000,
                "mean_return": float(result["mean_return"]),
                "mean_standing": float(result["mean_standing"]),
                "mean_hopping": float(result["mean_hopping"]),
                "mean_step_reward": float(result["mean_step_reward"]),
                "action_bound_fraction": float(result["action_bound_fraction"]),
            }
        )
        print(
            f"  eval_seed={seed_start + index * 1000}: "
            f"return={result['mean_return']:.1f} "
            f"standing={result['mean_standing']:.3f} "
            f"hopping={result['mean_hopping']:.3f}",
            flush=True,
        )
    returns = np.array([row["mean_return"] for row in per_seed])
    standings = np.array([row["mean_standing"] for row in per_seed])
    hoppings = np.array([row["mean_hopping"] for row in per_seed])
    aggregate = {
        "actor": actor_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_source": source,
        "eval_seeds": eval_seeds,
        "episodes_per_seed": episodes,
        "total_episodes": eval_seeds * episodes,
        "per_seed": per_seed,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "standing_mean": float(standings.mean()),
        "hopping_mean": float(hoppings.mean()),
    }
    print(
        json.dumps(
            {
                "return_mean": aggregate["return_mean"],
                "return_std": aggregate["return_std"],
                "return_min": aggregate["return_min"],
                "return_max": aggregate["return_max"],
                "standing": aggregate["standing_mean"],
                "hopping": aggregate["hopping_mean"],
            },
            indent=1,
        ),
        flush=True,
    )
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", default="BC-KMPC", choices=BC_ACTOR_ORDER)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--koopman",
        type=Path,
        default=Path("runs/hopper_hop/koopman_v2/best.pt"),
    )
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=20_270_804)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = evaluate_multiseed(
        args.actor,
        args.checkpoint,
        args.koopman,
        eval_seeds=args.eval_seeds,
        episodes=args.episodes,
        seed_start=args.seed_start,
        device_name=args.device,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        print(f"written: {args.output}", flush=True)


if __name__ == "__main__":
    main()
