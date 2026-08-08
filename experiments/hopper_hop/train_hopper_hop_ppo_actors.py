"""PPO training for the four HopperHop actors (PPO / KLQR / AB-PQ / KMPC).

Supports two initialization modes per actor:

  * ``--bc-checkpoint <name.pt>``  : BC-pretrained actor weights -> PPO
    fine-tuning (the recommended BC->PPO path).  The frozen Koopman lift and
    the BC dataset normalizers are reused exactly, so the fine-tuned policy
    sees the same input distribution it was BC-pretrained on.
  * default (no BC)               : train directly from scratch with PPO.

The policy is a Gaussian over each actor's deterministic mean:

  ``mean = actor_mean(obs)`` (raw 15-dim state for PPO; Koopman lift for
  KLQR / AB-PQ / KMPC),  ``a ~ Normal(mean, log_std.exp())``.

Hyperparameters match the PPO baseline trainer (num_envs=2048, rollout 100,
minibatch 6400, 8 update epochs, GAE lambda 0.95, gamma 0.99, annealed lr).
Resume is supported via ``latest.pt`` (atomic writes) and the best policy is
saved separately to ``best.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from experiments.hopper_hop.train_hopper_hop_bc import (
    BC_ACTOR_ORDER,
    BCConfig,
    canonical_actor_name,
    StandardPPOActor,
    closed_loop_evaluation,
    load_koopman,
    _make_builders,
)
from experiments.hopper_hop.train_hopper_hop_ppo import (
    STATE_DIM,
    ACTION_DIM,
    Actor as PPOBaselineActor,
    _atomic_json,
    _atomic_torch_save,
    _make_env,
)

TRAINING_SPEC_VERSION = "hopperhop_ppo_v1"


def _orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class ValueNetwork(nn.Module):
    """256x256 Tanh MLP critic (PPO: raw state; structured: lifted state)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        _orthogonal(self.network[0], math.sqrt(2.0))
        _orthogonal(self.network[2], math.sqrt(2.0))
        _orthogonal(self.network[4], 1.0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


@dataclass(frozen=True)
class ActorPPOConfig:
    actor_name: str = "PPO"
    num_envs: int = 2048
    rollout_steps: int = 100
    minibatch_size: int = 6400
    update_epochs: int = 8
    total_timesteps: int = 50_000_000
    learning_rate: float = 3e-4
    anneal_learning_rate: bool = True
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 1e-3
    initial_std: float = 1.0  # matches the official PPO baseline config
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    checkpoint_interval_updates: int = 10
    max_wall_time_seconds: float | None = None
    seed: int = 20_240_101
    bc_checkpoint: str | None = None

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps


def _actor_mean(
    name: str,
    actor: nn.Module,
    state: torch.Tensor,
    lifted: torch.Tensor,
) -> torch.Tensor:
    """Deterministic mean of each actor's policy (no context in HopperHop)."""
    if name == "PPO":
        return actor(state)
    if name == "KLQR":
        return actor(lifted).action
    if name == "AB-PQ":
        return actor(lifted, lifted).action
    return actor(lifted).action


def _value_estimate(
    name: str,
    value: nn.Module,
    state: torch.Tensor,
    lifted: torch.Tensor,
) -> torch.Tensor:
    if name == "PPO":
        return value(state)
    return value(lifted)


def _optional_mean(values: deque[float]) -> float | None:
    return float(np.mean(values)) if values else None


def train(
    actor_name: str,
    koopman_path: Path,
    output_dir: Path,
    config: ActorPPOConfig,
    device_name: str = "auto",
    resume: bool = True,
) -> dict[str, Any]:
    if actor_name not in BC_ACTOR_ORDER:
        raise ValueError(f"Unsupported actor {actor_name!r}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    koopman, koopman_payload = load_koopman(koopman_path, device)

    bc_payload: dict[str, Any] | None = None
    bc_checkpoint: Path | None = None
    if config.bc_checkpoint:
        bc_checkpoint = Path(config.bc_checkpoint)
        bc_payload = torch.load(
            bc_checkpoint, map_location=device, weights_only=False
        )
        if canonical_actor_name(bc_payload.get("name")) != canonical_actor_name(
            actor_name
        ):
            raise ValueError(
                f"BC checkpoint actor {bc_payload.get('name')!r} does not "
                f"match requested actor {actor_name!r}"
            )
        if bc_payload.get("kind") != "hopperhop_bc_actor":
            raise ValueError(f"{bc_checkpoint} is not a HopperHop BC actor")

    # actor architecture hyperparameters come from BCConfig; training
    # hyperparameters come from ActorPPOConfig.
    actor_config = BCConfig(seed=config.seed)
    builders = _make_builders(koopman, actor_config, device)
    # The PPO route must use the SAME actor as the official ManiSkill PPO
    # baseline (3x256 Tanh MLP with tanh-squashed output, orthogonal init) so
    # the comparison is a strict apples-to-apples baseline.  The BC-builder
    # StandardPPOActor (linear output) is NOT the official baseline and would
    # handicap PPO.
    builders["PPO"] = lambda: PPOBaselineActor(koopman.state_dim, ACTION_DIM)
    actor = builders[actor_name]().to(device)
    if bc_payload is not None:
        actor.load_state_dict(bc_payload["actor_state"])
    value_input_dim = (
        koopman.state_dim if actor_name == "PPO" else koopman.lifted_dim
    )
    value_net = ValueNetwork(value_input_dim).to(device)
    log_std = nn.Parameter(
        torch.full((ACTION_DIM,), math.log(config.initial_std), device=device)
    )

    # normalizers: BC dataset normalizers on the BC path, Koopman normalizers
    # otherwise (PPO route ignores them entirely -- it uses the raw state).
    if bc_payload is not None:
        normalizer = bc_payload["normalizer"]
        center = torch.as_tensor(
            normalizer["state_center"], device=device, dtype=torch.float32
        )
        scale = torch.as_tensor(
            normalizer["state_scale"], device=device, dtype=torch.float32
        )
    else:
        center = torch.as_tensor(
            koopman_payload["normalizer"]["center"],
            device=device,
            dtype=torch.float32,
        )
        scale = torch.as_tensor(
            koopman_payload["normalizer"]["scale"],
            device=device,
            dtype=torch.float32,
        )

    actor_parameters = list(actor.parameters())
    auxiliary_parameters = [*value_net.parameters(), log_std]
    optimizer = torch.optim.Adam(
        [
            {"params": actor_parameters, "lr": config.learning_rate},
            {"params": auxiliary_parameters, "lr": config.learning_rate},
        ],
        eps=1e-5,
    )

    env = _make_env(config.num_envs, config.seed)
    latest_path = output_dir / "latest.pt"
    start_update = 0
    global_step = 0
    # best-checkpoint tracking: ``best.pt`` = best recent_episode_return seen,
    # ``latest.pt`` = most recent state (both atomic, resumable).
    best_return = -float("inf")
    best_return_update = 0
    if resume and latest_path.exists():
        payload = torch.load(
            latest_path, map_location=device, weights_only=False
        )
        if canonical_actor_name(payload.get("actor_name")) != canonical_actor_name(
            actor_name
        ):
            raise ValueError("Resume checkpoint actor does not match requested actor")
        if payload.get("training_spec_version") != TRAINING_SPEC_VERSION:
            raise ValueError(
                "Resume checkpoint uses an incompatible training spec; "
                "use a new output directory or --no-resume"
            )
        actor.load_state_dict(payload["actor_state"])
        value_net.load_state_dict(payload["value_state"])
        log_std.data.copy_(payload["log_std"].to(device))
        optimizer.load_state_dict(payload["optimizer_state"])
        center = payload["center"].to(device)
        scale = payload["scale"].to(device)
        start_update = int(payload["update"])
        global_step = int(payload["global_step"])
        best_return = float(payload.get("best_return", -float("inf")))
        best_return_update = int(payload.get("best_return_update", 0))
        print(
            f"resumed {actor_name} PPO from update {start_update} "
            f"step {global_step:,}",
            flush=True,
        )

    number_updates = math.ceil(config.total_timesteps / config.batch_size)
    metrics_path = output_dir / "metrics.jsonl"
    episode_returns: deque[float] = deque(maxlen=100)
    episode_lengths: deque[float] = deque(maxlen=100)
    metadata = {
        "kind": (
            "hopperhop_ppo_actor_bc_finetune"
            if bc_payload is not None
            else "hopperhop_ppo_actor_from_scratch"
        ),
        "actor_name": actor_name,
        "training_spec_version": TRAINING_SPEC_VERSION,
        "bc_checkpoint": (
            str(bc_checkpoint.resolve()) if bc_checkpoint is not None else None
        ),
        "bc_validation_mse": (
            float(bc_payload["report"]["best_validation_mse"])
            if bc_payload is not None
            else None
        ),
        "config": asdict(config),
        "env": "MS-HopperHop-v1",
        "obs_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "device": str(device),
    }
    _atomic_json(output_dir / "run_config.json", metadata)

    started = time.perf_counter()
    observation, _ = env.reset(
        seed=[config.seed + i for i in range(config.num_envs)]
    )
    try:
        for update in range(start_update + 1, number_updates + 1):
            if config.anneal_learning_rate:
                fraction = 1.0 - (update - 1.0) / number_updates
                for group in optimizer.param_groups:
                    group["lr"] = fraction * config.learning_rate
            states: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []
            log_probs: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            dones: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            obs_t = observation
            for _ in range(config.rollout_steps):
                full = obs_t.to(device)
                with torch.no_grad():
                    normalized = (full - center) / scale
                    lifted = koopman.lift(normalized)
                    mean = _actor_mean(actor_name, actor, full, lifted)
                    std = log_std.exp().expand_as(mean)
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)
                    value = _value_estimate(
                        actor_name, value_net, full, lifted
                    )
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = torch.logical_or(
                    torch.as_tensor(terminated, device=device)
                    .bool()
                    .reshape(-1),
                    torch.as_tensor(truncated, device=device).bool().reshape(-1),
                )
                if bool(done.any()):
                    final_info = info.get("final_info", info)
                    episode = final_info.get("episode", {})
                    for item in torch.as_tensor(
                        episode.get("return", reward), device=device
                    ).reshape(-1)[done].tolist():
                        episode_returns.append(float(item))
                    for item in torch.as_tensor(
                        episode.get(
                            "episode_len",
                            torch.zeros(config.num_envs, device=device),
                        ),
                        device=device,
                    ).reshape(-1)[done].tolist():
                        episode_lengths.append(float(item))
                states.append(full)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(
                    torch.as_tensor(reward, device=device).float().reshape(-1)
                )
                dones.append(done.float())
                values.append(value)
                obs_t = next_obs
            observation = obs_t

            state_batch = torch.stack(states)
            action_batch = torch.stack(actions)
            old_log_prob = torch.stack(log_probs)
            reward_batch = torch.stack(rewards)
            done_batch = torch.stack(dones)
            old_value = torch.stack(values)

            # GAE
            with torch.no_grad():
                normalized_next = (observation.to(device) - center) / scale
                lifted_next = koopman.lift(normalized_next)
                next_value = _value_estimate(
                    actor_name, value_net, observation.to(device), lifted_next
                )
            advantages = torch.zeros_like(reward_batch)
            last_gae = torch.zeros(config.num_envs, device=device)
            for t in range(config.rollout_steps - 1, -1, -1):
                following = (
                    next_value
                    if t == config.rollout_steps - 1
                    else old_value[t + 1]
                )
                nonterminal = 1.0 - done_batch[t]
                delta = (
                    reward_batch[t]
                    + config.discount * following * nonterminal
                    - old_value[t]
                )
                last_gae = (
                    delta
                    + config.discount
                    * config.gae_lambda
                    * nonterminal
                    * last_gae
                )
                advantages[t] = last_gae
            returns = advantages + old_value

            flat_state = state_batch.flatten(0, 1)
            flat_action = action_batch.flatten(0, 1)
            flat_old_log_prob = old_log_prob.flatten()
            flat_old_value = old_value.flatten()
            flat_advantage = advantages.flatten()
            flat_return = returns.flatten()
            with torch.no_grad():
                flat_lifted = koopman.lift((flat_state - center) / scale)

            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            kl_approximations: list[float] = []
            clip_fractions: list[float] = []
            early_stopped = False
            for _ in range(config.update_epochs):
                order = torch.randperm(config.batch_size, device=device)
                for start in range(0, config.batch_size, config.minibatch_size):
                    index = order[start : start + config.minibatch_size]
                    adv = flat_advantage[index]
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                    mean = _actor_mean(
                        actor_name, actor, flat_state[index], flat_lifted[index]
                    )
                    dist = Normal(mean, log_std.exp().expand_as(mean))
                    new_log_prob = dist.log_prob(flat_action[index]).sum(-1)
                    ratio = (new_log_prob - flat_old_log_prob[index]).exp()
                    if config.target_kl is not None:
                        with torch.no_grad():
                            kl = (
                                (ratio - 1.0)
                                - (new_log_prob - flat_old_log_prob[index])
                            ).mean()
                        if float(kl) > config.target_kl:
                            early_stopped = True
                            break
                    policy_loss = -torch.minimum(
                        ratio * adv,
                        torch.clamp(
                            ratio,
                            1.0 - config.clip_ratio,
                            1.0 + config.clip_ratio,
                        )
                        * adv,
                    ).mean()
                    predicted_value = _value_estimate(
                        actor_name,
                        value_net,
                        flat_state[index],
                        flat_lifted[index],
                    )
                    value_loss = 0.5 * (
                        predicted_value - flat_return[index]
                    ).square().mean()
                    entropy = dist.entropy().sum(-1).mean()
                    loss = (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        [*actor_parameters, *auxiliary_parameters],
                        config.max_grad_norm,
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("non-finite gradient")
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = (
                            (ratio - 1.0)
                            - (new_log_prob - flat_old_log_prob[index])
                        ).mean()
                        clip_frac = (
                            (ratio - 1.0)
                            .abs()
                            .gt(config.clip_ratio)
                            .float()
                            .mean()
                        )
                    policy_losses.append(float(policy_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                    entropies.append(float(entropy.detach()))
                    kl_approximations.append(float(approx_kl.detach()))
                    clip_fractions.append(float(clip_frac.detach()))
                if early_stopped:
                    break
            global_step = min(
                update * config.batch_size, config.total_timesteps
            )

            report = {
                "update": update,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "recent_episode_return": _optional_mean(episode_returns),
                "recent_episode_length": _optional_mean(episode_lengths),
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "approximate_kl": float(np.mean(kl_approximations)),
                "clip_fraction": float(np.mean(clip_fractions)),
                "log_std": log_std.detach().cpu().tolist(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, sort_keys=True) + "\n")
            checkpoint = {
                **metadata,
                "actor_name": actor_name,
                "actor_state": actor.state_dict(),
                "value_state": value_net.state_dict(),
                "log_std": log_std.detach(),
                "optimizer_state": optimizer.state_dict(),
                "center": center,
                "scale": scale,
                "update": update,
                "global_step": global_step,
                "last_report": report,
                "best_return": float(best_return),
                "best_return_update": int(best_return_update),
            }
            is_best = False
            if (
                report["recent_episode_return"] is not None
                and float(report["recent_episode_return"]) > best_return
            ):
                best_return = float(report["recent_episode_return"])
                best_return_update = int(update)
                is_best = True
                checkpoint["best_return"] = best_return
                checkpoint["best_return_update"] = best_return_update
                checkpoint["is_best"] = True
                _atomic_torch_save(output_dir / "best.pt", checkpoint)
            if (
                update % config.checkpoint_interval_updates == 0
                or update == number_updates
                or (
                    config.max_wall_time_seconds is not None
                    and report["elapsed_seconds"]
                    >= config.max_wall_time_seconds
                )
            ):
                _atomic_torch_save(latest_path, checkpoint)
            _atomic_json(
                output_dir / "status.json",
                {"state": "running", **report, "pid": os.getpid()},
            )
            print(json.dumps(report, sort_keys=True), flush=True)
            if (
                config.max_wall_time_seconds is not None
                and report["elapsed_seconds"] >= config.max_wall_time_seconds
            ):
                break
    finally:
        env.close()

    # final closed-loop evaluation (apples-to-apples with the BC evals)
    eval_config = BCConfig(
        seed=config.seed,
        action_limit=1.0,
        evaluation_episodes=64,
        evaluation_num_envs=64,
    )
    with torch.no_grad():
        center_np = center.detach().cpu().numpy()
        scale_np = scale.detach().cpu().numpy()
    evaluation = closed_loop_evaluation(
        actor_name,
        actor,
        koopman,
        center_np,
        scale_np,
        eval_config,
        device,
    )
    final = {
        "kind": metadata["kind"],
        "actor_name": actor_name,
        "update": update,
        "global_step": global_step,
        "elapsed_seconds": time.perf_counter() - started,
        "last_report": report,
        "best_return": float(best_return),
        "best_return_update": int(best_return_update),
        "evaluation": evaluation,
    }
    _atomic_json(output_dir / "final.json", final)
    print(
        f"actor={actor_name} ppo_done update={update} "
        f"eval_return={evaluation['mean_return']:.1f} "
        f"standing={evaluation['mean_standing']:.3f} "
        f"hopping={evaluation['mean_hopping']:.3f}",
        flush=True,
    )
    return final


# small holder so eval uses the same action_limit default as BC training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", default="PPO", choices=BC_ACTOR_ORDER)
    parser.add_argument(
        "--koopman",
        type=Path,
        default=Path("runs/hopper_hop/koopman_v2/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=6400)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20_240_101)
    parser.add_argument("--bc-checkpoint", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        mode = "finetune" if args.bc_checkpoint else "scratch"
        args.output_dir = (
            Path("runs/hopper_hop") / f"ppo_{args.actor}_{mode}"
        )
    config = ActorPPOConfig(
        actor_name=args.actor,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        initial_std=args.initial_std,
        seed=args.seed,
        bc_checkpoint=(
            str(args.bc_checkpoint.resolve())
            if args.bc_checkpoint is not None
            else None
        ),
    )
    train(
        args.actor,
        args.koopman,
        args.output_dir,
        config,
        args.device,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
