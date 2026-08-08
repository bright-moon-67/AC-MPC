"""Train PPO from scratch on MS-HopperHop with periodic trajectory saving.

The PPO baseline follows the ManiSkill3 official-style configuration for the
MS-* locomotion family (state obs, pd_joint_delta_pos, normalized_dense
reward, gamma=0.99, GAE=0.95, 2048 envs, entropy 1e-3).  During training a
*budgeted* subset of rollout transitions is periodically persisted to disk so
the collected data can later train a global Deep Koopman dynamics model.

The 15-dim flat observation ``[qpos(6), qvel(7), toe_touch(1), heel_touch(1)]``
is used directly as the closed-loop dynamics state.  Only within-episode
transitions are saved (cross-episode pairs from auto-reset are dropped), with
globally-unique episode ids and consecutive step indices so chunks are
directly consumable by ``train_pandareach_koopman.build_windows``.  The total
budget is capped by transition count and by bytes (default well below 2 GB).
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

from mani_skill.utils.registration import make

# state(15) + action(4) + next_state(15) in float32 + int64 episode/step (8B)
STATE_DIM = 15
ACTION_DIM = 4
BYTES_PER_TRANSITION = 4 * (STATE_DIM + ACTION_DIM + STATE_DIM) + 8 + 8


@dataclass(frozen=True)
class HopperPPOConfig:
    num_envs: int = 2048
    rollout_steps: int = 100
    minibatch_size: int = 6400  # batch 204800 / 32 minibatches
    update_epochs: int = 8
    total_timesteps: int = 20_000_000
    learning_rate: float = 3e-4
    anneal_learning_rate: bool = True
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 1e-3
    initial_std: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    checkpoint_interval_updates: int = 10
    max_wall_time_seconds: float | None = None
    seed: int = 20_240_101

    # --- data collection ---
    collect_dir: str | None = None
    collect_every_updates: int = 5
    collect_max_transitions: int = 1_000_000
    collect_max_bytes: int = 1_900_000_000  # hard safety cap (~1.9 GB)
    collect_start_update: int = 1
    # Per-flush cap so the budget is spread across training stages instead of
    # being consumed entirely by the first flush(s). None = auto (budget /
    # number of flush points).
    collect_per_flush_cap: int | None = None

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps


def _orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def _mlp(
    input_dim: int, hidden: int = 256, output_dim: int | None = None
) -> nn.Sequential:
    out = hidden if output_dim is None else output_dim
    net = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out),
    )
    _orthogonal(net[0], math.sqrt(2.0))
    _orthogonal(net[2], math.sqrt(2.0))
    _orthogonal(net[4], 0.01 if output_dim is None else 1.0)
    return net


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = _mlp(obs_dim, output_dim=action_dim)
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = _mlp(obs_dim, output_dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _make_env(num_envs: int, seed: int):
    import gymnasium as gym
    import mani_skill.envs  # registers tasks into the gymnasium registry
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    # ``gymnasium.make`` applies the TimeLimit wrapper (episode horizon 600),
    # which the raw ``mani_skill.utils.registration.make`` does NOT. Without
    # it the base env never reports truncation and no episode ever completes.
    base = gym.make(
        "MS-HopperHop-v1",
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        sim_backend="gpu" if torch.cuda.is_available() else "cpu",
        render_backend="none",
    )
    env = ManiSkillVectorEnv(
        base,
        num_envs,
        auto_reset=True,
        ignore_terminations=False,
        record_metrics=True,
    )
    env.reset(seed=[seed + i for i in range(num_envs)])
    return env


class TransitionCollector:
    """Budgeted, periodic persistence of COMPLETE on-policy episodes.

    Episodes are buffered per-env during rollout and emitted whole when they
    finish, so every saved transition chain is contiguous (step_index
    consecutive within an episode) and directly consumable by the existing
    K-step Koopman trainer (``train_pandareach_koopman.build_windows``).
    A stage tag (``update`` / ``global_step``) is stored per transition so the
    dataset can later be balanced across training stages.  The total budget is
    capped by transition count and bytes (default well below 2 GB).
    """

    def __init__(
        self,
        config: HopperPPOConfig,
        device: torch.device,
        per_flush_cap: int | None = None,
    ):
        self.config = config
        self.device = device
        self.per_flush_cap = per_flush_cap
        self.total_transitions = 0
        self.total_bytes = 0
        self.output_dir = (
            Path(config.collect_dir) if config.collect_dir else None
        )
        self.min_episode_len = 20  # K-step window length
        self._max_pending_episodes = 4096
        # per-env current episode buffers: list of lists of tensors
        self._buf_state: list[list] = []
        self._buf_action: list[list] = []
        self._buf_next: list[list] = []
        self._buf_step: list[list] = []
        self._pending: list[dict[str, Any]] = []  # finalized episodes
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.status_path = self.output_dir / "collection_status.json"

    def init_episode_buffers(self, num_envs: int) -> None:
        self._buf_state = [[] for _ in range(num_envs)]
        self._buf_action = [[] for _ in range(num_envs)]
        self._buf_next = [[] for _ in range(num_envs)]
        self._buf_step = [[] for _ in range(num_envs)]

    def push(
        self,
        env_idx: int,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        step_index: int,
    ) -> None:
        if self.output_dir is None:
            return
        self._buf_state[env_idx].append(state)
        self._buf_action[env_idx].append(action)
        self._buf_next[env_idx].append(next_state)
        self._buf_step[env_idx].append(step_index)

    def finalize_episode(
        self,
        env_idx: int,
        episode_id: int,
        update: int,
        global_step: int,
    ) -> None:
        states = self._buf_state[env_idx]
        actions = self._buf_action[env_idx]
        nexts = self._buf_next[env_idx]
        steps = self._buf_step[env_idx]
        length = len(states)
        self._buf_state[env_idx] = []
        self._buf_action[env_idx] = []
        self._buf_next[env_idx] = []
        self._buf_step[env_idx] = []
        if self.output_dir is None or length < self.min_episode_len:
            return
        self._pending.append(
            {
                "episode_id": int(episode_id),
                "update": int(update),
                "global_step": int(global_step),
                "state": torch.stack(states),
                "action": torch.stack(actions),
                "next_state": torch.stack(nexts),
                "step_index": torch.stack(steps),
            }
        )
        if len(self._pending) > self._max_pending_episodes:
            # oldest episodes dropped first; keeps memory bounded
            self._pending = self._pending[-self._max_pending_episodes :]

    def active(self) -> bool:
        return (
            self.output_dir is not None
            and self.total_transitions < self.config.collect_max_transitions
            and self.total_bytes < self.config.collect_max_bytes
        )

    def should_flush(self, update: int) -> bool:
        return (
            self.active()
            and update >= self.config.collect_start_update
            and update % self.config.collect_every_updates == 0
        )

    def _fit_budget(self, count: int) -> int:
        remaining_transitions = (
            self.config.collect_max_transitions - self.total_transitions
        )
        remaining_bytes = self.config.collect_max_bytes - self.total_bytes
        by_bytes = max(0, remaining_bytes) // BYTES_PER_TRANSITION
        return max(0, min(count, remaining_transitions, by_bytes))

    @torch.no_grad()
    def flush(self, update: int, global_step: int) -> None:
        if self.output_dir is None or not self.should_flush(update):
            return
        if not self._pending:
            return
        # Budget: fit as many whole episodes as possible.
        lengths = np.array(
            [len(ep["state"]) for ep in self._pending], dtype=np.int64
        )
        cumulative = np.cumsum(lengths)
        budget = self._fit_budget(int(cumulative[-1]))
        if self.per_flush_cap is not None:
            budget = min(budget, self.per_flush_cap)
        if budget <= 0:
            return
        take = int(np.searchsorted(cumulative, budget, side="right"))
        take = max(1, take)
        selected = self._pending[:take]
        self._pending = self._pending[take:]

        states = torch.cat([ep["state"] for ep in selected], dim=0)
        actions = torch.cat([ep["action"] for ep in selected], dim=0)
        next_states = torch.cat([ep["next_state"] for ep in selected], dim=0)
        episode_ids = torch.cat(
            [
                torch.full(
                    (len(ep["state"]),),
                    ep["episode_id"],
                    dtype=torch.int64,
                    device=self.device,
                )
                for ep in selected
            ],
            dim=0,
        )
        step_indices = torch.cat(
            [ep["step_index"] for ep in selected], dim=0
        )
        updates = torch.cat(
            [
                torch.full(
                    (len(ep["state"]),),
                    ep["update"],
                    dtype=torch.int64,
                    device=self.device,
                )
                for ep in selected
            ],
            dim=0,
        )
        global_steps = torch.cat(
            [
                torch.full(
                    (len(ep["state"]),),
                    ep["global_step"],
                    dtype=torch.int64,
                    device=self.device,
                )
                for ep in selected
            ],
            dim=0,
        )
        chunk = {
            "state": states.detach().cpu().numpy(),
            "action": actions.detach().cpu().numpy(),
            "next_state": next_states.detach().cpu().numpy(),
            "episode_id": episode_ids.detach().cpu().numpy(),
            "step_index": step_indices.detach().cpu().numpy(),
            "update": updates.detach().cpu().numpy(),
            "global_step": global_steps.detach().cpu().numpy(),
        }
        path = self.output_dir / f"coverage_{update:06d}.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **chunk)
        os.replace(temporary, path)
        self.total_transitions += len(chunk["state"])
        self.total_bytes += len(chunk["state"]) * BYTES_PER_TRANSITION
        _atomic_json(
            self.status_path,
            {
                "total_transitions": int(self.total_transitions),
                "total_bytes": int(self.total_bytes),
                "last_update": int(update),
                "last_global_step": int(global_step),
                "last_chunk_episodes": int(len(selected)),
                "last_chunk_transitions": int(len(chunk["state"])),
                "pid": os.getpid(),
            },
        )
        print(
            f"[collector] update={update} saved {len(selected)} episodes / "
            f"{len(chunk['state']):,} transitions "
            f"(total {self.total_transitions:,}, {self.total_bytes/1e9:.2f} GB)",
            flush=True,
        )

    def state_dict(self) -> dict[str, int]:
        return {
            "total_transitions": int(self.total_transitions),
            "total_bytes": int(self.total_bytes),
        }

    def load_state_dict(self, payload: dict[str, int]) -> None:
        self.total_transitions = int(payload.get("total_transitions", 0))
        self.total_bytes = int(payload.get("total_bytes", 0))


def train(
    config: HopperPPOConfig,
    output_dir: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(config.num_envs, config.seed)
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    print(
        f"HopperHop PPO | obs={obs_dim} act={action_dim} "
        f"num_envs={config.num_envs} batch={config.batch_size} "
        f"total_steps={config.total_timesteps:,}",
        flush=True,
    )
    number_updates = math.ceil(config.total_timesteps / config.batch_size)
    if config.collect_per_flush_cap is None and config.collect_dir:
        flush_points = max(1, number_updates // config.collect_every_updates)
        per_flush_cap = max(1, config.collect_max_transitions // flush_points)
    else:
        per_flush_cap = config.collect_per_flush_cap
    actor = Actor(obs_dim, action_dim).to(device)
    critic = Critic(obs_dim).to(device)
    log_std = nn.Parameter(
        torch.full((action_dim,), math.log(config.initial_std), device=device)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": actor.parameters(), "lr": config.learning_rate},
            {"params": critic.parameters(), "lr": config.learning_rate},
            {"params": [log_std], "lr": config.learning_rate},
        ],
        eps=1e-5,
    )
    collector = TransitionCollector(config, device, per_flush_cap=per_flush_cap)
    collector.init_episode_buffers(config.num_envs)
    episode_counter = config.num_envs
    metrics_path = output_dir / "metrics.jsonl"
    latest_path = output_dir / "latest.pt"
    start_update = 0
    global_step = 0
    if latest_path.exists():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        actor.load_state_dict(payload["actor_state"])
        critic.load_state_dict(payload["critic_state"])
        log_std.data.copy_(payload["log_std"].to(device))
        optimizer.load_state_dict(payload["optimizer_state"])
        start_update = int(payload["update"])
        global_step = int(payload["global_step"])
        collector.load_state_dict(payload["collector_state"])
        episode_counter = int(payload.get("episode_counter", episode_counter))
        print(f"resumed from update {start_update} step {global_step:,}", flush=True)
    metadata = {
        "kind": "hopper_hop_ppo",
        "config": asdict(config),
        "env": "MS-HopperHop-v1",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": STATE_DIM,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    _atomic_json(output_dir / "run_config.json", metadata)

    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    started = time.perf_counter()
    # per-env episode bookkeeping; episode ids are globally unique so later
    # dataset assembly can merge chunks without id collisions.
    episode_id = torch.arange(config.num_envs, dtype=torch.int64, device=device)
    step_index = torch.zeros(config.num_envs, dtype=torch.int64, device=device)

    try:
        observation, _ = env.reset(
            seed=[config.seed + i for i in range(config.num_envs)]
        )
        for update in range(start_update + 1, number_updates + 1):
            if config.anneal_learning_rate:
                fraction = 1.0 - (update - 1.0) / number_updates
                for group in optimizer.param_groups:
                    group["lr"] = fraction * config.learning_rate
            states: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []
            next_states: list[torch.Tensor] = []
            log_probs: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            dones: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            obs_t = observation
            for _ in range(config.rollout_steps):
                full = obs_t.to(device)
                with torch.no_grad():
                    mean = actor(full)
                    std = log_std.exp().expand_as(mean)
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)
                    value = critic(full)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = torch.logical_or(
                    torch.as_tensor(terminated, device=device).bool().reshape(-1),
                    torch.as_tensor(truncated, device=device).bool().reshape(-1),
                )
                next_full = next_obs.to(device)
                reward_t = (
                    torch.as_tensor(reward, device=device).float().reshape(-1)
                )
                if bool(done.any()):
                    final_info = info.get("final_info", info)
                    episode = final_info.get("episode", {})
                    for item in torch.as_tensor(
                        episode.get("return", reward_t), device=device
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
                # push this transition into the collector's episode buffers
                if collector.output_dir is not None:
                    for env_idx in range(config.num_envs):
                        collector.push(
                            env_idx,
                            full[env_idx],
                            action[env_idx],
                            next_full[env_idx],
                            step_index[env_idx],
                        )
                    for env_idx in done.nonzero(as_tuple=False).flatten().tolist():
                        collector.finalize_episode(
                            env_idx,
                            int(episode_id[env_idx]),
                            update,
                            global_step,
                        )
                        episode_id[env_idx] = episode_counter
                        episode_counter += 1
                states.append(full)
                actions.append(action)
                next_states.append(next_full)
                log_probs.append(log_prob)
                rewards.append(reward_t)
                dones.append(done.float())
                values.append(value)
                step_index = torch.where(
                    done, torch.zeros_like(step_index), step_index + 1
                )
                obs_t = next_obs
            observation = obs_t

            state_batch = torch.stack(states)
            action_batch = torch.stack(actions)
            next_state_batch = torch.stack(next_states)
            old_log_prob = torch.stack(log_probs)
            reward_batch = torch.stack(rewards)
            done_batch = torch.stack(dones)
            old_value = torch.stack(values)

            # GAE
            with torch.no_grad():
                next_value = critic(observation.to(device))
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
            flat_next = next_state_batch.flatten(0, 1)
            flat_old_log_prob = old_log_prob.flatten()
            flat_old_value = old_value.flatten()
            flat_advantage = advantages.flatten()
            flat_return = returns.flatten()

            # periodic budgeted flush of completed episodes
            collector.flush(update, global_step)

            # PPO update
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
                    mean = actor(flat_state[index])
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
                    predicted_value = critic(flat_state[index])
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
                        [*actor.parameters(), *critic.parameters(), log_std],
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
                            (ratio - 1.0).abs().gt(config.clip_ratio).float().mean()
                        )
                    policy_losses.append(float(policy_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                    entropies.append(float(entropy.detach()))
                    kl_approximations.append(float(approx_kl.detach()))
                    clip_fractions.append(float(clip_frac.detach()))
                if early_stopped:
                    break
            global_step = min(update * config.batch_size, config.total_timesteps)

            report = {
                "update": update,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "recent_episode_return": (
                    float(np.mean(episode_returns)) if episode_returns else None
                ),
                "recent_episode_length": (
                    float(np.mean(episode_lengths)) if episode_lengths else None
                ),
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
                "actor_state": actor.state_dict(),
                "critic_state": critic.state_dict(),
                "log_std": log_std.detach(),
                "optimizer_state": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "collector_state": collector.state_dict(),
                "episode_counter": int(episode_counter),
                "last_report": report,
            }
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
        final = {
            "kind": "hopper_hop_ppo_result",
            "update": update,
            "global_step": global_step,
            "elapsed_seconds": time.perf_counter() - started,
            "collected_transitions": collector.total_transitions,
            "collected_bytes": collector.total_bytes,
            "last_report": report,
        }
        _atomic_json(output_dir / "final.json", final)
        print(json.dumps(final, sort_keys=True), flush=True)
        return final
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=6400)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--initial-std", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=10)
    parser.add_argument("--max-wall-time-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20_240_101)
    parser.add_argument("--collect-dir", type=str, default=None)
    parser.add_argument("--collect-every-updates", type=int, default=5)
    parser.add_argument("--collect-max-transitions", type=int, default=1_000_000)
    parser.add_argument("--collect-start-update", type=int, default=1)
    parser.add_argument("--collect-per-flush-cap", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        HopperPPOConfig(
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            minibatch_size=args.minibatch_size,
            update_epochs=args.update_epochs,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            entropy_coefficient=args.entropy_coefficient,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            initial_std=args.initial_std,
            checkpoint_interval_updates=args.checkpoint_interval_updates,
            max_wall_time_seconds=(
                args.max_wall_time_minutes * 60.0
                if args.max_wall_time_minutes
                else None
            ),
            seed=args.seed,
            collect_dir=args.collect_dir,
            collect_every_updates=args.collect_every_updates,
            collect_max_transitions=args.collect_max_transitions,
            collect_start_update=args.collect_start_update,
            collect_per_flush_cap=args.collect_per_flush_cap,
        ),
        args.output_dir,
        args.device,
    )


if __name__ == "__main__":
    main()
