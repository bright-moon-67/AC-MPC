"""Collect MuJoCo Hopper transitions in the exact PPO-collector chunk schema.

The MuJoCo branch reuses the original Koopman trainer end-to-end, so the
collected chunks must match ``train_hopper_hop_ppo.TransitionCollector.flush``
byte-for-byte in schema:

    state [N,15] (legacy15), action [N,4] (canonical), next_state [N,15],
    episode_id [N] (globally unique), step_index [N] (consecutive),
    update [N] (policy stage tag), global_step [N] (monotonic)

Each chunk contains only COMPLETE episodes (contiguous step_index), exactly
like the PPO collector. ``update`` tags mimic multi-stage PPO data so the
existing 8/1/1 stage-interleaved split in ``build_hopperhop_dataset.py`` works
unchanged: random-policy stage -> update=10, PhysX-PPO-policy stage -> update=20.

Usage (one process per config x seed; run 9 in parallel on a multicore box):

    python -m experiments.hopper_hop_mujoco.collect.collect_hopperhop_mujoco \
        --contact mujoco_compliant --seed-index 0 --transitions-per-stage 200000

Writes chunks under
    runs/hopper_hop_mujoco/data/<contact>/seed_2024020{1+seed_index}/coverage_*.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from experiments.hopper_hop_mujoco.envs.mujoco_hopper import (
    MuJoCoHopper,
    MuJoCoHopperConfig,
)
from experiments.hopper_hop_mujoco.envs.contact_config import PRESET_CONTACT_CONFIGS

MIN_EPISODE_LEN = 20  # must cover the K-step window length (K=20)
SEED_DIRS = ("seed_20240201", "seed_20240202", "seed_20240203")
RANDOM_UPDATE = 10   # early-policy stage tag (mimics PPO start)
PPO_UPDATE = 20      # late-policy stage tag (mimics trained PPO)
PPO_CKPT = "runs/hopper_hop/ppo_fair/PPO/latest.pt"
PPO_NOISE_STD = 0.1  # small exploration noise around the deterministic policy


def _load_ppo_actor(device: torch.device):
    """Load the PhysX-trained PPO actor (state->action, tanh output in [-1,1]).

    Runs single-threaded: the actor is a tiny 256x256 MLP and, more importantly,
    several collector processes running multi-threaded torch inference in
    parallel oversubscribe the CPU OpenMP pools and thrash (this stalled the
    PPO stage when 9 collectors ran together).
    """
    import sys

    torch.set_num_threads(1)
    sys.path.insert(0, "experiments/hopper_hop")
    from train_hopper_hop_ppo import Actor

    payload = torch.load(PPO_CKPT, map_location=device, weights_only=False)
    actor = Actor(obs_dim=15, action_dim=4)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()
    return actor


def _flush(
    output_dir: Path,
    chunk_index: int,
    episodes: list[dict],
    status: dict,
) -> None:
    states = np.concatenate([ep["state"] for ep in episodes], axis=0)
    actions = np.concatenate([ep["action"] for ep in episodes], axis=0)
    nexts = np.concatenate([ep["next_state"] for ep in episodes], axis=0)
    episode_ids = np.concatenate(
        [np.full(len(ep["state"]), ep["episode_id"], dtype=np.int64) for ep in episodes]
    )
    step_idx = np.concatenate(
        [np.arange(len(ep["state"]), dtype=np.int64) for ep in episodes]
    )
    updates = np.concatenate(
        [np.full(len(ep["state"]), ep["update"], dtype=np.int64) for ep in episodes]
    )
    global_steps = np.concatenate(
        [ep["global_step"] for ep in episodes], axis=0
    ).astype(np.int64)
    chunk = {
        "state": states.astype(np.float32),
        "action": actions.astype(np.float32),
        "next_state": nexts.astype(np.float32),
        "episode_id": episode_ids,
        "step_index": step_idx,
        "update": updates,
        "global_step": global_steps,
    }
    path = output_dir / f"coverage_{chunk_index:06d}.npz"
    # numpy 2.x appends ".npz" to the given filename, so the temp name must
    # already end in ".npz" for the atomic replace to target the right file
    temporary = path.with_name(path.stem + "_tmp.npz")
    np.savez_compressed(temporary, **chunk)
    os.replace(temporary, path)
    status["total_transitions"] += len(chunk["state"])
    status["chunks_written"] += 1
    print(
        f"[collector] chunk {chunk_index:06d}: {len(chunk['state']):,} "
        f"transitions (total {status['total_transitions']:,})",
        flush=True,
    )


def collect(
    contact: str,
    seed_index: int,
    *,
    physics_dt: float,
    transitions_per_stage: int,
    max_episode_steps: int,
    output_root: Path,
    ppo_policy: bool,
    random_policy: bool,
) -> dict:
    if contact not in PRESET_CONTACT_CONFIGS:
        raise ValueError(f"unknown contact {contact!r}")
    seed_dir = SEED_DIRS[seed_index]
    output_dir = output_root / contact / seed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {"total_transitions": 0, "chunks_written": 0}
    status_path = output_dir / "collection_status.json"
    if status_path.exists():
        status.update(json.loads(status_path.read_text()))

    # ---- resume support -----------------------------------------------
    # Existing chunks are kept; new flushes continue the numbering. A
    # ``{policy}_done.txt`` marker means that stage is complete and is skipped
    # (so a killed run does not waste already-collected transitions).
    existing = sorted(output_dir.glob("coverage_*.npz"))
    chunk_index = len(existing)
    for chunk_path in existing:
        with np.load(chunk_path, allow_pickle=False) as archive:
            status["total_transitions"] += int(len(archive["state"]))
    status["chunks_written"] = chunk_index
    print(
        f"[collector] {contact}/{seed_dir}: resuming with {chunk_index} "
        f"existing chunks ({status['total_transitions']:,} transitions)",
        flush=True,
    )

    env = MuJoCoHopper(
        MuJoCoHopperConfig(mjcf_path="experiments/hopper_hop_mujoco/assets/hopper_mujoco.xml",
                           contact=contact, seed=seed_index, physics_dt=physics_dt)
    )
    rng = np.random.default_rng(1000 + seed_index)
    actor = _load_ppo_actor(torch.device("cpu")) if ppo_policy else None

    episode_id = 1000000 * seed_index  # unique across seeds
    global_step = 0
    episodes: list[dict] = []
    pending_transitions = 0

    def _stage_done(policy: str) -> bool:
        return (output_dir / f"{policy}_done.txt").exists()

    def _mark_stage_done(policy: str) -> None:
        (output_dir / f"{policy}_done.txt").write_text("done\n")

    def _run_policy(policy: str, target: int) -> int:
        nonlocal episode_id, global_step, chunk_index, episodes, pending_transitions
        if _stage_done(policy):
            print(f"[collector] stage {policy!r} already done, skipping", flush=True)
            return 0
        collected = 0
        while collected < target:
            env.reset(seed=seed_index * 100 + (episode_id % 1000))
            obs = env.get_legacy15_state()
            buf_s, buf_a, buf_n = [], [], []
            for step in range(max_episode_steps):
                if policy == "random":
                    a = rng.uniform(-1.0, 1.0, 4).astype(np.float32)
                else:
                    with torch.no_grad():
                        a = actor(torch.from_numpy(obs).float()).numpy()
                    if PPO_NOISE_STD > 0.0:
                        a = a + rng.normal(0.0, PPO_NOISE_STD, 4)
                    a = np.clip(a, -1.0, 1.0).astype(np.float32)
                obs_next, _rew, done, _info = env.step(a)
                # env.step returns an obs *dict*; the canonical Koopman state
                # is the legacy15 vector (same one fed to the PPO actor)
                obs_next_state = env.get_legacy15_state()
                buf_s.append(obs.astype(np.float32))
                buf_a.append(a)
                buf_n.append(obs_next_state.astype(np.float32))
                obs = obs_next_state
                global_step += 1
                if done:
                    break
            if len(buf_s) < MIN_EPISODE_LEN:
                continue  # too short for a K-step window; drop like the PPO collector
            episodes.append(
                {
                    "state": np.stack(buf_s),
                    "action": np.stack(buf_a),
                    "next_state": np.stack(buf_n),
                    "episode_id": int(episode_id),
                    "update": RANDOM_UPDATE if policy == "random" else PPO_UPDATE,
                    "global_step": np.arange(
                        global_step - len(buf_s), global_step, dtype=np.int64
                    ),
                }
            )
            episode_id += 1
            pending_transitions += len(buf_s)
            collected += len(buf_s)
            if pending_transitions >= 50_000:
                _flush(output_dir, chunk_index, episodes, status)
                episodes = []
                pending_transitions = 0
                chunk_index += 1
        _mark_stage_done(policy)
        return collected

    n_random = _run_policy("random", transitions_per_stage) if random_policy else 0
    n_ppo = _run_policy("ppo", transitions_per_stage) if ppo_policy else 0
    if episodes:
        _flush(output_dir, chunk_index, episodes, status)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True))
    print(
        f"[collector] DONE {contact}/{seed_dir}: random={n_random:,} "
        f"ppo={n_ppo:,} total={status['total_transitions']:,}",
        flush=True,
    )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact", choices=list(PRESET_CONTACT_CONFIGS), required=True)
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--transitions-per-stage", type=int, default=200_000)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--output-root", type=Path, default=Path("runs/hopper_hop_mujoco/data"))
    parser.add_argument("--no-ppo", action="store_true", help="skip the PPO stage")
    parser.add_argument("--no-random", action="store_true", help="skip the random stage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        args.contact,
        args.seed_index,
        physics_dt=args.physics_dt,
        transitions_per_stage=args.transitions_per_stage,
        max_episode_steps=args.max_episode_steps,
        output_root=args.output_root,
        ppo_policy=not args.no_ppo,
        random_policy=not args.no_random,
    )


if __name__ == "__main__":
    main()
