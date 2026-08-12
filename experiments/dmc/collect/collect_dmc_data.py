"""Collect DMC transitions with resumable, protocol-stamped chunks.

The default is the native DMC protocol and random coverage only.  Expert
collection is opt-in and requires a checkpoint.  Every chunk contains complete
episodes (or an explicitly marked collector truncation), applied and requested
actions, reward/discount/done, globally unique episode/global-step ids, and a
canonical JSON description of the environment protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from experiments.dmc.actors import ActorConfig, StandardPPOActor
from experiments.dmc.protocol import canonical_json as canonical_protocol_json
from experiments.dmc.protocol import protocol_fingerprint
from experiments.dmc.tasks.registry import DMC_CUSTOM_PROTOCOL, get_task_spec


SEED_DIRS = ("seed_20240201", "seed_20240202", "seed_20240203")
RANDOM_UPDATE = 10
EXPERT_UPDATE = 20
COLLECTION_SCHEMA_VERSION = 3


class PPOExpert(StandardPPOActor):
    """Backward-compatible name for the shared DMC PPO actor architecture."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__(
            obs_dim,
            action_dim,
            hidden_dim=hidden,
            action_limit=1.0,
        )


def _build_expert_module(obs_dim: int, action_dim: int, module_spec: str) -> nn.Module:
    if not module_spec:
        return PPOExpert(obs_dim, action_dim)
    module_name, separator, class_name = module_spec.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("--expert-module-spec must have the form module:ClassName")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(obs_dim, action_dim)


def load_expert(
    checkpoint: Path,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    module_spec: str = "",
) -> nn.Module:
    """Load an expert policy from a checkpoint, tolerating common layouts."""
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if module_spec:
        expert = _build_expert_module(obs_dim, action_dim, module_spec)
    else:
        if isinstance(payload, dict) and payload.get("actor_type") not in (None, "PPO"):
            raise ValueError("The standalone expert collector requires a PPO actor")
        actor_config = ActorConfig.from_mapping(
            payload.get("actor_config") if isinstance(payload, dict) else None
        )
        expert = PPOExpert(
            obs_dim,
            action_dim,
            hidden=actor_config.ppo_hidden_dim,
        )
    for key in ("actor_state", "model", "actor"):
        if isinstance(payload, dict) and key in payload:
            state = payload[key]
            break
    else:
        state = payload if isinstance(payload, dict) else None
    if state is None or not any(
        key.startswith(("net", "network", "0.")) for key in state
    ):
        raise ValueError(
            f"Unrecognized expert checkpoint {checkpoint}: no actor_state/"
            "model/actor key and no obvious MLP state dict"
        )
    # Schema-v1 DMC checkpoints use ``network.*``.  Accept the abandoned
    # pre-migration ``net.*`` spelling explicitly so old local smoke artifacts
    # fail only on real architectural differences, not a prefix rename.
    state = {
        ("network." + key[len("net.") :] if key.startswith("net.") else key): value
        for key, value in state.items()
    }
    expert.load_state_dict(state)
    expert.to(device).eval()
    return expert


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _protocol_fingerprint(protocol_json: str) -> str:
    return hashlib.sha256(protocol_json.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_status(path: Path, status: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(status, indent=2, sort_keys=True) + "\n")


def _scalar(archive: Any, name: str) -> Any:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"Chunk field {name!r} must be scalar, got {value.shape}")
    return value.item()


def _scan_existing_chunks(
    output_dir: Path,
    *,
    task_name: str,
    seed_index: int,
    protocol_json: str,
    environment_protocol_json: str,
) -> dict[str, Any]:
    files = sorted(output_dir.glob("coverage_*.npz"))
    seen_episode_ids: set[int] = set()
    seen_global_steps: set[int] = set()
    stage_transitions = {"random": 0, "expert": 0}
    total_transitions = 0
    max_episode_id: Optional[int] = None
    max_global_step: Optional[int] = None
    max_chunk_index = -1
    rng_state: Optional[dict[str, Any]] = None

    required = {
        "state",
        "requested_action",
        "action",
        "next_state",
        "reward",
        "discount",
        "done",
        "terminated",
        "truncated",
        "collector_truncated",
        "episode_id",
        "step_index",
        "update",
        "global_step",
        "reset_seed",
        "collection_schema_version",
        "protocol_json",
        "environment_protocol_json",
        "protocol_fingerprint",
        "task",
        "seed_index",
        "rng_state_after_json",
    }
    for path in files:
        try:
            chunk_index = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid chunk filename {path.name!r}") from exc
        max_chunk_index = max(max_chunk_index, chunk_index)
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"Existing chunk {path} uses an obsolete/incomplete schema; "
                    f"missing {sorted(missing)}. Use a fresh output directory."
                )
            if int(_scalar(archive, "collection_schema_version")) != COLLECTION_SCHEMA_VERSION:
                raise ValueError(f"Unsupported collection schema in {path}")
            if str(_scalar(archive, "protocol_json")) != protocol_json:
                raise ValueError(f"Protocol mismatch while resuming from {path}")
            if (
                str(_scalar(archive, "environment_protocol_json"))
                != environment_protocol_json
            ):
                raise ValueError(
                    f"Environment protocol mismatch while resuming from {path}"
                )
            if str(_scalar(archive, "protocol_fingerprint")) != _protocol_fingerprint(
                environment_protocol_json
            ):
                raise ValueError(f"Protocol fingerprint mismatch in {path}")
            if str(_scalar(archive, "task")) != task_name:
                raise ValueError(f"Task mismatch while resuming from {path}")
            if int(_scalar(archive, "seed_index")) != seed_index:
                raise ValueError(f"Seed mismatch while resuming from {path}")

            count = len(archive["state"])
            for name in (
                "requested_action",
                "action",
                "next_state",
                "reward",
                "discount",
                "done",
                "terminated",
                "truncated",
                "collector_truncated",
                "episode_id",
                "step_index",
                "update",
                "global_step",
                "reset_seed",
            ):
                if len(archive[name]) != count:
                    raise ValueError(f"Length mismatch for {name!r} in {path}")

            episode_ids = np.asarray(archive["episode_id"], dtype=np.int64)
            global_steps = np.asarray(archive["global_step"], dtype=np.int64)
            chunk_episode_ids = {int(value) for value in np.unique(episode_ids)}
            duplicate_episodes = seen_episode_ids & chunk_episode_ids
            if duplicate_episodes:
                raise ValueError(
                    f"Duplicate episode ids while resuming: {sorted(duplicate_episodes)}"
                )
            if len(np.unique(global_steps)) != len(global_steps):
                raise ValueError(f"Duplicate global_step values within {path}")
            duplicate_steps = seen_global_steps & {int(value) for value in global_steps}
            if duplicate_steps:
                raise ValueError(
                    f"Duplicate global_step values while resuming from {path}"
                )
            seen_episode_ids.update(chunk_episode_ids)
            seen_global_steps.update(int(value) for value in global_steps)
            if len(episode_ids):
                max_episode_id = max(
                    int(episode_ids.max()),
                    -1 if max_episode_id is None else max_episode_id,
                )
                max_global_step = max(
                    int(global_steps.max()),
                    -1 if max_global_step is None else max_global_step,
                )
            updates = np.asarray(archive["update"], dtype=np.int64)
            unknown_updates = set(int(value) for value in np.unique(updates)) - {
                RANDOM_UPDATE,
                EXPERT_UPDATE,
            }
            if unknown_updates:
                raise ValueError(f"Unknown update tags in {path}: {unknown_updates}")
            stage_transitions["random"] += int(np.sum(updates == RANDOM_UPDATE))
            stage_transitions["expert"] += int(np.sum(updates == EXPERT_UPDATE))
            total_transitions += count
            rng_state = json.loads(str(_scalar(archive, "rng_state_after_json")))

    return {
        "files": files,
        "chunk_index": max_chunk_index + 1,
        "total_transitions": total_transitions,
        "stage_transitions": stage_transitions,
        "next_episode_id": None if max_episode_id is None else max_episode_id + 1,
        "next_global_step": 0 if max_global_step is None else max_global_step + 1,
        "rng_state": rng_state,
    }


def _flush(
    output_dir: Path,
    chunk_index: int,
    episodes: list[dict[str, Any]],
    status: dict[str, Any],
    status_path: Path,
) -> None:
    if not episodes:
        return

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([episode[name] for episode in episodes], axis=0)

    states = concatenate("state").astype(np.float32)
    requested_actions = concatenate("requested_action").astype(np.float32)
    actions = concatenate("action").astype(np.float32)
    next_states = concatenate("next_state").astype(np.float32)
    rewards = concatenate("reward").astype(np.float32)
    discounts = concatenate("discount").astype(np.float32)
    dones = concatenate("done").astype(np.bool_)
    terminated = concatenate("terminated").astype(np.bool_)
    truncated = concatenate("truncated").astype(np.bool_)
    collector_truncated = concatenate("collector_truncated").astype(np.bool_)
    episode_ids = np.concatenate(
        [
            np.full(len(episode["state"]), episode["episode_id"], dtype=np.int64)
            for episode in episodes
        ]
    )
    step_indices = np.concatenate(
        [np.arange(len(episode["state"]), dtype=np.int64) for episode in episodes]
    )
    updates = np.concatenate(
        [
            np.full(len(episode["state"]), episode["update"], dtype=np.int64)
            for episode in episodes
        ]
    )
    global_steps = concatenate("global_step").astype(np.int64)
    reset_seeds = np.concatenate(
        [
            np.full(len(episode["state"]), episode["reset_seed"], dtype=np.int64)
            for episode in episodes
        ]
    )
    protocol_json = _canonical_json(status["protocol"])
    environment_protocol_json = canonical_protocol_json(
        status["environment_protocol"]
    )
    rng_state_json = _canonical_json(status["rng_state"])
    chunk = {
        "state": states,
        "requested_action": requested_actions,
        "action": actions,
        "next_state": next_states,
        "reward": rewards,
        "discount": discounts,
        "done": dones,
        "terminated": terminated,
        "truncated": truncated,
        "collector_truncated": collector_truncated,
        "episode_id": episode_ids,
        "step_index": step_indices,
        "update": updates,
        "global_step": global_steps,
        "reset_seed": reset_seeds,
        "collection_schema_version": np.asarray(
            COLLECTION_SCHEMA_VERSION, dtype=np.int64
        ),
        "protocol_json": np.asarray(protocol_json),
        "environment_protocol_json": np.asarray(environment_protocol_json),
        "protocol_fingerprint": np.asarray(
            protocol_fingerprint(status["environment_protocol"])
        ),
        "task": np.asarray(status["task"]),
        "seed_index": np.asarray(status["seed_index"], dtype=np.int64),
        "seed_dir": np.asarray(status["seed_dir"]),
        "rng_state_after_json": np.asarray(rng_state_json),
    }
    path = output_dir / f"coverage_{chunk_index:06d}.npz"
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **chunk)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    count = len(states)
    status["total_transitions"] += count
    status["chunks_written"] += 1
    status["stage_transitions"]["random"] += int(
        np.sum(updates == RANDOM_UPDATE)
    )
    status["stage_transitions"]["expert"] += int(
        np.sum(updates == EXPERT_UPDATE)
    )
    status["next_episode_id"] = int(episode_ids.max()) + 1
    status["next_global_step"] = int(global_steps.max()) + 1
    _write_status(status_path, status)
    print(
        f"[collector] chunk {chunk_index:06d}: {count:,} transitions "
        f"(total {status['total_transitions']:,})",
        flush=True,
    )


def _validate_arguments(
    *,
    seed_index: int,
    transitions_per_stage: int,
    random_policy: bool,
    expert_policy: bool,
    expert_checkpoint: Optional[Path],
    expert_noise_std: float,
    max_episode_steps: Optional[int],
    chunk_flush_every: int,
) -> None:
    if seed_index not in range(len(SEED_DIRS)):
        raise ValueError(f"seed_index must be in {range(len(SEED_DIRS))}")
    if transitions_per_stage <= 0:
        raise ValueError("transitions_per_stage must be positive")
    if not random_policy and not expert_policy:
        raise ValueError("At least one collection policy must be enabled")
    if expert_policy:
        if expert_checkpoint is None:
            raise ValueError("expert_policy=True requires expert_checkpoint")
        if not expert_checkpoint.is_file():
            raise FileNotFoundError(expert_checkpoint)
    if not np.isfinite(expert_noise_std) or expert_noise_std < 0:
        raise ValueError("expert_noise_std must be finite and non-negative")
    if max_episode_steps is not None and max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if chunk_flush_every <= 0:
        raise ValueError("chunk_flush_every must be positive")


def collect(
    task_name: str,
    seed_index: int,
    *,
    transitions_per_stage: int,
    output_root: Path,
    random_policy: bool = True,
    expert_policy: bool = False,
    expert_checkpoint: Optional[Path] = None,
    expert_noise_std: float = 0.1,
    expert_module_spec: str = "",
    control_timestep: Optional[float] = None,
    max_episode_steps: Optional[int] = None,
    chunk_flush_every: int = 50_000,
) -> dict[str, Any]:
    """Collect one task/seed, safely resuming from durable chunks."""
    from experiments.dmc.tasks.adapter import make_dmc_adapter

    _validate_arguments(
        seed_index=seed_index,
        transitions_per_stage=transitions_per_stage,
        random_policy=random_policy,
        expert_policy=expert_policy,
        expert_checkpoint=expert_checkpoint,
        expert_noise_std=expert_noise_std,
        max_episode_steps=max_episode_steps,
        chunk_flush_every=chunk_flush_every,
    )
    spec = get_task_spec(task_name)
    seed_dir = SEED_DIRS[seed_index]
    output_dir = output_root / task_name / seed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "collection_status.json"

    env = make_dmc_adapter(
        task_name, seed=seed_index, control_timestep=control_timestep
    )
    try:
        max_steps = env.step_limit if max_episode_steps is None else int(max_episode_steps)
        if max_steps > env.step_limit:
            raise ValueError(
                f"max_episode_steps={max_steps} exceeds environment step_limit="
                f"{env.step_limit}"
            )
        if max_steps < spec.k_step:
            raise ValueError(
                f"max_episode_steps={max_steps} is shorter than k_step={spec.k_step}"
            )
        environment_protocol = env.protocol_metadata()
        protocol = dict(environment_protocol)
        protocol["collector_max_episode_steps"] = max_steps
        protocol["collector_truncates_episodes"] = max_steps < env.step_limit
        if protocol["collector_truncates_episodes"]:
            protocol["environment_protocol_name"] = protocol["protocol_name"]
            protocol["protocol_name"] = DMC_CUSTOM_PROTOCOL
        protocol_json = _canonical_json(protocol)
        environment_protocol_json = canonical_protocol_json(environment_protocol)
        environment_fingerprint = protocol_fingerprint(environment_protocol)

        scanned = _scan_existing_chunks(
            output_dir,
            task_name=task_name,
            seed_index=seed_index,
            protocol_json=protocol_json,
            environment_protocol_json=environment_protocol_json,
        )
        base_episode_id = 1_000_000 * seed_index
        episode_id = (
            base_episode_id
            if scanned["next_episode_id"] is None
            else int(scanned["next_episode_id"])
        )
        if not base_episode_id <= episode_id < base_episode_id + 1_000_000:
            raise ValueError(
                f"Episode id {episode_id} is outside seed {seed_index}'s namespace"
            )
        global_step = int(scanned["next_global_step"])
        chunk_index = int(scanned["chunk_index"])

        rng = np.random.default_rng(1000 + seed_index)
        if scanned["rng_state"] is not None:
            rng.bit_generator.state = scanned["rng_state"]
        status: dict[str, Any] = {
            "collection_schema_version": COLLECTION_SCHEMA_VERSION,
            "task": task_name,
            "seed_index": seed_index,
            "seed_dir": seed_dir,
            "obs_dim": spec.obs_dim,
            "action_dim": spec.action_dim,
            "protocol": protocol,
            "environment_protocol": environment_protocol,
            "protocol_fingerprint": environment_fingerprint,
            "total_transitions": int(scanned["total_transitions"]),
            "chunks_written": len(scanned["files"]),
            "stage_transitions": dict(scanned["stage_transitions"]),
            "next_episode_id": episode_id,
            "next_global_step": global_step,
            "rng_state": rng.bit_generator.state,
        }
        _write_status(status_path, status)
        print(
            f"[collector] {task_name}/{seed_dir}: resuming from "
            f"{status['chunks_written']} chunks "
            f"({status['total_transitions']:,} transitions)",
            flush=True,
        )

        device = torch.device("cpu")
        expert = (
            load_expert(
                expert_checkpoint,
                spec.obs_dim,
                spec.action_dim,
                device,
                module_spec=expert_module_spec,
            )
            if expert_policy and expert_checkpoint is not None
            else None
        )
        episodes: list[dict[str, Any]] = []
        pending_transitions = 0

        def flush_pending() -> None:
            nonlocal episodes, pending_transitions, chunk_index
            if not episodes:
                return
            status["rng_state"] = rng.bit_generator.state
            _flush(output_dir, chunk_index, episodes, status, status_path)
            episodes = []
            pending_transitions = 0
            chunk_index += 1

        def marker_path(policy: str) -> Path:
            return output_dir / f"{policy}_done.txt"

        def mark_stage_done(policy: str, target: int) -> None:
            marker = {
                "policy": policy,
                "protocol_json": protocol_json,
                "target_transitions": target,
                "durable_transitions": status["stage_transitions"][policy],
            }
            _atomic_write_text(
                marker_path(policy), json.dumps(marker, indent=2, sort_keys=True) + "\n"
            )

        def run_policy(policy: str, target: int) -> int:
            nonlocal episode_id, global_step, pending_transitions
            existing_count = int(status["stage_transitions"][policy])
            if marker_path(policy).exists() and existing_count < target:
                try:
                    previous_marker = json.loads(
                        marker_path(policy).read_text(encoding="utf-8")
                    )
                    marker_is_valid = (
                        previous_marker.get("protocol_json") == protocol_json
                        and int(previous_marker.get("target_transitions", -1))
                        <= existing_count
                        and int(previous_marker.get("durable_transitions", -1))
                        <= existing_count
                    )
                except (json.JSONDecodeError, OSError, TypeError, ValueError):
                    marker_is_valid = False
                if not marker_is_valid:
                    raise RuntimeError(
                        f"{marker_path(policy)} exists but only {existing_count} "
                        f"of {target} requested transitions are durable"
                    )
                print(
                    f"[collector] extending completed {policy!r} stage from "
                    f"{existing_count:,} to at least {target:,} transitions",
                    flush=True,
                )
            if existing_count >= target:
                _write_status(status_path, status)
                mark_stage_done(policy, target)
                print(
                    f"[collector] stage {policy!r} already complete "
                    f"({existing_count:,} transitions)",
                    flush=True,
                )
                return 0

            collected = 0
            stage_progress = existing_count
            update = RANDOM_UPDATE if policy == "random" else EXPERT_UPDATE
            while stage_progress < target:
                reset_seed = seed_index * 1_000_000 + (episode_id - base_episode_id)
                env.reset(seed=reset_seed)
                obs = env.get_state()
                buffers: dict[str, list[Any]] = {
                    "state": [],
                    "requested_action": [],
                    "action": [],
                    "next_state": [],
                    "reward": [],
                    "discount": [],
                    "done": [],
                    "terminated": [],
                    "truncated": [],
                    "collector_truncated": [],
                }
                for _step in range(max_steps):
                    if policy == "random":
                        requested = rng.uniform(
                            -1.0, 1.0, spec.action_dim
                        ).astype(np.float32)
                    else:
                        if expert is None:
                            raise RuntimeError("Expert policy was not initialized")
                        with torch.no_grad():
                            requested = (
                                expert(torch.from_numpy(obs).float().unsqueeze(0))
                                .squeeze(0)
                                .numpy()
                            )
                        if expert_noise_std > 0.0:
                            requested = requested + rng.normal(
                                0.0, expert_noise_std, spec.action_dim
                            )
                    next_obs, reward, done, info = env.step(requested)
                    applied = np.asarray(info["applied_action"], dtype=np.float32)
                    buffers["state"].append(obs.astype(np.float32))
                    buffers["requested_action"].append(
                        np.asarray(info["requested_action"], dtype=np.float32)
                    )
                    buffers["action"].append(applied)
                    buffers["next_state"].append(next_obs.astype(np.float32))
                    buffers["reward"].append(float(reward))
                    discount = info.get("discount")
                    buffers["discount"].append(
                        np.nan if discount is None else float(discount)
                    )
                    buffers["done"].append(bool(done))
                    terminated = bool(info.get("terminated", False))
                    truncated = bool(info.get("truncated", False))
                    if done and not (terminated or truncated):
                        # Compatibility with adapters that only expose the DMC
                        # discount: LAST+discount=1 is a time-limit truncation;
                        # LAST+discount=0 is a true terminal transition.
                        terminated = discount is not None and float(discount) == 0.0
                        truncated = not terminated
                    buffers["terminated"].append(terminated)
                    buffers["truncated"].append(truncated)
                    buffers["collector_truncated"].append(False)
                    obs = next_obs
                    global_step += 1
                    if done:
                        break

                episode_length = len(buffers["state"])
                if episode_length < spec.k_step:
                    raise RuntimeError(
                        f"DMC episode length {episode_length} is shorter than "
                        f"k_step={spec.k_step}; refusing to create a gapped resume"
                    )
                if not buffers["done"][-1]:
                    buffers["done"][-1] = True
                    buffers["collector_truncated"][-1] = True
                    buffers["truncated"][-1] = True
                episode = {
                    name: np.asarray(values)
                    for name, values in buffers.items()
                }
                episode.update(
                    {
                        "episode_id": int(episode_id),
                        "update": update,
                        "global_step": np.arange(
                            global_step - episode_length,
                            global_step,
                            dtype=np.int64,
                        ),
                        "reset_seed": int(reset_seed),
                    }
                )
                episodes.append(episode)
                episode_id += 1
                pending_transitions += episode_length
                collected += episode_length
                stage_progress += episode_length
                if pending_transitions >= chunk_flush_every:
                    flush_pending()

            # The marker is only durable after every episode in the stage is.
            flush_pending()
            status["rng_state"] = rng.bit_generator.state
            _write_status(status_path, status)
            mark_stage_done(policy, target)
            return collected

        random_count = (
            run_policy("random", transitions_per_stage) if random_policy else 0
        )
        expert_count = (
            run_policy("expert", transitions_per_stage) if expert_policy else 0
        )
        _write_status(status_path, status)
        print(
            f"[collector] DONE {task_name}/{seed_dir}: "
            f"random_new={random_count:,} expert_new={expert_count:,} "
            f"total={status['total_transitions']:,}",
            flush=True,
        )
        return status
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    from experiments.dmc.tasks.registry import TASK_SPECS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--transitions-per-stage", type=int, default=200_000)
    parser.add_argument(
        "--control-timestep",
        type=float,
        default=None,
        help="explicit custom protocol only; default uses native DMC timing",
    )
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--chunk-flush-every", type=int, default=50_000)
    parser.add_argument("--output-root", type=Path, default=Path("runs/dmc/data"))
    parser.add_argument("--no-random", action="store_true", help="skip random stage")
    parser.add_argument(
        "--expert",
        action="store_true",
        help="opt in to expert collection (requires --expert-checkpoint)",
    )
    parser.add_argument("--expert-checkpoint", type=Path, default=None)
    parser.add_argument("--expert-noise-std", type=float, default=0.1)
    parser.add_argument("--expert-module-spec", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        args.task,
        args.seed_index,
        transitions_per_stage=args.transitions_per_stage,
        output_root=args.output_root,
        random_policy=not args.no_random,
        expert_policy=args.expert,
        expert_checkpoint=args.expert_checkpoint,
        expert_noise_std=args.expert_noise_std,
        expert_module_spec=args.expert_module_spec,
        control_timestep=args.control_timestep,
        max_episode_steps=args.max_episode_steps,
        chunk_flush_every=args.chunk_flush_every,
    )


if __name__ == "__main__":
    main()
