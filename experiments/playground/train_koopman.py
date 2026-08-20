"""GPU-native reward-free K-step Koopman training for staged trajectory data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from experiments.playground.tasks import PLAYGROUND_COMMIT, TASKS
from experiments.playground.train_ppo import _atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relocated_path(path: str) -> Path:
    """Resolve legacy prepared-data paths after the formal storage move."""

    candidate = Path(path).resolve()
    if candidate.exists():
        return candidate
    old_prefix = "/root/autodl-tmp/AC-MPC/runs/o2o/data/"
    new_prefix = "/root/acmpc-o2o-nonformal-20260819/data/"
    if path.startswith(old_prefix):
        relocated = Path(new_prefix + path[len(old_prefix) :]).resolve()
        if relocated.exists():
            return relocated
    return candidate


def _manifest_stage_order(manifest: dict[str, Any]) -> tuple[str, ...]:
    order = manifest.get("stage_order")
    stages = manifest.get("stages")
    if order is None and isinstance(stages, dict) and set(stages) == {
        "early",
        "mid",
        "late",
    }:
        order = ["early", "mid", "late"]
    if (
        not isinstance(order, list)
        or not order
        or not all(isinstance(name, str) and name for name in order)
        or len(order) != len(set(order))
        or not isinstance(stages, dict)
        or set(order) != set(stages)
    ):
        raise ValueError("Dataset manifest has an invalid stage_order/stages contract")
    return tuple(order)


def _load_data(directory: Path, task: str, manifest: dict[str, Any]):
    states, actions, splits, stage_labels = [], [], [], []
    stage_order = _manifest_stage_order(manifest)
    manifest_stages = manifest["stages"]
    manifest_counts = manifest.get("stage_episode_counts")
    if manifest_counts is None:
        manifest_counts = {
            name: metadata.get("episodes")
            for name, metadata in manifest_stages.items()
            if isinstance(metadata, dict)
        }
    if not isinstance(manifest_counts, dict) or set(manifest_counts) != set(stage_order):
        raise ValueError("Dataset stage episode counts are missing or inconsistent")
    for stage_index, stage in enumerate(stage_order):
        path = directory / f"{stage}.npz"
        stage_metadata = manifest_stages[stage]
        if not isinstance(stage_metadata, dict):
            raise ValueError(f"Dataset manifest stage {stage!r} is invalid")
        if _relocated_path(stage_metadata.get("path", "")) != path.resolve():
            raise ValueError(f"Dataset manifest path differs for stage {stage!r}")
        if stage_metadata.get("sha256") != _sha256(path):
            raise ValueError(f"Dataset stage SHA256 differs for {stage!r}")
        with np.load(path, allow_pickle=False) as archive:
            state = np.asarray(archive["states"], dtype=np.float32)
            action = np.asarray(archive["actions"], dtype=np.float32)
            # Reward is deliberately outside the Koopman training contract.
            # It is owned by the optional MPVE/reward pipeline.
        expected = TASKS[task]
        if state.ndim != 3 or state.shape[1:] != (
            expected.episode_steps + 1,
            expected.observation_dim,
        ):
            raise ValueError(f"{stage} states have the wrong shape {state.shape}")
        if action.shape != (
            state.shape[0], expected.episode_steps, expected.action_dim
        ):
            raise ValueError(f"{stage} state/action shapes are inconsistent")
        if not all(np.isfinite(x).all() for x in (state, action)):
            raise FloatingPointError(f"{stage} contains NaN or Inf")
        if state.shape[0] != manifest_counts[stage]:
            raise ValueError(f"{stage} episode count differs from the manifest")
        episode_mod = np.arange(state.shape[0]) % 10
        split = np.where(episode_mod < 8, 0, np.where(episode_mod == 8, 1, 2))
        states.append(state)
        actions.append(action)
        splits.append(split)
        stage_labels.extend([stage_index] * state.shape[0])
    return (
        np.concatenate(states),
        np.concatenate(actions),
        np.concatenate(splits),
        np.asarray(stage_labels, dtype=np.int32),
        stage_order,
    )


def _init_linear(key: Any, output_dim: int, input_dim: int, *, zero: bool = False):
    import jax
    import jax.numpy as jp

    if zero:
        return {
            "weight": jp.zeros((output_dim, input_dim), dtype=jp.float32),
            "bias": jp.zeros((output_dim,), dtype=jp.float32),
        }
    weight_key, bias_key = jax.random.split(key)
    bound = 1.0 / math.sqrt(input_dim)
    return {
        "weight": jax.random.uniform(
            weight_key, (output_dim, input_dim), minval=-bound, maxval=bound
        ),
        "bias": jax.random.uniform(
            bias_key, (output_dim,), minval=-bound, maxval=bound
        ),
    }


def _init_params(key: Any, obs_dim: int, action_dim: int, lift_dim: int):
    import jax
    import jax.numpy as jp

    keys = jax.random.split(key, 10)
    lifted_dim = obs_dim + lift_dim
    encoder = (
        _init_linear(keys[0], 256, obs_dim),
        _init_linear(keys[1], 256, 256),
        _init_linear(keys[2], lift_dim, 256),
    )
    return {
        "encoder": encoder,
        "A": jp.eye(lifted_dim) + 0.001 * jax.random.normal(keys[6], (lifted_dim, lifted_dim)),
        "B": 0.01 * jax.random.normal(keys[7], (lifted_dim, action_dim)),
    }


def _mlp(layers: Any, value: Any, *, sigmoid_final: bool = False):
    import jax.nn as jnn

    for index, layer in enumerate(layers):
        value = value @ layer["weight"].T + layer["bias"]
        if index + 1 < len(layers):
            value = jnn.silu(value)
    return jnn.sigmoid(value) if sigmoid_final else value


def _lift(params: Any, state: Any):
    import jax.numpy as jp

    return jp.concatenate((state, _mlp(params["encoder"], state)), axis=-1)


def _differentiable_spectral_radius(matrix: Any, iterations: int = 32):
    """Estimate the dominant eigenvalue magnitude without a GPU eigensolver.

    JAX does not implement nonsymmetric eigendecomposition on CUDA.  Power
    iteration is differentiable, inexpensive for the 15/72 dimensional
    Koopman matrices used here, and converges to the spectral radius when the
    dominant mode is unique.  Exact NumPy eigvals are still recorded once per
    epoch after the small matrix has moved to the host.
    """

    import jax
    import jax.numpy as jp

    size = matrix.shape[0]
    vector = jp.arange(1, size + 1, dtype=matrix.dtype)
    vector = vector / jp.maximum(jp.linalg.norm(vector), 1e-12)

    def step(current: Any, _unused: Any):
        following = matrix @ current
        following = following / jp.maximum(jp.linalg.norm(following), 1e-12)
        return following, None

    vector, _ = jax.lax.scan(step, vector, None, length=iterations)
    return jp.linalg.norm(matrix @ vector)


def _loss(
    params: Any,
    states: Any,
    actions: Any,
    *,
    rollout_discount: float,
    spectral_radius_limit: float,
):
    import jax
    import jax.numpy as jp

    z0 = _lift(params, states[:, 0])

    def step(current: Any, action: Any):
        following = current @ params["A"].T + action @ params["B"].T
        return following, following

    _last, predicted_t = jax.lax.scan(step, z0, jp.swapaxes(actions, 0, 1))
    predicted = jp.swapaxes(predicted_t, 0, 1)
    true_lifts = _lift(params, states[:, 1:])
    lifted_step_error = jp.mean((predicted - true_lifts) ** 2, axis=(0, 2))
    state_step_error = jp.mean(
        (predicted[..., : states.shape[-1]] - states[:, 1:]) ** 2,
        axis=(0, 2),
    )
    weights = rollout_discount ** jp.arange(actions.shape[1], dtype=states.dtype)
    linear = jp.sum(lifted_step_error * weights) / jp.sum(weights)
    rollout = jp.sum(state_step_error * weights) / jp.sum(weights)
    phi = _mlp(params["encoder"], states.reshape(-1, states.shape[-1]))
    phi_std = jp.std(phi, axis=0)
    latent_std = jp.mean(jp.maximum(0.0, 1.0 - phi_std) ** 2)
    spectral_radius = _differentiable_spectral_radius(params["A"])
    stability = jp.maximum(0.0, spectral_radius - spectral_radius_limit) ** 2
    identity = jp.mean((params["A"] - jp.eye(params["A"].shape[0])) ** 2)
    dynamics = (
        10.0 * linear
        + rollout
        + stability
        + 0.1 * latent_std
        + 1e-4 * identity
    )
    total = dynamics
    return total, {
        "total": total,
        "dynamics": dynamics,
        "linear": linear,
        "rollout": rollout,
        "stability": stability,
        "latent_std": latent_std,
        "spectral_radius": spectral_radius,
    }


def _atomic_export(
    path: Path,
    params: Any,
    center: np.ndarray,
    scale: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    import jax

    host = jax.tree.map(np.asarray, params)
    obs_dim = center.shape[0]
    lifted_dim = host["A"].shape[0]
    C = np.zeros((obs_dim, lifted_dim), dtype=np.float32)
    C[:, :obs_dim] = np.eye(obs_dim, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "A": host["A"].astype(np.float32),
        "B": host["B"].astype(np.float32),
        "C": C,
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    }
    for index, layer in enumerate(host["encoder"]):
        arrays[f"encoder_{index}_weight"] = layer["weight"].astype(np.float32)
        arrays[f"encoder_{index}_bias"] = layer["bias"].astype(np.float32)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jp
    import optax

    task = TASKS[args.task]
    data_manifest = json.loads(
        (args.data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if data_manifest.get("task") != args.task:
        raise ValueError("Dataset manifest task does not match --task")
    states_np, actions_np, split_np, stages_np, stage_order = _load_data(
        args.data_dir, args.task, data_manifest
    )
    train_episode = np.flatnonzero(split_np == 0)
    validation_episode = np.flatnonzero(split_np == 1)
    center = np.mean(states_np[train_episode, :-1], axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.std(states_np[train_episode, :-1], axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    states = jp.asarray((states_np - center) / scale)
    actions = jp.asarray(actions_np)
    del states_np, actions_np
    train_episode_jax = jp.asarray(train_episode, dtype=jp.int32)
    max_start = task.episode_steps - args.k_step
    if max_start < 1:
        raise ValueError("K-step horizon is too long for the task episode")
    validation_rng = np.random.default_rng(args.seed + 1)
    val_episodes = validation_rng.choice(
        validation_episode, size=args.validation_windows, replace=True
    ).astype(np.int32)
    val_starts = validation_rng.integers(
        0, max_start + 1, size=args.validation_windows, dtype=np.int32
    )
    val_episodes_jax = jp.asarray(val_episodes)
    val_starts_jax = jp.asarray(val_starts)
    offsets_state = jp.arange(args.k_step + 1, dtype=jp.int32)
    offsets_action = jp.arange(args.k_step, dtype=jp.int32)

    def make_batch(episode_index: Any, start_index: Any):
        batch_states = states[
            episode_index[:, None], start_index[:, None] + offsets_state[None, :]
        ]
        batch_actions = actions[
            episode_index[:, None], start_index[:, None] + offsets_action[None, :]
        ]
        return batch_states, batch_actions

    spectral_limit = args.spectral_radius_limit ** (
        task.control_timestep / args.stability_reference_dt
    )
    params = _init_params(
        jax.random.PRNGKey(args.seed), task.observation_dim, task.action_dim, args.lift_dim
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.add_decayed_weights(1e-6),
        optax.adam(args.learning_rate, eps=1e-8),
    )
    optimizer_state = optimizer.init(params)
    steps_per_epoch = math.ceil(args.max_windows / args.batch_size)

    def one_update(carry: Any, unused: Any):
        current_params, current_optimizer, key = carry
        key, episode_key, start_key = jax.random.split(key, 3)
        episode_selector = jax.random.randint(
            episode_key, (args.batch_size,), 0, train_episode_jax.shape[0]
        )
        episode_index = train_episode_jax[episode_selector]
        start_index = jax.random.randint(
            start_key, (args.batch_size,), 0, max_start + 1
        )
        batch = make_batch(episode_index, start_index)
        (loss, metrics), gradient = jax.value_and_grad(_loss, has_aux=True)(
            current_params,
            *batch,
            rollout_discount=0.99,
            spectral_radius_limit=spectral_limit,
        )
        updates, current_optimizer = optimizer.update(
            gradient, current_optimizer, current_params
        )
        current_params = optax.apply_updates(current_params, updates)
        return (current_params, current_optimizer, key), metrics

    @jax.jit
    def train_epoch(current_params: Any, current_optimizer: Any, key: Any):
        (current_params, current_optimizer, key), metrics = jax.lax.scan(
            one_update,
            (current_params, current_optimizer, key),
            None,
            length=steps_per_epoch,
        )
        return current_params, current_optimizer, key, jax.tree.map(jp.mean, metrics)

    @jax.jit
    def validate(current_params: Any):
        batch = make_batch(val_episodes_jax, val_starts_jax)
        _total, metrics = _loss(
            current_params,
            *batch,
            rollout_discount=0.99,
            spectral_radius_limit=spectral_limit,
        )
        return metrics

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    run_metadata = {
        "kind": "mujoco_playground_koopman_training_v1",
        "task": args.task,
        "playground_commit": PLAYGROUND_COMMIT,
        "data_dir": str(args.data_dir.resolve()),
        "data_manifest_sha256": _sha256(args.data_dir / "manifest.json"),
        "data_collection": {
            key: data_manifest.get(key)
            for key in (
                "kind",
                "policy",
                "total_transitions",
                "behaviors",
                "uniform_held_action_steps",
            )
            if key in data_manifest
        },
        "episodes": int(len(split_np)),
        "train_episodes": int(len(train_episode)),
        "validation_episodes": int(len(validation_episode)),
        "stage_episode_counts": {
            name: int(np.sum(stages_np == index))
            for index, name in enumerate(stage_order)
        },
        "stage_order": list(stage_order),
        "architecture": {
            "architecture": "fullA_history_v2_adapted",
            "state_dim": task.observation_dim,
            "action_dim": task.action_dim,
            "lift_dim": args.lift_dim,
            "hidden_dims": [256, 256],
            "activation": "silu",
        },
        "reward_training": "disabled; reward is outside the Koopman contract",
        "k_step": args.k_step,
        "batch_size": args.batch_size,
        "max_windows_per_epoch": args.max_windows,
        "steps_per_epoch": steps_per_epoch,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "spectral_radius_limit_reference": args.spectral_radius_limit,
        "stability_reference_dt": args.stability_reference_dt,
        "effective_spectral_radius_limit": spectral_limit,
        "spectral_radius_training_estimator": "differentiable_power_iteration_32_v1",
        "spectral_radius_audit": "exact_numpy_eigvals_once_per_epoch",
        "seed": args.seed,
        "encoder_layer_count": 3,
        "reward_layer_count": 0,
        "started_unix_seconds": time.time(),
    }
    _atomic_json(args.output_dir / "run.json", run_metadata)
    best = float("inf")
    best_epoch = 0
    stale = 0
    key = jax.random.PRNGKey(args.seed + 2)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        params, optimizer_state, key, train_metrics = train_epoch(
            params, optimizer_state, key
        )
        validation_metrics = validate(params)
        train_host, validation_host = jax.tree.map(
            lambda value: float(np.asarray(value)),
            (train_metrics, validation_metrics),
        )
        exact_spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(np.asarray(params["A"]))))
        )
        train_host["spectral_radius_exact"] = exact_spectral_radius
        validation_host["spectral_radius_exact"] = exact_spectral_radius
        joint = validation_host["rollout"]
        row = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - started,
            **{f"train_{key}": value for key, value in train_host.items()},
            **{f"validation_{key}": value for key, value in validation_host.items()},
            "validation_joint_objective": joint,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if joint < best:
            best = joint
            best_epoch = epoch
            stale = 0
            export_metadata = {
                **run_metadata,
                "kind": "playground_koopman_export_v1",
                "source_path": str(args.data_dir.resolve()),
                "source_sha256": run_metadata["data_manifest_sha256"],
                "source_checkpoint_kind": "playground_jax_best",
                "best_epoch": best_epoch,
                "best_validation_joint_objective": best,
                "best_validation_rollout_normalized_mse": validation_host["rollout"],
                "best_spectral_radius_exact": exact_spectral_radius,
                "dataset_sha256": run_metadata["data_manifest_sha256"],
                "source_protocol_fingerprint": None,
            }
            _atomic_export(
                args.output_dir / "best.npz", params, center, scale, export_metadata
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= args.patience:
            print(
                f"epoch={epoch} train={train_host['total']:.6g} "
                f"val_joint={joint:.6g} val_rollout={validation_host['rollout']:.6g} "
                f"rho_est={validation_host['spectral_radius']:.6g} "
                f"rho_exact={exact_spectral_radius:.6g} best={best:.6g}",
                flush=True,
            )
        if stale >= args.patience:
            break
    run_metadata.update(
        completed=True,
        completed_epochs=epoch,
        best_epoch=best_epoch,
        best_validation_joint_objective=best,
        wall_time_seconds=time.time() - started,
    )
    _atomic_json(args.output_dir / "run.json", run_metadata)
    return run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lift-dim", type=int)
    parser.add_argument("--k-step", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=500000)
    parser.add_argument("--validation-windows", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--spectral-radius-limit", type=float, default=0.95)
    parser.add_argument("--stability-reference-dt", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    task = TASKS[args.task]
    if args.lift_dim is None:
        args.lift_dim = task.koopman_lift_dim
    if args.k_step is None:
        args.k_step = task.koopman_horizon_steps
    for name in (
        "lift_dim", "k_step", "batch_size", "max_windows", "validation_windows", "epochs", "patience"
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
