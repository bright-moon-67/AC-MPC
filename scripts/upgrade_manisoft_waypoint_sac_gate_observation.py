#!/usr/bin/env python
"""Expand a legacy 70-D waypoint SAC checkpoint to the optional 74-D mode.

The original 45-D ManiSoft physical state and every legacy observation index
remain unchanged. Four new columns are appended. Their first-layer weights are
zero-initialized, so the migrated deterministic actor is exactly the source
actor until fine-tuning learns to use the new gate/prior semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from antmaze_ac.envs.manisoft_waypoint_sac_env import (
    MANISOFT_WAYPOINT_SAC_GATE_OBSERVATION_DIM,
    MANISOFT_WAYPOINT_SAC_OBSERVATION_DIM,
)
from antmaze_ac.koopman.checkpoint import sha256


class _SpaceOnlyEnv(gym.Env):
    def __init__(self, observation_dim: int, action_space: gym.Space) -> None:
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(observation_dim,), dtype=np.float32
        )
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        del action
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


def _expanded_actor_state(
    state: dict[str, torch.Tensor], old_dim: int, new_dim: int
) -> dict[str, torch.Tensor]:
    expanded = {key: value.detach().clone() for key, value in state.items()}
    key = "latent_pi.0.weight"
    old = state[key]
    if old.shape[1] != old_dim:
        raise ValueError(f"unexpected actor input shape: {tuple(old.shape)}")
    weight = old.new_zeros((old.shape[0], new_dim))
    weight[:, :old_dim] = old
    expanded[key] = weight
    return expanded


def _expanded_critic_state(
    state: dict[str, torch.Tensor], old_dim: int, new_dim: int, action_dim: int
) -> dict[str, torch.Tensor]:
    expanded = {key: value.detach().clone() for key, value in state.items()}
    for key in ("qf0.0.weight", "qf1.0.weight"):
        old = state[key]
        if old.shape[1] != old_dim + action_dim:
            raise ValueError(f"unexpected critic input shape for {key}: {old.shape}")
        weight = old.new_zeros((old.shape[0], new_dim + action_dim))
        weight[:, :old_dim] = old[:, :old_dim]
        weight[:, new_dim:] = old[:, old_dim:]
        expanded[key] = weight
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-vec-normalize", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_model).expanduser().resolve()
    source_vec_path = Path(args.source_vec_normalize).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source_path.is_file() or not source_vec_path.is_file():
        raise FileNotFoundError("source model and VecNormalize files are required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    source = SAC.load(str(source_path), device=args.device)
    old_dim = int(source.observation_space.shape[0])
    new_dim = MANISOFT_WAYPOINT_SAC_GATE_OBSERVATION_DIM
    if old_dim != MANISOFT_WAYPOINT_SAC_OBSERVATION_DIM:
        raise ValueError(f"source observation dimension must be 70, got {old_dim}")
    action_dim = int(source.action_space.shape[0])
    if action_dim != 2:
        raise ValueError("gate74 migration currently requires the compact 2-D action")

    old_dummy = DummyVecEnv(
        [lambda: _SpaceOnlyEnv(old_dim, source.action_space)]
    )
    source_vec = VecNormalize.load(str(source_vec_path), old_dummy)
    new_dummy = DummyVecEnv(
        [lambda: _SpaceOnlyEnv(new_dim, source.action_space)]
    )
    target_vec = VecNormalize(
        new_dummy,
        training=source_vec.training,
        norm_obs=source_vec.norm_obs,
        norm_reward=source_vec.norm_reward,
        clip_obs=source_vec.clip_obs,
        clip_reward=source_vec.clip_reward,
        gamma=source_vec.gamma,
        epsilon=source_vec.epsilon,
    )
    target_vec.obs_rms.mean[:old_dim] = source_vec.obs_rms.mean
    target_vec.obs_rms.var[:old_dim] = source_vec.obs_rms.var
    target_vec.obs_rms.mean[old_dim:] = 0.0
    target_vec.obs_rms.var[old_dim:] = 1.0
    target_vec.obs_rms.count = source_vec.obs_rms.count
    target_vec.ret_rms.mean = source_vec.ret_rms.mean
    target_vec.ret_rms.var = source_vec.ret_rms.var
    target_vec.ret_rms.count = source_vec.ret_rms.count

    target = SAC(
        "MlpPolicy",
        target_vec,
        learning_rate=float(source.learning_rate),
        buffer_size=int(source.buffer_size),
        learning_starts=int(source.learning_starts),
        batch_size=int(source.batch_size),
        tau=float(source.tau),
        gamma=float(source.gamma),
        train_freq=int(source.train_freq.frequency),
        gradient_steps=int(source.gradient_steps),
        ent_coef=source.ent_coef,
        target_entropy=float(source.target_entropy),
        policy_kwargs=dict(source.policy_kwargs),
        device=args.device,
        seed=0,
        verbose=0,
    )
    target.actor.load_state_dict(
        _expanded_actor_state(source.actor.state_dict(), old_dim, new_dim)
    )
    target.critic.load_state_dict(
        _expanded_critic_state(
            source.critic.state_dict(), old_dim, new_dim, action_dim
        )
    )
    target.critic_target.load_state_dict(
        _expanded_critic_state(
            source.critic_target.state_dict(), old_dim, new_dim, action_dim
        )
    )
    if source.log_ent_coef is not None and target.log_ent_coef is not None:
        target.log_ent_coef.data.copy_(source.log_ent_coef.data)
    target.num_timesteps = int(source.num_timesteps)
    target._n_updates = int(source._n_updates)

    # The four appended actor columns are exactly zero, so deterministic raw
    # actions must be bit-level close before the migrated model is accepted.
    rng = np.random.default_rng(20260823)
    old_observation = rng.normal(size=(128, old_dim)).astype(np.float32)
    new_observation = np.concatenate(
        (old_observation, rng.normal(size=(128, new_dim - old_dim))), axis=1
    ).astype(np.float32)
    old_action, _ = source.predict(old_observation, deterministic=True)
    new_action, _ = target.predict(new_observation, deterministic=True)
    maximum_action_error = float(np.max(np.abs(old_action - new_action)))
    if maximum_action_error > 1e-6:
        raise RuntimeError(
            "expanded actor changed legacy behavior: "
            f"max action error {maximum_action_error:.3e}"
        )

    model_output = output / "sac_gate74_initialized.zip"
    vec_output = output / "vecnormalize_gate74.pkl"
    target.save(str(model_output))
    target_vec.save(str(vec_output))
    report = {
        "kind": "manisoft_waypoint_sac_observation_upgrade",
        "physical_state_dimension": 45,
        "source_observation_dimension": old_dim,
        "target_observation_dimension": new_dim,
        "new_feature_order": [
            "current_cartesian_prior_action_x",
            "current_cartesian_prior_action_y",
            "effective_cartesian_prior_weight",
            "normalized_internal_waypoint_capture_error",
        ],
        "new_first_layer_weights_initialized_to_zero": True,
        "maximum_deterministic_action_error": maximum_action_error,
        "source_model": str(source_path),
        "source_model_sha256": sha256(source_path),
        "source_vecnormalize": str(source_vec_path),
        "source_vecnormalize_sha256": sha256(source_vec_path),
        "model": str(model_output),
        "model_sha256": sha256(model_output),
        "vecnormalize": str(vec_output),
        "vecnormalize_sha256": sha256(vec_output),
    }
    (output / "upgrade_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    source_vec.close()
    target_vec.close()


if __name__ == "__main__":
    main()
