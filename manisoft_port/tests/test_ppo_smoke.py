import gymnasium as gym
import numpy as np
import torch

from antmaze_ac.envs.delta_action_wrapper import DeltaActionWrapper
from antmaze_ac.envs.process_vector_env import ProcessVectorEnv
from antmaze_ac.rl.delta_policy import DeltaPolicy
from antmaze_ac.rl.ppo import (
    collect_rollout,
    collect_vector_rollout,
    ppo_update,
)


class ShortEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def __init__(self):
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.full(3, self.steps / 10, dtype=np.float32),
            float(self.steps == 3),
            self.steps >= 5,
            False,
            {},
        )


def test_collect_and_sb3_aligned_ppo_update_smoke():
    torch.manual_seed(5)
    env = DeltaActionWrapper(ShortEnv())
    observation, _ = env.reset(seed=5)
    env._ppo_observation = observation
    policy = DeltaPolicy(
        5,
        2,
        torch.zeros(5),
        torch.ones(5),
        hidden_dims=(8, 8),
        log_std_init=0.0,
        activation="relu",
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rollout = collect_rollout(env, policy, 16, 0.99, 0.95, torch.device("cpu"))
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = ppo_update(
        policy,
        optimizer,
        rollout,
        update_epochs=2,
        minibatch_size=8,
        clip_range=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
        max_grad_norm=0.5,
    )
    assert set(metrics) == {
        "policy",
        "value",
        "entropy",
        "grad_norm",
        "approx_kl",
        "clip_fraction",
        "update_dare_retry_fraction",
        "update_dare_fallback_fraction",
        "ppo_optimizer_steps",
        "ppo_early_stopped",
        "ppo_early_stop_kl",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))
    assert len(rollout.episode_returns) == 3
    env.close()


def test_vector_rollout_batches_policy_and_isolates_episode_state():
    torch.manual_seed(7)
    envs = [DeltaActionWrapper(ShortEnv()) for _ in range(4)]
    for index, env in enumerate(envs):
        observation, _ = env.reset(seed=7 + index)
        env._ppo_observation = observation
    policy = DeltaPolicy(
        5,
        2,
        torch.zeros(5),
        torch.ones(5),
        hidden_dims=(8, 8),
        log_std_init=0.0,
        activation="relu",
    )
    rollout = collect_vector_rollout(
        envs,
        policy,
        steps=40,
        gamma=0.99,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    )
    assert rollout.observations.shape == (40, 5)
    assert rollout.actions.shape == (40, 2)
    assert rollout.returns.shape == (40,)
    assert rollout.advantages.shape == (40,)
    assert len(rollout.episode_returns) == 8
    assert np.all(rollout.episode_lengths == 5)
    assert all(env._ppo_episode_length == 0 for env in envs)
    for env in envs:
        env.close()


def test_vector_rollout_requires_divisible_transition_count():
    envs = [DeltaActionWrapper(ShortEnv()) for _ in range(3)]
    policy = DeltaPolicy(
        5,
        2,
        torch.zeros(5),
        torch.ones(5),
        hidden_dims=(8,),
        activation="relu",
    )
    with np.testing.assert_raises_regex(ValueError, "must be divisible"):
        collect_vector_rollout(
            envs,
            policy,
            steps=10,
            gamma=0.99,
            gae_lambda=0.95,
            device=torch.device("cpu"),
        )
    for env in envs:
        env.close()


def test_process_vector_rollout_matches_vector_contract():
    torch.manual_seed(17)
    envs = ProcessVectorEnv(
        [lambda: DeltaActionWrapper(ShortEnv()) for _ in range(4)]
    )
    envs.reset([17, 18, 19, 20])
    policy = DeltaPolicy(
        5,
        2,
        torch.zeros(5),
        torch.ones(5),
        hidden_dims=(8, 8),
        log_std_init=0.0,
        activation="relu",
    )
    try:
        rollout = collect_vector_rollout(
            envs,
            policy,
            steps=40,
            gamma=0.99,
            gae_lambda=0.95,
            device=torch.device("cpu"),
        )
        assert rollout.observations.shape == (40, 5)
        assert rollout.actions.shape == (40, 2)
        assert len(rollout.episode_returns) == 8
        assert np.all(rollout.episode_lengths == 5)
        assert np.all(envs.episode_lengths == 0)
    finally:
        envs.close()
