import gymnasium as gym
import numpy as np
import torch

from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.koopman.checkpoint import save_checkpoint, sha256
from antmaze_ac.koopman.history_model import HistoryDeepKoopman
from antmaze_ac.rl.critic import Critic
from antmaze_ac.rl.history_mlp_policy import HistoryMLPActor, HistoryMLPPolicy
from antmaze_ac.rl.serialization import (
    load_history_mlp_checkpoint,
    make_history_mlp_policy,
)
from antmaze_ac.rl.ppo import collect_rollout, ppo_update


class _ManiSoftShapeEnv(gym.Env):
    observation_space = gym.spaces.Box(
        -np.inf,
        np.inf,
        shape=(45,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(-0.3, 0.3, shape=(18,), dtype=np.float32)

    def __init__(self):
        self.target_tip = np.asarray([0.01, 0.02, 0.03], dtype=np.float32)
        self.state = np.zeros(45, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state.fill(0.0)
        self.steps = 0
        return self.state.copy(), {"distance": float(np.linalg.norm(self.target_tip))}

    def step(self, action):
        self.steps += 1
        self.state[:18] += np.asarray(action, dtype=np.float32)
        distance = float(np.linalg.norm(self.state[30:33] - self.target_tip))
        terminated = self.steps >= 4
        return self.state.copy(), -distance, terminated, False, {
            "distance": distance,
            "is_success": False,
        }


def _policy() -> HistoryMLPPolicy:
    state_dim = 45
    action_dim = 18
    history_steps = 10
    feature_dim = history_steps * (state_dim + action_dim) + 6
    actor = HistoryMLPActor(
        feature_dim,
        action_dim,
        hidden_dims=(32, 32),
        action_low=-0.30,
        action_high=0.30,
    )
    critic = Critic(feature_dim, hidden_dims=(32,), activation="tanh")
    return HistoryMLPPolicy(
        actor,
        critic,
        torch.zeros(state_dim),
        torch.ones(state_dim),
        state_dim=state_dim,
        action_dim=action_dim,
        history_steps=history_steps,
    )


def _observations(batch: int = 4) -> torch.Tensor:
    policy = _policy()
    observations = torch.zeros(batch, policy.observation_dim)
    observations[:, : policy.state_dim] = 0.1 * torch.randn(
        batch, policy.state_dim
    )
    action_start = (
        policy.state_dim + policy.history_steps * policy.state_dim
    )
    action_stop = policy.state_dim + policy.history_context_dim
    observations[:, action_start:action_stop] = 0.02 * torch.randn(
        batch,
        policy.history_steps * policy.action_dim,
    )
    observations[:, -3:] = torch.tensor([0.2, 0.3, 0.4])
    return observations


def test_history_mlp_45d_18d_h10_action_support_and_gradients():
    torch.manual_seed(23)
    policy = _policy()
    observations = _observations()
    assert policy.observation_dim == 678
    assert policy.feature_dim == 636

    output = policy(observations)
    assert output.mean.shape == (4, 18)
    assert output.value.shape == (4,)
    assert output.features.shape == (4, 636)
    actions = output.distribution.sample().detach()
    previous = policy.split_observation(observations).previous_action
    lower, upper = policy.actor.feasible_interval(previous)
    assert torch.all(output.mean >= lower)
    assert torch.all(output.mean <= upper)
    assert torch.all(actions >= lower)
    assert torch.all(actions <= upper)

    log_prob, entropy, values, evaluated = policy.evaluate_actions(
        observations,
        actions,
    )
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    loss = -log_prob.mean() - 0.01 * entropy.mean() + values.square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in policy.actor.parameters()
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in policy.critic.parameters()
    )
    assert evaluated.raw_mean.shape == (4, 18)


def test_history_mlp_checkpoint_round_trip(tmp_path):
    torch.manual_seed(29)
    koopman = HistoryDeepKoopman(
        state_dim=45,
        action_dim=18,
        lift_dim=4,
        hidden_dims=(8,),
        history_steps=10,
    )
    metadata_path = tmp_path / "history_koopman.pt"
    config = {
        "history_mlp_baseline": {
            "hidden_dims": [16],
            "activation": "tanh",
            "critic_hidden_dims": [16],
            "critic_activation": "tanh",
            "log_std_init": -0.5,
        }
    }
    normalizers = {
        "state": {
            "mean": torch.zeros(45),
            "std": torch.ones(45),
        }
    }
    save_checkpoint(
        metadata_path,
        koopman,
        epoch=0,
        best_validation=0.0,
        config=config,
        normalizers=normalizers,
        elapsed_seconds=0.0,
    )
    policy, _ = make_history_mlp_policy(metadata_path, torch.device("cpu"))
    observations = torch.zeros(2, policy.observation_dim)
    expected = policy.actor_mean(observations).detach()
    bc_path = tmp_path / "history_mlp_bc.pt"
    torch.save(
        {
            "method": "history_mlp_bc",
            "koopman_checkpoint": str(metadata_path),
            "koopman_checkpoint_sha256": sha256(metadata_path),
            "actor": policy.actor.state_dict(),
            "runtime": {
                "absolute_action_limit": 0.30,
                "max_delta": 0.001,
            },
        },
        bc_path,
    )
    restored, payload, _ = load_history_mlp_checkpoint(
        bc_path,
        torch.device("cpu"),
    )
    torch.testing.assert_close(restored.actor_mean(observations), expected)
    assert payload["method"] == "history_mlp_bc"


def test_history_mlp_ppo_smoke_with_h10_wrapper():
    torch.manual_seed(31)
    policy = _policy()
    env = HistoryContextTrackingWrapper(
        _ManiSoftShapeEnv(),
        history_steps=10,
        state_mean=np.zeros(45, dtype=np.float32),
        state_std=np.ones(45, dtype=np.float32),
    )
    observation, _ = env.reset(seed=31)
    env._ppo_observation = observation
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rollout = collect_rollout(
        env,
        policy,
        steps=8,
        gamma=0.99,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    )
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = ppo_update(
        policy,
        optimizer,
        rollout,
        update_epochs=1,
        minibatch_size=4,
        clip_range=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
        max_grad_norm=0.5,
        target_kl=0.1,
    )
    assert rollout.observations.shape == (8, 678)
    assert rollout.actions.shape == (8, 18)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, policy.parameters())
    )
    env.close()
