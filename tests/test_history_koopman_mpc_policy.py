import gymnasium as gym
import numpy as np
import torch

from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.koopman.history_model import HistoryDeepKoopman
from antmaze_ac.rl.critic import Critic
from antmaze_ac.rl.history_koopman_mpc_policy import HistoryKoopmanMPCPolicy
from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor


class _HistoryEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(5,), dtype=np.float32)
    action_space = gym.spaces.Box(-0.3, 0.3, shape=(2,), dtype=np.float32)

    def __init__(self):
        self.target_tip = np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
        self.state = np.zeros(5, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state.fill(0.0)
        return self.state.copy(), {"target_tip": self.target_tip.copy()}

    def step(self, action):
        self.state[:2] += np.asarray(action, dtype=np.float32)
        return self.state.copy(), 0.0, False, False, {}


class _ThreeWaypointHistoryEnv(_HistoryEnv):
    waypoint_count = 3

    def __init__(self):
        super().__init__()
        self.waypoints = np.asarray(
            [[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.3, 0.2, 0.1]],
            dtype=np.float32,
        )
        self.active_waypoint_index = 0
        self.target_tip = self.waypoints[0].copy()

    def step(self, action):
        self.active_waypoint_index = min(self.active_waypoint_index + 1, 2)
        self.target_tip = self.waypoints[self.active_waypoint_index].copy()
        return super().step(action)


def test_history_wrapper_alignment_and_absolute_action_constraints():
    wrapped = HistoryContextTrackingWrapper(
        _HistoryEnv(),
        history_steps=3,
        state_mean=np.zeros(5),
        state_std=np.ones(5),
        tip_indices=(0, 1, 2),
    )
    observation, _ = wrapped.reset(seed=3)
    assert observation.shape == (5 + 3 * (5 + 2) + 3,)
    context = observation[5:-3]
    np.testing.assert_allclose(context, 0.0)

    following, _, _, _, info = wrapped.step(np.asarray([0.2, -0.2]))
    np.testing.assert_allclose(info["applied_action"], [0.2, -0.2])
    following_context = following[5:-3]
    action_context = following_context[3 * 5 :].reshape(3, 2)
    np.testing.assert_allclose(action_context[-1], [0.2, -0.2])
    np.testing.assert_allclose(following[-3:], wrapped.target_tip)


def test_history_wrapper_exposes_all_waypoints_and_active_stage():
    wrapped = HistoryContextTrackingWrapper(
        _ThreeWaypointHistoryEnv(),
        history_steps=2,
        state_mean=np.zeros(5),
        state_std=np.ones(5),
        tip_indices=(0, 1, 2),
    )
    observation, _ = wrapped.reset(seed=3)
    assert observation.shape == (5 + 2 * (5 + 2) + 12,)
    np.testing.assert_allclose(
        observation[-12:-3], wrapped.env.waypoints.reshape(-1)
    )
    np.testing.assert_allclose(observation[-3:], [1.0, 0.0, 0.0])

    following, _, _, _, _ = wrapped.step(np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(following[-3:], [0.0, 1.0, 0.0])


def test_history_mpc_policy_reconstructs_actor_and_critic_gradients():
    torch.manual_seed(17)
    koopman = HistoryDeepKoopman(
        state_dim=5,
        action_dim=2,
        lift_dim=2,
        hidden_dims=(8,),
        history_steps=3,
    )
    with torch.no_grad():
        koopman.A.copy_(torch.eye(7) * 0.85)
        koopman.B.fill_(0.05)
    actor = KoopmanMPCActor(
        koopman.A,
        koopman.B,
        koopman.C,
        horizon=3,
        context_dim=6,
        hidden_dims=(12,),
        action_low=-0.3,
        action_high=0.3,
        solver_iterations=5,
    )
    critic = Critic(koopman.lifted_dim + 6, hidden_dims=(12,))
    policy = HistoryKoopmanMPCPolicy(
        koopman,
        actor,
        critic,
        torch.zeros(5),
        torch.ones(5),
        tip_indices=(0, 1, 2),
    )
    observation_dim = 5 + koopman.context_dim + 3
    observations = torch.randn(4, observation_dim)
    # The action part of the history must be within the configured support.
    action_context_start = 5 + koopman.history_steps * koopman.state_dim
    observations[:, action_context_start : 5 + koopman.context_dim] *= 0.05
    output = policy(observations)
    assert output.mean.shape == (4, 2)
    assert output.value.shape == (4,)
    assert output.mpc.action_sequence.shape == (4, 3, 2)
    actions = output.distribution.sample().detach()
    assert actions.shape == (4, 2)
    assert torch.isfinite(actions).all()
    log_prob, entropy, values, _ = policy.evaluate_actions(observations, actions)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    loss = -log_prob.mean() - 0.01 * entropy.mean() + values.square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in actor.parameters()
    )
    assert all(parameter.grad is None for parameter in koopman.parameters())


def test_history_mpc_policy_three_waypoint_context():
    torch.manual_seed(23)
    koopman = HistoryDeepKoopman(
        state_dim=5,
        action_dim=2,
        lift_dim=2,
        hidden_dims=(8,),
        history_steps=2,
    )
    actor = KoopmanMPCActor(
        koopman.A,
        koopman.B,
        koopman.C,
        horizon=2,
        context_dim=12,
        hidden_dims=(12,),
        action_low=-0.3,
        action_high=0.3,
        solver_iterations=3,
    )
    critic = Critic(koopman.lifted_dim + 12, hidden_dims=(12,))
    policy = HistoryKoopmanMPCPolicy(
        koopman,
        actor,
        critic,
        torch.zeros(5),
        torch.ones(5),
        waypoint_count=3,
        tip_indices=(0, 1, 2),
    )
    observations = torch.zeros(2, policy.observation_dim)
    observations[:, -12:-3] = torch.arange(9, dtype=torch.float32)
    observations[:, -3:] = torch.tensor([0.0, 1.0, 0.0])
    split, _, actor_context, physical_reference, action_reference = policy.features(
        observations
    )
    assert split.task_context.shape == (2, 12)
    assert actor_context.shape == (2, 12)
    torch.testing.assert_close(actor_context[:, -3:], observations[:, -3:])
    # Stage one selects the second waypoint.  Non-tip coordinates retain the
    # current normalized state (zero in this fixture).
    torch.testing.assert_close(
        physical_reference[:, :3],
        torch.tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]),
    )
    torch.testing.assert_close(physical_reference[:, 3:], torch.zeros(2, 2))
    torch.testing.assert_close(action_reference, torch.zeros(2, 2))
    output = policy(observations)
    assert output.mean.shape == (2, 2)
