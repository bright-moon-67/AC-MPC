import math

import gymnasium as gym
import numpy as np
import torch

from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.koopman.checkpoint import save_checkpoint, sha256
from antmaze_ac.koopman.history_model import HistoryDeepKoopman
from antmaze_ac.rl.manisoft_ppo_policies import (
    load_manisoft_ppo_checkpoint,
    make_manisoft_ppo_policy,
)
from antmaze_ac.rl.ppo import collect_rollout, ppo_update


class _ThreeWaypointEnv(gym.Env):
    waypoint_count = 3
    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(45,), dtype=np.float32
    )
    action_space = gym.spaces.Box(-0.3, 0.3, shape=(3,), dtype=np.float32)

    def __init__(self):
        self.waypoints = np.asarray(
            [[0.1, 0.0, 0.0], [0.15, 0.05, 0.0], [0.2, 0.1, 0.05]],
            dtype=np.float32,
        )
        self.target_tip = self.waypoints[0].copy()
        self.active_waypoint_index = 0
        self.state = np.zeros(45, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state.fill(0.0)
        self.steps = 0
        self.active_waypoint_index = 0
        self.target_tip = self.waypoints[0].copy()
        return self.state.copy(), {"distance": 0.1}

    def step(self, action):
        self.steps += 1
        self.state[:3] += 0.01 * np.asarray(action, dtype=np.float32)
        distance = float(
            np.linalg.norm(self.state[30:33] - self.target_tip)
        )
        done = self.steps >= 4
        return self.state.copy(), -distance, done, False, {
            "distance": distance,
            "is_success": False,
            "waypoints_completed": 0,
        }


def _checkpoint(tmp_path):
    torch.manual_seed(4)
    model = HistoryDeepKoopman(
        state_dim=45,
        action_dim=3,
        lift_dim=2,
        hidden_dims=(8,),
        history_steps=2,
    )
    path = tmp_path / "koopman.pt"
    save_checkpoint(
        path,
        model,
        epoch=0,
        best_validation=0.0,
        config={},
        normalizers={
            "state": {"mean": torch.zeros(45), "std": torch.ones(45)}
        },
        elapsed_seconds=0.0,
    )
    return path


def _make(name, checkpoint, *, cost_parameterization="full", **kwargs):
    return make_manisoft_ppo_policy(
        name,
        checkpoint,
        torch.device("cpu"),
        mlp_hidden_dims=(16, 16),
        kmpc_hidden_dims=(12,),
        horizon=2,
        solver_iterations=4,
        kmpc_cost_parameterization=cost_parameterization,
        **kwargs,
    )[0]


def test_lifted_mlp_and_kmpc_have_aligned_inputs_and_gradients(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    mlp = _make("ppo_mlp", checkpoint)
    observations = torch.zeros(5, mlp.observation_dim)
    observations[:, -3:] = torch.tensor([1.0, 0.0, 0.0])
    assert isinstance(mlp.koopman, HistoryDeepKoopman)
    assert mlp.feature_dim == mlp.koopman.lifted_dim + 12
    mlp_output = mlp(observations)
    assert mlp_output.mean.shape == (5, 3)
    assert mlp_output.value.shape == (5,)
    mlp_actions = mlp_output.distribution.sample().detach()
    log_prob, entropy, values, _ = mlp.evaluate_actions(
        observations, mlp_actions
    )
    (-log_prob.mean() - 1e-4 * entropy.mean() + values.square().mean()).backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in mlp.actor.parameters()
    )
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in mlp.critic.parameters()
    )
    assert all(
        not p.requires_grad and p.grad is None
        for p in mlp.koopman.parameters()
    )

    kmpc = _make("ppo_kmpc", checkpoint)
    kmpc_output = kmpc(observations)
    assert kmpc_output.mean.shape == (5, 3)
    assert kmpc_output.mpc.action_sequence.shape == (5, 2, 3)
    torch.testing.assert_close(
        mlp_output.features[..., : kmpc.koopman.lifted_dim],
        kmpc_output.lifted_state,
    )
    torch.testing.assert_close(
        mlp_output.features[..., kmpc.koopman.lifted_dim :],
        kmpc_output.actor_context,
    )
    kmpc_actions = kmpc_output.distribution.sample().detach()
    log_prob, entropy, values, _ = kmpc.evaluate_actions(
        observations, kmpc_actions
    )
    (-log_prob.mean() - 1e-4 * entropy.mean() + values.square().mean()).backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in kmpc.actor.parameters()
    )
    assert all(not p.requires_grad and p.grad is None for p in kmpc.koopman.parameters())


def test_from_scratch_ppo_update_and_std_bounds(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    policy = _make("ppo_mlp", checkpoint)
    env = HistoryContextTrackingWrapper(
        _ThreeWaypointEnv(),
        history_steps=2,
        state_mean=np.zeros(45, dtype=np.float32),
        state_std=np.ones(45, dtype=np.float32),
    )
    observation, _ = env.reset(seed=9)
    env._ppo_observation = observation
    with torch.no_grad():
        policy.log_std.fill_(1.0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rollout = collect_rollout(
        env,
        policy,
        steps=8,
        gamma=0.99,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    )
    metrics = ppo_update(
        policy,
        optimizer,
        rollout,
        update_epochs=1,
        minibatch_size=4,
        clip_range=0.2,
        value_coefficient=0.5,
        entropy_coefficient=1e-4,
        max_grad_norm=0.5,
        target_kl=0.3,
        clip_value_loss=True,
        minimum_log_std=math.log(0.001),
        maximum_log_std=math.log(0.2),
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert torch.all(policy.log_std <= math.log(0.2) + 1e-7)
    assert rollout.action_bound.shape == (8,)
    assert rollout.applied_delta_action_l2.shape == (8,)
    env.close()


def _separated_optimizer_update(tmp_path, *, hard_multiplier):
    policy = _make("ppo_mlp", _checkpoint(tmp_path))
    policy.log_std.requires_grad_(False)
    env = HistoryContextTrackingWrapper(
        _ThreeWaypointEnv(),
        history_steps=2,
        state_mean=np.zeros(45, dtype=np.float32),
        state_std=np.ones(45, dtype=np.float32),
    )
    observation, _ = env.reset(seed=17)
    env._ppo_observation = observation
    rollout = collect_rollout(
        env,
        policy,
        steps=8,
        gamma=0.99,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    )
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=1e-3)
    actor_before = [p.detach().clone() for p in policy.actor.parameters()]
    critic_before = [p.detach().clone() for p in policy.critic.parameters()]
    metrics = ppo_update(
        policy,
        actor_optimizer,
        rollout,
        update_epochs=2,
        minibatch_size=4,
        clip_range=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
        max_grad_norm=0.5,
        target_kl=1e-12,
        critic_optimizer=critic_optimizer,
        kl_soft_stop_multiplier=1.5,
        kl_hard_rollback_multiplier=hard_multiplier,
    )
    env.close()
    return policy, actor_before, critic_before, metrics


def test_soft_kl_stop_keeps_actor_step_and_finishes_critic_updates(tmp_path):
    policy, actor_before, critic_before, metrics = _separated_optimizer_update(
        tmp_path,
        hard_multiplier=1e20,
    )
    assert metrics["ppo_kl_soft_stopped"] == 1.0
    assert metrics["ppo_kl_hard_rollbacks"] == 0.0
    assert metrics["ppo_actor_optimizer_steps"] == 1.0
    assert metrics["ppo_critic_optimizer_steps"] == 4.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, policy.actor.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, policy.critic.parameters())
    )


def test_hard_kl_rollback_restores_only_actor_and_trains_critic(tmp_path):
    policy, actor_before, critic_before, metrics = _separated_optimizer_update(
        tmp_path,
        hard_multiplier=3.0,
    )
    assert metrics["ppo_kl_hard_rollbacks"] == 1.0
    assert metrics["ppo_actor_optimizer_steps"] == 0.0
    assert metrics["ppo_critic_optimizer_steps"] == 4.0
    assert all(
        torch.equal(before, after)
        for before, after in zip(actor_before, policy.actor.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, policy.critic.parameters())
    )


def test_kmpc_rollout_uses_same_bounded_normalized_delta_for_ppo_and_env(
    tmp_path,
):
    checkpoint = _checkpoint(tmp_path)
    policy = _make("ppo_kmpc", checkpoint)
    env = HistoryContextTrackingWrapper(
        _ThreeWaypointEnv(),
        history_steps=2,
        state_mean=np.zeros(45, dtype=np.float32),
        state_std=np.ones(45, dtype=np.float32),
        max_delta=0.001,
    )
    observation, _ = env.reset(seed=11)
    env._ppo_observation = observation
    rollout = collect_rollout(
        env,
        policy,
        steps=8,
        gamma=0.99,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    )
    assert bool((rollout.actions.abs() <= 1.0 + 1e-6).all())
    assert float(rollout.applied_delta_action_abs_max.max()) <= 0.001 + 1e-7
    with torch.no_grad():
        recomputed = policy.evaluate_actions(
            rollout.observations, rollout.actions
        )[0]
    torch.testing.assert_close(recomputed, rollout.old_log_probs)
    env.close()


def test_structured_kmpc_policy_reduces_cost_map_and_backpropagates(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    full = _make("ppo_kmpc", checkpoint)
    structured = _make(
        "ppo_kmpc",
        checkpoint,
        cost_parameterization="structured",
    )
    observations = torch.zeros(5, structured.observation_dim)
    observations[:, -3] = 1.0
    full_output = full(observations)
    structured_output = structured(observations)
    torch.testing.assert_close(structured_output.mean, full_output.mean)
    assert structured.actor.network[-1].out_features == 5
    assert sum(p.numel() for p in structured.actor.parameters()) < sum(
        p.numel() for p in full.actor.parameters()
    )
    actions = structured_output.distribution.sample().detach()
    log_prob, entropy, values, _ = structured.evaluate_actions(
        observations,
        actions,
    )
    (-log_prob.mean() - 1e-4 * entropy.mean() + values.square().mean()).backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in structured.actor.parameters()
    )
    assert structured.cost_initialization == "structured_reference_weights_v1"


def test_structured_v2_uses_eleven_outputs_and_zero_velocity_reference(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    policy = _make(
        "ppo_kmpc",
        checkpoint,
        cost_parameterization="structured_v2",
    )
    observations = torch.zeros(5, policy.observation_dim)
    observations[:, :45] = torch.arange(45, dtype=torch.float32)
    observations[:, -12:-9] = torch.tensor([1.0, 2.0, 3.0])
    observations[:, -3] = 1.0
    (
        _,
        _,
        _,
        physical_reference,
        action_reference,
    ) = policy.features(observations)
    velocity_indices = torch.tensor(
        [
            *range(9, 15),
            *range(24, 30),
            *range(39, 45),
        ]
    )
    shape_indices = torch.tensor(
        [
            *range(0, 9),
            *range(15, 24),
            *range(33, 39),
        ]
    )
    torch.testing.assert_close(
        physical_reference[:, velocity_indices],
        torch.zeros(5, 18),
    )
    torch.testing.assert_close(
        physical_reference[:, shape_indices],
        observations[:, shape_indices],
    )
    torch.testing.assert_close(
        physical_reference[:, 30:33],
        torch.tensor([1.0, 2.0, 3.0]).expand(5, 3),
    )
    torch.testing.assert_close(
        action_reference,
        torch.zeros_like(action_reference),
    )

    output = policy(observations)
    assert policy.actor.network[-1].out_features == 11
    assert output.mpc.quadratic_diagonal.shape == (5, 2, 48)
    assert policy.cost_initialization == "structured_reference_groups_v2"
    actions = output.distribution.sample().detach()
    log_prob, entropy, values, _ = policy.evaluate_actions(
        observations,
        actions,
    )
    (
        -log_prob.mean()
        - 1e-4 * entropy.mean()
        + values.square().mean()
        + 1e-3 * output.mpc.quadratic_diagonal.mean()
    ).backward()
    assert any(
        parameter.grad is not None
        and float(parameter.grad.abs().sum()) > 0
        for parameter in policy.actor.parameters()
    )


def test_source_style_ablation_axes_are_independent(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    observations = torch.zeros(5, 45 + 2 * (45 + 3) + 12)
    observations[:, -3] = 1.0

    torch.manual_seed(901)
    baseline = _make(
        "ppo_kmpc",
        checkpoint,
        cost_parameterization="structured",
    )
    torch.manual_seed(901)
    implicit = _make(
        "ppo_kmpc",
        checkpoint,
        cost_parameterization="structured",
        kmpc_reference_mode="implicit",
    )
    assert implicit.cost_initialization == (
        "structured_q_implicit_stage_linear_zero_v1"
    )
    assert implicit.actor.reference_mode == "implicit"
    # The wider implicit-p actor consumes a matched RNG substream so its
    # critic is paired with the v15e baseline.
    for baseline_parameter, implicit_parameter in zip(
        baseline.critic.parameters(), implicit.critic.parameters()
    ):
        torch.testing.assert_close(baseline_parameter, implicit_parameter)
    assert implicit(observations).mean.shape == (5, 3)

    torch.manual_seed(902)
    terminal_on = _make(
        "ppo_kmpc",
        checkpoint,
        cost_parameterization="structured",
    )
    torch.manual_seed(902)
    terminal_off = _make(
        "ppo_kmpc",
        checkpoint,
        cost_parameterization="structured",
        structured_terminal_multiplier=False,
    )
    torch.testing.assert_close(
        terminal_on(observations).mean,
        terminal_off(observations).mean,
    )
    assert not terminal_off.actor.use_terminal_multiplier
    assert terminal_off.actor.network[-1].out_features == 5

    absolute = _make(
        "ppo_kmpc",
        checkpoint,
        cost_parameterization="structured",
        max_delta=None,
        initial_action_std=0.00225,
    )
    absolute_output = absolute(observations)
    assert absolute.actor.max_delta is None
    assert absolute.ACTION_DISTRIBUTION == "diagonal_normal_v1"
    assert float(absolute_output.mean.abs().max()) <= 0.3 + 1e-6


def test_comparison_checkpoint_round_trip(tmp_path):
    koopman_path = _checkpoint(tmp_path)
    cases = (
        ("ppo_mlp", "full"),
        ("ppo_kmpc", "full"),
        ("ppo_kmpc", "structured"),
        ("ppo_kmpc", "structured_v2"),
    )
    for actor_name, cost_parameterization in cases:
        policy = _make(
            actor_name,
            koopman_path,
            cost_parameterization=cost_parameterization,
        )
        observations = torch.zeros(2, policy.observation_dim)
        observations[:, -3] = 1.0
        expected = policy(observations).mean.detach()
        runtime = {
            "absolute_action_limit": 0.3,
            "progress_reward_scale": 1.0,
            "initial_action_std": 0.015,
            "waypoint_count": 3,
            "mlp_hidden_dims": [16, 16],
            "kmpc_hidden_dims": [12],
            "horizon": 2,
            "solver_iterations": 4,
            "quadratic_log_scale": 1.5,
            "linear_scale": 10.0,
            "action_quadratic_scale": 1.0,
            "tip_weight": 1.0,
            "max_delta": 0.001 if actor_name == "ppo_kmpc" else None,
            "kmpc_cost_parameterization": cost_parameterization,
            "structured_log_scale": math.log(2.0),
            "structured_shape_weight": 1e-3,
            "structured_linear_velocity_weight": 1e-2,
            "structured_angular_velocity_weight": 1e-2,
            "structured_normalized_delta_weight": 1e-4,
        }
        path = tmp_path / f"{actor_name}_{cost_parameterization}.pt"
        torch.save(
            {
                "method": "manisoft_ppo_from_scratch",
                "actor_name": actor_name,
                "koopman_checkpoint": str(koopman_path),
                "koopman_checkpoint_sha256": sha256(koopman_path),
                "runtime": runtime,
                "policy": policy.state_dict(),
            },
            path,
        )
        restored, payload, _ = load_manisoft_ppo_checkpoint(
            path, torch.device("cpu")
        )
        torch.testing.assert_close(restored(observations).mean, expected)
        assert payload["actor_name"] == actor_name


def test_ablation_checkpoint_round_trip(tmp_path):
    koopman_path = _checkpoint(tmp_path)
    cases = (
        (
            "implicit",
            {"kmpc_reference_mode": "implicit"},
            {"kmpc_reference_mode": "implicit", "max_delta": 0.001},
        ),
        (
            "absolute",
            {"max_delta": None, "initial_action_std": 0.00225},
            {"max_delta": None, "initial_action_std": 0.00225},
        ),
        (
            "no_terminal",
            {"structured_terminal_multiplier": False},
            {"structured_terminal_multiplier": False, "max_delta": 0.001},
        ),
    )
    for name, factory_kwargs, runtime_overrides in cases:
        policy = _make(
            "ppo_kmpc",
            koopman_path,
            cost_parameterization="structured",
            **factory_kwargs,
        )
        observations = torch.zeros(2, policy.observation_dim)
        observations[:, -3] = 1.0
        expected = policy(observations).mean.detach()
        runtime = {
            "absolute_action_limit": 0.3,
            "progress_reward_scale": 1.0,
            "initial_action_std": 0.015,
            "waypoint_count": 3,
            "mlp_hidden_dims": [16, 16],
            "kmpc_hidden_dims": [12],
            "horizon": 2,
            "solver_iterations": 4,
            "quadratic_log_scale": 1.5,
            "linear_scale": 10.0,
            "action_quadratic_scale": 1.0,
            "tip_weight": 1.0,
            "max_delta": 0.001,
            "kmpc_cost_parameterization": "structured",
            "structured_log_scale": math.log(2.0),
            **runtime_overrides,
        }
        path = tmp_path / f"ablation_{name}.pt"
        torch.save(
            {
                "method": "manisoft_ppo_from_scratch",
                "actor_name": "ppo_kmpc",
                "koopman_checkpoint": str(koopman_path),
                "koopman_checkpoint_sha256": sha256(koopman_path),
                "runtime": runtime,
                "policy": policy.state_dict(),
            },
            path,
        )
        restored, _, _ = load_manisoft_ppo_checkpoint(
            path, torch.device("cpu")
        )
        torch.testing.assert_close(restored(observations).mean, expected)
