import copy
import json
import subprocess
from pathlib import Path

import torch

from antmaze_ac.koopman.model import DeepKoopman
from antmaze_ac.rl.ac_koopman_policy import KoopmanLQRPolicy
from antmaze_ac.rl.cost_actor import CostActor
from antmaze_ac.rl.critic import Critic
from antmaze_ac.rl.td3_bc import (
    OfflineTransitionBatch,
    TD3BCTrainer,
    TwinActionValueCritic,
    offline_validation_metrics,
)
from scripts.train_td3_bc import (
    environment_evaluation_due,
    reconcile_environment_evaluations,
    run_environment_evaluation,
)


def make_system():
    torch.manual_seed(9)
    state_dim, action_dim = 3, 1
    koopman = DeepKoopman(state_dim, action_dim, lift_dim=1, hidden_dims=(4,))
    with torch.no_grad():
        koopman.A.copy_(torch.eye(4) * 0.7)
        koopman.B.fill_(0.2)
    policy = KoopmanLQRPolicy(
        koopman,
        CostActor(
            state_dim,
            action_dim,
            hidden_dims=(8,),
            q_max=2.0,
            p_max=1.0,
        ),
        Critic(state_dim, hidden_dims=(8,)),
        torch.zeros(state_dim),
        torch.ones(state_dim),
        dare_tolerance=1e-8,
        dare_max_iterations=100,
        mean_action_limit=2.0,
    )
    policy.requires_grad_(False)
    policy.actor.requires_grad_(True)
    target_policy = copy.deepcopy(policy)
    target_policy.requires_grad_(False)
    critic = TwinActionValueCritic(
        state_dim,
        action_dim,
        torch.zeros(state_dim),
        torch.ones(state_dim),
        hidden_dims=(8,),
        action_scale=2.0,
    )
    target_critic = copy.deepcopy(critic)
    target_critic.requires_grad_(False)
    trainer = TD3BCTrainer(
        policy,
        target_policy,
        critic,
        target_critic,
        torch.optim.Adam(policy.actor.parameters(), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        discount=0.99,
        tau=0.01,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_frequency=2,
        alpha=2.5,
        bc_weight=1.0,
        bc_warmup_steps=10,
        max_delta_action=2.0,
        reward_scale=1.0,
        reward_bias=0.0,
        max_grad_norm=1.0,
    )
    return policy, critic, trainer


def test_td3_bc_update_preserves_dare_actor_and_action_support():
    policy, critic, trainer = make_system()
    batch = OfflineTransitionBatch(
        state=torch.randn(8, 3),
        action=torch.empty(8, 1).uniform_(-2.0, 2.0),
        next_state=torch.randn(8, 3),
        reward=torch.tensor([0, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float32),
        done=torch.tensor([0, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float32),
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.actor.named_parameters()
    }
    warmup_metrics = trainer.update(batch, gradient_step=2)
    assert warmup_metrics["actor_updated"] == 1
    assert warmup_metrics["td3_bc_lambda"] == 0.0
    assert warmup_metrics["behavior_cloning_loss"] is not None
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in policy.actor.named_parameters()
    )
    output = policy(batch.state)
    assert torch.max(torch.abs(output.mean)) <= 2.0
    assert torch.isfinite(output.mean).all()
    assert torch.all(output.solver_valid)

    td3_metrics = trainer.update(batch, gradient_step=12)
    assert td3_metrics["actor_updated"] == 1
    assert float(td3_metrics["td3_bc_lambda"]) > 0.0
    validation = offline_validation_metrics(policy, critic, batch)
    assert validation["behavior_cloning_loss"] >= 0.0
    assert validation["dare_fallback_fraction"] == 0.0


def test_environment_evaluation_schedule_includes_near_initial_policy():
    assert environment_evaluation_due(1, 2500)
    assert environment_evaluation_due(2500, 2500)
    assert not environment_evaluation_due(2499, 2500)
    assert not environment_evaluation_due(2500, 0)


def test_periodic_environment_evaluation_writes_paths_and_trend(
    tmp_path,
    monkeypatch,
):
    project_root = tmp_path / "project"
    (project_root / "scripts").mkdir(parents=True)
    output = tmp_path / "run"
    checkpoint_payload = {
        "format_version": 1,
        "method": "td3_bc_koopman_lqr",
        "policy": {"weight": torch.ones(2)},
        "koopman_checkpoint": "/tmp/koopman.pt",
        "koopman_checkpoint_sha256": "koopman-digest",
        "gradient_step": 1,
        "seed": 0,
        "elapsed_seconds": 12.0,
        "best_validation_behavior_cloning_loss": 0.5,
        "config": {"experiment": {"env_id": "antmaze-umaze-v2"}},
        "runtime": {"max_delta_action": 2.0},
    }

    def fake_run(command, **kwargs):
        del kwargs
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        report = Path(command[command.index("--output") + 1])
        plot = Path(command[command.index("--path-plot-output") + 1])
        episodes = int(command[command.index("--episodes") + 1])
        from antmaze_ac.koopman.checkpoint import sha256

        plot.write_bytes(b"png")
        plot.with_suffix(".npz").write_bytes(b"npz")
        report.write_text(
            json.dumps(
                {
                    "resolved_backend": "legacy",
                    "episodes": episodes,
                    "checkpoint_sha256": sha256(checkpoint),
                    "success_mean": 0.2,
                    "return_mean": 0.2,
                    "d4rl_normalized_score": 20.0,
                    "goal_progress_fraction_mean": 0.4,
                    "minimum_goal_distance_mean": 1.3,
                    "final_goal_distance_mean": 1.8,
                    "dare_failure_count": 0,
                    "dare_retry_count": 0,
                    "closed_loop_spectral_radius_max": 0.99,
                    "path_plot_png": str(plot),
                    "path_data_npz": str(plot.with_suffix(".npz")),
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.train_td3_bc.subprocess.run", fake_run)
    first = run_environment_evaluation(
        project_root=project_root,
        output=output,
        gradient_step=1,
        training_elapsed_seconds=12.0,
        checkpoint_payload=checkpoint_payload,
        episodes=5,
        plot_paths=5,
        device="cpu",
        seed_offset=100_000,
        timeout_seconds=30.0,
    )
    checkpoint_payload["gradient_step"] = 2500
    second = run_environment_evaluation(
        project_root=project_root,
        output=output,
        gradient_step=2500,
        training_elapsed_seconds=3600.0,
        checkpoint_payload=checkpoint_payload,
        episodes=5,
        plot_paths=5,
        device="cpu",
        seed_offset=100_000,
        timeout_seconds=30.0,
    )

    evaluation_root = output / "periodic_evaluation"
    history = [
        json.loads(line)
        for line in (evaluation_root / "history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert first["status"] == second["status"] == "ok"
    assert [row["gradient_step"] for row in history] == [1, 2500]
    assert (evaluation_root / "trend.png").exists()
    assert (evaluation_root / "step_00002500_legacy_5ep_paths.png").exists()
    assert (evaluation_root / "step_00002500_legacy_5ep_paths.npz").exists()

    removed = reconcile_environment_evaluations(evaluation_root, gradient_step=1)
    assert removed == 4
    assert not (evaluation_root / "step_00002500.pt").exists()
