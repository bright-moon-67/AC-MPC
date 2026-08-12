from pathlib import Path
import math

import numpy as np
import pytest
import torch

from antmaze_ac.koopman.model import DeepKoopman
from experiments.dmc.actors import (
    ACTOR_TYPES,
    HAIKU_DEFAULT_LINEAR_INITIALIZATION,
    ActorConfig,
    actor_config_from_checkpoint,
    actor_mean,
    build_actor,
    checkpoint_protocol_fingerprint,
    load_koopman,
    normalizer_arrays,
)
from experiments.dmc.tasks.registry import get_task_spec


def _stable_cartpole_koopman() -> DeepKoopman:
    spec = get_task_spec("cartpole_swingup")
    model = DeepKoopman(spec.obs_dim, spec.action_dim, lift_dim=4, hidden_dims=(8,))
    with torch.no_grad():
        model.A.copy_(0.9 * torch.eye(model.lifted_dim))
        model.B.normal_(std=0.01)
    return model


def test_shared_actor_factory_outputs_finite_bounded_actions():
    model = _stable_cartpole_koopman().freeze_dynamics()
    config = ActorConfig(kmpc_horizon=3, kmpc_solver_iterations=2)
    state = torch.zeros(2, model.state_dim)
    lifted = model.lift(state)
    for actor_type in ACTOR_TYPES:
        actor = build_actor(
            actor_type,
            "cartpole_swingup",
            torch.device("cpu"),
            koopman=model if actor_type != "PPO" else None,
            config=config,
        )
        action = actor_mean(
            actor_type,
            actor,
            state,
            lifted if actor_type != "PPO" else None,
        )
        assert action.shape == (2, 1)
        assert torch.isfinite(action).all()
        assert (action.abs() <= 1.0 + 1e-6).all()


def test_ac_mpc_mpve_actor_is_exactly_the_kmpc_architecture():
    model = _stable_cartpole_koopman().freeze_dynamics()
    config = ActorConfig(kmpc_horizon=3, kmpc_solver_iterations=2)
    torch.manual_seed(11)
    kmpc = build_actor(
        "KMPC",
        "cartpole_swingup",
        torch.device("cpu"),
        koopman=model,
        config=config,
    )
    torch.manual_seed(11)
    mpve = build_actor(
        "AC-MPC-MPVE",
        "cartpole_swingup",
        torch.device("cpu"),
        koopman=model,
        config=config,
    )
    assert type(mpve) is type(kmpc)
    for name, value in kmpc.state_dict().items():
        assert torch.equal(value, mpve.state_dict()[name])


def test_koopman_best_checkpoint_round_trip(tmp_path: Path):
    model = _stable_cartpole_koopman()
    path = tmp_path / "best.pt"
    torch.save(
        {
            "architecture": model.architecture(),
            "model_state": model.state_dict(),
            "state_kind": "cartpole_swingup",
            "normalizer": {
                "center": torch.zeros(model.state_dim),
                "scale": torch.ones(model.state_dim),
            },
        },
        path,
    )
    loaded, payload = load_koopman(
        path, "cartpole_swingup", torch.device("cpu")
    )
    center, scale = normalizer_arrays(payload, "cartpole_swingup")
    assert loaded.state_dim == model.state_dim
    assert np.array_equal(center, np.zeros(model.state_dim, dtype=np.float32))
    assert np.array_equal(scale, np.ones(model.state_dim, dtype=np.float32))
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_actor_config_checkpoint_contract():
    config = ActorConfig(hidden_dim=64, kmpc_horizon=7)
    assert actor_config_from_checkpoint({"actor_config": config.to_dict()}) == config


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"ppo_hidden_layers": 2}, "ppo_hidden_layers=3"),
        ({"ppo_activation": "tanh"}, "ppo_activation='relu'"),
        ({"ppo_distribution": "normal"}, "tanh-squashed"),
    ],
)
def test_primary_ppo_architecture_contract_is_fail_closed(override, message):
    with pytest.raises(ValueError, match=message):
        ActorConfig(**override).validate()


def test_plain_ppo_torso_uses_haiku_default_linear_initialization():
    actor = build_actor(
        "PPO",
        "cartpole_swingup",
        torch.device("cpu"),
        config=ActorConfig(ppo_hidden_dim=32),
    )
    assert HAIKU_DEFAULT_LINEAR_INITIALIZATION.endswith("bounds_2sigma_v1")
    for layer in (
        module for module in actor.network if isinstance(module, torch.nn.Linear)
    ):
        stddev = 1.0 / math.sqrt(layer.in_features)
        assert float(layer.weight.abs().max()) <= 2.0 * stddev
        assert torch.equal(layer.bias, torch.zeros_like(layer.bias))
    head_bound = math.sqrt(3.0 / actor.hidden_dim)
    for head in (actor.loc_layer, actor.scale_layer):
        assert float(head.weight.abs().max()) <= head_bound
        assert torch.equal(head.bias, torch.zeros_like(head.bias))


def test_dmc_koopman_protocol_identity_is_fail_closed():
    with pytest.raises(ValueError, match="protocol identity"):
        checkpoint_protocol_fingerprint(
            {
                "kind": "dmc_k_step_koopman",
                "state_kind": "cartpole_swingup",
            }
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"hidden_dim": True},
        {"kmpc_horizon": 2.5},
        {"action_limit": float("nan")},
    ],
)
def test_actor_config_rejects_ambiguous_or_nonfinite_values(updates):
    values = ActorConfig().to_dict()
    values.update(updates)
    with pytest.raises(ValueError):
        ActorConfig.from_mapping(values)
