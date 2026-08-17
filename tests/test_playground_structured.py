from __future__ import annotations

from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
jp = pytest.importorskip("jax.numpy")
pytest.importorskip("brax")

from brax.training import types
from brax.training.agents.ppo.losses import PPONetworkParams

from experiments.playground.koopman import (
    KoopmanParameters,
    cartpole_exact_reward,
    humanoid_run_exact_reward,
    load_export,
    reacher_hard_exact_reward,
)
from experiments.playground.mpve import make_mpve_ppo_loss
from experiments.playground.run_structured_after_koopman import (
    _child_command,
    _wait_for_koopman,
)
from experiments.playground.structured_networks import make_structured_ppo_networks


EXPORT = Path("runs/playground/koopman/cartpole_swingup/current_best.npz")


def _identity_parameters(state_dim: int, action_dim: int) -> KoopmanParameters:
    return KoopmanParameters(
        A=jp.eye(state_dim),
        B=jp.zeros((state_dim, action_dim)),
        C=jp.eye(state_dim),
        center=jp.zeros((state_dim,)),
        scale=jp.ones((state_dim,)),
        encoder_weights=(),
        encoder_biases=(),
        reward_weights=(),
        reward_biases=(),
    )


@pytest.fixture(scope="module")
def export_path() -> Path:
    if not EXPORT.is_file():
        pytest.skip("Local Cartpole Koopman export is unavailable")
    return EXPORT


def _network(method: str, export_path: Path):
    return make_structured_ppo_networks(
        (5,),
        1,
        lambda observation, unused: observation,
        method=method,
        koopman_path=str(export_path),
    )


def test_kmpc_and_mpve_share_controller_initialization(export_path: Path) -> None:
    kmpc = _network("KMPC", export_path)
    mpve = _network("AC-MPC-MPVE", export_path)
    key = jax.random.PRNGKey(7)
    kmpc_params = kmpc.policy_network.init(key)
    mpve_params = mpve.policy_network.init(key)
    assert all(
        bool(equal)
        for equal in jax.tree.leaves(
            jax.tree.map(
                lambda left, right: jp.array_equal(left, right),
                kmpc_params,
                mpve_params,
            )
        )
    )
    observation = jp.ones((4, 5), dtype=jp.float32)
    kmpc_logits = kmpc.policy_network.apply(None, kmpc_params, observation)
    mpve_logits = mpve.policy_network.apply(None, mpve_params, observation)
    assert jp.array_equal(kmpc_logits, mpve_logits[..., :2])


def test_projected_exact_reward_is_bounded(export_path: Path) -> None:
    parameters, _ = load_export(export_path)
    predicted = jp.asarray(
        [[0.0, 100.0, -4.0, 0.0, 0.0], [0.0, -100.0, -4.0, 0.0, 0.0]]
    )
    reward = cartpole_exact_reward(parameters, jp.zeros((2, 1)), predicted)
    assert bool(jp.all(reward >= 0.0))
    assert bool(jp.all(reward <= 1.0))


def test_mpve_auxiliary_loss_has_no_policy_gradient(export_path: Path) -> None:
    network = _network("AC-MPC-MPVE", export_path)
    policy = network.policy_network.init(jax.random.PRNGKey(1))
    value = network.value_network.init(jax.random.PRNGKey(2))
    params = PPONetworkParams(policy=policy, value=value)
    observation = jp.ones((2, 3, 5), dtype=jp.float32)
    logits = network.policy_network.apply(None, policy, observation)
    data = types.Transition(
        observation=observation,
        action=jp.zeros((2, 3, 1)),
        reward=jp.zeros((2, 3)),
        discount=jp.ones((2, 3)),
        next_observation=observation,
        extras={
            "policy_extras": {
                "distribution_params": logits,
                "mpve_terminal_value": jp.zeros((2, 3)),
            }
        },
    )

    def zero_base(*args, **kwargs):
        del args, kwargs
        return jp.asarray(0.0), {"total_loss": jp.asarray(0.0)}

    loss = make_mpve_ppo_loss(
        zero_base,
        koopman_path=str(export_path),
        action_size=1,
        horizon=10,
        coefficient=1.0,
        reward_source="exact_cartpole",
    )

    def objective(current):
        return loss(current, None, data, jax.random.PRNGKey(3), network)[0]

    gradient = jax.grad(objective)(params)
    assert all(
        bool(jp.array_equal(leaf, jp.zeros_like(leaf)))
        for leaf in jax.tree.leaves(gradient.policy)
    )
    assert any(
        bool(jp.any(jp.abs(leaf) > 0)) for leaf in jax.tree.leaves(gradient.value)
    )


def test_supervisor_accepts_only_a_completed_koopman_run(tmp_path: Path) -> None:
    (tmp_path / "best.npz").write_bytes(b"checkpoint")
    (tmp_path / "run.json").write_text('{"completed": true}', encoding="utf-8")
    assert _wait_for_koopman(tmp_path, 0.001) == (tmp_path / "best.npz").resolve()


def test_supervisor_child_command_keeps_method_specific_output(tmp_path: Path) -> None:
    command = _child_command(
        task="CartpoleSwingup",
        method="KMPC",
        koopman=tmp_path / "best.npz",
        output=tmp_path / "KMPC",
        seed=20260812,
        timesteps=1234,
    )
    assert command[1:3] == ["-m", "experiments.playground.train_structured"]
    assert command[command.index("--method") + 1] == "KMPC"
    assert command[command.index("--timesteps") + 1] == "1234"


def test_reacher_hard_exact_reward_uses_observed_target_vector() -> None:
    parameters = _identity_parameters(6, 2)
    observation = jp.zeros((2, 6)).at[:, 2].set(jp.asarray([0.024, 0.026]))
    reward = reacher_hard_exact_reward(
        parameters, jp.zeros((2, 2)), observation
    )
    assert jp.array_equal(reward, jp.asarray([1.0, 0.0]))


def test_humanoid_run_exact_reward_reaches_one_at_all_targets() -> None:
    parameters = _identity_parameters(67, 21)
    observation = jp.zeros((1, 67))
    observation = observation.at[:, 21].set(1.4)
    observation = observation.at[:, 36].set(0.9)
    observation = observation.at[:, 37].set(10.0)
    reward = humanoid_run_exact_reward(
        parameters, jp.zeros((1, 21)), observation
    )
    assert reward == pytest.approx(jp.asarray([1.0]))


def test_invalid_structured_critic_input_is_rejected(export_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown critic input"):
        make_structured_ppo_networks(
            (5,),
            1,
            lambda observation, unused: observation,
            method="KMPC",
            koopman_path=str(export_path),
            critic_input="wrong",
        )


def test_lifted_critic_uses_koopman_lift_dimension(export_path: Path) -> None:
    network = make_structured_ppo_networks(
        (5,),
        1,
        lambda observation, unused: observation,
        method="KMPC",
        koopman_path=str(export_path),
        critic_input="lifted_state",
    )
    value = network.value_network.init(jax.random.PRNGKey(12))
    assert any(15 in leaf.shape for leaf in jax.tree.leaves(value))
    estimate = network.value_network.apply(
        None, value, jp.zeros((3, 5), dtype=jp.float32)
    )
    assert estimate.shape == (3,)
    assert bool(jp.all(jp.isfinite(estimate)))
