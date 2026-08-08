import pytest
import torch

from antmaze_ac.koopman.visual_losses import visual_koopman_loss
from antmaze_ac.koopman.visual_model import VisualLinearKoopman
from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor


def test_identity_model_shapes_rollout_and_external_action() -> None:
    torch.manual_seed(10)
    model = VisualLinearKoopman(
        robot_dim=5,
        action_dim=2,
        visual_feature_dim=12,
        visual_latent_dim=3,
        encoder_hidden_dims=(9,),
    )
    robot = torch.randn(4, 5)
    feature = torch.randn(4, 12)
    state = model.make_state(robot, feature)
    lifted = model.lift(state)

    assert model.state_dim == model.lifted_dim == 8
    assert model.lift_dim == 0
    assert state.shape == lifted.shape == (4, 8)
    torch.testing.assert_close(lifted, state)
    torch.testing.assert_close(model.reconstruct(lifted), state)
    assert not isinstance(model.T, torch.nn.Parameter)
    assert not isinstance(model.C, torch.nn.Parameter)

    action_0 = torch.zeros(4, 2)
    action_1 = torch.randn(4, 2)
    following_0 = model.linear_step(lifted, action_0)
    following_1 = model.linear_step(lifted, action_1)
    torch.testing.assert_close(
        following_1 - following_0,
        action_1 @ model.B.mT,
    )

    actions = torch.randn(4, 6, 2)
    predicted_states, predicted_lifts = model.rollout(state, actions)
    assert predicted_states.shape == (4, 6, 8)
    assert predicted_lifts.shape == (4, 6, 8)
    manual = lifted
    for step in range(actions.shape[1]):
        manual = model.linear_step(manual, actions[:, step])
        torch.testing.assert_close(predicted_lifts[:, step], manual)
        torch.testing.assert_close(
            predicted_states[:, step],
            model.reconstruct(manual),
        )

    architecture = model.architecture()
    assert architecture["architecture"] == "visual_linear_controlled_v1"
    assert architecture["transform_mode"] == "identity"
    assert architecture["visual_latent_dim"] == 3


def test_learned_transform_loss_backward_and_architecture_roundtrip() -> None:
    torch.manual_seed(11)
    model = VisualLinearKoopman(
        robot_dim=3,
        action_dim=2,
        visual_feature_dim=7,
        visual_latent_dim=2,
        encoder_hidden_dims=(8,),
        activation="gelu",
        transform_mode="learned",
    )
    robot_states = torch.randn(6, 5, 3)
    visual_features = torch.randn(6, 5, 7)
    actions = torch.randn(6, 4, 2) * 0.2

    initial_states = model.make_state(robot_states, visual_features)
    torch.testing.assert_close(
        model.reconstruct(model.lift(initial_states)),
        initial_states,
    )
    assert isinstance(model.T, torch.nn.Parameter)
    assert isinstance(model.C, torch.nn.Parameter)

    losses = visual_koopman_loss(
        model,
        robot_states,
        visual_features,
        actions,
        spectral_radius_limit=1.05,
    )
    assert all(torch.isfinite(value) for value in losses.__dict__.values())
    losses.total.backward()
    named_gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
    }
    assert all(gradient is not None for gradient in named_gradients.values())
    assert all(
        torch.isfinite(gradient).all()
        for gradient in named_gradients.values()
        if gradient is not None
    )
    for prefix in ("A", "B", "T", "C", "visual_encoder", "feature_decoder"):
        selected = [
            gradient
            for name, gradient in named_gradients.items()
            if name == prefix or name.startswith(prefix + ".")
        ]
        assert selected
        assert sum(float(gradient.abs().sum()) for gradient in selected) > 0.0

    architecture = model.architecture().copy()
    architecture.pop("architecture")
    restored = VisualLinearKoopman(**architecture)
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        expected = model.make_state(robot_states[:, 0], visual_features[:, 0])
        actual = restored.make_state(robot_states[:, 0], visual_features[:, 0])
    torch.testing.assert_close(actual, expected)


def test_detached_linear_target_keeps_target_side_transform_gradient() -> None:
    """Stop-gradient applies to the encoder target, not the learned T."""

    torch.manual_seed(110)
    model = VisualLinearKoopman(
        robot_dim=2,
        action_dim=1,
        visual_feature_dim=4,
        visual_latent_dim=1,
        encoder_hidden_dims=(5,),
        transform_mode="learned",
    )
    # Remove the initial-state path from the linear prediction.  A nonzero T
    # gradient can then only come through the target lift T s_next.
    with torch.no_grad():
        model.A.zero_()
        model.B.zero_()
    robot = torch.randn(4, 4, 2)
    feature = torch.randn(4, 4, 4)
    action = torch.zeros(4, 3, 1)

    losses = visual_koopman_loss(model, robot, feature, action)
    transform_gradient, *encoder_gradients = torch.autograd.grad(
        losses.linear,
        (model.T, *tuple(model.visual_encoder.parameters())),
        allow_unused=True,
    )

    assert torch.isfinite(transform_gradient).all()
    assert float(transform_gradient.abs().sum()) > 0.0
    assert all(
        gradient is None or float(gradient.abs().sum()) == 0.0
        for gradient in encoder_gradients
    )


def test_learned_inverse_uses_exact_differentiable_solve_and_no_c_parameter() -> None:
    torch.manual_seed(111)
    model = VisualLinearKoopman(
        robot_dim=3,
        action_dim=2,
        visual_feature_dim=7,
        visual_latent_dim=2,
        encoder_hidden_dims=(8,),
        transform_mode="learned_inverse",
    )
    with torch.no_grad():
        model.T.copy_(
            torch.tensor(
                [
                    [1.3, 0.2, 0.0, 0.1, 0.0],
                    [0.0, 0.8, 0.1, 0.0, 0.0],
                    [0.1, 0.0, 1.1, 0.2, 0.0],
                    [0.0, 0.1, 0.0, 0.9, 0.1],
                    [0.0, 0.0, 0.1, 0.0, 1.2],
                ]
            )
        )
    state = torch.randn(4, 6, model.state_dim)
    lifted = model.lift(state)
    reconstructed = model.reconstruct(lifted)
    torch.testing.assert_close(reconstructed, state, rtol=2e-6, atol=2e-6)

    readout = model.readout_matrix()
    torch.testing.assert_close(
        readout @ model.T,
        torch.eye(model.state_dim),
        rtol=2e-6,
        atol=2e-6,
    )
    assert "C" not in dict(model.named_parameters())
    assert "C" not in model.state_dict()

    # Use a lifted vector independent of T so the solve has a nonzero T
    # derivative (reconstruct(lift(s)) is mathematically T-invariant).
    arbitrary_lift = torch.randn(9, model.lifted_dim)
    objective = model.reconstruct(arbitrary_lift).square().mean()
    transform_gradient = torch.autograd.grad(objective, model.T)[0]
    assert torch.isfinite(transform_gradient).all()
    assert float(transform_gradient.abs().sum()) > 0.0


def test_learned_inverse_loss_penalizes_near_singular_transform() -> None:
    torch.manual_seed(112)
    model = VisualLinearKoopman(
        robot_dim=2,
        action_dim=1,
        visual_feature_dim=5,
        visual_latent_dim=2,
        encoder_hidden_dims=(6,),
        transform_mode="learned_inverse",
    )
    with torch.no_grad():
        model.T[-1].mul_(0.05)
    robot = torch.randn(5, 4, 2)
    features = torch.randn(5, 4, 5)
    actions = 0.1 * torch.randn(5, 3, 1)
    losses = visual_koopman_loss(
        model,
        robot,
        features,
        actions,
        spectral_radius_limit=1.05,
        transform_minimum_singular_value=0.25,
    )
    assert float(losses.transform_min_singular_value) < 0.25
    assert float(losses.transform_max_singular_value) >= 1.0
    assert float(losses.transform_singular_value) > 0.0
    assert float(losses.transform_condition_number) > 1.0
    assert float(losses.identity) > 0.0
    # Exact inverse decoding keeps C(Ts)-s at numerical precision even though
    # the chosen T is deliberately ill-conditioned.
    assert float(losses.transform_reconstruction) < 1e-10
    losses.total.backward()
    assert model.T.grad is not None
    assert torch.isfinite(model.T.grad).all()
    assert float(model.T.grad.abs().sum()) > 0.0


def test_learned_orthogonal_transform_is_exact_well_conditioned_and_differentiable() -> None:
    torch.manual_seed(114)
    model = VisualLinearKoopman(
        robot_dim=3,
        action_dim=2,
        visual_feature_dim=7,
        visual_latent_dim=2,
        encoder_hidden_dims=(8,),
        transform_mode="learned_orthogonal",
    )
    with torch.no_grad():
        model.S.copy_(0.2 * torch.randn_like(model.S))

    transform = model.transform_matrix()
    readout = model.readout_matrix(transform=transform)
    identity = torch.eye(model.state_dim)
    torch.testing.assert_close(
        transform.mT @ transform,
        identity,
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(readout, transform.mT)
    assert "S" in model.state_dict()
    assert "T" not in model.state_dict()
    assert "C" not in model.state_dict()

    state = torch.randn(4, 6, model.state_dim)
    lifted = model.lift(state, transform=transform)
    torch.testing.assert_close(
        model.reconstruct(lifted, transform=transform),
        state,
        rtol=2e-5,
        atol=2e-5,
    )
    asymmetric_objective = lifted[..., 0].sum()
    generator_gradient = torch.autograd.grad(
        asymmetric_objective,
        model.S,
    )[0]
    assert torch.isfinite(generator_gradient).all()
    assert float(generator_gradient.abs().sum()) > 0.0

    robot = torch.randn(5, 4, 3)
    features = torch.randn(5, 4, 7)
    actions = 0.1 * torch.randn(5, 3, 2)
    losses = visual_koopman_loss(
        model,
        robot,
        features,
        actions,
        spectral_radius_limit=1.05,
    )
    assert float(losses.transform_reconstruction) < 1e-10
    assert float(losses.transform_condition) < 1e-10
    assert float(losses.transform_singular_value) == 0.0
    assert float(losses.transform_min_singular_value) == pytest.approx(
        1.0,
        abs=2e-5,
    )
    assert float(losses.transform_max_singular_value) == pytest.approx(
        1.0,
        abs=2e-5,
    )
    assert float(losses.transform_condition_number) == pytest.approx(
        1.0,
        abs=3e-5,
    )
    losses.total.backward()
    assert model.S.grad is not None
    assert torch.isfinite(model.S.grad).all()
    assert float(model.S.grad.abs().sum()) > 0.0


def test_learned_orthogonal_loss_materializes_transform_once(monkeypatch) -> None:
    model = VisualLinearKoopman(
        robot_dim=2,
        action_dim=1,
        visual_feature_dim=4,
        visual_latent_dim=2,
        encoder_hidden_dims=(5,),
        transform_mode="learned_orthogonal",
    )
    original_matrix_exp = torch.matrix_exp
    calls = 0

    def counted_matrix_exp(value):
        nonlocal calls
        calls += 1
        return original_matrix_exp(value)

    monkeypatch.setattr(torch, "matrix_exp", counted_matrix_exp)
    losses = visual_koopman_loss(
        model,
        torch.randn(3, 6, 2),
        torch.randn(3, 6, 4),
        torch.randn(3, 5, 1),
        spectral_radius_limit=1.05,
    )

    assert torch.isfinite(losses.total)
    assert calls == 1


def test_visual_loss_rejects_invalid_windows_and_nonfinite_inputs() -> None:
    model = VisualLinearKoopman(
        robot_dim=2,
        action_dim=1,
        visual_feature_dim=4,
        visual_latent_dim=2,
        encoder_hidden_dims=(),
    )
    robot = torch.randn(3, 4, 2)
    feature = torch.randn(3, 4, 4)
    action = torch.randn(3, 3, 1)
    with pytest.raises(ValueError, match="one more step"):
        visual_koopman_loss(model, robot[:, :-1], feature[:, :-1], action)

    feature[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        visual_koopman_loss(model, robot, feature, action)


def test_short_synthetic_training_reduces_visual_koopman_loss() -> None:
    torch.manual_seed(12)
    batch = 64
    horizon = 4
    actions = 0.3 * torch.randn(batch, horizon, 1)
    true_joint = torch.zeros(batch, horizon + 1, 2)
    true_joint[:, 0] = torch.randn(batch, 2) * 0.5
    true_A = torch.tensor([[0.82, 0.15], [0.08, 0.88]])
    true_B = torch.tensor([[0.45], [-0.25]])
    for step in range(horizon):
        true_joint[:, step + 1] = (
            true_joint[:, step] @ true_A.mT
            + actions[:, step] @ true_B.mT
        )
    robot = true_joint[..., :1]
    latent = true_joint[..., 1:]
    features = torch.cat((latent, 0.5 * latent), dim=-1)

    model = VisualLinearKoopman(
        robot_dim=1,
        action_dim=1,
        visual_feature_dim=2,
        visual_latent_dim=1,
        encoder_hidden_dims=(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_kwargs = {
        "linear_weight": 2.0,
        "robot_rollout_weight": 1.0,
        "feature_reconstruction_weight": 1.0,
        "future_feature_reconstruction_weight": 1.0,
        "latent_variance_weight": 0.01,
        "stability_weight": 0.0,
    }
    initial = float(
        visual_koopman_loss(
            model, robot, features, actions, **loss_kwargs
        ).total.detach()
    )
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        total = visual_koopman_loss(
            model, robot, features, actions, **loss_kwargs
        ).total
        total.backward()
        optimizer.step()
    final = float(
        visual_koopman_loss(
            model, robot, features, actions, **loss_kwargs
        ).total.detach()
    )
    assert final < 0.5 * initial


def test_equal_dimension_model_connects_to_existing_mpc_actor() -> None:
    torch.manual_seed(13)
    model = VisualLinearKoopman(
        robot_dim=4,
        action_dim=2,
        visual_feature_dim=10,
        visual_latent_dim=3,
        encoder_hidden_dims=(8,),
    )
    mpc = KoopmanMPCActor(
        model.A,
        model.B,
        model.C,
        horizon=3,
        hidden_dims=(12,),
        action_low=-0.2,
        action_high=0.25,
        solver_iterations=5,
    )
    robot = torch.randn(5, 4)
    feature = torch.randn(5, 10)
    lifted = model.lift(model.make_state(robot, feature)).detach()
    output = mpc(lifted)

    assert output.action.shape == (5, 2)
    assert output.action_sequence.shape == (5, 3, 2)
    assert bool((output.action_sequence >= -0.2).all())
    assert bool((output.action_sequence <= 0.25).all())
    loss = output.action.square().mean()
    gradients = torch.autograd.grad(loss, tuple(mpc.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_learned_inverse_exports_exact_readout_to_existing_mpc_actor() -> None:
    torch.manual_seed(113)
    model = VisualLinearKoopman(
        robot_dim=3,
        action_dim=2,
        visual_feature_dim=8,
        visual_latent_dim=2,
        encoder_hidden_dims=(7,),
        transform_mode="learned_inverse",
    )
    with torch.no_grad():
        model.T.add_(0.08 * torch.randn_like(model.T))
    readout = model.readout_matrix()
    mpc = KoopmanMPCActor(
        model.A,
        model.B,
        readout,
        horizon=2,
        hidden_dims=(8,),
        solver_iterations=3,
    )
    torch.testing.assert_close(mpc.C @ model.T, torch.eye(model.state_dim))
    state = model.make_state(torch.randn(4, 3), torch.randn(4, 8))
    output = mpc(model.lift(state))
    assert output.action.shape == (4, 2)
    assert torch.isfinite(output.action).all()


def test_learned_orthogonal_exports_transpose_readout_to_existing_mpc_actor() -> None:
    torch.manual_seed(115)
    model = VisualLinearKoopman(
        robot_dim=3,
        action_dim=2,
        visual_feature_dim=8,
        visual_latent_dim=2,
        encoder_hidden_dims=(7,),
        transform_mode="learned_orthogonal",
    )
    with torch.no_grad():
        model.S.copy_(0.15 * torch.randn_like(model.S))
    transform = model.transform_matrix()
    readout = model.readout_matrix(transform=transform)
    mpc = KoopmanMPCActor(
        model.A,
        model.B,
        readout,
        horizon=2,
        hidden_dims=(8,),
        solver_iterations=3,
    )

    torch.testing.assert_close(
        mpc.C @ transform,
        torch.eye(model.state_dim),
        rtol=2e-5,
        atol=2e-5,
    )
    state = model.make_state(torch.randn(4, 3), torch.randn(4, 8))
    output = mpc(model.lift(state, transform=transform))
    assert output.action.shape == (4, 2)
    assert torch.isfinite(output.action).all()
