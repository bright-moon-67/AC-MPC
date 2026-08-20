import tempfile
from pathlib import Path

import torch

from antmaze_ac.koopman.checkpoint import load_checkpoint, save_checkpoint
from antmaze_ac.koopman.losses import koopman_loss
from antmaze_ac.koopman.model import DeepKoopman


def test_shape_loss_rollout_backward_and_checkpoint():
    torch.manual_seed(1)
    model = DeepKoopman(7, 2, lift_dim=4, hidden_dims=(16, 16))
    states = torch.randn(5, 21, 7)
    actions = torch.randn(5, 20, 2)
    predicted, lifted = model.rollout(states[:, 0], actions)
    assert predicted.shape == (5, 20, 7)
    assert lifted.shape == (5, 20, 11)
    losses = koopman_loss(model, states, actions)
    losses.total.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        save_checkpoint(
            path,
            model,
            optimizer=optimizer,
            epoch=2,
            best_validation=1.0,
            config={},
            normalizers={},
            elapsed_seconds=3.0,
        )
        restored, payload = load_checkpoint(path)
        assert payload["epoch"] == 2
        for expected, actual in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, actual)


def test_full_a_reference_initialization_and_one_step_forward():
    torch.manual_seed(7)
    model = DeepKoopman(7, 2, lift_dim=4, hidden_dims=(16, 16))
    identity = torch.eye(model.lifted_dim)
    assert torch.max(torch.abs(model.A - identity)) < 0.01
    assert torch.count_nonzero(model.A - torch.diag(torch.diag(model.A))) > 0
    assert 0 < float(model.B.std()) < 0.03
    states = torch.randn(5, 7)
    actions = torch.randn(5, 2)
    predicted, lifted = model(states, actions)
    assert predicted.shape == (5, 7)
    assert lifted.shape == (5, 11)
    torch.testing.assert_close(predicted, model.reconstruct(lifted))


def test_exact_previous_action_integrator_in_normalized_coordinates():
    model = DeepKoopman(7, 2, lift_dim=4, hidden_dims=(16, 16))
    action_std = torch.tensor([0.2, 0.5])
    model.configure_action_integrator(action_std)
    state = torch.randn(5, 7)
    delta_action = torch.randn(5, 2) * 0.01

    predicted, _ = model(state, delta_action)

    torch.testing.assert_close(
        predicted[:, -2:],
        state[:, -2:] + delta_action / action_std,
    )


def test_loss_rejects_nonfinite_inputs_and_preserves_identity_reconstruction():
    model = DeepKoopman(3, 1, lift_dim=2, hidden_dims=(8,))
    states = torch.randn(2, 3, 3)
    actions = torch.randn(2, 2, 1)
    losses = koopman_loss(model, states, actions)
    assert float(losses.reconstruction) == 0.0
    states[0, 0, 0] = float("nan")
    try:
        koopman_loss(model, states, actions)
    except FloatingPointError as error:
        assert "NaN or Inf" in str(error)
    else:
        raise AssertionError("non-finite input must be rejected")


def test_short_training_smoke_reduces_loss():
    torch.manual_seed(2)
    model = DeepKoopman(3, 1, lift_dim=2, hidden_dims=(8,))
    states = torch.zeros(16, 6, 3)
    actions = torch.randn(16, 5, 1) * 0.1
    for index in range(5):
        states[:, index + 1] = 0.8 * states[:, index]
        states[:, index + 1, 0] += actions[:, index, 0]
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial = float(koopman_loss(model, states, actions).total)
    for _ in range(20):
        optimizer.zero_grad()
        loss = koopman_loss(model, states, actions).total
        loss.backward()
        optimizer.step()
    final = float(koopman_loss(model, states, actions).total)
    assert final < initial
