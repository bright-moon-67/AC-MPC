from __future__ import annotations

import numpy as np
import torch

from antmaze_ac.data.build_sequences import Normalizer
from antmaze_ac.data.history_windows import AbsoluteActionHistoryWindowDataset
from antmaze_ac.koopman.checkpoint import load_checkpoint, save_checkpoint
from antmaze_ac.koopman.history_losses import history_koopman_loss
from antmaze_ac.koopman.history_model import HistoryDeepKoopman


def test_absolute_action_history_alignment() -> None:
    states = np.arange(12, dtype=np.float32).reshape(-1, 1)
    actions = (100 + np.arange(12, dtype=np.float32)).reshape(-1, 1)
    dataset = AbsoluteActionHistoryWindowDataset(
        states,
        actions,
        episode_ids=np.zeros(12, dtype=np.int64),
        step_indices=np.arange(12, dtype=np.int64),
        normalizer=Normalizer(
            mean=np.zeros(1, dtype=np.float32),
            std=np.ones(1, dtype=np.float32),
        ),
        transitions=3,
        history_steps=2,
        starts=np.asarray([2], dtype=np.int64),
    )

    contexts, state_sequence, controls = dataset[0]
    np.testing.assert_allclose(contexts[0].numpy(), [1, 2, 100, 101])
    np.testing.assert_allclose(contexts[1].numpy(), [2, 3, 101, 102])
    np.testing.assert_allclose(state_sequence.numpy().reshape(-1), [2, 3, 4, 5])
    np.testing.assert_allclose(controls.numpy().reshape(-1), [102, 103, 104])


def test_history_model_loss_and_gradients() -> None:
    torch.manual_seed(3)
    model = HistoryDeepKoopman(
        state_dim=3,
        action_dim=2,
        lift_dim=4,
        hidden_dims=(8,),
        activation="silu",
        history_steps=2,
    )
    contexts = torch.randn(5, 4, model.context_dim)
    states = torch.randn(5, 4, model.state_dim)
    actions = torch.randn(5, 3, model.action_dim)

    losses = history_koopman_loss(model, contexts, states, actions)
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert model.A.grad is not None and float(model.A.grad.abs().sum()) > 0
    assert model.B.grad is not None and float(model.B.grad.abs().sum()) > 0
    assert all(parameter.grad is not None for parameter in model.encoder.parameters())


def test_physical_tip_loss_uses_state_scale_and_enters_total() -> None:
    torch.manual_seed(4)
    model = HistoryDeepKoopman(
        state_dim=5,
        action_dim=2,
        lift_dim=4,
        hidden_dims=(8,),
        activation="silu",
        history_steps=2,
    )
    contexts = torch.randn(3, 4, model.context_dim)
    states = torch.randn(3, 4, model.state_dim)
    actions = torch.randn(3, 3, model.action_dim)
    state_std = torch.tensor([0.01, 0.02, 0.03, 2.0, 3.0])

    baseline = history_koopman_loss(model, contexts, states, actions)
    weighted = history_koopman_loss(
        model,
        contexts,
        states,
        actions,
        tip_position_weight=10000.0,
        tip_position_slice=(0, 3),
        state_std=state_std,
    )

    torch.testing.assert_close(
        weighted.total - baseline.total,
        10000.0 * weighted.tip_position,
    )
    assert float(weighted.tip_position) > 0
    assert float(weighted.tip_position_h1) > 0
    assert weighted.scalars()["tip_position_rmse_mm"] > 0


def test_current_action_is_not_part_of_current_context() -> None:
    states = np.arange(10, dtype=np.float32).reshape(-1, 1)
    actions = np.arange(10, dtype=np.float32).reshape(-1, 1)
    dataset = AbsoluteActionHistoryWindowDataset(
        states,
        actions,
        episode_ids=np.zeros(10, dtype=np.int64),
        step_indices=np.arange(10, dtype=np.int64),
        normalizer=Normalizer(
            mean=np.zeros(1, dtype=np.float32),
            std=np.ones(1, dtype=np.float32),
        ),
        transitions=2,
        history_steps=3,
        starts=np.asarray([3], dtype=np.int64),
    )
    contexts, _, controls = dataset[0]
    current_action = float(controls[0, 0])
    action_history = contexts[0, 3:].numpy()
    assert current_action == 3.0
    np.testing.assert_allclose(action_history, [0.0, 1.0, 2.0])


def test_episode_start_uses_state_repeat_and_zero_action_padding() -> None:
    states = np.arange(8, dtype=np.float32).reshape(-1, 1)
    actions = (10 + np.arange(8, dtype=np.float32)).reshape(-1, 1)
    dataset = AbsoluteActionHistoryWindowDataset(
        states,
        actions,
        episode_ids=np.zeros(8, dtype=np.int64),
        step_indices=np.arange(8, dtype=np.int64),
        normalizer=Normalizer(
            mean=np.zeros(1, dtype=np.float32),
            std=np.ones(1, dtype=np.float32),
        ),
        transitions=2,
        history_steps=3,
    )

    contexts, state_sequence, controls = dataset[0]
    np.testing.assert_allclose(contexts[0].numpy(), [0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(contexts[1].numpy(), [0, 0, 1, 0, 0, 10])
    np.testing.assert_allclose(state_sequence.numpy().reshape(-1), [0, 1, 2])
    np.testing.assert_allclose(controls.numpy().reshape(-1), [10, 11])


def test_history_checkpoint_round_trip(tmp_path) -> None:
    model = HistoryDeepKoopman(3, 2, lift_dim=4, hidden_dims=(8,), history_steps=2)
    path = tmp_path / "history.pt"
    save_checkpoint(
        path,
        model,
        epoch=0,
        best_validation=1.0,
        config={},
        normalizers={},
        elapsed_seconds=0.0,
    )

    loaded, payload = load_checkpoint(path)
    assert isinstance(loaded, HistoryDeepKoopman)
    assert loaded.history_steps == 2
    assert payload["architecture"]["architecture"] == "fullA_history_context_v1"
