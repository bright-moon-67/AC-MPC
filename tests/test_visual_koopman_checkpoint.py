from __future__ import annotations

import pytest
import torch

from antmaze_ac.koopman.checkpoint import load_checkpoint, save_checkpoint
from antmaze_ac.koopman.visual_model import VisualLinearKoopman


@pytest.mark.parametrize(
    "transform_mode",
    ["identity", "learned_inverse", "learned_orthogonal"],
)
def test_visual_checkpoint_round_trip(tmp_path, transform_mode: str) -> None:
    torch.manual_seed(7)
    model = VisualLinearKoopman(
        robot_dim=21,
        action_dim=8,
        visual_feature_dim=512,
        visual_latent_dim=16,
        encoder_hidden_dims=(64, 32),
        transform_mode=transform_mode,
    )
    if transform_mode == "learned_inverse":
        with torch.no_grad():
            model.T.add_(0.03 * torch.randn_like(model.T))
    elif transform_mode == "learned_orthogonal":
        with torch.no_grad():
            model.S.copy_(0.1 * torch.randn_like(model.S))
    path = tmp_path / "visual.pt"
    save_checkpoint(
        path,
        model,
        epoch=3,
        best_validation=0.25,
        config={"horizon": 10},
        normalizers={"robot_mean": [0.0] * 21},
        elapsed_seconds=1.5,
    )

    restored, payload = load_checkpoint(path)

    assert isinstance(restored, VisualLinearKoopman)
    assert restored.architecture() == model.architecture()
    assert payload["epoch"] == 3
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
    restored_transform = restored.transform_matrix()
    torch.testing.assert_close(
        restored.readout_matrix(transform=restored_transform) @ restored_transform,
        torch.eye(restored.state_dim),
        rtol=2e-5,
        atol=2e-5,
    )
    if transform_mode == "identity":
        # Identity checkpoints retain the historical persistent C buffer.
        assert "C" in restored.state_dict()
    elif transform_mode == "learned_inverse":
        assert "C" not in restored.state_dict()
        assert "T" in restored.state_dict()
    else:
        assert "S" in restored.state_dict()
        assert "T" not in restored.state_dict()
        assert "C" not in restored.state_dict()
