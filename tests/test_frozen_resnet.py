from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("torchvision")
from torchvision.models import ResNet18_Weights

import antmaze_ac.vision.frozen_resnet as frozen_resnet_module
from antmaze_ac.vision import FrozenResNet18


class DummyResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.fc = nn.Linear(512, 1000)
        self.last_input: torch.Tensor | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.last_input = images.detach().clone()
        pooled = images.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        features = pooled.repeat(1, 512) * self.scale
        return self.fc(features)


def make_encoder(monkeypatch: pytest.MonkeyPatch) -> tuple[FrozenResNet18, DummyResNet]:
    calls: list[object] = []
    dummy = DummyResNet()

    def fake_resnet18(*, weights: object) -> DummyResNet:
        calls.append(weights)
        return dummy

    monkeypatch.setattr(frozen_resnet_module, "resnet18", fake_resnet18)
    encoder = FrozenResNet18()
    assert calls == [ResNet18_Weights.IMAGENET1K_V1]
    return encoder, dummy


def test_frozen_resnet_preprocesses_nhwc_and_stays_frozen(monkeypatch):
    encoder, backbone = make_encoder(monkeypatch)
    images = torch.randint(0, 256, (2, 31, 47, 3), dtype=torch.uint8)
    features = encoder(images)

    assert features.shape == (2, 512)
    assert isinstance(backbone.fc, nn.Identity)
    assert backbone.last_input is not None
    assert backbone.last_input.shape == (2, 3, 224, 224)
    assert not encoder.training
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    encoder.train()
    assert not encoder.training
    assert not backbone.training


def test_frozen_resnet_nchw_float_and_nhwc_uint8_agree(monkeypatch):
    encoder, _ = make_encoder(monkeypatch)
    images = torch.randint(0, 256, (3, 19, 23, 3), dtype=torch.uint8)
    expected = encoder(images)
    actual_unit = encoder(images.permute(0, 3, 1, 2).float() / 255.0)
    actual_255 = encoder(images.float())
    torch.testing.assert_close(actual_unit, expected)
    torch.testing.assert_close(actual_255, expected)


def test_frozen_resnet_unbatched_and_invalid_inputs(monkeypatch):
    encoder, _ = make_encoder(monkeypatch)
    assert encoder(torch.zeros(12, 13, 3, dtype=torch.uint8)).shape == (512,)
    with pytest.raises(ValueError, match="three RGB channels"):
        encoder(torch.zeros(2, 12, 13, 4, dtype=torch.uint8))
    with pytest.raises(TypeError, match="uint8 or a floating dtype"):
        encoder(torch.zeros(2, 12, 13, 3, dtype=torch.int16))
    bad = torch.zeros(2, 12, 13, 3)
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        encoder(bad)


class MeanFeatureEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = images.float().reshape(images.shape[0], -1).mean(dim=1, keepdim=True)
        return values.repeat(1, 512)


def test_feature_cache_schema_metadata_and_overwrite_refusal(tmp_path):
    h5py = pytest.importorskip("h5py")
    from experiments.maniskill_pick_visual.cache_resnet_features import (
        cache_resnet_features,
    )

    source = tmp_path / "visual.h5"
    output = tmp_path / "visual.resnet18.h5"
    with h5py.File(source, "w") as handle:
        handle.create_group("traj_2").create_dataset(
            "rgb", data=np.full((2, 8, 9, 3), 20, dtype=np.uint8)
        )
        handle.create_group("traj_0").create_dataset(
            "rgb", data=np.full((3, 3, 7, 6), 10, dtype=np.uint8)
        )

    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = cache_resnet_features(
        source,
        output,
        batch_size=2,
        device="cpu",
        encoder=MeanFeatureEncoder(),
    )
    assert summary.trajectories == 2
    assert summary.frames == 5
    assert summary.source_sha256 == source_digest

    with h5py.File(output, "r") as handle:
        assert bool(handle.attrs["complete"])
        assert handle.attrs["source_sha256"] == source_digest
        assert handle["metadata"].attrs["weights"] == "IMAGENET1K_V1"
        assert handle["traj_0/resnet18"].shape == (3, 512)
        assert handle["traj_2/resnet18"].shape == (2, 512)
        np.testing.assert_allclose(handle["traj_0/resnet18"][:], 10.0)
        np.testing.assert_allclose(handle["traj_2/resnet18"][:], 20.0)

    output_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="overwrite is disabled"):
        cache_resnet_features(
            source,
            output,
            device="cpu",
            encoder=MeanFeatureEncoder(),
        )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == output_digest
