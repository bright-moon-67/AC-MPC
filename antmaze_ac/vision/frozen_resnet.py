"""A deterministic, frozen ImageNet ResNet-18 feature extractor."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class FrozenResNet18(nn.Module):
    """Return the 512-dimensional pre-classifier ResNet-18 feature.

    Inputs may be batched ``NHWC`` or ``NCHW`` RGB tensors. ``uint8``
    tensors are scaled from ``[0, 255]``; floating tensors may be in either
    ``[0, 1]`` or ``[0, 255]``. Images are resized to 224 by 224 and
    normalized with the ImageNet statistics associated with
    :attr:`ResNet18_Weights.IMAGENET1K_V1`.

    The backbone is permanently kept in evaluation mode and all of its
    parameters are frozen. Calling :meth:`train` on this wrapper therefore
    cannot accidentally enable batch-normalization updates.
    """

    output_dim = 512
    weights = ResNet18_Weights.IMAGENET1K_V1

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=self.weights)
        self.backbone.fc = nn.Identity()
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenResNet18":
        """Keep the extractor in evaluation mode regardless of ``mode``."""

        super().train(False)
        self.backbone.eval()
        return self

    @staticmethod
    def _to_nchw(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}")
        if images.ndim not in (3, 4):
            raise ValueError(
                "Expected an HWC/CHW image or NHWC/NCHW batch, "
                f"got shape {tuple(images.shape)}"
            )

        unbatched = images.ndim == 3
        if unbatched:
            images = images.unsqueeze(0)

        if images.shape[1] == 3:
            nchw = images
        elif images.shape[-1] == 3:
            nchw = images.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "Expected exactly three RGB channels in axis 1 or the last axis, "
                f"got shape {tuple(images.shape)}"
            )
        if nchw.shape[0] == 0 or nchw.shape[-2] == 0 or nchw.shape[-1] == 0:
            raise ValueError(f"Empty image batches are unsupported: {tuple(images.shape)}")
        return nchw, unbatched

    def preprocess(self, images: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Convert an RGB tensor to normalized ``NCHW`` float32 images."""

        images, unbatched = self._to_nchw(images)
        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32).div_(255.0)
        elif images.is_floating_point():
            images = images.to(dtype=torch.float32)
            if not bool(torch.isfinite(images).all()):
                raise FloatingPointError("images contain NaN or Inf")
            minimum, maximum = torch.aminmax(images.detach())
            if float(minimum) < 0.0 or float(maximum) > 255.0:
                raise ValueError("Floating RGB inputs must be in [0, 1] or [0, 255]")
            if float(maximum) > 1.0:
                images = images / 255.0
        else:
            raise TypeError(
                "RGB inputs must have dtype uint8 or a floating dtype, "
                f"got {images.dtype}"
            )

        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.imagenet_mean) / self.imagenet_std
        return images, unbatched

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized, unbatched = self.preprocess(images)
        features = self.backbone(normalized)
        if features.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"ResNet-18 returned {features.shape[-1]} features, expected {self.output_dim}"
            )
        return features[0] if unbatched else features
