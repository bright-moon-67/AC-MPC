"""Trainable visual encoder for single-stage visual BC.

Pipeline (KOROL-style structure, single-stage training):

    RGBD(4ch) --[optional 2D DCT, no_grad]--> 8ch
      -> trainable ResNet18 (ImageNet warm start, 8-channel conv1)
      -> b(512)
      -> MLP head 512 -> hidden -> v(v_dim)
      -> pos_branch v -> 3            (privileged active-waypoint regression)

The encoder is trained end-to-end by the BC action loss through the
cost-map / differentiable MPC chain, plus the optional ``pos_branch`` geometric
supervision that anchors the latent against collapse.  ``pos_branch`` is used
only during training; at inference only ``v`` is consumed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _to_nchw(images: torch.Tensor, channels: int, name: str) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if images.ndim not in (3, 4):
        raise ValueError(f"{name} must be CHW/HWC or NCHW/NHWC")
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.shape[1] == channels:
        return images
    if images.shape[-1] == channels:
        return images.permute(0, 3, 1, 2)
    raise ValueError(
        f"Expected {channels} channels in axis 1 or last axis for {name}, "
        f"got {tuple(images.shape)}"
    )


def build_dct_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Orthonormal DCT-II matrix of shape ``[size, size]``.

    Matches ``scipy.fft.dct(x, type=2, norm='ortho', axis=...)`` so that
    ``y = C @ x`` is the type-II cosine transform along a dimension.
    """

    indices = torch.arange(size, dtype=torch.float64, device=device)
    basis = torch.cos(
        (math.pi / size) * indices.unsqueeze(1) * (indices.unsqueeze(0) + 0.5)
    )
    scale = torch.full((size,), math.sqrt(2.0 / size), dtype=torch.float64, device=device)
    scale[0] = math.sqrt(1.0 / size)
    return (basis * scale.unsqueeze(1)).to(dtype=dtype)


def dct_2d(x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply 2D DCT-II to the last two dimensions of ``x``.

    ``x`` may have leading batch/channel dims; ``transform`` is ``[N, N]``
    with ``N`` equal to the spatial size.
    """

    y = torch.einsum("...hw,kw->...hk", x, transform)  # DCT along width
    y = torch.einsum("...hn,kh->...kn", y, transform)  # DCT along height
    return y


def _activation(name: str) -> type[nn.Module]:
    choices = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported encoder activation {name!r}") from exc


class VisualEncoder(nn.Module):
    """Trainable 8-channel ResNet18 -> ``v`` + privileged ``pos_branch``."""

    def __init__(
        self,
        v_dim: int = 16,
        *,
        hidden_dims: Sequence[int] = (128,),
        activation: str = "gelu",
        use_dct: bool = True,
        # ManiSkill's minimal shader returns depth in millimeters (uint16).
        # 2500 mm = 2.5 m normalizes the near-field table scene to [0, 1].
        depth_scale: float = 2500.0,
        pos_dim: int = 3,
    ) -> None:
        super().__init__()
        if v_dim < 1:
            raise ValueError("v_dim must be positive")
        if depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
        if pos_dim < 1:
            raise ValueError("pos_dim must be positive")
        self.v_dim = int(v_dim)
        self.use_dct = bool(use_dct)
        self.pos_dim = int(pos_dim)
        self.register_buffer("depth_scale", torch.tensor(float(depth_scale)))
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self._dct_cache: dict[int, torch.Tensor] = {}

        # Trainable backbone with an 8-channel (or 4-channel) first convolution,
        # warm-started from ImageNet weights: RGB channels inherit the pretrained
        # filter, depth/DCT channels start from a small scaled copy.
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        input_channels = 4 if not self.use_dct else 8
        conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        with torch.no_grad():
            pretrained = backbone.conv1.weight  # [64, 3, 7, 7]
            conv1.weight[:, :3] = pretrained
            if input_channels > 3:
                conv1.weight[:, 3:] = pretrained.mean(dim=1, keepdim=True) * 0.1
        backbone.conv1 = conv1
        backbone.fc = nn.Identity()
        self.backbone = backbone

        activation_type = _activation(activation)
        dimensions = [512, *map(int, hidden_dims), self.v_dim]
        head_layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            head_layers.append(nn.Linear(in_dim, out_dim))
            if index < len(dimensions) - 2:
                head_layers.append(activation_type())
        self.head = nn.Sequential(*head_layers)

        # Privileged geometric anchor: regresses the active waypoint position
        # from ``v``.  Training-only; never consumed at inference.
        self.pos_branch = nn.Linear(self.v_dim, self.pos_dim)
        nn.init.zeros_(self.pos_branch.weight)
        nn.init.zeros_(self.pos_branch.bias)

    @torch.no_grad()
    def _dct_transform(self, size: int) -> torch.Tensor:
        transform = self._dct_cache.get(size)
        if transform is None or transform.device != self.depth_scale.device:
            transform = build_dct_matrix(
                size, self.depth_scale.dtype, self.depth_scale.device
            )
            self._dct_cache[size] = transform
        return transform

    def preprocess(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized ``NCHW`` float input with ``4`` or ``8`` channels."""

        rgb_nchw = _to_nchw(rgb, 3, "rgb").float()
        depth_nchw = _to_nchw(depth, 1, "depth").float()
        if rgb_nchw.shape[-2:] != depth_nchw.shape[-2:]:
            raise ValueError("RGB and depth resolutions differ")
        if rgb_nchw.dtype == torch.uint8:
            rgb_nchw = rgb_nchw / 255.0
        rgb_nchw = (rgb_nchw - self.imagenet_mean) / self.imagenet_std
        depth_norm = (depth_nchw / self.depth_scale).clamp(0.0, 1.0)
        spatial = torch.cat((rgb_nchw, depth_norm), dim=1)  # [B, 4, H, W]
        if not self.use_dct:
            return spatial
        height = spatial.shape[-2]
        if height != spatial.shape[-1]:
            raise ValueError("2D DCT requires square images")
        transform = self._dct_transform(height)
        with torch.no_grad():
            frequency = dct_2d(spatial, transform)
        return torch.cat((spatial, frequency), dim=1)  # [B, 8, H, W]

    def backbone_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled 512-d feature ``b`` for a preprocessed input batch."""

        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return self.backbone.avgpool(x).flatten(1)

    def spatial_features(self, images: torch.Tensor) -> torch.Tensor:
        """Spatial feature map before global pooling ``[B, 512, h, w]`` (for CAM)."""

        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(v, pos)`` with shapes ``[..., v_dim]`` and ``[..., pos_dim]``."""

        images = self.preprocess(rgb, depth)
        pooled = self.backbone_forward(images)
        v = self.head(pooled)
        pos = self.pos_branch(v)
        return v, pos

    def activation_map(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """KOROL-style CAM saliency ``[B, h, w]`` from the spatial feature map.

        The pooled-to-``v`` head's first linear layer weights are used as the
        class-like weighting over the 512 spatial channels.
        """

        first = self.head[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("Expected head[0] to be Linear for CAM")
        images = self.preprocess(rgb, depth)
        spatial = self.spatial_features(images)  # [B, 512, h, w]
        weights = first.weight.detach().mean(dim=0)  # [512]
        return (spatial * weights.view(1, -1, 1, 1)).sum(dim=1)  # [B, h, w]


__all__ = ["VisualEncoder", "build_dct_matrix", "dct_2d"]
