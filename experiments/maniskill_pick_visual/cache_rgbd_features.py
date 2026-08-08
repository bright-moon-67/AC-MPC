"""Cache frozen RGB and depth-adapter ResNet-18 features.

The RGB branch is the existing frozen ImageNet ResNet-18.  Depth is converted
to a fixed three-channel pseudo-image (normalized depth, valid-depth mask, and
inverse normalized depth) and passed through the same frozen backbone.  The
cached ``rgbd_resnet18`` feature is the concatenation of the two 512-D
features.  Keeping the two component datasets in the sidecar makes the cache
auditable while allowing the existing visual Koopman trainer to consume the
1024-D concatenated feature without a second dynamics implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

from antmaze_ac.vision import FrozenResNet18


_TRAJECTORY_PATTERN = re.compile(r"traj_(\d+)$")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CacheSummary:
    source_path: str
    output_path: str
    source_sha256: str
    trajectories: int
    frames: int
    feature_dim: int = 2 * FrozenResNet18.output_dim


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def _trajectory_names(handle: h5py.File) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for name, value in handle.items():
        match = _TRAJECTORY_PATTERN.fullmatch(name)
        if match is not None and isinstance(value, h5py.Group):
            indexed.append((int(match.group(1)), name))
    if not indexed:
        raise ValueError("Source HDF5 contains no traj_i groups")
    return [name for _, name in sorted(indexed)]


def _depth_pseudo_rgb(depth: np.ndarray, depth_scale: float) -> torch.Tensor:
    array = np.asarray(depth)
    if array.ndim != 4 or array.shape[-1] != 1:
        raise ValueError(f"Expected NHW1 depth, got {array.shape}")
    values = torch.from_numpy(array[..., 0].astype(np.float32, copy=False))
    valid = torch.isfinite(values) & (values > 0.0)
    normalized = (values / float(depth_scale)).clamp(0.0, 1.0)
    # A fixed geometry-oriented adapter: depth, validity, inverse depth.
    pseudo = torch.stack((normalized, valid.to(torch.float32), 1.0 - normalized), dim=-1)
    return pseudo


def cache_rgbd_features(
    source_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 128,
    depth_scale: float = 10000.0,
    max_episodes: int | None = None,
    device: str | torch.device = "auto",
    overwrite: bool = False,
) -> CacheSummary:
    source = Path(source_path)
    output = Path(output_path)
    if batch_size < 1 or depth_scale <= 0.0:
        raise ValueError("batch_size and depth_scale must be positive")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive when provided")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("The sidecar output must differ from the source trajectory")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}")

    resolved_device = _resolve_device(device)
    source_digest = sha256(source)
    encoder = FrozenResNet18().to(resolved_device).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    trajectory_count = 0
    frame_count = 0
    with h5py.File(source, "r") as source_file, h5py.File(output, mode) as output_file:
        attrs = {
            "complete": False,
            "schema": "acmpc.rgbd-resnet18-feature-cache",
            "schema_version": SCHEMA_VERSION,
            "source_path": str(source.resolve()),
            "source_sha256": source_digest,
            "encoder": "resnet18_rgb_plus_depth_adapter",
            "weights": "IMAGENET1K_V1",
            "feature_dim": 2 * FrozenResNet18.output_dim,
            "rgb_feature_dim": FrozenResNet18.output_dim,
            "depth_feature_dim": FrozenResNet18.output_dim,
            "depth_scale": float(depth_scale),
            "device": str(resolved_device),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in attrs.items():
            output_file.attrs[key] = value
        metadata = output_file.create_group("metadata")
        for key, value in attrs.items():
            metadata.attrs[key] = value

        with torch.inference_mode():
            trajectories = _trajectory_names(source_file)
            if max_episodes is not None:
                trajectories = trajectories[:max_episodes]
            output_file.attrs["episode_limit"] = (
                -1 if max_episodes is None else int(max_episodes)
            )
            for trajectory in trajectories:
                source_group = source_file[trajectory]
                if "rgb" not in source_group or "depth" not in source_group:
                    raise KeyError(f"{trajectory} must contain rgb and depth")
                rgb = source_group["rgb"]
                depth = source_group["depth"]
                if rgb.ndim != 4 or rgb.shape[0] != depth.shape[0]:
                    raise ValueError(f"RGB/depth frame mismatch in {trajectory}")
                if depth.ndim != 4 or depth.shape[-1] != 1:
                    raise ValueError(f"Expected {trajectory}/depth as NHW1")
                frames = int(rgb.shape[0])
                group = output_file.create_group(trajectory)
                rgb_features = group.create_dataset(
                    "rgb_resnet18", shape=(frames, 512), dtype=np.float32,
                    chunks=(min(batch_size, frames), 512), compression="gzip",
                    compression_opts=4, shuffle=True,
                )
                depth_features = group.create_dataset(
                    "depth_resnet18", shape=(frames, 512), dtype=np.float32,
                    chunks=(min(batch_size, frames), 512), compression="gzip",
                    compression_opts=4, shuffle=True,
                )
                fused = group.create_dataset(
                    "rgbd_resnet18", shape=(frames, 1024), dtype=np.float32,
                    chunks=(min(batch_size, frames), 1024), compression="gzip",
                    compression_opts=4, shuffle=True,
                )
                for start in range(0, frames, batch_size):
                    stop = min(start + batch_size, frames)
                    rgb_batch = torch.from_numpy(np.asarray(rgb[start:stop])).to(resolved_device)
                    depth_batch = _depth_pseudo_rgb(
                        np.asarray(depth[start:stop]), depth_scale
                    ).to(resolved_device)
                    rgb_feature = encoder(rgb_batch)
                    depth_feature = encoder(depth_batch)
                    feature = torch.cat((rgb_feature, depth_feature), dim=-1)
                    rgb_features[start:stop] = rgb_feature.cpu().numpy()
                    depth_features[start:stop] = depth_feature.cpu().numpy()
                    fused[start:stop] = feature.cpu().numpy()
                trajectory_count += 1
                frame_count += frames
        output_file.attrs["complete"] = True

    return CacheSummary(
        source_path=str(source.resolve()), output_path=str(output.resolve()),
        source_sha256=source_digest, trajectories=trajectory_count,
        frames=frame_count,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--depth-scale", type=float, default=10000.0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = cache_rgbd_features(
        args.source_h5, args.output_h5, batch_size=args.batch_size,
        depth_scale=args.depth_scale, max_episodes=args.max_episodes,
        device=args.device, overwrite=args.overwrite,
    )
    print(summary)
