"""Cache frozen ImageNet ResNet-18 features for a visual trajectory HDF5.

The input schema is ``traj_i/rgb`` with RGB observations shaped either
``[T+1, H, W, 3]`` or ``[T+1, 3, H, W]``. The output is a sidecar HDF5 with
one ``traj_i/resnet18`` float32 dataset of shape ``[T+1, 512]`` per episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from antmaze_ac.vision import FrozenResNet18


SCHEMA_VERSION = 1
_TRAJECTORY_PATTERN = re.compile(r"traj_(\d+)$")


@dataclass(frozen=True)
class CacheSummary:
    source_path: str
    output_path: str
    source_sha256: str
    trajectories: int
    frames: int
    feature_dim: int = FrozenResNet18.output_dim


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_output_path(source_path: str | Path) -> Path:
    source = Path(source_path)
    return source.with_name(f"{source.stem}.resnet18.h5")


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {resolved} requested but CUDA is unavailable")
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


def _validate_rgb(dataset: h5py.Dataset, trajectory: str) -> None:
    if dataset.ndim != 4:
        raise ValueError(
            f"{trajectory}/rgb must have rank 4, got shape {dataset.shape}"
        )
    if dataset.shape[0] < 1:
        raise ValueError(f"{trajectory}/rgb must contain at least one frame")
    if dataset.shape[1] != 3 and dataset.shape[-1] != 3:
        raise ValueError(
            f"{trajectory}/rgb must be NHWC or NCHW RGB, got shape {dataset.shape}"
        )
    if dataset.dtype != np.uint8 and not np.issubdtype(dataset.dtype, np.floating):
        raise TypeError(
            f"{trajectory}/rgb must be uint8 or floating, got {dataset.dtype}"
        )


def _write_metadata(
    output: h5py.File,
    *,
    source: Path,
    source_digest: str,
    device: torch.device,
) -> None:
    values: dict[str, object] = {
        "schema": "acmpc.resnet18-feature-cache",
        "schema_version": SCHEMA_VERSION,
        "source_path": str(source.resolve()),
        "source_sha256": source_digest,
        "encoder": "resnet18",
        "weights": "IMAGENET1K_V1",
        "feature_dim": FrozenResNet18.output_dim,
        "resize_height": 224,
        "resize_width": 224,
        "device": str(device),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    for key, value in values.items():
        output.attrs[key] = value
    metadata = output.create_group("metadata")
    for key, value in values.items():
        metadata.attrs[key] = value
    metadata.attrs["imagenet_mean"] = np.asarray(
        (0.485, 0.456, 0.406), dtype=np.float32
    )
    metadata.attrs["imagenet_std"] = np.asarray(
        (0.229, 0.224, 0.225), dtype=np.float32
    )


def cache_resnet_features(
    source_path: str | Path,
    output_path: str | Path | None = None,
    *,
    batch_size: int = 256,
    device: str | torch.device = "auto",
    overwrite: bool = False,
    encoder: nn.Module | None = None,
) -> CacheSummary:
    """Create a ResNet-18 feature sidecar without mutating the source file."""

    source = Path(source_path)
    output = default_output_path(source) if output_path is None else Path(output_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"Source trajectory does not exist: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("The sidecar output must differ from the source trajectory")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists and overwrite is disabled: {output}"
        )

    resolved_device = _resolve_device(device)
    source_digest = sha256(source)
    feature_encoder = FrozenResNet18() if encoder is None else encoder
    feature_encoder = feature_encoder.to(resolved_device)
    feature_encoder.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    trajectory_count = 0
    frame_count = 0
    with h5py.File(source, "r") as source_file, h5py.File(output, mode) as output_file:
        output_file.attrs["complete"] = False
        _write_metadata(
            output_file,
            source=source,
            source_digest=source_digest,
            device=resolved_device,
        )
        with torch.inference_mode():
            for trajectory in _trajectory_names(source_file):
                source_group = source_file[trajectory]
                if "rgb" not in source_group:
                    raise KeyError(f"Source is missing {trajectory}/rgb")
                rgb = source_group["rgb"]
                if not isinstance(rgb, h5py.Dataset):
                    raise TypeError(f"{trajectory}/rgb must be an HDF5 dataset")
                _validate_rgb(rgb, trajectory)

                frames = int(rgb.shape[0])
                group = output_file.create_group(trajectory)
                features = group.create_dataset(
                    "resnet18",
                    shape=(frames, FrozenResNet18.output_dim),
                    dtype=np.float32,
                    chunks=(min(batch_size, frames), FrozenResNet18.output_dim),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                for start in range(0, frames, batch_size):
                    stop = min(start + batch_size, frames)
                    images = torch.from_numpy(np.asarray(rgb[start:stop])).to(
                        resolved_device
                    )
                    batch_features = feature_encoder(images)
                    if batch_features.shape != (
                        stop - start,
                        FrozenResNet18.output_dim,
                    ):
                        raise RuntimeError(
                            "Encoder returned shape "
                            f"{tuple(batch_features.shape)}, expected "
                            f"{(stop - start, FrozenResNet18.output_dim)}"
                        )
                    if not bool(torch.isfinite(batch_features).all()):
                        raise FloatingPointError(
                            f"Encoder produced NaN or Inf for {trajectory}"
                        )
                    features[start:stop] = (
                        batch_features.detach().to(device="cpu", dtype=torch.float32).numpy()
                    )
                trajectory_count += 1
                frame_count += frames
        output_file.attrs["trajectory_count"] = trajectory_count
        output_file.attrs["frame_count"] = frame_count
        output_file.attrs.modify("complete", True)

    return CacheSummary(
        source_path=str(source.resolve()),
        output_path=str(output.resolve()),
        source_sha256=source_digest,
        trajectories=trajectory_count,
        frames=frame_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing sidecar. By default an existing output is refused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = cache_resnet_features(
        args.source,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
