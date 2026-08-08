"""Run the four core visual Koopman ablations on one fixed data split.

The experiment matrix is intentionally small and ordered:

``visual_latent_dim in {16, 32} x transform in {identity, learned_orthogonal}``.

``learned_orthogonal`` learns the square linear transform
``T=exp(S-S^T)`` and uses the exact inverse ``C=T^T``.  The hard orthogonality
constraint makes the comparison immune to the coordinate-scale collapse of a
free invertible transform.  Every run receives the same trajectory, feature
cache, split seed, and optimization settings.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.maniskill_pick_visual.train_visual_koopman import (
    TrainConfig,
    train,
)


SUMMARY_NAME = "ablation_summary.json"
SCHEMA = "acmpc.visual-koopman-ablation.v1"


@dataclass(frozen=True)
class AblationSpec:
    visual_latent_dim: int
    transform: str

    @property
    def name(self) -> str:
        return f"latent{self.visual_latent_dim}_{self.transform}"

    @property
    def train_transform_mode(self) -> str:
        if self.transform not in {"identity", "learned_orthogonal"}:
            raise ValueError(f"Unknown transform ablation {self.transform!r}")
        return self.transform


ABLATION_SPECS = tuple(
    AblationSpec(latent, transform)
    for latent in (16, 32)
    for transform in ("identity", "learned_orthogonal")
)


@dataclass(frozen=True)
class AblationConfig:
    trajectory_h5: Path
    feature_h5: Path
    output_dir: Path
    horizon: int = 20
    epochs: int = 250
    patience: int = 50
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-6
    seed: int = 43
    workers: int = 0
    device: str = "auto"
    preload: bool = False
    wandb_offline: bool = True
    wandb_project: str = "acmpc-visual-pickcube"
    wandb_group: str | None = None

    def validated(self) -> "AblationConfig":
        trajectory_h5 = self.trajectory_h5.expanduser().resolve()
        feature_h5 = self.feature_h5.expanduser().resolve()
        output_dir = self.output_dir.expanduser().resolve()
        if not trajectory_h5.is_file():
            raise FileNotFoundError(trajectory_h5)
        if not feature_h5.is_file():
            raise FileNotFoundError(feature_h5)
        positive = {
            "horizon": self.horizon,
            "epochs": self.epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if self.lr_patience < 0:
            raise ValueError("lr_patience must be non-negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.lr_factor) or not 0.0 < self.lr_factor < 1.0:
            raise ValueError("lr_factor must lie in (0, 1)")
        if not math.isfinite(self.min_lr) or not 0.0 <= self.min_lr <= self.learning_rate:
            raise ValueError("min_lr must lie in [0, learning_rate]")
        if not self.device:
            raise ValueError("device must not be empty")
        if not self.wandb_project:
            raise ValueError("wandb_project must not be empty")
        return AblationConfig(
            trajectory_h5=trajectory_h5,
            feature_h5=feature_h5,
            output_dir=output_dir,
            horizon=int(self.horizon),
            epochs=int(self.epochs),
            patience=int(self.patience),
            batch_size=int(self.batch_size),
            learning_rate=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
            lr_factor=float(self.lr_factor),
            lr_patience=int(self.lr_patience),
            min_lr=float(self.min_lr),
            seed=int(self.seed),
            workers=int(self.workers),
            device=self.device,
            preload=bool(self.preload),
            wandb_offline=bool(self.wandb_offline),
            wandb_project=self.wandb_project,
            wandb_group=self.wandb_group,
        )


def _train_config(
    config: AblationConfig,
    spec: AblationSpec,
) -> tuple[TrainConfig, dict[str, Any]]:
    """Construct a TrainConfig while tolerating not-yet-landed future fields."""

    available = {field.name for field in fields(TrainConfig)}
    required: dict[str, Any] = {
        "trajectory_h5": config.trajectory_h5,
        "feature_h5": config.feature_h5,
        "output_dir": config.output_dir / spec.name,
        "epochs": config.epochs,
        "patience": config.patience,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "lr_factor": config.lr_factor,
        "lr_patience": config.lr_patience,
        "min_lr": config.min_lr,
        "horizon": config.horizon,
        "visual_latent_dim": spec.visual_latent_dim,
        "transform_mode": spec.train_transform_mode,
        "seed": config.seed,
        "workers": config.workers,
    }
    missing_required = set(required).difference(available)
    if missing_required:
        raise TypeError(
            "TrainConfig is missing required ablation fields: "
            f"{sorted(missing_required)}"
        )

    optional: dict[str, Any] = {
        "preload": config.preload,
        "wandb_mode": "offline" if config.wandb_offline else "disabled",
        "wandb_offline": config.wandb_offline,
        "wandb_project": config.wandb_project,
        "wandb_group": config.wandb_group,
        "wandb_name": spec.name,
    }
    passed_optional = {
        name: value
        for name, value in optional.items()
        if name in available and value is not None
    }
    train_config = TrainConfig(**required, **passed_optional)
    compatibility = {
        "requested_preload": config.preload,
        "effective_preload": (
            bool(getattr(train_config, "preload")) if "preload" in available else False
        ),
        "train_config_optional_fields_passed": sorted(passed_optional),
        "train_config_optional_fields_unavailable": sorted(
            name for name in optional if name not in available
        ),
    }
    return train_config, compatibility


@contextmanager
def _wandb_environment(
    config: AblationConfig,
    spec: AblationSpec,
) -> Iterator[None]:
    values: dict[str, str] = {
        "WANDB_PROJECT": config.wandb_project,
        "WANDB_NAME": spec.name,
    }
    if config.wandb_group is not None:
        values["WANDB_RUN_GROUP"] = config.wandb_group
    if config.wandb_offline:
        values["WANDB_MODE"] = "offline"
    missing = object()
    previous: dict[str, str | object] = {
        name: os.environ.get(name, missing) for name in values
    }
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)


def _call_train(
    trainer: Callable[..., Mapping[str, Any]],
    train_config: TrainConfig,
    *,
    device: str,
) -> Mapping[str, Any]:
    parameters = inspect.signature(trainer).parameters
    if "device_name" in parameters:
        return trainer(train_config, device_name=device)
    if "device" in parameters:
        return trainer(train_config, device=device)
    return trainer(train_config)


def _horizon_metrics(test_metrics: Mapping[str, Any]) -> dict[str, Any]:
    horizons = test_metrics.get("horizons", {})
    if not isinstance(horizons, Mapping):
        raise TypeError("test_metrics.horizons must be a mapping")
    return {str(step): horizons.get(str(step)) for step in (1, 5, 10, 20)}


def _summarize_run(
    spec: AblationSpec,
    train_config: TrainConfig,
    report: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    test_metrics = report.get("test_metrics")
    if not isinstance(test_metrics, Mapping):
        raise KeyError(f"{spec.name} report is missing test_metrics")
    action_ablation = test_metrics.get("one_step_action_ablation")
    if not isinstance(action_ablation, Mapping):
        raise KeyError(f"{spec.name} report is missing one_step_action_ablation")
    required = ("elapsed_seconds",)
    missing = [name for name in required if name not in report]
    if missing:
        raise KeyError(f"{spec.name} report is missing {missing}")
    if "spectral_radius" not in test_metrics:
        raise KeyError(f"{spec.name} report is missing spectral_radius")
    compatibility_values = dict(compatibility)
    if "best_validation_observable" in report:
        best_validation_observable = float(report["best_validation_observable"])
        compatibility_values["validation_metric_source"] = (
            "best_validation_observable"
        )
    elif "best_validation_total" in report:
        # Reports produced before the observable metric was introduced only
        # expose the full validation objective.  Keep them readable while
        # making the fallback explicit in the ablation summary.
        best_validation_observable = float(report["best_validation_total"])
        compatibility_values["validation_metric_source"] = (
            "legacy_best_validation_total"
        )
    else:
        raise KeyError(
            f"{spec.name} report is missing best_validation_observable"
        )
    if "best_validation_total_at_best" in report:
        best_validation_total_at_best = float(
            report["best_validation_total_at_best"]
        )
    elif "best_validation_total" in report:
        best_validation_total_at_best = float(report["best_validation_total"])
    else:
        raise KeyError(
            f"{spec.name} report is missing best_validation_total_at_best"
        )
    spectral_radius = float(test_metrics["spectral_radius"])
    elapsed = float(report["elapsed_seconds"])
    if not all(
        map(
            math.isfinite,
            (
                best_validation_observable,
                best_validation_total_at_best,
                spectral_radius,
                elapsed,
            ),
        )
    ):
        raise ValueError(f"{spec.name} report contains non-finite summary values")
    horizons = _horizon_metrics(test_metrics)
    test_loss = report.get("test_loss", {})
    if not isinstance(test_loss, Mapping):
        raise TypeError(f"{spec.name} report.test_loss must be a mapping")
    checkpoint = report.get("checkpoint")
    return {
        "name": spec.name,
        "run_dir": str(train_config.output_dir),
        "visual_latent_dim": spec.visual_latent_dim,
        "latent": spec.visual_latent_dim,
        "transform": spec.transform,
        "train_transform_mode": spec.train_transform_mode,
        "best_epoch": report.get("best_epoch"),
        "validation_selection_metric": "observable_validation",
        "best_validation_observable": best_validation_observable,
        "best_validation_total_at_best": best_validation_total_at_best,
        # Compatibility aliases.  ``best_val`` follows the ranking metric;
        # ``best_validation_total`` remains the full training objective.
        "best_validation_total": best_validation_total_at_best,
        "best_val": best_validation_observable,
        "epochs_completed": report.get("epochs_completed"),
        "stopped_early": report.get("stopped_early"),
        "final_lr": report.get("final_lr"),
        "test_loss": dict(test_loss),
        "test_horizons": horizons,
        "test": horizons,
        "action_ablation": dict(action_ablation),
        "spectral_radius": spectral_radius,
        "rho": spectral_radius,
        "elapsed_seconds": elapsed,
        "elapsed": elapsed,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "compatibility": compatibility_values,
    }


def run_ablation(
    config: AblationConfig,
    *,
    trainer: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all ablations in fixed order and atomically write one summary."""

    config = config.validated()
    trainer = train if trainer is None else trainer
    summary_path = config.output_dir / SUMMARY_NAME
    run_dirs = [config.output_dir / spec.name for spec in ABLATION_SPECS]
    occupied = [path for path in [summary_path, *run_dirs] if path.exists()]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite existing ablation outputs: "
            + ", ".join(map(str, occupied))
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []
    for spec in ABLATION_SPECS:
        train_config, compatibility = _train_config(config, spec)
        with _wandb_environment(config, spec):
            report = _call_train(
                trainer,
                train_config,
                device=config.device,
            )
        run_summaries.append(
            _summarize_run(spec, train_config, report, compatibility)
        )

    best = min(
        run_summaries,
        key=lambda run: run["best_validation_observable"],
    )
    serialized_config = {
        **asdict(config),
        "trajectory_h5": str(config.trajectory_h5),
        "feature_h5": str(config.feature_h5),
        "output_dir": str(config.output_dir),
    }
    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": serialized_config,
        "run_order": [spec.name for spec in ABLATION_SPECS],
        "best_run_by_observable": best["name"],
        # Backward-compatible name; validation ranking is now observable-only.
        "best_run_by_validation": best["name"],
        "runs": run_summaries,
    }
    # Mode "x" preserves the same no-overwrite contract even if another
    # process races the preflight check.
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _parse_args() -> AblationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-h5", type=Path, required=True)
    parser.add_argument("--feature-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request dataset preload when supported by TrainConfig.",
    )
    parser.add_argument(
        "--wandb-offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local W&B offline logging (default: enabled).",
    )
    parser.add_argument(
        "--wandb-project",
        default="acmpc-visual-pickcube",
    )
    parser.add_argument("--wandb-group")
    args = parser.parse_args()
    return AblationConfig(
        trajectory_h5=args.trajectory_h5,
        feature_h5=args.feature_h5,
        output_dir=args.output_dir,
        horizon=args.horizon,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
        seed=args.seed,
        workers=args.workers,
        device=args.device,
        preload=args.preload,
        wandb_offline=args.wandb_offline,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
    )


if __name__ == "__main__":
    run_ablation(_parse_args())
