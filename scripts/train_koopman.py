#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from antmaze_ac.config import load_config
from antmaze_ac.data.build_sequences import Normalizer
from antmaze_ac.data.windows import KoopmanWindowDataset, load_npz_dataset
from antmaze_ac.koopman.checkpoint import load_checkpoint, save_checkpoint, sha256
from antmaze_ac.koopman.losses import koopman_loss
from antmaze_ac.koopman.model import DeepKoopman


def initialize_wandb(config: dict, output: Path, requested_mode: str | None):
    tracking = config.get("tracking", {})
    if tracking.get("provider") != "wandb":
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError("Install the 'tracking' extra to enable required W&B logging") from error

    mode = requested_mode or tracking.get("mode", "auto")
    if mode == "auto":
        try:
            mode = "online" if bool(wandb.Api(timeout=5).api_key) else "offline"
        except Exception:
            mode = "offline"
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported W&B mode {mode!r}")
    run_id_path = output / "wandb_run_id.txt"
    run_id = run_id_path.read_text(encoding="utf-8").strip() if run_id_path.exists() else wandb.util.generate_id()
    run = wandb.init(
        project=tracking["project"],
        entity=tracking.get("entity"),
        name=tracking.get("run_name"),
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(output),
        config=config,
        job_type="koopman-training",
        tags=["antmaze-umaze-v2", "fullA_history_v2", "batch4096", "formal"],
    )
    run_id_path.write_text(run.id + "\n", encoding="utf-8")
    tracking_status = {
        "provider": "wandb",
        "mode": mode,
        "run_id": run.id,
        "run_name": run.name,
        "project": tracking["project"],
        "url": run.url,
        "sync_command": None
        if mode == "online"
        else f"wandb sync {Path(run.dir).resolve().parent}",
    }
    (output / "wandb_status.json").write_text(
        json.dumps(tracking_status, indent=2),
        encoding="utf-8",
    )
    return run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def average_loss(model, loader, device, loss_kwargs):
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for states, actions in loader:
            states, actions = states.to(device), actions.to(device)
            losses = koopman_loss(model, states, actions, **loss_kwargs)
            batch = len(states)
            count += batch
            for key, value in losses.scalars().items():
                totals[key] = totals.get(key, 0.0) + value * batch
    return {key: value / count for key, value in totals.items()}


def reconcile_history(history_path: Path, start_epoch: int) -> tuple[int | None, int]:
    """Keep one authoritative row per completed checkpoint epoch on resume."""

    if not history_path.exists():
        return None, 0
    rows_by_epoch: dict[int, dict] = {}
    row_count = 0
    for line_number, line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            epoch = int(row["epoch"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Malformed history row {line_number} in {history_path}"
            ) from error
        row_count += 1
        if epoch < start_epoch:
            # A later resumed segment is authoritative if an old run already
            # produced a duplicate epoch.
            rows_by_epoch[epoch] = row
    retained = [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]
    serialized = "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in retained
    )
    temporary = history_path.with_suffix(".jsonl.reconcile")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(history_path)
    best_epoch = (
        min(retained, key=lambda row: float(row["validation"]["total"]))["epoch"]
        if retained
        else None
    )
    return None if best_epoch is None else int(best_epoch), row_count - len(retained)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/antmaze_umaze.yaml")
    parser.add_argument("--data", default="data/processed/antmaze-umaze-v2")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=None, help="Smoke override; formal default comes from YAML")
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-train-windows", type=int, default=None, help="Smoke-test subset")
    parser.add_argument("--max-validation-windows", type=int, default=None, help="Smoke-test subset")
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default=None,
        help="Override tracking.mode; auto uses online only when an API key is configured",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    koopman_config = config["koopman"]
    set_seed(config["experiment"]["seed"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    data_root = Path(args.data)
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    normalizer = Normalizer(
        np.asarray(metadata["normalizer"]["mean"], dtype=np.float32),
        np.asarray(metadata["normalizer"]["std"], dtype=np.float32),
    )
    train_data = load_npz_dataset(data_root / "train.npz")
    validation_data = load_npz_dataset(data_root / "validation.npz")
    k_step = int(koopman_config["K_step"])
    train_windows = KoopmanWindowDataset(
        train_data,
        k_step,
        normalizer,
        np.load(data_root / f"train_K{k_step}_starts.npy"),
    )
    validation_windows = KoopmanWindowDataset(
        validation_data,
        k_step,
        normalizer,
        np.load(data_root / f"validation_K{k_step}_starts.npy"),
    )
    if args.max_train_windows:
        train_windows.starts = train_windows.starts[: args.max_train_windows]
    if args.max_validation_windows:
        validation_windows.starts = validation_windows.starts[: args.max_validation_windows]
    train_loader = DataLoader(
        train_windows,
        batch_size=koopman_config["batch_size"],
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_windows,
        batch_size=koopman_config["eval_batch_size"],
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    state_dim = train_data.x.shape[1]
    action_dim = train_data.delta_action.shape[1]
    output = Path(args.output or config["experiment"]["output_dir"]) / "koopman"
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    if args.resume is None and history_path.exists():
        raise FileExistsError(
            f"{history_path} already exists; use a fresh output or pass --resume"
        )
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    wandb_run = initialize_wandb(config, output, args.wandb_mode)

    model = DeepKoopman(
        state_dim,
        action_dim,
        koopman_config["lift_dim"],
        koopman_config["encoder_hidden_dims"],
        koopman_config["encoder_activation"],
    ).to(device)
    exact_action_integrator = bool(
        koopman_config.get("exact_action_integrator", False)
    )
    if exact_action_integrator:
        model.configure_action_integrator(normalizer.std[-action_dim:])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=koopman_config["learning_rate"],
        weight_decay=koopman_config["weight_decay"],
    )
    start_epoch = 0
    best_validation = float("inf")
    elapsed_before = 0.0
    history_best_epoch = None
    if args.resume:
        model, payload = load_checkpoint(args.resume, map_location=device)
        if (model.state_dim, model.action_dim) != (state_dim, action_dim):
            raise ValueError(
                "Refusing incompatible resume: checkpoint state/action "
                f"dimensions are ({model.state_dim},{model.action_dim}), "
                f"dataset requires ({state_dim},{action_dim})"
            )
        checkpoint_architecture = payload.get("config", {}).get("koopman", {}).get("architecture")
        if checkpoint_architecture != koopman_config["architecture"]:
            raise ValueError(
                "Refusing incompatible resume: checkpoint Koopman architecture is "
                f"{checkpoint_architecture!r}, requested {koopman_config['architecture']!r}. "
                "Start a fresh fullA_history_v2-adapted run."
            )
        checkpoint_exact_integrator = bool(
            payload.get("config", {})
            .get("koopman", {})
            .get("exact_action_integrator", False)
        )
        if checkpoint_exact_integrator != exact_action_integrator:
            raise ValueError(
                "Refusing incompatible resume: exact_action_integrator "
                f"changed from {checkpoint_exact_integrator} to "
                f"{exact_action_integrator}"
            )
        model.to(device)
        if exact_action_integrator:
            model.configure_action_integrator(normalizer.std[-action_dim:])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=koopman_config["learning_rate"],
            weight_decay=koopman_config["weight_decay"],
        )
        if payload["optimizer"] is not None:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_validation = float(payload["best_validation"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        if payload.get("rng_state") is not None:
            restore_rng_state(payload["rng_state"])
        history_best_epoch, removed_history_rows = reconcile_history(
            history_path,
            start_epoch,
        )
        if removed_history_rows:
            print(
                json.dumps(
                    {
                        "event": "history_reconciled",
                        "resume_checkpoint_epoch": start_epoch - 1,
                        "removed_or_duplicate_rows": removed_history_rows,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    loss_weights = koopman_config["loss_weights"]
    loss_kwargs = {
        "rollout_discount": koopman_config["rollout_discount"],
        "linear_weight": loss_weights["linear"],
        "rollout_weight": loss_weights["rollout"],
        "stability_weight": loss_weights["stability"],
        "latent_std_weight": loss_weights["latent_std"],
        "identity_weight": loss_weights["identity"],
        "controllability_svd_weight": loss_weights["controllability_svd"],
        "augmentation_weight": loss_weights["augmentation"],
        "reconstruction_weight": loss_weights["reconstruction"],
        "spectral_radius_limit": koopman_config["spectral_radius_limit"],
        "target_latent_std": koopman_config["target_latent_std"],
    }
    max_epochs = args.max_epochs or int(koopman_config["max_epochs"])
    max_hours = args.max_wall_time_hours or float(koopman_config["max_wall_time_hours"])
    wall_limit = max_hours * 3600
    started = time.monotonic()
    best_epoch = (
        int(history_best_epoch)
        if history_best_epoch is not None
        else start_epoch - 1
    )
    stop_reason = "max_epochs"
    last_epoch = start_epoch - 1
    normalizers = {"state": normalizer.state_dict(), "delta_action": "physical_units"}

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_sums: dict[str, float] = {}
        train_count = 0
        gradient_sums = {
            "gradient_norm_before_clip": 0.0,
            "A_gradient_norm": 0.0,
            "B_gradient_norm": 0.0,
            "encoder_gradient_norm": 0.0,
        }
        gradient_batches = 0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = koopman_loss(model, states, actions, **loss_kwargs)
            losses.total.backward()
            encoder_squared = torch.zeros((), device=device)
            for parameter in model.encoder.parameters():
                if parameter.grad is not None:
                    encoder_squared = encoder_squared + parameter.grad.detach().square().sum()
            gradient_sums["A_gradient_norm"] += float(model.A.grad.detach().norm())
            gradient_sums["B_gradient_norm"] += float(model.B.grad.detach().norm())
            gradient_sums["encoder_gradient_norm"] += float(torch.sqrt(encoder_squared))
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), koopman_config["gradient_clip_norm"]
            )
            gradient_sums["gradient_norm_before_clip"] += float(gradient_norm)
            gradient_batches += 1
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite Koopman gradient")
            optimizer.step()
            if exact_action_integrator:
                model.project_action_integrator()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("Koopman parameter became NaN or Inf after optimizer step")
            batch = len(states)
            train_count += batch
            for key, value in losses.scalars().items():
                train_sums[key] = train_sums.get(key, 0.0) + value * batch
        train_metrics = {key: value / train_count for key, value in train_sums.items()}
        train_metrics.update(
            {key: value / max(gradient_batches, 1) for key, value in gradient_sums.items()}
        )
        validation_metrics = average_loss(model, validation_loader, device, loss_kwargs)
        elapsed = elapsed_before + time.monotonic() - started
        row = {
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if wandb_run is not None:
            wandb_metrics = {
                "epoch": epoch,
                "elapsed_seconds": elapsed,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"validation/{key}": value for key, value in validation_metrics.items()},
            }
            wandb_run.log(wandb_metrics, step=epoch)
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            save_checkpoint(
                output / "best_validation.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation=best_validation,
                config=config,
                normalizers=normalizers,
                elapsed_seconds=elapsed,
                rng_state=capture_rng_state(),
            )
        if (epoch + 1) % int(koopman_config["checkpoint_interval"]) == 0:
            save_checkpoint(
                output / f"recovery_epoch_{epoch:04d}.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation=best_validation,
                config=config,
                normalizers=normalizers,
                elapsed_seconds=elapsed,
                rng_state=capture_rng_state(),
            )
        last_epoch = epoch
        print(json.dumps(row, sort_keys=True), flush=True)
        if elapsed >= wall_limit:
            stop_reason = "max_wall_time"
            break

    elapsed = elapsed_before + time.monotonic() - started
    save_checkpoint(
        output / "last.pt",
        model,
        optimizer=optimizer,
        epoch=last_epoch,
        best_validation=best_validation,
        config=config,
        normalizers=normalizers,
        elapsed_seconds=elapsed,
        rng_state=capture_rng_state(),
    )
    if not (output / "best_validation.pt").exists():
        raise RuntimeError("Training ended without producing best_validation.pt")
    status = {
        "actual_epochs_this_run": max(0, last_epoch - start_epoch + 1),
        "last_epoch": last_epoch,
        "elapsed_seconds_total": elapsed,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "stop_reason": stop_reason,
        "best_checkpoint_sha256": sha256(output / "best_validation.pt"),
        "last_checkpoint_sha256": sha256(output / "last.pt"),
        "device": str(device),
        "batch_size": int(koopman_config["batch_size"]),
        "eval_batch_size": int(koopman_config["eval_batch_size"]),
        "architecture": koopman_config["architecture"],
    }
    (output / "training_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update(status)
        if config["tracking"].get("log_checkpoint_artifact", True):
            artifact = __import__("wandb").Artifact(
                name=f"{config['tracking']['run_name']}-checkpoints",
                type="model",
                metadata=status,
            )
            for artifact_path in (
                output / "best_validation.pt",
                output / "last.pt",
                output / "training_status.json",
                output / "resolved_config.json",
            ):
                artifact.add_file(str(artifact_path), name=artifact_path.name)
            wandb_run.log_artifact(artifact)
        wandb_run.finish()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
