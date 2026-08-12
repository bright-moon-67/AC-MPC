"""Resumable, fail-fast orchestration for the five-method O2O matrix.

This module deliberately does not import the trainer, evaluator, learner, or
environment.  Every expensive operation is an explicit child process whose
exact argv, PID, log, timestamps, and return code are recorded atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, IO, Mapping, Sequence

from experiments.dmc.o2o.config import METHODS, O2OConfig


MANIFEST_KIND = "acmpc_dmc_o2o_matrix_manifest_v1"
STATUS_KIND = "acmpc_dmc_o2o_matrix_status_v1"
RUN_KIND = "acmpc_dmc_o2o_run_v1"
EVALUATION_KIND = "acmpc_dmc_o2o_checkpoint_evaluation_v1"
AGGREGATE_KIND = "acmpc_dmc_o2o_aggregate_v1"
MATRIX_METHODS = tuple(METHODS)
MPVE_METHOD = "Cal-RLPD-AC-KMPC-MPVE"
MPVE_SOURCE_METHOD = "Cal-RLPD-AC-KMPC"
DEFAULT_SEEDS = (20260821, 20260822, 20260823)
CHILD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

_TRAINING_SOURCE_FILES = (
    "experiments/dmc/o2o/__init__.py",
    "experiments/dmc/o2o/checkpoint.py",
    "experiments/dmc/o2o/config.py",
    "experiments/dmc/o2o/dataset.py",
    "experiments/dmc/o2o/koopman.py",
    "experiments/dmc/o2o/learner.py",
    "experiments/dmc/o2o/networks.py",
    "experiments/dmc/o2o/train.py",
    "experiments/dmc/ppo/vector_env.py",
    "experiments/dmc/reward_oracle.py",
)
_RESULT_SOURCE_FILES = (
    "experiments/dmc/o2o/evaluate.py",
    "experiments/dmc/o2o/aggregate.py",
    "experiments/dmc/o2o/plot.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required file does not exist: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_seeds(text: str) -> tuple[int, ...]:
    pieces = [piece.strip() for piece in text.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError("--seeds must be a comma-separated integer list")
    try:
        seeds = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds contains a non-integer") from exc
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("--seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    return seeds


@dataclass(frozen=True)
class MatrixSpec:
    repo_root: Path
    dataset: Path
    koopman: Path
    root: Path
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    device: str = "cuda"
    eval_device: str = "cpu"
    offline_updates: int = 500_000
    online_steps: int = 100_000
    online_utd: int = 20
    num_envs: int = 5
    env_workers: int = 5
    cql_weight: float = 0.01
    eval_episodes: int = 10
    max_parallel: int = 1
    python: str = sys.executable

    def resolved(self) -> "MatrixSpec":
        executable = shutil.which(self.python)
        if executable is None:
            candidate = Path(self.python).expanduser()
            if candidate.is_file():
                executable = str(candidate.resolve())
            else:
                raise FileNotFoundError(f"Python executable does not exist: {self.python}")
        result = replace(
            self,
            repo_root=self.repo_root.expanduser().resolve(),
            dataset=self.dataset.expanduser().resolve(),
            koopman=self.koopman.expanduser().resolve(),
            root=self.root.expanduser().resolve(),
            python=str(Path(executable).resolve()),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.repo_root.is_dir() or not (self.repo_root / ".git").exists():
            raise ValueError(f"Not an AC-MPC git worktree: {self.repo_root}")
        if not self.dataset.is_file():
            raise FileNotFoundError(f"Offline dataset does not exist: {self.dataset}")
        if not self.koopman.is_file():
            raise FileNotFoundError(f"Koopman artifact does not exist: {self.koopman}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique tuple")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.seeds
        ):
            raise ValueError("seeds must contain non-negative integers")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.eval_device not in {"cpu", "cuda", "auto"}:
            raise ValueError("eval_device must be cpu, cuda, or auto")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        if self.eval_episodes != 10:
            raise ValueError("The formal result protocol requires exactly 10 episodes")
        for seed in self.seeds:
            for method in MATRIX_METHODS:
                self.config(method, seed).validate()

    def config(self, method: str, seed: int) -> O2OConfig:
        return O2OConfig(
            method=method,
            seed=seed,
            device=self.device,
            offline_updates=self.offline_updates,
            online_steps=self.online_steps,
            online_utd=self.online_utd,
            num_envs=self.num_envs,
            env_workers=self.env_workers,
            cql_weight=self.cql_weight,
            eval_episodes=self.eval_episodes,
        )

    def run_dir(self, seed: int, method: str) -> Path:
        return self.root / f"seed_{seed}" / method

    def offline_source(self, seed: int) -> Path:
        return self.run_dir(seed, MPVE_SOURCE_METHOD) / "offline.pt"


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    argv: tuple[str, ...]
    log_path: Path
    output_paths: tuple[Path, ...]
    seed: int | None = None
    method: str | None = None
    run_dir: Path | None = None
    depends_on: tuple[str, ...] = ()
    artifact_dependency: Path | None = None
    mode_at_plan: str = "run"

    def manifest_dict(self, repo_root: Path) -> dict[str, Any]:
        def display(path: Path) -> str:
            try:
                return str(path.relative_to(repo_root))
            except ValueError:
                return str(path)

        dependency: dict[str, Any] | None = None
        if self.artifact_dependency is not None:
            dependency = {
                "kind": "atomic_artifact_exists",
                "path": display(self.artifact_dependency),
                "validated_by_child": "experiments.dmc.o2o.train",
            }
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "seed": self.seed,
            "method": self.method,
            "argv": list(self.argv),
            "cwd": str(repo_root),
            "log": display(self.log_path),
            "outputs": [display(path) for path in self.output_paths],
            "depends_on": list(self.depends_on),
            "artifact_dependency": dependency,
            "mode_at_plan": self.mode_at_plan,
        }


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _source_paths(repo_root: Path, explicit: Sequence[str]) -> tuple[Path, ...]:
    paths = [repo_root / relative for relative in explicit]
    paths.extend(sorted((repo_root / "experiments/dmc/tasks").rglob("*.py")))
    unique = tuple(sorted(set(path.resolve() for path in paths)))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source snapshot is missing {missing[0]}")
    return unique


def _source_snapshot(repo_root: Path, explicit: Sequence[str]) -> dict[str, Any]:
    paths = _source_paths(repo_root, explicit)
    entries = [
        {
            "path": str(path.relative_to(repo_root)),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    porcelain = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *(str(path.relative_to(repo_root)) for path in paths),
    )
    lines = porcelain.splitlines() if porcelain else []
    return {
        "sha256": _json_fingerprint({"files": entries}),
        "files": entries,
        "git_porcelain": lines,
        "git_porcelain_sha256": hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
    }


def _git_identity(repo_root: Path) -> dict[str, Any]:
    branch = _git(repo_root, "branch", "--show-current") or "DETACHED"
    porcelain = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    lines = porcelain.splitlines() if porcelain else []
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": branch,
        "dirty": bool(lines),
        "porcelain": lines,
        "porcelain_sha256": hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
    }


def _runtime_identity() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    # Some hosting shells export OMP_NUM_THREADS=0.  Importing PyTorch under
    # that value emits a libgomp warning before the runner can launch its
    # correctly isolated children, so make the probe itself safe as well.
    previous_threads = {key: os.environ.get(key) for key in CHILD_ENVIRONMENT}
    os.environ.update(CHILD_ENVIRONMENT)
    try:
        import torch

        result.update(
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
            cuda_available=bool(torch.cuda.is_available()),
            cuda_device_count=int(torch.cuda.device_count()),
        )
        if torch.cuda.is_available():
            result["cuda_devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover - only for broken installations
        result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for key, value in previous_threads.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return result


def _identity_bundle(spec: MatrixSpec) -> dict[str, Any]:
    git = _git_identity(spec.repo_root)
    training_source = _source_snapshot(spec.repo_root, _TRAINING_SOURCE_FILES)
    result_source = _source_snapshot(spec.repo_root, _RESULT_SOURCE_FILES)
    runner_path = Path(__file__).resolve()
    return {
        "git": git,
        "training_source": training_source,
        "result_source": result_source,
        "runner": {
            "path": str(runner_path.relative_to(spec.repo_root)),
            "sha256": _sha256_file(runner_path),
        },
        "runtime": _runtime_identity(),
    }


def _assert_source_unchanged(
    spec: MatrixSpec, identity: Mapping[str, Any], *, include_results: bool
) -> None:
    current_git = _git_identity(spec.repo_root)
    original_git = identity["git"]
    for key in ("commit", "branch"):
        if current_git.get(key) != original_git.get(key):
            raise RuntimeError(f"Git {key} changed while the matrix runner was active")
    current_training = _source_snapshot(spec.repo_root, _TRAINING_SOURCE_FILES)
    if current_training["sha256"] != identity["training_source"]["sha256"]:
        raise RuntimeError("O2O training source changed; refusing to mix implementations")
    if (
        current_training["git_porcelain_sha256"]
        != identity["training_source"]["git_porcelain_sha256"]
    ):
        raise RuntimeError("O2O training source dirty status changed during the matrix")
    if include_results:
        current_results = _source_snapshot(spec.repo_root, _RESULT_SOURCE_FILES)
        if current_results["sha256"] != identity["result_source"]["sha256"]:
            raise RuntimeError("O2O result source changed; refusing to mix result protocols")
        if (
            current_results["git_porcelain_sha256"]
            != identity["result_source"]["git_porcelain_sha256"]
        ):
            raise RuntimeError("O2O result source dirty status changed during the matrix")
    runner_path = spec.repo_root / identity["runner"]["path"]
    if _sha256_file(runner_path) != identity["runner"]["sha256"]:
        raise RuntimeError("O2O matrix runner changed while it was active")


def _expected_dataset(spec: MatrixSpec) -> dict[str, str]:
    return {"path": str(spec.dataset), "sha256": _sha256_file(spec.dataset)}


def _expected_koopman(spec: MatrixSpec) -> dict[str, str]:
    return {"path": str(spec.koopman), "sha256": _sha256_file(spec.koopman)}


def _check_run_identity(
    spec: MatrixSpec,
    *,
    seed: int,
    method: str,
    dataset: Mapping[str, str],
    koopman: Mapping[str, str],
) -> str:
    """Return fresh/resume/completed or reject ambiguous/cross-protocol state."""

    run_dir = spec.run_dir(seed, method)
    metadata_path = run_dir / "run.json"
    known_artifacts = (
        run_dir / "latest.pt",
        run_dir / "best.pt",
        run_dir / "offline.pt",
        run_dir / "metrics.jsonl",
    )
    if not metadata_path.exists():
        if any(path.exists() for path in known_artifacts):
            raise RuntimeError(f"Run artifacts exist without run.json: {run_dir}")
        return "fresh"
    metadata = _read_json(metadata_path)
    config = spec.config(method, seed)
    if metadata.get("kind") != RUN_KIND:
        raise ValueError(f"Unsupported run kind in {metadata_path}")
    if metadata.get("config") != config.to_dict():
        raise ValueError(f"Existing run config differs: {run_dir}")
    if metadata.get("config_fingerprint") != config.fingerprint:
        raise ValueError(f"Existing run config fingerprint differs: {run_dir}")
    saved_dataset = metadata.get("dataset")
    if not isinstance(saved_dataset, Mapping) or any(
        saved_dataset.get(key) != value for key, value in dataset.items()
    ):
        raise ValueError(f"Existing run dataset identity differs: {run_dir}")
    saved_koopman = metadata.get("koopman")
    if not isinstance(saved_koopman, Mapping) or any(
        saved_koopman.get(key) != value for key, value in koopman.items()
    ):
        raise ValueError(f"Existing run Koopman identity differs: {run_dir}")
    latest = run_dir / "latest.pt"
    if metadata.get("completed") is True:
        expected_offline = config.offline_updates if config.uses_calql else 0
        if metadata.get("offline_updates_completed") != expected_offline:
            raise ValueError(f"Completed run has the wrong offline budget: {run_dir}")
        if metadata.get("online_steps_completed") != config.online_steps:
            raise ValueError(f"Completed run has the wrong online budget: {run_dir}")
        if not latest.is_file():
            raise FileNotFoundError(f"Completed run is missing latest.pt: {run_dir}")
        if method == MPVE_SOURCE_METHOD and not spec.offline_source(seed).is_file():
            raise FileNotFoundError(
                f"Completed AC-KMPC run is missing immutable offline.pt: {run_dir}"
            )
        return "completed"
    if not latest.is_file():
        raise RuntimeError(
            f"Partial run has no resumable latest.pt; preserve it for audit and use "
            f"a fresh run directory: {run_dir}"
        )
    return "resume"


def _evaluation_is_current(
    spec: MatrixSpec,
    *,
    seed: int,
    method: str,
    dataset: Mapping[str, str],
    koopman: Mapping[str, str],
) -> bool:
    path = spec.run_dir(seed, method) / "evaluation_latest_10.json"
    if not path.is_file():
        return False
    try:
        report = _read_json(path)
    except ValueError:
        return False
    latest = spec.run_dir(seed, method) / "latest.pt"
    if not latest.is_file():
        return False
    config = spec.config(method, seed)
    protocol = report.get("evaluation_protocol")
    saved_dataset = report.get("dataset")
    saved_koopman = report.get("koopman")
    return bool(
        report.get("kind") == EVALUATION_KIND
        and report.get("method") == method
        and report.get("training_seed") == seed
        and report.get("checkpoint_name") == "latest"
        and report.get("checkpoint_sha256") == _sha256_file(latest)
        and report.get("config_fingerprint") == config.fingerprint
        and report.get("online_step") == config.online_steps
        and isinstance(protocol, Mapping)
        and protocol.get("deterministic") is True
        and protocol.get("episodes") == 10
        and isinstance(saved_dataset, Mapping)
        and saved_dataset.get("sha256") == dataset["sha256"]
        and isinstance(saved_koopman, Mapping)
        and saved_koopman.get("sha256") == koopman["sha256"]
    )


def _train_argv(
    spec: MatrixSpec, *, seed: int, method: str, mode: str
) -> tuple[str, ...]:
    argv = (
        spec.python,
        "-m",
        "experiments.dmc.o2o.train",
        "--method",
        method,
        "--dataset",
        str(spec.dataset),
        "--koopman",
        str(spec.koopman),
        "--output-dir",
        str(spec.run_dir(seed, method)),
        "--seed",
        str(seed),
        "--device",
        spec.device,
        "--offline-updates",
        str(spec.offline_updates),
        "--online-steps",
        str(spec.online_steps),
        "--online-utd",
        str(spec.online_utd),
        "--num-envs",
        str(spec.num_envs),
        "--env-workers",
        str(spec.env_workers),
        "--cql-weight",
        str(spec.cql_weight),
        "--eval-episodes",
        str(spec.eval_episodes),
    )
    if method == MPVE_METHOD and mode != "resume":
        argv += ("--initialize-from-offline", str(spec.offline_source(seed)))
    return argv


def build_jobs(
    spec: MatrixSpec,
    *,
    dataset: Mapping[str, str],
    koopman: Mapping[str, str],
) -> tuple[Job, ...]:
    jobs: list[Job] = []
    evaluation_ids: list[str] = []
    logs = spec.root / "logs"
    for seed in spec.seeds:
        train_ids: dict[str, str] = {}
        for method in MATRIX_METHODS:
            job_id = f"train.seed_{seed}.{method}"
            mode = _check_run_identity(
                spec,
                seed=seed,
                method=method,
                dataset=dataset,
                koopman=koopman,
            )
            train_ids[method] = job_id
            jobs.append(
                Job(
                    job_id=job_id,
                    stage="train",
                    argv=_train_argv(spec, seed=seed, method=method, mode=mode),
                    log_path=logs / f"{job_id}.log",
                    output_paths=(spec.run_dir(seed, method) / "latest.pt",),
                    seed=seed,
                    method=method,
                    run_dir=spec.run_dir(seed, method),
                    artifact_dependency=(
                        spec.offline_source(seed) if method == MPVE_METHOD else None
                    ),
                    mode_at_plan=mode,
                )
            )
        for method in MATRIX_METHODS:
            job_id = f"evaluate.seed_{seed}.{method}.latest"
            evaluation_ids.append(job_id)
            jobs.append(
                Job(
                    job_id=job_id,
                    stage="evaluate",
                    argv=(
                        spec.python,
                        "-m",
                        "experiments.dmc.o2o.evaluate",
                        "--run-dir",
                        str(spec.run_dir(seed, method)),
                        "--checkpoint",
                        "latest",
                        "--dataset",
                        str(spec.dataset),
                        "--koopman",
                        str(spec.koopman),
                        "--device",
                        spec.eval_device,
                        "--output",
                        str(
                            spec.run_dir(seed, method)
                            / "evaluation_latest_10.json"
                        ),
                    ),
                    log_path=logs / f"{job_id}.log",
                    output_paths=(
                        spec.run_dir(seed, method) / "evaluation_latest_10.json",
                    ),
                    seed=seed,
                    method=method,
                    run_dir=spec.run_dir(seed, method),
                    depends_on=(train_ids[method],),
                    mode_at_plan=(
                        "completed"
                        if _evaluation_is_current(
                            spec,
                            seed=seed,
                            method=method,
                            dataset=dataset,
                            koopman=koopman,
                        )
                        else "run"
                    ),
                )
            )
    aggregate_path = spec.root / "results" / "cartpole_proto1m.json"
    aggregate_id = "aggregate.latest"
    aggregate_argv: tuple[str, ...] = (
        spec.python,
        "-m",
        "experiments.dmc.o2o.aggregate",
        "--output",
        str(aggregate_path),
    )
    for seed in spec.seeds:
        for method in MATRIX_METHODS:
            aggregate_argv += ("--run-dir", str(spec.run_dir(seed, method)))
    jobs.append(
        Job(
            job_id=aggregate_id,
            stage="aggregate",
            argv=aggregate_argv,
            log_path=logs / f"{aggregate_id}.log",
            output_paths=(aggregate_path,),
            depends_on=tuple(evaluation_ids),
        )
    )
    plot_prefix = spec.root / "results" / "cartpole_proto1m"
    jobs.append(
        Job(
            job_id="plot.latest",
            stage="plot",
            argv=(
                spec.python,
                "-m",
                "experiments.dmc.o2o.plot",
                "--aggregate",
                str(aggregate_path),
                "--output-prefix",
                str(plot_prefix),
            ),
            log_path=logs / "plot.latest.log",
            output_paths=(
                plot_prefix.with_suffix(".png"),
                plot_prefix.with_suffix(".pdf"),
            ),
            depends_on=(aggregate_id,),
        )
    )
    return tuple(jobs)


def _manifest(
    spec: MatrixSpec,
    *,
    dataset: Mapping[str, str],
    koopman: Mapping[str, str],
    identity: Mapping[str, Any],
    jobs: Sequence[Job],
) -> dict[str, Any]:
    experiment = {
        "task": "cartpole_swingup",
        "methods": list(MATRIX_METHODS),
        "seeds": list(spec.seeds),
        "dataset": dict(dataset),
        "koopman": dict(koopman),
        "root": str(spec.root),
        "device": spec.device,
        "eval_device": spec.eval_device,
        "offline_updates": spec.offline_updates,
        "online_steps": spec.online_steps,
        "online_utd": spec.online_utd,
        "num_envs": spec.num_envs,
        "env_workers": spec.env_workers,
        "cql_weight": spec.cql_weight,
        "eval_episodes": spec.eval_episodes,
        "max_parallel": spec.max_parallel,
        "python": spec.python,
        "child_environment": dict(CHILD_ENVIRONMENT),
        "inherited_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    reproducibility = {
        "experiment": experiment,
        "git_commit": identity["git"]["commit"],
        "git_branch": identity["git"]["branch"],
        "training_source_sha256": identity["training_source"]["sha256"],
        "result_source_sha256": identity["result_source"]["sha256"],
        "runner_sha256": identity["runner"]["sha256"],
        "runtime": identity["runtime"],
    }
    return {
        "kind": MANIFEST_KIND,
        "created_utc": _utc_now(),
        "matrix_fingerprint": _json_fingerprint(reproducibility),
        "experiment": experiment,
        "source_identity": dict(identity),
        "jobs": [job.manifest_dict(spec.repo_root) for job in jobs],
    }


def _pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _compatible_existing_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return manifest
    existing = _read_json(path)
    if existing.get("kind") != MANIFEST_KIND:
        raise ValueError(f"Unsupported matrix manifest: {path}")
    if existing.get("matrix_fingerprint") != manifest["matrix_fingerprint"]:
        raise RuntimeError(
            "Existing matrix manifest has a different data/config/source identity; "
            "use a new --root"
        )
    manifest["created_utc"] = existing.get("created_utc", manifest["created_utc"])
    manifest["last_invoked_utc"] = _utc_now()
    return manifest


def _initial_status(
    status_path: Path, manifest: Mapping[str, Any], jobs: Sequence[Job]
) -> dict[str, Any]:
    previous: dict[str, Any] | None = None
    if status_path.exists():
        previous = _read_json(status_path)
        if previous.get("kind") != STATUS_KIND:
            raise ValueError(f"Unsupported matrix status: {status_path}")
        if previous.get("matrix_fingerprint") != manifest["matrix_fingerprint"]:
            raise RuntimeError("Existing matrix status belongs to another manifest")
        if previous.get("state") in {"running", "fail_fast_waiting"} and _pid_alive(
            previous.get("runner_pid")
        ):
            raise RuntimeError(
                f"Matrix runner PID {previous['runner_pid']} is still active; refusing "
                "to launch a competing runner"
            )
        for value in previous.get("jobs", {}).values():
            if isinstance(value, Mapping) and value.get("state") == "running" and _pid_alive(
                value.get("pid")
            ):
                raise RuntimeError(
                    f"Child PID {value['pid']} is still active; refusing to duplicate it"
                )
    previous_jobs = previous.get("jobs", {}) if previous else {}
    job_status: dict[str, Any] = {}
    for job in jobs:
        old = previous_jobs.get(job.job_id, {})
        attempts = old.get("attempts", []) if isinstance(old, Mapping) else []
        job_status[job.job_id] = {
            "state": "pending",
            "stage": job.stage,
            "seed": job.seed,
            "method": job.method,
            "pid": None,
            "argv": list(job.argv),
            "log": str(job.log_path),
            "returncode": None,
            "started_utc": None,
            "finished_utc": None,
            "attempts": list(attempts) if isinstance(attempts, list) else [],
        }
    invocations = previous.get("invocations", []) if previous else []
    invocation = {"runner_pid": os.getpid(), "started_utc": _utc_now()}
    return {
        "kind": STATUS_KIND,
        "matrix_fingerprint": manifest["matrix_fingerprint"],
        "state": "initializing",
        "runner_pid": os.getpid(),
        "started_utc": invocation["started_utc"],
        "updated_utc": invocation["started_utc"],
        "finished_utc": None,
        "error": None,
        "invocations": [*invocations, invocation],
        "jobs": job_status,
    }


def _write_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_utc"] = _utc_now()
    _atomic_json(path, status)


def _set_skipped(status: dict[str, Any], job: Job, reason: str) -> None:
    value = status["jobs"][job.job_id]
    value.update(
        state="skipped",
        skip_reason=reason,
        finished_utc=_utc_now(),
        returncode=0,
    )


@dataclass
class _ActiveJob:
    job: Job
    process: Any
    log_handle: IO[str]
    attempt_index: int


PopenFactory = Callable[..., Any]


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(CHILD_ENVIRONMENT)
    return environment


def _start_job(
    job: Job,
    *,
    spec: MatrixSpec,
    identity: Mapping[str, Any],
    status: dict[str, Any],
    status_path: Path,
    popen_factory: PopenFactory,
) -> _ActiveJob:
    _assert_source_unchanged(
        spec, identity, include_results=job.stage != "train"
    )
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log_path.open("a", encoding="utf-8", buffering=1)
    started = _utc_now()
    handle.write(
        "\n=== O2O matrix child ===\n"
        f"started_utc={started}\n"
        f"job_id={job.job_id}\n"
        f"argv={json.dumps(list(job.argv), ensure_ascii=False)}\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    try:
        process = popen_factory(
            list(job.argv),
            cwd=spec.repo_root,
            env=_child_environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        handle.close()
        raise
    value = status["jobs"][job.job_id]
    attempt = {
        "pid": int(process.pid),
        "argv": list(job.argv),
        "started_utc": started,
        "finished_utc": None,
        "returncode": None,
    }
    value["attempts"].append(attempt)
    attempt_index = len(value["attempts"]) - 1
    value.update(
        state="running",
        pid=int(process.pid),
        argv=list(job.argv),
        started_utc=started,
        finished_utc=None,
        returncode=None,
    )
    _write_status(status_path, status)
    return _ActiveJob(job, process, handle, attempt_index)


def _validate_job_output(
    job: Job,
    *,
    spec: MatrixSpec,
    dataset: Mapping[str, str],
    koopman: Mapping[str, str],
) -> None:
    if job.stage == "train":
        assert job.seed is not None and job.method is not None
        mode = _check_run_identity(
            spec,
            seed=job.seed,
            method=job.method,
            dataset=dataset,
            koopman=koopman,
        )
        if mode != "completed":
            raise RuntimeError(f"Training child exited zero but run is {mode}: {job.job_id}")
        return
    if job.stage == "evaluate":
        assert job.seed is not None and job.method is not None
        if not _evaluation_is_current(
            spec,
            seed=job.seed,
            method=job.method,
            dataset=dataset,
            koopman=koopman,
        ):
            raise RuntimeError(
                f"Evaluation child exited zero without a current report: {job.job_id}"
            )
        return
    if job.stage == "aggregate":
        value = _read_json(job.output_paths[0])
        if value.get("kind") != AGGREGATE_KIND:
            raise ValueError("Aggregate child produced an unsupported result")
        return
    if job.stage == "plot":
        missing = [path for path in job.output_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Plot child did not create {missing[0]}")
        return
    raise AssertionError(f"Unknown stage {job.stage}")


def _finish_job(
    active: _ActiveJob,
    returncode: int,
    *,
    spec: MatrixSpec,
    dataset: Mapping[str, str],
    koopman: Mapping[str, str],
    status: dict[str, Any],
    status_path: Path,
) -> str | None:
    active.log_handle.close()
    job = active.job
    value = status["jobs"][job.job_id]
    finished = _utc_now()
    error: str | None = None
    if returncode == 0:
        try:
            _validate_job_output(job, spec=spec, dataset=dataset, koopman=koopman)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = f"child exited with return code {returncode}"
    state = "succeeded" if error is None else "failed"
    value.update(
        state=state,
        pid=int(active.process.pid),
        returncode=int(returncode),
        finished_utc=finished,
    )
    if error is not None:
        value["error"] = error
    attempt = value["attempts"][active.attempt_index]
    attempt.update(finished_utc=finished, returncode=int(returncode))
    if error is not None:
        attempt["error"] = error
    _write_status(status_path, status)
    return error


def _training_dependency_ready(job: Job) -> bool:
    return job.artifact_dependency is None or job.artifact_dependency.is_file()


def _run_training_jobs(
    jobs: Sequence[Job],
    *,
    spec: MatrixSpec,
    dataset: Mapping[str, str],
    koopman: Mapping[str, str],
    identity: Mapping[str, Any],
    status: dict[str, Any],
    status_path: Path,
    popen_factory: PopenFactory,
    sleep_fn: Callable[[float], None],
) -> None:
    pending: list[Job] = []
    for job in jobs:
        if job.mode_at_plan == "completed":
            _set_skipped(status, job, "strictly_valid_completed_run")
        else:
            pending.append(job)
    _write_status(status_path, status)
    active: dict[str, _ActiveJob] = {}
    failures: list[str] = []
    try:
        while pending or active:
            for job_id, running in list(active.items()):
                returncode = running.process.poll()
                if returncode is None:
                    continue
                del active[job_id]
                error = _finish_job(
                    running,
                    int(returncode),
                    spec=spec,
                    dataset=dataset,
                    koopman=koopman,
                    status=status,
                    status_path=status_path,
                )
                if error is not None:
                    failures.append(f"{job_id}: {error}")
            if failures:
                status["state"] = "fail_fast_waiting" if active else "failed"
                _write_status(status_path, status)
            else:
                while len(active) < spec.max_parallel:
                    ready_index = next(
                        (
                            index
                            for index, job in enumerate(pending)
                            if _training_dependency_ready(job)
                        ),
                        None,
                    )
                    if ready_index is None:
                        break
                    job = pending.pop(ready_index)
                    running = _start_job(
                        job,
                        spec=spec,
                        identity=identity,
                        status=status,
                        status_path=status_path,
                        popen_factory=popen_factory,
                    )
                    active[job.job_id] = running
            if not active and pending and not failures:
                blocked = ", ".join(job.job_id for job in pending)
                failures.append(
                    "No runnable training jobs; required AC-KMPC offline.pt is "
                    f"missing for: {blocked}"
                )
                status["state"] = "failed"
                _write_status(status_path, status)
            if failures and not active:
                for job in pending:
                    value = status["jobs"][job.job_id]
                    value.update(
                        state="blocked",
                        finished_utc=_utc_now(),
                        skip_reason="not_started_after_fail_fast",
                    )
                _write_status(status_path, status)
                raise RuntimeError("; ".join(failures))
            if pending or active:
                sleep_fn(0.5)
    except BaseException:
        # start_new_session keeps children alive if the runner itself is
        # interrupted.  Close only our copies of the log handles and leave the
        # recorded PIDs untouched so a rerun can refuse to duplicate them.
        for running in active.values():
            running.log_handle.close()
        raise


def _run_serial_result_jobs(
    jobs: Sequence[Job],
    *,
    spec: MatrixSpec,
    dataset: Mapping[str, str],
    koopman: Mapping[str, str],
    identity: Mapping[str, Any],
    status: dict[str, Any],
    status_path: Path,
    popen_factory: PopenFactory,
    sleep_fn: Callable[[float], None],
) -> None:
    for job in jobs:
        if job.stage == "evaluate":
            assert job.seed is not None and job.method is not None
            if _evaluation_is_current(
                spec,
                seed=job.seed,
                method=job.method,
                dataset=dataset,
                koopman=koopman,
            ):
                _set_skipped(status, job, "strictly_current_latest_evaluation")
                _write_status(status_path, status)
                continue
        running = _start_job(
            job,
            spec=spec,
            identity=identity,
            status=status,
            status_path=status_path,
            popen_factory=popen_factory,
        )
        try:
            while True:
                returncode = running.process.poll()
                if returncode is not None:
                    break
                sleep_fn(0.5)
        except BaseException:
            running.log_handle.close()
            raise
        error = _finish_job(
            running,
            int(returncode),
            spec=spec,
            dataset=dataset,
            koopman=koopman,
            status=status,
            status_path=status_path,
        )
        if error is not None:
            raise RuntimeError(f"{job.job_id}: {error}")


def run_matrix(
    spec: MatrixSpec,
    *,
    dry_run: bool = False,
    popen_factory: PopenFactory = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    spec = spec.resolved()
    dataset = _expected_dataset(spec)
    koopman = _expected_koopman(spec)
    identity = _identity_bundle(spec)
    jobs = build_jobs(spec, dataset=dataset, koopman=koopman)
    manifest = _manifest(
        spec, dataset=dataset, koopman=koopman, identity=identity, jobs=jobs
    )
    spec.root.mkdir(parents=True, exist_ok=True)
    manifest_path = spec.root / "matrix_manifest.json"
    status_path = spec.root / "matrix_status.json"
    manifest = _compatible_existing_manifest(manifest_path, manifest)
    status = _initial_status(status_path, manifest, jobs)
    _atomic_json(manifest_path, manifest)
    _write_status(status_path, status)

    if dry_run:
        for job in jobs:
            if job.stage == "train" and job.mode_at_plan == "completed":
                _set_skipped(status, job, "strictly_valid_completed_run")
            elif job.stage == "evaluate" and job.mode_at_plan == "completed":
                _set_skipped(status, job, "strictly_current_latest_evaluation")
            else:
                status["jobs"][job.job_id]["state"] = "planned"
        status.update(state="dry_run", finished_utc=_utc_now())
        status["invocations"][-1]["finished_utc"] = status["finished_utc"]
        status["invocations"][-1]["state"] = "dry_run"
        _write_status(status_path, status)
        return status

    status["state"] = "running"
    _write_status(status_path, status)
    training_jobs = [job for job in jobs if job.stage == "train"]
    result_jobs = [job for job in jobs if job.stage != "train"]
    try:
        _run_training_jobs(
            training_jobs,
            spec=spec,
            dataset=dataset,
            koopman=koopman,
            identity=identity,
            status=status,
            status_path=status_path,
            popen_factory=popen_factory,
            sleep_fn=sleep_fn,
        )
        _run_serial_result_jobs(
            result_jobs,
            spec=spec,
            dataset=dataset,
            koopman=koopman,
            identity=identity,
            status=status,
            status_path=status_path,
            popen_factory=popen_factory,
            sleep_fn=sleep_fn,
        )
    except BaseException as exc:
        status.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_utc=_utc_now(),
        )
        status["invocations"][-1].update(
            state=status["state"], finished_utc=status["finished_utc"]
        )
        _write_status(status_path, status)
        raise
    status.update(state="completed", finished_utc=_utc_now(), error=None)
    status["invocations"][-1].update(
        state="completed", finished_utc=status["finished_utc"]
    )
    _write_status(status_path, status)
    return status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--koopman", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated training seeds (default: three formal seeds)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--eval-device", choices=("cpu", "cuda", "auto"), default="cpu"
    )
    parser.add_argument("--offline-updates", type=int, default=500_000)
    parser.add_argument("--online-steps", type=int, default=100_000)
    parser.add_argument("--online-utd", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--env-workers", type=int, default=5)
    parser.add_argument("--cql-weight", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    spec = MatrixSpec(
        repo_root=repo_root,
        dataset=args.dataset,
        koopman=args.koopman,
        root=args.root,
        seeds=args.seeds,
        device=args.device,
        eval_device=args.eval_device,
        offline_updates=args.offline_updates,
        online_steps=args.online_steps,
        online_utd=args.online_utd,
        num_envs=args.num_envs,
        env_workers=args.env_workers,
        cql_weight=args.cql_weight,
        eval_episodes=args.eval_episodes,
        max_parallel=args.max_parallel,
        python=args.python,
    )
    status = run_matrix(spec, dry_run=args.dry_run)
    summary = {
        "state": status["state"],
        "manifest": str(spec.root.expanduser().resolve() / "matrix_manifest.json"),
        "status": str(spec.root.expanduser().resolve() / "matrix_status.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
