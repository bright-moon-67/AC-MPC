"""Content identity for code that can affect a DMC training result."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _source_paths(root: Path) -> list[Path]:
    paths = list((root / "experiments" / "dmc").rglob("*.py"))
    paths.extend((root / "antmaze_ac" / "control").glob("*.py"))
    paths.extend((root / "antmaze_ac" / "koopman").glob("*.py"))
    paths.extend(
        [
            root / "antmaze_ac" / "rl" / "koopman_mpc_actor.py",
            root / "antmaze_ac" / "rl" / "quadratic_actors.py",
            root / "pyproject.toml",
        ]
    )
    return sorted({path.resolve() for path in paths if path.is_file()})


def source_identity(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Hash relevant tracked or untracked sources without relying on Git state."""

    root = root.resolve()
    digest = hashlib.sha256()
    files: list[dict[str, str]] = []
    for path in _source_paths(root):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Source path {path} escapes repository root {root}") from exc
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative, "sha256": file_digest})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    if not files:
        raise RuntimeError("No DMC source files found for source identity")
    return {
        "algorithm": "sha256_path_and_content_v1",
        "fingerprint": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }
