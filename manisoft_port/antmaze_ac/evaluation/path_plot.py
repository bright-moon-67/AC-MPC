from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def antmaze_geometry(env) -> dict[str, Any] | None:
    """Extract legacy D4RL maze geometry without depending on its classes."""

    current = env
    visited = set()
    candidates = []
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        candidates.append(current)
        legacy = getattr(current, "legacy_env", None)
        if legacy is not None:
            candidates.extend((legacy, getattr(legacy, "unwrapped", legacy)))
        current = getattr(current, "env", None)
    for candidate in candidates:
        maze_map = getattr(candidate, "_maze_map", None)
        if maze_map is None:
            continue
        return {
            "maze_map": [list(row) for row in maze_map],
            "scale": float(candidate._maze_size_scaling),
            "origin_x": float(candidate._init_torso_x),
            "origin_y": float(candidate._init_torso_y),
        }
    return None


def target_goal(env) -> np.ndarray | None:
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        legacy = getattr(current, "legacy_env", None)
        for candidate in (
            current,
            legacy,
            getattr(legacy, "unwrapped", None),
            getattr(current, "unwrapped", None),
        ):
            if candidate is None:
                continue
            goal = getattr(candidate, "target_goal", None)
            if goal is None:
                goal = getattr(candidate, "_goal", None)
            if goal is not None:
                array = np.asarray(goal, dtype=np.float64).reshape(-1)
                if len(array) >= 2 and np.isfinite(array[:2]).all():
                    return array[:2]
        current = getattr(current, "env", None)
    return None


def path_progress(xy: np.ndarray, goal: np.ndarray | None) -> dict[str, float | None]:
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 1 or xy.shape[1:] != (2,):
        raise ValueError("xy path must have shape [steps, 2]")
    path_length = (
        float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        if len(xy) > 1
        else 0.0
    )
    if goal is None:
        return {
            "path_length_xy": path_length,
            "start_goal_distance": None,
            "final_goal_distance": None,
            "minimum_goal_distance": None,
            "goal_progress_fraction": None,
        }
    goal = np.asarray(goal, dtype=np.float64)
    distances = np.linalg.norm(xy - goal[None, :], axis=1)
    start_distance = float(distances[0])
    minimum_distance = float(distances.min())
    progress = (
        (start_distance - minimum_distance) / start_distance
        if start_distance > 1e-12
        else 0.0
    )
    return {
        "path_length_xy": path_length,
        "start_goal_distance": start_distance,
        "final_goal_distance": float(distances[-1]),
        "minimum_goal_distance": minimum_distance,
        "goal_progress_fraction": float(progress),
    }


def save_path_diagnostics(
    paths: list[dict[str, Any]],
    geometry: dict[str, Any],
    png_path: str | Path,
) -> tuple[Path, Path]:
    """Save per-episode U-Maze XY plots plus the underlying paths."""

    if not paths:
        raise ValueError("At least one path is required")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Rectangle

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(3, len(paths))
    rows = int(np.ceil(len(paths) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 4.6 * rows),
        squeeze=False,
    )
    maze_map = geometry["maze_map"]
    scale = float(geometry["scale"])
    origin_x = float(geometry["origin_x"])
    origin_y = float(geometry["origin_y"])
    centers = []
    for row_index, maze_row in enumerate(maze_map):
        for column_index, cell in enumerate(maze_row):
            center_x = column_index * scale - origin_x
            center_y = row_index * scale - origin_y
            centers.append((center_x, center_y))
            if cell == 1:
                for axis in axes.flat:
                    axis.add_patch(
                        Rectangle(
                            (center_x - scale / 2, center_y - scale / 2),
                            scale,
                            scale,
                            facecolor="#5f554d",
                            edgecolor="#322d29",
                            linewidth=0.5,
                            zorder=0,
                        )
                    )
    x_values = [value[0] for value in centers]
    y_values = [value[1] for value in centers]
    x_limits = (min(x_values) - scale / 2, max(x_values) + scale / 2)
    y_limits = (min(y_values) - scale / 2, max(y_values) + scale / 2)

    archive: dict[str, np.ndarray] = {}
    for path_index, (axis, path) in enumerate(zip(axes.flat, paths)):
        xy = np.asarray(path["xy"], dtype=np.float64)
        goal = np.asarray(path["goal"], dtype=np.float64)
        if len(xy) > 1:
            segments = np.stack((xy[:-1], xy[1:]), axis=1)
            collection = LineCollection(
                segments,
                cmap="viridis",
                linewidth=2.0,
                zorder=2,
            )
            collection.set_array(np.linspace(0.0, 1.0, len(segments)))
            axis.add_collection(collection)
        axis.scatter(*xy[0], marker="o", s=45, color="#1f77b4", zorder=4)
        axis.scatter(*xy[-1], marker="x", s=55, color="#d62728", zorder=4)
        axis.scatter(*goal, marker="*", s=140, color="#ffbf00", edgecolor="black", zorder=5)
        axis.add_patch(
            Circle(
                goal,
                radius=0.5,
                fill=False,
                edgecolor="#ffbf00",
                linestyle="--",
                linewidth=1.2,
                zorder=3,
            )
        )
        inset = axis.inset_axes([0.53, 0.04, 0.43, 0.34])
        inset.plot(xy[:, 0], xy[:, 1], color="#2a9d8f", linewidth=1.2)
        inset.scatter(*xy[0], marker="o", s=22, color="#1f77b4", zorder=3)
        inset.scatter(*xy[-1], marker="x", s=28, color="#d62728", zorder=3)
        path_span = np.ptp(xy, axis=0)
        margin = max(float(path_span.max()) * 0.2, 0.03)
        inset.set_xlim(float(xy[:, 0].min() - margin), float(xy[:, 0].max() + margin))
        inset.set_ylim(float(xy[:, 1].min() - margin), float(xy[:, 1].max() + margin))
        inset.set_aspect("equal")
        inset.tick_params(labelsize=6)
        inset.grid(alpha=0.2)
        inset.set_title("trajectory zoom", fontsize=7)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(
            f"episode {path['episode']} | success={int(path['success'])}\n"
            f"min→goal={path['minimum_goal_distance']:.2f}, "
            f"final→goal={path['final_goal_distance']:.2f}"
        )
        archive[f"episode_{path_index:03d}_xy"] = xy.astype(np.float32)
        archive[f"episode_{path_index:03d}_goal"] = goal.astype(np.float32)
    for axis in axes.flat[len(paths) :]:
        axis.set_visible(False)
    figure.suptitle(
        "AntMaze trajectories: blue=start, red=final, gold=goal",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    npz_path = png_path.with_suffix(".npz")
    np.savez_compressed(npz_path, **archive)
    return png_path, npz_path
