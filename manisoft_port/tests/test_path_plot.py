import numpy as np

from antmaze_ac.evaluation.path_plot import path_progress, save_path_diagnostics


def test_path_progress_reports_failed_episode_extent():
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    metrics = path_progress(xy, np.asarray([4.0, 0.0]))
    assert metrics["path_length_xy"] == 2.0
    assert metrics["start_goal_distance"] == 4.0
    assert metrics["final_goal_distance"] == 2.0
    assert metrics["minimum_goal_distance"] == 2.0
    assert metrics["goal_progress_fraction"] == 0.5


def test_save_path_diagnostics_writes_png_and_numeric_archive(tmp_path):
    progress = path_progress(
        np.asarray([[0.0, 0.0], [1.0, 0.5]]),
        np.asarray([4.0, 4.0]),
    )
    png, archive = save_path_diagnostics(
        [
            {
                "episode": 0,
                "success": 0.0,
                "xy": np.asarray([[0.0, 0.0], [1.0, 0.5]]),
                "goal": np.asarray([4.0, 4.0]),
                **progress,
            }
        ],
        {
            "maze_map": [[1, 1, 1], [1, "r", 1], [1, "g", 1]],
            "scale": 4.0,
            "origin_x": 4.0,
            "origin_y": 4.0,
        },
        tmp_path / "paths.png",
    )
    assert png.exists() and png.stat().st_size > 0
    assert archive.exists()
    with np.load(archive) as payload:
        np.testing.assert_allclose(
            payload["episode_000_xy"],
            [[0.0, 0.0], [1.0, 0.5]],
        )
