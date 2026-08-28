from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from antmaze_ac.envs.manisoft_teacher_tracking_sac_env import (
    MANISOFT_TEACHER_TRACKING_OBSERVATION_DIM,
    ManiSoftTeacherTrackingSACEnv,
    load_smooth_wall_teacher_episode,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "configs/manisoft_strong_bend_e2mpa_r45mm_damping7.yaml"
TASK = ROOT / (
    "configs/manisoft_wall_route_collection_strong_bend_e2mpa_r45mm_"
    "t45_a060_wall_y027_x010.yaml"
)
TEACHER = ROOT / (
    "data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/"
    "smooth_teacher_damping7_ts250_as250_v1/teacher_episode.npz"
)
ARCHED_TEACHER = ROOT / (
    "data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/"
    "arched_teacher_z030_v1/teacher_episode.npz"
)
LOW_TIP_TEACHER = ROOT / (
    "data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/"
    "arched_lowtip_z015_teacher_v1/teacher_episode.npz"
)
SILKY_LOW_TIP_TEACHER = ROOT / (
    "data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/"
    "silky_bridged_braked_teacher_v2/teacher_episode.npz"
)
POSE_MATCHED_TEACHER = ROOT / (
    "data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/"
    "silky_pose_matched_teacher_v1/teacher_episode.npz"
)
NEGATIVE_X_TEACHER = ROOT / (
    "data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/"
    "silky_negx2cm_speed018_teacher_v1/teacher_episode.npz"
)
POSE_REFERENCE_TEACHER = ROOT / (
    "data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/"
    "silky_lowtip_teacher_v2/teacher_episode.npz"
)

_REQUIRED_TEACHER_ARTIFACTS = (
    TEACHER,
    ARCHED_TEACHER,
    LOW_TIP_TEACHER,
    SILKY_LOW_TIP_TEACHER,
    POSE_MATCHED_TEACHER,
    NEGATIVE_X_TEACHER,
    POSE_REFERENCE_TEACHER,
)
_MISSING_TEACHER_ARTIFACTS = tuple(
    path for path in _REQUIRED_TEACHER_ARTIFACTS if not path.is_file()
)
pytestmark = pytest.mark.skipif(
    bool(_MISSING_TEACHER_ARTIFACTS),
    reason=(
        "wall-teacher integration tests require local experiment artifacts; "
        "see README_ACMPC_MANISOFT.md section 12.9"
    ),
)


def test_teacher_episode_alignment_and_hash_metadata():
    episode = load_smooth_wall_teacher_episode(TEACHER)
    assert episode.actions.shape == (1091, 18)
    assert episode.physical_states.shape == (1092, 45)
    assert episode.node_positions.shape == (1092, 21, 3)
    assert episode.route_side == 1
    assert np.max(np.abs(episode.actions)) <= 0.60


def test_teacher_tracking_observation_batch_is_actor_ready():
    env = ManiSoftTeacherTrackingSACEnv(
        SCENARIO,
        task_config_path=TASK,
        teacher_episode_path=TEACHER,
    )
    observations = env.teacher_observation_batch()
    assert observations.shape == (
        env.teacher.transition_count,
        MANISOFT_TEACHER_TRACKING_OBSERVATION_DIM,
    )
    assert np.isfinite(observations).all()
    np.testing.assert_allclose(observations[:, 63:108], 0.0, atol=0.0)
    np.testing.assert_allclose(
        observations[:, 111:129],
        env.teacher.actions / env.absolute_action_limit,
        atol=1e-7,
    )
    env.close()


def test_arched_teacher_terminal_requires_and_reaches_30cm_arch():
    episode = load_smooth_wall_teacher_episode(ARCHED_TEACHER)
    env = ManiSoftTeacherTrackingSACEnv(
        SCENARIO,
        task_config_path=TASK,
        teacher_episode_path=ARCHED_TEACHER,
        episode_steps=2,
        terminal_minimum_crossed_fraction=0.40,
        terminal_maximum_tip_speed=0.25,
        arch_height_target=0.30,
        arch_y_margin=0.05,
        arch_enforcement_start_progress=0.78,
        arch_deficit_penalty_scale=2000.0,
    )
    env.reset(seed=19, options={"start_index": episode.transition_count - 1})
    _, _, terminated, _, info = env.step(np.zeros(18, dtype=np.float32))
    assert terminated and info["is_success"]
    assert info["arch_height"] >= 0.30
    assert info["target_plane_distance"] <= 0.005
    env.close()


def test_low_tip_teacher_keeps_arch_and_finishes_near_15cm():
    episode = load_smooth_wall_teacher_episode(LOW_TIP_TEACHER)
    env = ManiSoftTeacherTrackingSACEnv(
        SCENARIO,
        task_config_path=TASK,
        teacher_episode_path=LOW_TIP_TEACHER,
        episode_steps=2,
        terminal_minimum_crossed_fraction=0.40,
        terminal_maximum_tip_speed=0.40,
        arch_height_target=0.30,
        arch_y_margin=0.05,
        arch_enforcement_start_progress=0.78,
        arch_deficit_penalty_scale=2000.0,
    )
    env.reset(seed=23, options={"start_index": episode.transition_count - 1})
    _, _, terminated, _, info = env.step(np.zeros(18, dtype=np.float32))
    final_tip = env._rod_arrays()[0][-1]
    assert terminated and info["is_success"]
    assert info["arch_height"] >= 0.30
    assert info["target_plane_distance"] <= 0.005
    assert info["distal_crossed_fraction"] >= 0.40
    assert final_tip[1] > env.geometry.wall_maximum[1]
    assert final_tip[2] == pytest.approx(0.15, abs=0.005)
    env.close()


def test_silky_teacher_has_no_long_constant_action_hold():
    episode = load_smooth_wall_teacher_episode(SILKY_LOW_TIP_TEACHER)
    constant = (
        np.linalg.norm(np.diff(episode.actions.astype(np.float64), axis=0), axis=1)
        < 1e-8
    )
    longest = current = 0
    longest_stop = 0
    current_start = 0
    for index, value in enumerate(constant):
        if value:
            if current == 0:
                current_start = index
            current += 1
            if current > longest:
                longest = current
                longest_stop = current_start
        else:
            current = 0
    # A short terminal target-action hold is permitted while the arm is still
    # dynamically braking; the multi-second stationary middle hold is not.
    assert longest <= 15
    held_tip_speeds = np.linalg.norm(
        episode.node_velocities[longest_stop : longest_stop + longest + 1, -1], axis=1
    )
    assert np.min(held_tip_speeds) >= 0.10
    assert episode.transition_count == 844
    assert episode.node_positions[-1, -1, 2] == pytest.approx(0.15, abs=0.005)
    assert np.linalg.norm(episode.node_velocities[-1, -1]) <= 0.17


def test_pose_matched_teacher_preserves_oblique_distal_posture():
    episode = load_smooth_wall_teacher_episode(POSE_MATCHED_TEACHER)
    reference = load_smooth_wall_teacher_episode(POSE_REFERENCE_TEACHER)
    nodes = episode.node_positions[-1]
    relative = nodes[-6:] - nodes[-1]
    reference_nodes = reference.node_positions[-1]
    reference_relative = reference_nodes[-6:] - reference_nodes[-1]
    pose_rmse = np.sqrt(np.mean((relative - reference_relative) ** 2))
    tangent = nodes[-1] - nodes[-2]
    tangent /= np.linalg.norm(tangent)
    angle_to_yz = np.degrees(np.arcsin(abs(tangent[0])))
    assert pose_rmse <= 0.01
    assert angle_to_yz >= 10.0
    assert np.linalg.norm(episode.node_velocities[-1, -1]) <= 0.30


def test_negative_x_teacher_reaches_left_side_slowly_and_obliquely():
    episode = load_smooth_wall_teacher_episode(NEGATIVE_X_TEACHER)
    nodes = episode.node_positions[-1]
    tangent = nodes[-1] - nodes[-2]
    tangent /= np.linalg.norm(tangent)
    angle_to_yz = np.degrees(np.arcsin(abs(tangent[0])))
    assert nodes[-1, 0] == pytest.approx(-0.02, abs=0.003)
    assert np.linalg.norm(episode.node_velocities[-1, -1]) <= 0.21
    assert angle_to_yz >= 14.7


@pytest.mark.parametrize("start_index", [0, 500, 1090])
def test_teacher_action_replays_one_transition_from_restored_snapshot(start_index):
    env = ManiSoftTeacherTrackingSACEnv(
        SCENARIO,
        task_config_path=TASK,
        teacher_episode_path=TEACHER,
        episode_steps=2,
    )
    observation, info = env.reset(seed=17, options={"start_index": start_index})
    assert observation.shape == (MANISOFT_TEACHER_TRACKING_OBSERVATION_DIM,)
    assert info["reference_index"] == start_index
    _, _, terminated, _, next_info = env.step(np.zeros(18, dtype=np.float32))
    assert not terminated or next_info["is_success"]
    assert next_info["reference_index"] == start_index + 1
    assert next_info["node_tracking_rmse"] < 1e-8
    assert next_info["tip_tracking_error"] < 1e-8
    env.close()
