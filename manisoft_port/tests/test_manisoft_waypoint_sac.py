from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym
from scipy.spatial import Delaunay

from antmaze_ac.envs.fixed_seed_panel import FixedSeedPanelWrapper
from antmaze_ac.envs.frozen_base_residual import FrozenBaseResidualActionWrapper
from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv
from antmaze_ac.envs.manisoft_waypoint_sac_env import (
    MANISOFT_WAYPOINT_SAC_OBSERVATION_DIM,
    ManiSoftWaypointSACEnv,
)
from antmaze_ac.envs.waypoint_paths import (
    ReferencePath,
    WaypointPathGenerator,
    WaypointWorkspace,
)
from antmaze_ac.rl.anchored_sac import AnchoredSAC


class _ResidualInnerEnv(gym.Env):
    observation_space = gym.spaces.Box(-10.0, 10.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self) -> None:
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        observation = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        return observation, 2.0, False, False, {"inner": True}


class _ResidualInner18Env(_ResidualInnerEnv):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=np.float32)


class _ResidualNormalizer:
    def normalize_obs(self, observation):
        return np.asarray(observation, dtype=np.float32) / 2.0


class _ResidualBasePolicy:
    def __init__(self) -> None:
        self.last_observation = None

    def predict(self, observation, *, deterministic=True):
        assert deterministic
        self.last_observation = np.asarray(observation).copy()
        return np.asarray([[0.2, -0.3]], dtype=np.float32), None


class _ResidualBase18Policy:
    def predict(self, observation, *, deterministic=True):
        del observation
        assert deterministic
        return np.linspace(-0.4, 0.4, 18, dtype=np.float32)[None, :], None


def test_anchored_sac_source_policy_warmup_uses_deterministic_actor() -> None:
    env = gym.make("Pendulum-v1")
    model = AnchoredSAC(
        "MlpPolicy",
        env,
        learning_starts=16,
        buffer_size=64,
        batch_size=8,
        policy_kwargs={"net_arch": [16]},
        device="cpu",
        seed=7,
    )
    model._last_obs = model.get_env().reset()
    model.num_timesteps = 0
    expected, _ = model.predict(model._last_obs, deterministic=True)
    model.enable_source_policy_warmup()

    action, buffer_action = model._sample_action(16, n_envs=1)

    np.testing.assert_allclose(action, expected, atol=1e-7)
    np.testing.assert_allclose(
        buffer_action,
        model.policy.scale_action(expected),
        atol=1e-7,
    )
    env.close()


def test_anchored_sac_actor_delay_keeps_source_actor_frozen() -> None:
    env = gym.make("Pendulum-v1")
    model = AnchoredSAC(
        "MlpPolicy",
        env,
        learning_starts=0,
        buffer_size=64,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [16]},
        device="cpu",
        seed=9,
    )
    model.enable_actor_anchor(10.0)
    model.delay_actor_updates_until(100)
    before = [parameter.detach().clone() for parameter in model.actor.parameters()]

    model.learn(total_timesteps=4)

    for expected, actual in zip(before, model.actor.parameters()):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(), expected.cpu().numpy(), atol=0.0
        )
    env.close()


def test_frozen_base_residual_zero_action_exactly_reproduces_base() -> None:
    inner = _ResidualInnerEnv()
    base = _ResidualBasePolicy()
    env = FrozenBaseResidualActionWrapper(
        inner,
        residual_action_scale=0.10,
        residual_action_penalty_scale=0.50,
        base_model=base,
        observation_normalizer=_ResidualNormalizer(),
    )
    env.reset(seed=7)
    _, reward, terminated, truncated, info = env.step(np.zeros(2))
    np.testing.assert_allclose(base.last_observation, [[0.5, 1.0, 1.5, 2.0]])
    np.testing.assert_allclose(inner.last_action, [0.2, -0.3])
    np.testing.assert_allclose(info["combined_policy_action"], [0.2, -0.3])
    assert reward == pytest.approx(2.0)
    assert not terminated and not truncated


def test_frozen_base_residual_supports_eighteen_action_axes() -> None:
    inner = _ResidualInner18Env()
    env = FrozenBaseResidualActionWrapper(
        inner,
        residual_action_scale=0.02,
        base_model=_ResidualBase18Policy(),
        observation_normalizer=_ResidualNormalizer(),
    )
    env.reset(seed=8)
    _, _, _, _, info = env.step(np.zeros(18, dtype=np.float32))
    expected = np.linspace(-0.4, 0.4, 18, dtype=np.float32)
    np.testing.assert_allclose(inner.last_action, expected)
    np.testing.assert_allclose(info["combined_policy_action"], expected)


def test_frozen_base_residual_is_scaled_clipped_and_penalized() -> None:
    inner = _ResidualInnerEnv()
    env = FrozenBaseResidualActionWrapper(
        inner,
        residual_action_scale=0.20,
        residual_action_penalty_scale=0.50,
        base_model=_ResidualBasePolicy(),
        observation_normalizer=_ResidualNormalizer(),
    )
    env.reset()
    _, reward, _, _, info = env.step(np.asarray([1.0, -1.0]))
    np.testing.assert_allclose(inner.last_action, [0.4, -0.5])
    np.testing.assert_allclose(info["residual_scaled_action"], [0.2, -0.2])
    assert info["residual_action_penalty"] == pytest.approx(0.5)
    assert reward == pytest.approx(1.5)


def test_frozen_residual_can_ramp_only_after_waypoint_progress_stalls() -> None:
    inner = _ResidualInnerEnv()
    inner.step_count = 250
    inner.last_waypoint_improvement_step = 100
    env = FrozenBaseResidualActionWrapper(
        inner,
        residual_action_scale=0.20,
        residual_stall_activation_steps=100,
        residual_stall_ramp_steps=100,
        base_model=_ResidualBasePolicy(),
        observation_normalizer=_ResidualNormalizer(),
    )
    env.reset()
    _, _, _, _, info = env.step(np.asarray([1.0, -1.0]))
    assert info["residual_activation_factor"] == pytest.approx(0.5)
    np.testing.assert_allclose(inner.last_action, [0.3, -0.4])

    inner.step_count = 150
    inner.last_waypoint_improvement_step = 100
    _, _, _, _, info = env.step(np.asarray([1.0, -1.0]))
    assert info["residual_activation_factor"] == pytest.approx(0.0)
    np.testing.assert_allclose(inner.last_action, [0.2, -0.3])


def test_reference_path_arc_length_sampling() -> None:
    path = ReferencePath.from_points(
        "corner",
        np.asarray([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.1, 0.2, 0.5]]),
        np.asarray([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.1, 0.2, 0.5]]),
    )
    assert path.length == pytest.approx(0.3)
    np.testing.assert_allclose(path.sample(0.05), [0.05, 0.0, 0.5])
    np.testing.assert_allclose(path.sample(0.20), [0.1, 0.1, 0.5])
    np.testing.assert_allclose(path.sample(2.0), [0.1, 0.2, 0.5])


def test_workspace_can_enforce_an_xy_convex_hull() -> None:
    workspace = WaypointWorkspace.from_bounds(
        [0.0, 0.0, 0.4],
        [1.0, 1.0, 0.6],
        max_reach=2.0,
        xy_hull_equations=np.asarray(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, -1.0]]
        ),
    )
    assert workspace.contains(np.asarray([0.2, 0.3, 0.5]))
    assert not workspace.contains(np.asarray([0.8, 0.8, 0.5]))


def test_workspace_can_reject_under_sampled_delaunay_simplices() -> None:
    triangulation = Delaunay(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    )
    accepted_simplex, rejected_simplex = 0, 1
    accepted_point = np.append(
        np.mean(triangulation.points[triangulation.simplices[accepted_simplex]], axis=0),
        0.5,
    )
    rejected_point = np.append(
        np.mean(triangulation.points[triangulation.simplices[rejected_simplex]], axis=0),
        0.5,
    )
    valid = np.zeros(len(triangulation.simplices), dtype=bool)
    valid[accepted_simplex] = True
    workspace = WaypointWorkspace.from_bounds(
        [0.0, 0.0, 0.4],
        [1.0, 1.0, 0.6],
        max_reach=2.0,
        xy_triangulation=triangulation,
        xy_valid_simplices=valid,
    )
    assert workspace.contains(accepted_point)
    assert not workspace.contains(rejected_point)


def test_workspace_can_enforce_a_three_dimensional_pose_hull() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.4],
            [1.0, 0.0, 0.4],
            [0.0, 1.0, 0.4],
            [0.0, 0.0, 0.6],
        ]
    )
    triangulation = Delaunay(points)
    workspace = WaypointWorkspace.from_bounds(
        [0.0, 0.0, 0.4],
        [1.0, 1.0, 0.6],
        max_reach=2.0,
        xy_triangulation=triangulation,
        xy_valid_simplices=np.ones(len(triangulation.simplices), dtype=bool),
        interpolation_dimensions=3,
    )
    assert workspace.contains(np.asarray([0.2, 0.2, 0.5]))
    assert not workspace.contains(np.asarray([0.6, 0.6, 0.5]))


def test_pose_map_action_can_interpolate_xyz_without_changing_state_shape() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.4],
            [1.0, 0.0, 0.4],
            [0.0, 1.0, 0.4],
            [0.0, 0.0, 0.6],
        ],
        dtype=np.float64,
    )
    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.pose_map_interpolation_dimensions = 3
    env.pose_map_triangulation = Delaunay(points)
    env.pose_map_valid_simplices = np.ones(
        len(env.pose_map_triangulation.simplices), dtype=bool
    )
    env.pose_map_policy_actions = np.zeros((len(points), 18), dtype=np.float32)
    env.pose_map_policy_actions[:, :3] = points
    query = np.asarray([0.2, 0.1, 0.48])

    action = env._pose_map_action(query)

    np.testing.assert_allclose(action[:3], query, atol=1e-7)
    assert action.shape == (18,)


def test_long_waypoint_chords_respect_ten_to_twenty_centimetres() -> None:
    workspace = WaypointWorkspace.from_bounds(
        [0.0, 0.0, 0.44],
        [0.24, 0.24, 0.47],
        max_reach=1.0,
    )
    generator = WaypointPathGenerator(
        workspace,
        waypoint_segment_count_range=(10, 10),
        waypoint_segment_count_probabilities=(1.0,),
        waypoint_segment_length_range=(0.10, 0.20),
        waypoint_maximum_extent=0.24,
        waypoint_minimum_turn_degrees=0.0,
        waypoint_maximum_turn_degrees=175.0,
        waypoint_vertical_delta_range=(0.0, 0.0),
    )
    start = np.asarray([0.12, 0.04, 0.455])
    path = generator.generate(
        np.random.default_rng(20260895),
        start,
        curriculum="table_waypoint_polyline",
        family="waypoint_polyline",
    )
    lengths = np.linalg.norm(np.diff(path.anchors, axis=0), axis=1)
    assert len(lengths) == 10
    assert np.all(lengths >= 0.10 - 1e-6)
    assert np.all(lengths <= 0.20 + 1e-6)
    assert np.all(workspace.contains(path.anchors))


def test_waypoint_polyline_can_use_a_short_entry_segment_before_long_chords() -> None:
    workspace = WaypointWorkspace.from_bounds(
        [0.0, 0.0, 0.44],
        [0.30, 0.30, 0.49],
        max_reach=1.0,
    )
    generator = WaypointPathGenerator(
        workspace,
        waypoint_segment_count_range=(8, 8),
        waypoint_segment_count_probabilities=(1.0,),
        waypoint_segment_length_range=(0.10, 0.20),
        waypoint_first_segment_length_range=(0.03, 0.08),
        waypoint_maximum_extent=0.30,
        waypoint_maximum_turn_degrees=175.0,
        waypoint_vertical_delta_range=(-0.01, 0.01),
    )
    path = generator.generate(
        np.random.default_rng(20261071),
        np.asarray([0.15, 0.05, 0.465]),
        curriculum="table_waypoint_polyline",
        family="waypoint_polyline",
    )
    lengths = np.linalg.norm(np.diff(path.anchors, axis=0), axis=1)
    assert 0.03 - 1e-6 <= lengths[0] <= 0.081
    assert np.all(lengths[1:] >= 0.10 - 1e-6)
    assert np.all(lengths[1:] <= 0.201)
    assert np.all(workspace.contains(path.anchors))


def test_custom_path_does_not_prepend_a_roundoff_duplicate_start() -> None:
    generator = WaypointPathGenerator()
    start = np.asarray([0.13752586, 0.72552967, 0.50559115])
    anchors = np.asarray(
        [
            start + np.asarray([2e-7, 0.0, 0.0]),
            start + np.asarray([-0.10, 0.01, 0.0]),
        ]
    )
    path = generator.generate(
        np.random.default_rng(1),
        start,
        curriculum="table_long_waypoints",
        anchors=anchors,
    )
    assert len(path.anchors) == 2
    np.testing.assert_allclose(path.anchors[0], start)


def test_reference_path_projection_is_geometric_and_window_bounded() -> None:
    path = ReferencePath.from_points(
        "out_and_back",
        np.asarray([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5], [0.0, 0.0, 0.5]]),
        np.asarray([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5], [0.0, 0.0, 0.5]]),
    )
    progress, distance, projected = path.project(
        [0.05, 0.01, 0.5], minimum_distance=0.0, maximum_distance=0.10
    )
    assert progress == pytest.approx(0.05)
    assert distance == pytest.approx(0.01)
    np.testing.assert_allclose(projected, [0.05, 0.0, 0.5])
    # The identical returning segment is outside the permitted forward window.
    assert progress < 0.10


def test_reference_path_projection_can_ignore_vertical_error() -> None:
    path = ReferencePath.from_points(
        "rising_line",
        np.asarray([[0.0, 0.0, 0.44], [0.2, 0.0, 0.49]]),
        np.asarray([[0.0, 0.0, 0.44], [0.2, 0.0, 0.49]]),
    )
    progress, distance, projected = path.project(
        [0.1, 0.01, 0.44],
        coordinate_weights=(1.0, 1.0, 0.0),
    )
    assert progress == pytest.approx(0.5 * path.length)
    assert distance == pytest.approx(0.01)
    np.testing.assert_allclose(projected, [0.1, 0.0, 0.465])


def test_vertical_tolerance_relaxes_z_without_changing_xy_capture() -> None:
    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.tracking_vertical_tolerance = 0.025
    target = np.asarray([0.30, 0.40, 0.45])

    assert env._task_space_distance(
        target + np.asarray([0.01, 0.0, 0.020]), target
    ) == pytest.approx(0.01)
    assert env._within_task_tolerance(
        target + np.asarray([0.009, 0.0, 0.024]), target, 0.010
    )
    assert not env._within_task_tolerance(
        target + np.asarray([0.011, 0.0, 0.0]), target, 0.010
    )
    assert not env._within_task_tolerance(
        target + np.asarray([0.0, 0.0, 0.026]), target, 0.010
    )
    assert env._task_space_distance(
        target + np.asarray([0.0, 0.0, 0.035]), target
    ) == pytest.approx(0.010)


@pytest.mark.parametrize(
    "family", ["line", "polyline", "bezier", "s_curve", "reverse"]
)
def test_generated_table_paths_are_seeded_finite_and_reachable(family: str) -> None:
    workspace = WaypointWorkspace.from_bounds()
    generator = WaypointPathGenerator(workspace)
    start = np.asarray([0.0, 0.0, 0.94])
    first = generator.generate(
        np.random.default_rng(123), start, curriculum="mixed", family=family
    )
    second = generator.generate(
        np.random.default_rng(123), start, curriculum="mixed", family=family
    )
    np.testing.assert_allclose(first.points, second.points)
    assert first.length > 0.05
    assert np.isfinite(first.points).all()
    np.testing.assert_allclose(first.points[0], start)
    assert bool(workspace.contains(first.points[-1]))
    # All points after the entry segment reaches the table box are valid.
    inside = workspace.contains(first.points)
    first_inside = int(np.flatnonzero(inside)[0])
    assert np.all(inside[first_inside:])


def test_custom_waypoints_prepend_current_tip_and_validate_workspace() -> None:
    generator = WaypointPathGenerator()
    start = np.asarray([0.0, 0.0, 0.94])
    anchors = np.asarray([[0.0, 0.52, 0.54], [0.12, 0.60, 0.48]])
    path = generator.generate(
        np.random.default_rng(1), start, anchors=anchors, curriculum="mixed"
    )
    assert path.family == "custom"
    np.testing.assert_allclose(path.anchors[0], start)
    np.testing.assert_allclose(path.anchors[1:], anchors)
    with pytest.raises(ValueError, match="workspace"):
        generator.generate(
            np.random.default_rng(1),
            start,
            anchors=[[1.2, 0.5, 0.5]],
            curriculum="mixed",
        )


@pytest.mark.parametrize("family", ["line", "polyline", "bezier", "s_curve", "reverse"])
def test_table_local_curriculum_uses_short_near_planar_segments(family: str) -> None:
    workspace = WaypointWorkspace.from_bounds(
        low=(-0.30, 0.50, 0.445), high=(0.30, 0.80, 0.54), max_reach=0.91
    )
    generator = WaypointPathGenerator(workspace)
    start = np.asarray([0.1375, 0.7255, 0.5056])
    path = generator.generate(
        np.random.default_rng(77),
        start,
        curriculum="table_local",
        family=family,
    )
    anchor_lengths = np.linalg.norm(np.diff(path.anchors, axis=0), axis=1)
    assert np.max(anchor_lengths) <= 0.056
    assert np.max(np.abs(np.diff(path.anchors[:, 2]))) <= 0.0121
    assert np.all(workspace.contains(path.points))


def test_low_table_entry_descent_stays_inside_safe_workspace() -> None:
    workspace = WaypointWorkspace.from_bounds(
        low=(-0.30, 0.50, 0.445), high=(0.30, 0.80, 0.54), max_reach=0.91
    )
    generator = WaypointPathGenerator(workspace)
    start = np.asarray([0.2073, 0.7169, 0.4810])
    for seed in range(20):
        path = generator.generate(
            np.random.default_rng(seed),
            start,
            curriculum="table_local",
            family="line",
        )
        assert path.points[-1, 2] >= workspace.low[2]
        assert path.points[-1, 2] >= start[2] - 0.0121


def test_table_local_line_curriculum_is_short_and_nearly_planar() -> None:
    generator = WaypointPathGenerator(
        WaypointWorkspace.from_bounds(
            low=(-0.30, 0.50, 0.445), high=(0.30, 0.80, 0.54), max_reach=0.91
        )
    )
    start = np.asarray([0.1375, 0.7255, 0.5056])
    for seed in range(50):
        path = generator.generate(
            np.random.default_rng(seed),
            start,
            curriculum="table_local_line",
        )
        assert path.family == "line"
        assert 0.0179 <= path.length <= 0.0301
        assert abs(float(path.points[-1, 2] - start[2])) <= 1e-6


def test_table_waypoint_polyline_is_planar_short_segmented_and_local() -> None:
    workspace = WaypointWorkspace.from_bounds(
        low=(-0.30, 0.50, 0.445), high=(0.30, 0.80, 0.54), max_reach=0.91
    )
    generator = WaypointPathGenerator(
        workspace,
        waypoint_segment_count_range=(2, 4),
        waypoint_segment_length_range=(0.015, 0.030),
        waypoint_maximum_extent=0.045,
        waypoint_maximum_turn_degrees=135.0,
        waypoint_vertical_delta_range=(0.0, 0.0),
    )
    starts = (
        np.asarray([0.2694, 0.6911, 0.4975]),
        np.asarray([0.1375, 0.7255, 0.5056]),
        np.asarray([-0.2759, 0.6929, 0.4813]),
    )
    for start in starts:
        for seed in range(50):
            path = generator.generate(
                np.random.default_rng(seed),
                start,
                curriculum="table_waypoint_polyline",
            )
            assert path.family == "waypoint_polyline"
            assert 2 <= len(path.anchors) - 1 <= 4
            segment_vectors = np.diff(path.anchors, axis=0)
            segment_lengths = np.linalg.norm(segment_vectors, axis=1)
            assert np.all(segment_lengths >= 0.0149)
            assert np.all(segment_lengths <= 0.0301)
            assert np.max(np.linalg.norm(path.anchors - start, axis=1)) <= 0.0451
            assert np.max(np.abs(path.anchors[:, 2] - start[2])) <= 1e-6
            assert np.all(workspace.contains(path.points))
            if len(segment_vectors) > 1:
                unit = segment_vectors / segment_lengths[:, None]
                turns = np.rad2deg(
                    np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1))
                )
                assert np.max(turns) <= 135.1


def test_table_waypoint_rehearsal_probability_and_explicit_family() -> None:
    start = np.asarray([0.1375, 0.7255, 0.5056])
    line_generator = WaypointPathGenerator(
        waypoint_single_line_probability=1.0
    )
    assert (
        line_generator.generate(
            np.random.default_rng(1),
            start,
            curriculum="table_waypoint_polyline",
        ).family
        == "line"
    )
    # An explicit evaluation family must never be replaced by rehearsal.
    assert (
        line_generator.generate(
            np.random.default_rng(1),
            start,
            curriculum="table_waypoint_polyline",
            family="waypoint_polyline",
        ).family
        == "waypoint_polyline"
    )


def test_segment_count_probabilities_control_multipoint_curriculum() -> None:
    generator = WaypointPathGenerator(
        waypoint_segment_count_range=(2, 3),
        waypoint_segment_count_probabilities=(0.0, 1.0),
        waypoint_segment_length_range=(0.012, 0.022),
        waypoint_maximum_extent=0.035,
        waypoint_maximum_turn_degrees=90.0,
    )
    start = np.asarray([0.1375, 0.7255, 0.5056])
    for seed in range(30):
        path = generator.generate(
            np.random.default_rng(seed),
            start,
            curriculum="table_waypoint_polyline",
            family="waypoint_polyline",
        )
        assert len(path.anchors) - 1 == 3
    waypoint_generator = WaypointPathGenerator(
        waypoint_single_line_probability=0.0
    )
    assert (
        waypoint_generator.generate(
            np.random.default_rng(1),
            start,
            curriculum="table_waypoint_polyline",
        ).family
        == "waypoint_polyline"
    )


def test_hard_turn_episode_mixture_can_oversample_reversal_paths() -> None:
    generator = WaypointPathGenerator(
        WaypointWorkspace.from_bounds(
            low=(-0.25, -0.25, 0.44),
            high=(0.25, 0.25, 0.48),
            max_reach=1.0,
        ),
        waypoint_segment_count_range=(4, 4),
        waypoint_segment_count_probabilities=(1.0,),
        waypoint_segment_length_range=(0.02, 0.03),
        waypoint_maximum_extent=0.12,
        waypoint_minimum_turn_degrees=0.0,
        waypoint_maximum_turn_degrees=175.0,
        waypoint_hard_turn_probability=1.0,
        waypoint_hard_turn_range_degrees=(120.0, 150.0),
    )
    path = generator.generate(
        np.random.default_rng(20268103),
        np.asarray([0.0, 0.0, 0.46]),
        curriculum="table_waypoint_polyline",
        family="waypoint_polyline",
    )
    vectors = np.diff(path.anchors[:, :2], axis=0)
    headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    turns = np.abs(
        np.rad2deg(
            np.arctan2(
                np.sin(np.diff(headings)), np.cos(np.diff(headings))
            )
        )
    )
    assert np.all(turns >= 120.0 - 1e-6)
    assert np.all(turns <= 150.0 + 1e-6)


def test_tight_turn_polyline_has_deterministic_feasible_fallback() -> None:
    generator = WaypointPathGenerator(
        waypoint_segment_count_range=(3, 3),
        waypoint_segment_length_range=(0.012, 0.022),
        waypoint_maximum_extent=0.035,
        waypoint_maximum_turn_degrees=30.0,
        waypoint_vertical_delta_range=(0.0, 0.0),
    )
    start = np.asarray([0.2694103, 0.6911209, 0.4974925])
    anchors = generator._minimum_length_polyline_fallback(
        start=start,
        segment_count=3,
        minimum_turn=0.0,
        maximum_turn=np.deg2rad(30.0),
        minimum_length=0.012,
        minimum_revisit=0.0054,
    )
    assert anchors is not None
    segment_vectors = np.diff(anchors, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    np.testing.assert_allclose(segment_lengths, 0.012, atol=1e-9)
    unit = segment_vectors / segment_lengths[:, None]
    turns = np.rad2deg(
        np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0))
    )
    assert np.max(turns) <= 30.0001
    assert np.max(np.linalg.norm(anchors - start, axis=1)) <= 0.0350001
    assert np.all(generator.workspace.contains(anchors))


def test_tight_turn_polyline_fallback_preserves_path_diversity() -> None:
    generator = WaypointPathGenerator(
        waypoint_segment_count_range=(3, 3),
        waypoint_segment_length_range=(0.012, 0.022),
        waypoint_maximum_extent=0.035,
        waypoint_maximum_turn_degrees=30.0,
        waypoint_vertical_delta_range=(0.0, 0.0),
    )
    start = np.asarray([0.2694103, 0.6911209, 0.4974925])
    paths = [
        generator._short_waypoint_polyline(np.random.default_rng(seed), start)
        for seed in range(60)
    ]
    unique_offsets = {
        tuple(np.round((anchors - anchors[0]).ravel(), 6)) for anchors in paths
    }
    assert len(unique_offsets) >= 50
    for anchors in paths:
        segments = np.diff(anchors, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        unit = segments / lengths[:, None]
        turns = np.rad2deg(
            np.arccos(
                np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
            )
        )
        assert np.min(lengths) >= 0.012 - 1e-9
        assert np.max(lengths) <= 0.022 + 1e-9
        assert np.max(turns) <= 30.0001
        assert np.max(np.linalg.norm(anchors - start, axis=1)) <= 0.0350001
        assert np.all(generator.workspace.contains(anchors))


def test_waypoint_generation_mode_reports_random_or_fallback() -> None:
    generator = WaypointPathGenerator(
        waypoint_segment_count_range=(3, 3),
        waypoint_segment_length_range=(0.012, 0.022),
        waypoint_maximum_extent=0.035,
        waypoint_maximum_turn_degrees=30.0,
        waypoint_vertical_delta_range=(0.0, 0.0),
    )
    start = np.asarray([0.2694103, 0.6911209, 0.4974925])
    modes = {
        generator.generate(
            np.random.default_rng(seed),
            start,
            curriculum="table_waypoint_polyline",
            family="waypoint_polyline",
        ).generation_mode
        for seed in range(30)
    }
    assert modes <= {"random", "curved_fallback", "deterministic_fallback"}
    assert modes


def test_fixed_seed_panel_cycles_and_rewinds() -> None:
    class SeedRecorder(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

        def __init__(self) -> None:
            self.seeds: list[int | None] = []

        def reset(self, *, seed=None, options=None):
            del options
            self.seeds.append(seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            del action
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    base = SeedRecorder()
    env = FixedSeedPanelWrapper(base, 730000)
    env.reset(seed=1)
    env.reset(seed=2)
    env.rewind_evaluation_panel()
    env.reset(seed=3)
    assert base.seeds == [730000, 730001, 730000]


def test_entry_sampling_weights_can_focus_a_weak_posture() -> None:
    class Bank:
        trajectory_count = 3

    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.entry_bank = Bank()
    env.active_curriculum = "table_waypoint_polyline"
    env.entry_sampling_weights = np.asarray([0.0, 0.0, 1.0])
    env._np_random = np.random.default_rng(123)
    assert {env._select_entry_index({}) for _ in range(20)} == {2}
    assert env._select_entry_index({"entry_index": 1}) == 1


def test_internal_waypoint_gate_requires_geometric_capture_in_order() -> None:
    path = ReferencePath.from_points(
        "waypoint_polyline",
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.02, 0.0, 0.5],
                [0.02, 0.02, 0.5],
                [0.0, 0.02, 0.5],
            ]
        ),
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.02, 0.0, 0.5],
                [0.02, 0.02, 0.5],
                [0.0, 0.02, 0.5],
            ]
        ),
    )
    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.path = path
    env.internal_waypoint_capture_radius = 0.01
    env.path_progress = 0.0
    env._initialize_internal_waypoint_gate()

    assert env._internal_waypoint_progress_limit(0.04) == pytest.approx(0.02)
    # Being spatially close is not enough when the ordered path has not
    # advanced near this waypoint yet.
    assert not env._capture_internal_waypoint(np.asarray([0.02, 0.0, 0.5]))
    env.path_progress = 0.015
    assert env._capture_internal_waypoint(np.asarray([0.019, 0.001, 0.5]))
    assert env.internal_waypoints_completed == 1
    assert env.next_internal_waypoint_index == 2
    assert env._internal_waypoint_progress_limit(0.06) == pytest.approx(0.04)


def test_reference_target_stops_at_uncaptured_internal_waypoint() -> None:
    path = ReferencePath.from_points(
        "waypoint_polyline",
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.02, 0.0, 0.5],
                [0.02, 0.02, 0.5],
                [0.0, 0.02, 0.5],
            ]
        ),
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.02, 0.0, 0.5],
                [0.02, 0.02, 0.5],
                [0.0, 0.02, 0.5],
            ]
        ),
    )
    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.path = path
    env.internal_waypoint_capture_radius = 0.01
    env.target_lead_distance = 0.008
    env.lookahead_distance = 0.018
    env.path_progress = 0.018
    env._initialize_internal_waypoint_gate()
    env._update_reference_targets()
    np.testing.assert_allclose(env.current_target, path.anchors[1], atol=1e-8)
    assert env.lookahead_target[1] > 0.0

    env.next_internal_waypoint_index = 2
    env._update_reference_targets()
    assert env.current_target[1] > 0.0


def test_reference_action_does_not_extrapolate_beyond_path_end() -> None:
    path = ReferencePath.from_points(
        "waypoint_polyline",
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.1, 0.0, 0.5],
                [0.2, 0.0, 0.5],
            ]
        ),
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.1, 0.0, 0.5],
                [0.2, 0.0, 0.5],
            ]
        ),
    )
    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.path = path
    env.internal_waypoint_capture_radius = 0.0
    env.target_lead_distance = 0.05
    env.lookahead_distance = 0.02
    env.path_progress = path.length - 0.01
    env.path_anchor_policy_actions = np.asarray(
        [[0.0, 0.0], [0.5, 0.25], [1.0, 0.75]], dtype=np.float32
    )
    env.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(2,), dtype=np.float32
    )
    env._initialize_internal_waypoint_gate()

    env._update_reference_targets()

    np.testing.assert_allclose(env.current_target, path.anchors[-1], atol=1e-8)
    np.testing.assert_allclose(
        env.reference_policy_action,
        env.path_anchor_policy_actions[-1],
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "setting",
    (
        {"internal_waypoint_bonus": 1.0},
        {"internal_waypoint_progress_scale": 1.0},
        {"internal_waypoint_distance_penalty_scale": 1.0},
    ),
)
def test_internal_waypoint_rewards_require_capture_radius(
    tmp_path, setting
) -> None:
    with pytest.raises(ValueError, match="requires a positive capture radius"):
        ManiSoftWaypointSACEnv(
            _scenario(tmp_path),
            internal_waypoint_capture_radius=0.0,
            **setting,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"waypoint_segment_count_range": (0, 2)}, "count_range"),
        ({"waypoint_segment_length_range": (0.03, 0.01)}, "length_range"),
        ({"waypoint_maximum_extent": 0.005}, "maximum_extent"),
        ({"waypoint_maximum_turn_degrees": 181.0}, "turn_degrees"),
        ({"waypoint_vertical_delta_range": (0.01, -0.01)}, "vertical_delta"),
        ({"waypoint_single_line_probability": 1.01}, "single_line_probability"),
        (
            {
                "waypoint_segment_count_range": (2, 3),
                "waypoint_segment_count_probabilities": (1.0,),
            },
            "count_probabilities",
        ),
    ],
)
def test_table_waypoint_polyline_validates_sampling_configuration(
    kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        WaypointPathGenerator(**kwargs)


class _FakeMuscle:
    def __init__(self) -> None:
        self.activation = np.zeros((6, 3))

    def set_activation(self, activation: np.ndarray) -> None:
        self.activation = np.asarray(activation).copy()

    def evaluate(self, element_lengths: np.ndarray) -> np.ndarray:
        return np.zeros((20, 3))


class _FakeSimulation:
    def __init__(self) -> None:
        class SoftRobotState:
            element_positions = np.asarray(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.9]], dtype=np.float64
            )

        class Backend:
            softrobot_state = SoftRobotState()

        self._backend = Backend()

    def step_with_torque_callback(self, callback) -> None:
        callback(np.ones(20))


def _scenario(tmp_path) -> str:
    path = tmp_path / "scenario.yaml"
    path.write_text(
        "backend:\n  dt: 0.0002\n"
        "environment:\n  update_interval: 100\n",
        encoding="utf-8",
    )
    return str(path)


def test_entry_mid_anchor_sampling_balances_episode_length_bias(tmp_path) -> None:
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="entry_mid",
        entry_mid_anchor_probability=0.75,
        entry_mid_advance_fraction_range=(0.30, 0.55),
        entry_mid_anchor_fraction_range=(0.55, 0.85),
    )
    env._np_random = np.random.default_rng(123)
    samples = np.asarray(
        [env._sample_entry_fraction("entry_mid", None) for _ in range(2000)]
    )
    assert np.all((samples >= 0.30) & (samples <= 0.85))
    assert np.mean(samples >= 0.55) == pytest.approx(0.75, abs=0.03)
    assert env._sample_entry_fraction("entry_mid", 0.42) == pytest.approx(0.42)


def test_entry_sampling_mixes_exact_upright_starts_with_retention_anchors(
    tmp_path,
) -> None:
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="entry",
        entry_anchor_probability=0.65,
        entry_anchor_fraction_range=(0.30, 0.65),
    )
    env._np_random = np.random.default_rng(321)
    samples = np.asarray(
        [env._sample_entry_fraction("entry", None) for _ in range(2000)]
    )
    anchors = samples > 0.0
    assert np.mean(anchors) == pytest.approx(0.65, abs=0.03)
    assert np.all((samples[anchors] >= 0.30) & (samples[anchors] <= 0.65))
    assert np.all(samples[~anchors] == 0.0)
    assert env._sample_entry_fraction("entry", 0.12) == pytest.approx(0.12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"entry_mid_anchor_probability": 1.1}, "anchor_probability"),
        (
            {"entry_mid_advance_fraction_range": (0.60, 0.30)},
            "advance_fraction_range",
        ),
        (
            {"entry_mid_anchor_fraction_range": (0.55, 1.10)},
            "anchor_fraction_range",
        ),
        ({"entry_anchor_probability": -0.1}, "entry_anchor_probability"),
        (
            {"entry_anchor_fraction_range": (0.65, 0.30)},
            "entry_anchor_fraction_range",
        ),
    ],
)
def test_entry_mid_anchor_sampling_validates_configuration(
    tmp_path, kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        ManiSoftWaypointSACEnv(_scenario(tmp_path), **kwargs)


def test_environment_observation_and_action_rate_limit(monkeypatch, tmp_path) -> None:
    state = np.zeros(45, dtype=np.float32)
    state[30:33] = [0.0, 0.0, 0.94]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path), curriculum="point", episode_steps=10
    )
    monkeypatch.setattr(env, "_physical_state", lambda: state.copy())
    observation, info = env.reset(seed=5)
    assert observation.shape == (MANISOFT_WAYPOINT_SAC_OBSERVATION_DIM,)
    assert env.observation_space.contains(observation)
    assert info["path_family"] == "point"
    initial_progress = info["path_progress_m"]

    next_observation, _, terminated, truncated, info = env.step(
        np.full(18, 0.30, dtype=np.float32)
    )
    assert not terminated and not truncated
    np.testing.assert_allclose(info["applied_action"], 0.01, atol=1e-7)
    np.testing.assert_allclose(info["applied_delta_action"], 0.01, atol=1e-7)
    assert info["action_rate_clipped_ratio"] == pytest.approx(1.0)
    assert info["requested_rate_penalty"] > 0
    # A stationary tip no longer earns clock/tolerance-gated reference progress.
    assert info["path_progress_m"] == pytest.approx(initial_progress)
    assert not info["reference_advanced"]
    np.testing.assert_allclose(next_observation[45:63], 0.01, atol=1e-7)


def test_gate74_appends_features_without_changing_physical_state(
    monkeypatch, tmp_path
) -> None:
    state = np.zeros(45, dtype=np.float32)
    state[30:33] = [0.0, 0.0, 0.94]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="point",
        action_mode="table_equilibrium",
        observation_mode="gate74",
    )
    observation, _ = env.reset(seed=16)
    assert observation.shape == (74,)
    np.testing.assert_allclose(observation[:45], state)
    np.testing.assert_allclose(observation[70:72], 0.0)
    assert observation[72] == pytest.approx(0.0)


def test_internal_waypoint_potential_preserves_original_45_state(
    monkeypatch, tmp_path
) -> None:
    reset_state = np.zeros(45, dtype=np.float32)
    reset_state[30:33] = [0.0, 0.0, 0.94]
    moved_state = reset_state.copy()
    moved_state[30:33] = [0.005, 0.0, 0.94]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return reset_state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="point",
        internal_waypoint_capture_radius=0.010,
        internal_waypoint_progress_scale=0.25,
        internal_waypoint_distance_penalty_scale=0.10,
    )
    env.reset(seed=13)
    anchors = np.asarray(
        [[0.0, 0.0, 0.94], [0.02, 0.0, 0.94], [0.02, 0.02, 0.94]]
    )
    env.path = ReferencePath.from_points("potential_test", anchors, anchors)
    env.path_progress = 0.0
    env.desired_speed = 0.02
    env._initialize_internal_waypoint_gate()
    env._update_reference_targets()
    monkeypatch.setattr(env, "_physical_state", lambda: moved_state.copy())

    observation, _, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and not truncated
    assert info["internal_waypoint_distance_delta"] == pytest.approx(0.005)
    assert info["normalized_internal_waypoint_progress"] == pytest.approx(2.0)
    assert info["normalized_internal_waypoint_capture_error"] == pytest.approx(0.5)
    np.testing.assert_allclose(observation[:45], moved_state)
    assert observation.shape == (70,)


def test_active_waypoint_distance_stall_has_dedicated_truncation(
    monkeypatch, tmp_path
) -> None:
    state = np.zeros(45, dtype=np.float32)
    state[30:33] = [0.0, 0.0, 0.94]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="point",
        internal_waypoint_capture_radius=0.010,
        waypoint_stall_steps=2,
        stall_grace_steps=100,
        stall_window_steps=100,
    )
    env.reset(seed=14)
    anchors = np.asarray(
        [[0.0, 0.0, 0.94], [0.02, 0.0, 0.94], [0.02, 0.02, 0.94]]
    )
    env.path = ReferencePath.from_points("waypoint_stall_test", anchors, anchors)
    env.path_progress = 0.0
    env._initialize_internal_waypoint_gate()
    env._update_reference_targets()
    monkeypatch.setattr(env, "_physical_state", lambda: state.copy())

    _, _, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and truncated
    assert info["waypoint_stalled"]
    assert not info["path_progress_stalled"]
    assert info["stalled"]
    assert reward < -1.0


def test_table_equilibrium_action_maps_two_dimensions_to_physical_action(
    tmp_path,
) -> None:
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="table_local_line",
        action_mode="table_equilibrium",
        equilibrium_rotation_degrees=4.0,
        equilibrium_xy_scale_delta=0.03,
    )
    base = np.linspace(-0.20, 0.20, 18, dtype=np.float32)
    env.equilibrium_action = base.copy()
    physical = env._physical_action_request(np.asarray([1.0, 1.0]))
    angle = np.deg2rad(4.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    expected = base.reshape(6, 3).astype(np.float64)
    expected[:, :2] = 1.03 * expected[:, :2] @ rotation.T
    np.testing.assert_allclose(physical, expected.reshape(-1), atol=1e-7)
    assert env.action_space.shape == (2,)
    with pytest.raises(ValueError, match="2-D action"):
        env._physical_action_request(np.zeros(18))


def test_cartesian_delta_action_integrates_signed_calibrated_columns(
    tmp_path,
) -> None:
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="table_local_line",
        action_mode="table_equilibrium",
        cartesian_action_step_scale=0.04,
        cartesian_action_leak=0.0,
    )
    env.action_mode = "table_cartesian_delta"
    env.entry_index = 0
    env.previous_action = np.linspace(-0.10, 0.10, 18, dtype=np.float32)
    env.cartesian_positive_deltas = np.zeros((1, 2, 18), dtype=np.float32)
    env.cartesian_negative_deltas = np.zeros((1, 2, 18), dtype=np.float32)
    env.cartesian_positive_deltas[0, 0, 3] = 0.10
    env.cartesian_negative_deltas[0, 1, 7] = -0.20
    physical = env._physical_action_request(np.asarray([0.5, -0.25]))
    expected = env.previous_action.copy()
    expected[3] += 0.04 * 0.5 * 0.10
    expected[7] += 0.04 * 0.25 * -0.20
    np.testing.assert_allclose(physical, expected, atol=1e-7)


def test_cartesian_prior_blends_policy_with_target_feedback() -> None:
    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    env.action_mode = "table_cartesian_delta"
    env.cartesian_prior_weight = 0.60
    env.cartesian_prior_internal_waypoints_only = False
    env.cartesian_prior_proportional_gain = 20.0
    env.cartesian_prior_feedforward_scale = 1.0
    env.cartesian_command_distance = 0.01
    env.cartesian_action_step_scale = 0.035
    env.cartesian_action_leak = 0.008
    env.cartesian_prior_start_tip = np.asarray([0.0, 0.0, 0.5])
    env.current_target = np.asarray([0.006, -0.003, 0.5])
    env.last_physical_state = np.zeros(45, dtype=np.float32)
    env.last_physical_state[30:33] = [0.002, -0.001, 0.5]

    raw = np.asarray([0.4, 0.2], dtype=np.float32)
    blended = env._blend_cartesian_prior(raw)
    steady_displacement = 0.01 * 0.035 / 0.008
    expected_prior = np.asarray(
        [
            0.006 / steady_displacement + 20.0 * 0.004,
            -0.003 / steady_displacement - 20.0 * 0.002,
        ],
        dtype=np.float32,
    )
    expected = 0.40 * raw + 0.60 * expected_prior
    np.testing.assert_allclose(env.last_raw_policy_action, raw)
    np.testing.assert_allclose(env.last_controller_prior_action, expected_prior)
    np.testing.assert_allclose(blended, expected)


def test_cartesian_prior_can_switch_off_after_internal_waypoints() -> None:
    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    env.action_mode = "table_cartesian_delta"
    env.cartesian_prior_weight = 0.50
    env.cartesian_prior_internal_waypoints_only = True
    env.path = ReferencePath.from_points(
        "waypoint_polyline",
        np.asarray([[0.0, 0.0, 0.5], [0.02, 0.0, 0.5], [0.04, 0.0, 0.5]]),
        np.asarray([[0.0, 0.0, 0.5], [0.02, 0.0, 0.5], [0.04, 0.0, 0.5]]),
    )
    env.next_internal_waypoint_index = 2
    raw = np.asarray([0.3, -0.2], dtype=np.float32)
    np.testing.assert_allclose(env._blend_cartesian_prior(raw), raw)
    np.testing.assert_allclose(env.last_controller_prior_action, 0.0)


def test_equilibrium_path_prior_adds_bounded_sac_residual() -> None:
    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    env.cartesian_prior_weight = 0.0
    env.cartesian_prior_internal_waypoints_only = False
    env.equilibrium_path_prior_weight = 1.0
    env.equilibrium_path_residual_scale = 0.03
    env.reference_policy_action = np.asarray([0.25, 0.0], dtype=np.float32)
    raw = np.asarray([0.50, -0.50], dtype=np.float32)
    blended = env._blend_cartesian_prior(raw)
    np.testing.assert_allclose(blended, [0.265, -0.015], atol=1e-7)
    np.testing.assert_allclose(env.last_controller_prior_action, [0.25, 0.0])


def test_cartesian_prior_adds_bounded_sac_residual() -> None:
    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    env.action_mode = "table_cartesian_delta"
    env.cartesian_prior_weight = 1.0
    env.cartesian_prior_residual_scale = 0.10
    env.cartesian_prior_internal_waypoints_only = False
    env.equilibrium_path_prior_weight = 0.0
    env.equilibrium_path_residual_scale = 0.0
    env.last_physical_state = np.zeros(45, dtype=np.float32)
    env._cartesian_prior_action = lambda _: np.asarray([0.25, -0.10])

    blended = env._blend_cartesian_prior(np.asarray([0.50, -0.50]))

    np.testing.assert_allclose(blended, [0.30, -0.15], atol=1e-7)


def test_certified_long_path_samples_three_to_five_commanded_points() -> None:
    class Generator:
        waypoint_segment_count_range = (3, 5)
        waypoint_segment_count_probabilities = np.asarray([0.2, 0.3, 0.5])
        dense_spacing = 0.005

    env = ManiSoftWaypointSACEnv.__new__(ManiSoftWaypointSACEnv)
    env.equilibrium_path_entry_index = 1
    env.equilibrium_path_tip_positions = np.column_stack(
        (
            np.arange(5, dtype=np.float32) * -0.10,
            np.full(5, 0.72, dtype=np.float32),
            np.full(5, 0.505, dtype=np.float32),
        )
    )
    env.equilibrium_path_policy_actions = np.column_stack(
        (np.linspace(0.0, 1.0, 5, dtype=np.float32), np.zeros(5))
    )
    env.path_generator = Generator()
    env._np_random = np.random.default_rng(8)
    commanded_point_counts = {
        len(env._table_equilibrium_long_path(1).anchors) - 1 for _ in range(100)
    }
    assert commanded_point_counts == {3, 4, 5}
    five_target_path = None
    for _ in range(100):
        candidate = env._table_equilibrium_long_path(1)
        if len(candidate.anchors) == 6:
            five_target_path = candidate
            break
    assert five_target_path is not None
    np.testing.assert_allclose(
        five_target_path.anchors[:, 0],
        [0.0, -0.1, -0.2, -0.3, -0.4, -0.3],
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="not covered"):
        env._table_equilibrium_long_path(0)


def test_stationary_policy_is_truncated_as_stalled(monkeypatch, tmp_path) -> None:
    state = np.zeros(45, dtype=np.float32)
    state[30:33] = [0.0, 0.0, 0.94]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="point",
        stall_grace_steps=1,
        stall_window_steps=2,
    )
    monkeypatch.setattr(env, "_physical_state", lambda: state.copy())
    env.reset(seed=9)
    _, _, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and truncated
    assert info["stalled"]
    assert reward < -1.0


def test_environment_terminates_below_virtual_table(monkeypatch, tmp_path) -> None:
    reset_state = np.zeros(45, dtype=np.float32)
    reset_state[30:33] = [0.0, 0.0, 0.94]
    below_table = reset_state.copy()
    below_table[30:33] = [0.0, 0.4, 0.39]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return reset_state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path), curriculum="point", table_violation_penalty=300.0
    )
    env.reset(seed=7)
    monkeypatch.setattr(env, "_physical_state", lambda: below_table.copy())
    _, reward, terminated, truncated, info = env.step(np.zeros(18))
    assert terminated and not truncated
    assert info["table_violation"]
    assert reward < -299.0


def test_environment_can_log_table_violation_without_terminating(
    monkeypatch, tmp_path
) -> None:
    reset_state = np.zeros(45, dtype=np.float32)
    reset_state[30:33] = [0.0, 0.0, 0.94]
    below_table = reset_state.copy()
    below_table[30:33] = [0.0, 0.4, 0.39]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return reset_state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="point",
        table_violation_penalty=0.0,
        terminate_on_table_violation=False,
    )
    env.reset(seed=7)
    monkeypatch.setattr(env, "_physical_state", lambda: below_table.copy())
    _, _, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and not truncated
    assert info["table_violation"]


def test_environment_bounds_terminal_precision_attempt(monkeypatch, tmp_path) -> None:
    reset_state = np.zeros(45, dtype=np.float32)
    reset_state[30:33] = [0.0, 0.0, 0.94]
    off_endpoint = reset_state.copy()
    off_endpoint[30:33] = [0.02, 0.015, 0.94]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return reset_state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="point",
        success_threshold=0.010,
        terminal_capture_radius=0.020,
        terminal_settle_steps=2,
        terminal_timeout_penalty=50.0,
    )
    env.reset(seed=11)
    points = np.asarray([[0.0, 0.0, 0.94], [0.02, 0.0, 0.94]])
    env.path = ReferencePath.from_points("terminal_test", points, points)
    env.path_progress = 0.0
    env._update_reference_targets()
    monkeypatch.setattr(env, "_physical_state", lambda: off_endpoint.copy())

    _, _, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and not truncated
    assert not info["terminal_timeout"]
    env.step(np.zeros(18))
    _, reward, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and truncated
    assert info["terminal_timeout"]
    assert reward < -49.0


def test_projection_end_outside_capture_radius_does_not_start_terminal_timeout(
    monkeypatch, tmp_path
) -> None:
    reset_state = np.zeros(45, dtype=np.float32)
    reset_state[30:33] = [0.0, 0.0, 0.94]
    far_from_endpoint = reset_state.copy()
    far_from_endpoint[30:33] = [0.02, 0.05, 0.94]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return reset_state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="point",
        success_threshold=0.010,
        terminal_capture_radius=0.020,
        terminal_settle_steps=2,
    )
    env.reset(seed=12)
    points = np.asarray([[0.0, 0.0, 0.94], [0.02, 0.0, 0.94]])
    env.path = ReferencePath.from_points("terminal_capture_test", points, points)
    env.path_progress = 0.0
    env._update_reference_targets()
    monkeypatch.setattr(env, "_physical_state", lambda: far_from_endpoint.copy())

    for _ in range(4):
        _, _, terminated, truncated, info = env.step(np.zeros(18))
        assert not terminated and not truncated
        assert info["geometric_path_end"]
        assert not info["terminal_capture"]
        assert not info["terminal_timeout"]


def test_terminal_timeout_requires_consecutive_capture_dwell(
    monkeypatch, tmp_path
) -> None:
    reset_state = np.zeros(45, dtype=np.float32)
    reset_state[30:33] = [0.0, 0.0, 0.94]
    inside = reset_state.copy()
    inside[30:33] = [0.02, 0.015, 0.94]
    outside = reset_state.copy()
    outside[30:33] = [0.02, 0.030, 0.94]

    def fake_reset(self, *, seed=None, options=None):
        self._np_random = np.random.default_rng(seed)
        self.sim = _FakeSimulation()
        self.muscle = _FakeMuscle()
        return reset_state.copy(), {}

    monkeypatch.setattr(ManiSoftTipTrackingEnv, "reset", fake_reset)
    env = ManiSoftWaypointSACEnv(
        _scenario(tmp_path),
        curriculum="point",
        success_threshold=0.010,
        terminal_capture_radius=0.020,
        terminal_settle_steps=2,
    )
    env.reset(seed=15)
    points = np.asarray([[0.0, 0.0, 0.94], [0.02, 0.0, 0.94]])
    env.path = ReferencePath.from_points("terminal_dwell_test", points, points)
    env.path_progress = 0.0
    env._initialize_internal_waypoint_gate()
    env._update_reference_targets()
    states = iter((inside, outside, inside))
    monkeypatch.setattr(env, "_physical_state", lambda: next(states).copy())

    env.step(np.zeros(18))
    assert env.terminal_entry_step == 1
    env.step(np.zeros(18))
    assert env.terminal_entry_step is None
    _, _, terminated, truncated, info = env.step(np.zeros(18))
    assert not terminated and not truncated
    assert env.terminal_entry_step == 3
    assert not info["terminal_timeout"]
