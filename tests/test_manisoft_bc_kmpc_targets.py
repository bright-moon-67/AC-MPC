import numpy as np

from scripts.train_manisoft_bc_kmpc_bc import _future_targets


def test_future_targets_stop_at_episode_and_waypoint_boundaries():
    actions = np.arange(12, dtype=np.float32).reshape(6, 2)
    episodes = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    stages = np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int64)

    future, mask = _future_targets(actions, episodes, stages, horizon=3)

    np.testing.assert_allclose(future[0, :2], actions[:2])
    np.testing.assert_allclose(mask[0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(mask[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(mask[2], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(mask[3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(mask[4], [1.0, 1.0, 0.0])
