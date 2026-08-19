import gymnasium as gym
import numpy as np

from antmaze_ac.envs.delta_action_wrapper import DeltaActionWrapper, make_delta_vector_env


class CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def __init__(self, horizon=3):
        self.horizon = horizon
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return (
            np.full(3, self.step_count, dtype=np.float32),
            0.0,
            self.step_count >= self.horizon,
            False,
            {},
        )


def test_reset_accumulation_clipping_and_info():
    env = DeltaActionWrapper(CountingEnv())
    observation, _ = env.reset()
    assert observation.shape == (5,)
    np.testing.assert_array_equal(observation[-2:], 0)

    observation, *_rest, info = env.step(np.array([0.4, -0.25], dtype=np.float32))
    np.testing.assert_allclose(observation[-2:], [0.4, -0.25])
    observation, *_rest, info = env.step(np.array([0.8, -2.0], dtype=np.float32))
    np.testing.assert_allclose(observation[-2:], [1.0, -1.0])
    np.testing.assert_allclose(info["applied_delta_action"], [0.6, -0.75])
    assert info["action_saturation_ratio"] == 1.0

    observation, _ = env.reset()
    np.testing.assert_array_equal(observation[-2:], 0)


def test_shape_and_finite_checks():
    env = DeltaActionWrapper(CountingEnv())
    env.reset()
    for bad in (np.zeros(3), np.array([np.nan, 0])):
        try:
            env.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad action must fail")


def test_vector_environments_reset_independently():
    vector = make_delta_vector_env([lambda: CountingEnv(1), lambda: CountingEnv(3)])
    observation, _ = vector.reset()
    assert observation.shape == (2, 5)
    action = np.array([[0.5, 0.0], [0.2, 0.0]], dtype=np.float32)
    _, _, terminated, _, _ = vector.step(action)
    np.testing.assert_array_equal(terminated, [True, False])
    # Gymnasium NEXT_STEP autoreset resets only env 0 on the following call;
    # that call returns the reset observation without applying its action.
    observation, *_ = vector.step(action)
    np.testing.assert_allclose(observation[0, -2:], [0.0, 0.0])
    np.testing.assert_allclose(observation[1, -2:], [0.4, 0.0])
    observation, *_ = vector.step(action)
    np.testing.assert_allclose(observation[0, -2:], [0.5, 0.0])
    vector.close()
