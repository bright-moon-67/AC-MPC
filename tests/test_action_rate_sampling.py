import numpy as np

from antmaze_ac.data.action_rate_sampling import (
    ActionRateStratifiedSampler,
    rate_bin_indices,
    stratified_sample_indices,
    transition_action_rates,
    window_action_rates,
)


def test_transition_rates_reset_previous_action_at_episode_boundary():
    actions = np.array([[0.0], [0.2], [-0.1], [0.3]], dtype=np.float32)
    episode_ids = np.array([0, 0, 1, 1])

    rates = transition_action_rates(actions, episode_ids)

    np.testing.assert_allclose(rates, [0.0, 0.2, 0.1, 0.4])


def test_window_rates_include_history_and_future_without_crossing_episode():
    actions = np.array(
        [[0.0], [0.01], [0.02], [0.12], [0.0], [0.3], [-0.3]],
        dtype=np.float32,
    )
    episode_ids = np.array([0, 0, 0, 0, 1, 1, 1])
    starts = np.array([2, 4])

    rates = window_action_rates(
        actions, episode_ids, starts, transitions=2, history_steps=2
    )

    np.testing.assert_allclose(rates, [0.1, 0.3])


def test_stratified_sampling_meets_requested_proportions():
    rates = np.concatenate(
        (
            np.full(100, 0.001),
            np.full(20, 0.02),
            np.full(5, 0.2),
        )
    )
    selected = stratified_sample_indices(
        rates,
        edges=[0.005, 0.05],
        fractions=[0.5, 0.35, 0.15],
        num_samples=100,
        rng=np.random.default_rng(42),
    )

    counts = np.bincount(
        rate_bin_indices(rates[selected], [0.005, 0.05]), minlength=3
    )
    np.testing.assert_array_equal(counts, [50, 35, 15])


def test_sampler_changes_each_epoch_but_is_reproducible():
    rates = np.linspace(0.0, 0.2, 200)
    first = ActionRateStratifiedSampler(
        rates, [0.005, 0.05], [0.5, 0.35, 0.15], 80, seed=7
    )
    second = ActionRateStratifiedSampler(
        rates, [0.005, 0.05], [0.5, 0.35, 0.15], 80, seed=7
    )

    first.set_epoch(3)
    second.set_epoch(3)
    epoch_three = list(first)
    assert epoch_three == list(second)
    first.set_epoch(4)
    assert epoch_three != list(first)
