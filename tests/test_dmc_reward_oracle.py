from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.dmc.reward_oracle import (
    OFFICIAL_OBSERVATION_ORACLE,
    ORACLE_PARITY_MAX_ABS_ERROR,
    ExactObservationRewardOracle,
    cartpole_swingup_official_reward,
    exact_reward_oracle_metadata,
    walker_run_exact_reward_numpy,
    walker_run_official_reward,
)
from experiments.dmc.tasks.adapter import make_dmc_adapter


def test_cartpole_exact_reward_matches_live_dmc_transitions() -> None:
    pytest.importorskip("dm_control")
    env = make_dmc_adapter("cartpole_swingup", seed=321)
    rng = np.random.default_rng(20260811)
    max_error = 0.0
    try:
        env.reset(seed=321)
        for _ in range(1000):
            action = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
            next_observation, reward, done, info = env.step(action)
            predicted = cartpole_swingup_official_reward(
                torch.from_numpy(next_observation).unsqueeze(0),
                torch.from_numpy(info["applied_action"]).unsqueeze(0),
            )
            max_error = max(max_error, abs(float(predicted.item()) - reward))
            if done:
                env.reset(seed=321)
    finally:
        env.close()
    assert max_error <= ORACLE_PARITY_MAX_ABS_ERROR


def test_walker_exact_reward_matches_live_dmc_transitions() -> None:
    pytest.importorskip("dm_control")
    env = make_dmc_adapter("walker_run", seed=654)
    rng = np.random.default_rng(20260818)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    try:
        env.reset(seed=654)
        for _ in range(200):
            action = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
            next_observation, reward, done, info = env.step(action)
            observations.append(next_observation)
            actions.append(info["applied_action"])
            rewards.append(reward)
            if done:
                env.reset(seed=654)
    finally:
        env.close()

    observation_batch = np.asarray(observations, dtype=np.float32)
    action_batch = np.asarray(actions, dtype=np.float32)
    live_reward = np.asarray(rewards, dtype=np.float32)
    numpy_reward = walker_run_exact_reward_numpy(observation_batch, action_batch)
    torch_reward = walker_run_official_reward(
        torch.from_numpy(observation_batch), torch.from_numpy(action_batch)
    ).numpy()
    np.testing.assert_allclose(
        numpy_reward, live_reward, rtol=0.0, atol=ORACLE_PARITY_MAX_ABS_ERROR
    )
    np.testing.assert_allclose(
        torch_reward, live_reward, rtol=0.0, atol=ORACLE_PARITY_MAX_ABS_ERROR
    )


def test_exact_oracle_undoes_koopman_normalization() -> None:
    center = torch.tensor([0.2, -0.1, 0.3, 0.4, -0.2])
    scale = torch.tensor([2.0, 0.5, 1.5, 3.0, 0.25])
    next_observation = torch.tensor(
        [[0.5, 0.75, -0.4, 0.1, 1.25], [-0.2, -0.6, 0.7, -1.0, -2.0]]
    )
    action = torch.tensor([[0.3], [-0.8]])
    oracle = ExactObservationRewardOracle(
        "cartpole_swingup", center=center, scale=scale
    )
    normalized_next = (next_observation - center) / scale
    actual = oracle(torch.zeros_like(normalized_next), action, normalized_next)
    expected = cartpole_swingup_official_reward(next_observation, action)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert oracle.metadata() == exact_reward_oracle_metadata("cartpole_swingup")
    assert oracle.metadata()["source"] == OFFICIAL_OBSERVATION_ORACLE
    assert all(not parameter.requires_grad for parameter in oracle.parameters())


@pytest.mark.parametrize(
    ("observation", "action", "message"),
    [
        (torch.zeros(2, 4), torch.zeros(2, 1), "next_observation"),
        (torch.zeros(2, 5), torch.zeros(2, 2), "applied_action"),
        (torch.zeros(2, 5), torch.zeros(3, 1), "batch shapes"),
        (torch.full((2, 5), float("nan")), torch.zeros(2, 1), "NaN or Inf"),
    ],
)
def test_cartpole_exact_reward_rejects_malformed_inputs(
    observation: torch.Tensor,
    action: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((ValueError, FloatingPointError), match=message):
        cartpole_swingup_official_reward(observation, action)


def test_exact_oracle_rejects_unverified_task() -> None:
    with pytest.raises(ValueError, match="No verified exact"):
        ExactObservationRewardOracle(
            "hopper_hop", torch.zeros(15), torch.ones(15)
        )
