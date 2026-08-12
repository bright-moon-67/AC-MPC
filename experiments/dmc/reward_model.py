"""Transition reward model shared by DMC Koopman training and MVE actors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def transition_reward_input_contract() -> dict[str, Any]:
    """Return the exact, checkpointed semantics of reward-model features."""

    return {
        "kind": "dmc_transition_reward_input_v1",
        "state": "koopman_train_split_normalized_state",
        "action": "applied_action",
        "next_state": "koopman_train_split_normalized_next_state",
        "target": "dmc_reward_in_closed_interval_0_1",
        "training_sample": "first_transition_of_each_k_step_window",
    }


class TransitionRewardModel(nn.Module):
    """Predict DMC reward from one normalized transition.

    Both states use the train-split Koopman normalizer.  ``applied_action`` is
    the command actually passed to DMC (the dataset's ``action`` field), not a
    potentially unclipped requested action.  DMC task rewards are bounded in
    ``[0, 1]``, so the network ends in a sigmoid and preserves that contract for
    imagined transitions used by MVE.
    """

    ARCHITECTURE = "dmc_transition_reward_mlp_v1"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(action_dim) is not int or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if isinstance(hidden_dims, (str, bytes)) or not isinstance(
            hidden_dims, Sequence
        ):
            raise ValueError("hidden_dims must be a sequence of positive integers")
        if not hidden_dims or any(
            type(width) is not int or width < 1
            for width in hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integers")
        widths = tuple(int(width) for width in hidden_dims)
        if type(activation) is not str or activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation {activation!r}")
        activation_name = activation
        activation_type = _ACTIVATIONS[activation_name]

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = widths
        self.activation = activation_name
        dimensions = [
            2 * self.state_dim + self.action_dim,
            *self.hidden_dims,
            1,
        ]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.append(activation_type())
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        normalized_state: torch.Tensor,
        applied_action: torch.Tensor,
        normalized_next_state: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_state.shape[-1:] != (self.state_dim,):
            raise ValueError(
                f"Expected state dimension {self.state_dim}, got "
                f"{normalized_state.shape[-1:]}"
            )
        if normalized_next_state.shape != normalized_state.shape:
            raise ValueError("normalized_next_state must match normalized_state")
        if applied_action.shape[:-1] != normalized_state.shape[:-1] or (
            applied_action.shape[-1:] != (self.action_dim,)
        ):
            raise ValueError(
                "applied_action must share the state batch shape and have "
                f"dimension {self.action_dim}"
            )
        features = torch.cat(
            (normalized_state, applied_action, normalized_next_state), dim=-1
        )
        return self.network(features).squeeze(-1)

    def architecture(self) -> dict[str, Any]:
        return {
            "architecture": self.ARCHITECTURE,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
        }

    @classmethod
    def from_architecture(cls, architecture: Mapping[str, Any]) -> "TransitionRewardModel":
        if not isinstance(architecture, Mapping):
            raise ValueError("Reward-model architecture must be a mapping")
        expected_keys = {
            "architecture",
            "state_dim",
            "action_dim",
            "hidden_dims",
            "activation",
        }
        if set(architecture) != expected_keys:
            raise ValueError(
                "Reward-model architecture fields differ from the strict contract"
            )
        if architecture["architecture"] != cls.ARCHITECTURE:
            raise ValueError(
                f"Unsupported reward-model architecture "
                f"{architecture['architecture']!r}"
            )
        if not isinstance(architecture["hidden_dims"], list):
            raise ValueError("Reward-model hidden_dims must be a JSON list")
        model = cls(
            state_dim=architecture["state_dim"],
            action_dim=architecture["action_dim"],
            hidden_dims=architecture["hidden_dims"],
            activation=architecture["activation"],
        )
        if model.architecture() != dict(architecture):
            raise ValueError("Reward-model architecture is not canonical")
        return model


def reward_model_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> TransitionRewardModel:
    """Strictly reconstruct the reward model embedded in a Koopman checkpoint."""

    if not isinstance(payload, Mapping):
        raise ValueError("Koopman checkpoint must be a mapping")
    required = {
        "reward_model_architecture",
        "reward_model_input_contract",
        "reward_model_state",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(
            f"Koopman checkpoint is missing reward-model fields: {sorted(missing)}"
        )
    if payload["reward_model_input_contract"] != transition_reward_input_contract():
        raise ValueError("Koopman checkpoint reward-model input contract is invalid")
    state = payload["reward_model_state"]
    if not isinstance(state, Mapping):
        raise ValueError("reward_model_state must be a mapping")
    model = TransitionRewardModel.from_architecture(
        payload["reward_model_architecture"]
    ).to(device)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("reward_model_state contains NaN or Inf")
    return model
