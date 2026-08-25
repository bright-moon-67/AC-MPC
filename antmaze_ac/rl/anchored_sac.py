"""SAC with a frozen source-policy trust-region penalty for curriculum transfer."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class AnchoredSAC(SAC):
    """Keep fine-tuning close to the policy present when anchoring is enabled.

    The regularizer compares deterministic, squashed actions on the same
    normalized replay observations used by SAC.  The source actor is frozen,
    excluded from serialization, and therefore cannot affect inference.
    """

    anchor_actor = None
    actor_anchor_coef = 0.0
    source_policy_warmup = False
    actor_learning_starts = 0

    def enable_actor_anchor(self, coefficient: float) -> None:
        if coefficient <= 0:
            raise ValueError("actor anchor coefficient must be positive")
        self.actor_anchor_coef = float(coefficient)
        self.anchor_actor = deepcopy(self.actor).to(self.device)
        self.anchor_actor.set_training_mode(False)
        for parameter in self.anchor_actor.parameters():
            parameter.requires_grad_(False)

    def enable_source_policy_warmup(self) -> None:
        """Collect a new curriculum buffer without uniform random actions.

        SB3 normally samples the entire action box before ``learning_starts``.
        That is appropriate for a fresh low-dimensional policy, but unsafe for
        an 18-D soft-arm residual policy transferred to a nearby curriculum.
        Use the loaded source actor deterministically during that buffer-only
        phase; standard stochastic SAC actions resume once learning starts.
        """

        self.source_policy_warmup = True

    def delay_actor_updates_until(self, timestep: int) -> None:
        if timestep < self.num_timesteps:
            raise ValueError("actor update timestep cannot be in the past")
        self.actor_learning_starts = int(timestep)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.source_policy_warmup and self.num_timesteps < learning_starts:
            assert self._last_obs is not None, "last observation is unavailable"
            unscaled_action, _ = self.predict(
                self._last_obs, deterministic=True
            )
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(
                    scaled_action + action_noise(), -1.0, 1.0
                )
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
            return action, buffer_action
        return super()._sample_action(
            learning_starts,
            action_noise=action_noise,
            n_envs=n_envs,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "anchor_actor",
            "source_policy_warmup",
            "actor_learning_starts",
        ]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.anchor_actor is None or self.actor_anchor_coef <= 0:
            raise RuntimeError("enable_actor_anchor() must be called before learning")

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        sac_actor_losses: list[float] = []
        anchor_losses: list[float] = []
        anchor_action_rmses: list[float] = []
        critic_losses: list[float] = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size, env=self._vec_normalize_env
            )
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self.num_timesteps >= self.actor_learning_starts:
                q_values_pi = th.cat(
                    self.critic(replay_data.observations, actions_pi), dim=1
                )
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                sac_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                current_mean_action = self.actor(
                    replay_data.observations, deterministic=True
                )
                with th.no_grad():
                    source_mean_action = self.anchor_actor(
                        replay_data.observations, deterministic=True
                    )
                anchor_loss = F.mse_loss(
                    current_mean_action, source_mean_action
                )
                actor_loss = (
                    sac_actor_loss + self.actor_anchor_coef * anchor_loss
                )

                actor_losses.append(float(actor_loss.item()))
                sac_actor_losses.append(float(sac_actor_loss.item()))
                anchor_losses.append(float(anchor_loss.item()))
                anchor_action_rmses.append(float(th.sqrt(anchor_loss).item()))
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
            else:
                with th.no_grad():
                    current_mean_action = self.actor(
                        replay_data.observations, deterministic=True
                    )
                    source_mean_action = self.anchor_actor(
                        replay_data.observations, deterministic=True
                    )
                    anchor_loss = F.mse_loss(
                        current_mean_action, source_mean_action
                    )
                anchor_losses.append(float(anchor_loss.item()))
                anchor_action_rmses.append(float(th.sqrt(anchor_loss).item()))

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.batch_norm_stats, self.batch_norm_stats_target, 1.0
                )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record(
            "train/actor_update_active",
            float(self.num_timesteps >= self.actor_learning_starts),
        )
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record(
                "train/actor_sac_loss", np.mean(sac_actor_losses)
            )
        self.logger.record("train/actor_anchor_loss", np.mean(anchor_losses))
        self.logger.record(
            "train/actor_anchor_action_rmse", np.mean(anchor_action_rmses)
        )
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
