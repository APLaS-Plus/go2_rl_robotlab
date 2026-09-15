# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with HIMLoco's joint estimator update.

Ported from the official HIMLoco implementation (https://github.com/MarkFzp/himloco,
arXiv 2405.10645, ``rsl_rl/algorithms/him_ppo.py``) onto the PPO base class of
this repository.  The official update loop interleaves, per mini-batch and in
this order: KL-based adaptive learning-rate scheduling, one estimator
(teacher-student contrastive) update that inherits the adapted learning rate,
then the standard PPO surrogate/value step.  That interleaving is preserved
here, which is why :meth:`update` re-implements the base loop instead of
calling ``super().update()``.

Two deliberate deviations from the official code (both commented inline):

* the estimator parameters are excluded from the PPO optimizer.  In the
  official code the PPO Adam also stepped the estimator with stale gradients
  left over from its own update, double-stepping it; the same exclusion is
  used by ``DreamWaQPPO`` for its CENet.
* transitions whose ``next_critic_obs`` is a post-reset observation are
  masked out of the estimator update instead of being replaced by the
  pre-reset privileged observation, which Isaac Lab does not expose.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_him import ActorCriticHIM


class HIMPPO(PPO):
    policy: ActorCriticHIM

    def __init__(self, policy: ActorCriticHIM, storage, **kwargs) -> None:
        super().__init__(policy, storage, **kwargs)
        if self.is_multi_gpu:
            raise NotImplementedError("HIM estimator training currently supports one GPU only.")
        if self.rnd or self.symmetry:
            raise NotImplementedError("HIMPPO does not support RND or symmetry auxiliary objectives.")
        # Estimator outputs are detached inside ActorCriticHIM._update_distribution,
        # so PPO gradients never reach it; train it exclusively with its own
        # optimizer (held inside the estimator module, as in HIMLoco).
        self.ppo_parameters = [p for name, p in self.policy.named_parameters() if not name.startswith("estimator.")]
        self.optimizer = optim.Adam(self.ppo_parameters, lr=self.learning_rate)

    def act(self, obs: TensorDict) -> torch.Tensor:
        rollout_obs = self.policy.prepare_rollout_observations(obs)
        self.transition.actions = self.policy.act(rollout_obs).detach()
        self.transition.values = self.policy.evaluate(rollout_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations (including the module-side history) before env.step()
        self.transition.observations = rollout_obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Attach the privileged O_{t+1} to the already captured O_t transition
        # before storage copies it.  For terminated environments Isaac Lab
        # reports the post-reset observation, so flag the transition as an
        # invalid estimator target (HIMLoco's env substituted the pre-reset
        # privileged observation instead).
        self.transition.observations[self.policy.NEXT_CRITIC_OBS_KEY] = self.policy.get_critic_obs(obs).detach()
        self.transition.observations[self.policy.VALID_TRANSITION_KEY] = (
            (~dones.reshape(-1, 1)).to(self.transition.observations[self.policy.NEXT_CRITIC_OBS_KEY].dtype)
        )
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_entropy = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _,
            _,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute the action distribution for the current parameters.
            # The estimator outputs are detached inside the policy, matching
            # HIMLoco's act() during updates.
            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL divergence and adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Estimator update at the official position: after the learning
            # rate adaptation (which it inherits) and before the PPO step.
            estimation_loss, swap_loss = self.policy.estimator.update(
                obs_batch[self.policy.HISTORY_KEY],
                obs_batch[self.policy.NEXT_CRITIC_OBS_KEY],
                valid_mask=obs_batch[self.policy.VALID_TRANSITION_KEY],
                lr=self.learning_rate,
            )

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step (clipped over PPO parameters only; the estimator
            # already clipped its own gradients inside its update)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        mean_entropy /= num_updates
        self.storage.clear()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimation": mean_estimation_loss,
            "swap": mean_swap_loss,
        }
