# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with the DreamWaQ CENet auxiliary objective."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_dreamwaq import ActorCriticDreamWaQ


class DreamWaQPPO(PPO):
    policy: ActorCriticDreamWaQ

    def __init__(
        self,
        policy: ActorCriticDreamWaQ,
        storage,
        velocity_loss_coef: float = 1.0,
        reconstruction_loss_coef: float = 1.0,
        kl_loss_coef: float = 1.0,
        cenet_learning_rate: float = 1.0e-3,
        cenet_num_learning_epochs: int | None = None,
        cenet_num_mini_batches: int | None = None,
        beta: float = 1.0,
        beta_limit: float = 4.0,
        beta_growth: float = 1.01,
        velocity_bootstrap_iterations: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(policy, storage, **kwargs)
        if self.is_multi_gpu:
            raise NotImplementedError("DreamWaQ CENet training currently supports one GPU only.")
        self.velocity_loss_coef = velocity_loss_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.cenet_num_learning_epochs = cenet_num_learning_epochs or self.num_learning_epochs
        self.cenet_num_mini_batches = cenet_num_mini_batches or self.num_mini_batches
        self.beta = beta
        self.beta_limit = beta_limit
        self.beta_growth = beta_growth
        self.velocity_bootstrap_iterations = max(int(velocity_bootstrap_iterations), 0)
        self.current_update = 0

        # PPO treats the sampled CENet latent as rollout data.  CENet receives
        # gradients only from its supervised Beta-VAE objective.
        ppo_parameters = [p for name, p in policy.named_parameters() if not name.startswith("cenet.")]
        self.optimizer = optim.Adam(ppo_parameters, lr=self.learning_rate)
        self.cenet_optimizer = optim.Adam(policy.cenet.parameters(), lr=cenet_learning_rate)

    def act(self, obs: TensorDict) -> torch.Tensor:
        rollout_obs = self.policy.prepare_rollout_observations(obs)
        # DreamWaQ's adaptive bootstrap lets the actor learn locomotion from
        # ground-truth velocity before progressively handing control to CENet.
        # Both signals use the same fixed scale from the environment contract.
        if self.velocity_bootstrap_iterations == 0:
            estimate_probability = 1.0
        else:
            estimate_probability = min(self.current_update / self.velocity_bootstrap_iterations, 1.0)
        if estimate_probability < 1.0:
            latent = rollout_obs[self.policy.LATENT_KEY]
            if estimate_probability <= 0.0:
                velocity_input = obs["velocity"]
            else:
                use_estimate = torch.rand(latent.shape[0], 1, device=latent.device) < estimate_probability
                velocity_input = torch.where(use_estimate, latent[:, :3], obs["velocity"])
            rollout_obs[self.policy.LATENT_KEY] = torch.cat((velocity_input, latent[:, 3:]), dim=-1)
        self.transition.actions = self.policy.act(rollout_obs).detach()
        self.transition.values = self.policy.evaluate(rollout_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = rollout_obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Attach O_(t+1) to the already captured O_t transition before storage
        # copies it.  This is the CENet reconstruction target.
        self.transition.observations[self.policy.NEXT_POLICY_OBS_KEY] = (
            self.policy.get_single_observation(obs).detach()
        )
        super().process_env_step(obs, rewards, dones, extras)

    def _update_cenet(self) -> dict[str, float]:
        observations = self.storage.observations.flatten(0, 1)
        valid_reconstruction = ~self.storage.dones.flatten(0, 1).reshape(-1).bool()
        batch_size = observations.batch_size[0]
        mini_batch_size = batch_size // self.cenet_num_mini_batches
        totals = {"cenet_velocity": 0.0, "cenet_reconstruction": 0.0, "cenet_kl": 0.0, "cenet_total": 0.0}
        updates = 0

        for _ in range(self.cenet_num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for mini_batch in range(self.cenet_num_mini_batches):
                idx = indices[mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size]
                # Isaac Lab returns a reset observation for terminal environments.
                # Exclude those transitions so CENet never reconstructs across episodes.
                idx = idx[valid_reconstruction[idx]]
                if idx.numel() == 0:
                    continue
                batch = observations[idx]
                reconstruction, estimated_velocity, _, mu, log_var = self.policy.cenet(
                    batch[self.policy.HISTORY_KEY], sample=True
                )
                velocity_target = batch["velocity"]
                reconstruction_target = batch[self.policy.NEXT_POLICY_OBS_KEY]
                velocity_loss = nn.functional.mse_loss(estimated_velocity, velocity_target)
                reconstruction_loss = nn.functional.mse_loss(reconstruction, reconstruction_target)
                kl_loss = -0.5 * (1.0 + log_var - mu.square() - log_var.exp()).sum(dim=-1).mean()
                total_loss = (
                    self.velocity_loss_coef * velocity_loss
                    + self.reconstruction_loss_coef * reconstruction_loss
                    + self.kl_loss_coef * self.beta * kl_loss
                )

                self.cenet_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.cenet.parameters(), self.max_grad_norm)
                self.cenet_optimizer.step()
                self.cenet_optimizer.zero_grad(set_to_none=True)

                totals["cenet_velocity"] += velocity_loss.item()
                totals["cenet_reconstruction"] += reconstruction_loss.item()
                totals["cenet_kl"] += kl_loss.item()
                totals["cenet_total"] += total_loss.item()
                updates += 1

        self.beta = min(self.beta * self.beta_growth, self.beta_limit)
        return {name: value / max(updates, 1) for name, value in totals.items()}

    def update(self) -> dict[str, float]:
        estimate_probability = (
            1.0
            if self.velocity_bootstrap_iterations == 0
            else min(self.current_update / self.velocity_bootstrap_iterations, 1.0)
        )
        cenet_losses = self._update_cenet()
        ppo_losses = super().update()
        self.current_update += 1
        return {
            **ppo_losses,
            **cenet_losses,
            "velocity_estimate_probability": estimate_probability,
        }
