# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ actor-critic and context-aided estimator network.

The deployed actor consumes one 45-dimensional proprioceptive observation and
five such observations of history.  CENet estimates body linear velocity and a
compact terrain/context latent from the history.  Privileged observations are
used by the critic and as CENet training targets only; they never enter the
deployed actor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import MLP


class CENet(nn.Module):
    """Beta-VAE context encoder used by DreamWaQ."""

    def __init__(
        self,
        observation_dim: int,
        history_length: int = 5,
        latent_dim: int = 16,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128, 48),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.estimated_velocity_dim = 3
        self.encoder = MLP(
            observation_dim * history_length,
            self.estimated_velocity_dim + 2 * latent_dim,
            encoder_hidden_dims,
            activation,
        )
        self.decoder = MLP(
            self.estimated_velocity_dim + latent_dim,
            observation_dim,
            decoder_hidden_dims,
            activation,
        )

    def encode(
        self, history: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.encoder(history)
        estimated_velocity = params[..., : self.estimated_velocity_dim]
        mu, log_var = params[..., self.estimated_velocity_dim :].chunk(2, dim=-1)
        # Bounding log variance prevents exp() overflow during early PPO updates.
        log_var = log_var.clamp(-20.0, 10.0)
        if sample:
            context = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        else:
            context = mu
        return estimated_velocity, context, mu, log_var

    def forward(
        self, history: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        estimated_velocity, context, mu, log_var = self.encode(history, sample=sample)
        reconstruction = self.decoder(torch.cat((estimated_velocity, context), dim=-1))
        return reconstruction, estimated_velocity, context, mu, log_var


class ActorCriticDreamWaQ(ActorCritic):
    """RSL-RL 3.3 compatible DreamWaQ policy.

    Environment observations remain ordinary ``TensorDict`` entries.  During
    rollout the module adds history, sampled CENet latent, and next-observation
    targets to the transition TensorDict.  PPO therefore uses the exact latent
    that generated an action instead of resampling it during an update.
    """

    HISTORY_KEY = "dreamwaq_history"
    LATENT_KEY = "dreamwaq_latent"
    NEXT_POLICY_OBS_KEY = "dreamwaq_next_policy_obs"
    # The real-robot and MuJoCo deployers keep one temporal block per sensor
    # feature: ang_vel[5], gravity[5], command[5], then the three joint-sized
    # terms.  This differs from simply flattening five complete frames.
    DEFAULT_FEATURE_DIMS = (3, 3, 3, 12, 12, 12)

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        history_length: int = 5,
        latent_dim: int = 16,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128, 48),
        history_feature_dims: tuple[int, ...] | list[int] = DEFAULT_FEATURE_DIMS,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        if actor_obs_normalization:
            raise ValueError("DreamWaQ uses fixed observation scales; actor RMS normalization must be disabled.")

        actor_hidden_dims = kwargs.get("actor_hidden_dims", (512, 256, 128))
        activation = kwargs.get("activation", "elu")
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=False,
            critic_obs_normalization=critic_obs_normalization,
            **kwargs,
        )

        self.single_observation_dim = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        self.num_actions = num_actions
        self.num_single_obs = self.single_observation_dim
        self.num_actor_obs = self.single_observation_dim * history_length
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.actor_input_dim = self.single_observation_dim + 3 + latent_dim
        self.history_feature_dims = tuple(history_feature_dims)
        if sum(self.history_feature_dims) != self.single_observation_dim:
            raise ValueError(
                "history_feature_dims must sum to the single policy observation dimension: "
                f"{sum(self.history_feature_dims)} != {self.single_observation_dim}"
            )
        self.cenet = CENet(
            observation_dim=self.single_observation_dim,
            history_length=history_length,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
        )
        # Replace the base actor (which was sized for a single observation).
        if self.state_dependent_std:
            self.actor = MLP(self.actor_input_dim, (2, num_actions), actor_hidden_dims, activation)
        else:
            self.actor = MLP(self.actor_input_dim, num_actions, actor_hidden_dims, activation)

        self._rollout_history: torch.Tensor | None = None
        self._rollout_reset_mask: torch.Tensor | None = None

    def _flatten_history_term_major(self, history: torch.Tensor) -> torch.Tensor:
        """Convert ``[batch, time, feature]`` to the deployment layout.

        Each sensor term contains all old-to-new frames before the next term:
        ``ang_vel[t-4:t], gravity[t-4:t], ..., action[t-4:t]``.
        """
        blocks = []
        start = 0
        for dim in self.history_feature_dims:
            blocks.append(history[..., start : start + dim].flatten(1))
            start += dim
        return torch.cat(blocks, dim=-1)

    def get_single_observation(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups["policy"]], dim=-1)

    def make_storage_observations(self, obs: TensorDict) -> TensorDict:
        """Return an observation schema containing DreamWaQ rollout fields."""
        result = obs.clone()
        batch = obs.batch_size[0]
        options = {"device": obs.device, "dtype": self.get_single_observation(obs).dtype}
        result[self.HISTORY_KEY] = torch.zeros(
            batch, self.history_length * self.single_observation_dim, **options
        )
        result[self.LATENT_KEY] = torch.zeros(batch, 3 + self.latent_dim, **options)
        result[self.NEXT_POLICY_OBS_KEY] = torch.zeros(batch, self.single_observation_dim, **options)
        return result

    def prepare_rollout_observations(self, obs: TensorDict) -> TensorDict:
        """Append the current frame to history and sample the rollout latent."""
        single_obs = self.get_single_observation(obs)
        batch = single_obs.shape[0]
        history_shape = (batch, self.history_length, self.single_observation_dim)
        if self._rollout_history is None or self._rollout_history.shape != history_shape:
            # Match the real-robot deployer: at startup, repeat the first
            # measurement across all history slots instead of injecting four
            # synthetic zero frames.
            self._rollout_history = single_obs.unsqueeze(1).repeat(1, self.history_length, 1)
            self._rollout_reset_mask = torch.zeros(batch, dtype=torch.bool, device=single_obs.device)
        else:
            reset_mask = self._rollout_reset_mask
            self._rollout_history = torch.roll(self._rollout_history, shifts=-1, dims=1)
            self._rollout_history[:, -1] = single_obs
            if reset_mask is not None and reset_mask.any():
                self._rollout_history[reset_mask] = single_obs[reset_mask].unsqueeze(1)
                self._rollout_reset_mask[reset_mask] = False
        flat_history = self._flatten_history_term_major(self._rollout_history)
        estimated_velocity, context, _, _ = self.cenet.encode(flat_history, sample=True)

        result = obs.clone()
        result[self.HISTORY_KEY] = flat_history.detach()
        result[self.LATENT_KEY] = torch.cat((estimated_velocity, context), dim=-1).detach()
        result[self.NEXT_POLICY_OBS_KEY] = torch.zeros_like(single_obs)
        return result

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self._rollout_history = None
            self._rollout_reset_mask = None
            return
        if self._rollout_history is not None and self._rollout_reset_mask is not None:
            done_mask = dones.reshape(-1).bool()
            self._rollout_reset_mask[done_mask] = True

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        single_obs = self.get_single_observation(obs)
        if self.LATENT_KEY not in obs.keys():
            raise KeyError(f"Actor input is missing '{self.LATENT_KEY}'.")
        return torch.cat((single_obs, obs[self.LATENT_KEY]), dim=-1)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        if self.HISTORY_KEY in obs.keys():
            history = obs[self.HISTORY_KEY]
            single_obs = self.get_single_observation(obs)
        else:
            prepared = self.prepare_rollout_observations(obs)
            history = prepared[self.HISTORY_KEY]
            single_obs = self.get_single_observation(prepared)
        estimated_velocity, context, _, _ = self.cenet.encode(history, sample=False)
        actor_obs = torch.cat((single_obs, estimated_velocity, context), dim=-1)
        return self.actor(actor_obs)

    def update_normalization(self, obs: TensorDict) -> None:
        # Actor values are already normalized by fixed, deployment-identical
        # scales in the environment observation terms.
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))
