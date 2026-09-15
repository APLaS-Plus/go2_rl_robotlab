# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HIM (Hybrid Internal Model) actor-critic with an internal height estimator.

Ported from the official HIMLoco implementation (https://github.com/MarkFzp/himloco,
arXiv 2405.10645, files ``rsl_rl/modules/him_actor_critic.py`` and
``rsl_rl/modules/him_estimator.py``) onto the TensorDict-based RSL-RL API of this
repository (same adaptation pattern as ``actor_critic_dreamwaq.py``).

Obs-group contract (established by ``robot_lab/tasks/go2/him_env_cfg.py``):

* ``policy``: one 45-value proprioceptive frame per control step (Go2:
  commands 3, ang_vel 3, gravity 3, joint_pos 12, joint_vel 12, actions 12).
  The six-frame history of HIMLoco is maintained inside this module during
  rollouts, not by the simulator-side observation manager.
* ``critic``: HIMLoco privileged layout
  ``[one-step proprio (45) | base_lin_vel (3) | height_scan (187)]``.  The
  ordering is load-bearing: :class:`HIMEstimator` takes its velocity target
  from ``critic[:, 45:48]`` and its teacher input from ``critic[:, 3:48]``.
  (The official code carries an extra 3-dim external-wrench block that the
  RobotLab environment does not provide; it is dropped here.)

Adaptations relative to the official code are commented inline.  The core HIM
structure is preserved: the actor consumes only proprioception plus the
estimator's detached velocity/latent outputs, the estimator is trained by a
contrastive (Sinkhorn/swapped-prediction) objective against a target network
that sees privileged data, and the critic sees the full privileged group.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import MLP
from rsl_rl.utils import resolve_nn_activation


class HIMEstimator(nn.Module):
    """Velocity and terrain-latent estimator with teacher-student contrastive training.

    Faithful port of HIMLoco's ``him_estimator.py``:

    * ``encoder`` (student): maps the flattened proprioceptive history to a
      3-dim velocity estimate plus a ``num_latent``-dim terrain embedding;
    * ``target`` (teacher): maps the *privileged* next one-step observation
      (true velocity included) to the terrain-embedding space;
    * ``proto``: learnable prototypes used for SwAV-style swapped prediction
      with Sinkhorn-Knopp normalized assignments.

    The module owns its optimizer, as in the official implementation, so that
    PPO can coordinate its learning rate while keeping its gradients separate.
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        enc_hidden_dims: tuple[int, ...] | list[int] = (128, 64, 16),
        tar_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 10.0,
        num_prototype: int = 32,
        temperature: float = 3.0,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "HIMEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # Encoder (student): full proprioceptive history -> velocity + latent
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for layer in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[layer]), activation]
            enc_input_dim = enc_hidden_dims[layer]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target (teacher): privileged one-step observation -> latent
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for layer in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[layer]), activation]
            tar_input_dim = tar_hidden_dims[layer]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototypes for the contrastive assignment
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer (kept inside the module, as in the official implementation)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def encode(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        lr: float | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """One estimator step: velocity regression + swapped contrastive loss.

        Args:
            obs_history: Flattened proprioceptive history at time t.
            next_critic_obs: Privileged observation at time t+1 laid out as
                ``[one-step proprio | base_lin_vel | height_scan]``.
            lr: Optional learning-rate override (HIMLoco's PPO passes its
                adaptive learning rate here).
            valid_mask: Optional ``[batch, 1]`` flag marking transitions whose
                ``next_critic_obs`` is a valid (non-terminal) target.

        Returns:
            Tuple of the estimation (velocity MSE) and swap losses.
        """
        # Isaac Lab returns the post-reset observation for terminated
        # environments, while HIMLoco's env replaced it with the pre-reset
        # privileged observation.  Such transitions are skipped instead.
        if valid_mask is not None:
            keep = valid_mask.reshape(-1) > 0.5
            if not bool(keep.all()):
                obs_history = obs_history[keep]
                next_critic_obs = next_critic_obs[keep]
                if obs_history.shape[0] == 0:
                    return 0.0, 0.0

        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        vel = next_critic_obs[:, self.num_one_step_obs : self.num_one_step_obs + 3].detach()
        next_obs = next_critic_obs.detach()[:, 3 : self.num_one_step_obs + 3]

        z_s = self.encoder(obs_history)
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for _ in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


class ActorCriticHIM(ActorCritic):
    """RSL-RL compatible HIMLoco policy.

    Mirrors the official ``HIMActorCritic``: the actor input is the latest
    proprioceptive frame concatenated with the estimator's 3-dim velocity
    estimate and 16-dim height latent, while the critic is a plain MLP over
    the full privileged group.  During PPO updates the estimator outputs are
    recomputed gradient-free (exactly as in HIMLoco's ``act``), so the
    estimator is trained only by its internal contrastive objective.
    """

    HISTORY_KEY = "him_history"
    NEXT_CRITIC_OBS_KEY = "him_next_critic_obs"
    VALID_TRANSITION_KEY = "him_valid_transition"

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        history_length: int = 6,
        latent_dim: int = 16,
        enc_hidden_dims: tuple[int, ...] | list[int] = (128, 64, 16),
        tar_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        estimator_learning_rate: float = 1.0e-3,
        estimator_max_grad_norm: float = 10.0,
        num_prototype: int = 32,
        temperature: float = 3.0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        if actor_obs_normalization:
            raise ValueError("HIM uses fixed observation scales; actor RMS normalization must be disabled.")
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
        self.history_length = history_length
        self.latent_dim = latent_dim
        # Official HIMActorCritic: mlp_input_dim_a = num_one_step_obs + 3 + 16
        self.actor_input_dim = self.single_observation_dim + 3 + latent_dim
        if enc_hidden_dims[-1] != latent_dim:
            raise ValueError(
                "enc_hidden_dims[-1] must equal latent_dim: "
                f"{enc_hidden_dims[-1]} != {latent_dim}"
            )

        # Estimator (official HIMActorCritic holds exactly one HIMEstimator)
        self.estimator = HIMEstimator(
            temporal_steps=history_length,
            num_one_step_obs=self.single_observation_dim,
            enc_hidden_dims=list(enc_hidden_dims),
            tar_hidden_dims=list(tar_hidden_dims),
            activation=activation,
            learning_rate=estimator_learning_rate,
            max_grad_norm=estimator_max_grad_norm,
            num_prototype=num_prototype,
            temperature=temperature,
        )

        # Replace the base actor, which was sized for a single frame without
        # the estimator outputs.
        if self.state_dependent_std:
            self.actor = MLP(self.actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(self.actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Estimator encoder: {self.estimator.encoder}")

        self._rollout_history: torch.Tensor | None = None
        self._rollout_reset_mask: torch.Tensor | None = None

    def get_single_observation(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups["policy"]], dim=-1)

    def prepare_rollout_observations(self, obs: TensorDict) -> TensorDict:
        """Append the current frame to the module-side history and attach HIM fields."""
        single_obs = self.get_single_observation(obs)
        batch = single_obs.shape[0]
        history_shape = (batch, self.history_length, self.single_observation_dim)
        if self._rollout_history is None or self._rollout_history.shape != history_shape:
            # At startup, repeat the first measurement across all history
            # slots instead of injecting synthetic zero frames (deployment
            # behaviour, matching the DreamWaQ runner convention).
            self._rollout_history = single_obs.unsqueeze(1).repeat(1, self.history_length, 1)
            self._rollout_reset_mask = torch.zeros(batch, dtype=torch.bool, device=single_obs.device)
        else:
            self._rollout_history = torch.roll(self._rollout_history, shifts=-1, dims=1)
            self._rollout_history[:, -1] = single_obs
            reset_mask = self._rollout_reset_mask
            if reset_mask is not None and reset_mask.any():
                self._rollout_history[reset_mask] = single_obs[reset_mask].unsqueeze(1)
                self._rollout_reset_mask[reset_mask] = False

        # Frame-major chronological order (oldest -> newest).  HIMLoco's env
        # buffer is newest-first; the ordering is irrelevant for the encoder
        # MLP, and the newest frame is ``history[..., -45:]`` here versus
        # ``history[:, :45]`` in the official code.
        flat_history = self._rollout_history.flatten(1)

        result = obs.clone()
        options = {"device": obs.device, "dtype": single_obs.dtype}
        result[self.HISTORY_KEY] = flat_history.detach()
        result[self.NEXT_CRITIC_OBS_KEY] = torch.zeros(batch, self.get_critic_obs(obs).shape[-1], **options)
        result[self.VALID_TRANSITION_KEY] = torch.ones(batch, 1, **options)
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
        if self.HISTORY_KEY not in obs.keys():
            raise KeyError(f"Actor input is missing '{self.HISTORY_KEY}'.")
        return obs[self.HISTORY_KEY]

    def _update_distribution(self, history: torch.Tensor) -> None:
        # Official update_distribution(): the estimator runs gradient-free, so
        # no PPO gradient ever flows into it.
        with torch.no_grad():
            vel, latent = self.estimator(history)
        actor_input = torch.cat((history[..., -self.single_observation_dim :], vel, latent), dim=-1)
        if self.state_dependent_std:
            mean_and_std = self.actor(actor_input)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(actor_input)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        if self.HISTORY_KEY in obs.keys():
            history = obs[self.HISTORY_KEY]
        else:
            prepared = self.prepare_rollout_observations(obs)
            history = prepared[self.HISTORY_KEY]
        with torch.no_grad():
            vel, latent = self.estimator(history)
        actor_input = torch.cat((history[..., -self.single_observation_dim :], vel, latent), dim=-1)
        return self.actor(actor_input)

    def update_normalization(self, obs: TensorDict) -> None:
        # Actor values are already normalized by fixed, deployment-identical
        # scales in the environment observation terms.
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))
