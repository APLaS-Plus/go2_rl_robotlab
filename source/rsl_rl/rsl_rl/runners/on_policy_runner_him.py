# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL runner adapter for HIMLoco's rollout-only TensorDict fields.

Ported from the official HIMLoco implementation (https://github.com/MarkFzp/himloco,
arXiv 2405.10645, ``rsl_rl/runners/him_on_policy_runner.py``).  HIMLoco trains
in a single stage: the teacher-student contrastive learning lives inside the
estimator (the target network sees privileged data, the history encoder does
not) and is updated jointly with PPO, so the standard runner loop is reused.
This subclass only widens the rollout-storage schema with HIM's private
transition fields and extends save/load with the estimator optimizer state,
following the ``OnPolicyRunnerDreamWaQ`` convention.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class OnPolicyRunnerHIM(OnPolicyRunner):
    """Use the standard runner loop with an expanded rollout-storage schema."""

    def _construct_algorithm(self, obs: TensorDict):
        # Construct policy/algorithm first using a temporary schema.  The policy
        # sizes all public observation groups dynamically, while storage also
        # needs the private HIM transition fields.
        from rsl_rl.modules import ActorCriticHIM

        policy_cfg = self.policy_cfg
        history_length = policy_cfg.get("history_length", 6)
        single_dim = sum(obs[name].shape[-1] for name in self.cfg["obs_groups"]["policy"])
        critic_dim = sum(obs[name].shape[-1] for name in self.cfg["obs_groups"]["critic"])
        expanded_obs = obs.clone()
        options = {"device": obs.device, "dtype": next(iter(obs.values())).dtype}
        expanded_obs[ActorCriticHIM.HISTORY_KEY] = torch.zeros(
            obs.batch_size[0], history_length * single_dim, **options
        )
        expanded_obs[ActorCriticHIM.NEXT_CRITIC_OBS_KEY] = torch.zeros(obs.batch_size[0], critic_dim, **options)
        expanded_obs[ActorCriticHIM.VALID_TRANSITION_KEY] = torch.zeros(obs.batch_size[0], 1, **options)
        return super()._construct_algorithm(expanded_obs)

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            # The estimator owns its optimizer (official HIMLoco layout)
            "estimator_optimizer_state_dict": self.alg.policy.estimator.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "estimator_optimizer_state_dict" in loaded_dict:
                self.alg.policy.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
