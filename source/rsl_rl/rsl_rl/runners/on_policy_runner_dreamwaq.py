# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL runner adapter for DreamWaQ's rollout-only TensorDict fields."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class OnPolicyRunnerDreamWaQ(OnPolicyRunner):
    """Use the standard runner loop with an expanded rollout-storage schema."""

    def _construct_algorithm(self, obs: TensorDict):
        # Construct policy/algorithm first using a temporary schema.  The policy
        # sizes all public observation groups dynamically, while storage also
        # needs the private DreamWaQ transition fields.
        from rsl_rl.modules import ActorCriticDreamWaQ

        policy_cfg = self.policy_cfg
        history_length = policy_cfg.get("history_length", 5)
        latent_dim = policy_cfg.get("latent_dim", 16)
        single_dim = sum(obs[name].shape[-1] for name in self.cfg["obs_groups"]["policy"])
        expanded_obs = obs.clone()
        options = {"device": obs.device, "dtype": next(iter(obs.values())).dtype}
        expanded_obs[ActorCriticDreamWaQ.HISTORY_KEY] = torch.zeros(
            obs.batch_size[0], history_length * single_dim, **options
        )
        expanded_obs[ActorCriticDreamWaQ.LATENT_KEY] = torch.zeros(
            obs.batch_size[0], 3 + latent_dim, **options
        )
        expanded_obs[ActorCriticDreamWaQ.NEXT_POLICY_OBS_KEY] = torch.zeros(
            obs.batch_size[0], single_dim, **options
        )
        return super()._construct_algorithm(expanded_obs)

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "cenet_optimizer_state_dict": self.alg.cenet_optimizer.state_dict(),
            "cenet_beta": self.alg.beta,
            "dreamwaq_update": self.alg.current_update,
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
            if "cenet_optimizer_state_dict" in loaded_dict:
                self.alg.cenet_optimizer.load_state_dict(loaded_dict["cenet_optimizer_state_dict"])
        self.alg.beta = loaded_dict.get("cenet_beta", self.alg.beta)
        self.alg.current_update = loaded_dict.get("dreamwaq_update", loaded_dict.get("iter", 0))
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
