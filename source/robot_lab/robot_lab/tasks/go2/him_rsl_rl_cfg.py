# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL configuration for the RobotLab Go2 HIM (HIMLoco) task.

Hyperparameters follow the official HIMLoco defaults
(https://github.com/MarkFzp/himloco, arXiv 2405.10645,
``legged_gym/envs/base/legged_robot_config.py::LeggedRobotCfgPPO``): a
45-value proprioceptive frame with six frames of history, a 16-dim height
latent plus a 3-dim velocity estimate from the internal estimator, and PPO
with an adaptive KL schedule.  Only ``num_steps_per_env``/``max_iterations``
follow the RobotLab repo convention (HIMLoco used 100/200000).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlHIMActorCriticCfg(RslRlPpoActorCriticCfg):
    """HIM actor, asymmetric critic, and internal height-estimator settings."""

    class_name = "ActorCriticHIM"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    # Six 45-value frames are maintained by ActorCriticHIM.  The estimator
    # predicts three body-frame velocity values plus a 16-dim terrain latent
    # from the history, matching the original HIMLoco deployment contract.
    history_length = 6
    latent_dim = 16
    enc_hidden_dims = [128, 64, 16]
    tar_hidden_dims = [128, 64]

    # Estimator (kept inside the module, trained by its own Adam)
    estimator_learning_rate = 1.0e-3
    estimator_max_grad_norm = 10.0
    num_prototype = 32
    temperature = 3.0


@configclass
class RslRlHIMAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO and HIM estimator-loss settings (official HIMLoco defaults)."""

    class_name = "HIMPPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class HIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default 4096-environment HIM experiment."""

    class_name = "OnPolicyRunnerHIM"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "go2_him"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlHIMActorCriticCfg()
    algorithm = RslRlHIMAlgorithmCfg()
