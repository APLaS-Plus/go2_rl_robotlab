# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL configuration for the RobotLab Go2 DreamWaQ task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlDreamWaQActorCriticCfg(RslRlPpoActorCriticCfg):
    """DreamWaQ actor, asymmetric critic, and context encoder settings."""

    class_name = "ActorCriticDreamWaQ"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    # Five 45-value frames are maintained by OnPolicyRunnerDreamWaQ.  The
    # encoder predicts three body-frame velocity values plus a 16-value
    # stochastic context, matching the original DreamWaQ deployment contract.
    history_length = 5
    latent_dim = 16
    encoder_hidden_dims = [128, 64]
    decoder_hidden_dims = [64, 128, 48]


@configclass
class RslRlDreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO and DreamWaQ auxiliary-loss settings."""

    class_name = "DreamWaQPPO"
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

    velocity_loss_coef = 1.0
    reconstruction_loss_coef = 1.0
    kl_loss_coef = 1.0
    cenet_learning_rate = 1.0e-2
    velocity_bootstrap_iterations = 1000


@configclass
class DreamWaQRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default 4096-environment, 5000-iteration DreamWaQ experiment."""

    class_name = "OnPolicyRunnerDreamWaQ"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "go2_dreamwaq"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlDreamWaQActorCriticCfg()
    algorithm = RslRlDreamWaQAlgorithmCfg()
