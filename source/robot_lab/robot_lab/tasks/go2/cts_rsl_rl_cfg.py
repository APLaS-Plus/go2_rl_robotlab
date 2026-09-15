# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL configuration for the RobotLab Go2 vanilla Cross Teacher-Student (CTS) task.

Hyperparameters follow the ``go2_cts`` baseline of the reference repository
go2_rl_gym (https://arxiv.org/abs/2405.10830, ``legged_gym/envs/go2/
go2_config.py::GO2CfgCTS`` on top of ``LeggedRobotCfgCTS``): a single plain
MLP student encoder over the 5-frame stacked proprioceptive history (no
mixture-of-experts, no load-balance loss), an l2-normalized 32-dim latent,
and the concurrent teacher-student PPO update of :class:`CTS`.

Differences to the MoE-CTS runner configuration
(``rsl_rl_cfg.py::MoECTSRunnerCfg``):

* ``policy``: ``ActorCriticCTS`` instead of ``ActorCriticMoECTS`` --
  ``expert_num`` is dropped and ``student_encoder_hidden_dims`` is the vanilla
  ``[512, 256]`` instead of the MoE ``[512, 256, 256]``;
* ``algorithm``: ``CTS`` instead of ``MoECTS`` -- ``load_balance_coef`` is
  dropped, everything else (including ``teacher_env_ratio = 0.75``) is kept;
* ``max_iterations``: 150000 as in go2_rl_gym (MoE-CTS of this repo uses
  300000).

The ``policy``/``algorithm`` ``class_name`` strings are resolved by
``OnPolicyRunnerCTS._construct_algorithm`` via ``eval()`` inside the runner
module, which only sees the classes imported there (``ActorCriticMoECTS`` /
``MoECTS``).  Since the runner file must stay untouched, the new classes are
referenced through the repository's own ``resolve_callable`` utility, which is
imported by the runner and resolves qualified ``module:Class`` names through
``importlib``.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlCtsActorCriticCfg(RslRlPpoActorCriticCfg):
    """Vanilla CTS actor, asymmetric critic, and teacher/student encoder settings."""

    # Resolved by OnPolicyRunnerCTS via eval(); the runner module does not
    # import ActorCriticCTS itself, so resolve it through `resolve_callable`.
    class_name = "resolve_callable('rsl_rl.modules:ActorCriticCTS')"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    teacher_encoder_hidden_dims = [512, 256]
    student_encoder_hidden_dims = [512, 256]  # vanilla CTS dims (MoE-CTS uses [512, 256, 256])
    activation = "elu"
    latent_dim = 32
    norm_type = "l2norm"  # normalization type for encoders: l2norm, simnorm


@configclass
class RslRlCtsAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Concurrent teacher-student PPO settings (go2_rl_gym GO2CfgCTS defaults)."""

    # Resolved by OnPolicyRunnerCTS via eval(); see the note in RslRlCtsActorCriticCfg.
    class_name = "resolve_callable('rsl_rl.algorithms:CTS')"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    student_encoder_learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    betas = (0.9, 0.999)
    weight_decay = 0.0
    desired_kl = 0.01
    max_grad_norm = 1.0
    teacher_env_ratio = 0.75  # percentage of envs assigned to teacher


@configclass
class CTSRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default 16384-environment, 150000-iteration vanilla CTS experiment."""

    class_name = "OnPolicyRunnerCTS"  # shared with the MoE-CTS task
    num_steps_per_env = 24
    max_iterations = 150000
    save_interval = 500
    experiment_name = "go2_cts"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlCtsActorCriticCfg()
    algorithm = RslRlCtsAlgorithmCfg()
