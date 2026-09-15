# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""HIMLoco observation contract for the RobotLab Go2 environment.

Source: HIMLoco / Hybrid Internal Model (https://github.com/MarkFzp/himloco,
arXiv 2405.10645), ``legged_gym/envs/base/legged_robot_config.py`` and
``legged_gym/envs/base/legged_robot.py::compute_observations``.

The environment dynamics intentionally inherit :class:`Go2EnvCfg` unchanged
(same rewards, curriculum, domain randomization, commands and the 187-ray
``height_scanner`` scene as the validated RobotLab baseline).  Only the
observation layout and the default number of parallel environments are
specialized:

* ``policy``: 45 proprioceptive values per control step in HIMLoco order
  (commands, ang vel, gravity, joint pos, joint vel, actions).  No height
  measurements ever reach the policy.  The six-frame history is maintained by
  ``ActorCriticHIM``, so the manager-side history length is one.
* ``critic``: the HIMLoco privileged layout
  ``[one-step proprio (45) | base_lin_vel (3) | height_scan (187)]``.  This
  ordering is load-bearing: :class:`HIMEstimator` slices its velocity target
  from ``critic[:, 45:48]`` and its teacher input from ``critic[:, 3:48]``.
  The official 3-dim external-wrench block is omitted because the RobotLab
  environment has no such sensor.

Term scales/noises follow the validated RobotLab Go2 task (as in the
DreamWaQ port), not HIMLoco's legged_gym scales: commands enter unscaled
(scale 1.0) and the height scan keeps ``clip=(-1, 1), scale=2.5``.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.go2.env_cfg import JOINT_NAMES, Go2EnvCfg


@configclass
class HIMObservationsCfg:
    """Observation groups consumed by HIM training.

    Shapes for a Go2 instance are:

    * ``policy``: 45 values from the current control step (proprioception
      only -- the deployed actor never sees height measurements);
    * ``critic``: 235 privileged values laid out as
      ``[proprio 45 | base_lin_vel 3 | height_scan 187]``.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive one-step observations in HIMLoco term order."""

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            # History belongs to ActorCriticHIM, not the simulator-side
            # observation manager.  A length of one keeps the returned tensor
            # flat and preserves the 45-value term ordering above.
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations in HIMLoco layout (ordering is load-bearing).

        ``[0:45)`` mirrors the policy frame above, ``[45:48)`` is the true
        base linear velocity (estimator regression target), and ``[48:235)``
        is the 187-ray height scan (teacher terrain evidence).
        """

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=2.5,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2HIMEnvCfg(Go2EnvCfg):
    """RobotLab Go2 dynamics with the HIMLoco observation contract."""

    observations: HIMObservationsCfg = HIMObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
