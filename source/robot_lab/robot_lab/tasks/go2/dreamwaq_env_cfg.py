# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""DreamWaQ observation contract for the RobotLab Go2 environment.

The environment dynamics intentionally inherit :class:`Go2EnvCfg` unchanged.
Only the observation layout and the default number of parallel environments
are specialized here.  The DreamWaQ runner owns the five-frame history so the
deployed policy receives the same 45-value frame in Isaac Lab and MuJoCo.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.go2.env_cfg import Go2EnvCfg, ObservationsCfg


@configclass
class DreamWaQObservationsCfg:
    """Observation groups consumed by DreamWaQ training.

    Shapes for a Go2 instance are:

    * ``policy``: 45 values from the current control step;
    * ``critic``: the original RobotLab privileged observation group;
    * ``velocity``: 3-value ground-truth body-frame linear velocity used only
      as the estimator target during training.

    The scales are inherited from the validated RobotLab Go2 task.  In
    particular, joint velocity uses 0.05 and body linear velocity uses 2.0.
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            # History belongs to the DreamWaQ runner, not the simulator-side
            # observation manager.  A length of one keeps the returned tensor
            # flat and preserves the original 45-value term ordering.
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        """Unmodified privileged critic observations from RobotLab Go2."""

    @configclass
    class VelocityCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=2.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    velocity: VelocityCfg = VelocityCfg()


@configclass
class Go2DreamWaQEnvCfg(Go2EnvCfg):
    """RobotLab Go2 dynamics with the DreamWaQ observation contract."""

    observations: DreamWaQObservationsCfg = DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
