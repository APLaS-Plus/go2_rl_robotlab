# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Environment configuration for the RobotLab Go2 vanilla Cross Teacher-Student (CTS) task.

This task ports the ``go2_cts`` baseline of the reference repository go2_rl_gym
(https://arxiv.org/abs/2405.10830, the ``CTS vanilla`` row of the RoboGauge
benchmark table in the README) to Isaac Lab / RobotLab.

The environment dynamics intentionally inherit :class:`Go2EnvCfg` unchanged,
exactly like the MoE-CTS task (``RobotLab-Go2-v0``).  The only specialization
is the observation history length of the ``policy`` group: the vanilla CTS
baseline stacks 5 proprioceptive frames (go2_rl_gym
``LeggedRobotCfgCTS.history_length = 5``), while the MoE-CTS task of this
repository stacks 10.  The observation layout itself is shared:

* ``policy``: 5 x 45 flattened history, input of the student encoder
  (history stacking and reset-on-done are handled by the observation manager);
* ``critic``: the original RobotLab privileged observation group, input of the
  teacher encoder and the critic;
* ``single_obs``: 45-value current frame, concatenated with the latent for the
  actor (also the input contract of the deployed jit/onnx policy).
"""

from isaaclab.utils import configclass

from robot_lab.tasks.go2.env_cfg import Go2EnvCfg, ObservationsCfg


@configclass
class CTSObservationsCfg(ObservationsCfg):
    """Observation groups consumed by vanilla CTS training."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            # Five 45-value frames are stacked by the observation manager,
            # matching the vanilla CTS baseline (go2_rl_gym
            # LeggedRobotCfgCTS.history_length = 5). The MoE-CTS task uses 10.
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2CTSEnvCfg(Go2EnvCfg):
    """RobotLab Go2 dynamics with the vanilla CTS observation contract."""

    observations: CTSObservationsCfg = CTSObservationsCfg()
