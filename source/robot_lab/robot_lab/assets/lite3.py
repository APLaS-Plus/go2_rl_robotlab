# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Articulation configuration for the DEEP Robotics Lite3 quadruped.

物理参数移植来源：money12532/Lite3_RL_Project
    https://github.com/money12532/Lite3_RL_Project
    - PD 增益 / 力矩限幅 / 速度限幅：rl_training/source/rl_training/rl_training/
      assets/deeprobotics.py 的 ``DEEPROBOTICS_LITE3_CFG``
    - 默认站姿与初始高度：同上 ``init_state``，并与部署端
      Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.cpp 的
      ``dof_pos_default`` 对齐
    - 关节限位：Lite3 URDF/MJCF（deep_robotics_model/Lite3）

本文件仅移植物理参数（关节限位、力矩/速度限幅、PD 增益、默认站姿、初始高度、
关节顺序），不复用其奖励函数设计 —— 奖励沿用 RobotLab 既有配置。
"""

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

# 物理参数来源：money12532/Lite3_RL_Project -> rl_training/.../assets/deeprobotics.py
LITE3_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/lite3/Lite3_usd/Lite3.usd"

# 关节顺序与 Lite3 参考仓完全一致（FL,FR,HL,HR x HipX,HipY,Knee），
# 也是部署端 robot2policy_idx 使用的顺序。
LITE3_JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]

LITE3_BASE_LINK_NAME = "TORSO"
LITE3_FOOT_LINK_NAME = ".*_FOOT"

# 物理参数来源：money12532/Lite3_RL_Project
LITE3_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        # 分层 USD：Lite3.usd 引用 configuration/ 下的 base/physics/robot/sensor 四个子层，
        # 由 deep_robotics_model 提供，整机质量 11.9376 kg。
        usd_path=LITE3_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=None,
        articulation_props=None,
    ),
    # 初始高度 0.375 m 与默认站姿（HipX 0.0 / HipY -0.65 / Knee 1.3）
    # 来自 money12532/Lite3_RL_Project -> deeprobotics.py 的 init_state，
    # 与部署端 lite3_test_policy_runner_onnx.cpp 的 dof_pos_default 一致。
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.375),
        joint_pos={
            ".*HipX_joint": 0.0,
            ".*HipY_joint": -0.65,
            ".*Knee_joint": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    # soft_joint_pos_limit_factor = 0.99，与参考仓一致（URDF/MJCF 已给真实限位）。
    soft_joint_pos_limit_factor=0.99,
    # 执行器拆成 Hip / Knee 两组，PD 增益与限幅直接取自参考仓 deeprobotics.py：
    #   Hip (HipX, HipY): effort 24 N*m, velocity 26.2 rad/s, stiffness 30, damping 1
    #   Knee           : effort 36 N*m, velocity 17.3 rad/s, stiffness 30, damping 1
    # 关节限位来自 Lite3 URDF/MJCF：
    #   HipX  -0.523 ~  0.523
    #   HipY  -2.670 ~  0.314
    #   Knee   0.524 ~  2.792
    # 参考仓使用 DelayedPDActuatorCfg(min_delay=0, max_delay=1)，
    # 与官方 DeepRoboticsLab/rl_training 相同，此处保留该延迟配置。
    actuators={
        "Hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip[X,Y]_joint"],
            effort_limit=24.0,
            velocity_limit=26.2,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        "Knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Knee_joint"],
            effort_limit=36.0,
            velocity_limit=17.3,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
    },
)
