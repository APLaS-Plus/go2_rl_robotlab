from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# 地形难度：按身体高度等比例缩放
#
# 缩放来源与公式：
#   LITE3_BODY_HEIGHT_REF = 0.375 m  (money12532/Lite3_RL_Project
#       rl_training/.../assets/deeprobotics.py 的 init_state.pos.z)
#   GO2_BODY_HEIGHT_REF   = 0.38 m   (RobotLab Go2 任务 BASE_HEIGHT_TARGET)
#   TERRAIN_HEIGHT_SCALE  = 0.375 / 0.38 = 0.98684
#
# 所有"垂直方向"的地形参数（台阶高、障碍高、噪声幅值、坡度）都乘以该比例，
# 使 Lite3 面对的地形难度相对自身体型与 Go2 面临的难度等价。
# 水平方向参数（台阶宽、平台宽、网格宽、地形尺寸）不缩放，与 Go2 保持一致。
#
# 数值对照（Go2 原值 -> Lite3 缩放值）：
#   stairs step_height_range  (0.05, 0.257) -> (0.0493, 0.2536)
#   obstacle_height_range     (0.05, 0.275) -> (0.0493, 0.2714)
#   boxes grid_height_range   (0.025, 0.10) -> (0.0247, 0.0987)
#   random_rough noise_range  (0.01, 0.06)  -> (0.0099, 0.0592)
#   slope_range               (0.1, 0.568)  -> (0.0987, 0.5605)
# ---------------------------------------------------------------------------

TERRAIN_HEIGHT_SCALE = 0.375 / 0.38  # ≈ 0.98684


def _scale(rng: tuple[float, float]) -> tuple[float, float]:
    """Scale a vertical (height) range by the body-height ratio."""
    return (round(rng[0] * TERRAIN_HEIGHT_SCALE, 5), round(rng[1] * TERRAIN_HEIGHT_SCALE, 5))


@configclass
class Lite3TerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    """Terrain generator for Lite3: Go2 difficulty scaled by body height."""

    pass


# 地形类型、比例、水平尺寸沿用 RobotLab Go2 的 TERRAIN_CFG 设计，
# 仅垂直难度按身体高度缩放。
TERRAIN_CFG = Lite3TerrainGeneratorCfg(
    size=(9.0, 9.0),  # 8.0 terrain + 0.5*2 sub_terrain_border_width
    border_width=25.0,
    sub_terrain_border_width=0.5,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_gym_difficulty=False,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.05,
            amplitude_range=_scale((0.1, 0.28)),
            num_waves=5,
            noise_range=_scale((-0.05, 0.05)),
            noise_step=0.005,
            downsampled_scale=0.2,
        ),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=_scale((0.1, 0.568)),
            platform_width=3.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=_scale((0.1, 0.568)),
            platform_width=3.0,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=_scale((0.05, 0.257)),
            step_width=0.31,
            platform_width=3.0,
        ),
        "stairs_down": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=_scale((0.05, 0.257)),
            step_width=0.31,
            platform_width=3.0,
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=_scale((0.05, 0.275)),
            num_obstacles=20,
            platform_width=3.0,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.0,
            stone_height_max=0.0,
            stone_width_range=(0.075, 1.575),
            stone_distance_range=(0.05, 0.10),
            holes_depth=-10.0,
            platform_width=4.0,
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.0,
            gap_width_range=(0.0, 0.9),
            platform_width=3.0,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
    },
)
