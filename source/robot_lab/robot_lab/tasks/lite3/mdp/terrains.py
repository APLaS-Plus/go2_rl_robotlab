from __future__ import annotations

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh

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
#   stairs step_height_range  (0.05, 0.257) -> (0.04934, 0.25362)
#   obstacle_height_range     (0.05, 0.275) -> (0.04934, 0.27138)
#   random_rough noise_range  (0.01, 0.06)  -> (0.00987, 0.05921)
#   slope_range               (0.1, 0.568)  -> (0.09868, 0.56052)
#   wave amplitude_range      (0.1, 0.28)   -> (0.09868, 0.27632)
# ---------------------------------------------------------------------------

TERRAIN_HEIGHT_SCALE = 0.375 / 0.38  # ~ 0.98684


def _scale(rng):
    """Scale a vertical (height) range by the body-height ratio."""
    return (round(rng[0] * TERRAIN_HEIGHT_SCALE, 5), round(rng[1] * TERRAIN_HEIGHT_SCALE, 5))


class PerSubTerrainSlopeThresholdGenerator(TerrainGenerator):
    """Terrain generator that lets each height-field sub-terrain override slope_threshold.

    与 Go2 的 PerSubTerrainSlopeThresholdGenerator 一致（按 sub-terrain 单独控制坡度修正）。
    """

    def __init__(self, cfg, device="cpu"):
        self._apply_sub_terrain_border_width(cfg)
        super().__init__(cfg, device)

    def _apply_sub_terrain_border_width(self, cfg):
        border_width = getattr(cfg, "sub_terrain_border_width", None)
        if border_width is None:
            return
        for sub_cfg in cfg.sub_terrains.values():
            if hasattr(sub_cfg, "border_width"):
                sub_cfg.border_width = border_width

    def _get_terrain_mesh(self, difficulty, cfg):
        difficulty = self._maybe_use_gym_difficulty(difficulty)
        override = getattr(cfg, "slope_threshold_override", None)
        if override is None:
            return super()._get_terrain_mesh(difficulty, cfg)
        original_slope_threshold = getattr(cfg, "slope_threshold", None)
        cfg.slope_threshold = override
        try:
            return super()._get_terrain_mesh(difficulty, cfg)
        finally:
            cfg.slope_threshold = original_slope_threshold

    def _maybe_use_gym_difficulty(self, difficulty):
        if not getattr(self.cfg, "use_gym_difficulty", False):
            return difficulty
        lower, upper = self.cfg.difficulty_range
        if upper > lower:
            normalized_difficulty = (float(difficulty) - lower) / (upper - lower)
        else:
            normalized_difficulty = float(difficulty)
        normalized_difficulty = np.clip(normalized_difficulty, 0.0, 1.0)
        num_levels = self.cfg.num_rows
        level = min(int(np.floor(normalized_difficulty * num_levels)), num_levels - 1)
        return level / num_levels


@configclass
class Lite3TerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    class_type = PerSubTerrainSlopeThresholdGenerator
    sub_terrain_border_width = None
    use_gym_difficulty = False


def with_slope_threshold(sub_terrain_cfg, slope_threshold):
    """Attach a per-sub-terrain slope-threshold override to a terrain config."""
    sub_terrain_cfg.slope_threshold_override = slope_threshold
    return sub_terrain_cfg


# -----------------------------------------------------------------------------
# Lite3 terrain setup (Go2 design, vertical difficulty scaled by body height)
# -----------------------------------------------------------------------------
@height_field_to_mesh
def wave_terrain(difficulty, cfg):
    """wave terrain: wave plus random uniform roughness."""
    wave = hf_terrains.wave_terrain.__wrapped__(difficulty, cfg)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.rint(wave + rough).astype(np.int16)


@height_field_to_mesh
def rough_slope_terrain(difficulty, cfg):
    """rough slope terrain: slope plus random uniform roughness."""
    slope = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.rint(slope + rough).astype(np.int16)


@configclass
class WaveTerrainCfg(terrain_gen.HfWaveTerrainCfg):
    """HfWaveTerrainCfg + random uniform roughness (same as Go2 WaveTerrainCfg)."""

    function = wave_terrain
    amplitude_range = (0.1, 0.28)
    num_waves = 5
    noise_range = (-0.05, 0.05)
    noise_step = 0.005
    downsampled_scale = 0.2


@configclass
class RoughSlopeTerrainCfg(terrain_gen.HfPyramidSlopedTerrainCfg):
    """Sloped terrain + random uniform roughness (same as Go2 RoughSlopeTerrainCfg)."""

    function = rough_slope_terrain
    slope_range = (0.1, 0.568)
    platform_width = 3.0
    noise_range = (-0.05, 0.05)
    noise_step = 0.005
    downsampled_scale = 0.2


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
    # slope correction = 0.75 ~ 36.9 degrees by default,
    # but recommended to set for each terrain type separately using with_slope_threshold
    slope_threshold=0.75,
    use_gym_difficulty=False,
    use_cache=False,
    sub_terrains={
        "wave": with_slope_threshold(
            WaveTerrainCfg(
                proportion=0.05,
                amplitude_range=_scale((0.1, 0.28)),
                noise_range=_scale((-0.05, 0.05)),
            ),
            10.0,  # effectively disable slope correction for wave terrain
        ),
        "slope_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=_scale((0.1, 0.568)),
                platform_width=3.0,
            ),
            10.0,  # effectively disable slope correction for slope_up terrain
        ),
        "slope_down": with_slope_threshold(
            terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=_scale((0.1, 0.568)),
                platform_width=3.0,
            ),
            10.0,  # effectively disable slope correction for slope_down terrain
        ),
        "rough_slope": with_slope_threshold(
            RoughSlopeTerrainCfg(
                proportion=0.05,
                slope_range=_scale((0.1, 0.568)),
                noise_range=_scale((-0.05, 0.05)),
            ),
            10.0,  # effectively disable slope correction for rough_slope terrain
        ),
        "stairs_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                proportion=0.25,
                step_height_range=_scale((0.05, 0.257)),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25,  # enable slope correction for stairs terrain by 14.0 degrees
        ),
        "stairs_down": with_slope_threshold(
            terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=_scale((0.05, 0.257)),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25,  # enable slope correction for stairs terrain by 14.0 degrees
        ),
        "obstacles": with_slope_threshold(
            terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.20,
                obstacle_width_range=(1.0, 2.0),
                obstacle_height_range=_scale((0.05, 0.275)),
                num_obstacles=20,
                platform_width=3.0,
            ),
            0.25,  # enable slope correction for obstacles terrain by 14.0 degrees
        ),
        "stepping_stones": with_slope_threshold(
            terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.0,
                stone_height_max=0.0,
                stone_width_range=(0.075, 1.575),
                stone_distance_range=(0.05, 0.10),
                holes_depth=-10.0,
                platform_width=4.0,
            ),
            0.25,
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.0,
            gap_width_range=(0.0, 0.9),
            platform_width=3.0,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
    },
)
