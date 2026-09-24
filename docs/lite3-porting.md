# Porting a New Robot: Lite3 Case Study

How `RobotLab-Lite3-v0` was built from the Go2 task, and what to watch out for when
porting another robot. Written for someone who has never seen this repo.

## What was copied vs. what changed

The Lite3 task **reuses the Go2 environment class and the entire MoE-CTS network**.
Nothing about learning was reinvented — only the robot changed.

| Layer | Reused from Go2 | Changed for Lite3 |
|---|---|---|
| Env class | `robot_lab.tasks.go2.env.go2_env:Go2Env` (registered as-is) | — |
| Action manager | `ActionManagerGo2` (provides `_prev_prev_action` for `action_smoothness_l2`) | — |
| Network | MoE 8 experts `[512,256,256]`, teacher `[512,256]`+L2Norm, actor/critic `[512,256,128]`, latent 32 ELU | — |
| Algorithm | `MoECTS`, `teacher_env_ratio=0.75`, `load_balance_coef=0.01` | — |
| Observation contract | 45-dim policy obs, history 10 → 450; critic 263 | — |
| Rewards | **all of them, unchanged** | — |
| Asset | — | `resources/lite3/` (USD/MJCF/URDF, 12MB) |
| Actuators | — | kp=30, kd=1; Hip 24 N·m / Knee 36 N·m; Hip 26.2 / Knee 17.3 rad/s |
| Joint limits | — | HipX ±0.523, HipY −2.67~0.314, Knee 0.524~2.792 |
| Default pose | — | HipX 0, HipY −0.65, Knee 1.3; base z = 0.375 |
| Action scale | — | HipX 0.125, others 0.25 |
| Obs scale | — | ang_vel 0.25, dof_pos 1.0, dof_vel 0.05 |
| Terrain | same types/proportions/horizontal sizes | **vertical difficulty × 0.98684** |

Physical parameters come from [`money12532/Lite3_RL_Project`](https://github.com/money12532/Lite3_RL_Project).
Only physical parameters were taken — not its reward design. Everyporting parameter is (One exception, found after the first 20k run: the
stance-height target -- see *A third trap* below.)
commented in `source/robot_lab/robot_lab/tasks/lite3/env_cfg.py`.

## Terrain scaling

Lite3 is shorter than Go2 (0.375 m vs 0.38 m nominal body height), so identical
terrain is relatively harder. Vertical parameters are scaled by the body-height
ratio; horizontal parameters are left alone.

```
TERRAIN_HEIGHT_SCALE = 0.375 / 0.38 = 0.98684
```

| Parameter | Go2 | Lite3 |
|---|---|---|
| stairs `step_height_range` | (0.05, 0.257) | (0.04934, 0.25362) |
| obstacle `obstacle_height_range` | (0.05, 0.275) | (0.04934, 0.27138) |
| slope `slope_range` | (0.1, 0.568) | (0.09868, 0.56052) |
| wave `amplitude_range` | (0.1, 0.28) | (0.09868, 0.27632) |
| rough `noise_range` | (0.01, 0.06) | (0.00987, 0.05921) |

The scale factor and the full before/after table live at the top of
`source/robot_lab/robot_lab/tasks/lite3/mdp/terrains.py`.

## Two traps that cost real debugging time

### 1. `HfWaveTerrainCfg` does not accept `noise_range`

IsaacLab's `HfWaveTerrainCfg` only has `amplitude_range` and `num_waves`. The Go2
task defines its own subclasses that add roughness on top:

```python
@height_field_to_mesh
def wave_terrain(difficulty, cfg):
    wave = hf_terrains.wave_terrain.__wrapped__(difficulty, cfg)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.rint(wave + rough).astype(np.int16)

@configclass
class WaveTerrainCfg(terrain_gen.HfWaveTerrainCfg):
    function = wave_terrain
    amplitude_range = (0.1, 0.28)
    num_waves = 5
    noise_range = (-0.05, 0.05)      # <- only exists on the subclass
    noise_step = 0.005
    downsampled_scale = 0.2
```

If you copy the terrain config but not these subclasses, you get
`TypeError: HfWaveTerrainCfg.__init__() got an unexpected keyword argument 'noise_range'`.
`RoughSlopeTerrainCfg` and `PerSubTerrainSlopeThresholdGenerator` must come along too.

### 2. Body-name regexes are case-sensitive and robot-specific

Go2's USD uses lowercase link names (`*_thigh`, `*_calf`); Lite3's USD uses
uppercase (`FL_THIGH`, `FL_SHANK`, `*_FOOT`). Copying the regex verbatim makes it
match nothing, and IsaacLab fails at env construction with:

```
ValueError: Error while parsing 'undesired_contacts:sensor_cfg'.
Not all regular expressions are matched! .*_thigh|.*_calf: []
Available strings: ['TORSO', 'FL_HIP', 'FL_THIGH', 'FL_SHANK', 'FL_FOOT', ...]
```

Always grep every `body_names=` / `joint_names=` regex against the target robot's
actual prim names before launching.

## A third trap: the stance-height target (fixed 2026-09-25)

`RewardsCfg` is the Go2 design unchanged, **except one number**: the stance height consumed
by `base_height_l2` and `feet_regulation`. It was copied as `BASE_HEIGHT_TARGET = 0.55`
from the reference repo's *rough reward config*, and that value is an upstream mistake:

- Official `DeepRoboticsLab/rl_training` (which `money12532` re-publishes) kept
  `base_height_l2 = 0.35 m / weight -10` from 2025-08 until commit `61443d28`
  (2026-04-25), which changed it to `0.55 m / weight -50`. That same commit raised the
  spawn stance by only 4 cm (`pos.z 0.35 -> 0.375`, `HipY -0.8 -> -0.65`, `Knee 1.6 -> 1.3`).
- 0.55 m is unreachable for Lite3: the official standing size is 610x370x**406 mm** (whole
  robot); the MJCF places the base at **0.3485 m** at the nominal stance and at most
  **~0.42 m** with the knee at its lower limit (0.524 rad). The term measures *base height
  above the local terrain*, so the penalty can never reach 0 -- it degenerates into a
  constant drag plus a 'stretch the legs' gradient.
- Inverting the logged `Episode_Reward/base_height_l2` of the first 20k run gives a stance
  of ~**0.39 m** at the end (nominal 0.3485 m) and a constant penalty of about
  **-6.4 per episode** (~19% of the return). Go2, whose target (0.38 m) is reachable, pays
  only -0.23..-0.55 per episode for the same term.

Fix: `BASE_HEIGHT_TARGET = 0.35`, consistent with the reference's own *flat* config (0.35),
its IsaacGym config (0.36), and the measured stance. The rule this violated: a height
**target** is a reward hyperparameter and must not travel with the pose / spawn / actuator
numbers, even though it looks like a physical quantity.

## Running it

```bash
cd ~/go2_rl_robotlab
.venv/bin/python scripts/rsl_rl/train.py \
  --task=RobotLab-Lite3-v0 --headless \
  --num_envs 16384 --max_iterations 20000 --logger tensorboard
```

Smoke-test first with `--num_envs 2 --max_iterations 1`; Isaac Sim takes 2-3 minutes
to load before the first iteration prints.

Expected speed: **~8.9 s/iter** at 16384 envs. That is ~65% slower than Go2's 5.4 s/iter
because Lite3 is lighter (11.9 kg) and the scaled terrain is relatively harder, so
episodes end sooner and resets are more frequent.

## Reference measurements (iter ~9600 of 20000)

| Metric | Lite3 | Go2 (same iter) |
|---|---|---|
| teacher reward | 34.1 | ~44-46 |
| student reward | 24.7 | ~39 |
| `terrain_levels` | 6.08 | 6.08 |
| `time_out` ratio | 0.90 | 0.94 |
| `illegal_contact` | 0.10 | 0.06 |
| `mean_noise_std` | 0.46 | 0.38 |

`terrain_levels` matching Go2 is the key signal that the physical port did not handicap
terrain adaptation. `illegal_contact` and `noise_std` running higher is consistent with
Lite3 being light with small feet on rough ground.
