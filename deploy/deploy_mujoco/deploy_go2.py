"""Run a Go2 policy deployment in MuJoCo with joystick command input.

Overview:
This script loads a TorchScript policy, steps MuJoCo simulation with PD control, and
queries the policy at control decimation using only current-frame observations. Both
the exported CTS and DreamWaQ policies manage their observation history internally.

Quick Start:
    python deploy/deploy_mujoco/deploy_go2.py
    python deploy/deploy_mujoco/deploy_go2.py --config go2_dreamwaq.yaml
"""

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from utils import (
    build_delay_buffers,
    display_current_command,
    gravity_from_quat,
    init_joystick,
    load_config,
    open_video_writer,
    pd_control,
    read_joystick_command,
    sample_delayed_targets,
    set_initial_state,
    setup_tracking_camera,
    MujocoRenderUtils,
)


CONFIG_NAME = "go2.yaml"
VIDEO_DIR = Path(__file__).with_name("videos")
ACTUATOR_GROUPS = (
    np.array([0, 3, 6, 9], dtype=np.int64),
    np.array([1, 4, 7, 10], dtype=np.int64),
    np.array([2, 5, 8, 11], dtype=np.int64),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=CONFIG_NAME,
        help="Config filename under deploy_mujoco/configs, or a config file path.",
    )
    parser.add_argument("--policy", type=Path, help="Override policy_path from the config.")
    parser.add_argument("--xml", type=Path, help="Override xml_path from the config.")
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--steps", type=int, help="Stop after this many MuJoCo physics steps.")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace simulation to wall-clock time.")
    return parser.parse_args()


def build_features(data, action: np.ndarray, cmd: np.ndarray, cfg):
    """Build the current single-frame feature dictionary for policy inference.

    Args:
        data: MuJoCo runtime data object.
        action: Latest policy action in policy-action order.
        cmd: Current command vector.
        cfg: Loaded deployment configuration.

    Returns:
        A feature dictionary keyed by observation group name.
    """
    joint_pos = (data.qpos[7:] - cfg.default_angles) * cfg.dof_pos_scale
    joint_vel = data.qvel[6:] * cfg.dof_vel_scale
    return {
        "ang_vel": data.qvel[3:6] * cfg.ang_vel_scale,
        "gravity": gravity_from_quat(data.qpos[3:7]),
        "cmd": cmd * cfg.cmd_scale,
        "joint_pos": joint_pos[cfg.idx_mj2obs],
        "joint_vel": joint_vel[cfg.idx_mj2obs],
        "last_action": action,
    }


def action_to_target(action: np.ndarray, cfg):
    """Convert normalized action output to joint position targets.

    Args:
        action: Action in model joint order.
        cfg: Loaded deployment configuration.

    Returns:
        Target joint positions in MuJoCo qpos order.
    """
    action_mj = action[cfg.idx_mj2action]
    if np.isscalar(cfg.action_pos_scale):
        scale_mj = cfg.action_pos_scale
    else:
        scale_mj = cfg.action_pos_scale[cfg.idx_mj2action]
    return cfg.default_angles + action_mj * scale_mj


def disable_robot_self_collisions(model: mujoco.MjModel) -> None:
    """Apply Unitree training's robot-vs-robot contact filter in memory."""
    collision_geoms = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    robot_geoms = (model.geom_bodyid != 0) & collision_geoms
    model.geom_contype[robot_geoms] = 1
    model.geom_conaffinity[robot_geoms] = 2


def control_order_from_model(model: mujoco.MjModel, cfg) -> np.ndarray:
    """Map XML actuator slots to the policy-action joint order."""
    if model.nu != cfg.num_actions:
        raise ValueError(f"Expected {cfg.num_actions} actuators, found {model.nu}.")
    actuator_joint_names: list[str] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            raise ValueError(f"Actuator {actuator_id} is not attached to a named joint.")
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            raise ValueError(f"Actuator {actuator_id} is not attached to a named joint.")
        actuator_joint_names.append(joint_name)
    # model_joint_names is represented by idx_action2mj; recover names through
    # the qpos-order names from config rather than relying on XML actuator order.
    # idx_action2mj maps action index -> qpos index, so its inverse expresses
    # exactly the policy action order against the actuator joint names.
    action_index_by_mj_index = cfg.idx_mj2action
    qpos_names = getattr(cfg, "mujoco_joint_names", None)
    if qpos_names is None:
        # Existing configs without names are identity ordered by construction.
        return np.arange(model.nu, dtype=np.int64)
    return np.asarray(
        [action_index_by_mj_index[qpos_names.index(name)] for name in actuator_joint_names],
        dtype=np.int64,
    )


def build_single_obs(features: dict[str, np.ndarray], layout: list[tuple[str, int]]) -> np.ndarray:
    """Flatten current feature groups into one single observation vector.

    Args:
        features: Current-step feature dictionary.
        layout: Ordered feature layout specification.

    Returns:
        A concatenated single-frame observation vector.
    """
    return np.concatenate([features[name] for name, _ in layout], axis=0).astype(np.float32, copy=False)


def main() -> None:
    """Run MuJoCo simulation and deploy the policy in closed-loop control."""
    args = parse_args()
    cfg = load_config(args.config)
    if args.policy is not None:
        cfg.policy_path = args.policy.resolve()
    if args.xml is not None:
        cfg.xml_path = args.xml.resolve()
    if args.command is not None:
        cfg.cmd_init = np.asarray(args.command, dtype=np.float32)
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    layout = [
        ("ang_vel", 3),
        ("gravity", 3),
        ("cmd", 3),
        ("joint_pos", cfg.num_actions),
        ("joint_vel", cfg.num_actions),
        ("last_action", cfg.num_actions),
    ]
    if sum(dim for _, dim in layout) != cfg.num_obs:
        raise ValueError(f"Observation layout does not match num_obs={cfg.num_obs}.")

    joystick = None if args.headless else init_joystick()
    cmd = cfg.cmd_init.copy()
    if not args.headless:
        display_current_command(cmd)

    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    data = mujoco.MjData(model)
    if cfg.self_collisions == "training-disabled":
        disable_robot_self_collisions(model)
    model.opt.timestep = cfg.dt
    set_initial_state(data, cfg.base_init_pos, cfg.base_init_quat, cfg.default_angles)
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(str(cfg.policy_path))
    if hasattr(policy, "reset"):
        policy.reset()
    writer, frame_skip, video_path = open_video_writer(
        cfg.save_video, policy_path=cfg.policy_path, cmd=cmd, dt=cfg.dt, video_dir=VIDEO_DIR, video_fps=cfg.video_fps
    )
    if args.headless and writer:
        raise ValueError("Headless mode does not support video recording; set save_video: false.")
    renderer = mujoco.Renderer(model, height=360, width=640) if writer else None

    action = np.zeros(cfg.num_actions, dtype=np.float32)
    target_pos = cfg.default_angles.copy()
    target_vel = np.zeros(cfg.num_actions, dtype=np.float32)
    pos_history = build_delay_buffers(target_pos, delay_max=cfg.delay_max)
    delay_rng = np.random.default_rng(cfg.delay_seed)
    control_action_indices = control_order_from_model(model, cfg)
    render_substeps = max(1, int((1.0 / cfg.render_fps) / cfg.dt))
    mujoco_render_utils = MujocoRenderUtils()

    def update_policy() -> None:
        nonlocal action, target_pos
        action_tensor = policy(torch.from_numpy(build_single_obs(build_features(data, action, cmd, cfg), layout)).unsqueeze(0))
        action = action_tensor.detach().cpu().numpy().squeeze()
        pos_history.append(action_to_target(action, cfg).copy())
        target_pos = sample_delayed_targets(
            pos_history, ACTUATOR_GROUPS, cfg.delay_min, cfg.delay_max, delay_rng
        )

    viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    with viewer_context as viewer:
        if viewer is not None:
            setup_tracking_camera(viewer)
        start_time = time.time()
        counter = 0

        while (
            (viewer is None or viewer.is_running())
            and time.time() - start_time < cfg.duration
            and (args.steps is None or counter < args.steps)
        ):
            step_start = time.time()
            if joystick and counter % cfg.decimation == 0:
                cmd = read_joystick_command(joystick, cfg.max_cmd)
            if cfg.policy_update_before_step and counter % cfg.decimation == 0:
                update_policy()

            torque_mj = pd_control(target_pos, data.qpos[7:], cfg.kps, target_vel, data.qvel[6:], cfg.kds)
            torque_action = torque_mj[cfg.idx_action2mj]
            control = torque_action[control_action_indices]
            if cfg.use_model_torque_limits:
                limits = np.minimum(
                    np.abs(model.actuator_ctrlrange[:, 0]),
                    np.abs(model.actuator_ctrlrange[:, 1]),
                )
                control = np.clip(control, -limits, limits)
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            mujoco_render_utils.update(cmd, data)

            if writer and renderer is not None and counter % frame_skip == 0:
                try:
                    renderer.update_scene(data, camera=viewer.cam)
                    mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                    writer.append_data(renderer.render())
                except Exception as exc:
                    print(f"Render error: {exc}")

            counter += 1
            if not cfg.policy_update_before_step and counter % cfg.decimation == 0:
                update_policy()
                if not args.headless:
                    display_current_command(cmd)

            if viewer is not None and counter % render_substeps == 0:
                mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
                viewer.sync()
            if not args.no_realtime:
                sleep_time = cfg.dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    if writer:
        writer.close()
        print(f"Video saved: {video_path}")
    base_pos = data.qpos[:3]
    base_vel = data.qvel[:3]
    gravity = gravity_from_quat(data.qpos[3:7])
    print(
        f"Completed {counter} MuJoCo physics steps; finite={bool(np.isfinite(data.qpos).all())}; "
        f"base_pos={np.array2string(base_pos, precision=3)}; "
        f"base_vel={np.array2string(base_vel, precision=3)}; "
        f"projected_gravity={np.array2string(gravity, precision=3)}"
    )


if __name__ == "__main__":
    main()
