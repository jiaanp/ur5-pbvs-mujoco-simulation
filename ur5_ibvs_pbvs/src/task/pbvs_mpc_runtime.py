import cv2
import numpy as np

from src.perception.target_motion import TargetMotionController
from src.utils.transforms import make_transform, rotation_matrix_to_quaternion


def build_error_state_world(e_p_cam, e_r_cam, camera_rotation_world, r_mj_camera_from_cv_camera):
    e_p_mj = r_mj_camera_from_cv_camera @ e_p_cam
    e_r_mj = r_mj_camera_from_cv_camera @ e_r_cam

    e_p_world = camera_rotation_world @ e_p_mj
    e_r_world = camera_rotation_world @ e_r_mj

    return np.concatenate([e_p_world, e_r_world], axis=0)


def compute_orientation_lock_error_in_camera_frame(
    camera_rotation_world,
    locked_camera_rotation_world,
):
    r_err_cam = camera_rotation_world.T @ locked_camera_rotation_world
    e_r_cam, _ = cv2.Rodrigues(r_err_cam)
    return e_r_cam.ravel()


def build_lift_error_state_world(lift_position_error_world):
    return np.concatenate(
        [np.asarray(lift_position_error_world, dtype=np.float64), np.zeros(3, dtype=np.float64)],
        axis=0,
    )


def compute_desired_camera_feedforward_world(
    target_quat_world,
    target_linear_world,
    target_angular_world,
    t_tag_camera_desired,
    linear_gain=1.0,
    angular_gain=1.0,
):
    r_world_tag = TargetMotionController.quaternion_to_rotation_matrix(target_quat_world)
    p_tag_camera_desired = t_tag_camera_desired[:3, 3]
    desired_offset_world = r_world_tag @ p_tag_camera_desired

    linear_ff_world = target_linear_world + np.cross(target_angular_world, desired_offset_world)
    angular_ff_world = target_angular_world

    return np.concatenate(
        [linear_gain * linear_ff_world, angular_gain * angular_ff_world],
        axis=0,
    )


def build_reference_error_trajectory(
    target_quat_world,
    target_linear_world,
    target_angular_world,
    t_tag_camera_desired,
    horizon,
    dt,
    reference_preview_gain,
):
    ff_velocity_world = compute_desired_camera_feedforward_world(
        target_quat_world,
        target_linear_world,
        target_angular_world,
        t_tag_camera_desired,
        linear_gain=1.0,
        angular_gain=1.0,
    )

    reference = np.zeros((horizon, 6), dtype=np.float64)
    for i in range(horizon):
        step_scale = float(i + 1) * dt
        reference[i] = -reference_preview_gain * step_scale * ff_velocity_world

    return reference


def set_gripper_ctrl(data, actuator_name, ctrl_value):
    data.actuator(actuator_name).ctrl = float(ctrl_value)


def clip_gripper_ctrl(ctrl_value, min_value=-70.0, max_value=5.0):
    return float(np.clip(ctrl_value, min_value, max_value))


def get_gripper_ctrl_for_phase(phase, gripper_open_ctrl, gripper_close_ctrl):
    if phase in ("track", "approach", "release", "home", "done"):
        return gripper_open_ctrl
    if phase in ("attach", "lift", "place"):
        return gripper_close_ctrl
    return gripper_open_ctrl


def smooth_error_vector(last_error, current_error, alpha):
    last_error = np.asarray(last_error, dtype=np.float64)
    current_error = np.asarray(current_error, dtype=np.float64)
    return (1.0 - alpha) * last_error + alpha * current_error


def print_target_control_instructions(target_is_mocap, conveyor_enabled):
    print("Target control:")
    if target_is_mocap:
        print("  Focus the 'End-Effector Camera' window, then use W/S A/D R/F to move the tag.")
        print("  Use I/K J/L U/O to rotate the tag, SPACE to reset, M to toggle auto motion.")
    else:
        print("  Current target is a free rigid body. Manual mocap target motion is disabled.")
        if conveyor_enabled:
            print("  Physical-looking conveyor mode enabled: moving belt strips carry the target.")


def get_target_home_pose(data, target_body_id, target_mocap_id, target_is_mocap):
    if target_is_mocap:
        return (
            data.mocap_pos[target_mocap_id].copy(),
            data.mocap_quat[target_mocap_id].copy(),
        )
    return data.xpos[target_body_id].copy(), data.xquat[target_body_id].copy()


def update_target_pose_for_step(
    env,
    data,
    target_body_name,
    site_name,
    target_body_id,
    target_mocap_id,
    target_is_mocap,
    current_site_pos,
    grasp_state_machine,
    target_motion,
    attached_target_offset_world,
    attached_target_quat,
):
    if target_is_mocap:
        target_pos, target_quat = target_motion.get_target_pose(data.time)

        if grasp_state_machine.attached:
            target_pos = current_site_pos + attached_target_offset_world
            target_quat = attached_target_quat.copy()

        env.set_target_pose(target_body_name, target_pos, target_quat)
        env.forward()
        current_site_pos, _ = env.get_site_pose(site_name)

        current_target_pos = data.mocap_pos[target_mocap_id].copy()
        current_target_quat = data.mocap_quat[target_mocap_id].copy()
    else:
        current_target_pos = data.xpos[target_body_id].copy()
        current_target_quat = data.xquat[target_body_id].copy()

    return current_site_pos, current_target_pos, current_target_quat


def draw_runtime_overlay(
    vis,
    target_is_mocap,
    target_motion,
    conveyor_active,
    conveyor_speed,
    grasp_state_machine,
    gripper_target_ctrl,
):
    if target_is_mocap:
        vis = target_motion.draw_overlay(vis)
    else:
        cv2.putText(
            vis,
            f"PHYSICAL TARGET MODE  BELT {'ON' if conveyor_active else 'OFF'}  {conveyor_speed:.2f} m/s",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    vis = grasp_state_machine.draw_overlay(vis)
    cv2.putText(
        vis,
        f"GRIPPER CTRL: {gripper_target_ctrl:.2f}  AUTO BY PHASE: {grasp_state_machine.phase.upper()}",
        (10, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    return vis


def solve_cartesian_transport_qdot(
    env,
    robot_kin,
    mpc_controller,
    error_world,
    last_q_dot,
    max_q_dot,
    arm_dof_count,
):
    error_state_world = build_lift_error_state_world(error_world)
    jacobian = robot_kin.compute_frame_jacobian(env.get_joint_positions(arm_dof_count))
    q_dot = mpc_controller.solve(
        error_state_world,
        jacobian,
        last_q_dot=last_q_dot,
    )
    return np.clip(q_dot, -max_q_dot, max_q_dot)


def estimate_tag_world_pose_from_camera_observation(
    camera_position_world,
    camera_rotation_world,
    t_camera_tag_cv,
    r_mj_camera_from_cv_camera,
):
    """
    根据某个相机观测到的 T_camera_tag（OpenCV 相机系），恢复目标在世界系下的位姿。
    """
    t_world_mj_camera = make_transform(camera_rotation_world, camera_position_world)
    t_mj_camera_cv_camera = make_transform(r_mj_camera_from_cv_camera, np.zeros(3, dtype=np.float64))
    t_world_tag = t_world_mj_camera @ t_mj_camera_cv_camera @ np.asarray(t_camera_tag_cv, dtype=np.float64)

    world_pos = t_world_tag[:3, 3].copy()
    world_quat = rotation_matrix_to_quaternion(t_world_tag[:3, :3])
    return world_pos, world_quat, t_world_tag


def project_image_point_to_world_plane(
    pixel_uv,
    camera_matrix,
    camera_position_world,
    camera_rotation_world,
    r_mj_camera_from_cv_camera,
    plane_z_world,
):
    """
    将图像像素点通过相机模型投影到世界系 z=plane_z_world 的平面上。
    这里用的是：
    图像中心点 -> OpenCV 相机系射线 -> MuJoCo 相机局部系 -> 世界系 -> 与平面求交。
    """
    pixel_h = np.array([pixel_uv[0], pixel_uv[1], 1.0], dtype=np.float64)
    ray_cv = np.linalg.inv(np.asarray(camera_matrix, dtype=np.float64)) @ pixel_h
    ray_mj = np.asarray(r_mj_camera_from_cv_camera, dtype=np.float64) @ ray_cv
    ray_world = np.asarray(camera_rotation_world, dtype=np.float64) @ ray_mj

    if abs(ray_world[2]) < 1e-9:
        return None

    scale = (float(plane_z_world) - float(camera_position_world[2])) / float(ray_world[2])
    if scale <= 0.0:
        return None

    return np.asarray(camera_position_world, dtype=np.float64) + scale * ray_world
