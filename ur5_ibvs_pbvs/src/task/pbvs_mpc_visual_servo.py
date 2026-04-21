import numpy as np

from src.controllers.pbvs_controller import soften_error_vector
from src.perception.pose_estimator import draw_pose_axes
from src.task.pbvs_mpc_runtime import (
    build_error_state_world,
    build_reference_error_trajectory,
    compute_orientation_lock_error_in_camera_frame,
    smooth_error_vector,
)
from src.utils.transforms import rvec_tvec_to_transform


def run_visual_servo_step(
    env,
    data,
    camera_id,
    robot_kin,
    pose_estimator,
    pbvs_controller,
    mpc_controller,
    tracking_mpc_controller,
    grasp_state_machine,
    target_motion,
    controller_dt,
    current_target_pos,
    current_target_quat,
    current_site_pos,
    target_linear_world,
    target_angular_world,
    target_motion_speed,
    tracking_target,
    t_tag_camera_desired,
    tag,
    vis,
    camera_matrix,
    dist_coeffs,
    width,
    height,
    last_q_dot,
    last_e_p_cam,
    last_e_r_cam,
    locked_approach_camera_rotation_world,
    position_deadband,
    position_soft_zone,
    rotation_deadband,
    rotation_soft_zone,
    error_smoothing_alpha,
    target_motion_speed_threshold,
    max_tracking_q_dot,
    max_q_dot,
    reference_preview_gain,
    r_mj_camera_from_cv_camera,
    actuator_names,
    arm_dof_count,
    target_is_mocap,
):
    success, rvec, tvec = pose_estimator.estimate_pose(
        tag,
        camera_matrix,
        dist_coeffs,
    )

    if not success:
        return {
            "success": False,
            "vis": vis,
            "last_q_dot": last_q_dot,
            "last_e_p_cam": last_e_p_cam,
            "last_e_r_cam": last_e_r_cam,
            "locked_approach_camera_rotation_world": locked_approach_camera_rotation_world,
            "phase": None,
            "attached_target_offset_world": None,
            "attached_target_quat": None,
        }

    vis = draw_pose_axes(
        vis,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        axis_length=0.05,
    )

    t_camera_tag = rvec_tvec_to_transform(rvec, tvec)
    e_p_cam_raw, e_r_cam_raw, _ = pbvs_controller.compute_pbvs_error_in_camera_frame(
        t_camera_tag,
        t_tag_camera_desired,
    )
    e_p_cam = soften_error_vector(
        e_p_cam_raw,
        position_deadband,
        position_soft_zone,
    )
    e_r_cam = soften_error_vector(
        e_r_cam_raw,
        rotation_deadband,
        rotation_soft_zone,
    )
    e_p_cam = smooth_error_vector(last_e_p_cam, e_p_cam, error_smoothing_alpha)
    e_r_cam = smooth_error_vector(last_e_r_cam, e_r_cam, error_smoothing_alpha)
    last_e_p_cam = e_p_cam.copy()
    last_e_r_cam = e_r_cam.copy()

    camera_rotation_world = data.cam_xmat[camera_id].reshape(3, 3)
    if grasp_state_machine.phase == "approach":
        if locked_approach_camera_rotation_world is None:
            locked_approach_camera_rotation_world = camera_rotation_world.copy()
        e_r_cam = compute_orientation_lock_error_in_camera_frame(
            camera_rotation_world,
            locked_approach_camera_rotation_world,
        )
    else:
        locked_approach_camera_rotation_world = None

    error_state_world = build_error_state_world(
        e_p_cam,
        e_r_cam,
        camera_rotation_world,
        r_mj_camera_from_cv_camera,
    )

    jacobian = robot_kin.compute_frame_jacobian(env.get_joint_positions(arm_dof_count))
    active_mpc_controller = tracking_mpc_controller if tracking_target else mpc_controller
    active_max_q_dot = max_tracking_q_dot if tracking_target else max_q_dot
    reference_trajectory = build_reference_error_trajectory(
        current_target_quat,
        target_linear_world,
        target_angular_world,
        t_tag_camera_desired,
        active_mpc_controller.horizon,
        controller_dt,
        reference_preview_gain,
    )

    q_dot_mpc = active_mpc_controller.solve(
        error_state_world,
        jacobian,
        last_q_dot=last_q_dot,
        reference_trajectory=reference_trajectory,
    )

    q_dot = np.clip(q_dot_mpc, -active_max_q_dot, active_max_q_dot)
    env.apply_joint_velocity(actuator_names, q_dot)
    last_q_dot = q_dot.copy()

    phase = grasp_state_machine.update(
        np.linalg.norm(e_p_cam_raw),
        np.linalg.norm(e_r_cam_raw),
        target_motion_speed=target_motion_speed,
        motion_threshold=target_motion_speed_threshold,
    )

    attached_target_offset_world = None
    attached_target_quat = None
    if phase == "lift":
        if target_is_mocap:
            target_motion.auto_mode = False
            target_motion.manual_pos = current_target_pos.copy()
            target_motion.manual_quat = current_target_quat.copy()
            attached_target_offset_world = current_target_pos - current_site_pos
            attached_target_quat = current_target_quat.copy()
        grasp_state_machine.start_lift(current_site_pos)
    elif phase == "approach" and locked_approach_camera_rotation_world is None:
        locked_approach_camera_rotation_world = camera_rotation_world.copy()

    print("e_p_cam_raw =", np.round(e_p_cam_raw, 4))
    print("e_r_cam_raw =", np.round(e_r_cam_raw, 4))
    print("e_p_cam =", np.round(e_p_cam, 4))
    print("e_r_cam =", np.round(e_r_cam, 4))
    print("phase =", grasp_state_machine.phase)
    print("attached =", grasp_state_machine.attached)
    print("error_state_world =", np.round(error_state_world, 4))
    print("tracking_target =", tracking_target)
    print("target_motion_speed =", np.round(target_motion_speed, 4))
    print("reference_trajectory[0] =", np.round(reference_trajectory[0], 4))
    print("q_dot_mpc =", np.round(q_dot_mpc, 4))
    print("q_dot =", np.round(q_dot, 4))

    return {
        "success": True,
        "vis": vis,
        "last_q_dot": last_q_dot,
        "last_e_p_cam": last_e_p_cam,
        "last_e_r_cam": last_e_r_cam,
        "locked_approach_camera_rotation_world": locked_approach_camera_rotation_world,
        "phase": phase,
        "attached_target_offset_world": attached_target_offset_world,
        "attached_target_quat": attached_target_quat,
    }
