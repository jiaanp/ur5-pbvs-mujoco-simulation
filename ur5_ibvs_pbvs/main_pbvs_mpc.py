
import time

import cv2
import mujoco
import numpy as np

from src.config import (
    ACTUATOR_NAMES,
    ARM_DOF_COUNT,
    BLIND_ATTACH_FRAMES,
    CONVEYOR_ENABLED,
    CONVEYOR_SPEED,
    ERROR_SMOOTHING_ALPHA,
    GRASP_YAW_ABOUT_TAG_NORMAL_RAD,
    GRIPPER_ACTUATOR_NAME,
    GRIPPER_CLOSE_CTRL,
    GRIPPER_OPEN_CTRL,
    GUI_SLEEP,
    HEIGHT,
    HOME_JOINT_KP,
    HOME_JOINT_TOL,
    HOME_MAX_Q_DOT,
    LOST_TAG_DECAY,
    LOST_TAG_HOLD_FRAMES,
    MAX_PLACE_Q_DOT,
    MAX_Q_DOT,
    MAX_TRACKING_Q_DOT,
    MAX_TRANSPORT_Q_DOT,
    POSITION_DEADBAND,
    POSITION_SOFT_ZONE,
    REFERENCE_PREVIEW_GAIN,
    RELEASE_HOLD_FRAMES,
    ROTATION_DEADBAND,
    ROTATION_SOFT_ZONE,
    R_MJ_CAMERA_FROM_CV_CAMERA,
    SCENE_XML,
    SITE_NAME,
    TARGET_BODY_NAME,
    TARGET_MOTION_SPEED_THRESHOLD,
    WIDTH,
)
from src.task.pbvs_mpc_phases import (
    handle_done_phase,
    handle_home_phase,
    handle_lift_phase,
    handle_place_phase,
    handle_release_phase,
)
from src.task.pbvs_mpc_runtime import (
    clip_gripper_ctrl,
    draw_runtime_overlay,
    get_gripper_ctrl_for_phase,
    set_gripper_ctrl,
    update_target_pose_for_step,
)
from src.task.pbvs_mpc_setup import build_runtime
from src.task.pbvs_mpc_visual_servo import run_visual_servo_step
from src.utils.transforms import (
    build_desired_tag_camera_transform_from_grasp,
    make_transform,
)

def main():
    runtime = build_runtime(SCENE_XML)
    env = runtime["env"]
    model = runtime["model"]
    data = runtime["data"]
    controller_dt = runtime["controller_dt"]
    renderer = runtime["renderer"]
    robot_kin = runtime["robot_kin"]
    pose_estimator = runtime["pose_estimator"]
    pbvs_controller = runtime["pbvs_controller"]
    mpc_controller = runtime["mpc_controller"]
    tracking_mpc_controller = runtime["tracking_mpc_controller"]
    camera_id = runtime["camera_id"]
    camera_matrix = runtime["camera_matrix"]
    dist_coeffs = runtime["dist_coeffs"]
    detector = runtime["detector"]
    last_q_dot = runtime["last_q_dot"]
    gripper_target_ctrl = runtime["gripper_target_ctrl"]
    release_frames_remaining = runtime["release_frames_remaining"]
    lost_tag_count = runtime["lost_tag_count"]
    attached_target_offset_world = runtime["attached_target_offset_world"]
    attached_target_quat = runtime["attached_target_quat"]
    locked_approach_camera_rotation_world = runtime["locked_approach_camera_rotation_world"]
    last_e_p_cam = runtime["last_e_p_cam"]
    last_e_r_cam = runtime["last_e_r_cam"]
    conveyor_active = runtime["conveyor_active"]
    target_body_id = runtime["target_body_id"]
    target_mocap_id = runtime["target_mocap_id"]
    target_is_mocap = runtime["target_is_mocap"]
    conveyor_actuator_id = runtime["conveyor_actuator_id"]
    target_motion = runtime["target_motion"]
    grasp_state_machine = runtime["grasp_state_machine"]
    home_qpos = runtime["home_qpos"]
    place_site_target_world = runtime["place_site_target_world"]
    t_grasp_camera = runtime["t_grasp_camera"]

    with renderer.create_viewer() as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        while viewer.is_running():
            if CONVEYOR_ENABLED and conveyor_actuator_id >= 0:
                data.ctrl[conveyor_actuator_id] = CONVEYOR_SPEED if conveyor_active else 0.0
            if release_frames_remaining > 0:
                gripper_target_ctrl = GRIPPER_OPEN_CTRL
            else:
                gripper_target_ctrl = get_gripper_ctrl_for_phase(
                    grasp_state_machine.phase,
                    GRIPPER_OPEN_CTRL,
                    GRIPPER_CLOSE_CTRL,
                )
            set_gripper_ctrl(data, GRIPPER_ACTUATOR_NAME, gripper_target_ctrl)
            env.forward()
            current_site_pos, _ = env.get_site_pose(SITE_NAME)

            current_site_pos, current_target_pos, current_target_quat = update_target_pose_for_step(
                env,
                data,
                TARGET_BODY_NAME,
                SITE_NAME,
                target_body_id,
                target_mocap_id,
                target_is_mocap,
                current_site_pos,
                grasp_state_machine,
                target_motion,
                attached_target_offset_world,
                attached_target_quat,
            )

            target_linear_world, target_angular_world = target_motion.estimate_target_spatial_velocity_world(
                current_target_pos,
                current_target_quat,
                controller_dt,
            )
            t_tag_camera_desired = build_desired_tag_camera_transform_from_grasp(
                standoff=grasp_state_machine.get_desired_standoff(),
                t_grasp_camera=t_grasp_camera,
                yaw_about_normal_rad=GRASP_YAW_ABOUT_TAG_NORMAL_RAD,
            )
            active_standoff = grasp_state_machine.get_desired_standoff()
            target_motion_speed = np.linalg.norm(target_linear_world) + active_standoff * np.linalg.norm(target_angular_world)
            tracking_target = target_motion_speed > TARGET_MOTION_SPEED_THRESHOLD

            bgr = renderer.render_camera_bgr()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            vis = draw_apriltags(bgr, tags)
            vis = draw_runtime_overlay(
                vis,
                target_is_mocap,
                target_motion,
                conveyor_active,
                CONVEYOR_SPEED,
                grasp_state_machine,
                gripper_target_ctrl,
            )

            control_applied = False

            if grasp_state_machine.phase == "release":
                last_q_dot, release_frames_remaining = handle_release_phase(
                    env,
                    ACTUATOR_NAMES,
                    ARM_DOF_COUNT,
                    vis,
                    HEIGHT,
                    release_frames_remaining,
                    grasp_state_machine,
                )
                control_applied = True

            if grasp_state_machine.phase == "lift" and grasp_state_machine.attached:
                last_q_dot = handle_lift_phase(
                    env,
                    robot_kin,
                    mpc_controller,
                    grasp_state_machine,
                    current_site_pos,
                    place_site_target_world,
                    last_q_dot,
                    vis,
                    ACTUATOR_NAMES,
                    ARM_DOF_COUNT,
                    HEIGHT,
                    MAX_TRANSPORT_Q_DOT,
                )
                control_applied = True

            if grasp_state_machine.phase == "place" and grasp_state_machine.attached:
                last_q_dot, place_done = handle_place_phase(
                    env,
                    robot_kin,
                    mpc_controller,
                    grasp_state_machine,
                    current_site_pos,
                    last_q_dot,
                    vis,
                    ACTUATOR_NAMES,
                    ARM_DOF_COUNT,
                    HEIGHT,
                    MAX_PLACE_Q_DOT,
                )
                control_applied = True

                if place_done:
                    release_frames_remaining = RELEASE_HOLD_FRAMES

            if grasp_state_machine.phase == "home":
                last_q_dot = handle_home_phase(
                    env,
                    home_qpos,
                    grasp_state_machine,
                    vis,
                    ACTUATOR_NAMES,
                    ARM_DOF_COUNT,
                    HEIGHT,
                    HOME_JOINT_KP,
                    HOME_JOINT_TOL,
                    HOME_MAX_Q_DOT,
                )
                control_applied = True

            if grasp_state_machine.phase == "done":
                last_q_dot = handle_done_phase(env, vis, ACTUATOR_NAMES, ARM_DOF_COUNT, HEIGHT)
                control_applied = True

            if (not control_applied) and len(tags) > 0:
                servo_result = run_visual_servo_step(
                    env=env,
                    data=data,
                    camera_id=camera_id,
                    robot_kin=robot_kin,
                    pose_estimator=pose_estimator,
                    pbvs_controller=pbvs_controller,
                    mpc_controller=mpc_controller,
                    tracking_mpc_controller=tracking_mpc_controller,
                    grasp_state_machine=grasp_state_machine,
                    target_motion=target_motion,
                    controller_dt=controller_dt,
                    current_target_pos=current_target_pos,
                    current_target_quat=current_target_quat,
                    current_site_pos=current_site_pos,
                    target_linear_world=target_linear_world,
                    target_angular_world=target_angular_world,
                    target_motion_speed=target_motion_speed,
                    tracking_target=tracking_target,
                    t_tag_camera_desired=t_tag_camera_desired,
                    tag=tags[0],
                    vis=vis,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    width=WIDTH,
                    height=HEIGHT,
                    last_q_dot=last_q_dot,
                    last_e_p_cam=last_e_p_cam,
                    last_e_r_cam=last_e_r_cam,
                    locked_approach_camera_rotation_world=locked_approach_camera_rotation_world,
                    position_deadband=POSITION_DEADBAND,
                    position_soft_zone=POSITION_SOFT_ZONE,
                    rotation_deadband=ROTATION_DEADBAND,
                    rotation_soft_zone=ROTATION_SOFT_ZONE,
                    error_smoothing_alpha=ERROR_SMOOTHING_ALPHA,
                    target_motion_speed_threshold=TARGET_MOTION_SPEED_THRESHOLD,
                    max_tracking_q_dot=MAX_TRACKING_Q_DOT,
                    max_q_dot=MAX_Q_DOT,
                    reference_preview_gain=REFERENCE_PREVIEW_GAIN,
                    r_mj_camera_from_cv_camera=R_MJ_CAMERA_FROM_CV_CAMERA,
                    actuator_names=ACTUATOR_NAMES,
                    arm_dof_count=ARM_DOF_COUNT,
                    target_is_mocap=target_is_mocap,
                )

                if servo_result["success"]:
                    vis = servo_result["vis"]
                    last_q_dot = servo_result["last_q_dot"]
                    last_e_p_cam = servo_result["last_e_p_cam"]
                    last_e_r_cam = servo_result["last_e_r_cam"]
                    locked_approach_camera_rotation_world = servo_result[
                        "locked_approach_camera_rotation_world"
                    ]
                    control_applied = True
                    lost_tag_count = 0

                    if servo_result["phase"] == "lift":
                        conveyor_active = False
                        if servo_result["attached_target_offset_world"] is not None:
                            attached_target_offset_world = servo_result["attached_target_offset_world"]
                        if servo_result["attached_target_quat"] is not None:
                            attached_target_quat = servo_result["attached_target_quat"]

            if not control_applied:
                lost_tag_count += 1

                if grasp_state_machine.phase == "approach" and lost_tag_count >= BLIND_ATTACH_FRAMES:
                    grasp_state_machine.force_attach()
                    locked_approach_camera_rotation_world = None
                    if target_is_mocap:
                        target_motion.auto_mode = False
                        target_motion.manual_pos = current_target_pos.copy()
                        target_motion.manual_quat = target_motion.normalize_quaternion(
                            current_target_quat.copy()
                        )
                        attached_target_offset_world = current_target_pos - current_site_pos
                        attached_target_quat = current_target_quat.copy()
                    grasp_state_machine.start_lift(current_site_pos)

                if lost_tag_count <= LOST_TAG_HOLD_FRAMES:
                    q_dot_hold = np.clip(
                        last_q_dot * (LOST_TAG_DECAY ** lost_tag_count),
                        -MAX_TRACKING_Q_DOT,
                        MAX_TRACKING_Q_DOT,
                    )
                    env.apply_joint_velocity(ACTUATOR_NAMES, q_dot_hold)
                    last_q_dot = q_dot_hold.copy()
                    cv2.putText(
                        vis,
                        "TARGET LOST: BLIND APPROACH HOLD",
                        (10, HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 200, 255),
                        2,
                    )
                else:
                    env.zero_joint_velocity(ACTUATOR_NAMES)
                    last_q_dot = np.zeros(ARM_DOF_COUNT, dtype=np.float64)
                    last_e_p_cam[:] = 0.0
                    last_e_r_cam[:] = 0.0
                    grasp_state_machine.settled_count = 0

            renderer.show_camera_image(vis)

            key = cv2.waitKey(1) & 0xFF
            if target_is_mocap:
                toggled_auto_mode = target_motion.handle_key(key)
                if toggled_auto_mode and not target_motion.auto_mode:
                    target_motion.manual_pos = data.mocap_pos[target_mocap_id].copy()
                    target_motion.manual_quat = target_motion.normalize_quaternion(
                        data.mocap_quat[target_mocap_id].copy()
                    )

            if key == 27:
                break

            env.step(SIM_STEPS_PER_CONTROL)
            viewer.sync()
            time.sleep(GUI_SLEEP)

    renderer.close()


if __name__ == "__main__":
    main()
