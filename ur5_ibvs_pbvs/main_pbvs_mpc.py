
import time

import cv2
import mujoco
import numpy as np
from pupil_apriltags import Detector

from src.controllers.mpc_controller import MPCController
from src.controllers.pbvs_controller import PBVSController, soften_error_vector
from src.perception.camera_model import build_camera_matrix, build_zero_distortion
from src.perception.pose_estimator import (
    AprilTagPoseEstimator,
    draw_apriltags,
    draw_pose_axes,
)
from src.perception.target_motion import TargetMotionController
from src.robotics.ur5_kinematics import UR5Kinematics
from src.sim.mujoco_env import MujocoEnv
from src.sim.rendering import MujocoRenderer
from src.task.grasp_state_machine import GraspStateMachine
from src.utils.transforms import (
    build_desired_tag_camera_transform,
    rvec_tvec_to_transform,
)


SCENE_XML = "/home/adrian/ur5-pbvs-mujoco-simulation/ur5_ibvs_pbvs/model/scene.xml"
CAMERA_NAME = "end_effector_camera"
SITE_NAME = "attachment_site"
TARGET_BODY_NAME = "target"

WIDTH = 640
HEIGHT = 480

TAG_SIZE = 0.10
DESIRED_STANDOFF = 0.15
PLACE_OFFSET_FROM_HOME_WORLD = np.array([0.0, -0.25, 0.0], dtype=np.float64)




ACTUATOR_NAMES = [
    "shoulder_pan_vel_init",
    "shoulder_lift_vel_init",
    "elbow_vel_init",
    "wrist_1_vel_init",
    "wrist_2_vel_init",
    "wrist_3_vel_init",
]
ARM_DOF_COUNT = len(ACTUATOR_NAMES)

POSITION_GAIN = 1.0
ROTATION_GAIN = 4.0
MAX_Q_DOT = 2.0
MAX_TRACKING_Q_DOT = 5.0
JACOBIAN_DAMPING = 0.02
GUI_SLEEP = 0.0
POSITION_DEADBAND = 0.004
POSITION_SOFT_ZONE = 0.03
ROTATION_DEADBAND = np.deg2rad(0.8)
ROTATION_SOFT_ZONE = np.deg2rad(6.0)
AUTO_MOVE_TARGET = False
TARGET_MOTION_AMPLITUDE = np.array([0.06, 0.04, 0.02], dtype=np.float64)
TARGET_MOTION_FREQUENCY = np.array([0.45, 0.80, 0.60], dtype=np.float64)
TARGET_MOTION_PHASE = np.array([0.0, 0.6, 1.1], dtype=np.float64)
TARGET_TRANSLATION_STEP = 0.01
TARGET_ROTATION_STEP_RAD = np.deg2rad(5.0)
TARGET_LINEAR_FEEDFORWARD_GAIN = 1.8
TARGET_ANGULAR_FEEDFORWARD_GAIN = 1.4
TARGET_MOTION_SPEED_THRESHOLD = 1e-3
PREDICTION_HORIZON = 60
REFERENCE_PREVIEW_GAIN = 1.6
SIM_STEPS_PER_CONTROL = 4
LOST_TAG_HOLD_FRAMES = 8
LOST_TAG_DECAY = 0.92
BLIND_ATTACH_FRAMES = 3
HOME_JOINT_KP = 2.0
HOME_JOINT_TOL = 0.03
HOME_MAX_Q_DOT = 1.5

# OpenCV 相机坐标系 -> MuJoCo 相机局部坐标系
R_MJ_CAMERA_FROM_CV_CAMERA = np.diag([1.0, -1.0, -1.0])


def build_error_state_world(e_p_cam, e_r_cam, camera_rotation_world):
    """
    把当前相机坐标系下的 6 维 PBVS 误差，转换到世界坐标系。
    这样 MPC 内部就可以统一使用世界系误差状态：
        x = [position_error_world, rotation_error_world]
    """
    e_p_mj = R_MJ_CAMERA_FROM_CV_CAMERA @ e_p_cam
    e_r_mj = R_MJ_CAMERA_FROM_CV_CAMERA @ e_r_cam

    e_p_world = camera_rotation_world @ e_p_mj
    e_r_world = camera_rotation_world @ e_r_mj

    return np.concatenate([e_p_world, e_r_world], axis=0)


def build_lift_error_state_world(lift_position_error_world):
    """
    lift 阶段不再依赖视觉误差，直接在世界系给一个向上抬升的位置误差。
    姿态先保持不变，因此旋转误差置零。
    """
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
    """
    根据目标本身的运动，构造理想相机在世界系下的前馈速度。
    当目标在动时，理想相机也应该跟着一起平移/转动，而不是只靠误差滞后修正。
    """
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
):
    """
    生成 tracking MPC 用的未来参考误差轨迹。

    当前 MPC 的状态定义是“世界系误差”，而目标未来运动会让误差自然漂移。
    这里用一个最小可用近似：
    - 假设目标在预测域内保持当前线速度/角速度
    - 理想相机也应跟随目标做同样的未来运动
    - 因此参考误差轨迹取为“未来目标诱发误差”的负值
      这样 MPC 会主动去抵消这部分未来漂移
    """
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
        reference[i] = -REFERENCE_PREVIEW_GAIN * step_scale * ff_velocity_world

    return reference


def main():
    env = MujocoEnv(SCENE_XML)
    env.reset_to_home()

    model = env.model
    data = env.data

    env.initialize_intvelocity_actuators(ACTUATOR_NAMES)

    controller_dt = model.opt.timestep * SIM_STEPS_PER_CONTROL

    renderer = MujocoRenderer(model, data, CAMERA_NAME, width=WIDTH, height=HEIGHT)
    robot_kin = UR5Kinematics(model, data, SITE_NAME, arm_dof_count=ARM_DOF_COUNT)
    pose_estimator = AprilTagPoseEstimator(TAG_SIZE)
    pbvs_controller = PBVSController(R_MJ_CAMERA_FROM_CV_CAMERA)
    mpc_controller = MPCController(
        horizon=PREDICTION_HORIZON,
        dt=controller_dt,
        q_weights=[18.0, 18.0, 18.0, 9.0, 9.0, 9.0],
        r_weights=[0.02, 0.02, 0.02, 0.012, 0.012, 0.012],
        du_weights=[0.10, 0.10, 0.10, 0.06, 0.06, 0.06],
        qdot_limit=MAX_Q_DOT,
    )
    tracking_mpc_controller = MPCController(
        horizon=PREDICTION_HORIZON,
        dt=controller_dt,
        q_weights=[36.0, 36.0, 36.0, 16.0, 16.0, 16.0],
        r_weights=[0.006, 0.006, 0.006, 0.004, 0.004, 0.004],
        du_weights=[0.025, 0.025, 0.025, 0.015, 0.015, 0.015],
        qdot_limit=MAX_TRACKING_Q_DOT,
    )

    camera_id = env.get_camera_id(CAMERA_NAME)
    camera_matrix = build_camera_matrix(model, camera_id, WIDTH, HEIGHT)
    dist_coeffs = build_zero_distortion()

    detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        refine_edges=1,
    )

    print("camera_matrix =")
    print(camera_matrix)
    print("Target control:")
    print("  Focus the 'End-Effector Camera' window, then use W/S A/D R/F to move the tag.")
    print("  Use I/K J/L U/O to rotate the tag, SPACE to reset, M to toggle auto motion.")

    last_q_dot = np.zeros(ARM_DOF_COUNT, dtype=np.float64)
    lost_tag_count = 0
    attached_target_offset_world = np.zeros(3, dtype=np.float64)
    attached_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    target_body_id = env.get_body_id(TARGET_BODY_NAME)
    target_mocap_id = model.body_mocapid[target_body_id]
    if target_mocap_id < 0:
        raise ValueError(f"body is not mocap-controlled: {TARGET_BODY_NAME}")
    target_home_pos = data.mocap_pos[target_mocap_id].copy()
    target_home_quat = data.mocap_quat[target_mocap_id].copy()
    target_motion = TargetMotionController(
        target_home_pos,
        target_home_quat,
        auto_move=AUTO_MOVE_TARGET,
        motion_amplitude=TARGET_MOTION_AMPLITUDE,
        motion_frequency=TARGET_MOTION_FREQUENCY,
        motion_phase=TARGET_MOTION_PHASE,
        translation_step=TARGET_TRANSLATION_STEP,
        rotation_step_rad=TARGET_ROTATION_STEP_RAD,
    )
    grasp_state_machine = GraspStateMachine(
        track_standoff=DESIRED_STANDOFF,
        approach_standoff=0.015,
        track_position_tol=0.02,
        track_rotation_tol_rad=np.deg2rad(3.0),
        settle_frames_required=4,
        attach_position_tol=0.014,
        attach_rotation_tol_rad=np.deg2rad(8.0),
        attach_settle_frames_required=2,
        dynamic_track_position_tol=0.035,
        dynamic_track_rotation_tol_rad=np.deg2rad(8.0),
        dynamic_track_settle_frames_required=2,
        dynamic_attach_position_tol=0.022,
        dynamic_attach_rotation_tol_rad=np.deg2rad(12.0),
        dynamic_attach_settle_frames_required=1,
        lift_offset_world=[0.0, 0.0, 0.08],
        place_offset_world=PLACE_OFFSET_FROM_HOME_WORLD,
        place_position_tol=0.01,
    )
    home_qpos = env.get_joint_positions(ARM_DOF_COUNT)
    home_site_pos, _ = env.get_site_pose(SITE_NAME)
    place_site_target_world = home_site_pos + PLACE_OFFSET_FROM_HOME_WORLD

    with renderer.create_viewer() as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        while viewer.is_running():
            target_pos, target_quat = target_motion.get_target_pose(data.time)
            env.forward()
            current_site_pos, _ = env.get_site_pose(SITE_NAME)

            if grasp_state_machine.attached:
                target_pos = current_site_pos + attached_target_offset_world
                target_quat = attached_target_quat.copy()

            env.set_target_pose(TARGET_BODY_NAME, target_pos, target_quat)
            env.forward()
            current_site_pos, _ = env.get_site_pose(SITE_NAME)

            current_target_pos = data.mocap_pos[target_mocap_id].copy()
            current_target_quat = data.mocap_quat[target_mocap_id].copy()
            target_linear_world, target_angular_world = target_motion.estimate_target_spatial_velocity_world(
                current_target_pos,
                current_target_quat,
                controller_dt,
            )
            t_tag_camera_desired = grasp_state_machine.get_desired_tag_camera_transform()
            active_standoff = grasp_state_machine.get_desired_standoff()
            target_motion_speed = np.linalg.norm(target_linear_world) + active_standoff * np.linalg.norm(target_angular_world)
            tracking_target = target_motion_speed > TARGET_MOTION_SPEED_THRESHOLD

            bgr = renderer.render_camera_bgr()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            vis = draw_apriltags(bgr, tags)
            vis = target_motion.draw_overlay(vis)
            vis = grasp_state_machine.draw_overlay(vis)

            control_applied = False

            if grasp_state_machine.phase == "lift" and grasp_state_machine.attached:
                lift_position_error_world, lift_done = grasp_state_machine.get_lift_error_world(
                    current_site_pos
                )
                error_state_world = build_lift_error_state_world(lift_position_error_world)
                J = robot_kin.compute_site_jacobian()

                q_dot_lift = mpc_controller.solve(
                    error_state_world,
                    J,
                    last_q_dot=last_q_dot,
                )
                q_dot_lift = np.clip(q_dot_lift, -MAX_Q_DOT, MAX_Q_DOT)

                if lift_done:
                    q_dot_lift[:] = 0.0

                env.apply_joint_velocity(ACTUATOR_NAMES, q_dot_lift)
                last_q_dot = q_dot_lift.copy()
                control_applied = True

                cv2.putText(
                    vis,
                    "LIFTING ATTACHED TARGET",
                    (10, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                print("phase = lift")
                print("lift_position_error_world =", np.round(lift_position_error_world, 4))
                print("q_dot_lift =", np.round(q_dot_lift, 4))

                if lift_done:
                    grasp_state_machine.start_place(
                        current_site_pos,
                        place_target_pos_world=place_site_target_world,
                    )

            if grasp_state_machine.phase == "place" and grasp_state_machine.attached:
                place_position_error_world, place_done = grasp_state_machine.get_place_error_world(
                    current_site_pos
                )
                error_state_world = build_lift_error_state_world(place_position_error_world)
                J = robot_kin.compute_site_jacobian()

                q_dot_place = mpc_controller.solve(
                    error_state_world,
                    J,
                    last_q_dot=last_q_dot,
                )
                q_dot_place = np.clip(q_dot_place, -MAX_Q_DOT, MAX_Q_DOT)

                if place_done:
                    q_dot_place[:] = 0.0

                env.apply_joint_velocity(ACTUATOR_NAMES, q_dot_place)
                last_q_dot = q_dot_place.copy()
                control_applied = True

                cv2.putText(
                    vis,
                    "PLACING ATTACHED TARGET",
                    (10, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

                print("phase = place")
                print("place_position_error_world =", np.round(place_position_error_world, 4))
                print("q_dot_place =", np.round(q_dot_place, 4))

                if place_done:
                    target_motion.auto_mode = False
                    target_motion.manual_pos = current_target_pos.copy()
                    target_motion.manual_quat = target_motion.normalize_quaternion(
                        current_target_quat.copy()
                    )
                    grasp_state_machine.start_home()

            if grasp_state_machine.phase == "home":
                current_qpos = env.get_joint_positions(ARM_DOF_COUNT)
                q_error = home_qpos - current_qpos
                q_dot_home = np.clip(HOME_JOINT_KP * q_error, -HOME_MAX_Q_DOT, HOME_MAX_Q_DOT)
                home_done = np.linalg.norm(q_error) < HOME_JOINT_TOL

                if home_done:
                    q_dot_home[:] = 0.0
                    grasp_state_machine.mark_done()

                env.apply_joint_velocity(ACTUATOR_NAMES, q_dot_home)
                last_q_dot = q_dot_home.copy()
                control_applied = True

                cv2.putText(
                    vis,
                    "RETURNING HOME",
                    (10, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )

                print("phase = home")
                print("q_error =", np.round(q_error, 4))
                print("q_dot_home =", np.round(q_dot_home, 4))

            if grasp_state_machine.phase == "done":
                env.zero_joint_velocity(ACTUATOR_NAMES)
                last_q_dot = np.zeros(ARM_DOF_COUNT, dtype=np.float64)
                control_applied = True

                cv2.putText(
                    vis,
                    "TASK COMPLETE",
                    (10, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if (not control_applied) and len(tags) > 0:
                tag = tags[0]

                success, rvec, tvec = pose_estimator.estimate_pose(
                    tag,
                    camera_matrix,
                    dist_coeffs,
                )

                if success:
                    vis = draw_pose_axes(
                        vis,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        axis_length=0.05,
                    )

                    T_camera_tag = rvec_tvec_to_transform(rvec, tvec)

                    e_p_cam_raw, e_r_cam_raw, T_tag_camera = pbvs_controller.compute_pbvs_error_in_camera_frame(
                        T_camera_tag,
                        t_tag_camera_desired,
                    )
                    e_p_cam = soften_error_vector(
                        e_p_cam_raw,
                        POSITION_DEADBAND,
                        POSITION_SOFT_ZONE,
                    )
                    e_r_cam = soften_error_vector(
                        e_r_cam_raw,
                        ROTATION_DEADBAND,
                        ROTATION_SOFT_ZONE,
                    )

                    # 这里必须使用“真实相机”的姿态，而不是 attachment_site 的姿态。
                    # PBVS 误差 e_p_cam / e_r_cam 是在相机坐标系下定义的，
                    # 如果把 site 姿态误当成相机姿态，会把控制速度映射到错误方向，
                    # 导致机械臂离目标越来越远。
                    camera_rotation_world = data.cam_xmat[camera_id].reshape(3, 3)

                    error_state_world = build_error_state_world(
                        e_p_cam,
                        e_r_cam,
                        camera_rotation_world,
                    )

                    J = robot_kin.compute_site_jacobian()
                    active_mpc_controller = tracking_mpc_controller if tracking_target else mpc_controller
                    active_max_q_dot = MAX_TRACKING_Q_DOT if tracking_target else MAX_Q_DOT
                    reference_trajectory = build_reference_error_trajectory(
                        current_target_quat,
                        target_linear_world,
                        target_angular_world,
                        t_tag_camera_desired,
                        active_mpc_controller.horizon,
                        controller_dt,
                    )

                    q_dot_mpc = active_mpc_controller.solve(
                        error_state_world,
                        J,
                        last_q_dot=last_q_dot,
                        reference_trajectory=reference_trajectory,
                    )

                    # 现在把“未来目标运动”主要交给 tracking MPC 自己处理，
                    # 不再在外面额外叠加一层 q_dot 前馈，避免双重补偿后仍然滞后或发散。
                    q_dot_ff = np.zeros(ARM_DOF_COUNT, dtype=np.float64)
                    q_dot = q_dot_mpc

                    q_dot = np.clip(q_dot, -active_max_q_dot, active_max_q_dot)

                    env.apply_joint_velocity(ACTUATOR_NAMES, q_dot)
                    last_q_dot = q_dot.copy()
                    control_applied = True
                    lost_tag_count = 0
                    phase = grasp_state_machine.update(
                        np.linalg.norm(e_p_cam_raw),
                        np.linalg.norm(e_r_cam_raw),
                        target_motion_speed=target_motion_speed,
                        motion_threshold=TARGET_MOTION_SPEED_THRESHOLD,
                    )

                    if phase == "lift":
                        # 吸附版：进入 lift 阶段时，先冻结目标，后面再接真正的抬升动作
                        target_motion.auto_mode = False
                        target_motion.manual_pos = current_target_pos.copy()
                        target_motion.manual_quat = current_target_quat.copy()
                        attached_target_offset_world = current_target_pos - current_site_pos
                        attached_target_quat = current_target_quat.copy()
                        grasp_state_machine.start_lift(current_site_pos)

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
                    print("q_dot_ff =", np.round(q_dot_ff, 4))
                    print("q_dot =", np.round(q_dot, 4))

            if not control_applied:
                lost_tag_count += 1

                if grasp_state_machine.phase == "approach" and lost_tag_count >= BLIND_ATTACH_FRAMES:
                    grasp_state_machine.force_attach()
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
                    grasp_state_machine.settled_count = 0

            renderer.show_camera_image(vis)

            key = cv2.waitKey(1) & 0xFF
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
