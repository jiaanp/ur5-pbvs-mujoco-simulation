import time

import cv2
import mujoco
import numpy as np
from pupil_apriltags import Detector

from src.controllers.pbvs_controller import PBVSController, soften_error_vector
from src.perception.camera_model import build_camera_matrix, build_zero_distortion
from src.perception.pose_estimator import (
    AprilTagPoseEstimator,
    draw_apriltags,
    draw_pose_axes,
)
#from src.robotics.ur5_kinematics import UR5Kinematics
from src.robotics.pinocchio_kinematics import PinocchioKinematics

from src.sim.mujoco_env import MujocoEnv
from src.sim.rendering import MujocoRenderer
from src.utils.transforms import (
    build_desired_tag_camera_transform,
    rvec_tvec_to_transform,
)


SCENE_XML = "/home/adrian/ur5-pbvs-mujoco-simulation/ur5_ibvs_pbvs/model/scene.xml"
CAMERA_NAME = "end_effector_camera"
SITE_NAME = "attachment_site"

WIDTH = 640
HEIGHT = 480

TAG_SIZE = 0.10
DESIRED_STANDOFF = 0.15
URDF_PATH = "/home/adrian/models/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf"
EE_FRAME_NAME = "tool0"
PINOCCHIO_Q_SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
PINOCCHIO_EE_OFFSET_LOCAL = np.array(
    [-0.02484611, -0.00510126, -0.05614109],
    dtype=np.float64,
)





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
MAX_Q_DOT = 1.2
JACOBIAN_DAMPING = 0.02
GUI_SLEEP = 0.002
POSITION_DEADBAND = 0.004
POSITION_SOFT_ZONE = 0.03
ROTATION_DEADBAND = np.deg2rad(0.8)
ROTATION_SOFT_ZONE = np.deg2rad(6.0)

# OpenCV 相机坐标系 -> MuJoCo 相机局部坐标系
R_MJ_CAMERA_FROM_CV_CAMERA = np.diag([1.0, -1.0, -1.0])


def main():
    env = MujocoEnv(SCENE_XML)
    env.reset_to_home()

    model = env.model
    data = env.data

    env.initialize_intvelocity_actuators(ACTUATOR_NAMES)

    renderer = MujocoRenderer(model, data, CAMERA_NAME, width=WIDTH, height=HEIGHT)
    #robot_kin = UR5Kinematics(model, data, SITE_NAME, arm_dof_count=ARM_DOF_COUNT)
    robot_kin = PinocchioKinematics(
        urdf_path=URDF_PATH,
        ee_frame_name=EE_FRAME_NAME,
        arm_dof_count=ARM_DOF_COUNT,
        q_signs=PINOCCHIO_Q_SIGNS,
        ee_offset_local=PINOCCHIO_EE_OFFSET_LOCAL,
    )
    pose_estimator = AprilTagPoseEstimator(TAG_SIZE)
    pbvs_controller = PBVSController(R_MJ_CAMERA_FROM_CV_CAMERA)

    camera_id = env.get_camera_id(CAMERA_NAME)
    camera_matrix = build_camera_matrix(model, camera_id, WIDTH, HEIGHT)
    dist_coeffs = build_zero_distortion()

    detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        refine_edges=1,
    )

    T_tag_camera_desired = build_desired_tag_camera_transform(DESIRED_STANDOFF)

    print("camera_matrix =")
    print(camera_matrix)
    print("T_tag_camera_desired =")
    print(np.round(T_tag_camera_desired, 4))

    with renderer.create_viewer() as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        while viewer.is_running():
            bgr = renderer.render_camera_bgr()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            vis = draw_apriltags(bgr, tags)

            control_applied = False

            if len(tags) > 0:
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
                        T_tag_camera_desired,
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

                    v_e_desired_world = pbvs_controller.compute_desired_ee_velocity_world(
                        e_p_cam,
                        e_r_cam,
                        camera_rotation_world,
                        POSITION_GAIN,
                        ROTATION_GAIN,
                    )

                    # q_dot, J = robot_kin.compute_joint_velocity_from_ee_velocity(
                    #     v_e_desired_world,
                    #     damping=JACOBIAN_DAMPING,
                    # )
                    q_dot, J = robot_kin.compute_joint_velocity_from_ee_velocity(
                        env.get_joint_positions(ARM_DOF_COUNT),
                        v_e_desired_world,
                        damping=JACOBIAN_DAMPING,
                    )

                    q_dot = np.clip(q_dot, -MAX_Q_DOT, MAX_Q_DOT)

                    env.apply_joint_velocity(ACTUATOR_NAMES, q_dot)
                    control_applied = True

                    print("e_p_cam_raw =", np.round(e_p_cam_raw, 4))
                    print("e_r_cam_raw =", np.round(e_r_cam_raw, 4))
                    print("e_p_cam =", np.round(e_p_cam, 4))
                    print("e_r_cam =", np.round(e_r_cam, 4))
                    print("q_dot =", np.round(q_dot, 4))

            if not control_applied:
                env.zero_joint_velocity(ACTUATOR_NAMES)

            renderer.show_camera_image(vis)

            key = cv2.waitKey(1)
            if key == 27:
                break

            env.step(1)
            viewer.sync()
            time.sleep(GUI_SLEEP)

    renderer.close()


if __name__ == "__main__":
    main()
