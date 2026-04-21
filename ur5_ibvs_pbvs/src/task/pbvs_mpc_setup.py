import mujoco
import numpy as np
from pupil_apriltags import Detector

from src.config import (
    ACTUATOR_NAMES,
    ARM_DOF_COUNT,
    AUTO_MOVE_TARGET,
    CAMERA_NAME,
    CONVEYOR_ACTUATOR_NAME,
    CONVEYOR_ENABLED,
    DESIRED_STANDOFF,
    EE_FRAME_NAME,
    GRASP_TO_CAMERA_ROTATION,
    GRASP_TO_CAMERA_TRANSLATION,
    GRIPPER_ACTUATOR_NAME,
    GRIPPER_CLOSE_CTRL,
    GRIPPER_OPEN_CTRL,
    HEIGHT,
    MAX_Q_DOT,
    MAX_TRACKING_Q_DOT,
    PINOCCHIO_EE_OFFSET_LOCAL,
    PINOCCHIO_Q_SIGNS,
    PLACE_BOX_CENTER_WORLD,
    PLACE_RELEASE_HEIGHT,
    PREDICTION_HORIZON,
    R_MJ_CAMERA_FROM_CV_CAMERA,
    SIM_STEPS_PER_CONTROL,
    TAG_SIZE,
    TARGET_BODY_NAME,
    TARGET_MOTION_AMPLITUDE,
    TARGET_MOTION_FREQUENCY,
    TARGET_MOTION_PHASE,
    TARGET_ROTATION_STEP_RAD,
    TARGET_TRANSLATION_STEP,
    URDF_PATH,
    WIDTH,
)
from src.controllers.mpc_controller import MPCController
from src.controllers.pbvs_controller import PBVSController
from src.perception.camera_model import build_camera_matrix, build_zero_distortion
from src.perception.pose_estimator import AprilTagPoseEstimator
from src.perception.target_motion import TargetMotionController
from src.robotics.pinocchio_kinematics import PinocchioKinematics
from src.sim.mujoco_env import MujocoEnv
from src.sim.rendering import MujocoRenderer
from src.task.grasp_state_machine import GraspStateMachine
from src.task.pbvs_mpc_runtime import (
    get_gripper_ctrl_for_phase,
    get_target_home_pose,
    print_target_control_instructions,
    set_gripper_ctrl,
)
from src.utils.transforms import make_transform


def build_runtime(scene_xml):
    env = MujocoEnv(scene_xml)
    env.reset_to_home()

    model = env.model
    data = env.data
    env.initialize_intvelocity_actuators(ACTUATOR_NAMES)
    controller_dt = model.opt.timestep * SIM_STEPS_PER_CONTROL

    renderer = MujocoRenderer(model, data, CAMERA_NAME, width=WIDTH, height=HEIGHT)
    robot_kin = PinocchioKinematics(
        urdf_path=URDF_PATH,
        ee_frame_name=EE_FRAME_NAME,
        arm_dof_count=ARM_DOF_COUNT,
        q_signs=PINOCCHIO_Q_SIGNS,
        ee_offset_local=PINOCCHIO_EE_OFFSET_LOCAL,
    )
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

    last_q_dot = np.zeros(ARM_DOF_COUNT, dtype=np.float64)
    gripper_target_ctrl = get_gripper_ctrl_for_phase(
        "track",
        GRIPPER_OPEN_CTRL,
        GRIPPER_CLOSE_CTRL,
    )
    release_frames_remaining = 0
    lost_tag_count = 0
    attached_target_offset_world = np.zeros(3, dtype=np.float64)
    attached_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    locked_approach_camera_rotation_world = None
    last_e_p_cam = np.zeros(3, dtype=np.float64)
    last_e_r_cam = np.zeros(3, dtype=np.float64)
    conveyor_active = bool(CONVEYOR_ENABLED)
    target_body_id = env.get_body_id(TARGET_BODY_NAME)
    target_mocap_id = model.body_mocapid[target_body_id]
    target_is_mocap = target_mocap_id >= 0
    conveyor_actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        CONVEYOR_ACTUATOR_NAME,
    )
    print_target_control_instructions(target_is_mocap, CONVEYOR_ENABLED)
    target_home_pos, target_home_quat = get_target_home_pose(
        data,
        target_body_id,
        target_mocap_id,
        target_is_mocap,
    )
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
        settle_frames_required=10,
        attach_position_tol=0.014,
        attach_rotation_tol_rad=np.deg2rad(8.0),
        attach_settle_frames_required=2,
        dynamic_track_position_tol=0.035,
        dynamic_track_rotation_tol_rad=np.deg2rad(8.0),
        dynamic_track_settle_frames_required=6,
        dynamic_attach_position_tol=0.022,
        dynamic_attach_rotation_tol_rad=np.deg2rad(12.0),
        dynamic_attach_settle_frames_required=1,
        lift_offset_world=[0.0, 0.0, 0.08],
        place_offset_world=[0.0, 0.0, 0.0],
        place_position_tol=0.01,
    )
    home_qpos = env.get_joint_positions(ARM_DOF_COUNT)
    place_site_target_world = PLACE_BOX_CENTER_WORLD + np.array(
        [0.0, 0.0, PLACE_RELEASE_HEIGHT],
        dtype=np.float64,
    )
    set_gripper_ctrl(data, GRIPPER_ACTUATOR_NAME, gripper_target_ctrl)
    t_grasp_camera = make_transform(
        GRASP_TO_CAMERA_ROTATION,
        GRASP_TO_CAMERA_TRANSLATION,
    )

    return {
        "env": env,
        "model": model,
        "data": data,
        "controller_dt": controller_dt,
        "renderer": renderer,
        "robot_kin": robot_kin,
        "pose_estimator": pose_estimator,
        "pbvs_controller": pbvs_controller,
        "mpc_controller": mpc_controller,
        "tracking_mpc_controller": tracking_mpc_controller,
        "camera_id": camera_id,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "detector": detector,
        "last_q_dot": last_q_dot,
        "gripper_target_ctrl": gripper_target_ctrl,
        "release_frames_remaining": release_frames_remaining,
        "lost_tag_count": lost_tag_count,
        "attached_target_offset_world": attached_target_offset_world,
        "attached_target_quat": attached_target_quat,
        "locked_approach_camera_rotation_world": locked_approach_camera_rotation_world,
        "last_e_p_cam": last_e_p_cam,
        "last_e_r_cam": last_e_r_cam,
        "conveyor_active": conveyor_active,
        "target_body_id": target_body_id,
        "target_mocap_id": target_mocap_id,
        "target_is_mocap": target_is_mocap,
        "conveyor_actuator_id": conveyor_actuator_id,
        "target_motion": target_motion,
        "grasp_state_machine": grasp_state_machine,
        "home_qpos": home_qpos,
        "place_site_target_world": place_site_target_world,
        "t_grasp_camera": t_grasp_camera,
    }
