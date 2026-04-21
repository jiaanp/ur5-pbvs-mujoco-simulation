import numpy as np


SCENE_XML = "/home/adrian/ur5-pbvs-mujoco-simulation/ur5_ibvs_pbvs/model/scene_with_gripper.xml"
CAMERA_NAME = "end_effector_camera"
SITE_NAME = "attachment_site"
TARGET_BODY_NAME = "target"

WIDTH = 640
HEIGHT = 480

TAG_SIZE = 0.05
DESIRED_STANDOFF = 0.15
URDF_PATH = "/home/adrian/models/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf"
EE_FRAME_NAME = "tool0"
PINOCCHIO_Q_SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
PINOCCHIO_EE_OFFSET_LOCAL = np.array(
    [-0.02484611, -0.00510126, 0.04385891],
    dtype=np.float64,
)
GRASP_TO_CAMERA_TRANSLATION = np.array([0.0, 0.0, 0.0], dtype=np.float64)
GRASP_TO_CAMERA_ROTATION = np.eye(3, dtype=np.float64)
GRASP_YAW_ABOUT_TAG_NORMAL_RAD = 0.0
PLACE_BOX_CENTER_WORLD = np.array([-0.13, 0.72, 0.0], dtype=np.float64)
PLACE_RELEASE_HEIGHT = 0.30

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
HOME_MAX_Q_DOT = 2.0
RELEASE_HOLD_FRAMES = 20
ERROR_SMOOTHING_ALPHA = 0.45
CONVEYOR_ENABLED = True
CONVEYOR_SPEED = 0.35
CONVEYOR_ACTUATOR_NAME = "conveyor_drive"
MAX_TRANSPORT_Q_DOT = 3.0
MAX_PLACE_Q_DOT = 4.0

R_MJ_CAMERA_FROM_CV_CAMERA = np.diag([1.0, -1.0, -1.0])

GRIPPER_ACTUATOR_NAME = "2f85_ctrl"
GRIPPER_OPEN_CTRL = -60.0
GRIPPER_CLOSE_CTRL = 5.0
GRIPPER_CTRL_STEP = 2.0
