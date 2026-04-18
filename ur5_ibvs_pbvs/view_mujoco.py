import math
import time

import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np
from pupil_apriltags import Detector


# 你的 MuJoCo 场景文件
SCENE_XML = "/home/adrian/ur5-pbvs-mujoco-simulation/ur5_ibvs_pbvs/model/scene_with_gripper.xml"

# 末端相机名称
CAMERA_NAME = "end_effector_camera"

# 相机图像分辨率
WIDTH = 640
HEIGHT = 480

# AprilTag 实际边长，单位：米
TAG_SIZE = 0.10
DESIRED_STANDOFF = 0.20
SITE_NAME = "attachment_site"
TARGET_BODY_NAME = "target"
ACTUATOR_NAMES = [
    "shoulder_pan_vel_init",
    "shoulder_lift_vel_init",
    "elbow_vel_init",
    "wrist_1_vel_init",
    "wrist_2_vel_init",
    "wrist_3_vel_init",
]
ARM_DOF_COUNT = len(ACTUATOR_NAMES)
MAX_Q_DOT = 2.4
MAX_TRACKING_Q_DOT = 3.4
POSITION_GAIN = 2.0
ROTATION_GAIN = 8.0
TRACKING_POSITION_GAIN = 2.4
TRACKING_ROTATION_GAIN = 9.0
TARGET_LINEAR_FEEDFORWARD_GAIN = 1.8
TARGET_ANGULAR_FEEDFORWARD_GAIN = 1.4
TARGET_MOTION_SPEED_THRESHOLD = 1e-3
JACOBIAN_DAMPING = 0.02
LOST_TAG_HOLD_FRAMES = 15
LOST_TAG_DECAY = 0.92
LOST_TARGET_BACKOFF_SPEED = 0.18
LOST_TARGET_RECOVERY_Q_DOT = 2.2
R_MJ_CAMERA_FROM_CV_CAMERA = np.diag([1.0, -1.0, -1.0])
GUI_SLEEP = 0.0
SIM_STEPS_PER_FRAME = 2
DEBUG_PRINT_EVERY = 20
AUTO_MOVE_TARGET = False
TARGET_MOTION_AMPLITUDE = np.array([0.06, 0.04, 0.02], dtype=np.float64)
TARGET_MOTION_FREQUENCY = np.array([0.45, 0.80, 0.60], dtype=np.float64)
TARGET_MOTION_PHASE = np.array([0.0, 0.6, 1.1], dtype=np.float64)
TARGET_TRANSLATION_STEP = 0.01
TARGET_ROTATION_STEP_RAD = math.radians(5.0)
POSITION_DEADBAND = 0.004
POSITION_SOFT_ZONE = 0.03
ROTATION_DEADBAND = math.radians(0.8)
ROTATION_SOFT_ZONE = math.radians(6.0)
NEAR_GOAL_POSITION_TOL = 0.012
NEAR_GOAL_ROTATION_TOL = math.radians(1.8)
SETTLE_FRAMES_REQUIRED = 4
Q_DOT_SMOOTH_ALPHA = 0.80
Q_DOT_SMOOTH_ALPHA_NEAR_GOAL = 0.30
NEAR_GOAL_MAX_Q_DOT = 0.8



def build_camera_matrix(model, camera_id, width, height):
    """
    根据 MuJoCo 相机的 fovy 和图像尺寸，近似构造相机内参矩阵。
    """
    fovy_deg = model.cam_fovy[camera_id]
    fy = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0

    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return camera_matrix


def draw_apriltags(image, tags):
    """
    在图像上绘制 AprilTag 检测结果：
    - 四条边
    - 四个角点
    - 中心点
    - tag id
    """
    output = image.copy()

    for tag in tags:
        corners = np.array(tag.corners, dtype=np.int32)
        center = tuple(np.array(tag.center, dtype=np.int32))

        # 画四条边
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(output, p1, p2, (0, 255, 0), 2)

        # 画四个角点
        for i, corner in enumerate(corners):
            point = tuple(corner)
            cv2.circle(output, point, 4, (0, 0, 255), -1)
            cv2.putText(
                output,
                str(i),
                (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        # 画中心点
        cv2.circle(output, center, 5, (255, 0, 0), -1)

        # 显示 tag id
        cv2.putText(
            output,
            f"id={tag.tag_id}",
            (center[0] + 50, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    return output


def estimate_tag_pose(tag, camera_matrix, dist_coeffs, tag_size):
    """
    根据 AprilTag 的四个角点，用 solvePnP 估计 tag 相对相机的位姿。
    """
    half = tag_size / 2.0

    # 目标坐标系下的四个角点，顺序要和 tag.corners 对齐
    object_points = np.array(
        [
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )

    image_points = np.array(tag.corners, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    return success, rvec, tvec


def draw_pose_axes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length=0.05):
    """
    在图像上绘制 tag 坐标轴：
    - X 轴红色
    - Y 轴绿色
    - Z 轴蓝色
    """
    axis_points_3d = np.array(
        [
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, -axis_length],
        ],
        dtype=np.float32,
    )

    image_points, _ = cv2.projectPoints(
        axis_points_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    image_points = image_points.reshape(-1, 2).astype(np.int32)

    # 坐标轴原点就是 tvec 对应的投影点
    origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    origin_2d, _ = cv2.projectPoints(
        origin_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    origin = tuple(origin_2d.reshape(-1, 2).astype(np.int32)[0])

    # X red
    cv2.line(image, origin, tuple(image_points[0]), (0, 0, 255), 2)
    # Y green
    cv2.line(image, origin, tuple(image_points[1]), (0, 255, 0), 2)
    # Z blue
    cv2.line(image, origin, tuple(image_points[2]), (255, 0, 0), 2)

    return image



def rvec_tvec_to_transform(rvec, tvec):
    """
    把 OpenCV 的 rvec / tvec 转成 4x4 齐次变换矩阵。
    返回的是：目标坐标系相对相机坐标系的位姿 T_camera_tag
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = tvec.ravel()

    return transform



def make_transform(rotation_matrix, translation_vector):
    """
    根据旋转矩阵和平移向量构造 4x4 齐次变换矩阵。
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector
    return transform


def invert_transform(transform):
    """
    计算 4x4 齐次变换矩阵的逆。
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def build_desired_tag_camera_transform(standoff):
    """
    构造期望的 T_tag_camera：
    - 相机位于标签法线方向 standoff 米处
    - 相机正对 tag

    根据当前 solvePnP 的点定义和检测结果，当前“正面看向标签”时：
    - T_camera_tag 的旋转接近单位阵
    - 相机位于 tag 坐标系的 -Z 方向

    因此期望相机位姿应定义为：
    - R_tag_camera_desired = I
    - p_tag_camera_desired = [0, 0, -standoff]
    """
    desired_rotation = np.eye(3, dtype=np.float64)
    desired_translation = np.array([0.0, 0.0, -standoff], dtype=np.float64)
    return make_transform(desired_rotation, desired_translation)


def rotation_error_vector(R_current, R_desired):
    """
    计算旋转误差向量。
    这是 PBVS 里常用的一种姿态误差表示方式。
    """
    return 0.5 * (
        np.cross(R_desired[:, 0], R_current[:, 0]) +
        np.cross(R_desired[:, 1], R_current[:, 1]) +
        np.cross(R_desired[:, 2], R_current[:, 2])
    )


def compute_pose_error(T_current, T_desired):
    """
    根据当前位姿和期望位姿，计算：
    - 位置误差 e_p
    - 姿态误差 e_r
    """
    p_current = T_current[:3, 3]
    p_desired = T_desired[:3, 3]

    R_current = T_current[:3, :3]
    R_desired = T_desired[:3, :3]

    # 位置误差
    e_p = p_current - p_desired

    # 姿态误差
    e_r = rotation_error_vector(R_current, R_desired)

    return e_p, e_r


def compute_pbvs_error_in_camera_frame(T_camera_tag, T_tag_camera_desired):
    """
    基于当前 tag 位姿和期望相机位姿，计算“当前相机坐标系”下的 PBVS 误差。

    返回：
    - e_p_cam: 当前位置误差，在当前相机坐标系下表示
    - e_r_cam: 当前姿态到期望姿态的旋转误差，在当前相机坐标系下表示
    """
    T_tag_camera = invert_transform(T_camera_tag)

    R_tc = T_tag_camera[:3, :3]
    p_tc = T_tag_camera[:3, 3]

    R_tc_des = T_tag_camera_desired[:3, :3]
    p_tc_des = T_tag_camera_desired[:3, 3]

    # 先在标签坐标系下做差，再转到当前相机坐标系。
    p_err_tag = p_tc_des - p_tc
    e_p_cam = R_tc.T @ p_err_tag

    # 当前相机到期望相机的相对旋转，表达在当前相机坐标系。
    R_err_cam = R_tc.T @ R_tc_des
    e_r_cam, _ = cv2.Rodrigues(R_err_cam)
    e_r_cam = e_r_cam.ravel()

    return e_p_cam, e_r_cam, T_tag_camera


def compute_desired_ee_velocity_world(e_p_cam, e_r_cam, camera_rotation_world, k_p_pos=1.0, k_p_rot=1.0):
    """
    根据相机坐标系下的 PBVS 误差，计算并转换到世界坐标系下的末端期望速度。

    MuJoCo 的 Jacobian 对应的是世界坐标系下的线速度/角速度，
    所以这里要先把相机系速度转到世界系。
    """
    # solvePnP 给出的位姿和误差都在 OpenCV 相机坐标系下：
    # x right, y down, z forward。
    # MuJoCo 相机局部坐标更接近 OpenGL：
    # x right, y up, z backward。
    # 因此把相机系速度送入 MuJoCo Jacobian 前，需要做一次坐标变换。
    v_linear_cv = k_p_pos * e_p_cam
    v_angular_cv = k_p_rot * e_r_cam

    v_linear_mj = R_MJ_CAMERA_FROM_CV_CAMERA @ v_linear_cv
    v_angular_mj = R_MJ_CAMERA_FROM_CV_CAMERA @ v_angular_cv

    v_linear_world = camera_rotation_world @ v_linear_mj
    v_angular_world = camera_rotation_world @ v_angular_mj

    return np.concatenate([v_linear_world, v_angular_world], axis=0)


def compute_lost_target_recovery_velocity_world(camera_rotation_world, backoff_speed=0.1):
    """
    丢失目标后的恢复策略：
    沿 OpenCV 相机坐标系的负 z 方向后退一点，扩大视野，尝试重新看到标签。
    """
    v_linear_cv = np.array([0.0, 0.0, -backoff_speed], dtype=np.float64)
    v_angular_cv = np.zeros(3, dtype=np.float64)

    v_linear_mj = R_MJ_CAMERA_FROM_CV_CAMERA @ v_linear_cv
    v_angular_mj = R_MJ_CAMERA_FROM_CV_CAMERA @ v_angular_cv

    v_linear_world = camera_rotation_world @ v_linear_mj
    v_angular_world = camera_rotation_world @ v_angular_mj

    return np.concatenate([v_linear_world, v_angular_world], axis=0)


def soften_error_vector(vector, deadband, soft_zone):
    """
    给误差加一个“软死区”：
    - 很小的误差直接当作 0
    - 靠近目标时逐渐减小误差幅值
    这样能明显减少到位后的来回抖动。
    """
    norm = np.linalg.norm(vector)
    if norm <= deadband:
        return np.zeros_like(vector)
    if norm >= soft_zone:
        return vector

    scale = (norm - deadband) / max(soft_zone - deadband, 1e-9)
    return vector * scale


def blend_joint_velocity(previous_q_dot, current_q_dot, alpha):
    """
    对关节速度命令做一阶平滑，减少 solvePnP 抖动直接打到执行器上。
    alpha 越小，平滑越强。
    """
    return (1.0 - alpha) * previous_q_dot + alpha * current_q_dot


def compute_joint_velocity_from_ee_velocity(model, data, site_name, v_e_desired_world, damping=0.05):
    """
    用 MuJoCo 的 site Jacobian，把期望末端速度转成关节速度。

    参数：
    - site_name: 末端参考点，这里建议用 attachment_site
    - v_e_desired_world: 6维末端速度 [vx, vy, vz, wx, wy, wz]，在世界坐标系下
    - damping: 阻尼伪逆系数

    返回：
    - q_dot: 关节速度
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"cannot find site: {site_name}")

    # 线速度 Jacobian 和角速度 Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    # 拼成 6xnv Jacobian。
    # 当前模型在 UR5 末端后又挂了 Robotiq 夹爪，所以 nv 已经不再是 6。
    # 但视觉伺服只控制 UR5 的前 6 个关节，夹爪自由度不参与 PBVS 逆解。
    J_full = np.vstack([jacp, jacr])
    J = J_full[:, :ARM_DOF_COUNT]

    # 阻尼伪逆：J^T (J J^T + λ^2 I)^-1
    JT = J.T
    q_dot = JT @ np.linalg.inv(J @ JT + (damping ** 2) * np.eye(6)) @ v_e_desired_world

    return q_dot, J

def initialize_intvelocity_actuators(model, data, actuator_names):
    """
    初始化 intvelocity 执行器：
    - 清零速度控制输入
    - 把内部积分状态 data.act 同步到当前关节角
    """
    for name in actuator_names:
        data.actuator(name).ctrl = 0.0

    if getattr(data, "act", None) is not None and data.act.size >= model.na:
        for name in actuator_names:
            actuator_id = model.actuator(name).id
            joint_id = model.actuator_trnid[actuator_id][0]
            qpos_adr = model.jnt_qposadr[joint_id]
            data.act[actuator_id] = data.qpos[qpos_adr]


def apply_joint_velocity_directly(data, actuator_names, q_dot):
    """
    直接把求解得到的关节速度写到 intvelocity actuator。
    """
    for i, name in enumerate(actuator_names):
        data.actuator(name).ctrl = float(q_dot[i])


def normalize_quaternion(quat):
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def axis_angle_to_quaternion(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / axis_norm
    half = angle_rad * 0.5
    sin_half = math.sin(half)
    return np.array(
        [math.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=np.float64,
    )


def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation_matrix(quat):
    quat = normalize_quaternion(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def set_target_pose(data, mocap_id, pos, quat):
    data.mocap_pos[mocap_id] = pos
    data.mocap_quat[mocap_id] = normalize_quaternion(quat)


def update_moving_target(data, mocap_id, home_pos, home_quat, sim_time):
    """
    让 AprilTag 目标沿平滑的 3D 轨迹移动，便于验证 PBVS 是否能持续伺服。
    这里先只做平移，保持标签朝向不变。
    """
    offset = TARGET_MOTION_AMPLITUDE * np.sin(TARGET_MOTION_FREQUENCY * sim_time + TARGET_MOTION_PHASE)
    set_target_pose(data, mocap_id, home_pos + offset, home_quat)


def apply_manual_target_control(key, target_pos, target_quat, home_pos, home_quat):
    """
    使用 OpenCV 窗口按键手动控制标签位置和姿态。
    返回更新后的 (target_pos, target_quat, toggled_auto_mode)。
    """
    toggled_auto_mode = False

    if key in (-1, 255):
        return target_pos, target_quat, toggled_auto_mode

    # 平移：A/D -> X, W/S -> Y, R/F -> Z
    if key == ord("a"):
        target_pos[0] -= TARGET_TRANSLATION_STEP
    elif key == ord("d"):
        target_pos[0] += TARGET_TRANSLATION_STEP
    elif key == ord("w"):
        target_pos[1] += TARGET_TRANSLATION_STEP
    elif key == ord("s"):
        target_pos[1] -= TARGET_TRANSLATION_STEP
    elif key == ord("r"):
        target_pos[2] += TARGET_TRANSLATION_STEP
    elif key == ord("f"):
        target_pos[2] -= TARGET_TRANSLATION_STEP
    # 旋转：I/K pitch, J/L yaw, U/O roll
    elif key == ord("i"):
        delta = axis_angle_to_quaternion([1.0, 0.0, 0.0], TARGET_ROTATION_STEP_RAD)
        target_quat = multiply_quaternions(delta, target_quat)
    elif key == ord("k"):
        delta = axis_angle_to_quaternion([1.0, 0.0, 0.0], -TARGET_ROTATION_STEP_RAD)
        target_quat = multiply_quaternions(delta, target_quat)
    elif key == ord("j"):
        delta = axis_angle_to_quaternion([0.0, 0.0, 1.0], TARGET_ROTATION_STEP_RAD)
        target_quat = multiply_quaternions(delta, target_quat)
    elif key == ord("l"):
        delta = axis_angle_to_quaternion([0.0, 0.0, 1.0], -TARGET_ROTATION_STEP_RAD)
        target_quat = multiply_quaternions(delta, target_quat)
    elif key == ord("u"):
        delta = axis_angle_to_quaternion([0.0, 1.0, 0.0], TARGET_ROTATION_STEP_RAD)
        target_quat = multiply_quaternions(delta, target_quat)
    elif key == ord("o"):
        delta = axis_angle_to_quaternion([0.0, 1.0, 0.0], -TARGET_ROTATION_STEP_RAD)
        target_quat = multiply_quaternions(delta, target_quat)
    elif key == ord(" "):
        target_pos = home_pos.copy()
        target_quat = home_quat.copy()
    elif key == ord("m"):
        toggled_auto_mode = True

    return target_pos, normalize_quaternion(target_quat), toggled_auto_mode


def draw_control_overlay(image, auto_mode):
    overlay = image.copy()
    mode_text = "AUTO TARGET" if auto_mode else "MANUAL TARGET"
    cv2.putText(overlay, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, "W/S Y  A/D X  R/F Z", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(overlay, "I/K pitch  J/L yaw  U/O roll", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(overlay, "SPACE reset  M toggle auto", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return overlay


def estimate_target_spatial_velocity_world(prev_pos, prev_quat, curr_pos, curr_quat, dt):
    """
    由前后两帧 mocap 位姿估计目标在世界坐标系下的线速度和角速度。
    """
    dt = max(float(dt), 1e-6)
    linear_world = (curr_pos - prev_pos) / dt

    R_prev = quaternion_to_rotation_matrix(prev_quat)
    R_curr = quaternion_to_rotation_matrix(curr_quat)
    R_delta_world = R_curr @ R_prev.T
    rvec_world, _ = cv2.Rodrigues(R_delta_world)
    angular_world = rvec_world.ravel() / dt

    return linear_world, angular_world


def compute_desired_camera_feedforward_world(
    target_pos_world,
    target_quat_world,
    target_linear_world,
    target_angular_world,
    T_tag_camera_desired,
    linear_gain=1.0,
    angular_gain=1.0,
):
    """
    根据目标本身的运动，构造期望相机在世界系下的前馈速度。

    由于相机期望位姿是“固定在标签坐标系中的一个相对位姿”，
    当标签平移/旋转时，理想相机也应该同步平移/旋转。
    """
    del target_pos_world  # 当前位置目前不直接用，但保留接口更清晰

    R_world_tag = quaternion_to_rotation_matrix(target_quat_world)
    p_tag_camera_desired = T_tag_camera_desired[:3, 3]
    desired_offset_world = R_world_tag @ p_tag_camera_desired

    linear_ff_world = target_linear_world + np.cross(target_angular_world, desired_offset_world)
    angular_ff_world = target_angular_world

    return np.concatenate(
        [linear_gain * linear_ff_world, angular_gain * angular_ff_world],
        axis=0,
    )





def main():
    # 加载模型和对应的数据容器
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # 如果定义了 keyframe，就先把机器人重置到 home 姿态
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0

    # 根据当前 qpos / ctrl 等状态刷新所有前向结果
    initialize_intvelocity_actuators(model, data, ACTUATOR_NAMES)
    mujoco.mj_forward(model, data)

    # 找到末端相机的 id
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id < 0:
        raise ValueError(f"cannot find camera: {CAMERA_NAME}")

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)
    if target_body_id < 0:
        raise ValueError(f"cannot find body: {TARGET_BODY_NAME}")

    target_mocap_id = model.body_mocapid[target_body_id]
    if target_mocap_id < 0:
        raise ValueError(f"body is not mocap-controlled: {TARGET_BODY_NAME}")

    target_home_pos = data.mocap_pos[target_mocap_id].copy()
    target_home_quat = data.mocap_quat[target_mocap_id].copy()
    target_manual_pos = target_home_pos.copy()
    target_manual_quat = target_home_quat.copy()
    prev_target_pos = target_home_pos.copy()
    prev_target_quat = target_home_quat.copy()

    # 根据 MuJoCo 相机参数近似构造内参矩阵
    camera_matrix = build_camera_matrix(model, camera_id, WIDTH, HEIGHT)

    # 这里先假设无畸变
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)



    # 期望相对位姿：
    # 相机位于标签正面法线方向 15 cm 处，并且光轴正对标签。
    T_tag_camera_desired = build_desired_tag_camera_transform(DESIRED_STANDOFF)
    T_camera_tag_desired = invert_transform(T_tag_camera_desired)

    print("T_camera_tag_desired =")
    print(np.round(T_camera_tag_desired, 4))
    print("T_tag_camera_desired =")
    print(np.round(T_tag_camera_desired, 4))


    print("camera_matrix =")
    print(camera_matrix)

    # 初始化 AprilTag 检测器
    detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        refine_edges=1,
    )

    # 初始化 GLFW，用于离屏渲染末端相机图像
    if not glfw.init():
        raise RuntimeError("failed to initialize glfw")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    offscreen_window = glfw.create_window(WIDTH, HEIGHT, "offscreen", None, None)
    if not offscreen_window:
        glfw.terminate()
        raise RuntimeError("failed to create offscreen window")

    glfw.make_context_current(offscreen_window)

    # 创建离屏渲染所需的场景、上下文、相机和视口
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    viewport = mujoco.MjrRect(0, 0, WIDTH, HEIGHT)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id

    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

    # 创建 OpenCV 小窗，显示末端相机画面
    cv2.namedWindow("End-Effector Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("End-Effector Camera", WIDTH, HEIGHT)

    # 打开 MuJoCo 主窗口
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        last_q_dot = np.zeros(len(ACTUATOR_NAMES), dtype=np.float64)
        lost_tag_count = LOST_TAG_HOLD_FRAMES + 1
        auto_target_mode = AUTO_MOVE_TARGET
        frame_count = 0
        settled_count = 0

        print("Target control:")
        print("  Focus the 'End-Effector Camera' window, then use W/S A/D R/F to move the tag.")
        print("  Use I/K J/L U/O to rotate the tag, SPACE to reset, M to toggle auto motion.")

        while viewer.is_running():
            control_applied = False

            if auto_target_mode:
                update_moving_target(
                    data,
                    target_mocap_id,
                    target_home_pos,
                    target_home_quat,
                    data.time,
                )
            else:
                set_target_pose(
                    data,
                    target_mocap_id,
                    target_manual_pos,
                    target_manual_quat,
                )

            # 更新 mocap 目标后先做一次前向传播，让当前帧渲染和控制都看到最新目标位姿
            mujoco.mj_forward(model, data)

            current_target_pos = data.mocap_pos[target_mocap_id].copy()
            current_target_quat = data.mocap_quat[target_mocap_id].copy()
            target_linear_world, target_angular_world = estimate_target_spatial_velocity_world(
                prev_target_pos,
                prev_target_quat,
                current_target_pos,
                current_target_quat,
                model.opt.timestep,
            )
            target_motion_speed = np.linalg.norm(target_linear_world) + DESIRED_STANDOFF * np.linalg.norm(target_angular_world)
            tracking_target = target_motion_speed > TARGET_MOTION_SPEED_THRESHOLD
            dynamic_max_q_dot = MAX_TRACKING_Q_DOT if target_motion_speed > TARGET_MOTION_SPEED_THRESHOLD else MAX_Q_DOT
            dynamic_position_gain = TRACKING_POSITION_GAIN if tracking_target else POSITION_GAIN
            dynamic_rotation_gain = TRACKING_ROTATION_GAIN if tracking_target else ROTATION_GAIN

            # 切到离屏上下文，从末端相机渲染图像
            glfw.make_context_current(offscreen_window)

            mujoco.mjv_updateScene(
                model,
                data,
                mujoco.MjvOption(),
                mujoco.MjvPerturb(),
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            mujoco.mjr_render(viewport, scene, context)

            # 读取 RGB 像素
            rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, viewport, context)

            # MuJoCo 原点在左下，OpenCV 原点在左上，所以要翻转
            rgb = np.flipud(rgb)

            # OpenCV 用 BGR 显示
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # AprilTag 检测一般在灰度图上进行
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            # 先画检测结果
            vis = draw_apriltags(bgr, tags)
            vis = draw_control_overlay(vis, auto_target_mode)

            # 如果检测到了 tag，就继续做位姿估计
            for tag in tags:
                success, rvec, tvec = estimate_tag_pose(
                    tag,
                    camera_matrix,
                    dist_coeffs,
                    TAG_SIZE,
                )

                if success:
                    # 在图像上画坐标轴
                    vis = draw_pose_axes(
                        vis,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        axis_length=0.05,
                    )
                    T_camera_tag = rvec_tvec_to_transform(rvec, tvec)
                    e_p_cam_raw, e_r_cam_raw, T_tag_camera = compute_pbvs_error_in_camera_frame(
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
                    position_error_norm = np.linalg.norm(e_p_cam_raw)
                    rotation_error_norm = np.linalg.norm(e_r_cam_raw)
                    near_goal = (
                        position_error_norm < NEAR_GOAL_POSITION_TOL
                        and rotation_error_norm < NEAR_GOAL_ROTATION_TOL
                        and not tracking_target
                    )

                    if near_goal:
                        settled_count += 1
                    else:
                        settled_count = 0

                    camera_rotation_world = data.cam_xmat[camera_id].reshape(3, 3)
                    v_e_desired_world = compute_desired_ee_velocity_world(
                        e_p_cam,
                        e_r_cam,
                        camera_rotation_world,
                        k_p_pos=dynamic_position_gain,
                        k_p_rot=dynamic_rotation_gain,
                    )
                    v_e_feedforward_world = compute_desired_camera_feedforward_world(
                        current_target_pos,
                        current_target_quat,
                        target_linear_world,
                        target_angular_world,
                        T_tag_camera_desired,
                        linear_gain=TARGET_LINEAR_FEEDFORWARD_GAIN,
                        angular_gain=TARGET_ANGULAR_FEEDFORWARD_GAIN,
                    )
                    v_e_desired_world = v_e_desired_world + v_e_feedforward_world

                    q_dot, J = compute_joint_velocity_from_ee_velocity(
                        model,
                        data,
                        SITE_NAME,
                        v_e_desired_world,
                        damping=JACOBIAN_DAMPING,
                    )

                    smoothing_alpha = Q_DOT_SMOOTH_ALPHA_NEAR_GOAL if near_goal else Q_DOT_SMOOTH_ALPHA
                    q_dot = blend_joint_velocity(last_q_dot, q_dot, smoothing_alpha)

                    effective_max_q_dot = min(dynamic_max_q_dot, NEAR_GOAL_MAX_Q_DOT) if near_goal else dynamic_max_q_dot
                    q_dot = np.clip(q_dot, -effective_max_q_dot, effective_max_q_dot)

                    if settled_count >= SETTLE_FRAMES_REQUIRED:
                        q_dot = np.zeros_like(q_dot)

                    apply_joint_velocity_directly(
                        data,
                        ACTUATOR_NAMES,
                        q_dot,
                    )
                    last_q_dot = q_dot.copy()
                    lost_tag_count = 0
                    control_applied = True

                    if frame_count % DEBUG_PRINT_EVERY == 0:
                        print(f"id={tag.tag_id}")
                        print("desired ee velocity world =", np.round(v_e_desired_world, 4))
                        print("q_dot =", np.round(q_dot, 4))
                        print("rvec =", rvec.ravel())
                        print("tvec =", tvec.ravel())
                        print("T_camera_tag =")
                        print(np.round(T_camera_tag, 4))
                        print("T_tag_camera =")
                        print(np.round(T_tag_camera, 4))
                        print("position error e_p_cam_raw =", np.round(e_p_cam_raw, 4))
                        print("rotation error e_r_cam_raw =", np.round(e_r_cam_raw, 4))
                        print("position error e_p_cam =", np.round(e_p_cam, 4))
                        print("rotation error e_r_cam =", np.round(e_r_cam, 4))
                        print("near_goal =", near_goal, "settled_count =", settled_count)
                    break

            # 这一帧没有成功识别并生成控制时，保持零速度
            if not control_applied:
                settled_count = 0
                lost_tag_count += 1

                if lost_tag_count <= LOST_TAG_HOLD_FRAMES:
                    hold_scale = LOST_TAG_DECAY ** lost_tag_count
                    q_dot_hold = np.clip(last_q_dot * hold_scale, -dynamic_max_q_dot, dynamic_max_q_dot)
                    apply_joint_velocity_directly(
                        data,
                        ACTUATOR_NAMES,
                        q_dot_hold,
                    )
                else:
                    camera_rotation_world = data.cam_xmat[camera_id].reshape(3, 3)
                    v_e_recovery_world = compute_lost_target_recovery_velocity_world(
                        camera_rotation_world,
                        backoff_speed=LOST_TARGET_BACKOFF_SPEED,
                    )
                    q_dot_recovery, _ = compute_joint_velocity_from_ee_velocity(
                        model,
                        data,
                        SITE_NAME,
                        v_e_recovery_world,
                        damping=JACOBIAN_DAMPING,
                    )
                    q_dot_recovery = np.clip(
                        q_dot_recovery,
                        -LOST_TARGET_RECOVERY_Q_DOT,
                        LOST_TARGET_RECOVERY_Q_DOT,
                    )
                    apply_joint_velocity_directly(
                        data,
                        ACTUATOR_NAMES,
                        q_dot_recovery,
                    )
                    if frame_count % DEBUG_PRINT_EVERY == 0:
                        print("target lost -> recovery backoff")
                        print("recovery ee velocity world =", np.round(v_e_recovery_world, 4))
                        print("recovery q_dot =", np.round(q_dot_recovery, 4))
                    cv2.putText(
                        vis,
                        "TARGET LOST: BACKING OFF TO REACQUIRE",
                        (10, HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 180, 255),
                        2,
                    )

            # 在写入控制之后再推进仿真，这样当前控制会真正作用到系统。
            # 多跑几步 physics，可以明显提升体感运动速度。
            for _ in range(SIM_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)

            # 显示检测后的图像
            cv2.imshow("End-Effector Camera", vis)

            # 按键控制标签。需要先让 OpenCV 小窗获得焦点。
            key = cv2.waitKey(1) & 0xFF
            if not auto_target_mode:
                target_manual_pos, target_manual_quat, toggled_auto_mode = apply_manual_target_control(
                    key,
                    target_manual_pos,
                    target_manual_quat,
                    target_home_pos,
                    target_home_quat,
                )
                if toggled_auto_mode:
                    auto_target_mode = True
            elif key == ord("m"):
                auto_target_mode = False
                target_manual_pos = data.mocap_pos[target_mocap_id].copy()
                target_manual_quat = data.mocap_quat[target_mocap_id].copy()

            # 按 ESC 退出
            if key == 27:
                break

            # 把最新仿真状态同步到 MuJoCo 主窗口
            viewer.sync()

            prev_target_pos = current_target_pos
            prev_target_quat = current_target_quat
            frame_count += 1

            # 稍微降一点循环频率，避免空转太快
            time.sleep(GUI_SLEEP)

    # 清理资源
    cv2.destroyAllWindows()
    glfw.destroy_window(offscreen_window)
    glfw.terminate()


if __name__ == "__main__":
    main()
