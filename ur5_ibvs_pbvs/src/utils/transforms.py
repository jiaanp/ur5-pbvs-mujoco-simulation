
import cv2
import numpy as np


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


def rvec_tvec_to_transform(rvec, tvec):
    """
    把 OpenCV 的 rvec / tvec 转成 4x4 齐次变换矩阵。
    返回 T_camera_tag。
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = tvec.ravel()
    return transform


def rotation_matrix_to_quaternion(rotation_matrix):
    """
    把 3x3 旋转矩阵转换成 [w, x, y, z] 四元数。
    """
    r = np.asarray(rotation_matrix, dtype=np.float64)
    trace = np.trace(r)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def build_desired_tag_camera_transform(standoff):
    """
    构造期望的 T_tag_camera：
    - 相机位于标签法线方向外 standoff 米
    - 相机正对标签
    """
    desired_rotation = np.eye(3, dtype=np.float64)
    desired_translation = np.array([0.0, 0.0, -standoff], dtype=np.float64)
    return make_transform(desired_rotation, desired_translation)


def rotation_matrix_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = axis / axis_norm
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c

    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def build_desired_tag_grasp_transform(standoff, yaw_about_normal_rad=0.0):
    """
    构造期望的 T_tag_grasp：
    - 抓取中心位于标签法线方向外 standoff 米
    - 抓取方向沿标签法向
    - 允许绕标签法向再附加一个 yaw，用来调手指开口方向

    当前项目里默认把“抓取参考系”和之前的相机参考系在标签法向上保持一致，
    这样先把目标从“相机目标位姿”提升成“抓取目标位姿”，后面再通过固定外参
    换算出真正的相机目标位姿。
    """
    base_rotation = np.eye(3, dtype=np.float64)
    yaw_rotation = rotation_matrix_from_axis_angle([0.0, 0.0, 1.0], yaw_about_normal_rad)
    desired_rotation = base_rotation @ yaw_rotation
    desired_translation = np.array([0.0, 0.0, -standoff], dtype=np.float64)
    return make_transform(desired_rotation, desired_translation)


def build_desired_tag_camera_transform_from_grasp(
    standoff,
    t_grasp_camera,
    yaw_about_normal_rad=0.0,
):
    """
    先定义标签坐标系下的期望抓取位姿，再通过固定的 grasp->camera 外参
    推出期望的 tag->camera 位姿。
    """
    t_tag_grasp_desired = build_desired_tag_grasp_transform(
        standoff=standoff,
        yaw_about_normal_rad=yaw_about_normal_rad,
    )
    return t_tag_grasp_desired @ np.asarray(t_grasp_camera, dtype=np.float64)
