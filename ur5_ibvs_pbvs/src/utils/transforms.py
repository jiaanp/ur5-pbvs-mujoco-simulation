
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


def build_desired_tag_camera_transform(standoff):
    """
    构造期望的 T_tag_camera：
    - 相机位于标签法线方向外 standoff 米
    - 相机正对标签
    """
    desired_rotation = np.eye(3, dtype=np.float64)
    desired_translation = np.array([0.0, 0.0, -standoff], dtype=np.float64)
    return make_transform(desired_rotation, desired_translation)
