
import math
import numpy as np


def build_camera_matrix(model, camera_id, width, height):
    """
    根据 MuJoCo 相机 fovy 和图像尺寸，近似构造相机内参矩阵。
    """
    fovy_deg = model.cam_fovy[camera_id]
    fy = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def build_zero_distortion():
    """
    当前仿真先假设无畸变。
    """
    return np.zeros((5, 1), dtype=np.float32)
