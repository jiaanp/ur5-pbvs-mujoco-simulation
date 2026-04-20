import cv2
import numpy as np

from src.utils.transforms import invert_transform

class PBVSController:
    def __init__(self, r_mj_camera_from_cv_camera):
        """
        参数：
        - r_mj_camera_from_cv_camera:
          OpenCV 相机坐标系 -> MuJoCo 相机局部坐标系 的旋转矩阵
        """

        self.r_mj_camera_from_cv_camera = np.asarray(r_mj_camera_from_cv_camera, dtype=np.float64)

    

    def compute_pbvs_error_in_camera_frame(self, t_camera_tag, t_tag_camera_desired):
        """
        基于当前 tag 位姿和期望相机位姿，计算“当前相机坐标系”下的 PBVS 误差。

        返回：
        - e_p_cam: 位置误差，在当前相机坐标系下表示
        - e_r_cam: 姿态误差，在当前相机坐标系下表示
        - t_tag_camera: 当前相机在 tag 坐标系下的位姿
        """
        t_tag_camera = invert_transform(t_camera_tag)
        r_tc = t_tag_camera[:3, :3]
        p_tc = t_tag_camera[:3, 3]


        r_tc_des = t_tag_camera_desired[:3, :3]
        p_tc_des = t_tag_camera_desired[:3, 3]

        # 先在 tag 坐标系下做差，再转回当前相机坐标系
        p_err_tag = p_tc_des - p_tc
        e_p_cam = r_tc.T @ p_err_tag

         # 当前相机到期望相机的相对旋转，表达在当前相机坐标系
        r_err_cam = r_tc.T @ r_tc_des
        e_r_cam, _ = cv2.Rodrigues(r_err_cam)
        e_r_cam = e_r_cam.ravel()
        return e_p_cam, e_r_cam, t_tag_camera
    

    def compute_desired_ee_velocity_world(self, e_p_cam, e_r_cam, camera_rotation_world, k_p_pos, k_p_rot):

        """
        根据相机坐标系下的 PBVS 误差，计算世界坐标系下的末端期望速度。
        """
        v_linear_cv = k_p_pos * e_p_cam
        v_angular_cv = k_p_rot * e_r_cam

        v_linear_mj = self.r_mj_camera_from_cv_camera @ v_linear_cv
        v_angular_mj = self.r_mj_camera_from_cv_camera @ v_angular_cv


        v_linear_world = camera_rotation_world @ v_linear_mj
        v_angular_world = camera_rotation_world @ v_angular_mj
        return np.concatenate([v_linear_world, v_angular_world], axis=0)
    
def soften_error_vector(vector, deadband, soft_zone):
    """
    给误差加一个软死区：
    - 很小的误差直接置零
    - 靠近目标时逐渐缩小误差
    """
    norm = np.linalg.norm(vector)

    if norm <= deadband:
        return np.zeros_like(vector)

    if norm >= soft_zone:
        return vector

    scale = (norm - deadband) / max(soft_zone - deadband, 1e-9)
    return vector * scale
