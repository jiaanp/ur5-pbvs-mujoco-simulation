import math

import cv2
import numpy as np


class TargetMotionController:
    """
    管理标签目标的自动运动、手动控制、位姿重置，以及目标速度估计。
    """

    def __init__(
        self,
        home_pos,
        home_quat,
        auto_move=False,
        motion_amplitude=None,
        motion_frequency=None,
        motion_phase=None,
        translation_step=0.01,
        rotation_step_rad=None,
    ):
        self.home_pos = np.asarray(home_pos, dtype=np.float64).copy()
        self.home_quat = self.normalize_quaternion(home_quat)

        self.manual_pos = self.home_pos.copy()
        self.manual_quat = self.home_quat.copy()

        self.auto_mode = bool(auto_move)

        self.motion_amplitude = np.asarray(
            motion_amplitude if motion_amplitude is not None else [0.06, 0.04, 0.02],
            dtype=np.float64,
        )
        self.motion_frequency = np.asarray(
            motion_frequency if motion_frequency is not None else [0.45, 0.80, 0.60],
            dtype=np.float64,
        )
        self.motion_phase = np.asarray(
            motion_phase if motion_phase is not None else [0.0, 0.6, 1.1],
            dtype=np.float64,
        )

        self.translation_step = float(translation_step)
        self.rotation_step_rad = (
            float(rotation_step_rad)
            if rotation_step_rad is not None
            else math.radians(5.0)
        )

        self.prev_pos = self.home_pos.copy()
        self.prev_quat = self.home_quat.copy()

    @staticmethod
    def normalize_quaternion(quat):
        quat = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    @staticmethod
    def axis_angle_to_quaternion(axis, angle_rad):
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        axis = axis / axis_norm
        half = angle_rad * 0.5
        sin_half = math.sin(half)

        return np.array(
            [
                math.cos(half),
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
            ],
            dtype=np.float64,
        )

    @staticmethod
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

    @staticmethod
    def quaternion_to_rotation_matrix(quat):
        quat = TargetMotionController.normalize_quaternion(quat)
        w, x, y, z = quat
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def reset_manual_target(self):
        self.manual_pos = self.home_pos.copy()
        self.manual_quat = self.home_quat.copy()

    def update_auto_target(self, sim_time):
        offset = self.motion_amplitude * np.sin(
            self.motion_frequency * sim_time + self.motion_phase
        )
        pos = self.home_pos + offset
        quat = self.home_quat.copy()
        return pos, quat

    def handle_key(self, key):
        """
        返回：
        - toggled_auto_mode: 是否切换了自动模式
        """
        toggled_auto_mode = False

        if key in (-1, 255):
            return toggled_auto_mode

        if key == ord("a"):
            self.manual_pos[0] -= self.translation_step
        elif key == ord("d"):
            self.manual_pos[0] += self.translation_step
        elif key == ord("w"):
            self.manual_pos[1] += self.translation_step
        elif key == ord("s"):
            self.manual_pos[1] -= self.translation_step
        elif key == ord("r"):
            self.manual_pos[2] += self.translation_step
        elif key == ord("f"):
            self.manual_pos[2] -= self.translation_step
        elif key == ord("i"):
            delta = self.axis_angle_to_quaternion([1.0, 0.0, 0.0], self.rotation_step_rad)
            self.manual_quat = self.multiply_quaternions(delta, self.manual_quat)
        elif key == ord("k"):
            delta = self.axis_angle_to_quaternion([1.0, 0.0, 0.0], -self.rotation_step_rad)
            self.manual_quat = self.multiply_quaternions(delta, self.manual_quat)
        elif key == ord("j"):
            delta = self.axis_angle_to_quaternion([0.0, 0.0, 1.0], self.rotation_step_rad)
            self.manual_quat = self.multiply_quaternions(delta, self.manual_quat)
        elif key == ord("l"):
            delta = self.axis_angle_to_quaternion([0.0, 0.0, 1.0], -self.rotation_step_rad)
            self.manual_quat = self.multiply_quaternions(delta, self.manual_quat)
        elif key == ord("u"):
            delta = self.axis_angle_to_quaternion([0.0, 1.0, 0.0], self.rotation_step_rad)
            self.manual_quat = self.multiply_quaternions(delta, self.manual_quat)
        elif key == ord("o"):
            delta = self.axis_angle_to_quaternion([0.0, 1.0, 0.0], -self.rotation_step_rad)
            self.manual_quat = self.multiply_quaternions(delta, self.manual_quat)
        elif key == ord(" "):
            self.reset_manual_target()
        elif key == ord("m"):
            self.auto_mode = not self.auto_mode
            toggled_auto_mode = True

        self.manual_quat = self.normalize_quaternion(self.manual_quat)
        return toggled_auto_mode

    def get_target_pose(self, sim_time):
        if self.auto_mode:
            return self.update_auto_target(sim_time)
        return self.manual_pos.copy(), self.manual_quat.copy()

    def estimate_target_spatial_velocity_world(self, current_pos, current_quat, dt):
        dt = max(float(dt), 1e-6)

        linear_world = (current_pos - self.prev_pos) / dt

        r_prev = self.quaternion_to_rotation_matrix(self.prev_quat)
        r_curr = self.quaternion_to_rotation_matrix(current_quat)
        r_delta_world = r_curr @ r_prev.T
        rvec_world, _ = cv2.Rodrigues(r_delta_world)
        angular_world = rvec_world.ravel() / dt

        self.prev_pos = np.asarray(current_pos, dtype=np.float64).copy()
        self.prev_quat = self.normalize_quaternion(current_quat)

        return linear_world, angular_world

    def draw_overlay(self, image):
        overlay = image.copy()
        mode_text = "AUTO TARGET" if self.auto_mode else "MANUAL TARGET"
        cv2.putText(overlay, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(overlay, "W/S Y  A/D X  R/F Z", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(overlay, "I/K pitch  J/L yaw  U/O roll", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(overlay, "SPACE reset  M toggle auto", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return overlay
