import numpy as np
import pinocchio as pin


def _skew(vector):
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


class PinocchioKinematics:
    """
    使用 Pinocchio 的 UR5 运动学适配层。

    目标：
    - 保持和当前 UR5Kinematics 相近的接口
    - 先只负责 FK、末端位姿、Jacobian、速度逆解
    - 不接管 MuJoCo 仿真本身
    """

    def __init__(
        self,
        urdf_path,
        ee_frame_name="tool0",
        arm_dof_count=6,
        package_dirs=None,
        root_joint=None,
        q_signs=None,
        ee_offset_local=None,
    ):
        self.urdf_path = urdf_path
        self.ee_frame_name = ee_frame_name
        self.arm_dof_count = int(arm_dof_count)
        self.q_signs = np.asarray(
            q_signs if q_signs is not None else np.ones(self.arm_dof_count),
            dtype=np.float64,
        )
        self.ee_offset_local = np.asarray(
            ee_offset_local if ee_offset_local is not None else np.zeros(3),
            dtype=np.float64,
        )

        # 从 URDF 构建机器人模型
        if root_joint is None:
            if package_dirs is None:
                self.model = pin.buildModelFromUrdf(urdf_path)
            else:
                self.model = pin.buildModelFromUrdf(urdf_path, package_dirs)
        else:
            if package_dirs is None:
                self.model = pin.buildModelFromUrdf(urdf_path, root_joint)
            else:
                self.model = pin.buildModelFromUrdf(
                    urdf_path,
                    root_joint,
                    package_dirs,
                )

        self.data = self.model.createData()

        if not self.model.existFrame(self.ee_frame_name):
            raise ValueError(f"cannot find frame in URDF: {self.ee_frame_name}")

        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)

    def _normalize_q(self, q, apply_joint_mapping=True):
        """
        兼容传入长度不足 model.nq 的情况。
        对 UR5 这种 6 自由度机械臂，通常传入 6 维关节角即可。
        """
        q = np.asarray(q, dtype=np.float64).reshape(-1)

        if apply_joint_mapping:
            mapped_len = min(q.shape[0], self.q_signs.shape[0])
            q = q.copy()
            q[:mapped_len] = q[:mapped_len] * self.q_signs[:mapped_len]

        if q.shape[0] == self.model.nq:
            return q

        if q.shape[0] < self.model.nq:
            q_full = pin.neutral(self.model)
            q_full[: q.shape[0]] = q
            return q_full

        return q[: self.model.nq]

    def forward_kinematics(self, q, apply_joint_mapping=True):
        """
        计算末端 frame 在世界坐标系下的位姿。

        返回：
        - pos_world: 3维位置
        - rot_world: 3x3旋转矩阵
        - t_world_ee: 4x4齐次变换矩阵
        """
        q = self._normalize_q(q, apply_joint_mapping=apply_joint_mapping)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        placement = self.data.oMf[self.ee_frame_id]

        pos_world = placement.translation.copy()
        rot_world = placement.rotation.copy()
        pos_world = pos_world + rot_world @ self.ee_offset_local

        t_world_ee = np.eye(4, dtype=np.float64)
        t_world_ee[:3, :3] = rot_world
        t_world_ee[:3, 3] = pos_world

        return pos_world, rot_world, t_world_ee

    def compute_frame_jacobian(self, q):
        """
        计算末端 frame 的 Jacobian。

        这里使用 LOCAL_WORLD_ALIGNED：
        - 线速度/角速度方向更容易和世界系控制量对齐
        - 更适合你当前 PBVS / MPC 主程序里的世界系误差定义
        """
        q = self._normalize_q(q)

        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        j_full = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        offset_world = self.data.oMf[self.ee_frame_id].rotation @ self.ee_offset_local
        j_full[:3, :] = j_full[:3, :] - _skew(offset_world) @ j_full[3:, :]

        return j_full[:, : self.arm_dof_count].copy()

    def compute_joint_velocity_from_ee_velocity(self, q, v_e_desired_world, damping=0.05):
        """
        根据世界系下的末端期望速度，使用阻尼伪逆求解关节速度。

        参数：
        - q: 当前关节角
        - v_e_desired_world: 6维末端速度 [vx, vy, vz, wx, wy, wz]
        - damping: 阻尼系数

        返回：
        - q_dot: 关节速度
        - J: 当前 Jacobian
        """
        j = self.compute_frame_jacobian(q)
        jt = j.T

        q_dot = jt @ np.linalg.inv(
            j @ jt + (damping ** 2) * np.eye(6)
        ) @ np.asarray(v_e_desired_world, dtype=np.float64)

        return q_dot, j

    def get_end_effector_pose(self, q):
        """
        语义化别名，便于主程序里直接拿末端位姿。
        """
        return self.forward_kinematics(q)

    def get_end_effector_jacobian(self, q):
        """
        语义化别名，便于主程序里直接拿 Jacobian。
        """
        return self.compute_frame_jacobian(q)

    def blend_joint_velocity(self, previous_q_dot, current_q_dot, alpha):
        """
        和现有 UR5Kinematics 保持一致的速度平滑接口。
        """
        previous_q_dot = np.asarray(previous_q_dot, dtype=np.float64)
        current_q_dot = np.asarray(current_q_dot, dtype=np.float64)
        return (1.0 - alpha) * previous_q_dot + alpha * current_q_dot

    def solve_ik_position(self, target_pos_world, q_init):
        """
        Solve position-only inverse kinematics using scipy numerical optimization.

        Minimizes ||FK(q).position - target_pos_world||^2 subject to joint limits.

        Args:
            target_pos_world: (3,) world-frame target position
            q_init:           (6,) initial guess joint configuration

        Returns:
            (6,) joint configuration that achieves the target position
        """
        from scipy.optimize import minimize

        target = np.asarray(target_pos_world, dtype=np.float64).reshape(3)
        q0 = self._normalize_q(q_init, apply_joint_mapping=True)[:self.arm_dof_count].copy()

        # Joint limits
        lo = self.model.lowerPositionLimit[:self.arm_dof_count].copy()
        hi = self.model.upperPositionLimit[:self.arm_dof_count].copy()
        # Handle unbounded cases (Pinocchio may use very large values for unlimited joints)
        lo[lo < -10.0] = -6.283
        hi[hi > 10.0] = 6.283

        def cost(q):
            # q is already in Pinocchio space (normalized), skip joint mapping
            pos, _, _ = self.forward_kinematics(q, apply_joint_mapping=False)
            return np.sum((pos - target) ** 2)

        res = minimize(
            cost,
            q0,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi)),
            options={"maxiter": 200, "ftol": 1e-9},
        )

        q_result = res.x.copy()
        # Undo joint sign mapping: the caller expects MuJoCo-compatible q
        q_result = q_result * self.q_signs[:self.arm_dof_count]
        return q_result
