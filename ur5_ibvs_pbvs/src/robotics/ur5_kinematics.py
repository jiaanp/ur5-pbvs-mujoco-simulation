
import numpy as np
import mujoco


class UR5Kinematics:
    def __init__(self, model, data, site_name, arm_dof_count=6):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.arm_dof_count = arm_dof_count

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"cannot find site: {site_name}")

    def compute_site_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        J_full = np.vstack([jacp, jacr])
        J = J_full[:, : self.arm_dof_count]
        return J

    def compute_joint_velocity_from_ee_velocity(self, v_e_desired_world, damping=0.05):
        J = self.compute_site_jacobian()
        JT = J.T
        q_dot = JT @ np.linalg.inv(J @ JT + (damping ** 2) * np.eye(6)) @ v_e_desired_world
        return q_dot, J

    def blend_joint_velocity(self, previous_q_dot, current_q_dot, alpha):
        return (1.0 - alpha) * previous_q_dot + alpha * current_q_dot
