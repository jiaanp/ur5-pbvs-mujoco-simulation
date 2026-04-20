
import mujoco
import numpy as np


class MujocoEnv:
    def __init__(self, scene_xml):
        self.scene_xml = scene_xml
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

    def reset_to_home(self):
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n_steps=1):
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def get_body_id(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"cannot find body: {name}")
        return body_id

    def get_site_id(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"cannot find site: {name}")
        return site_id

    def get_camera_id(self, name):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            raise ValueError(f"cannot find camera: {name}")
        return camera_id

    def get_body_pose(self, body_name):
        body_id = self.get_body_id(body_name)
        pos = self.data.xpos[body_id].copy()
        rot = self.data.xmat[body_id].reshape(3, 3).copy()
        return pos, rot

    def get_site_pose(self, site_name):
        site_id = self.get_site_id(site_name)
        pos = self.data.site_xpos[site_id].copy()
        mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
        return pos, mat

    def get_joint_positions(self, n=6):
        return self.data.qpos[:n].copy()

    def initialize_intvelocity_actuators(self, actuator_names):
        for name in actuator_names:
            self.data.actuator(name).ctrl = 0.0

        if getattr(self.data, "act", None) is not None and self.data.act.size >= self.model.na:
            for name in actuator_names:
                actuator_id = self.model.actuator(name).id
                joint_id = self.model.actuator_trnid[actuator_id][0]
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.data.act[actuator_id] = self.data.qpos[qpos_adr]

    def apply_joint_velocity(self, actuator_names, q_dot):
        for i, name in enumerate(actuator_names):
            self.data.actuator(name).ctrl = float(q_dot[i])

    def zero_joint_velocity(self, actuator_names):
        for name in actuator_names:
            self.data.actuator(name).ctrl = 0.0

    def set_target_pose(self, target_body_name, pos, quat):
        body_id = self.get_body_id(target_body_name)
        mocap_id = self.model.body_mocapid[body_id]
        if mocap_id < 0:
            raise ValueError(f"body is not mocap-enabled: {target_body_name}")

        quat = np.asarray(quat, dtype=np.float64)
        quat = quat / max(np.linalg.norm(quat), 1e-12)

        self.data.mocap_pos[mocap_id] = pos
        self.data.mocap_quat[mocap_id] = quat
