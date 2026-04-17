import mujoco
import numpy as np


SCENE_XML = "/home/adrian/ur5_ibvs_pbvs/model/scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    print("model loaded")
    print(f"nq = {model.nq}, nv = {model.nv}, nu = {model.nu}")

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "end_effector_camera")

    print(f"target body id = {target_body_id}")
    print(f"attachment_site id = {site_id}")
    print(f"end_effector_camera id = {camera_id}")

    if target_body_id < 0:
        raise ValueError("cannot find body: target")
    if site_id < 0:
        raise ValueError("cannot find site: attachment_site")
    if camera_id < 0:
        raise ValueError("cannot find camera: end_effector_camera")

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        print("home keyframe loaded")
        print("qpos[:6] =", np.round(data.qpos[:6], 4))
    else:
        print("no keyframe found")

    print("model check passed")


if __name__ == "__main__":
    main()


