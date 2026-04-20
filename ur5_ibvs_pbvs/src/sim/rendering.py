
import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np


class MujocoRenderer:
    def __init__(self, model, data, camera_name, width=640, height=480):
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height

        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if self.camera_id < 0:
            raise ValueError(f"cannot find camera: {camera_name}")

        if not glfw.init():
            raise RuntimeError("failed to initialize glfw")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.offscreen_window = glfw.create_window(
            self.width, self.height, "offscreen", None, None
        )
        if not self.offscreen_window:
            glfw.terminate()
            raise RuntimeError("failed to create offscreen window")

        glfw.make_context_current(self.offscreen_window)

        self.scene = mujoco.MjvScene(model, maxgeom=1000)
        self.context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = self.camera_id

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        cv2.namedWindow("End-Effector Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("End-Effector Camera", self.width, self.height)

    def render_camera_rgb(self):
        glfw.make_context_current(self.offscreen_window)

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            mujoco.MjvOption(),
            mujoco.MjvPerturb(),
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(self.viewport, self.scene, self.context)

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, self.viewport, self.context)
        rgb = np.flipud(rgb)
        return rgb

    def render_camera_bgr(self):
        rgb = self.render_camera_rgb()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def show_camera_image(self, image):
        cv2.imshow("End-Effector Camera", image)

    def create_viewer(self):
        return mujoco.viewer.launch_passive(self.model, self.data)

    def close(self):
        cv2.destroyAllWindows()
        if self.offscreen_window is not None:
            glfw.destroy_window(self.offscreen_window)
        glfw.terminate()
