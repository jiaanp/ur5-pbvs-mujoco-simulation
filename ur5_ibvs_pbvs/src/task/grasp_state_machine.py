import numpy as np

from src.utils.transforms import build_desired_tag_camera_transform


class GraspStateMachine:
    """
    一个最小可用的“吸附抓取”阶段状态机。

    当前阶段：
    - track:    预抓取阶段，保持较远 standoff
    - approach: 接近阶段，减小 standoff
    - attach:   满足吸附条件，判定吸附成功
    - lift:     吸附后抬升
    - place:    把目标移动到机械臂旁边的放置点
    - home:     机械臂回到初始位姿
    - done:     全流程结束

    这一版不控制夹爪，只负责阶段切换与目标距离管理。
    """

    def __init__(
        self,
        track_standoff=0.15,
        approach_standoff=0.015,
        track_position_tol=0.02,
        track_rotation_tol_rad=np.deg2rad(3.0),
        settle_frames_required=4,
        attach_position_tol=0.014,
        attach_rotation_tol_rad=np.deg2rad(8.0),
        attach_settle_frames_required=2,
        dynamic_track_position_tol=0.035,
        dynamic_track_rotation_tol_rad=np.deg2rad(8.0),
        dynamic_track_settle_frames_required=2,
        dynamic_attach_position_tol=0.022,
        dynamic_attach_rotation_tol_rad=np.deg2rad(12.0),
        dynamic_attach_settle_frames_required=1,
        lift_offset_world=None,
        place_offset_world=None,
        place_position_tol=0.01,
    ):
        self.track_standoff = float(track_standoff)
        self.approach_standoff = float(approach_standoff)
        self.track_position_tol = float(track_position_tol)
        self.track_rotation_tol_rad = float(track_rotation_tol_rad)
        self.settle_frames_required = int(settle_frames_required)
        self.attach_position_tol = float(attach_position_tol)
        self.attach_rotation_tol_rad = float(attach_rotation_tol_rad)
        self.attach_settle_frames_required = int(attach_settle_frames_required)
        self.dynamic_track_position_tol = float(dynamic_track_position_tol)
        self.dynamic_track_rotation_tol_rad = float(dynamic_track_rotation_tol_rad)
        self.dynamic_track_settle_frames_required = int(dynamic_track_settle_frames_required)
        self.dynamic_attach_position_tol = float(dynamic_attach_position_tol)
        self.dynamic_attach_rotation_tol_rad = float(dynamic_attach_rotation_tol_rad)
        self.dynamic_attach_settle_frames_required = int(dynamic_attach_settle_frames_required)
        self.lift_offset_world = np.asarray(
            lift_offset_world if lift_offset_world is not None else [0.0, 0.0, 0.08],
            dtype=np.float64,
        )
        self.place_offset_world = np.asarray(
            place_offset_world if place_offset_world is not None else [0.0, -0.25, 0.0],
            dtype=np.float64,
        )
        self.place_position_tol = float(place_position_tol)

        self.phase = "track"
        self.settled_count = 0
        self.attached = False
        self.lift_start_pos_world = None
        self.lift_target_pos_world = None
        self.place_start_pos_world = None
        self.place_target_pos_world = None

    def reset(self):
        self.phase = "track"
        self.settled_count = 0
        self.attached = False
        self.lift_start_pos_world = None
        self.lift_target_pos_world = None
        self.place_start_pos_world = None
        self.place_target_pos_world = None

    def get_desired_standoff(self):
        if self.phase == "track":
            return self.track_standoff
        if self.phase == "approach":
            return self.approach_standoff
        if self.phase == "attach":
            return 0.005
        if self.phase == "lift":
            return 0.005
        if self.phase == "place":
            return 0.005
        if self.phase == "home":
            return self.track_standoff
        return self.track_standoff

    def get_desired_tag_camera_transform(self):
        return build_desired_tag_camera_transform(self.get_desired_standoff())

    def update(self, position_error_norm, rotation_error_norm, target_motion_speed=0.0, motion_threshold=1e-3):
        """
        根据当前误差和目标是否还在明显运动，更新阶段。
        """
        moving_target = target_motion_speed > motion_threshold

        if self.phase == "track":
            pos_tol = self.dynamic_track_position_tol if moving_target else self.track_position_tol
            rot_tol = self.dynamic_track_rotation_tol_rad if moving_target else self.track_rotation_tol_rad
            settle_required = (
                self.dynamic_track_settle_frames_required
                if moving_target
                else self.settle_frames_required
            )

            close_enough = (
                position_error_norm < pos_tol
                and rotation_error_norm < rot_tol
            )

            if close_enough:
                self.settled_count += 1
            else:
                self.settled_count = 0

            if self.settled_count >= settle_required:
                self.phase = "approach"
                self.settled_count = 0

        elif self.phase == "approach":
            pos_tol = self.dynamic_attach_position_tol if moving_target else self.attach_position_tol
            rot_tol = self.dynamic_attach_rotation_tol_rad if moving_target else self.attach_rotation_tol_rad
            settle_required = (
                self.dynamic_attach_settle_frames_required
                if moving_target
                else self.attach_settle_frames_required
            )

            close_enough = (
                position_error_norm < pos_tol
                and rotation_error_norm < rot_tol
            )

            if close_enough:
                self.settled_count += 1
            else:
                self.settled_count = 0

            if self.settled_count >= settle_required:
                self.phase = "attach"
                self.attached = True
                self.settled_count = 0

        elif self.phase == "attach":
            self.phase = "lift"

        elif self.phase == "lift":
            pass

        elif self.phase == "place":
            pass

        elif self.phase == "home":
            pass

        elif self.phase == "done":
            pass

        return self.phase

    def should_attach(self):
        return self.phase in ("attach", "lift") and self.attached

    def force_attach(self):
        """
        当近距离吸附阶段因视野过近临时丢失标签时，允许外部直接判定吸附成功。
        """
        self.phase = "lift"
        self.attached = True
        self.settled_count = 0

    def start_lift(self, current_pos_world):
        current_pos_world = np.asarray(current_pos_world, dtype=np.float64)
        self.lift_start_pos_world = current_pos_world.copy()
        self.lift_target_pos_world = current_pos_world + self.lift_offset_world

    def get_lift_error_world(self, current_pos_world):
        """
        返回 lift 阶段的世界系位移误差，以及是否完成抬升。
        """
        current_pos_world = np.asarray(current_pos_world, dtype=np.float64)

        if self.lift_target_pos_world is None:
            self.start_lift(current_pos_world)

        position_error = self.lift_target_pos_world - current_pos_world
        done = np.linalg.norm(position_error) < 0.005
        return position_error, done

    def start_place(self, current_pos_world, place_target_pos_world=None):
        current_pos_world = np.asarray(current_pos_world, dtype=np.float64)
        self.phase = "place"
        self.place_start_pos_world = current_pos_world.copy()
        if place_target_pos_world is None:
            self.place_target_pos_world = current_pos_world + self.place_offset_world
        else:
            self.place_target_pos_world = np.asarray(place_target_pos_world, dtype=np.float64).copy()

    def get_place_error_world(self, current_pos_world):
        current_pos_world = np.asarray(current_pos_world, dtype=np.float64)

        if self.place_target_pos_world is None:
            self.start_place(current_pos_world)

        position_error = self.place_target_pos_world - current_pos_world
        done = np.linalg.norm(position_error) < self.place_position_tol
        return position_error, done

    def start_home(self):
        self.phase = "home"
        self.attached = False
        self.settled_count = 0

    def mark_done(self):
        self.phase = "done"
        self.attached = False
        self.settled_count = 0

    def draw_overlay(self, image):
        import cv2

        overlay = image.copy()
        attach_text = "ATTACHED" if self.attached else "DETACHED"
        text = f"PHASE: {self.phase.upper()}  {attach_text}"
        cv2.putText(overlay, text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
        return overlay
