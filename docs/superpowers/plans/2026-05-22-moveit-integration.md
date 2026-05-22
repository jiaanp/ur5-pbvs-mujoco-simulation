# MoveIt Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OMPL direct calls with MoveIt 2 for lift/place/home motion planning

**Architecture:** MoveIt manages IK + collision-aware planning for post-grasp phases. MuJoCo handles physics and execution. State synchronized before each planning call.

**Tech Stack:** ROS2 Humble, MoveIt 2 (moveit_commander), MuJoCo, existing PBVS/MPC pipeline

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/config.py` | Modify | Add `BOX_WALLS` list, MoveIt config paths |
| `src/task/pbvs_moveit_setup.py` | Create | MoveIt initialization (move_group, planning_scene) |
| `src/task/pbvs_moveit_phases.py` | Create | `handle_lift_phase_moveit`, `handle_place_phase_moveit`, `handle_home_phase_moveit` |
| `main_pbvs_moveit.py` | Create | Main entry (copy of main_pbvs_mpc.py with MoveIt phases) |

---

### Task 1: Add BOX_WALLS to config

**Files:**
- Modify: `ur5_ibvs_pbvs/src/config.py`

- [ ] **Step 1: Append BOX_WALLS definition**

Add at end of config.py:

```python
# ===========================================================================
# Box obstacle definitions (shared by MuJoCo XML and MoveIt PlanningScene)
# ===========================================================================

BOX_WALLS = [
    {"name": "place_box_wall_left",  "type": "box", "pos": [-0.80, 0.0, 0.083], "size": [0.005, 0.20, 0.15]},
    {"name": "place_box_wall_right", "type": "box", "pos": [-0.40, 0.0, 0.083], "size": [0.005, 0.20, 0.15]},
    {"name": "place_box_wall_front", "type": "box", "pos": [-0.60, -0.20, 0.083], "size": [0.20, 0.005, 0.15]},
    {"name": "place_box_wall_back",  "type": "box", "pos": [-0.60, 0.20, 0.083], "size": [0.20, 0.005, 0.15]},
]

PLACE_BOX_BASE_POS = [-0.60, 0.0, 0.005]
PLACE_BOX_BASE_SIZE = [0.20, 0.20, 0.005]

# MoveIt config
MOVEIT_CONFIG_PACKAGE = "ur5_moveit_config"
MOVEIT_CONFIG_PATH = "/home/adrian/ur5-pbvs-mujoco-simulation/moveit2-learning/ur5_moveit_config"
PLANNING_GROUP = "arm"
```

- [ ] **Step 2: Commit**

---

### Task 2: MoveIt setup module

**Files:**
- Create: `ur5_ibvs_pbvs/src/task/pbvs_moveit_setup.py`

- [ ] **Step 1: Write `init_moveit()` function**

```python
"""
MoveIt initialization for UR5e motion planning.
"""

import numpy as np
import rclpy
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive

from src.config import (
    ARM_DOF_COUNT,
    BOX_WALLS,
    PLACE_BOX_BASE_POS,
    PLACE_BOX_BASE_SIZE,
    PLANNING_GROUP,
    PLACE_BOX_CENTER_WORLD,
    PLACE_RELEASE_HEIGHT,
)


def init_moveit():
    """
    Initialize ROS2 + MoveIt. Returns move_group and planning_scene.

    Returns:
        dict with:
            "node":              rclpy.Node
            "move_group":        MoveGroupCommander
            "planning_scene":    PlanningSceneInterface
            "robot_commander":   RobotCommander
    """
    rclpy.init(args=None)
    node = rclpy.create_node("ur5_mujoco_moveit_bridge")

    move_group = MoveGroupCommander(
        PLANNING_GROUP,
        wait_for_servers=10.0,
    )
    move_group.set_planning_time(3.0)
    move_group.set_num_planning_attempts(3)

    planning_scene = PlanningSceneInterface()
    robot_commander = RobotCommander()

    return {
        "node": node,
        "move_group": move_group,
        "planning_scene": planning_scene,
        "robot_commander": robot_commander,
    }


def sync_mujoco_to_moveit(move_group, qpos, arm_dof=ARM_DOF_COUNT):
    """Copy MuJoCo joint state to MoveIt's current state."""
    joint_values = [float(qpos[i]) for i in range(arm_dof)]
    move_group.set_joint_value_target(joint_values)
    # set_joint_value_target also updates the internal RobotState for subsequent planning


def add_box_to_planning_scene(planning_scene):
    """Add placement box walls and base to MoveIt collision scene."""
    for wall in BOX_WALLS:
        planning_scene.add_box(
            name=wall["name"],
            size=wall["size"],
            pose=PoseStamped(
                pose=Pose(
                    position=Point(**dict(zip("xyz", wall["pos"]))),
                    orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                )
            ),
        )
    # Add base
    planning_scene.add_box(
        name="place_box_base",
        size=PLACE_BOX_BASE_SIZE,
        pose=PoseStamped(
            pose=Pose(
                position=Point(**dict(zip("xyz", PLACE_BOX_BASE_POS))),
                orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            )
        ),
    )


def plan_to_joint_target(move_group, target_q):
    """Plan a trajectory to a joint-space target. Returns RobotTrajectory or None."""
    target_q = [float(v) for v in target_q]
    move_group.set_joint_value_target(target_q)
    plan = move_group.plan()
    if not plan or not plan.joint_trajectory.points:
        return None
    return plan


def plan_to_pose_target(move_group, x, y, z):
    """Plan a trajectory to a Cartesian pose target. Returns RobotTrajectory or None."""
    move_group.set_position_target([x, y, z])
    plan = move_group.plan()
    if not plan or not plan.joint_trajectory.points:
        return None
    return plan


def trajectory_to_waypoints(plan, arm_dof=ARM_DOF_COUNT):
    """Extract joint-space waypoints from a MoveIt trajectory."""
    traj = plan.joint_trajectory
    waypoints = []
    for point in traj.points:
        wp = np.array([point.positions[i] for i in range(arm_dof)])
        waypoints.append(wp)
    return waypoints
```

- [ ] **Step 2: Commit**

---

### Task 3: MoveIt phase handlers

**Files:**
- Create: `ur5_ibvs_pbvs/src/task/pbvs_moveit_phases.py`

- [ ] **Step 1: Write handlers**

```python
"""
MoveIt-based phase handlers for lift, place, home.
"""

import cv2
import numpy as np

from src.task.pbvs_moveit_setup import (
    plan_to_joint_target,
    plan_to_pose_target,
    trajectory_to_waypoints,
    sync_mujoco_to_moveit,
)
from src.task.pbvs_mpc_phases import WaypointTracker


def handle_lift_phase_moveit(
    env, move_group, grasp_state_machine,
    current_site_pos, place_site_target_world,
    actuator_names, arm_dof_count, vis, height,
    *, wp_tracker=None, max_q_dot=3.0,
):
    """MoveIt plans lift trajectory, P controller executes it."""
    if wp_tracker is None:
        current_q = env.get_joint_positions(arm_dof_count)
        sync_mujoco_to_moveit(move_group, current_q)
        lift_target = grasp_state_machine.lift_target_pos_world

        # Compute goal joint config for lift target
        move_group.set_position_target(lift_target.tolist())
        goal_q = move_group.get_joint_value_target()
        if goal_q is None:
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        plan = plan_to_joint_target(move_group, goal_q)
        if plan is None:
            cv2.putText(vis, "LIFT: MOVEIT FAILED", (10, height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        waypoints = trajectory_to_waypoints(plan)
        wp_tracker = WaypointTracker(waypoints, kp=6.0, max_q_dot=max_q_dot, waypoint_tol=0.10)

    q_dot, done = wp_tracker.step(env.get_joint_positions(arm_dof_count))
    if done:
        q_dot[:] = 0.0
        grasp_state_machine.start_place(current_site_pos,
                                         place_target_pos_world=place_site_target_world)

    env.apply_joint_velocity(actuator_names, q_dot)
    cv2.putText(vis, f"LIFTING (MoveIt) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
                (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return q_dot, done, wp_tracker


def handle_place_phase_moveit(
    env, move_group, grasp_state_machine,
    current_site_pos, actuator_names, arm_dof_count, vis, height,
    *, wp_tracker=None, max_q_dot=4.0,
):
    """MoveIt plans collision-aware place trajectory."""
    if wp_tracker is None:
        current_q = env.get_joint_positions(arm_dof_count)
        sync_mujoco_to_moveit(move_group, current_q)
        target_pos = grasp_state_machine.place_target_pos_world

        # Set target above box walls
        target_pos = target_pos.copy()
        target_pos[2] = max(target_pos[2], 0.25)

        plan = plan_to_pose_target(move_group, *target_pos)
        if plan is None:
            cv2.putText(vis, "PLACE: MOVEIT FAILED", (10, height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        waypoints = trajectory_to_waypoints(plan)
        wp_tracker = WaypointTracker(waypoints, kp=6.0, max_q_dot=max_q_dot, waypoint_tol=0.10)

    q_dot, done = wp_tracker.step(env.get_joint_positions(arm_dof_count))
    if done:
        q_dot[:] = 0.0

    env.apply_joint_velocity(actuator_names, q_dot)
    cv2.putText(vis, f"PLACING (MoveIt) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
                (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    return q_dot, done, wp_tracker


def handle_home_phase_moveit(
    env, move_group, home_qpos, grasp_state_machine,
    actuator_names, arm_dof_count, vis, height,
    *, wp_tracker=None, max_q_dot=2.0,
):
    """MoveIt plans safe return-to-home trajectory."""
    if wp_tracker is None:
        current_q = env.get_joint_positions(arm_dof_count)
        sync_mujoco_to_moveit(move_group, current_q)
        plan = plan_to_joint_target(move_group, home_qpos)
        if plan is None:
            cv2.putText(vis, "HOME: MOVEIT FAILED", (10, height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        waypoints = trajectory_to_waypoints(plan)
        wp_tracker = WaypointTracker(waypoints, kp=6.0, max_q_dot=max_q_dot, waypoint_tol=0.10)

    q_dot, done = wp_tracker.step(env.get_joint_positions(arm_dof_count))
    if done:
        q_dot[:] = 0.0
        grasp_state_machine.mark_done()

    env.apply_joint_velocity(actuator_names, q_dot)
    cv2.putText(vis, f"RETURNING HOME (MoveIt) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
                (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    return q_dot, done, wp_tracker
```

- [ ] **Step 2: Commit**

---

### Task 4: Main entry point

**Files:**
- Create: `ur5_ibvs_pbvs/main_pbvs_moveit.py`

- [ ] **Step 1: Create main script (copy main_pbvs_mpc.py, modify phase dispatch)**

```python
"""UR5e visual servoing with MoveIt motion planning for lift/place/home."""

import time
import cv2
import mujoco
import numpy as np
import threading

from src.config import (
    ACTUATOR_NAMES, ARM_DOF_COUNT, BLIND_ATTACH_FRAMES, CONVEYOR_ENABLED,
    CONVEYOR_SPEED, ENABLE_GLOBAL_CAMERA_WINDOW, ERROR_SMOOTHING_ALPHA,
    GRASP_YAW_ABOUT_TAG_NORMAL_RAD, GRIPPER_ACTUATOR_NAME, GRIPPER_CLOSE_CTRL,
    GRIPPER_OPEN_CTRL, GUI_SLEEP, HEIGHT, HOME_JOINT_KP, HOME_JOINT_TOL,
    HOME_MAX_Q_DOT, LOST_TAG_DECAY, LOST_TAG_HOLD_FRAMES, MAX_PLACE_Q_DOT,
    MAX_Q_DOT, MAX_TRACKING_Q_DOT, MAX_TRANSPORT_Q_DOT, POSITION_DEADBAND,
    POSITION_SOFT_ZONE, REFERENCE_PREVIEW_GAIN, RELEASE_HOLD_FRAMES,
    ROTATION_DEADBAND, ROTATION_SOFT_ZONE, R_MJ_CAMERA_FROM_CV_CAMERA,
    SCENE_XML, SIM_STEPS_PER_CONTROL, SITE_NAME, TARGET_BODY_NAME,
    TARGET_MOTION_SPEED_THRESHOLD, USE_GLOBAL_CAMERA_FOR_SPEED, WIDTH,
)
from src.task.pbvs_mpc_phases import (
    handle_done_phase, handle_release_phase, handle_home_phase,
)
from src.task.pbvs_moveit_phases import (
    handle_lift_phase_moveit, handle_place_phase_moveit, handle_home_phase_moveit,
)
from src.task.pbvs_moveit_setup import init_moveit, add_box_to_planning_scene
from src.task.pbvs_mpc_runtime import (
    draw_runtime_overlay, get_gripper_ctrl_for_phase,
    project_image_point_to_world_plane, set_gripper_ctrl, update_target_pose_for_step,
)
from src.task.pbvs_mpc_setup import build_runtime
from src.task.pbvs_mpc_visual_servo import run_visual_servo_step
from src.perception.pose_estimator import draw_apriltags
from src.sim.rendering import MujocoRenderer
from src.utils.transforms import (
    build_desired_tag_camera_transform_from_grasp, make_transform,
)


def spin_ros(node):
    """Spin ROS2 node in background thread."""
    rclpy.spin(node)


def main():
    runtime = build_runtime(SCENE_XML)
    env = runtime["env"]
    model = runtime["model"]
    data = runtime["data"]
    controller_dt = runtime["controller_dt"]
    renderer = runtime["renderer"]
    robot_kin = runtime["robot_kin"]
    pose_estimator = runtime["pose_estimator"]
    pbvs_controller = runtime["pbvs_controller"]
    mpc_controller = runtime["mpc_controller"]
    tracking_mpc_controller = runtime["tracking_mpc_controller"]
    camera_id = runtime["camera_id"]
    camera_matrix = runtime["camera_matrix"]
    global_camera_id = runtime["global_camera_id"]
    global_camera_matrix = runtime["global_camera_matrix"]
    dist_coeffs = runtime["dist_coeffs"]
    detector = runtime["detector"]
    last_q_dot = runtime["last_q_dot"]
    gripper_target_ctrl = runtime["gripper_target_ctrl"]
    release_frames_remaining = runtime["release_frames_remaining"]
    lost_tag_count = runtime["lost_tag_count"]
    attached_target_offset_world = runtime["attached_target_offset_world"]
    attached_target_quat = runtime["attached_target_quat"]
    locked_approach_camera_rotation_world = runtime["locked_approach_camera_rotation_world"]
    last_e_p_cam = runtime["last_e_p_cam"]
    last_e_r_cam = runtime["last_e_r_cam"]
    conveyor_active = runtime["conveyor_active"]
    target_body_id = runtime["target_body_id"]
    target_mocap_id = runtime["target_mocap_id"]
    target_is_mocap = runtime["target_is_mocap"]
    conveyor_actuator_id = runtime["conveyor_actuator_id"]
    target_motion = runtime["target_motion"]
    grasp_state_machine = runtime["grasp_state_machine"]
    home_qpos = runtime["home_qpos"]
    place_site_target_world = runtime["place_site_target_world"]
    t_grasp_camera = runtime["t_grasp_camera"]
    global_renderer = runtime["global_renderer"]

    # ---- MoveIt initialization ----
    mit = init_moveit()
    move_group = mit["move_group"]
    planning_scene = mit["planning_scene"]
    ros_node = mit["node"]
    add_box_to_planning_scene(planning_scene)

    # Spin ROS in background thread
    ros_thread = threading.Thread(target=spin_ros, args=(ros_node,), daemon=True)
    ros_thread.start()

    prev_global_plane_point = None
    wp_tracker_lift = None
    wp_tracker_place = None
    wp_tracker_home = None

    try:
        with renderer.create_viewer() as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            while viewer.is_running():
                # ... (same conveyor, gripper, tag logic as main_pbvs_mpc.py)
                # ... (same global_camera logic)

                # ---- Phase dispatch ----
                control_applied = False

                if grasp_state_machine.phase == "release":
                    last_q_dot, release_frames_remaining = handle_release_phase(
                        env, ACTUATOR_NAMES, ARM_DOF_COUNT, vis, HEIGHT,
                        release_frames_remaining, grasp_state_machine)
                    control_applied = True

                if grasp_state_machine.phase == "lift" and grasp_state_machine.attached:
                    last_q_dot, lift_done, wp_tracker_lift = handle_lift_phase_moveit(
                        env, move_group, grasp_state_machine,
                        current_site_pos, place_site_target_world,
                        ACTUATOR_NAMES, ARM_DOF_COUNT, vis, HEIGHT,
                        wp_tracker=wp_tracker_lift, max_q_dot=MAX_TRANSPORT_Q_DOT)
                    control_applied = True

                if grasp_state_machine.phase == "place" and grasp_state_machine.attached:
                    last_q_dot, place_done, wp_tracker_place = handle_place_phase_moveit(
                        env, move_group, grasp_state_machine,
                        current_site_pos, ACTUATOR_NAMES, ARM_DOF_COUNT, vis, HEIGHT,
                        wp_tracker=wp_tracker_place, max_q_dot=MAX_PLACE_Q_DOT)
                    control_applied = True
                    if place_done:
                        release_frames_remaining = RELEASE_HOLD_FRAMES

                if grasp_state_machine.phase == "home":
                    last_q_dot, home_done, wp_tracker_home = handle_home_phase_moveit(
                        env, move_group, home_qpos, grasp_state_machine,
                        ACTUATOR_NAMES, ARM_DOF_COUNT, vis, HEIGHT,
                        wp_tracker=wp_tracker_home, max_q_dot=HOME_MAX_Q_DOT)
                    control_applied = True

                if grasp_state_machine.phase == "done":
                    last_q_dot = handle_done_phase(env, vis, ACTUATOR_NAMES, ARM_DOF_COUNT, HEIGHT)
                    control_applied = True

                if (not control_applied) and len(tags) > 0:
                    servo_result = run_visual_servo_step(...)
                    # ... (same visual servo handling as main_pbvs_mpc.py)

                # ... (same lost_tag handling, key handling, rendering)

    finally:
        if global_renderer is not None:
            global_renderer.close()
        renderer.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

Note: Full main loop omitted for brevity — copy entire remaining body from `main_pbvs_mpc.py`.

- [ ] **Step 2: Commit**

---

### Task 5: Verify

- [ ] **Step 1: Launch MoveIt**

```bash
cd ~/ur5-pbvs-mujoco-simulation/moveit2-learning/ur5_moveit_config
colcon build --packages-select ur5_moveit_config
source install/setup.bash
ros2 launch ur5_moveit_config move_group.launch.py
```

- [ ] **Step 2: Run main script**

```bash
cd ~/ur5-pbvs-mujoco-simulation/ur5_ibvs_pbvs
python3 main_pbvs_moveit.py
```

- [ ] **Step 3: Verify**

Visual servo works. Place phase avoids box walls. Gripper releases into box.

---

## Dependencies

- Task 2 depends on Task 1 (config values)
- Task 3 depends on Task 2 (setup functions)
- Task 4 depends on Task 3 (handlers)
- Task 5 depends on Task 4 (everything wired)
