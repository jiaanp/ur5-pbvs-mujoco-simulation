"""
MoveIt initialization for UR5e motion planning.
Bridge between MuJoCo simulation and MoveIt 2 motion planning.
"""

import time
import numpy as np
import rclpy
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from shape_msgs.msg import SolidPrimitive

from src.config import (
    ARM_DOF_COUNT,
    BOX_WALLS,
    PLACE_BOX_BASE_POS,
    PLACE_BOX_BASE_SIZE,
    PLANNING_GROUP,
)

# Joint names in order -- must match MuJoCo's ordering
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def init_moveit(wait_time=10.0):
    """
    Initialize ROS2 + MoveIt components.

    Returns:
        dict with keys: "node", "move_group", "planning_scene", "robot_commander"
    """
    rclpy.init(args=None)
    node = rclpy.create_node("ur5_mujoco_moveit_bridge")

    move_group = MoveGroupCommander(
        PLANNING_GROUP,
        wait_for_servers=wait_time,
    )
    # Set planning parameters
    move_group.set_planning_time(3.0)
    move_group.set_num_planning_attempts(3)
    move_group.set_max_velocity_scaling_factor(0.5)
    move_group.set_max_acceleration_scaling_factor(0.5)

    planning_scene = PlanningSceneInterface(synchronous=True)
    robot_commander = RobotCommander()

    return {
        "node": node,
        "move_group": move_group,
        "planning_scene": planning_scene,
        "robot_commander": robot_commander,
    }


def sync_mujoco_to_moveit(move_group, qpos, arm_dof=ARM_DOF_COUNT):
    """
    Copy MuJoCo current joint positions into MoveIt's RobotState.
    Call this before each planning request so MoveIt plans from the
    actual current arm configuration.
    """
    joint_values = {name: float(qpos[i]) for i, name in enumerate(JOINT_NAMES)}
    move_group.set_joint_value_target(joint_values)
    # set_joint_value_target with dict also updates internal RobotState


def add_box_to_planning_scene(planning_scene):
    """
    Add placement box (base + 4 walls) to MoveIt's collision scene.
    Geometry comes from config.BOX_WALLS, PLACE_BOX_BASE_POS/SIZE.
    """
    # Small delay to ensure PlanningScene is ready
    time.sleep(1.0)

    # Add walls
    for wall in BOX_WALLS:
        p = wall["pos"]
        s = wall["size"]
        planning_scene.add_box(
            name=wall["name"],
            size=s,
            pose=PoseStamped(
                header={"frame_id": "world"},
                pose=Pose(
                    position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                    orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
            ),
        )

    # Add base
    planning_scene.add_box(
        name="place_box_base",
        size=PLACE_BOX_BASE_SIZE,
        pose=PoseStamped(
            header={"frame_id": "world"},
            pose=Pose(
                position=Point(x=float(PLACE_BOX_BASE_POS[0]),
                               y=float(PLACE_BOX_BASE_POS[1]),
                               z=float(PLACE_BOX_BASE_POS[2])),
                orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
        ),
    )


def plan_to_joint_target(move_group, target_q):
    """
    Plan a trajectory to a joint-space target configuration.

    Args:
        move_group: MoveGroupCommander
        target_q: list or ndarray of 6 joint angles

    Returns:
        RobotTrajectory on success, None on failure
    """
    target_q = [float(v) for v in target_q]
    move_group.set_joint_value_target(target_q)
    plan = move_group.plan()
    if not plan or not plan.joint_trajectory.points:
        return None
    return plan


def plan_to_pose_target(move_group, x, y, z):
    """
    Plan a trajectory to a Cartesian position target (orientation unconstrained).

    Args:
        move_group: MoveGroupCommander
        x, y, z: world-frame target position

    Returns:
        RobotTrajectory on success, None on failure
    """
    move_group.set_position_target([float(x), float(y), float(z)])
    plan = move_group.plan()
    if not plan or not plan.joint_trajectory.points:
        return None
    return plan


def trajectory_to_waypoints(plan, arm_dof=ARM_DOF_COUNT):
    """
    Extract joint-space waypoints from a MoveIt trajectory plan.

    Args:
        plan: MoveIt plan result (RobotTrajectory)
        arm_dof: number of arm joints to extract

    Returns:
        list of ndarray[arm_dof], one per trajectory point
    """
    traj = plan.joint_trajectory
    waypoints = []
    for point in traj.points:
        wp = np.array([point.positions[i] for i in range(arm_dof)])
        waypoints.append(wp)
    return waypoints
