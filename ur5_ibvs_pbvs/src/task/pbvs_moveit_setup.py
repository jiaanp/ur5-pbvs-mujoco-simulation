"""
MoveIt initialization for UR5e motion planning (MoveItPy API).
Bridge between MuJoCo simulation and MoveIt 2 via moveit.planning.
"""

import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

from src.config import (
    ARM_DOF_COUNT,
    BOX_WALLS,
    PLACE_BOX_BASE_POS,
    PLACE_BOX_BASE_SIZE,
    PLANNING_GROUP,
)

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _build_config_dict():
    configs = MoveItConfigsBuilder("ur5", package_name="ur5_moveit_config").to_moveit_configs()
    d = configs.to_dict()
    if isinstance(d.get("planning_pipelines"), list):
        d["planning_pipelines"] = {"pipeline_names": d["planning_pipelines"]}
    for pk in ["ompl", "chomp", "pilz_industrial_motion_planner"]:
        pipeline = d.get(pk, {})
        if "planner_configs" in pipeline:
            pipeline.setdefault("arm", {})
            pipeline["arm"]["planner_configs"] = list(pipeline["planner_configs"].keys())
    # Force RRTstar as default planner (finds shorter, less circuitous paths)
    d.setdefault("ompl", {})
    d["ompl"].setdefault("arm", {})
    d["ompl"]["arm"]["default_planner_config"] = "RRTstar"
    d["ompl"].setdefault("planner_configs", {})
    d["ompl"]["planner_configs"]["RRTstar"] = {
        "type": "geometric::RRTstar",
        "range": 0.5,
    }
    d["plan_request_params"] = {"planning_pipeline": "ompl"}
    d["moveit_simple_controller_manager"] = {
        "controller_names": ["arm_controller"],
        "arm_controller": {
            "type": "FollowJointTrajectory",
            "joints": JOINT_NAMES,
            "action_ns": "follow_joint_trajectory",
        },
    }
    return d


def _make_box_collision(name, x, y, z, sx, sy, sz):
    obj = CollisionObject()
    obj.id = name
    obj.header.frame_id = "base_link"
    obj.operation = CollisionObject.ADD
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [float(sx), float(sy), float(sz)]
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    obj.primitives = [box]
    obj.primitive_poses = [pose]
    return obj


def init_moveit():
    """Initialize ROS2 + MoveItPy. Returns dict with arm, publisher, etc."""
    rclpy.init(args=None)

    config_dict = _build_config_dict()
    moveit_py = MoveItPy(config_dict=config_dict, node_name="ur5_mujoco_moveit_bridge")
    arm = moveit_py.get_planning_component(PLANNING_GROUP)

    # ROS node for collision objects + joint state publishing
    bridge_node = Node("ur5_mujoco_bridge")
    collision_pub = bridge_node.create_publisher(CollisionObject, "/collision_object", 10)
    joint_state_pub = bridge_node.create_publisher(JointState, "/joint_states", 10)

    # Multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(bridge_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(0.3)

    # Add floor to prevent planning below ground (z=0)
    collision_pub.publish(_make_box_collision("floor", 0.0, 0.0, -0.01, 2.0, 2.0, 0.02))

    # Add box obstacles
    for wall in BOX_WALLS:
        collision_pub.publish(_make_box_collision(
            wall["name"],
            wall["pos"][0], wall["pos"][1], wall["pos"][2],
            wall["size"][0], wall["size"][1], wall["size"][2],
        ))
    collision_pub.publish(_make_box_collision(
        "place_box_base",
        PLACE_BOX_BASE_POS[0], PLACE_BOX_BASE_POS[1], PLACE_BOX_BASE_POS[2],
        PLACE_BOX_BASE_SIZE[0], PLACE_BOX_BASE_SIZE[1], PLACE_BOX_BASE_SIZE[2],
    ))
    time.sleep(0.5)

    robot_model = moveit_py.get_robot_model()

    return {
        "moveit_py": moveit_py,
        "arm": arm,
        "robot_model": robot_model,
        "joint_state_pub": joint_state_pub,
        "bridge_node": bridge_node,
    }


def publish_joint_state(pub, qpos, arm_dof=ARM_DOF_COUNT):
    """Publish current joint positions as /joint_states for RViz."""
    msg = JointState()
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.name = JOINT_NAMES[:arm_dof]
    msg.position = [float(qpos[i]) for i in range(arm_dof)]
    pub.publish(msg)


def plan_to_pose_target_with_start(arm, robot_model, x, y, z, start_q_urdf):
    """
    Plan trajectory to Cartesian target, starting from given joint config (URDF space).
    MoveIt's fix_start_state_collision adapter handles any slight model mismatch.
    """
    start_state = RobotState(robot_model)
    start_state.set_joint_group_positions(
        PLANNING_GROUP, [float(v) for v in start_q_urdf[:6]],
    )
    arm.set_start_state(robot_state=start_state)

    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)
    # Orientation: tool0 Z pointing downward (world -Z)
    # Quaternion = 180° about Y: cos(90°)=0, sin(90°)*(0,1,0) = (0,1,0)
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 1.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 0.0
    arm.set_goal_state(pose_stamped_msg=pose, pose_link="tool0")
    result = arm.plan()
    if result and result.trajectory is not None:
        return result.trajectory
    return None


def plan_to_pose_target(arm, x, y, z):
    """Plan trajectory to Cartesian position target from current internal state."""
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)
    # Orientation: tool0 Z pointing downward (world -Z)
    # Quaternion = 180° about Y: cos(90°)=0, sin(90°)*(0,1,0) = (0,1,0)
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 1.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 0.0
    arm.set_goal_state(pose_stamped_msg=pose, pose_link="tool0")
    result = arm.plan()
    if result and result.trajectory is not None:
        return result.trajectory
    return None


def trajectory_to_waypoints(traj, arm_dof=ARM_DOF_COUNT):
    """Extract joint-space waypoints from a MoveIt RobotTrajectory."""
    waypoints = []
    if traj is None:
        return waypoints
    msg = traj.get_robot_trajectory_msg()
    for point in msg.joint_trajectory.points:
        wp = np.array([point.positions[i] for i in range(arm_dof)])
        waypoints.append(wp)
    return waypoints
