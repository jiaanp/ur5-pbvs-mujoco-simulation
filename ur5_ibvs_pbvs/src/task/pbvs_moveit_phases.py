"""
MoveIt-based phase handlers. Plan via MoveIt, execute via P-control through
MuJoCo velocity actuators (avoids direct qpos jump).
"""

import cv2
import numpy as np

from src.task.pbvs_mpc_phases import WaypointTracker
from src.task.pbvs_moveit_setup import (
    plan_to_pose_target_with_start,
    publish_joint_state,
    trajectory_to_waypoints,
)

# Joint mapping: MuJoCo ↔ URDF. sign * val + offset
Q_SIGNS  = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
Q_OFFSET = [np.pi, 0.0, 0.0, 0.0, 0.0, 0.0]


def _urdf_to_mujoco(urdf_q, arm_dof):
    """Convert URDF-space joint values to MuJoCo space."""
    mj_q = urdf_q.copy()
    for i in range(min(arm_dof, len(Q_SIGNS))):
        mj_q[i] = urdf_q[i] * Q_SIGNS[i] + Q_OFFSET[i]
    return mj_q


def _mujoco_to_urdf(mj_q, arm_dof):
    """Convert MuJoCo-space joint values to URDF space."""
    urdf_q = mj_q.copy()
    for i in range(min(arm_dof, len(Q_SIGNS))):
        urdf_q[i] = (mj_q[i] - Q_OFFSET[i]) / Q_SIGNS[i]
    return urdf_q


def handle_place_phase_moveit(
    env,
    model,
    data,
    arm,
    robot_model,
    joint_state_pub,
    grasp_state_machine,
    current_site_pos,
    actuator_names,
    arm_dof_count,
    vis,
    height,
    *,
    wp_tracker=None,
):
    """
    First call: MoveIt plan to place target, create WaypointTracker.
    Subsequent calls: track waypoints via P-control (velocity actuators).

    Returns:
        (q_dot, done, wp_tracker)
    """
    if wp_tracker is None:
        target_pos = grasp_state_machine.place_target_pos_world.copy()
        target_pos[2] = max(target_pos[2], 0.25)

        # Current q → URDF → set as MoveIt start state
        current_q_mj = env.get_joint_positions(arm_dof_count)
        start_q_urdf = _mujoco_to_urdf(current_q_mj, arm_dof_count)

        traj = plan_to_pose_target_with_start(
            arm, robot_model, *target_pos, start_q_urdf)
        if traj is None:
            cv2.putText(
                vis, "PLACE: MOVEIT FAILED",
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        # Extract waypoints, convert to MuJoCo space
        waypoints_urdf = trajectory_to_waypoints(traj)
        waypoints_mj = [_urdf_to_mujoco(wp, arm_dof_count) for wp in waypoints_urdf]
        # Prepend current q so tracking starts from where arm actually is
        waypoints_mj.insert(0, current_q_mj.copy())

        wp_tracker = WaypointTracker(waypoints_mj, kp=10.0, max_q_dot=4.0, waypoint_tol=0.25)
        print(f"[PLACE] MoveIt planned {len(waypoints_mj)} waypoints (MuJoCo space)")

        # Publish first joint state for RViz
        publish_joint_state(joint_state_pub, waypoints_urdf[0] if waypoints_urdf else current_q_mj)

    # Track waypoints via velocity actuators
    q_dot, done = wp_tracker.step(env.get_joint_positions(arm_dof_count))

    if done:
        q_dot[:] = 0.0

    env.apply_joint_velocity(actuator_names, q_dot)

    cv2.putText(
        vis,
        f"PLACING (MoveIt) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
    )

    return q_dot, done, wp_tracker
