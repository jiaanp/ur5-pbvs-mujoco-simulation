"""
MoveIt-based phase handlers for lift, place, home motion.
Each handler: sync MuJoCo state -> MoveIt plan -> WaypointTracker execute.
"""

import cv2
import numpy as np

from src.task.pbvs_mpc_phases import WaypointTracker
from src.task.pbvs_moveit_setup import (
    plan_to_joint_target,
    plan_to_pose_target,
    trajectory_to_waypoints,
    sync_mujoco_to_moveit,
)


def handle_lift_phase_moveit(
    env,
    move_group,
    grasp_state_machine,
    current_site_pos,
    place_site_target_world,
    actuator_names,
    arm_dof_count,
    vis,
    height,
    *,
    wp_tracker=None,
    max_q_dot=3.0,
):
    """
    MoveIt plans a joint-space lift trajectory. On first call, syncs current
    MuJoCo state into MoveIt and plans. On subsequent calls, tracks waypoints
    with P control. On completion, transitions to place phase.

    Returns:
        (q_dot, done, wp_tracker): q_dot ndarray[6], done bool, WaypointTracker or None
    """
    if wp_tracker is None:
        # --- First call: plan via MoveIt ---
        current_q = env.get_joint_positions(arm_dof_count)
        sync_mujoco_to_moveit(move_group, current_q)

        # Use the lift target from the state machine
        lift_target = grasp_state_machine.lift_target_pos_world

        # MoveIt position target -> IK internally -> joint trajectory
        plan = plan_to_pose_target(move_group, *lift_target)
        if plan is None:
            cv2.putText(
                vis, "LIFT: MOVEIT FAILED",
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        waypoints = trajectory_to_waypoints(plan)
        wp_tracker = WaypointTracker(waypoints, kp=6.0, max_q_dot=max_q_dot, waypoint_tol=0.10)

    # --- Track waypoints ---
    q_dot, done = wp_tracker.step(env.get_joint_positions(arm_dof_count))

    if done:
        q_dot[:] = 0.0
        grasp_state_machine.start_place(
            current_site_pos,
            place_target_pos_world=place_site_target_world,
        )

    env.apply_joint_velocity(actuator_names, q_dot)

    cv2.putText(
        vis,
        f"LIFTING (MoveIt) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
    )

    return q_dot, done, wp_tracker


def handle_place_phase_moveit(
    env,
    move_group,
    grasp_state_machine,
    current_site_pos,
    actuator_names,
    arm_dof_count,
    vis,
    height,
    *,
    wp_tracker=None,
    max_q_dot=4.0,
):
    """
    MoveIt plans a collision-aware Cartesian place trajectory. Box walls
    must already be in the PlanningScene (add_box_to_planning_scene).

    Returns:
        (q_dot, done, wp_tracker)
    """
    if wp_tracker is None:
        current_q = env.get_joint_positions(arm_dof_count)
        sync_mujoco_to_moveit(move_group, current_q)

        target_pos = grasp_state_machine.place_target_pos_world.copy()
        # Ensure target is above box walls (walls top at z=0.158)
        target_pos[2] = max(target_pos[2], 0.25)

        plan = plan_to_pose_target(move_group, *target_pos)
        if plan is None:
            cv2.putText(
                vis, "PLACE: MOVEIT FAILED",
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        waypoints = trajectory_to_waypoints(plan)
        wp_tracker = WaypointTracker(waypoints, kp=6.0, max_q_dot=max_q_dot, waypoint_tol=0.10)

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


def handle_home_phase_moveit(
    env,
    move_group,
    home_qpos,
    grasp_state_machine,
    actuator_names,
    arm_dof_count,
    vis,
    height,
    *,
    wp_tracker=None,
    max_q_dot=2.0,
):
    """
    MoveIt plans a collision-aware joint-space trajectory back to home.

    Returns:
        (q_dot, done, wp_tracker)
    """
    if wp_tracker is None:
        current_q = env.get_joint_positions(arm_dof_count)
        sync_mujoco_to_moveit(move_group, current_q)

        plan = plan_to_joint_target(move_group, home_qpos)
        if plan is None:
            cv2.putText(
                vis, "HOME: MOVEIT FAILED",
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        waypoints = trajectory_to_waypoints(plan)
        wp_tracker = WaypointTracker(waypoints, kp=6.0, max_q_dot=max_q_dot, waypoint_tol=0.10)

    q_dot, done = wp_tracker.step(env.get_joint_positions(arm_dof_count))

    if done:
        q_dot[:] = 0.0
        grasp_state_machine.mark_done()

    env.apply_joint_velocity(actuator_names, q_dot)

    cv2.putText(
        vis,
        f"RETURNING HOME (MoveIt) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2,
    )

    return q_dot, done, wp_tracker
