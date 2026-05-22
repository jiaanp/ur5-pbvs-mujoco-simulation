import cv2
import numpy as np

from src.task.pbvs_mpc_runtime import solve_cartesian_transport_qdot


def handle_release_phase(
    env,
    actuator_names,
    arm_dof_count,
    vis,
    height,
    release_frames_remaining,
    grasp_state_machine,
):
    env.zero_joint_velocity(actuator_names)
    last_q_dot = np.zeros(arm_dof_count, dtype=np.float64)
    cv2.putText(
        vis,
        "RELEASING OBJECT",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 0),
        2,
    )
    release_frames_remaining = max(release_frames_remaining - 1, 0)
    if release_frames_remaining == 0:
        grasp_state_machine.start_home()
    return last_q_dot, release_frames_remaining


def handle_lift_phase(
    env,
    robot_kin,
    mpc_controller,
    grasp_state_machine,
    current_site_pos,
    place_site_target_world,
    last_q_dot,
    vis,
    actuator_names,
    arm_dof_count,
    height,
    max_transport_q_dot,
):
    lift_position_error_world, lift_done = grasp_state_machine.get_lift_error_world(current_site_pos)
    q_dot_lift = solve_cartesian_transport_qdot(
        env,
        robot_kin,
        mpc_controller,
        lift_position_error_world,
        last_q_dot,
        max_transport_q_dot,
        arm_dof_count,
    )

    if lift_done:
        q_dot_lift[:] = 0.0

    env.apply_joint_velocity(actuator_names, q_dot_lift)
    cv2.putText(
        vis,
        "LIFTING ATTACHED TARGET",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    if lift_done:
        grasp_state_machine.start_place(
            current_site_pos,
            place_target_pos_world=place_site_target_world,
        )

    return q_dot_lift


def handle_place_phase(
    env,
    robot_kin,
    mpc_controller,
    grasp_state_machine,
    current_site_pos,
    last_q_dot,
    vis,
    actuator_names,
    arm_dof_count,
    height,
    max_place_q_dot,
):
    place_position_error_world, place_done = grasp_state_machine.get_place_error_world(current_site_pos)
    q_dot_place = solve_cartesian_transport_qdot(
        env,
        robot_kin,
        mpc_controller,
        place_position_error_world,
        last_q_dot,
        max_place_q_dot,
        arm_dof_count,
    )

    if place_done:
        q_dot_place[:] = 0.0

    env.apply_joint_velocity(actuator_names, q_dot_place)
    cv2.putText(
        vis,
        "PLACING ATTACHED TARGET",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )

    return q_dot_place, place_done


def handle_home_phase(
    env,
    home_qpos,
    grasp_state_machine,
    vis,
    actuator_names,
    arm_dof_count,
    height,
    home_joint_kp,
    home_joint_tol,
    home_max_q_dot,
):
    current_qpos = env.get_joint_positions(arm_dof_count)
    q_error = home_qpos - current_qpos
    q_dot_home = np.clip(home_joint_kp * q_error, -home_max_q_dot, home_max_q_dot)
    home_done = np.linalg.norm(q_error) < home_joint_tol

    if home_done:
        q_dot_home[:] = 0.0
        grasp_state_machine.mark_done()

    env.apply_joint_velocity(actuator_names, q_dot_home)
    cv2.putText(
        vis,
        "RETURNING HOME",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
    )

    return q_dot_home


def handle_done_phase(env, vis, actuator_names, arm_dof_count, height):
    env.zero_joint_velocity(actuator_names)
    cv2.putText(
        vis,
        "TASK COMPLETE",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return np.zeros(arm_dof_count, dtype=np.float64)


# ===========================================================================
# OMPL obstacle-aware place phase
# ===========================================================================


class WaypointTracker:
    """Joint-space waypoint tracker using P control."""

    def __init__(self, waypoints, kp=4.0, max_q_dot=1.5, waypoint_tol=0.05):
        self.waypoints = waypoints
        self.index = 0
        self.kp = float(kp)
        self.max_q_dot = float(max_q_dot)
        self.waypoint_tol = float(waypoint_tol)
        self.done = False

    def step(self, current_q):
        """Compute joint velocity command to track next waypoint.

        Returns:
            (q_dot, done): q_dot ndarray[6], done bool
        """
        if self.done:
            return np.zeros(len(current_q)), True

        target = self.waypoints[self.index] if self.index < len(self.waypoints) else self.waypoints[-1]
        current_q = np.asarray(current_q, dtype=np.float64)
        q_error = target - current_q
        q_dot = np.clip(self.kp * q_error, -self.max_q_dot, self.max_q_dot)

        if np.linalg.norm(q_error) < self.waypoint_tol:
            self.index += 1
            if self.index >= len(self.waypoints):
                self.done = True

        return q_dot, self.done


def handle_place_phase_ompl(
    env,
    model,
    data,
    robot_kin,
    grasp_state_machine,
    current_site_pos,
    actuator_names,
    arm_dof_count,
    vis,
    height,
    *,
    wp_tracker=None,
    max_q_dot=1.5,
):
    """
    First call: run IK + OMPL planning, create WaypointTracker.
    Subsequent calls: track waypoints via P control.

    Returns:
        (q_dot, done, wp_tracker)
    """
    if wp_tracker is None:
        current_q = env.get_joint_positions(arm_dof_count)
        place_target_world = grasp_state_machine.place_target_pos_world

        # Target: 5cm above box floor
        drop_pos = place_target_world.copy()
        drop_pos[2] = 0.05

        # IK
        goal_q = robot_kin.solve_ik_position(drop_pos, current_q)

        # OMPL planning (avoid box walls)
        from src.planning.ompl_planner import plan_joint_space

        waypoints = plan_joint_space(
            model, data, current_q, goal_q,
            target_geom_prefixes=["place_box_wall_"],
            arm_dof=arm_dof_count,
            planning_time=3.0,
            planning_range=0.2,
        )

        if waypoints is None:
            cv2.putText(
                vis, "PLACE: OMPL FAILED",
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        wp_tracker = WaypointTracker(waypoints, max_q_dot=max_q_dot)

    q_dot, done = wp_tracker.step(env.get_joint_positions(arm_dof_count))

    if done:
        q_dot[:] = 0.0

    env.apply_joint_velocity(actuator_names, q_dot)

    cv2.putText(
        vis,
        f"PLACING (OMPL) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
    )

    return q_dot, done, wp_tracker


def handle_lift_phase_ompl(
    env,
    model,
    data,
    robot_kin,
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
    First call: run IK + OMPL planning to lift target, create WaypointTracker.
    Subsequent calls: track waypoints via P control.
    On completion: transitions state machine to place phase.

    Returns:
        (q_dot, done, wp_tracker)
    """
    if wp_tracker is None:
        current_q = env.get_joint_positions(arm_dof_count)
        lift_target = grasp_state_machine.lift_target_pos_world

        # IK to lift target
        goal_q = robot_kin.solve_ik_position(lift_target, current_q)

        from src.planning.ompl_planner import plan_joint_space

        waypoints = plan_joint_space(
            model, data, current_q, goal_q,
            target_geom_prefixes=[],  # no specific obstacle to avoid during lift
            arm_dof=arm_dof_count,
            planning_time=1.0,
            planning_range=0.2,
        )

        if waypoints is None:
            cv2.putText(
                vis, "LIFT: OMPL FAILED",
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        wp_tracker = WaypointTracker(waypoints, max_q_dot=max_q_dot)

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
        f"LIFTING (OMPL) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
    )

    return q_dot, done, wp_tracker


def handle_home_phase_ompl(
    env,
    model,
    data,
    robot_kin,
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
    First call: OMPL planning from current q to home_qpos, create WaypointTracker.
    Subsequent calls: track waypoints via P control.
    On completion: marks done in state machine.

    Returns:
        (q_dot, done, wp_tracker)
    """
    if wp_tracker is None:
        current_q = env.get_joint_positions(arm_dof_count)
        goal_q = np.asarray(home_qpos, dtype=np.float64)

        from src.planning.ompl_planner import plan_joint_space

        waypoints = plan_joint_space(
            model, data, current_q, goal_q,
            target_geom_prefixes=["place_box_wall_"],  # avoid box on way home
            arm_dof=arm_dof_count,
            planning_time=2.0,
            planning_range=0.2,
        )

        if waypoints is None:
            # Fallback: fall through to direct P control in main script
            return np.zeros(arm_dof_count, dtype=np.float64), False, None

        wp_tracker = WaypointTracker(waypoints, max_q_dot=max_q_dot)

    q_dot, done = wp_tracker.step(env.get_joint_positions(arm_dof_count))

    if done:
        q_dot[:] = 0.0
        grasp_state_machine.mark_done()

    env.apply_joint_velocity(actuator_names, q_dot)

    cv2.putText(
        vis,
        f"RETURNING HOME (OMPL) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2,
    )

    return q_dot, done, wp_tracker
