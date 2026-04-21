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

    print("phase = lift")
    print("lift_position_error_world =", np.round(lift_position_error_world, 4))
    print("q_dot_lift =", np.round(q_dot_lift, 4))

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

    print("phase = place")
    print("place_position_error_world =", np.round(place_position_error_world, 4))
    print("q_dot_place =", np.round(q_dot_place, 4))
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

    print("phase = home")
    print("q_error =", np.round(q_error, 4))
    print("q_dot_home =", np.round(q_dot_home, 4))
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
