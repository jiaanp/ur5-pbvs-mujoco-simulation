from src.robotics.pinocchio_kinematics import PinocchioKinematics
import numpy as np

robot_kin = PinocchioKinematics(
    urdf_path="/home/adrian/models/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf",
    ee_frame_name="tool0",
    arm_dof_count=6,
)

q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float64)

pos, rot, T = robot_kin.forward_kinematics(q)
print(pos)
print(rot)

J = robot_kin.compute_frame_jacobian(q)
print(J)

v = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.1], dtype=np.float64)
q_dot, _ = robot_kin.compute_joint_velocity_from_ee_velocity(q, v, damping=0.02)
print(q_dot)
