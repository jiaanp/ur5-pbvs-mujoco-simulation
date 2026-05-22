import time
import numpy as np
import mujoco
import ompl.base as ob
import ompl.geometric as og


def plan_joint_space(model, data, start_q, goal_q,
                     target_geom_prefixes=None,
                     arm_dof=6,
                     planning_time=3.0,
                     planning_range=0.2,
                     verbose=True):
    """
    In UR5e's 6D joint space, use RRTConnect to plan a collision-free trajectory.

    Collision detection: classifies geoms into robot (body_id > 0) vs environment (body_id == 0).
    At the start configuration, records baseline (robot, env) contact pairs that are allowed.
    Any NEW (robot, env) contact during planning is treated as illegal.

    Args:
        model, data:          MuJoCo model and data
        start_q, goal_q:      Start and goal joint configurations (ndarray[6])
        target_geom_prefixes: List of geom name prefixes to treat as obstacles.
                              None means check ALL new (robot, env) contacts.
        arm_dof:              Number of arm degrees of freedom
        planning_time:        Planning timeout in seconds
        planning_range:       RRT extension step size in radians

    Returns:
        List[ndarray[6]] | None: waypoints on success, None on failure
    """
    # Classify geoms: robot (body_id > 0) vs environment (body_id == 0)
    robot_geom_ids = set()
    env_geom_ids = set()
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == 0:
            env_geom_ids.add(gi)
        else:
            robot_geom_ids.add(gi)

    # Parse target_geom_prefixes to identify obstacle geoms
    if target_geom_prefixes is None:
        target_geom_prefixes = []
    target_geom_ids = set()
    for gi in env_geom_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if name and any(name.startswith(p) for p in target_geom_prefixes):
            target_geom_ids.add(gi)
    if verbose and target_geom_ids:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
                 for gi in sorted(target_geom_ids)]
        print(f"[OMPL] Obstacle geoms: {names}")

    # Baseline contacts at start config
    data.qpos[:arm_dof] = start_q.copy()
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_collision(model, data)

    baseline_pairs = set()
    for ci in range(data.ncon):
        g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
        if g1 in robot_geom_ids and g2 in env_geom_ids:
            baseline_pairs.add((g1, g2))
        elif g2 in robot_geom_ids and g1 in env_geom_ids:
            baseline_pairs.add((g2, g1))

    # State space
    ss = ob.RealVectorStateSpace(arm_dof)
    bounds = ob.RealVectorBounds(arm_dof)
    for i in range(arm_dof):
        bounds.setLow(i, model.jnt_range[i, 0])
        bounds.setHigh(i, model.jnt_range[i, 1])
    ss.setBounds(bounds)
    si = ob.SpaceInformation(ss)

    # Collision checker
    class Checker(ob.StateValidityChecker):
        def isValid(self, state):
            data.qpos[:arm_dof] = [state[i] for i in range(arm_dof)]
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_collision(model, data)
            for ci in range(data.ncon):
                g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
                if g1 in robot_geom_ids and g2 in env_geom_ids:
                    if (g1, g2) not in baseline_pairs:
                        return False
                elif g2 in robot_geom_ids and g1 in env_geom_ids:
                    if (g2, g1) not in baseline_pairs:
                        return False
            return True

    si.setStateValidityChecker(Checker(si))
    si.setup()

    # Start and goal states
    start_s = ss.allocState()
    goal_s = ss.allocState()
    for i in range(arm_dof):
        start_s[i] = float(start_q[i])
        goal_s[i] = float(goal_q[i])

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_s, goal_s)
    pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    # RRTConnect
    planner = og.RRTConnect(si)
    planner.setRange(planning_range)
    planner.setIntermediateStates(True)
    planner.setProblemDefinition(pdef)
    planner.setup()

    if verbose:
        print(f"[OMPL] Planning (timeout={planning_time}s, range={planning_range})...")
    t0 = time.perf_counter()
    solved = planner.solve(planning_time)
    elapsed = time.perf_counter() - t0

    if not solved and not pdef.hasApproximateSolution():
        if verbose:
            print(f"[OMPL] Planning FAILED ({elapsed:.3f}s)")
        return None

    sol_path = pdef.getSolutionPath() if pdef.hasSolution() else pdef.getApproximateSolutionPath()
    waypoints = []
    for i in range(sol_path.getStateCount()):
        st = sol_path.getState(i)
        waypoints.append(np.array([st[j] for j in range(arm_dof)]))

    if verbose:
        print(f"[OMPL] Planning SUCCESS ({elapsed:.3f}s), waypoints={len(waypoints)}")

    return waypoints
