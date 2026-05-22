# -*- coding: utf-8 -*-
"""
Demo 2: 换规划器 + 路径后处理
================================

在同一个规划问题中 (同一障碍物、同一起止构型), 分别用三种规划器求解:

  - RRTConnect : 双向 RRT, 速度快, 为大多数场景首选
  - RRTstar    : 渐近最优 RRT, 路径质量随时间逐步改善
  - KPIECE1    : 基于网格投影的探索, 高维空间 (>6 DOF) 更鲁棒

最后对路径做:
  1. PathSimplifier.simplifyMax() — 删除冗余 waypoint
  2. PathSimplifier.smoothBSpline() — B-spline 平滑

运行方式:
  python3 demo_ompl_planners.py
"""

import os
import time
import tempfile
from xml.etree import ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

import ompl.base as ob
import ompl.geometric as og

# ===========================================================================
# 全局常量 (与 demo_ompl_obstacle.py 完全一致)
# ===========================================================================

SCENE_XML = os.path.join(os.path.dirname(__file__), "model", "scene_with_gripper.xml")
ARM_DOF = 6

ACTUATOR_NAMES = [
    "shoulder_pan_vel_init",
    "shoulder_lift_vel_init",
    "elbow_vel_init",
    "wrist_1_vel_init",
    "wrist_2_vel_init",
    "wrist_3_vel_init",
]

OBSTACLES = [
    {
        "type": "sphere",
        "size": [0.10],
        "pos": [-0.72, 0.38, 0.28],
        "rgba": [1.0, 0.2, 0.2, 0.9],
    },
]

START_Q = np.zeros(6)
GOAL_Q  = np.array([-1.2, -0.8, 1.2, -1.2, -1.5, 0.0])

PLANNING_TIME = 2.0   # 每个规划器的超时 (秒), RRTstar 需要更多时间
PLANNING_RANGE = 0.2  # RRT 步长 (rad)


# ===========================================================================
# 复用 demo_ompl_obstacle.py 的障碍物场景构建函数
# ===========================================================================

def add_obstacles_to_scene(scene_path, obstacles):
    """与 demo_ompl_obstacle.py 完全一致"""
    tree = ET.parse(scene_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("场景 XML 中未找到 <worldbody>")

    for i, obs in enumerate(obstacles):
        geom = ET.SubElement(worldbody, "geom")
        geom.set("name", f"obstacle_{i}")
        geom.set("type", obs["type"])
        geom.set("size", " ".join(f"{x:.4f}" for x in obs["size"]))
        geom.set("pos", f"{obs['pos'][0]:.4f} {obs['pos'][1]:.4f} {obs['pos'][2]:.4f}")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("rgba", f"{obs['rgba'][0]:.3f} {obs['rgba'][1]:.3f} {obs['rgba'][2]:.3f} {obs['rgba'][3]:.3f}")

    model_dir = os.path.dirname(os.path.abspath(scene_path))
    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", prefix="scene_obs_", delete=False,
        mode="w", encoding="utf-8", dir=model_dir,
    )
    tree.write(tmp.name, encoding="utf-8", xml_declaration=True)
    return tmp.name


def load_model():
    """与 demo_ompl_obstacle.py 完全一致"""
    xml_path = add_obstacles_to_scene(SCENE_XML, OBSTACLES)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data, xml_path


# ===========================================================================
# 通用 OMPL 规划函数 — 接受规划器类型作为参数
# ===========================================================================

def plan_with_planner(model, data, start_q, goal_q, planner_type):
    """
    用指定规划器在关节空间求解。

    参数:
        planner_type : "RRTConnect" | "RRTstar" | "KPIECE1"

    返回:
        dict {
            "waypoints": List[ndarray[6]],   # 路径点列表
            "time":      float,              # 规划耗时 (秒)
            "calls":     int,                # isValid 回调次数
            "states":    int,                # OMPL 内部探索的状态数
        }
        失败时返回 None
    """
    # --- 状态空间 (与之前完全一致) ---
    ss = ob.RealVectorStateSpace(ARM_DOF)
    bounds = ob.RealVectorBounds(ARM_DOF)
    for i in range(ARM_DOF):
        bounds.setLow(i, model.jnt_range[i, 0])
        bounds.setHigh(i, model.jnt_range[i, 1])
    ss.setBounds(bounds)

    si = ob.SpaceInformation(ss)

    # --- 分类 geom: 机械臂 (body>0) vs 环境 (body=0) ---
    robot_geom_ids = set()
    env_geom_ids = set()
    for gi in range(model.ngeom):
        body_id = model.geom_bodyid[gi]
        if body_id == 0:
            env_geom_ids.add(gi)
        else:
            robot_geom_ids.add(gi)

    # --- 在起始构型记录基线接触 ---
    data.qpos[:ARM_DOF] = start_q.copy()
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_collision(model, data)

    baseline_pairs = set()
    for ci in range(data.ncon):
        g1 = data.contact[ci].geom1
        g2 = data.contact[ci].geom2
        if g1 in robot_geom_ids and g2 in env_geom_ids:
            baseline_pairs.add((g1, g2))
        elif g2 in robot_geom_ids and g1 in env_geom_ids:
            baseline_pairs.add((g2, g1))

    # --- 碰撞检测器 ---
    n_calls = 0

    class Checker(ob.StateValidityChecker):
        def isValid(self, state):
            nonlocal n_calls
            n_calls += 1
            data.qpos[:ARM_DOF] = [state[i] for i in range(ARM_DOF)]
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_collision(model, data)
            for ci in range(data.ncon):
                g1 = data.contact[ci].geom1
                g2 = data.contact[ci].geom2
                # (机械臂, 环境) 接触 → 基线内放行, 否则非法
                if g1 in robot_geom_ids and g2 in env_geom_ids:
                    if (g1, g2) not in baseline_pairs:
                        return False
                elif g2 in robot_geom_ids and g1 in env_geom_ids:
                    if (g2, g1) not in baseline_pairs:
                        return False
            return True

    si.setStateValidityChecker(Checker(si))
    si.setup()

    # --- 起止状态 ---
    start_state = ss.allocState()
    goal_state = ss.allocState()
    for i in range(ARM_DOF):
        start_state[i] = float(start_q[i])
        goal_state[i] = float(goal_q[i])

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)
    pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    # --- 选择规划器 ---
    planner_map = {
        "RRTConnect": og.RRTConnect,
        "RRTstar": og.RRTstar,
        "KPIECE1": og.KPIECE1,
    }
    planner_cls = planner_map[planner_type]
    planner = planner_cls(si)

    # 通用参数 (RRT 族有 setRange, KPIECE1 忽略)
    if hasattr(planner, "setRange"):
        planner.setRange(PLANNING_RANGE)
    if hasattr(planner, "setIntermediateStates"):
        planner.setIntermediateStates(True)

    planner.setProblemDefinition(pdef)
    planner.setup()

    # --- 求解 ---
    print(f"  [{planner_type}] 规划中 ...")
    t0 = time.perf_counter()
    solved = planner.solve(PLANNING_TIME)
    elapsed = time.perf_counter() - t0

    # --- 提取状态数 (OMPL 内部统计) ---
    n_states = 0
    try:
        n_states = planner.getPlannerData().numVertices()
    except Exception:
        pass

    if not solved and not pdef.hasApproximateSolution():
        print(f"  [{planner_type}] 失败 (无解)")
        return None
    if not solved and pdef.hasApproximateSolution():
        print(f"  [{planner_type}] 仅有近似解")

    # 优先取精确解, 否则取近似解
    if not pdef.hasSolution():
        if not pdef.hasApproximateSolution():
            return None
        sol_path = pdef.getApproximateSolutionPath()
    else:
        sol_path = pdef.getSolutionPath()

    # --- 提取路径 ---
    waypoints = []
    for i in range(sol_path.getStateCount()):
        st = sol_path.getState(i)
        waypoints.append(np.array([st[j] for j in range(ARM_DOF)]))

    return {
        "waypoints": waypoints,
        "time": elapsed,
        "calls": n_calls,
        "states": n_states,
        "path": sol_path,   # 保留 OMPL Path 对象, 方便后处理
        "ss": ss,
        "si": si,
    }


# ===========================================================================
# 路径后处理: 简化 + 平滑
# ===========================================================================

def simplify_path(path, si):
    """
    simplifyMax: 用贪心算法删除冗余 waypoint.
    返回简化后的 waypoints 列表。
    """
    simplifier = og.PathSimplifier(si)
    # simplifyMax 尽量删除中间节点, 同时保证路径仍然无碰撞
    result = simplifier.simplifyMax(path)
    print(f"  [简化]  简化完成" if result else f"  [简化]  未改变")
    return result


def smooth_path(path, si):
    """
    smoothBSpline: 用三次 B-spline 拟合路径, 然后碰撞检测重验证.
    返回平滑后的 waypoints 列表。
    """
    simplifier = og.PathSimplifier(si)
    result = simplifier.smoothBSpline(path)
    print(f"  [平滑]  平滑完成" if result else f"  [平滑]  未改变")
    return result


# ===========================================================================
# 路径执行: P 控制器 + 速度执行器
# ===========================================================================

def execute_path(model, data, waypoints, label=""):
    """
    在 MuJoCo viewer 中执行一条路径 (阻塞, 直到执行完毕或窗口关闭).
    """
    # 初始化执行器
    for name in ACTUATOR_NAMES:
        data.actuator(name).ctrl = 0.0
    if getattr(data, "act", None) is not None and data.act.size >= model.na:
        for name in ACTUATOR_NAMES:
            act_id = model.actuator(name).id
            joint_id = model.actuator_trnid[act_id][0]
            qpos_adr = model.jnt_qposadr[joint_id]
            data.act[act_id] = data.qpos[qpos_adr]

    wp_index = 0
    kp = 4.0
    max_q_dot = 1.5

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = -60
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [-0.3, 0.2, 0.3]

        while viewer.is_running():
            if wp_index < len(waypoints):
                target_q = waypoints[wp_index]
            else:
                target_q = waypoints[-1]

            current_q = data.qpos[:ARM_DOF].copy()
            q_error = target_q - current_q
            q_dot = np.clip(kp * q_error, -max_q_dot, max_q_dot)

            for i, name in enumerate(ACTUATOR_NAMES):
                data.actuator(name).ctrl = float(q_dot[i])

            mujoco.mj_step(model, data)

            if wp_index < len(waypoints) and np.linalg.norm(q_error) < 0.05:
                wp_index += 1

            viewer.sync()
            time.sleep(model.opt.timestep)

            if wp_index >= len(waypoints):
                time.sleep(0.5)
                break


# ===========================================================================
# 主程序
# ===========================================================================

def main():
    model, data, tmp_path = load_model()
    print(f"[INFO] 障碍物场景已加载")
    print(f"[INFO] 起始构型: {START_Q}")
    print(f"[INFO] 目标构型: {GOAL_Q}")

    # ====== 第 1 步: 三种规划器对比 ======
    print("\n" + "=" * 60)
    print("第 1 步: 三种规划器分别求解")
    print("=" * 60)
    print("        RRTConnect : 双向 RRT, 速度快")
    print("        RRTstar    : 渐近最优, 路径质量逐步改善")
    print("        KPIECE1    : 网格投影, 高维空间中更鲁棒")
    print()

    results = {}
    for planner_name in ["RRTConnect", "RRTstar", "KPIECE1"]:
        r = plan_with_planner(model, data, START_Q, GOAL_Q, planner_name)
        if r:
            results[planner_name] = r
            wp = r["waypoints"]

            path_len = 0.0
            for i in range(1, len(wp)):
                path_len += np.linalg.norm(wp[i] - wp[i - 1])

            print(f"    耗时:     {r['time']:.4f} s")
            print(f"    waypoints: {len(wp)}")
            print(f"    路径长度: {path_len:.3f} rad")
            print(f"    isValid 调用: {r['calls']}")
            print(f"    探索节点数:   {r['states']}")
        print()

    # ====== 第 2 步: 汇总对比 ======
    print("=" * 60)
    print("第 2 步: 汇总对比")
    print("=" * 60)
    print(f"{'规划器':<14} {'耗时(s)':<10} {'waypoints':<12} {'路径长度(rad)':<16} {'isValid调用':<14} {'节点数'}")
    print("-" * 78)
    for name, r in results.items():
        wp = r["waypoints"]
        path_len = sum(np.linalg.norm(wp[i] - wp[i - 1]) for i in range(1, len(wp)))
        print(f"{name:<14} {r['time']:<10.4f} {len(wp):<12} {path_len:<16.3f} {r['calls']:<14} {r['states']}")

    # ====== 第 3 步: 路径后处理 ======
    if "RRTConnect" not in results:
        print("\n[FAIL] RRTConnect 规划失败, 跳过后处理和路径执行")
        os.unlink(tmp_path)
        return

    print("\n" + "=" * 60)
    print("第 3 步: 路径后处理 (以 RRTConnect 路径为例)")
    print("=" * 60)
    print("  simplifyMax : 贪心删除冗余中间点, 保持无碰撞")
    print("  smoothBSpline: 三次 B-spline 拟合 + 碰撞重验证")
    print()

    r = results["RRTConnect"]
    orig_wp = r["waypoints"]
    path_obj = r["path"]
    si = r["si"]
    print(f"  原始 waypoints:     {len(orig_wp)}")

    # 3a. 简化
    simplified = simplify_path(path_obj, si)
    if simplified:
        simp_wp = [np.array([path_obj.getState(i)[j] for j in range(ARM_DOF)])
                   for i in range(path_obj.getStateCount())]
        print(f"  简化后 waypoints:   {len(simp_wp)}")
    else:
        simp_wp = orig_wp

    # 3b. 平滑
    smoothed = smooth_path(path_obj, si)
    if smoothed:
        smooth_wp = [np.array([path_obj.getState(i)[j] for j in range(ARM_DOF)])
                     for i in range(path_obj.getStateCount())]
        print(f"  平滑后 waypoints:   {len(smooth_wp)}")
    else:
        smooth_wp = simp_wp if simplified else orig_wp

    # ====== 第 4 步: 路径执行 ======
    print("\n" + "=" * 60)
    print("第 4 步: 在 MuJoCo viewer 中依次执行")
    print("=" * 60)
    print("  ① 原始路径 → ② 简化路径 → ③ 平滑路径")
    print("  每次执行完毕请在 viewer 窗口按 ESC 继续下一步")
    print()

    data.qpos[:ARM_DOF] = START_Q.copy()
    mujoco.mj_forward(model, data)

    print(">>> [1/3] 执行原始路径 (RRTConnect) ...")
    execute_path(model, data, orig_wp)

    print(">>> [2/3] 执行简化路径 (simplifyMax) ...")
    execute_path(model, data, simp_wp)

    if smoothed:
        print(">>> [3/3] 执行平滑路径 (smoothBSpline) ...")
        execute_path(model, data, smooth_wp)

    print("\n[DONE] 所有路径执行完成")
    os.unlink(tmp_path)


if __name__ == "__main__":
    main()
