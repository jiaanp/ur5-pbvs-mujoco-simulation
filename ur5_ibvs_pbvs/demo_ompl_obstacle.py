# -*- coding: utf-8 -*-
"""
UR5e + OMPL RRTConnect 障碍物规避 demo
========================================

工作流程：
  1. 在 MuJoCo 场景 XML 中动态添加障碍物 geom
  2. 用 OMPL RRTConnect 在 6 维关节空间规划无碰撞轨迹
  3. 通过速度执行器驱动 UR5e 执行规划路径

运行方式：
  python3 demo_ompl_obstacle.py

依赖：
  - mujoco        (物理仿真)
  - ompl (2.0+)   (运动规划, nanobind 绑定)
  - numpy         (数组运算)
"""

import os
import time
import tempfile
from xml.etree import ElementTree as ET  # 用于解析/修改 MuJoCo XML 场景

import mujoco
import mujoco.viewer  # MuJoCo 的被动可视化窗口
import numpy as np

# ---------------------------------------------------------------------------
# OMPL 导入
#   - ompl.base    : 核心类型 (状态空间、状态有效性检查、问题定义)
#   - ompl.geometric: 几何规划器 (RRTConnect, RRTstar, PRM 等)
# ---------------------------------------------------------------------------
import ompl.base as ob
import ompl.geometric as og

# ===========================================================================
# 全局常量
# ===========================================================================

# 基础场景 XML 路径 (不含障碍物)
SCENE_XML = os.path.join(os.path.dirname(__file__), "model", "scene_with_gripper.xml")

# UR5e 的 6 个臂关节 (不包含夹爪的 2 个驱动关节)
ARM_DOF = 6

# 6 个臂关节对应的 MuJoCo 速度执行器名称
# 这些名称为 MuJoCo XML 中 <intvelocity> 的 name 属性
ACTUATOR_NAMES = [
    "shoulder_pan_vel_init",   # 关节 0: 肩部旋转 (base)
    "shoulder_lift_vel_init",  # 关节 1: 肩部抬升
    "elbow_vel_init",          # 关节 2: 肘部
    "wrist_1_vel_init",        # 关节 3: 腕部 1
    "wrist_2_vel_init",        # 关节 4: 腕部 2
    "wrist_3_vel_init",        # 关节 5: 腕部 3 (末端旋转)
]

# ===========================================================================
# 第 1 步: 向场景 XML 的动态 <worldbody> 中添加障碍物 geom
# ===========================================================================

def add_obstacles_to_scene(scene_path, obstacles):
    """
    解析原始 MuJoCo XML, 在 <worldbody> 末尾插入障碍物 <geom> 标签,
    写入临时文件后返回路径。

    参数:
        scene_path (str) : 原始场景 .xml 路径
        obstacles (list) : 障碍物列表, 每项是 dict:
            {
                "type": "sphere" | "box" | "cylinder",
                "size": [rx, ry, rz],          # 球体只需 [r]
                "pos":  [x, y, z],             # 世界坐标下的中心位置
                "rgba": [r, g, b, a],          # 颜色 + 透明度
            }

    返回:
        str : 临时 .xml 文件路径
    """
    # --- 解析原始 XML 树 ---
    tree = ET.parse(scene_path)
    root = tree.getroot()

    # 找到 <worldbody> 节点 (所有静态 geom 和 body 的父节点)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("场景 XML 中未找到 <worldbody>")

    # --- 逐个障碍物: 创建 <geom> 子元素插入 <worldbody> ---
    for i, obs in enumerate(obstacles):
        geom = ET.SubElement(worldbody, "geom")       # 创建子元素
        geom.set("name", f"obstacle_{i}")             # 唯一名称, 方便后续识别
        geom.set("type", obs["type"])                 # MuJoCo geom 类型
        geom.set("size", " ".join(f"{x:.4f}" for x in obs["size"]))
        geom.set("pos", f"{obs['pos'][0]:.4f} {obs['pos'][1]:.4f} {obs['pos'][2]:.4f}")
        geom.set("contype", "1")                      # 碰撞类型位掩码 = 1
        geom.set("conaffinity", "1")                  # 碰撞亲和位掩码 = 1
        # conteype & conaffinity 都是 1, 表示可以和任何同为 1 的 geom 碰撞
        geom.set("rgba", f"{obs['rgba'][0]:.3f} {obs['rgba'][1]:.3f} {obs['rgba'][2]:.3f} {obs['rgba'][3]:.3f}")

    # --- 写入临时 .xml, 放在原始场景同目录 (因为原始 XML 里有相对路径 <include>) ---
    model_dir = os.path.dirname(os.path.abspath(scene_path))
    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", prefix="scene_obs_",
        delete=False,                  # 不自动删除, 方便调试时检查
        mode="w", encoding="utf-8",
        dir=model_dir,                 # 关键: 放在模型目录下, 保证 include 能解析
    )
    tree.write(tmp.name, encoding="utf-8", xml_declaration=True)
    print(f"[INFO] 障碍物场景已写入: {tmp.name}")
    return tmp.name


# ===========================================================================
# 第 2 步: 加载带障碍物的 MuJoCo 模型
# ===========================================================================

def load_model_with_obstacles(obstacles):
    """
    调用 add_obstacles_to_scene 得到临时 XML, 再加载为 MuJoCo 模型。

    返回:
        (model, data, xml_path)
    """
    xml_path = add_obstacles_to_scene(SCENE_XML, obstacles)

    # 从 .xml 编译 MuJoCo 模型 (包含运动学树、geom、执行器等)
    model = mujoco.MjModel.from_xml_path(xml_path)

    # 创建与该模型绑定的仿真数据 (qpos, qvel, contact 等)
    data = mujoco.MjData(model)

    return model, data, xml_path


# ===========================================================================
# 第 3 步: OMPL 关节空间规划 - 核心函数
# ===========================================================================

def plan_joint_space(model, data, start_q, goal_q,
                     planning_time=3.0, planning_range=0.05):
    """
    在 UR5e 的 6 维关节空间中用 RRTConnect 规划无碰撞轨迹。

    碰撞检测策略:
      只检测障碍物 geom (名称以 "obstacle_" 开头) 与机械臂的接触。
      忽略 floor ↔ 夹爪的固定接触 (否则起始构型会被误判为碰撞)。

    参数:
        model, data         : MuJoCo 模型和数据
        start_q (ndarray[6]): 起始 6 维关节构型
        goal_q  (ndarray[6]): 目标 6 维关节构型
        planning_time (float): 规划器超时 (秒)
        planning_range(float): RRT 每一步的最大扩展距离 (关节空间弧度)

    返回:
        List[ndarray[6]] | None : 成功时返回 waypoint 列表, 失败返回 None
    """

    # ---------- 3.1 创建 6 维实数状态空间 ----------
    # RealVectorStateSpace(n) 表示一个 n 维连续实数向量空间,
    # 每个维度的值就是该关节的角度 (rad)
    state_space = ob.RealVectorStateSpace(ARM_DOF)

    # 设置每个维度的下界和上界 (从 MuJoCo 关节限位读取)
    bounds = ob.RealVectorBounds(ARM_DOF)
    for i in range(ARM_DOF):
        lo, hi = model.jnt_range[i]         # 关节 i 的 [min, max] 弧度
        bounds.setLow(i, lo)
        bounds.setHigh(i, hi)
    state_space.setBounds(bounds)           # 绑定到状态空间

    # ---------- 3.2 创建 SpaceInformation ----------
    # si 是规划器的运行时上下文, 管理:
    #   - 状态有效性检查器 (碰撞检测)
    #   - 距离度量
    #   - 状态采样、插值
    si = ob.SpaceInformation(state_space)

    # ---------- 3.3 分类 geom: 机械臂 vs 环境 ----------
    # 机械臂 geom: 挂在非 world body 上的 (UR5e 连杆 + 夹爪)
    # 环境   geom: 挂在 world body 上的 (地面 + 传送带 + 放置盒 + 障碍物)
    # body_id == 0 表示直接挂在 <worldbody> 下的 geom
    robot_geom_ids = set()
    env_geom_ids = set()
    for gi in range(model.ngeom):
        body_id = model.geom_bodyid[gi]      # geom 所属的 body 的 ID
        if body_id == 0:
            env_geom_ids.add(gi)             # 没有父 body → 环境静态 geom
        else:
            robot_geom_ids.add(gi)           # 有父 body → 机械臂连杆 geom

    print(f"[INFO] 机械臂 geom: {len(robot_geom_ids)} 个")
    print(f"[INFO] 环境 geom:   {len(env_geom_ids)} 个")

    # ---------- 3.4 在起始构型记录基线接触 ----------
    # 起始构型已有的 (机械臂, 环境) 接触被认为是"固定/不可避免的",
    # 如: 夹爪手指贴地面、底座贴地板。这些在规划中予以放行。
    data.qpos[:ARM_DOF] = start_q.copy()
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_collision(model, data)

    baseline_pairs = set()                  # (robot_geom, env_geom) 元组
    obstacle_ids = env_geom_ids.copy()      # 障碍物也在 env 中, 但单独标记
    for ci in range(data.ncon):
        g1 = data.contact[ci].geom1
        g2 = data.contact[ci].geom2
        # 只关心 (机械臂, 环境) 的接触对
        if g1 in robot_geom_ids and g2 in env_geom_ids:
            baseline_pairs.add((g1, g2))
        elif g2 in robot_geom_ids and g1 in env_geom_ids:
            baseline_pairs.add((g2, g1))

    print(f"[INFO] 基线接触对: {len(baseline_pairs)} 个 (起始构型已有的环境接触)")
    for rg, eg in sorted(baseline_pairs):
        rn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, rg) or f"g{rg}"
        en = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, eg) or f"g{eg}"
        print(f"        {rn} ↔ {en}")

    # ---------- 3.5 实现状态有效性检查器 ----------
    # 在 OMPL 2.0 (nanobind 绑定) 中, 必须子类化 StateValidityChecker
    # 并重写 isValid(state) -> bool
    class MuJoCoValidityChecker(ob.StateValidityChecker):
        def isValid(self, state):
            """
            OMPL 每采样或扩展到一个关节构型时调用此方法。
            state[i] 是第 i 个关节的角度 (float)。

            碰撞判定规则:
              1. (机械臂, 环境) 接触 → 在基线里就允许, 否则非法
              2. 基线的含义: "这个接触在起始构型就存在, 是场景固有的"

            返回 True  → 构型有效 (无碰撞)
            返回 False → 构型无效 (有碰撞)
            """
            # 把 OMPL 状态拷贝到 MuJoCo 的 qpos 中
            data.qpos[:ARM_DOF] = [state[i] for i in range(ARM_DOF)]

            # mj_fwdPosition: 只做正向运动学 (更新所有 body/geom 的世界位姿),
            # 比 mj_forward 快, 因为它不计算速度/加速度
            mujoco.mj_fwdPosition(model, data)

            # 碰撞检测: 更新 data.ncon 和 data.contact[]
            mujoco.mj_collision(model, data)

            # 遍历所有接触对
            for ci in range(data.ncon):
                g1 = data.contact[ci].geom1
                g2 = data.contact[ci].geom2

                # 检测 (机械臂, 环境) 接触对
                if g1 in robot_geom_ids and g2 in env_geom_ids:
                    if (g1, g2) not in baseline_pairs:
                        return False          # 新出现的环境碰撞 → 非法
                elif g2 in robot_geom_ids and g1 in env_geom_ids:
                    if (g2, g1) not in baseline_pairs:
                        return False          # 新出现的环境碰撞 → 非法

            return True  # 无新增碰撞, 构型合法

    # 把检查器绑定到 si
    si.setStateValidityChecker(MuJoCoValidityChecker(si))

    # setup() 必须在设置完所有组件后调用, 内部会做:
    #   - 验证状态空间维度一致
    #   - 设置 longestValidSegment 等参数
    si.setup()

    # ---------- 3.5 创建起止状态 ----------
    # OMPL 2.0 中用 allocState() 分配状态, 不能用 ob.State(space) 构造
    start_state = state_space.allocState()
    goal_state  = state_space.allocState()

    for i in range(ARM_DOF):
        start_state[i] = float(start_q[i])
        goal_state[i]  = float(goal_q[i])

    # ---------- 3.6 定义规划问题 ----------
    # ProblemDefinition 封装了:
    #   - 起止状态
    #   - 优化目标 (这里用路径长度最小化)
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)

    # PathLengthOptimizationObjective: 最小化路径长度
    # 路径长度 = 状态空间中各相邻 waypoint 距离之和
    opt = ob.PathLengthOptimizationObjective(si)
    pdef.setOptimizationObjective(opt)

    # ---------- 3.7 选择并配置规划器 ----------
    # RRTConnect: 双向 RRT (从起点和终点同时扩展两棵树)
    #   优点: 快, 适合大多数场景
    #   其他可选: RRTstar (渐近最优), PRMstar (多次查询), KPIECE1 (高维)
    planner = og.RRTConnect(si)

    # setRange: RRT 树每次扩展的最大步长 (关节空间弧度)
    #   太小 → 节点过多, 规划慢
    #   太大 → 可能跳过窄通道
    planner.setRange(planning_range)

    # 保留中间状态 (用于路径平滑/简化)
    planner.setIntermediateStates(True)

    # 绑定问题定义
    planner.setProblemDefinition(pdef)

    # setup 必须在 solve 前调用
    planner.setup()

    # ---------- 3.8 求解 ----------
    print(f"[INFO] 开始规划 (timeout={planning_time}s, range={planning_range})...")
    t0 = time.time()
    solved = planner.solve(planning_time)   # 阻塞调用, 最多等 planning_time 秒
    elapsed = time.time() - t0
    print(f"[INFO] 规划{'成功' if solved else '失败'}, 耗时 {elapsed:.3f}s")

    if not solved:
        return None

    # ---------- 3.9 提取路径 (waypoints) ----------
    # getSolutionPath() 返回 Path 对象, 包含一连串 State
    path = pdef.getSolutionPath()
    waypoints = []
    for i in range(path.getStateCount()):
        st = path.getState(i)
        # 把 OMPL State 转成 numpy 数组
        waypoints.append(np.array([st[j] for j in range(ARM_DOF)]))
    print(f"[INFO] 路径点数量: {len(waypoints)}")

    return waypoints


# ===========================================================================
# 第 4 步: 主程序
# ===========================================================================

def main():
    # ---- 定义障碍物 ----
    # 每个障碍物是一个静态 geom, 直接挂在 <worldbody> 下 (无 body, 不可移动)
    obstacles = [
        {
            "type": "sphere",                   # 球体
            "size": [0.10],                     # 半径 0.10 m
            "pos": [-0.72, 0.38, 0.28],         # 世界坐标位置 (x, y, z)
            "rgba": [1.0, 0.2, 0.2, 0.9],       # 红色, 透明度 0.9
        },
    ]

    # ---- 加载带障碍物的模型 ----
    model, data, tmp_path = load_model_with_obstacles(obstacles)

    # ---- 设定起止关节构型 ----
    # 注意: UR5e 零位 (全 0) 手臂是向后平伸的, 不是竖直站立
    mujoco.mj_forward(model, data)
    start_q = data.qpos[:ARM_DOF].copy()         # 起始: 零位构型
    goal_q  = np.array([-1.2, -0.8, 1.2, -1.2, -1.5, 0.0])  # 目标: 前伸抓取构型

    # ---- 可选: 打印起止构型对应的末端位姿 ----
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    data.qpos[:ARM_DOF] = start_q
    mujoco.mj_forward(model, data)
    start_ee = data.site_xpos[site_id].copy()    # 起始末端执行器位置
    data.qpos[:ARM_DOF] = goal_q
    mujoco.mj_forward(model, data)
    goal_ee = data.site_xpos[site_id].copy()     # 目标末端执行器位置
    print(f"[INFO] 起始构型: {start_q}")
    print(f"[INFO] 目标构型: {goal_q}")
    print(f"[INFO] 起始 EE:  {start_ee}")
    print(f"[INFO] 目标 EE:  {goal_ee}")

    # ---- OMPL 规划 ----
    waypoints = plan_joint_space(
        model, data, start_q, goal_q,
        planning_time=10.0,                      # 最多等 10 秒
        planning_range=0.2,                      # RRT 步长 0.2 rad
    )

    if waypoints is None:
        print("[FAIL] 规划失败，退出。")
        return

    # =========================================================================
    # 第 5 步: 路径执行 - 用 P 控制器通过速度执行器驱动关节
    # =========================================================================

    print("[INFO] 启动 MuJoCo viewer，按 ESC 退出...")

    # ---- 5.1 初始化速度执行器 ----
    # UR5e 模型使用 intvelocity (积分速度) 执行器.
    # 需要把 act[] 数组的初始值设为当前 qpos, 避免积分漂移.
    for name in ACTUATOR_NAMES:
        data.actuator(name).ctrl = 0.0           # 清零控制信号

    # act 是执行器的内部状态 (积分器的当前值), 需要初始化为当前关节位置
    if getattr(data, "act", None) is not None and data.act.size >= model.na:
        for name in ACTUATOR_NAMES:
            act_id = model.actuator(name).id               # 执行器索引
            joint_id = model.actuator_trnid[act_id][0]     # 该执行器驱动的关节索引
            qpos_adr = model.jnt_qposadr[joint_id]         # 关节在 qpos 中的起始地址
            data.act[act_id] = data.qpos[qpos_adr]         # 同步 act ← qpos

    # ---- 5.2 P 控制器参数 ----
    wp_index = 0                           # 当前追踪的 waypoint 索引
    kp = 4.0                               # 关节空间比例增益 (rad/s per rad error)
    max_q_dot = 1.5                        # 单关节最大速度 (rad/s)

    # ---- 5.3 启动 MuJoCo 被动 viewer ----
    # launch_passive: 启动一个独立渲染线程, 主线程继续控制仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置初始相机视角
        viewer.cam.distance = 2.0          # 相机距离目标点的距离
        viewer.cam.azimuth = -60           # 水平方位角 (度)
        viewer.cam.elevation = -25         # 俯仰角 (度)
        viewer.cam.lookat[:] = [-0.3, 0.2, 0.3]  # 相机注视点 (世界坐标)

        # ---- 5.4 主控制循环 ----
        while viewer.is_running():         # viewer.is_running() 检测窗口是否关闭

            # 选择当前目标 waypoint
            if wp_index < len(waypoints):
                target_q = waypoints[wp_index]
            else:
                target_q = waypoints[-1]   # 已经到最后一个 waypoint

            # P 控制: 关节误差 → 关节速度
            current_q = data.qpos[:ARM_DOF].copy()
            q_error = target_q - current_q
            q_dot = np.clip(kp * q_error, -max_q_dot, max_q_dot)

            # 把速度指令写入各执行器的 ctrl 字段
            for i, name in enumerate(ACTUATOR_NAMES):
                data.actuator(name).ctrl = float(q_dot[i])

            # 物理仿真步进 (含正向动力学 + 碰撞检测)
            mujoco.mj_step(model, data)

            # 接近当前 waypoint 时切换到下一个
            if wp_index < len(waypoints) and np.linalg.norm(q_error) < 0.05:
                wp_index += 1
                if wp_index < len(waypoints):
                    print(f"[TRACK] -> wp {wp_index}/{len(waypoints)}")

            # 同步 viewer (刷新渲染)
            viewer.sync()

            # 按仿真时间步长休眠 (保持实时)
            time.sleep(model.opt.timestep)

            # 全部 waypoint 执行完毕
            if wp_index >= len(waypoints):
                time.sleep(1.0)
                print("[DONE] 路径执行完成")
                break

    # ---- 清理临时 XML 文件 ----
    os.unlink(tmp_path)


if __name__ == "__main__":
    main()
