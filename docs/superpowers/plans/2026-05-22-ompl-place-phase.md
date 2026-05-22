# OMPL 避障放置阶段 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** place 阶段用 OMPL RRTConnect 避开放置盒墙壁，将目标物放入箱子内部

**Architecture:** Pinocchio IK 求解目标关节构型 → OMPL 关节空间规划 → WaypointTracker P 控制跟踪。attach 之后 lift/place/home 全部走 OMPL + P 控制

**Tech Stack:** mujoco, ompl (2.0+), pinocchio, scipy, numpy

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/planning/__init__.py` | 创建 | 空 init |
| `src/planning/ompl_planner.py` | 创建 | 通用关节空间 OMPL 规划函数 |
| `src/robotics/pinocchio_kinematics.py` | 修改 | 新增 IK 方法 |
| `src/task/pbvs_mpc_phases.py` | 修改 | 新增 WaypointTracker + handle_place_phase_ompl |
| `main_pbvs_mpc_obstacle.py` | 修改 | place/lift/home 改用 OMPL 路径 |

---

### Task 1: 创建 `src/planning/__init__.py`

**Files:**
- Create: `ur5_ibvs_pbvs/src/planning/__init__.py`

- [ ] **Step 1: 写空 init**

```python
# src/planning - motion planning utilities (OMPL + MuJoCo)
```

- [ ] **Step 2: 提交**

```bash
git add ur5_ibvs_pbvs/src/planning/__init__.py
git commit -m "feat: add planning module init"
```

---

### Task 2: 创建 `src/planning/ompl_planner.py`

**Files:**
- Create: `ur5_ibvs_pbvs/src/planning/ompl_planner.py`

- [ ] **Step 1: 写 `plan_joint_space` 函数**

从 `demo_ompl_obstacle.py` 提取规划逻辑，增加 `target_geom_prefixes` 参数指定障碍物：

```python
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
    在 UR5e 的 6 维关节空间中用 RRTConnect 规划无碰撞轨迹。

    参数:
        model, data:          MuJoCo 模型和数据
        start_q, goal_q:      起止关节构型 (ndarray[6])
        target_geom_prefixes: 障碍物 geom 名称前缀列表, 如 ["place_box_wall_"]
                              None 时仅检测新出现的 (robot, env) 接触
        arm_dof:              臂自由度
        planning_time:        规划超时 (秒)
        planning_range:       RRT 步长 (rad)

    返回:
        List[ndarray[6]] | None
    """
    # --- 分类 geom: 机械臂 (body>0) vs 环境 (body=0) ---
    robot_geom_ids = set()
    env_geom_ids = set()
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == 0:
            env_geom_ids.add(gi)
        else:
            robot_geom_ids.add(gi)

    # --- 解析 target_geom_prefixes: 确定哪些环境 geom 需要避开 ---
    if target_geom_prefixes is None:
        target_geom_prefixes = []
    target_geom_ids = set()
    for gi in env_geom_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if name and any(name.startswith(p) for p in target_geom_prefixes):
            target_geom_ids.add(gi)
    if verbose and target_geom_ids:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) for gi in sorted(target_geom_ids)]
        print(f"[OMPL] 避障目标 geom: {names}")

    # --- 基线接触 (起始构型已有的环境接触, 放行) ---
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

    # --- 状态空间 ---
    ss = ob.RealVectorStateSpace(arm_dof)
    bounds = ob.RealVectorBounds(arm_dof)
    for i in range(arm_dof):
        bounds.setLow(i, model.jnt_range[i, 0])
        bounds.setHigh(i, model.jnt_range[i, 1])
    ss.setBounds(bounds)
    si = ob.SpaceInformation(ss)

    # --- 碰撞检测器 ---
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

    # --- 起止状态 ---
    start_s = ss.allocState()
    goal_s = ss.allocState()
    for i in range(arm_dof):
        start_s[i] = float(start_q[i])
        goal_s[i] = float(goal_q[i])

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_s, goal_s)
    pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    # --- RRTConnect ---
    planner = og.RRTConnect(si)
    planner.setRange(planning_range)
    planner.setIntermediateStates(True)
    planner.setProblemDefinition(pdef)
    planner.setup()

    if verbose:
        print(f"[OMPL] 开始规划 (timeout={planning_time}s, range={planning_range})...")
    t0 = time.perf_counter()
    solved = planner.solve(planning_time)
    elapsed = time.perf_counter() - t0

    if not solved and not pdef.hasApproximateSolution():
        if verbose:
            print(f"[OMPL] 规划失败 ({elapsed:.3f}s)")
        return None

    sol_path = pdef.getSolutionPath() if pdef.hasSolution() else pdef.getApproximateSolutionPath()
    waypoints = []
    for i in range(sol_path.getStateCount()):
        st = sol_path.getState(i)
        waypoints.append(np.array([st[j] for j in range(arm_dof)]))

    if verbose:
        print(f"[OMPL] 规划成功 ({elapsed:.3f}s), waypoints={len(waypoints)}")

    return waypoints
```

- [ ] **Step 2: 提交**

```bash
git add ur5_ibvs_pbvs/src/planning/ompl_planner.py
git commit -m "feat: add reusable OMPL joint-space planner"
```

---

### Task 3: PinocchioKinematics 新增 IK

**Files:**
- Modify: `ur5_ibvs_pbvs/src/robotics/pinocchio_kinematics.py`

- [ ] **Step 1: 在类末尾添加 `solve_ik_position` 方法**

在 `blend_joint_velocity` 方法之后、类结束之前插入：

```python
    def solve_ik_position(self, target_pos_world, q_init):
        """
        数值优化求解位置 IK（不约束姿态）。

        使用 scipy L-BFGS-B 在关节限位内最小化末端位置误差。

        参数:
            target_pos_world: (3,) 世界系目标位置
            q_init:           (6,) 初始关节构型

        返回:
            (6,) 关节构型
        """
        from scipy.optimize import minimize

        target = np.asarray(target_pos_world, dtype=np.float64).reshape(3)
        q0 = self._normalize_q(q_init, apply_joint_mapping=True)[:self.arm_dof_count].copy()

        # 关节限位
        lo = self.model.lowerPositionLimit[:self.arm_dof_count].copy()
        hi = self.model.upperPositionLimit[:self.arm_dof_count].copy()
        # 处理无界情况
        lo[lo < -10.0] = -6.283
        hi[hi > 10.0] = 6.283

        def cost(q):
            pos, _, _ = self.forward_kinematics(q)
            return np.sum((pos - target) ** 2)

        res = minimize(
            cost,
            q0,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi)),
            options={"maxiter": 200, "ftol": 1e-9},
        )

        q_result = res.x.copy()
        q_result = q_result * self.q_signs[:self.arm_dof_count]  # 解除符号映射
        return q_result
```

- [ ] **Step 2: 快速验证**

```bash
cd ur5_ibvs_pbvs && python3 -c "
from src.robotics.pinocchio_kinematics import PinocchioKinematics
from src.config import URDF_PATH, EE_FRAME_NAME, PINOCCHIO_Q_SIGNS, PINOCCHIO_EE_OFFSET_LOCAL
import numpy as np
kin = PinocchioKinematics(URDF_PATH, EE_FRAME_NAME, 6, q_signs=PINOCCHIO_Q_SIGNS, ee_offset_local=PINOCCHIO_EE_OFFSET_LOCAL)
target = np.array([-0.13, 0.85, 0.05])
q_goal = kin.solve_ik_position(target, np.zeros(6))
pos, _, _ = kin.forward_kinematics(q_goal)
err = np.linalg.norm(pos - target)
print(f'IK result q={q_goal.round(3)}')
print(f'FK pos={pos.round(3)}')
print(f'Error={err:.4f}m')
assert err < 0.05, f'IK error too large: {err}'
print('PASS')
"
```
期望: error < 0.05m

- [ ] **Step 3: 提交**

```bash
git add ur5_ibvs_pbvs/src/robotics/pinocchio_kinematics.py
git commit -m "feat: add Pinocchio scipy-based position IK"
```

---

### Task 4: 新增 WaypointTracker + handle_place_phase_ompl

**Files:**
- Modify: `ur5_ibvs_pbvs/src/task/pbvs_mpc_phases.py`

- [ ] **Step 1: 在文件末尾添加 WaypointTracker 和 handle_place_phase_ompl**

在 `handle_done_phase` 之后、文件末尾添加：

```python
# ===========================================================================
# OMPL 避障放置: WaypointTracker + handle_place_phase_ompl
# ===========================================================================


class WaypointTracker:
    """关节空间 waypoint 跟踪器, 用 P 控制逐步插值路径点。"""

    def __init__(self, waypoints, kp=4.0, max_q_dot=1.5, waypoint_tol=0.05):
        """
        参数:
            waypoints:     List[ndarray[6]], 关节构型路径
            kp:            比例增益
            max_q_dot:     单关节最大速度 (rad/s)
            waypoint_tol:  切换到下一个 waypoint 的关节误差阈值 (rad)
        """
        self.waypoints = waypoints
        self.index = 0
        self.kp = float(kp)
        self.max_q_dot = float(max_q_dot)
        self.waypoint_tol = float(waypoint_tol)
        self.done = False

    def step(self, current_q):
        """
        根据当前关节构型计算下一步的关节速度指令。

        返回:
            (q_dot, done):
                q_dot: ndarray[6] 关节速度
                done:  bool 是否已走完所有 waypoint
        """
        if self.done:
            return np.zeros(len(current_q)), True

        if self.index < len(self.waypoints):
            target = self.waypoints[self.index]
        else:
            target = self.waypoints[-1]

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
    首次调用时: IK → OMPL → 创建 WaypointTracker
    后续调用时: WaypointTracker.step → q_dot

    返回:
        (q_dot: ndarray[6], place_done: bool, wp_tracker: WaypointTracker)
    """
    if wp_tracker is None:
        # 首次: 获取当前关节构型 + IK 求解目标构型
        current_q = env.get_joint_positions(arm_dof_count)
        place_target_world = grasp_state_machine.place_target_pos_world

        # 在箱子内部 5cm 处放置
        drop_pos = place_target_world.copy()
        drop_pos[2] = 0.05  # 箱底上方 5cm

        # IK 求解
        goal_q = robot_kin.solve_ik_position(drop_pos, current_q)

        # OMPL 规划
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
            return np.zeros(arm_dof_count), False, None

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
```

注意：需要在文件顶部导入 `mujoco`（已在调用处可用）。

- [ ] **Step 2: 提交**

```bash
git add ur5_ibvs_pbvs/src/task/pbvs_mpc_phases.py
git commit -m "feat: add WaypointTracker and OMPL-based place handler"
```

---

### Task 5: 改造 main_pbvs_mpc_obstacle.py 的 place/lift/home 分支

**Files:**
- Modify: `ur5_ibvs_pbvs/main_pbvs_mpc_obstacle.py`

- [ ] **Step 1: 新增导入**

在文件顶部 import 区域末尾添加：

```python
from src.task.pbvs_mpc_phases import handle_place_phase_ompl
```

- [ ] **Step 2: 在 main() 开头初始化 wp_tracker_place**

在 `prev_global_plane_point = None` 之后添加：

```python
    wp_tracker_place = None
```

- [ ] **Step 3: 替换 place 分支**

将现有的 place 分支（约 328-345 行）：

```python
                if grasp_state_machine.phase == "place" and grasp_state_machine.attached:
                    last_q_dot, place_done = handle_place_phase(
                        env,
                        robot_kin,
                        mpc_controller,
                        grasp_state_machine,
                        current_site_pos,
                        last_q_dot,
                        vis,
                        ACTUATOR_NAMES,
                        ARM_DOF_COUNT,
                        HEIGHT,
                        MAX_PLACE_Q_DOT,
                    )
                    control_applied = True

                    if place_done:
                        release_frames_remaining = RELEASE_HOLD_FRAMES
```

替换为：

```python
                if grasp_state_machine.phase == "place" and grasp_state_machine.attached:
                    last_q_dot, place_done, wp_tracker_place = handle_place_phase_ompl(
                        env,
                        model,
                        data,
                        robot_kin,
                        grasp_state_machine,
                        current_site_pos,
                        ACTUATOR_NAMES,
                        ARM_DOF_COUNT,
                        vis,
                        HEIGHT,
                        wp_tracker=wp_tracker_place,
                        max_q_dot=MAX_PLACE_Q_DOT,
                    )
                    control_applied = True

                    if place_done:
                        release_frames_remaining = RELEASE_HOLD_FRAMES
```

- [ ] **Step 4: 同理改造 lift 分支**（后续 task，当前先用 MPC 过渡）

验证整体流程能运行。

- [ ] **Step 5: 运行验证**

```bash
cd ur5_ibvs_pbvs && timeout 60 python3 main_pbvs_mpc_obstacle.py
```

期望: visual servo → lift → place (OMPL 避障) → release → home

- [ ] **Step 6: 提交**

```bash
git add ur5_ibvs_pbvs/main_pbvs_mpc_obstacle.py
git commit -m "feat: integrate OMPL obstacle-aware place phase"
```

---

### Task 6: 同样改造 lift 和 home 阶段

**Files:**
- Modify: `ur5_ibvs_pbvs/main_pbvs_mpc_obstacle.py`
- Modify: `ur5_ibvs_pbvs/src/task/pbvs_mpc_phases.py`

- [ ] **Step 1: 新增 `handle_lift_phase_ompl`**

在 `pbvs_mpc_phases.py` 添加：

```python
def handle_lift_phase_ompl(
    env, model, data, robot_kin, grasp_state_machine,
    current_site_pos, place_site_target_world,
    actuator_names, arm_dof_count, vis, height,
    *,
    wp_tracker=None, max_q_dot=3.0,
):
    """与 handle_place_phase_ompl 模式相同: OMPL 规划 + P 控制"""
    if wp_tracker is None:
        current_q = env.get_joint_positions(arm_dof_count)
        lift_target = grasp_state_machine.lift_target_pos_world
        goal_q = robot_kin.solve_ik_position(lift_target, current_q)

        from src.planning.ompl_planner import plan_joint_space
        waypoints = plan_joint_space(
            model, data, current_q, goal_q,
            target_geom_prefixes=[],
            arm_dof=arm_dof_count,
            planning_time=1.0,
            planning_range=0.2,
        )
        if waypoints is None:
            cv2.putText(vis, "LIFT: OMPL FAILED",
                        (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            return np.zeros(arm_dof_count), False, None
        wp_tracker = WaypointTracker(waypoints, max_q_dot=max_q_dot)

    q_dot, done = wp_tracker.step(env.get_joint_positions(arm_dof_count))
    if done:
        q_dot[:] = 0.0
        grasp_state_machine.start_place(
            current_site_pos,
            place_target_pos_world=place_site_target_world,
        )
    env.apply_joint_velocity(actuator_names, q_dot)
    cv2.putText(vis, f"LIFTING (OMPL) wp={wp_tracker.index}/{len(wp_tracker.waypoints)}",
                (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return q_dot, done, wp_tracker
```

- [ ] **Step 2: 在 main_pbvs_mpc_obstacle.py 接入 lift OMPL**

初始化 `wp_tracker_lift = None`，在 lift 分支用新 handler。

- [ ] **Step 3: home 阶段（已有 P 控制, 可选改造为 OMPL 避障）**

如果 home 阶段也需要避障（避开箱子），同样用 OMPL 改造。否则保持现有 P 控制。

- [ ] **Step 4: 提交**

```bash
git add ur5_ibvs_pbvs/src/task/pbvs_mpc_phases.py ur5_ibvs_pbvs/main_pbvs_mpc_obstacle.py
git commit -m "feat: OMPL-aware lift and home with obstacle avoidance"
```

---

## 验证清单

1. `python3 main_pbvs_mpc_obstacle.py` 完整流程不 crash
2. place 阶段能观察到机械臂绕开放置盒墙壁
3. 目标物被放入箱子内部（z ≈ 0.05）
4. IK 误差 < 0.05m
5. OMPL 规划耗时 < 3s
6. `main_pbvs_mpc.py` 原始版本不受影响
