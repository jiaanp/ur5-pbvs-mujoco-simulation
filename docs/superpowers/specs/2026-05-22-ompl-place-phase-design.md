# OMPL 避障放置阶段集成设计

**日期**: 2026-05-22
**范围**: `main_pbvs_mpc_obstacle.py` 的 place 阶段替换为 OMPL RRTConnect 避障放置

## 动机

当前 place 阶段走直线笛卡尔 MPC 跟踪，无法避开放置盒墙壁。盒子已增大到 0.20×0.20m 底板 + 0.15m 高墙壁，需要规划避障路径将目标物放入箱子内部。

## 流程

```
track → approach → attach    ← PBVS + MPC (视觉跟踪)
                         │
                         ▼  (停止 MPC, 从此只用 P 控制)
                         │
                     ① IK: PLACE_BOX_CENTER + (0,0,0.05) → q_goal
                     ② OMPL: RRTConnect(q_start, q_goal) → waypoints
                     ③ P 控制器逐帧跟踪 waypoints → done
                         │
                    lift → place → home  (全部 OMPL + P)
```

## 新增文件

### `src/planning/ompl_planner.py`

提炼 `demo_ompl_obstacle.py` 的规划逻辑:

```python
def plan_joint_space(model, data, start_q, goal_q,
                     target_geom_names=None,   # 哪些环境 geom 作为障碍物
                     planning_time=3.0,
                     planning_range=0.2):
    """关节空间 RRTConnect 规划, 返回 waypoints 列表或 None"""
```

- 碰撞检测复用改进后的 robot-vs-env 基线模式
- `target_geom_names`: 指定障碍物 geom 名称前缀

## 修改文件

### `src/robotics/pinocchio_kinematics.py`

新增 `solve_ik_position(target_pos_world, q_init) -> np.ndarray`:
- 用 `scipy.optimize.minimize` 最小化 FK 位置误差
- 约束: 关节限位
- 无姿态要求, 只优化位置

### `src/task/pbvs_mpc_phases.py`

新增 `handle_place_phase_ompl(...)`:
- 首次调用: 运行 IK + OMPL 规划, 缓存 waypoints
- 后续调用: P 控制器跟踪 waypoints
- 返回 `(q_dot, place_done)`

新增 `WaypointTracker` 简单状态机:
- 存储 waypoints 列表和当前索引
- `step(current_q, dt) -> q_dot`

### `main_pbvs_mpc_obstacle.py`

place/lift/home 分支改用 OMPL + P 控制.

## 不改的文件

- `main_pbvs_mpc.py` — 原样保留
- `src/task/grasp_state_machine.py` — 不修改
- `demo_ompl_obstacle.py` / `demo_ompl_planners.py` — 不修改

## 箱子内部终点

```python
drop_target = PLACE_BOX_CENTER_WORLD + np.array([0.0, 0.0, 0.05])
            = [-0.13, 0.85, 0.05]
```

底板顶面 z=0.0075, 释放高度 5cm 在底板上方.

## 验证

1. 运行 `main_pbvs_mpc_obstacle.py`
2. 观察 place 阶段机械臂绕过箱子墙壁
3. 确认目标物被放入箱子内部
4. 确认 lift/home 阶段正常运行
