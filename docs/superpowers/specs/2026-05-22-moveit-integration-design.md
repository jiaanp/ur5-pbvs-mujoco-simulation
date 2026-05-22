# MoveIt 全面接管 lift/place/home 运动规划

## 动机

OMPL 直接调用存在 nanobind segfault、IK 模型不一致（Pinocchio vs MuJoCo）、碰撞检测复杂等问题。用 MoveIt 2 替代，通过 `moveit_commander` API 统一 IK + 规划 + 碰撞场景管理。

## 架构

MuJoCo + MoveIt 双模型，主脚本负责状态同步和轨迹执行：

- MuJoCo: PBVS + MPC 视觉伺服 (track/approach/attach)、物理仿真、渲染
- MoveIt: lift/place/home 的 IK、避障规划、时间参数化轨迹

## 数据流

```
attach done → 同步 qpos 到 MoveIt → 加载障碍物到 PlanningScene
  → MoveIt 规划 lift  → 执行轨迹
  → MoveIt 规划 place → 执行轨迹 (避箱壁)
  → MoveIt 规划 home  → 执行轨迹
```

## 文件

### 新建
- `main_pbvs_moveit.py` — 主入口
- `src/task/pbvs_moveit_phases.py` — 3 个 MoveIt phase handler
- `src/task/pbvs_moveit_setup.py` — MoveIt 初始化

### 修改
- `src/config.py` — 新增 `BOX_WALLS` 障碍物定义
- `model/scene_with_gripper.xml` — 墙壁 geom 从 config 动态生成

### 不改
- `main_pbvs_mpc.py`、`demo_ompl_*.py`、PBVS/MPC 控制器、状态机

## 障碍物同步

`config.py` 定义 `BOX_WALLS` 列表，MuJoCo XML 和 MoveIt PlanningScene 都从此读取：
```python
BOX_WALLS = [
    {"name": "place_box_wall_left",  "type": "box", "pos": [-0.80, 0, 0.083], "size": [0.005, 0.20, 0.15]},
    {"name": "place_box_wall_right", "type": "box", "pos": [-0.40, 0, 0.083], "size": [0.005, 0.20, 0.15]},
    {"name": "place_box_wall_front", "type": "box", "pos": [-0.60, -0.20, 0.083], "size": [0.20, 0.005, 0.15]},
    {"name": "place_box_wall_back",  "type": "box", "pos": [-0.60, 0.20, 0.083], "size": [0.20, 0.005, 0.15]},
]
```

## 验证

1. `ros2 launch ur5_moveit_config move_group.launch.py` 启动 MoveIt
2. `python3 main_pbvs_moveit.py` 完整流程
3. place 阶段绕开箱壁
