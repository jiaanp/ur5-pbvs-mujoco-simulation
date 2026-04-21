# ur5-pbvs-mujoco-simulation

UR5e eye-in-hand visual servoing and grasping simulation in MuJoCo with AprilTag, PBVS, MPC, and Pinocchio.

## 项目简介

这是一个基于 MuJoCo 的 UR5e 眼在手上视觉伺服与抓取仿真项目。  
当前项目已经不只是单纯的 PBVS 演示，而是扩展成了一条较完整的任务链：

- 末端相机识别 AprilTag
- `solvePnP` 估计目标位姿
- PBVS / MPC 跟踪目标
- 夹爪抓取
- 抬升、放置、归位
- 传送带场景中的动态抓取测试

项目同时保留了：

- `main_pbvs.py`：更简单的 PBVS 主入口
- `main_pbvs_mpc.py`：当前更完整的动态抓取主入口

## 当前能力

- UR5e + 末端相机 MuJoCo 仿真
- AprilTag 检测与位姿估计
- 基于 PBVS 的位姿误差闭环
- 基于 MPC 的动态目标跟踪
- Pinocchio 运动学接入
- 夹爪开合控制
- 物理目标物抓取、抬升、放置
- 传送带场景测试
- 放置盒场景

## 推荐入口

### 1. PBVS 基础入口

```bash
python3 /home/adrian/ur5-pbvs-mujoco-simulation/ur5_ibvs_pbvs/main_pbvs.py
```

适合验证：

- AprilTag 检测
- PBVS 基础闭环
- Pinocchio 运动学接入

### 2. 动态抓取入口

```bash
python3 /home/adrian/ur5-pbvs-mujoco-simulation/ur5_ibvs_pbvs/main_pbvs_mpc.py
```

适合验证：

- 动态目标跟踪
- 夹爪抓取
- 抬升 / 放置 / 归位
- 传送带测试

## 当前目录结构

```text
ur5-pbvs-mujoco-simulation/
├── README.md
└── ur5_ibvs_pbvs/
    ├── main_pbvs.py
    ├── main_pbvs_mpc.py
    ├── test_pinicchio.py
    ├── model/
    │   ├── scene.xml
    │   ├── scene_with_gripper.xml
    │   ├── ur5e.xml
    │   ├── ur5e_with_gripper.xml
    │   └── asset/
    └── src/
        ├── __init__.py
        ├── config.py
        ├── controllers/
        │   ├── ibvs_controller.py
        │   ├── mpc_controller.py
        │   └── pbvs_controller.py
        ├── perception/
        │   ├── camera_model.py
        │   ├── feature_tracker.py
        │   ├── pose_estimator.py
        │   └── target_motion.py
        ├── robotics/
        │   ├── pinocchio_kinematics.py
        │   ├── ur5_interface.py
        │   └── ur5_kinematics.py
        ├── sim/
        │   ├── mujoco_env.py
        │   └── rendering.py
        ├── task/
        │   ├── grasp_state_machine.py
        │   ├── pbvs_mpc_phases.py
        │   ├── pbvs_mpc_runtime.py
        │   ├── pbvs_mpc_setup.py
        │   └── pbvs_mpc_visual_servo.py
        └── utils/
            ├── math_utils.py
            └── transforms.py
```

## 依赖环境

建议环境：

- Python 3.10
- MuJoCo
- OpenCV
- NumPy
- GLFW
- `pupil_apriltags`
- `pinocchio`

常见安装方式：

```bash
pip install mujoco glfw opencv-python numpy pupil-apriltags pin
```

如果你的 `pinocchio` 不是通过 `pip` 安装，也可以使用 Conda 或系统包方式，只要运行时能正常导入即可。

## 主流程说明

### `main_pbvs.py`

主链大致是：

```text
相机图像
-> AprilTag 检测
-> solvePnP
-> PBVS 误差
-> Pinocchio Jacobian
-> 关节速度控制
```

### `main_pbvs_mpc.py`

主链大致是：

```text
相机图像
-> AprilTag 检测
-> solvePnP
-> PBVS 误差
-> MPC 跟踪
-> 抓取状态机
-> lift
-> place
-> home
```

当前状态机大致为：

```text
track -> approach -> attach -> lift -> place -> release -> home -> done
```

## 场景说明

当前带夹爪场景使用：

- `scene_with_gripper.xml`
- `ur5e_with_gripper.xml`

其中已经包含：

- 末端相机
- 夹爪
- 物理目标物
- 传送带场景
- 放置盒

## 目前更适合做什么

这个项目当前更适合：

- PBVS / MPC 视觉伺服实验
- 动态目标抓取演示
- 眼在手上视觉抓取流程验证
- Pinocchio 与 MuJoCo 联合使用实验

## 当前仍在持续整理的部分

- README 和文档仍在逐步同步最新代码结构
- `main_pbvs.py` 还可以继续按 `main_pbvs_mpc.py` 的方式整理
- 传送带目前以“测试可用”为主，不是工业级精细建模

## 致谢

- MuJoCo
- OpenCV
- pupil-apriltags
- Pinocchio
- MuJoCo Menagerie 的 UR5e 模型资源
