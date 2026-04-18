# UR5 在 MuJoCo 中的视觉伺服仿真

这是一个基于 MuJoCo 的 UR5e 眼在手上视觉伺服仿真项目。  
项目使用末端相机识别 AprilTag，利用 `solvePnP` 估计目标位姿，并通过 PBVS（Position-Based Visual Servoing，基于位姿的视觉伺服）控制机械臂运动。

## 项目功能

- 基于 MuJoCo 的 UR5e 机械臂仿真场景
- 末端相机（eye-in-hand camera）
- AprilTag 目标建模与渲染
- 基于 `pupil_apriltags` 的 AprilTag 检测
- 基于 OpenCV `solvePnP` 的目标位姿估计
- 基于位姿误差的 PBVS 控制
- 相机沿标签法线方向的目标位姿跟踪
- 手动键盘控制标签的位置和姿态
- 标签丢失后的恢复逻辑
- 运动目标跟踪（反馈 + 前馈）

## 当前演示入口

当前主要可运行的演示脚本是：

- [view_mujoco.py](/home/adrian/ur5_ibvs_pbvs/view_mujoco.py)

这个脚本目前已经实现：

- 从末端相机获取图像
- 检测 AprilTag
- 估计 `T_camera_tag`
- 计算 PBVS 位姿误差
- 驱动 UR5e 向目标位姿伺服
- 手动移动 / 旋转标签，测试闭环跟踪效果

当前控制目标是：

- 相机正对标签
- 相机位于标签法线方向上的指定距离处
- 当标签移动时，机械臂继续跟踪伺服

## 目录结构

```text
ur5_ibvs_pbvs/
├── view_mujoco.py              # 当前主演示：MuJoCo + AprilTag + PBVS
├── check_model.py              # 模型 / 相机 / 标签检查脚本
├── model/
│   ├── scene.xml               # MuJoCo 场景文件
│   ├── ur5e.xml                # UR5e 模型、末端相机、标签目标
│   └── assets/                 # 网格、AprilTag 贴图、目标平面 mesh
├── src/
│   ├── controllers/
│   ├── perception/
│   ├── robotics/
│   ├── sim/
│   └── utils/
├── configs/
├── tests/
└── docs/
```

## 依赖环境

当前项目主要使用：

- Python 3.10
- MuJoCo
- OpenCV
- GLFW
- NumPy
- `pupil_apriltags`

常见安装方式：

```bash
pip install mujoco glfw opencv-python numpy pupil-apriltags
```

## 运行方式

如果环境已经配置完成，可以直接运行：

```bash
python view_mujoco.py
```

如果想先检查模型是否正确加载：

```bash
python check_model.py
```

## 键盘控制

请先点击 `End-Effector Camera` 窗口，让它获得键盘焦点。

### 标签平移

- `W / S`：沿 `+Y / -Y` 方向移动
- `A / D`：沿 `-X / +X` 方向移动
- `R / F`：沿 `+Z / -Z` 方向移动

### 标签旋转

- `I / K`：pitch
- `J / L`：yaw
- `U / O`：roll

### 其他控制

- `Space`：将标签恢复到初始位置和姿态
- `M`：切换“手动控制 / 自动运动”
- `Esc`：退出程序

## 当前 PBVS 流程

当前主流程如下：

```text
末端相机图像
-> AprilTag 检测
-> solvePnP 位姿估计
-> 目标相对相机位姿
-> 位姿误差计算
-> 相机期望速度
-> MuJoCo Jacobian 逆解
-> UR5 关节速度控制
```

为了提升跟踪效果，当前控制器还加入了：

- 运动目标前馈
- 近目标减速与平滑
- 丢失目标后的后退重捕获策略

## 当前局限

- 目前还是一个以实验验证为主的仿真原型，不是完整的控制框架
- 控制参数仍需要根据任务继续调参
- 当前 PBVS 逻辑主要集中在 `view_mujoco.py` 中，后续应拆分到 `src/` 下的模块中
- 虽然仓库结构中预留了 IBVS / PBVS / MPC 入口，但当前最完整的是 `view_mujoco.py` 这条 MuJoCo PBVS 主链

## 后续计划

- 将当前演示脚本重构为 `src/` 下可复用模块
- 补全更清晰的 IBVS 流程
- 增加标签旋转跟踪实验
- 继续优化到位稳定性与收敛速度
- 增加位姿误差与关节速度的日志和可视化

## 致谢

- MuJoCo
- MuJoCo Menagerie 的 UR5e 模型资源
- OpenCV
- pupil-apriltags
