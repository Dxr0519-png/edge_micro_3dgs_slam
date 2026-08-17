# Edge-3DGS-SLAM

基于 **3D Gaussian Splatting SLAM** 与 **开放词汇大模型特征** 的 **边缘端具身感知大脑**，部署在 **Nvidia Jetson Orin NX** 上，输入 **Intel RealSense D435i** 的 RGB-D + IMU。

把开放词汇语义特征蒸馏进 RGB-D 3DGS-SLAM 的高斯场，使边缘机器人能**听懂自然语言指令**（如「找到黑色的椅子」）并在三维空间中定位、框选、导航到目标。

> 详细技术路线与逐阶段执行指南见 **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)**（唯一权威文档，请先通读）。

---

## 概览

| 模块 | 作用 | 技术栈 |
|---|---|---|
| `camera`（Phase 1） | D435i 数据流：RGB-D 对齐 / 内参 / IMU / 时间戳同步 | librealsense、realsense2-camera |
| `slam`（Phase 2） | RGB-D 3DGS-SLAM：Tracking + Mapping（免 COLMAP） | SplaTAM 蓝本、diff-gaussian-rasterization |
| `gaussian`（Phase 2） | 高斯模型 + 可微光栅化封装 | PyTorch CUDA |
| 优化（Phase 3） | Jetson 显存/算力优化：FP16、视锥剔除、高斯剪枝、关键帧 | — |
| `ws_src/*`（Phase 4） | ROS2 具身接口（位姿 / 高斯点云 / 查询） | ROS2 Humble、自定义 msg |
| `feature_factory`（Phase 5） | 边缘开放词汇特征工厂：MobileSAM + MobileCLIP | MobileSAM、MobileCLIP（open_clip） |
| `language_field` + `query`（Phase 6） | 语言特征高斯 + 特征光栅化 + 文本三维查询 | 自研 AE、DBSCAN |

**数据流**：

```
D435i ──► 对齐 RGB-D + IMU ──► Tracking(位姿) + Mapping(高斯几何场) ──┐
  │                                                                   ├─► 语言高斯场 ──► 文本查询 ──► 3D BBox / 热力图
  └──► MobileSAM 掩码 ──► MobileCLIP 特征图 ────────────────────────────┘
```

## 目标硬件与软件

| 项 | 值 |
|---|---|
| 边缘板卡 | **Jetson Orin NX Super（16GB + 256GB）** |
| JetPack | **6.2 / L4T R36.4.x（Ubuntu 22.04）** |
| GPU | Ampere `sm_87`（1024 CUDA 核 + 32 Tensor 核） |
| 容器 | Docker + nvidia-container-toolkit，`l4t-pytorch` 基础镜像（arm64） |
| ROS2 | **Humble**（arm64 apt） |
| PyTorch | **PyTorch for Jetson**（NGC aarch64 wheel） |
| 相机 | Intel RealSense D435i（RGB-D + IMU） |

> ⚠️ **最重要的前置认知**：Jetson 是 **aarch64 + 统一内存**，与桌面 x86 完全不同——不能用 `osrf/ros:humble-desktop-full`（无 arm64）、不能用 pypi 默认 `torch`（x86）。CUDA 架构为 `sm_87`，编译 3DGS CUDA 扩展时需 `TORCH_CUDA_ARCH_LIST="8.7"`。详见 `IMPLEMENTATION_PLAN.md` Phase 1。

## 目录结构

```
edge_micro_3dgs_slam/
├── IMPLEMENTATION_PLAN.md   # ★ 技术路线与执行指南（Phase 1–6）
├── docker/                  # 边缘版 Docker（L4T + PyTorch-for-Jetson + ROS2 Humble）
├── scripts/                 # 构建 / 启动 / 下载数据 / 评估脚本
├── config/                  # camera / slam / feature / ros2 配置
├── docs/                    # 分阶段细粒度设计文档（00 概述 + 01~06）
├── third_party/             # librealsense / SplaTAM / rasterizer（按文档 clone）
├── src/edge_3dgs_slam/      # 自研 Python 算法库（核心代码）
├── ws_src/                  # ROS2 工作空间（edge_3dgs_msgs + edge_3dgs_ros）
├── data/                    # 数据集 / 权重 / 结果（gitignore）
└── experiments/             # 实验脚本与 notebook
```

## 快速开始（环境骨架）

```bash
# 1. 构建并进入容器（L4T + PyTorch-for-Jetson + ROS2 Humble + librealsense）
./scripts/build_docker.sh
./scripts/run_container.sh

# 2. 容器内验证 Jetson 兼容（应输出 (8, 7)）
python -c "import torch; print(torch.cuda.get_device_capability())"
```

> 真机构建与 D435i 数据流打通的具体步骤见 `IMPLEMENTATION_PLAN.md` Phase 1。

## 分阶段路线

| Phase | 主题 | 依赖 |
|---|---|---|
| 1 | 边缘基础设施构建与 D435i 数据流打通 | — |
| 2 | 轻量级 RGB-D 3DGS-SLAM 基线（SplaTAM 蓝本，免 COLMAP） | Phase 1 |
| 3 | Jetson 显存与性能极致优化（简历壁垒） | Phase 2 |
| 4 | 系统封装与 ROS2 具身接口 | Phase 3 |
| 5 | [进阶] 边缘端轻量级 2D 开放词汇特征工厂 | Phase 1 |
| 6 | [进阶] Language-Embedded 3DGS 与空间查询 | Phase 4 + 5 |

每个 Phase 的**输入/输出格式、关键库、需复写的核心公式与逻辑、ARM64 特殊注意事项、验收标准**详见 `IMPLEMENTATION_PLAN.md` 对应章节。

## 文档目录

| # | 文档 | 内容 |
|---|---|---|
| 00 | [项目概述与总体架构](docs/00_项目概述与总体架构.md) | 总体设计、数据流、模块划分 |
| 01 | [Phase 1 边缘基础设施与 D435i 数据流](docs/01_Phase1_边缘基础设施与D435i数据流.md) | Docker + librealsense + 数据流 |
| 02 | [Phase 2 RGB-D 3DGS-SLAM 基线](docs/02_Phase2_RGBD_3DGS_SLAM基线.md) | SplaTAM Tracking/Mapping |
| 03 | [Phase 3 Jetson 显存性能优化](docs/03_Phase3_Jetson显存性能优化.md) | FP16 / 剪枝 / 视锥 / 关键帧 |
| 04 | [Phase 4 ROS2 具身接口](docs/04_Phase4_ROS2具身接口.md) | 节点封装 + 消息 |
| 05 | [Phase 5 边缘开放词汇特征工厂](docs/05_Phase5_边缘开放词汇特征工厂.md) | MobileSAM + MobileCLIP |
| 06 | [Phase 6 语言嵌入 3DGS 与空间查询](docs/06_Phase6_语言嵌入3DGS与空间查询.md) | 特征光栅化 + 查询引擎 |
