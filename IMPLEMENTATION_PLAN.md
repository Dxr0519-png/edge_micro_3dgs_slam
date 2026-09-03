# Edge-3DGS-SLAM · 项目技术路线与执行指南

> 本文档是本项目的**唯一权威实施指南**，按**严格工程依赖顺序**组织（Phase 1 → Phase 6），**不包含任何时间排期**。
> 每一 Phase 均明确给出：**目标 / 输入输出格式 / 关键依赖库 / 需要复写或修改的核心公式与逻辑 / Jetson ARM64 特殊编译与优化注意事项 / 验收标准 / 与后续 Phase 的接口契约**。
> 请在动手写代码前通读本文档，并严格按顺序推进：**前一个 Phase 的验收不通过，不要进入下一个 Phase**。

---

> **2026-09-02 状态修订**：本文档为规划文本（部分条目已执行），**执行结果与最终口径以 `docs/00 §1` 及各 Phase 验证报告为准**。原「**在线实时建图（Tracking 稳定 ≥15 FPS）**」目标经实测在 Jetson 上**不可达**（quality 档 ~9.1 Hz / fps 档 ~3–5 Hz，稳定 10 FPS 不可达；见 `docs/04_Phase4_验证报告.md`），交付定位修订为：**Jetson 采集 → 板卡本地离线两遍式 3DGS 稠密重建 + 开放词汇语义查询**（无实时约束、无外部算力）。本文档残余的「保实时」类规划文字均属该废弃目标的历史记录。

## 0. 项目定位与总体架构

### 0.1 一句话定位

把**开放词汇大模型特征**（MobileSAM 万物分割 + MobileCLIP 文本-图像对齐向量）蒸馏进 **RGB-D 3DGS 重建**的高斯场：在 **Jetson Orin NX** 上采集 D435i 的 RGB-D + IMU，**离线**以免 COLMAP 的 SplaTAM 式位姿估计与两遍式稠密建图生成场景模型，再嵌入语言特征，实现**自然语言三维空间定位**（在线实时口径见文首状态修订）。

### 0.2 与现有工作的关系

| 现有工作 | 我们借鉴什么 | 我们与之的区别 |
|---|---|---|
| 原生 3DGS (graphdeco-inria) | 可微光栅化、高斯参数化、自适应密度化 | 离线重建、无位姿追踪、无语义 |
| **SplaTAM** (CVPR'24) | RGB-D 输入、**深度反投影初始化（免 COLMAP）**、Tracking/Mapping 联合可微优化、各向同性正则 | 桌面 GPU、无语言特征、无边缘优化 |
| MonoGS / RTG-SLAM | Tracking/Mapping 可微优化范式、关键帧管理 | 单目（需 COLMAP 或真值先验）、纯几何 |
| LangSplat | SAM 分层掩码 + CLIP 特征 + 场景级自编码器压缩 + 特征光栅化 | 基于离线 3DGS（COLMAP 先验）、不是 SLAM、不输出实时位姿 |
| LERF | 坐标 → MLP → CLIP 特征的 NeRF 语言场 | 走 3DGS 显式高斯，查询快 1~2 个数量级 |

**本项目 = SplaTAM 的 RGB-D 位姿/建图骨架 + LangSplat 的语言特征蒸馏 + 边缘轻量模型（MobileSAM/MobileCLIP）+ Jetson 极致优化 + ROS2 具身接口。**

### 0.3 总体数据流

```
                         ┌──────────────────────────────────────────────┐
                         │              离线 / 在线 两种模式              │
                         └──────────────────────────────────────────────┘
  D435i ──► [Phase 1] 对齐 RGB-D + IMU + 内参 + 时间戳同步
                │
                ▼
          [Phase 2] Tracking(位姿) + Mapping(高斯几何)   ← 深度反投影初始化（免 COLMAP）
                │                                   ▲
                │  逐关键帧 RGB                      │  光度/SSIM/深度 Loss
                ▼                                   │
  RGB 图像 ──► [Phase 5] MobileSAM 掩码 ─► MobileCLIP 特征图缓存
                │                                   ▲
                │  掩码级 512 维监督                 │  语言特征渲染 Loss (L1+cos)
                ▼                                   │
          [Phase 6] 语言高斯场：每高斯存 D 维潜变量 ──┘
                │
                ▼
  文本查询：「黑色椅子」──► MobileCLIP Text Encoder(512)
                │
                ▼
       与所有高斯解码特征算余弦相似度 ──► 热力图 + 3D BBox ──► [Phase 4] ROS2 发布
```

### 0.4 模块依赖（严格串行）

```
Phase 1(边缘基础设施+D435i数据流)
        │
        ├──► Phase 2(RGB-D 3DGS-SLAM 基线) ──► Phase 3(Jetson 显存/性能优化) ──► Phase 4(ROS2 封装)
        │                                                                              ▲
        └──► Phase 5(边缘开放词汇特征工厂) ──► Phase 6(语言嵌入3DGS+空间查询) ──────────┘
```

- Phase 2 与 Phase 5 在代码上**互相独立**，可并行开发；但 **Phase 6 同时依赖 Phase 4 和 Phase 5 的产物**，是汇合点。
- Phase 3 依赖 Phase 2 的几何骨架；Phase 4 依赖 Phase 3 的优化后管线（封装 Phase 3 优化结果）。
- Phase 5 只依赖 Phase 1（相机数据流），可在 Phase 2/3 开发期间并行推进。

### 0.5 术语表

| 术语 | 含义 |
|---|---|
| **高斯 (Gaussian)** | 3DGS 中一个带属性的 3D 椭球基元：中心 `xyz`、不透明度 `opacity`、尺度 `scale`、旋转四元数 `rotation`、颜色（球谐系数，本项目可退化到单色 `features_dc`）、（Phase 6 新增）语言潜变量 `feature` |
| **光栅化 (Rasterization)** | 把 3D 高斯按深度排序后投影到 2D 图像平面做 α 混合，得到渲染图的过程 |
| **可微 (Differentiable)** | 渲染过程对高斯参数与相机位姿都可求导，从而能反向传播优化 |
| **Tracking** | 相机位姿追踪：固定高斯场，优化当前帧位姿 `T_wc` |
| **Mapping** | 高斯建图：固定位姿，优化/增删高斯参数（`xyz/scale/rotation/opacity/color`） |
| **关键帧 (Keyframe)** | 满足位姿增量阈值而保留、参与建图的帧；非关键帧只做 Tracking |
| **特征图 (Feature Map)** | Phase 5 输出的 2D 语义缓存：每个 MobileSAM 掩码对应一个 MobileCLIP 向量 |
| **潜变量 (Latent)** | Phase 5 自编码器把高维 CLIP 特征压缩后的低维（D 维）表示，存在高斯球上 |
| **relevance score** | Phase 6 中文本向量与某高斯特征向量的余弦相似度，即该高斯「有多像目标」 |
| **统一内存 (Unified Memory)** | Jetson 上 CPU 与 GPU 共享同一块物理内存（16GB），无 PCIe 拷贝 |

### 0.6 复现环境约定

| 项 | 值 |
|---|---|
| 边缘板卡 | **Jetson Orin NX Super（16GB + 256GB NVMe）** |
| JetPack | **6.2 / L4T R36.4.x** |
| GPU | **Ampere `sm_87`**（1024 CUDA 核 + 32 Tensor 核），统一内存 16GB（GPU 实际可用约 10~12GB） |
| 宿主机 OS | Ubuntu 22.04（L4T 根文件系统） |
| 容器 OS | Ubuntu 22.04（`nvcr.io/nvidia/l4t-*` 基础镜像，arm64） |
| ROS2 | **Humble**（arm64 apt 安装） |
| CUDA 工具链 | 随 JetPack 6.2 提供（CUDA 12.x），容器内经 `nvidia-container-toolkit` 挂载 `nvcc`；**以宿主机 `nvcc --version` 为准** |
| PyTorch | **PyTorch for Jetson**（NGC aarch64 wheel，torch 2.x；`torch.cuda.get_device_capability()` == `(8, 7)`） |
| Python | 3.10 |
| 相机 | Intel RealSense D435i（RGB + 深度 + IMU BMI055） |
| Docker | 24.x+ + nvidia-container-toolkit + Compose v2 |

> ⚠️ **最重要的前置风险（aarch64 兼容）**：SplaTAM / 原生 3DGS 官方 `environment.yml` pin 的是桌面 x86 的 **PyTorch 1.12 + CUDA 11.6**，该组合在 Jetson 上**既无 arm64 wheel 也无法运行**。本项目从 Phase 1 起就使用 **PyTorch for Jetson（NGC）+ JetPack 自带 CUDA 工具链**，并显式 `TORCH_CUDA_ARCH_LIST="8.7"` 编译 CUDA 扩展。这是全项目的第一道坎，务必先跑通再谈其它（详见 Phase 1 / Phase 2）。

---

## Phase 1：边缘端基础设施构建与 D435i 数据流打通

### 1.1 目标

在 Jetson 上跑通 **Docker（L4T）→ Librealsense SDK + ROS2 wrapper → RGB-D + IMU 数据流**，产出后续 SLAM 的标准输入：对齐的 RGB-D、相机内参、IMU、硬件时间戳同步。

### 1.2 输入 / 输出格式

**输入（宿主机）**：JetPack 6.2 已烧录 + `nvidia-container-toolkit` + `nvidia-docker` 运行时；D435i 通过 USB3 连接。

**输出（ROS2 话题，均为标准类型）**：

| 话题 | 类型 | 格式 |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | `encoding=rgb8`，`(H,W,3)` uint8，默认 640×480@30 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | `K=[fx,0,cx;0,fy,cy;0,0,1]`，`D`（plumb_bob，rectified 后≈0），`P` |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | `encoding=16UC1`，单位 **mm**，与 RGB 逐像素对齐 |
| `/camera/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | 与颜色内参**相同**（对齐后复用颜色 K） |
| `/imu` | `sensor_msgs/Imu` | gyro `rad/s` + accel `m/s²`，200/400Hz，硬件时间戳 |
| `/tf` / `/tf_static` | `tf2_msgs/TFMessage` | 相机光学坐标系与 body 之间的静态变换 |

> 深度尺度：默认 `depth_scale = 0.001`（即原始 16UC1 值 × 0.001 = 米）。真值深度 `d = raw_depth * depth_scale`。

### 1.3 关键库与版本清单

```text
# 基础镜像（二选一，文档默认 ①）
# ① l4t-pytorch：自带 PyTorch-for-Jetson（最省事）
#    nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.x-py3.10
# ② l4t-base：仅 L4T 根文件系统，PyTorch 用 NGC pip 索引另装
#    nvcr.io/nvidia/l4t-base:r36.4.0

# PyTorch for Jetson（若用方案②）
# pip install torch torchvision --index-url https://pypi.ngc.nvidia.com

# ROS 2 Humble（arm64 apt，Ubuntu 22.04 有官方 arm64 二进制）
#  ros-humble-ros-base  ros-humble-image-transport  ros-humble-image-transport-plugins
#  ros-humble-pcl-ros  ros-humble-tf2-ros  ros-humble-rosbag2  ros-humble-cv-bridge

# Librealsense（源码编译，Jetson 无官方 arm64 deb）
#   -DFORCE_RSUSB_BACKEND=ON   （RSUSB 用户态后端，免内核 patch，Jetson 推荐）
#   依赖：libusb-1.0-0-dev  libglfw3-dev  libgtk-3-dev  libssl-dev  cmake

# realsense2_camera（ROS2 wrapper，源码编译 ros2 分支）
#   git clone https://github.com/IntelRealSense/realsense-ros -b ros2-development
```

> ⚠️ **镜像 tag 务必核对（否则 `docker pull` 失败）**：上面的 `l4t-pytorch:r36.4.0-pth2.x-py3.10` 中 `pth2.x` 是**占位写法**；`docker/Dockerfile` 里 `ARG L4T_TAG=r36.4.0-pth2.6.0-py3.10` 也只是按 JetPack 6.2 / L4T 36.4 给的代表值，**并非一定存在的真实 tag**。构建前请到 [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch) 核对与你 JetPack 6.2 匹配的、实际存在的精确 tag，并回填 `docker/Dockerfile` 的 `L4T_TAG`。

### 1.4 核心公式 / 逻辑（需理解并实现）

**① 深度反投影（后续 SLAM 初始化的基础）**

```
P_cam = K⁻¹ · [u·d, v·d, d]ᵀ
d     = raw_depth × depth_scale          # 默认 0.001
```

其中 `(u,v)` 为像素坐标，`K` 为颜色相机内参。Phase 2 首帧/关键帧即用此式把深度图反投影成 3D 点云初始化高斯。

**② RGB-D 对齐逻辑**

- D435i 深度与颜色来自不同传感器，存在外参 `T_dc`；开启 `align_depth.enable=true` 后，`realsense2_camera` 内部把深度**重投影到颜色光心**并**重采样到颜色分辨率**。
- 对齐后：`aligned_depth` 与 `color` 逐像素对应，且**共享颜色内参 `K`**。这是 SLAM 端唯一需要的约定——**后续代码一律只消费 `/camera/aligned_depth_to_color/image_raw`，不再自行做深度配准**。

**③ 畸变模型**

- D435i 颜色输出已经过 rectification，采用针孔模型即可：`P = [X/Z, Y/Z]`，`u = fx·x + cx`。`CameraInfo.D` 的 plumb_bob 系数在 rectified 输出上≈0，**但保留读取以备万一手动配准未 rectified 图**。

**④ 时间戳同步（关键工程点）**

- D435i 的 RGB/Depth 有**全局快门 + 硬件时间戳**（帧元数据 `rs2_frame.get_timestamp()`），IMU 由 BMI055 以 200/400Hz 采样，三者**硬件对齐**。
- ROS2 端同步策略：
  - **RGB + Depth**：`message_filters.ApproximateTimeSynchronizer`（允许小幅抖动，queue_size≈10），或 `ExactTimeSynchronizer`（D435i 硬件同步好时可用）。
  - **IMU**：单独订阅并环形缓存（时间窗缓冲），供未来 VIO 融合；**Phase 1~6 的 RGB-D 核心不依赖 IMU，仅采集 + 存储**。
- 输出给 SLAM 的一帧数据契约：

```python
@dataclass
class SyncedFrame:
    rgb:   np.ndarray      # (H, W, 3) uint8
    depth: np.ndarray      # (H, W) float32, 单位米
    K:     np.ndarray      # (3, 3) float64  颜色内参
    stamp: float           # 硬件时间戳（秒）
    # 可选：imu_measurements: list[ImuSample]  # 该帧时间窗内的 IMU
```

### 1.5 Jetson ARM64 特殊编译与优化注意事项

1. **基础镜像必须 arm64**：`osrf/ros:humble-desktop-full` **无 arm64 标签**，必须用 `nvcr.io/nvidia/l4t-*` 为基础再装 ROS2。这是与桌面开发最大的差异。
2. **CUDA 工具链来自宿主机挂载**：容器内不装 NVIDIA 驱动，`nvcc`/`libcudart` 由 `nvidia-container-toolkit` 从宿主 L4T 挂载；容器内只需保证 `CUDA_HOME` 指向挂载的 `/usr/local/cuda`。
3. **`TORCH_CUDA_ARCH_LIST="8.7"`**：后续所有 CUDA 扩展（Phase 2 rasterizer）必须按 sm_87 编译，否则 nvcc 会试图为 `sm_50/60/70` 编译而失败或运行崩溃。
4. **GCC 版本**：JetPack 6 自带 GCC 11/12；`nvcc` 对 GCC 版本有上限，**不要手动升到比宿主更高版本**，直接用宿主版本即可。
5. **librealsense RSUSB 后端**：Jetson 内核默认不打 UVC metadata patch，用 `-DFORCE_RSUSB_BACKEND=ON` 让 SDK 走用户态 libusb，**免刷内核**，D435i 即插即用；同时需 udev 规则（`VID 8086`）允许非 root 访问。
6. **统一内存**：`pin_memory`/`to("cuda")` 在 Jetson 上不走 PCIe，但 `torch.cuda` 仍会保留独立内存池；Phase 3 会精细管理，本阶段只需确认 CUDA 可用。
7. **numpy 版本**：锁 `numpy<2`，避免与 `cv_bridge`/OpenCV 的 ABI 冲突（与 `light_weight_3dgs_slam` 项目同款经验）。

### 1.6 验收标准

- [ ] 容器内 `python -c "import torch; print(torch.cuda.get_device_capability())"` 输出 `(8, 7)`。
- [ ] `ros2 topic echo /camera/color/image_raw` 与 `/camera/aligned_depth_to_color/image_raw` 均有数据，且 `ros2 topic hz` 稳定在目标帧率。
- [ ] `/camera/color/camera_info` 的 `K` 合理（fx/fy ≈ 610~620 @640×480），`aligned_depth` 的 `camera_info.K` 与颜色一致。
- [ ] 写一段最小脚本：订阅 RGB+Depth 同步后，用深度反投影生成 3D 点云并保存 `.ply`，用 `open3d` 可视化无错位/重影。

---

## Phase 2：轻量级 RGB-D 3DGS-SLAM 基线构建（核心工程）

### 2.1 目标

以 **SplaTAM** 为蓝本，跑通 **Tracking（相机位姿追踪）** 与 **Mapping（高斯球建图）**：用 D435i 深度反投影初始化高斯（**抛弃 COLMAP**），验证 Jetson 上可微光栅化核心机制可用。

### 2.2 输入 / 输出格式

| 环节 | 输入 | 输出 |
|---|---|---|
| 数据加载 | Phase 1 的 `SyncedFrame`（对齐 RGB-D + K + stamp） | 归一化张量：`rgb ∈ (H,W,3) float32[0,1]`、`depth ∈ (H,W) float32 米`、`K ∈ (3,3)` |
| Tracking | 当前帧 RGB-D + 当前高斯场 + 初始位姿 | 优化后位姿 `T_wc ∈ SE(3)`（4×4，世界→相机，或按代码约定 `T_cw`） |
| Mapping | 关键帧 RGB-D + 位姿 + 高斯场 | 增删/优化后的高斯场 + 渲染图/深度图 |
| 持久化 | 高斯场 | `gaussians.ply`（含 `xyz/opacity/scale/rot/features_dc`） |

**高斯参数张量（SplaTAM 简化参数化，比原生 3DGS 省显存）**：

```
xyz      : (N, 3)  float32    # 中心
rotation : (N, 4)  float32    # 四元数 (w,x,y,z)
scale    : (N, 3)  float32    # 每轴尺度
opacity  : (N, 1)  float32    # 经 sigmoid 激活
color    : (N, 3)  float32    # 0 阶球谐（简化，可不用 SH）
```

> 注意：SplaTAM 默认用**简化球谐（单色或低阶）**而非原生 3DGS 的 16 通道 SH，显存更省、更适合边缘端。Phase 6 会在此之上新增 `feature` 通道。

### 2.3 关键库与版本清单

```text
# CUDA 光栅化：SplaTAM 的 diff-gaussian-rasterization 分支（带 depth 渲染 + surface 深度模式）
#   https://github.com/spla-tam/SplaTAM  （third_party/ 下 clone）
#   关键：它扩展了原版 rasterizer，支持渲染 depth 通道与「surface 深度」（见 2.4 ③）

# 数值/IO
numpy<2  scipy  plyfile  trimesh  open3d  munch  pyyaml  tqdm  rich  matplotlib

# 评估
evo   # ATE/RPE 轨迹评估

# 可视化（可选，边缘端可裁剪）
# PyOpenGL glfw PyGLM   # SplaTAM GUI，Jetson 上可裁剪掉换 headless
```

### 2.4 需复写 / 修改的核心公式与逻辑（务必逐条手推）

**① 3D 高斯协方差**

```
Σ = R · S · Sᵀ · Rᵀ
```
`R` 由四元数 `rotation` 给出，`S = diag(scale)`（3 维尺度）。

**② EWA 投影到 2D（图像平面协方差）**

```
Σ′ = J · W · Σ · Wᵀ · Jᵀ
```
`W` 是世界→相机旋转（`T_wc` 的旋转部分），`J` 是透视投影在均值点处的仿射近似的雅可比：

```
J = [ fx/Z   0    -fx·X/Z² ]
    [ 0     fy/Z  -fy·Y/Z² ]
```
2D 高斯中心为 `μ_2d = K · (X/Z, Y/Z, 1)ᵀ`。

**③ α 混合前向合成（渲染核心）**

```
C = Σᵢ cᵢ · αᵢ · Tᵢ ，  Tᵢ = Πⱼ<ᵢ (1 − αⱼ)
αᵢ = oᵢ · exp(−½ (x − μᵢ)ᵀ Σᵢ′⁻¹ (x − μᵢ))
```
高斯先按深度排序，从近到远累加颜色。**语言特征渲染（Phase 6）是同一套 α 权重，只是把 `cᵢ` 换成特征向量**。

**④ 深度渲染（SplaTAM 特有，与颜色渲染的关键差异）**

SplaTAM 用 **alpha 加权深度**渲染（非「最近表面深度」）：

```
D = Σᵢ dᵢ · αᵢ · Tᵢ / ( Σᵢ αᵢ · Tᵢ + ε )
```
其中 `dᵢ` 是该高斯中心在相机系下的深度。它在 `diff-gaussian-rasterization` 分支里作为额外输出通道（`depth` 模式），反向传播时深度损失梯度可回传到高斯的深度/中心。

> **为什么要区分**：颜色渲染的 α 混合天然「软」，但深度需要「表面感」；直接复用颜色混合权重并归一化，可在可微前提下得到稠密深度，供深度 Loss 与「高斯是否已覆盖该像素」的判据使用。

**⑤ 相机位姿追踪（Tracking）**

- 固定高斯场，只优化当前帧位姿 `T_wc`。
- 位姿参数化：`SE(3)` 用 6 维李代数扰动 `δ ∈ 𝔰𝔢(3)`，`T ← T · Exp(δ)`（**左乘右乘取决于代码约定，务必先确认并全程一致**；SplaTAM 默认世界→相机 `T_wc`，用 `Exp(δ)·T` 左扰动）。
- 迭代 `tracking_iters`（边缘端建议 5~10）步，用 Adam 优化 `δ`：

```
L_track = λ₁ ‖I_render − I_obs‖₁ + λ₂ (1 − SSIM(I_render, I_obs)) + λ₃ ‖D_render − D_obs‖₁
```
- 典型权重：`λ₁=0.8, λ₂=0.2, λ₃=1.0`（SplaTAM 默认，边缘端可微调）。

**⑥ 高斯建图（Mapping）**

- 新关键帧的**高斯新增**：对深度有效像素反投影 `P_world = T_wc⁻¹ · (K⁻¹·[u·d, v·d, d])`，在**尚无高斯覆盖**（用渲染深度与观测深度差异、或半径内无近邻高斯判据）处新增高斯，初始 opacity 小（如 0.5）、scale 取局部近邻均值。
- **属性优化**：光度 + SSIM + 深度损失，外加**各向同性正则**：

```
L_map = L_track + λ_iso · Σᵢ ‖scaleᵢ − mean(scaleᵢ)‖₂
```
各向同性正则约束高斯「不要太椭」，抑制伪影、提升几何稳定（SplaTAM 的关键 trick）。
- **高斯增删**：低 opacity（< 阈值，如 0.005）裁剪；过大 scale 或异常梯度剔除。
- **在线 vs 离线**：SplaTAM 支持逐帧在线；本项目后续按「在线 Tracking + 关键帧 Mapping」为主路径（见 Phase 3 关键帧管理）。

**⑦ 高斯剪枝判据（SplaTAM 风格，为 Phase 3 打基础）**

- 每关键帧渲染后，对「观测深度 ≪ 渲染深度（被遮挡）」或「投影出界」的高斯做标记。
- 每若干步统一 `prune`：opacity 低于阈值、scale 超过上限、梯度异常者删除。

### 2.5 Jetson ARM64 特殊编译与优化注意事项

1. **rasterizer 编译是「第一道坎」**：SplaTAM 的 `diff-gaussian-rasterization` 用 `nvcc` 编译，需：
   - `export TORCH_CUDA_ARCH_LIST="8.7"`（**不设会为错误的架构编译**）；
   - 检查 `setup.py` 是否硬编码 x86 架构名或 `-march=native`（有则去掉）；
   - GCC 用宿主版本（JetPack 6 的 GCC 11/12），**不要**用系统里新装的更高版本；
   - 确认 PyTorch 是 **Jetson aarch64 build**（`torch.__version__` 应带 `aarch64` 或来自 NGC），否则 `torch.utils.cpp_extension` 找不到匹配的 CUDA 头。
2. **验证脚本**（编译后立即跑）：
   ```bash
   python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer; print('ok')"
   ```
3. **降级方案**：若 CUDA 扩展编译不顺，先用**纯 PyTorch 自动求导**实现慢速 splat（显式深度排序 + α 混合，支持对 `xyz/opacity/color` 与位姿求导），**先验证数学正确性**，再替换 CUDA 提速。数学正确性优先于性能。
4. **显存预算（16GB 统一内存）**：本阶段高斯数控制在 ~10 万级即可，重点验证正确性；真正的数量/显存优化在 Phase 3。
5. **数据集**：优先用 **Replica**（自带 RGB-D + 真值位姿 + 内参）验证算法正确性；再切换 D435i 自采序列。Replica 下载脚本放 `scripts/download_data.sh`。

### 2.6 验收标准

- [ ] Replica 单序列跑通，ATE 达到 SplaTAM 报告量级（~cm 级），能输出渲染图与 `.ply`。
- [ ] 能完整说出 Tracking 与 Mapping 各自的**损失函数项与优化变量清单**。
- [ ] 能写最小脚本：给定 `(R, t, K)` 调 `render()` 输出合成 RGB 图，证明「可微光栅化可用」。
- [ ] 对 `xyz` 或 `opacity` 做一次**数值梯度 vs 解析梯度**校验，误差 < 1e-4。
- [ ] 已在本项目封装最小接口 `src/edge_3dgs_slam/slam/build_map(rgb, depth, K, poses) -> gaussian_model`，供 Phase 4/6 复用。

---

## Phase 3：针对 Jetson 平台的显存与性能极致优化（建立简历壁垒）

### 3.1 目标

在 16GB 统一内存、有限算力约束下控制建图/优化的显存峰值（防 OOM），把优化点沉淀为可量化消融（原「维持 15–30 FPS 实时 Tracking」目标已实测不可达并废弃，见文首状态修订）。

### 3.2 输入 / 输出

- **输入**：Phase 2 的几何 SLAM 管线 + 性能剖析数据（`torch.cuda.memory_stats`、逐帧耗时）。
- **输出**：优化后的管线（保留正确性），一份**内存/算力预算表 + 消融结果**（`docs/03` 记录）。

### 3.3 高斯球数量控制与剪枝策略（防 OOM）

| 策略 | 判据 | 说明 |
|---|---|---|
| **opacity 裁剪** | `opacity < τ_op`（如 0.005） | 每 `prune_interval` 步统一删 |
| **scale 异常裁剪** | `max(scale) > τ_scale`（世界尺度） | 抑制飞点/离群高斯 |
| **密度判据** | 反投影点半径 `r` 内已有近邻高斯 | **不新增**，避免重复堆积 |
| **硬上限淘汰** | 总数 `N > N_max`（如 15~30 万） | 按 opacity/贡献排序淘汰最低者 |
| **遮挡剔除** | `D_render < τ_occ · D_obs` | 被遮挡高斯降权或删除 |

> 显存核算：高斯参数紧凑（`xyz 3 + rot 4 + scale 3 + opacity 1 + color 3 ≈ 14 floats`，约 56B/高斯）。**30 万高斯 ≈ 17MB**，远小于渲染 tile buffer 与 NN 模型。因此**高斯数量上限的主要约束是「算力/渲染耗时」，不是显存**；显存压力来自（Phase 5/6 的）SAM/CLIP 模型与光栅化临时缓冲。明确这一点，才能把优化力气花在刀刃上。

### 3.4 渲染与训练管线优化

1. **半精度（FP16）混合训练**：
   - `scale/opacity/color` 用 FP16 存储、`xyz` 保留 FP32（中心精度影响几何，须保留）；
   - `torch.cuda.amp.autocast` 用于 Phase 5/6 的 NN 前向（MobileSAM/MobileCLIP/AE）；
   - **注意**：`diff-gaussian-rasterization` 的 CUDA kernel 内部用 `float`，FP16 张量需在送入前转换，或仅对存储层做 FP16、计算层保持 FP32——**边界转换要显式、可控**。
2. **视锥剔除（Frustum Culling）**：
   - 渲染时 rasterizer 已做 tile 级剔除；**优化时**只对「当前关键帧窗口内可见」的高斯反传梯度（手动 frustum 测试或用渲染输出的可见性掩码），冻结视野外高斯，减少反向图规模。
3. **渲染分辨率分档**：Tracking 用降采样（如 320×240）快速迭代，Mapping 用全分辨率（640×480）精化。
4. **限制每 tile 高斯数 / 总渲染高斯数**：`max_sh_degree=0`（不用 SH）、限制活跃高斯上限，直接砍渲染耗时。

### 3.5 关键帧管理机制（滑动窗口 + 异步建图，在线实验口径）

- **插入判据**：位姿相对上一关键帧的平移 `Δt > τ_t`（如 0.1m）或旋转 `Δθ > τ_θ`（如 3°）时插入新关键帧；或共视比 `< τ_cov` 时插入。
- **滑动窗口**：仅最近 `K`（如 5~8）个关键帧参与 Mapping 优化，旧关键帧高斯冻结，控制单步优化规模。
- **异步建图**：Tracking 在每帧实时跑；Mapping 在**独立 worker 线程**按关键帧节奏跑，二者通过线程安全的高斯场对象（加锁/双缓冲）通信。**Tracking 永不阻塞**。
- **丢帧策略**：若 Mapping 队列积压，跳过中间关键帧的 Mapping，只保留最新——优先保 Tracking 实时性。

### 3.6 性能剖析与验收

- 用 `torch.profiler` + 逐阶段计时打点，定位瓶颈（渲染 vs 反传 vs 位姿优化）。
- 记录优化前后：**FPS、峰值显存、高斯数、ATE/PSNR** 四项，写入 `docs/03`。

### 3.7 验收标准

- [x] ~~D435i 实时序列 Tracking 稳定 ≥15 FPS（目标 30）~~ → **实测取消**：真机稳定 10 FPS 不可达（quality 档 ~9.1 Hz / fps 档 ~3–5 Hz，见 docs/04_Phase4_验证报告.md §5）；在线口径废弃，交付改为离线重建（docs/00 §1）。
- [ ] 峰值显存 < 预算（给出预算表），全程无 OOM。
- [ ] 剪枝/量化后 ATE 与 PSNR 相对 Phase 2 **不显著劣化**（ATE 劣化 < 20%，PSNR 下降 < 2dB）。
- [ ] 形成一份可写进简历的**量化消融表**（FP16、视锥剔除、关键帧窗口各自带来的 FPS/显存增益）。

---

## Phase 4：系统封装与 ROS 2 具身接口

### 4.1 目标

把重建管线与数据链路封装为 ROS2 Humble Node，实现「订阅 RGB-D/IMU → 采集录制（bag/npz 分块）/ 回放评测 → 发布高斯点云与查询服务」的采集与接口层；在线端到端 Tracking/Mapping 模式仅作实验与瓶颈量化，结论见 `docs/04_Phase4_验证报告.md`。

### 4.2 订阅 / 发布接口

**订阅（Subscribers）**：

| 话题 | 类型 | 用途 |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 对齐深度 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 颜色内参 K |
| `/camera/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | 对齐深度内参（==颜色 K） |
| `/imu` | `sensor_msgs/Imu` | IMU（采集/缓存，为 VIO 预留） |

**发布（Publishers）**：

| 话题 | 类型 | 说明 |
|---|---|---|
| `/tf` | `tf2_msgs/TFMessage` | `map→odom→camera` 6DoF 位姿（离线重建/评测用） |
| `/odom` | `nav_msgs/Odometry` | 相机里程计位姿（可选冗余） |
| `/gaussian_map` | `edge_3dgs_msgs/msg/GaussianCloud` | 轻量化高斯点云（可视化） |
| `/rendered_image` | `sensor_msgs/Image` | 渲染合成图（可选，调试/质检） |

### 4.3 消息定义（`ws_src/edge_3dgs_msgs/msg/`，已在骨架中生成）

```text
Gaussian.msg        float32 x y z          # 中心
                    float32 opacity
                    float32 scale_x scale_y scale_z
                    float32 qx qy qz qw    # 旋转四元数
                    float32 r g b          # 颜色
                    float32[] feature      # 语言潜变量（Phase 6 启用，可变长度）

GaussianCloud.msg   std_msgs/Header header
                    Gaussian[] gaussians

QueryRequest.msg    string query           # 自然语言
                    float32 min_score      # 相似度阈值
                    int32 top_k

SemanticPoint.msg   float32 x y z
                    float32 score
                    string label

QueryResult.msg     QueryRequest request
                    SemanticPoint[] points
                    float32[3] bbox_center
                    float32[3] bbox_extent
                    float32[9] bbox_rotation
                    float32 confidence
```

### 4.4 节点架构（`ws_src/edge_3dgs_ros/`）

- 用 `rclpy` + `MultiThreadedExecutor`：
  - **Tracking 回调**：`ApproximateTimeSynchronizer` 同步 RGB+Depth，跑位姿优化，发布 `/tf`、`/odom`。
  - **Mapping worker**：独立线程按关键帧节奏建图，与 Tracking 解耦。
  - **Query 服务**：Phase 6 加入（`Query.srv`，`QueryRequest → QueryResult`）。
- `cv_bridge` 负责 `sensor_msgs/Image ↔ numpy` 转换；注意 `numpy<2` 的 ABI 约束。
- 可视化：`/gaussian_map` 用自定义 `GaussianCloud`，或降级为 `sensor_msgs/PointCloud2`（字段 `x,y,z + intensity(opacity) + rgb`）以便 `rviz2` 直接显示。

### 4.5 验收标准

- [ ] 节点能订阅到 ros2 bag 回放或真机的 RGB-D/IMU，`/tf` 与 `/gaussian_map` 稳定发布。
- [ ] `rviz2` 中能看到高斯点云与相机轨迹，位姿无漂移/跳变。
- [ ] `colcon build` 通过，`ros2 interface show edge_3dgs_msgs/msg/GaussianCloud` 输出正确。

---

## Phase 5：[进阶拓展] 边缘端轻量级 2D 开放词汇特征提取（Open-Vocab Feature Factory）

### 5.1 目标

用 **MobileSAM + MobileCLIP** 替代标准 SAM/CLIP，在边缘端生成降采样的 2D 语义特征图缓存，为 Phase 6 的 3D 特征投影做准备。

### 5.2 输入 / 输出格式

- **输入**：RGB `np.ndarray (H,W,3) uint8`。
- **中间（MobileSAM 掩码）**：每元素 `{segmentation:(H,W)bool, bbox:(x,y,w,h), area:int, score:float}`。
- **最终输出（特征图缓存，掩码级、非逐像素，省显存）**：

```python
# feature_map.pkl（或分帧 .npy）
{
  frame_id: {
    "H": int, "W": int,
    "masks": [
      {"mask_id": int, "bbox": [x,y,w,h],
       "clip_vec": np.ndarray(512, float32),   # L2 归一化
       "hier_level": int},                      # 分层掩码层级
    ],
  }
}
```

### 5.3 模型选择与理由

| 组件 | 选择 | 理由 |
|---|---|---|
| 分割 | **MobileSAM**（ViT-Tiny 蒸馏 SAM，~10MB） | 掩码质量对齐 SAM 但体积/算力小一个量级，可 CPU 实时 |
| 对齐 | **MobileCLIP-S0 / S1**（open_clip 预训练） | 512 维，边缘端低延迟；`open_clip` 直接 `create_model('MobileCLIP-S0', pretrained=...)` |
| 备选 | FastSAM（YOLOv8-seg）、TinyCLIP | FastSAM 更快但掩码边界较粗；TinyCLIP 维度/精度略低 |

### 5.4 流水线与核心逻辑

```python
# 伪代码（对应 src/edge_3dgs_slam/feature_factory/）
masks = mobile_sam(rgb)                       # 万物掩码（可用分层 mask_generator）
for m in masks:
    crop = crop_resize(rgb, m["bbox"], 224)   # 外接框 crop → 224×224
    v = mobile_clip.encode_image(crop)        # (1,512)
    v = v / v.norm(dim=-1, keepdim=True)      # L2 归一化
    store(frame_id, m, v, hier_level)
```

**关键细节（LangSplat 分层掩码思想）**：
- 单一尺度 SAM 掩码在物体边界/层级上含糊。用多级 SAM 掩码（不同 `points_per_side`/IoU 阈值）得到层次语义；每像素维护「掩码 ID → 语义层级」归属表，Phase 6 反投影监督时按层级加权，缓解边缘歧义。

### 5.5 显存友好的特征降维（为 Phase 6 准备）

- 场景级自编码器把 MobileCLIP 512 维压到低维潜变量 `D`（默认 **3~16 维**，LangSplat 用 3 维）：

```text
Encoder: 512 → 256 → 64 → 16 → D   (MLP + ReLU，末层无激活)
Decoder: D → 16 → 64 → 256 → 512   (MLP + ReLU，末层线性)
L_AE = ‖v̂ − v‖₁ + λ · (1 − cos(v̂, v))，  v̂ = Dec(Enc(v))
```

- 收敛判据：重建余弦相似度均值 > 0.95；保存 `data/checkpoints/lang_ae.pt`。
- **为什么降维**：30 万高斯 × 512 维 float32 ≈ 600MB，×D=3 维 ≈ 3.5MB。边缘端必须降维后再存进高斯。

### 5.6 Jetson ARM64 优化

- MobileSAM/MobileCLIP 轻量，CPU 或 CUDA 皆可；建议 **FP16 CUDA 推理 + 逐关键帧缓存**，与 Tracking 分时复用 GPU。
- **进阶（可选）**：MobileCLIP 经 ONNX→TensorRT 转成 INT8/FP16 engine，DLA 加速，进一步降延迟；记录在 `docs/05` 作为加分项。

### 5.7 验收标准

- [ ] 单帧 RGB 能输出数十个有语义意义的掩码；「椅子」文本与椅子掩码相似度明显高于背景。
- [ ] `feature_map.pkl` 可被 Phase 6 的 DataLoader 正确反序列化。
- [ ] AE 重建余弦相似度均值 > 0.95。

---

## Phase 6：[进阶拓展] Language-Embedded 3DGS 与空间查询

### 6.1 目标

为高斯新增低维特征维度，改写可微渲染管线使语言特征可 splat，并与 Phase 5 特征图算 Loss 优化；开发自然语言查询节点，发布 3D BBox / 热力图。

### 6.2 数据结构改造（`src/edge_3dgs_slam/gaussian/`）

```python
self._feature = nn.Parameter(torch.zeros(N, D, device="cuda"))   # D 维语言潜变量
```

- 在 `save_ply`/`load_ply` 中读写 `feature` 通道；D 默认 3（可配 16）。
- **冻结几何**：语言特征优化阶段冻结 `xyz/scale/rotation/opacity/color`（或极小学习率），避免破坏已收敛几何。

### 6.3 特征光栅化改造（核心难点，参考 LangSplat `language-gaussian-rasterization`）

**原理**：语言特征与颜色共享**同一套投影与 α 权重**，只是把「颜色通道」换成「D 维特征通道」。

在 CUDA kernel 中需改 3 处：

1. **`preprocess`**：把 D 维 `feature` 与颜色一起按 2D 高斯分配进 tile。
2. **`render`**：α 混合累加时，除颜色 `C` 与透射率 `T` 外，再累加特征 `F = Σᵢ fᵢ αᵢ Πⱼ<ᵢ(1−αⱼ)`，输出 `(N, D, H, W)` 特征图。
3. **`backward`**：特征通道梯度按 α 权重反传回 `feature`（以及共享的 `opacity/scale/rotation`）。

> **降级方案（务必先做）**：先用**纯 PyTorch 自动求导**实现慢速特征 splat，验证数学正确性（与 CUDA 版特征图误差 < 1e-3 互验），再替换 CUDA。数学正确性优先于性能。

### 6.4 损失函数与反向传播

```text
渲染：    F_2d (N,D,H,W) = rasterize_feature(gaussian_field, pose)
解码：    V_2d (N,512,H,W) = Decoder(F_2d)
监督：    Phase 5 掩码级 MobileCLIP 特征
Loss:     L = L1(V_2d, V_gt) + λ_cos · (1 − cos(V_2d, V_gt))
```

- **掩码级监督**：对每个 MobileSAM 掩码区域取渲染特征均值与该掩码 CLIP 向量对齐；用 Phase 5 分层掩码加权。
- **优化变量**：至少优化 `feature`；视显存可微调 `opacity/scale` 让语义边界贴合几何边界。

### 6.5 空间查询引擎（`src/edge_3dgs_slam/query/`）

```python
text = mobile_clip.encode_text(tokenize("黑色的椅子"))   # (1,512)，L2 归一化
V = Decoder(gaussians.feature)                          # (N,512)，L2 归一化
relevance = V @ text.T                                  # (N,) 余弦相似度
topk_idx = topk(relevance, k)                           # 候选高斯
clusters = DBSCAN(topk 高斯 3D 中心)                     # 去离群 + 聚类
bbox = AABB/OBB(cluster)                                # center + extent + rotation
```

- 输出结构（与 Phase 4 `QueryResult.msg` 对齐）：`{query, bbox, confidence, top_points}`。
- **热力图**：把 `relevance` 映射颜色，对 top-k 高斯中心用当前位姿前向渲染输出高亮图（`sensor_msgs/Image`）。
- **查询节点**：`ros2 service call` 触发，复用 Phase 4 的节点与服务接口。

### 6.6 验收标准

- [ ] 渲染语言特征图经解码后，与 Phase 5 特征图在掩码内余弦相似度 > 0.85。
- [ ] 查询「黑色的椅子」返回 bbox 落在目标附近（可视化目测 + 真值比对）；热力图高亮正确。
- [ ] 多物体场景能区分「椅子」「桌子」等不同查询。
- [ ] 纯 PyTorch 慢速 splat 与 CUDA 版特征图误差 < 1e-3（互验正确性）。

---

## 附录 A：跨 Phase 的接口契约速查

| 契约 | 生产者 | 消费者 | 格式 |
|---|---|---|---|
| 同步帧 `SyncedFrame` | Phase 1 | Phase 2/4/5 | `rgb(H,W,3)uint8` + `depth(H,W)f32米` + `K(3,3)` + `stamp` |
| 高斯几何场 | Phase 2/3 | Phase 4/6 | `.ply`（`xyz/opacity/scale/rot/features_dc`） |
| 特征图缓存 | Phase 5 | Phase 6 | `feature_map.pkl`（掩码级 512 维，L2 归一化） |
| 自编码器 | Phase 5 | Phase 5/6 | `lang_ae.pt`（Enc/Dec 权重，512↔D） |
| 语言高斯场 | Phase 6 | Phase 6 | `.ply`（新增 `feature` 通道，D 维） |
| 位姿 | Phase 2/3 | Phase 4/6 | `/tf` + TUM 格式 `t tx ty tz qx qy qz qw` |
| 查询结果 | Phase 6 | Phase 4 | 与 `QueryResult.msg` 对齐的 Python dict |

## 附录 B：风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| rasterizer 在 aarch64/sm_87 编译失败 | **高** | Phase 1/2 用 `TORCH_CUDA_ARCH_LIST="8.7"` + 宿主 GCC + Jetson PyTorch；降级用纯 PyTorch 慢速 splat 验证正确性 |
| 16GB 统一内存 OOM（SAM/CLIP+高斯训练并存） | 中 | MobileSAM/MobileCLIP 轻量化、特征降维到 D 维、关键帧采样、推理与训练分时复用 GPU、`empty_cache` |
| PyTorch-for-Jetson 版本绑定（NGC 与 JetPack 必须匹配） | 中 | 锁定 JetPack 6.2 对应 torch 版本；升级 JetPack 时同步升级 torch |
| D435i 驱动/时间戳异常 | 中 | RSUSB 后端免内核；udev 规则；用硬件时间戳 + ApproximateTimeSynchronizer |
| 语言特征破坏已收敛几何 | 中 | Phase 6 冻结几何或极小学习率；语言损失与几何损失解耦 |
| Tracking 实时性不达标 | 中 | Phase 3 关键帧 + 异步建图 + 降采样 + 视锥剔除 |

## 附录 C：如何按本文档逐阶段写代码（建议工作流）

1. 每个 Phase 开一个分支 `phase/N-xxx`；通过「验收标准」后才合并。
2. 每个 Phase 先写**最小可运行脚本**（放 `experiments/`），跑通后再抽象进 `src/edge_3dgs_slam/`。
3. 核心公式在 `docs/` 对应 Phase 文档留有手推页，动手前先手推一遍再写代码。
4. 所有跨 Phase 接口，先对齐「附录 A」的契约，再实现，避免返工。
5. 每个 Phase 结束时更新 `README.md` 的模块状态表，保持「文档 ↔ 代码」一致。
