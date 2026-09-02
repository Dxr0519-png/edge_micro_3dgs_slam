# 01 · Phase 1 边缘基础设施构建与 D435i 数据流打通

> 目标：在 Jetson 上**宿主机本地直接运行（无容器）**：PyTorch-for-Jetson + ROS2 Humble + pyrealsense2 → 对齐 RGB-D + IMU + 内参 + 时间戳同步。
> 依赖：无（本 Phase 是地基）。验收以 `IMPLEMENTATION_PLAN.md` 1.6 为准。
> 所有「下载 / 编译 / 安装」步骤遵循**就绪检查惯例**（`docs/00` §4）：先检查是否已就绪，未就绪才执行。

## 1. 前置确认与就绪检查（宿主机，Jetson）

逐项检查，**已就绪的直接跳过，未就绪的补齐后再继续**：

```bash
# ① 架构 / JetPack / CUDA —— 应分别为 aarch64 / R36.4.x（JetPack 6.2）/ 12.x
uname -m
cat /etc/nv_tegra_release
nvcc --version

# ② Python 3.10 + PyTorch for Jetson（aarch64 wheel；JetPack 6 起 NVIDIA 官方发布到 PyPI）
python3 --version
python3 -c "import torch, torchvision; print(torch.__version__, torch.version.cuda)"
#   未就绪才装：pip3 install torch==2.6.0 torchvision==0.21.0

# ③ numpy 必须 <2（cv_bridge / OpenCV ABI 约束）
python3 -c "import numpy; print(numpy.__version__)"    # 若 ≥2：pip3 install "numpy<2"

# ④ ROS2 Humble（apt 系统级，进不了 venv）
ls /opt/ros/humble/setup.bash
#   未就绪才按 docker/Dockerfile §2 的步骤 apt 安装（ros-humble-ros-base + 依赖）

# ⑤ D435i 被识别（VID 8086）
lsusb | grep -i intel
```

> 本项目不依赖 Docker 容器：宿主机即目标机（JetPack 6.2 自带 CUDA 12.6），PyTorch / ROS2 / numpy 均为系统级安装。`docker/` 保留为可选备用方案。

## 2. 本地环境收尾（当前宿主机已具备大部分）

2026-08-26 核查：**torch 2.11.0（CUDA 12.6）、torchvision、ROS2 Humble、numpy 1.26.4（<2）、colcon 均已就绪**，仅需补齐相机 SDK：

```bash
# ① 相机 SDK：pyrealsense2（aarch64 wheel，自带 RSUSB 用户态后端，免内核 patch、免编译）
python3 -c "import pyrealsense2" || pip3 install pyrealsense2

# ② 就绪验证（全部通过再进入下一步）
python3 -c "import torch; print(torch.cuda.get_device_capability())"   # 期望 (8, 7)
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import pyrealsense2; print(pyrealsense2.__version__)"
ros2 --help
```

> 备选：若坚持用 librealsense C++ SDK（`realsense-viewer` 等工具），按 `docker/Dockerfile` §3 手动编译（`cmake .. -DFORCE_RSUSB_BACKEND=ON ...`）。**本计划主路径用 pyrealsense2，无需编译。**

## 3. 相机访问权限（udev 规则）

pyrealsense2 的 RSUSB 后端以用户态 libusb 访问相机，仍需 udev 规则允许非 root 访问（宿主机执行）：

```bash
# 先检查规则是否已存在，已存在跳过：
ls /etc/udev/rules.d/ | grep -i realsense \
    || sudo cp /usr/local/lib/udev/rules.d/99-realsense-libusb.rules /etc/udev/rules.d/ 2>/dev/null \
    || sudo wget -O /etc/udev/rules.d/99-realsense-libusb.rules \
         https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
# 验证：拔插相机后 lsusb 应列出 VID 8086
```

> RSUSB 后端要点：免内核 patch、相机以用户态 libusb 运行，D435i 即插即用。

## 4. 启动相机，确认数据（二选一）

- **路线 A（推荐，零编译）**：pyrealsense2 直读。写 `experiments/phase1_camera_preview.py`：打开 D435i，打印 RGB/Depth 尺寸、内参 K 与帧率，存一帧 RGB-D 验证对齐。

```python
# experiments/phase1_camera_preview.py（骨架）
import pyrealsense2 as rs
pipe = rs.pipeline()
cfg = rs.config(); cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) \
                  ; cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)
# align 到 color：rs.align(rs.stream.color).process(frames)
# 打印 color_stream.get_intrinsics()（fx, fy, cx, cy）
```

- **路线 B（ROS2 节点，Phase 4 前置）**：编译 `realsense2_camera` 并 launch。按就绪检查惯例，**先检查是否已编译，未编译才编译**：

```bash
# ① 检查：未编译才执行下方编译
ros2 pkg prefix realsense2_camera >/dev/null 2>&1 || {
    git clone --depth 1 -b ros2-development https://github.com/IntelRealSense/realsense-ros.git /tmp/realsense-ros
    cd /tmp/realsense-ros && source /opt/ros/humble/setup.bash \
        && colcon build --symlink-install --packages-select realsense2_camera_msgs realsense2_camera
    source install/setup.bash
}

# ② 启动相机节点
# ⚠️ camera_namespace:=/ 必须加：本机 realsense-ros 旧版参数名是 camera_namespace（不是 namespace），
#    默认 'camera' 会产生 /camera/camera/... 双前缀；置 / 才是 docs 约定的 /camera/...
ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=/ \
    align_depth.enable:=true \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30

# ③ 查看话题
ros2 topic list | grep camera
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/color/camera_info --once     # 记下 K 与 D
ros2 topic echo /camera/aligned_depth_to_color/camera_info --once   # 应与颜色 K 一致
```

> 关键点：`align_depth.enable:=true` 后，`/camera/aligned_depth_to_color/image_raw` 与颜色逐像素对齐、共享颜色内参。**后续 SLAM 只消费 aligned_depth，不再自行配准。**

## 5. 编写数据读取/同步模块（`src/edge_3dgs_slam/camera/`）

建议文件：

```text
camera/
├── __init__.py
├── intrinsics.py      # CameraIntrinsics 数据类（fx,fy,cx,cy,w,h），from_camera_info()
├── synced_frame.py    # SyncedFrame 数据类（rgb,depth,K,stamp）
├── d435i_reader.py    # ROS2 订阅 + ApproximateTimeSynchronizer 封装
└── backproject.py     # 深度反投影 -> 点云（Phase 2 复用）
```

关键骨架：

```python
# camera/synced_frame.py
from dataclasses import dataclass
import numpy as np

@dataclass
class SyncedFrame:
    rgb:   np.ndarray   # (H, W, 3) uint8
    depth: np.ndarray   # (H, W) float32, 米
    K:     np.ndarray   # (3, 3) float64
    stamp: float        # 硬件时间戳（秒）
```

```python
# camera/d435i_reader.py（ROS2 订阅 + 同步）
import message_filters
from sensor_msgs.msg import Image, CameraInfo, Imu
from cv_bridge import CvBridge
import numpy as np

DEPTH_SCALE = 0.001

class D435iReader:
    def __init__(self, node):
        self.bridge = CvBridge()
        sub_rgb   = message_filters.Subscriber(node, Image, '/camera/color/image_raw')
        sub_depth = message_filters.Subscriber(node, Image, '/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], queue_size=10, slop=0.05)
        self.ts.registerCallback(self._on_sync)
        self.K = None
        node.create_subscription(CameraInfo, '/camera/color/camera_info', self._on_info, 10)
        node.create_subscription(Imu, '/imu', self._on_imu, 200)   # 环形缓存，为 VIO 预留

    def _on_sync(self, rgb_msg, depth_msg):
        rgb   = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32) * DEPTH_SCALE
        stamp = rgb_msg.header.stamp.to_sec()
        # 组装 SyncedFrame，交给 SLAM Tracking 回调
```

```python
# camera/backproject.py
def backproject(depth, K, T_wc):
    """把深度图反投影为世界系点云。depth: (H,W) 米；T_wc: 世界->相机 4x4。"""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)          # (H,W,3)
    R, t = T_wc[:3,:3], T_wc[:3,3]
    pts_world = pts_cam @ R.T + t                    # P_world = R^T P_cam + t（视 T_wc 约定）
    return pts_world, np.isfinite(z)                 # 返回点 + 有效掩码
```

## 6. 时间戳同步的两种路线（选其一）

- **路线 A（推荐，简单）**：`ApproximateTimeSynchronizer`（如上），D435i 硬件同步好，误差 < 5ms。
- **路线 B（精确）**：读帧元数据硬件时间戳手动对齐；但 ROS2 wrapper 已用 `header.stamp` 承载硬件时间戳，路线 A 足够。

IMU 单独缓存（时间窗环形 buffer），本 Phase 只采集不消费，为未来 VIO 预留。

## 7. 本 Phase 产出清单

- [x] 本地环境可用：`torch.cuda.get_device_capability()==(8,7)`、`pyrealsense2` 可 import。
- [x] 相机数据流跑通（路线 B），RGB/Depth/camera_info 齐全，color 与 aligned_depth 均 30fps。
- [x] `camera/` 模块跑通：订阅同步 → 反投影 → 存 `.ply`（验证实验 `experiments/phase1_verify_camera.py`，
      2026-08-26 实测：30 帧同步、间隔 33.4ms、回投误差 0.0000px、.ply 151019 点加载校验通过）。
- [x] 记录 D435i 内参到 `config/camera/d435i.yaml`（fx=607.61 fy=607.76 cx=331.14 cy=236.03）。

## 8. 查看点云效果（可视化）

产物位于 `data/outputs/phase1/`（frame_aligned.ply / frame_rgb.png / frame_depth.npy / view_grid.png / view_grid.html），按方便程度四种方式：

> **坐标约定（重要）**：`.ply` 存的是**世界系 z-up**（x 右 / y 前 / z 上，原点即相机位置），由 `backproject.to_zup_frame()` 从相机系（x 右 / y 下 / z 前，OpenCV 约定）转换而来——这是全项目世界系约定（Phase 2 起 `T_wc` 即此系）。若直接存相机系，CloudCompare 等查看器（z-up 显示）打开会"立起来/反着"，需转 180° 才能看——不是数据错误，是坐标约定不同。

**① 直接看图（最快）**：`data/outputs/phase1/view_grid.png`，四面板对照图：

| 面板 | 内容 | 说明 |
| --- | --- | --- |
| 1 | 原图 RGB | D435i 颜色流 |
| 2 | 深度图（viridis，米） | 深度值伪彩 |
| 3 | 点云重投影渲染 | **对齐证明**：把点云按内参"画回"画面，与面板 1 逐像素一致（实测 MAE 2.73/255）即颜色几何严格对位 |
| 4 | 点云侧视（RGB 着色） | 绕 Y 轴 50° + 俯仰 25°，看场景立体层次 |

**② 带解释的页面**：`data/outputs/phase1/view_grid.html`（图片内嵌、离线可用），浏览器打开。

**③ 交互式 3D 旋转**（Jetson 有桌面时）：

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    python3 experiments/phase1_view_pointcloud.py --window
```

**④ 专业查看器**（apt 包 `cloudcompare` 安装的是大写二进制，无 `cloudcompare` 命令）：

```bash
# 桌面终端里直接跑；VSCode 远程 / SSH 终端必须先带显示授权（见下方备注）
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority
ccViewer data/outputs/phase1/frame_aligned.ply        # 轻量查看器，看单个点云够用
CloudCompare data/outputs/phase1/frame_aligned.ply    # 完整 GUI（测量/切片/网格化）
```

> **GUI 打不开（QSocketNotifier: Can only be used with threads started with QThread）**：shell 里没有 DISPLAY / XAUTHORITY（VSCode 远程、SSH 终端常见）。本机桌面会话在 `:1`，由 GDM 管理，授权令牌在 `/run/user/1000/gdm/Xauthority`（`loginctl list-sessions` 确认你的会话号）。最省事：坐在 Jetson 前从桌面终端（Ctrl+Alt+T）运行，无需任何环境变量。
> **报 "File doesn't exist"**：ccViewer 按当前工作目录解析相对路径，`pwd` 确认在项目根，或直接用绝对路径 `/home/zc/edge_micro_3dgs_slam/data/outputs/phase1/frame_aligned.ply`。

重新生成：相机节点运行中执行 `python3 experiments/phase1_verify_camera.py 30`（采集新帧），再 `python3 experiments/phase1_view_pointcloud.py`（重新渲染）。

## 9. 常见坑

1. **相机打不开 / 无权限**：udev 规则缺失（见 §3）；先 `lsusb | grep -i intel` 确认 VID 8086。
2. **`torch` 装成 x86 了**：必须 aarch64 wheel（JetPack 6 起官方发布在 PyPI）；用 `python3 -c "import torch; print(torch.__version__, torch.version.cuda)"` 核对。
3. **aligned_depth 无话题**：路线 B 下忘了 `align_depth.enable:=true`。
4. **cv_bridge 报 numpy ABI 错**：锁 `numpy<2`，**永远不要 `pip3 install -U numpy`**。
5. **`pip3 install -U` 带崩系统包**：apt 与 pip 共享 site-packages，只按各 Phase 清单装，不整包升级。
6. **流启动失败（`Hardware Notification: Depth stream start failure` / `control_transfer ... Resource temporarily unavailable` 刷屏、有话题但 0 帧）**：相机被之前异常退出的进程留在坏状态。先停干净所有占用进程再重开：`pkill -f realsense2_camera_node`（注意用 `[e]` 转义避免误杀自身命令行），必要时拔插 USB。本机实测：进程停干净后重新打开即恢复，无需动硬件。
7. **话题变成 `/camera/camera/...` 双前缀**：本机 realsense-ros 是旧版，launch 参数名是 **`camera_namespace`**（不是新版 `namespace`），默认 `'camera'` 导致节点命名空间 + 节点名重复。启动时加 `camera_namespace:=/` 即恢复 docs 约定的 `/camera/...`（见 §4 路线 B）。
