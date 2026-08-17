# 01 · Phase 1 边缘基础设施构建与 D435i 数据流打通

> 目标：在 Jetson 上跑通 Docker（L4T）→ Librealsense + ROS2 wrapper → 对齐 RGB-D + IMU + 内参 + 时间戳同步。
> 依赖：无（本 Phase 是地基）。验收以 `IMPLEMENTATION_PLAN.md` 1.6 为准。

## 1. 前置确认（宿主机，Jetson）

```bash
# 确认 JetPack / L4T 版本
cat /etc/nv_tegra_release                    # 应为 R36.4.x（JetPack 6.2）
nvcc --version                               # 记录 CUDA 版本（12.x）

# 确认 Docker + nvidia-container-toolkit
docker --version
docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.4.0 nvidia-smi   # 若报错先装 toolkit

# 安装 toolkit（如缺失）
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker

# 确认 D435i 被识别（VID 8086）
lsusb | grep -i intel
```

## 2. 构建容器

```bash
./scripts/build_docker.sh     # 等价 docker compose build
./scripts/run_container.sh    # 进入容器
```

> 若 `l4t-pytorch` 基础镜像 tag 拉取失败，去 [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch) 选与你 JetPack 匹配的精确 tag，改 `docker/Dockerfile` 的 `ARG L4T_TAG`。

容器内验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"  # 期望 (8, 7)
python -c "import cv2; print(cv2.__version__)"
ros2 --help
```

## 3. 编译 Librealsense（已在 Dockerfile 内完成，此处为手动备用）

若需手动重建：

```bash
git clone --depth 1 https://github.com/IntelRealSense/librealsense.git
cd librealsense && mkdir build && cd build
cmake .. -DFORCE_RSUSB_BACKEND=ON -DBUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && sudo make install
# 验证
realsense-viewer   # 或 rs-enumerate-devices
```

> RSUSB 后端要点：免内核 patch、相机以用户态 libusb 运行。仍需 udev 规则允许非 root 访问 USB（宿主机执行）：
> ```bash
> sudo cp /usr/local/lib/udev/rules.d/99-realsense-libusb.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules
> ```

## 4. 启动相机节点，确认话题

```bash
# 容器内（或宿主机 ROS2 环境）
ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30

# 查看话题
ros2 topic list | grep camera
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/color/camera_info --once     # 记下 K 与 D
ros2 topic echo /camera/aligned_depth_to_color/camera_info --once   # 应与颜色 K 一致
```

> 关键点：`align_depth.enable:=true` 后，`/camera/aligned_depth_to_color/image_raw` 与颜色逐像素对齐、共享颜色内参。**后续 SLAM 只消费 aligned_depth，不再自行配准。**

## 5. 编写数据读取/同步模块（`src/edge_3dgs_slam/camera/`）

建议文件：

```
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

- [ ] 容器可运行，`torch.cuda.get_device_capability()==(8,7)`。
- [ ] `rs_launch.py` 起相机，RGB/Depth/IMU/camera_info 话题齐全且帧率稳定。
- [ ] `camera/` 模块跑通：订阅同步 → 反投影 → 存 `.ply`，`open3d` 可视化无错位。
- [ ] 记录 D435i 内参到 `config/camera/d435i.yaml`。

## 8. 常见坑

1. **相机在容器内打不开**：`privileged: true` + `/dev/bus/usb` 透传 + udev 规则缺一不可。
2. **`torch` 装成 x86 了**：必须用 NGC index 或 l4t-pytorch 镜像；`pip list | grep torch` 看来源。
3. **aligned_depth 无话题**：忘了 `align_depth.enable:=true`。
4. **cv_bridge 报 numpy ABI 错**：锁 `numpy<2`。
