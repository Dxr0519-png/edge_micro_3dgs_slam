# 04 · Phase 4 系统封装与 ROS 2 具身接口

> 目标：把优化后的 SLAM 管线封装为 ROS2 Humble Node，订阅 RGB-D/IMU，发布位姿与高斯点云。
> 依赖：Phase 3 的优化管线。消息定义已在骨架 `ws_src/edge_3dgs_msgs` 生成。

## 1. 先编译消息包

```bash
cd ws_src && colcon build --packages-select edge_3dgs_msgs
source install/setup.bash
ros2 interface show edge_3dgs_msgs/msg/GaussianCloud
```

## 2. 节点骨架（`ws_src/edge_3dgs_ros/`）

Phase 4 需补充的文件（骨架 CMakeLists 已预留）：

```
edge_3dgs_ros/
├── setup.py
├── setup.cfg
├── package.xml               # 已生成
├── resource/edge_3dgs_ros    # 空标记文件
└── edge_3dgs_ros/
    ├── __init__.py
    ├── node.py               # 主节点：订阅 + Tracking 回调 + Mapping 线程
    ├── tf_publisher.py       # /tf + /odom 发布
    └── cloud_publisher.py    # GaussianCloud -> 话题
```

`node.py` 骨架：

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

class Edge3DGSSlamNode(Node):
    def __init__(self):
        super().__init__('edge_3dgs_slam_node')
        self.reader = D435iReader(self)                 # Phase 1 的同步订阅
        self.backend = SLAMBackend()                    # Phase 3 的异步后端
        self.tf_pub = TFPublisher(self)
        self.cloud_pub = CloudPublisher(self)
        # 每来一帧：track -> 发布 tf/odom；关键帧 -> 入 Mapping 队列

def main():
    rclpy.init()
    node = Edge3DGSSlamNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
```

## 3. 发布位姿（`tf_publisher.py`）

```python
def publish_pose(self, T_wc, stamp):
    # map -> odom -> camera 两段：把 T_wc（世界->相机）拆成 map->odom（固定）与 odom->camera
    # 或简化：直接发 map->camera 动态 tf + nav_msgs/Odometry
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = stamp; t.header.frame_id = 'map'; t.child_frame_id = 'camera'
    t.transform.translation = to_vector3(T_wc[:3,3])
    t.transform.rotation    = to_quaternion(T_wc[:3,:3])
    self.br.sendTransform(t)
```

## 4. 发布高斯点云（`cloud_publisher.py`）

```python
def publish_gaussians(self, gaussians, stamp):
    msg = GaussianCloud()
    msg.header.stamp = stamp; msg.header.frame_id = 'map'
    for i in range(gaussians.n):
        g = Gaussian()
        g.x, g.y, g.z = gaussians.xyz[i].tolist()
        g.opacity = gaussians.opacity[i].item()
        g.scale_x, g.scale_y, g.scale_z = gaussians.scale[i].tolist()
        g.qx, g.qy, g.qz, g.qw = gaussians.rot[i].tolist()
        g.r, g.g, g.b = gaussians.color[i].tolist()
        msg.gaussians.append(g)
    self.pub.publish(msg)
```

> 若 `rviz2` 不便直接显示自定义 msg，可降级为 `sensor_msgs/PointCloud2`（字段 x/y/z + intensity=opacity + rgb）。

## 5. 启动与可视化

```bash
# 起相机 + SLAM 节点
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true ...
ros2 run edge_3dgs_ros edge_3dgs_slam_node

# 查看
ros2 topic hz /gaussian_map
ros2 run tf2_tools view_frames        # 看 tf 树
rviz2
```

## 6. 验收清单

- [ ] `colcon build` 通过，消息接口可 `ros2 interface show`。
- [ ] 订阅 ros2 bag 或真机，`/tf` 与 `/gaussian_map` 稳定发布。
- [ ] `rviz2` 中看到高斯点云与轨迹，位姿无跳变。

## 7. 常见坑

1. **回调里做重活**：Tracking 必须轻量，Mapping 丢 worker 线程，否则 DDS 掉消息。
2. **`numpy<2` ABI**：`cv_bridge` 与 opencv-python 冲突时锁 numpy 版本。
3. **tf 时间戳**：用 `frame.header.stamp`，别用 `node.get_clock().now()`，避免 rviz 报 tf 抖动。
4. **host 网络**：容器 `network_mode: host`，`ROS_DOMAIN_ID` 与宿主机一致。
