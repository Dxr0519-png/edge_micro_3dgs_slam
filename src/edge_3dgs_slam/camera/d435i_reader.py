"""D435i ROS2 数据流：订阅 RGB-D + 时间戳同步 + 内参 + IMU 环形缓存。

用法（Phase 1 验证 / Phase 4 节点通用）：
    class Sink(D435iReader):
        def on_frame(self, frame: SyncedFrame):
            ...   # 处理同步帧

    node = rclpy.create_node('demo')
    reader = Sink(node)
    D435iReader.spin(node)          # 或自定义 executor
"""
import collections
import rclpy
import message_filters
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, Imu
from cv_bridge import CvBridge

from .intrinsics import CameraIntrinsics
from .synced_frame import SyncedFrame

DEPTH_SCALE = 0.001                 # 16UC1 mm -> 米
DEFAULT_TOPICS = dict(
    color_topic="/camera/color/image_raw",
    depth_topic="/camera/aligned_depth_to_color/image_raw",
    info_topic="/camera/color/camera_info",
    imu_topic="/imu",
)


class D435iReader:
    def __init__(self, node, slop=0.05, queue_size=10, **topics):
        self.node = node
        self.bridge = CvBridge()
        t = {**DEFAULT_TOPICS, **topics}
        sub_rgb   = message_filters.Subscriber(node, Image, t["color_topic"])
        sub_depth = message_filters.Subscriber(node, Image, t["depth_topic"])
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth], queue_size=queue_size, slop=slop)
        self.ts.registerCallback(self._on_sync)
        self.K: CameraIntrinsics | None = None
        node.create_subscription(CameraInfo, t["info_topic"], self._on_info, 10)
        node.create_subscription(Imu, t["imu_topic"], self._on_imu, 200)
        self.imu_buffer = collections.deque(maxlen=400)   # 环形缓存，为 VIO 预留

    # ---- 回调 ----
    def _on_info(self, msg):
        if self.K is None:                      # 只取首帧内参
            self.K = CameraIntrinsics.from_camera_info(msg)
            self.node.get_logger().info(f"K: fx={self.K.fx:.2f} fy={self.K.fy:.2f} "
                                        f"cx={self.K.cx:.2f} cy={self.K.cy:.2f}")

    def _on_imu(self, msg):
        self.imu_buffer.append((self._stamp_sec(msg.header.stamp),
                                np.array(msg.linear_acceleration, dtype=np.float64),
                                np.array(msg.angular_velocity, dtype=np.float64)))

    def _on_sync(self, rgb_msg, depth_msg):
        if self.K is None:                      # camera_info 未到先丢帧
            return
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8").copy()
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1").astype(np.float32) * DEPTH_SCALE
        frame = SyncedFrame(rgb=rgb, depth=depth, K=self.K.K(),
                            stamp=self._stamp_sec(rgb_msg.header.stamp))
        self.on_frame(frame)

    @staticmethod
    def _stamp_sec(stamp) -> float:
        """builtin_interfaces/Time -> 秒（Time 消息类无 to_sec()）。"""
        return stamp.sec + stamp.nanosec * 1e-9

    def on_frame(self, frame: SyncedFrame):
        """子类重写：处理同步帧。"""
        raise NotImplementedError

    @staticmethod
    def spin(node):
        rclpy.spin(node)
