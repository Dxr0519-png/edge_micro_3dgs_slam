"""位姿发布（docs/04 §3）：map→odom 静态 + odom→camera 动态 + /odom。

tf 语义修正（docs/04 §3 回填）：全项目 T_wc 为**世界→相机**（w2c，`p_cam = T @ p_world`），
相机在世界系中的位姿 = inv(T_wc) = [Rᵀ | -Rᵀt]。骨架直接把 T_wc 当 tf 发布有误。
落地为 REP-105 两段：map→odom（静态单位阵，/tf_static）与 odom→camera（动态）。
时间戳一律用帧时间戳 frame.stamp（docs/04 §7 坑 3），不用 node.get_clock().now()。
"""
from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def _to_time(stamp) -> Time:
    """float 秒（frame.stamp）或 builtin_interfaces/Time → Time。"""
    if isinstance(stamp, Time):
        return stamp
    sec = int(stamp)
    return Time(sec=sec, nanosec=int(round((stamp - sec) * 1e9)))


def _matrix_to_quat_np(R: np.ndarray) -> np.ndarray:
    """(3,3) 旋转矩阵 → 四元数 (x,y,z,w)（纯 numpy，无 CUDA 排队——真机建图
    占 GPU 时 CUDA 版转换会排队，每帧 2 次拉高 on_frame 延迟，实测坑）。"""
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = np.zeros(4)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
        q[3] = 0.25 * s
    else:
        i = int(np.argmax(np.diag(R)))
        j = (i + 1) % 3
        k = (j + 1) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([0.0, 0.0, 0.0, 1.0])


def _to_quaternion_xyzw(R: np.ndarray) -> Quaternion:
    """(3,3) 旋转矩阵 → geometry_msgs/Quaternion（xyzw，纯 numpy）。"""
    q = _matrix_to_quat_np(R)
    return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


class TFPublisher:
    """map→odom（静态）+ odom→camera（动态）+ /odom（nav_msgs/Odometry）。"""

    def __init__(self, node, world_frame: str = "map", odom_frame: str = "odom",
                 camera_frame: str = "camera", publish_odom: bool = True):
        self.node = node
        self.world_frame = world_frame
        self.odom_frame = odom_frame
        self.camera_frame = camera_frame
        self._static_br = StaticTransformBroadcaster(node)
        self._br = TransformBroadcaster(node)
        self._odom_pub = (node.create_publisher(Odometry, "/odom", 10)
                          if publish_odom else None)
        self._send_static_identity()

    def _send_static_identity(self) -> None:
        """map→odom 静态单位阵（/tf_static，latched，后订阅者可见）。"""
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self._static_br.sendTransform(t)

    def publish_pose(self, T_wc: np.ndarray, stamp) -> None:
        """每帧发布位姿（处理帧与外推帧都发，保持 30Hz 位姿流）。

        参数:
            T_wc: (4,4) 世界→相机（w2c）
            stamp: 帧时间戳（builtin_interfaces/Time，来自 frame.stamp）
        """
        R = T_wc[:3, :3]
        t = T_wc[:3, 3]
        # inv(T_wc) = [Rᵀ | -Rᵀt]：相机在世界中的位姿
        R_wc = R.T
        t_wc = -R_wc @ t

        # 1) 动态 tf：odom→camera
        tf = TransformStamped()
        tf.header.stamp = _to_time(stamp)
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.camera_frame
        tf.transform.translation.x = float(t_wc[0])
        tf.transform.translation.y = float(t_wc[1])
        tf.transform.translation.z = float(t_wc[2])
        tf.transform.rotation = _to_quaternion_xyzw(R_wc)
        self._br.sendTransform(tf)

        # 2) /odom：位置+朝向同上；twist 全 0（Phase 4 不估速度，VIO 位留）
        if self._odom_pub is not None:
            odom = Odometry()
            odom.header.stamp = _to_time(stamp)
            odom.header.frame_id = self.odom_frame
            odom.child_frame_id = self.camera_frame
            odom.pose.pose.position.x = float(t_wc[0])
            odom.pose.pose.position.y = float(t_wc[1])
            odom.pose.pose.position.z = float(t_wc[2])
            odom.pose.pose.orientation = _to_quaternion_xyzw(R_wc)
            self._odom_pub.publish(odom)
