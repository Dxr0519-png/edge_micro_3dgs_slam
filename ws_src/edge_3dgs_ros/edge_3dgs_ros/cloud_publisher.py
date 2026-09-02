"""高斯点云发布（docs/04 §4）：/gaussian_map（GaussianCloud）+ /gaussian_map_pc2（PointCloud2）。

发布策略（Phase 4 定稿）：
- 1Hz 定时器线程调用 publish()，消息构建在**锁外**（快照已由 snapshot_model 锁内拷出）；
- max_points 均匀抽稀（默认 50k）：200k 全量 Python 构建 ~1-3s，绝不能进 sync 回调；
- rviz2 显示用 PointCloud2（numpy tobytes，毫秒级）；GaussianCloud 为 Phase 6/外部
  消费者的 API 契约（Gaussian.feature 留空，Phase 6 填语言潜变量）。
"""
from __future__ import annotations

import numpy as np
from edge_3dgs_msgs.msg import Gaussian, GaussianCloud
from sensor_msgs.msg import PointCloud2, PointField

from .tf_publisher import _to_time

_PC2_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("intensity", "<f4"), ("rgb", "<u4")])   # 20B/点


def _pack_rgb(rgb01: np.ndarray) -> np.ndarray:
    """(N,3) 0~1 RGB → packed uint32（r<<16 | g<<8 | b）。"""
    rgb = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint32)
    return (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]


def _make_pointcloud2(means: np.ndarray, opacity: np.ndarray,
                      rgb01: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """numpy 结构化数组 → PointCloud2（毫秒级，无 Python 逐元素循环）。"""
    n = means.shape[0]
    arr = np.zeros(n, dtype=_PC2_DTYPE)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["intensity"] = np.asarray(opacity).reshape(-1)[:n]
    arr["rgb"] = _pack_rgb(rgb01)
    msg = PointCloud2()
    msg.header.stamp = _to_time(stamp)
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = n
    msg.point_step = _PC2_DTYPE.itemsize
    msg.row_step = msg.point_step * n
    msg.is_dense = False
    msg.is_bigendian = False
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=16, datatype=PointField.UINT32, count=1),
    ]
    msg.data = arr.tobytes()
    return msg


class CloudPublisher:
    """GaussianCloud + PointCloud2 双发（同一快照、同频率、同 frame_id）。"""

    def __init__(self, node, publish_gaussian_map: bool = True,
                 publish_pointcloud2: bool = True,
                 world_frame: str = "map", max_points: int = 50000):
        self.node = node
        self.world_frame = world_frame
        self.max_points = max_points
        self._gpub = (node.create_publisher(GaussianCloud, "/gaussian_map", 1)
                      if publish_gaussian_map else None)
        self._p2pub = (node.create_publisher(PointCloud2, "/gaussian_map_pc2", 1)
                       if publish_pointcloud2 else None)

    def publish(self, snap: dict, stamp, gaussian_cloud: bool = True) -> None:
        """发布一次快照（snapshot_model 的返回 dict；调用方已在锁外）。

        gaussian_cloud=False 时只发 PointCloud2（毫秒级）——GaussianCloud 构建
        ~600-800ms（Python 逐元素循环），实测即使在独立 Reentrant 回调组也会
        饿死 sync 回调（on_frame 延迟 ~570ms，tf 降到 ~1Hz），故高频走 PC2、
        GaussianCloud 低频（节点侧每 5 次 timer 发一次）。
        """
        n = len(snap["means"])
        if n == 0:
            return
        if n > self.max_points:
            stride = (n + self.max_points - 1) // self.max_points
            snap = {k: v[::stride] for k, v in snap.items()}

        if gaussian_cloud and self._gpub is not None:
            cloud = GaussianCloud()
            cloud.header.stamp = _to_time(stamp)
            cloud.header.frame_id = self.world_frame
            means = snap["means"]
            rgb = snap["rgb"]
            op = snap["opacity"]
            sc = snap["scales"]
            rot = snap["rot"]
            cloud.gaussians = [
                Gaussian(x=float(means[i, 0]), y=float(means[i, 1]), z=float(means[i, 2]),
                         opacity=float(op[i, 0]),
                         scale_x=float(sc[i, 0]), scale_y=float(sc[i, 1]), scale_z=float(sc[i, 2]),
                         qx=float(rot[i, 0]), qy=float(rot[i, 1]),
                         qz=float(rot[i, 2]), qw=float(rot[i, 3]),
                         r=float(rgb[i, 0]), g=float(rgb[i, 1]), b=float(rgb[i, 2]),
                         feature=[])          # Phase 6 语言潜变量
                for i in range(len(means))]
            self._gpub.publish(cloud)

        if self._p2pub is not None:
            self._p2pub.publish(_make_pointcloud2(
                snap["means"], snap["opacity"], snap["rgb"], stamp, self.world_frame))
