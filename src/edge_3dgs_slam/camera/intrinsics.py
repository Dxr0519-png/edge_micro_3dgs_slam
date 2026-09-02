"""相机内参数据类：从 ROS2 CameraInfo 构造，供全项目复用。"""
from dataclasses import dataclass
import numpy as np


@dataclass
class CameraIntrinsics:
    """D435i 颜色内参（rectified 后深度已对齐到颜色，共用同一 K）。"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_info(cls, info) -> "CameraIntrinsics":
        """从 sensor_msgs/msg/CameraInfo 构造（取 K 矩阵的前两行）。"""
        return cls(
            fx=float(info.k[0]), fy=float(info.k[4]),
            cx=float(info.k[2]), cy=float(info.k[5]),
            width=int(info.width), height=int(info.height),
        )

    def K(self) -> np.ndarray:
        """返回 3x3 内参矩阵（float64）。"""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]], dtype=np.float64)
