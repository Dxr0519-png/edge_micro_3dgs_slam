"""同步帧数据类：RGB-D + 内参 + 时间戳，SLAM 各模块的标准输入。"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SyncedFrame:
    rgb:   np.ndarray   # (H, W, 3) uint8
    depth: np.ndarray   # (H, W) float32, 米
    K:     np.ndarray   # (3, 3) float64
    stamp: float        # 硬件时间戳（秒）
