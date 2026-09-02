"""Phase 3 §5 关键帧管理：按位姿平移/旋转阈值插入关键帧。

阈值与文档 §5 一致：dt > 0.1m 或 dth > 3° 即插入。
"""
from __future__ import annotations

import numpy as np


def angle_between(R1, R2) -> float:
    """两旋转矩阵的相对角（度）：R_rel = R1 @ R2.T，θ = arccos((tr−1)/2)。"""
    R_rel = np.asarray(R1, np.float64) @ np.asarray(R2, np.float64).T
    cos_t = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_t)))


class KeyframeManager:
    """按文档 §5：dt > dt_thresh 或 dth > dth_thresh 则插入关键帧。

    无状态——T_last 由调用方维护（在线循环里用上一关键帧位姿）。
    T 为 w2c (4,4)。

    ⚠️ 位移度量修正：平移差用**相机位置差**（c2w 平移）而非 w2c 平移差。
    w2c 平移 = −Rᵀ·c，旋转与平移耦合——相机绕场景中心俯视时（前向近似
    垂直），w2c 平移差会虚高（实测帧间实际位移 5.5cm、w2c 差 51cm），
    导致关键帧误插入。文档 §5 的"dt > 0.1m"语义即相机物理位移，故取 c2w。
    """

    def __init__(self, dt_thresh: float = 0.1, dth_thresh_deg: float = 3.0):
        self.dt_thresh = dt_thresh
        self.dth_thresh_deg = dth_thresh_deg

    def should_insert(self, T_new, T_last) -> bool:
        T_new = np.asarray(T_new, np.float64)
        T_last = np.asarray(T_last, np.float64)
        c_new = _c2w_translation(T_new)
        c_last = _c2w_translation(T_last)
        dt = np.linalg.norm(c_new - c_last)          # 相机位置位移（米）
        dth = angle_between(T_new[:3, :3], T_last[:3, :3])
        return dt > self.dt_thresh or dth > self.dth_thresh_deg


def _c2w_translation(T_w2c: np.ndarray) -> np.ndarray:
    """w2c (4,4) → 相机在世界系的位置（c2w 平移 = −Rᵀ·t）。"""
    R = T_w2c[:3, :3]
    return -R.T @ T_w2c[:3, 3]
