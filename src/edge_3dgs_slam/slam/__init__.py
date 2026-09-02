"""§8 统一 SLAM 接口（供 Phase 4/6 复用）。

- `build_map(rgb, depth, K, poses)`：离线建图（给定位姿序列 → 高斯模型）
- `track_one_frame(frame, model, T_init)`：在线追踪一帧位姿
- `KeyframeManager` / `SLAMBackend`：Phase 3 §5 关键帧管理 + 异步建图后端
"""
from __future__ import annotations

import numpy as np
import torch

from ..camera import SyncedFrame
from ..gaussian.model import GaussianModel
from ..utils.se3 import invert_pose, se3_log
from .backend import SLAMBackend
from .init import init_from_depth
from .keyframe import KeyframeManager
from .mapping import map_keyframe
from .tracking import track


def build_map(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, poses: np.ndarray,
              keyframe_every: int = 5, map_iters: int = 60, stride: int = 2,
              verbose: bool = False) -> GaussianModel:
    """离线建图：RGB-D 序列 + 已知位姿（w2c）→ 世界系高斯模型。

    参数:
        rgb:   (T, H, W, 3) uint8
        depth: (T, H, W) float32 米
        K:     (3, 3) 内参
        poses: (T, 4, 4) 世界→相机（w2c）位姿
        keyframe_every: 每 N 帧取一个关键帧参与建图
        map_iters:      每关键帧的属性优化迭代数

    返回:
        GaussianModel（CUDA）
    """
    T = rgb.shape[0]
    frames = [SyncedFrame(rgb=rgb[t], depth=depth[t], K=K, stamp=float(t)) for t in range(T)]

    model = init_from_depth(frames[0], poses[0], stride=stride)
    if verbose:
        print(f"[build_map] 初始 {model.num_gaussians} 高斯（首帧反投影）")
    for t in range(keyframe_every, T, keyframe_every):
        stat = map_keyframe(frames[t], model, poses[t], iters=map_iters)
        if verbose:
            print(f"[build_map] 关键帧 {t}: {stat}")
    return model


def track_one_frame(frame: SyncedFrame, model: GaussianModel, T_init: np.ndarray,
                    iters: int = 8, lr: float = 1e-2, T_prev: np.ndarray | None = None,
                    max_rot_deg: float = 90.0, max_trans_m: float = 1.0) -> np.ndarray | None:
    """在线追踪一帧，返回 (4,4) w2c numpy 位姿。

    T_prev 给定时做**时间一致性检测**（SLAM 标准鲁棒机制）：
    估计位姿与上一帧的旋转/平移差超过阈值（相机运动物理上连续，
    出现 0.x°→~180° 的跳变一定是跟踪落入翻转极小/表示符号翻转），
    判定本帧跟踪失败并返回 None——调用方应回退到连续外推初值，
    避免翻转结果污染后续轨迹（外推基于干净帧时污染链即被断开）。
    """
    T = track(frame, model, T_init, iters=iters, lr=lr)
    T_np = T.detach().cpu().numpy()
    if T_prev is not None:
        d = se3_log(T @ invert_pose(torch.as_tensor(T_prev, dtype=torch.float32).cuda()))
        d = d.detach().cpu().numpy()
        if np.linalg.norm(d[:3]) > np.deg2rad(max_rot_deg) or np.linalg.norm(d[3:]) > max_trans_m:
            return None
    return T_np
