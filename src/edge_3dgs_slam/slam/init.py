"""§4 深度反投影初始化：首帧 RGB-D → 世界系高斯（免 COLMAP）。

流程：Phase 1 的 `backproject` 反投影出世界系点云 → 可选均匀下采样 →
按像素-世界尺度初始化高斯 scale（SplaTAM 的 projective 法）→ GaussianModel。
"""
from __future__ import annotations

import numpy as np

from ..camera import SyncedFrame, backproject
from ..gaussian.model import GaussianModel


def init_from_depth(frame: SyncedFrame, T_wc: np.ndarray,
                    stride: int = 2, max_pts: int | None = None) -> GaussianModel:
    """从一帧 RGB-D 初始化世界系高斯模型。

    参数:
        frame:  SyncedFrame（rgb (H,W,3) uint8，depth (H,W) 米，K (3,3)）
        T_wc:   (4,4) 世界→相机位姿（w2c）
        stride: 像素下采样步长（控制初始高斯数，如 stride=2 → 1/4 像素数）
        max_pts: 上限保护（超出则再下采样）

    返回:
        GaussianModel（参数在 CUDA），variables 中 scene_radius 已按首帧深度设置。
    """
    H, W = frame.depth.shape
    pts_world, valid = backproject(frame.depth, frame.K, T_wc)   # Phase 1 反投影
    valid = valid & (frame.depth > 0)

    # 下采样：先在像素网格上取 stride 步长
    y_idx, x_idx = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    idx = y_idx.ravel(), x_idx.ravel()
    valid_s = valid[idx]

    pts = pts_world[idx][valid_s]
    colors = (frame.rgb[idx][valid_s] / 255.0).astype(np.float32)
    depth_z = frame.depth[idx][valid_s].astype(np.float64)      # 相机系 z == 深度值

    # 上限保护：超出则均匀抽稀
    if max_pts is not None and pts.shape[0] > max_pts:
        keep = np.random.default_rng(0).choice(pts.shape[0], max_pts, replace=False)
        pts, colors, depth_z = pts[keep], colors[keep], depth_z[keep]

    # 尺度初始化（SplaTAM projective 法）：像素对应世界尺寸 ≈ z / focal。
    # ×2 补偿低分辨率（320x240 下高斯间距 2px，σ 需 ≥2px 才连续覆盖表面）
    focal = (frame.K[0, 0] + frame.K[1, 1]) / 2.0
    scales = 2.0 * depth_z / focal

    model = GaussianModel.create_from_points(pts, colors, scales=scales, opacity=0.5)
    # 场景半径（用于后续剪枝/密化的尺度阈值，SplaTAM 约定）
    model.variables["scene_radius"] = float(np.max(depth_z)) / 3.0
    return model
