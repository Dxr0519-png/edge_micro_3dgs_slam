"""Phase 3 §4 视锥剔除：世界系高斯 → 相机视锥内的布尔掩码。

优化时只对 visible 高斯反传；不可见高斯冻结（不进入光栅化 → 无梯度 → Adam 不步进）。
"""
from __future__ import annotations

import numpy as np
import torch


def frustum_visible(xyz: torch.Tensor, T_wc, K, H: int, W: int,
                    margin: float | None = None, near: float = 0.01,
                    scales: torch.Tensor | None = None,
                    safety: float = 1.5) -> torch.Tensor:
    """世界系高斯 → 视锥内布尔掩码 (N,)。

    判据：相机系 z > near，且投影像素 u ∈ [−margin, W+margin]、v ∈ [−margin, H+margin]。
    margin 必须覆盖高斯的**屏幕足迹**（f·σ/z）——中心在图像边缘外的高斯
    其半透明尾巴仍可能贡献边缘像素，余量不足会误剔除（剔除前后渲染不一致）。

    margin 默认自动计算：max(fx,fy)·max_scale/z_min·safety（需要传 scales）；
    显式传 margin 则覆盖（文档 §4 签名兼容）。

    参数:
        xyz:    (N, 3) 世界系坐标（cuda）
        T_wc:   (4,4) w2c（tensor 或 numpy）
        K:      (3,3) 内参（tensor 或 numpy）
        H, W:   渲染分辨率
        margin: 像素余量；None 且给 scales 时自动（推荐）
        scales: (N,) 或 (N,3) 各高斯尺度（米），自动余量用
        safety: 自动余量安全系数（足迹 ≈ f·σ/z 是近似，留余量）
    """
    if not isinstance(T_wc, torch.Tensor):
        T_wc = torch.as_tensor(np.asarray(T_wc, dtype=np.float32)).cuda().float()
    if not isinstance(K, torch.Tensor):
        K = torch.as_tensor(np.asarray(K, dtype=np.float32), device=xyz.device)

    pc = xyz @ T_wc[:3, :3].T + T_wc[:3, 3]          # (N, 3) 相机系
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # z 略 ≤ 0 的点除法发散 → clamp 防 NaN（随后被 z > near 过滤）
    z = pc[:, 2].clamp(min=1e-6)
    if margin is None:
        if scales is not None:
            s = scales if scales.dim() == 1 else scales.max(dim=-1).values
            foot = max(float(fx), float(fy)) * s / z.clamp(min=near)
            margin = float(safety * foot.max())
        else:
            margin = 0.2
    u = fx * pc[:, 0] / z + cx
    v = fy * pc[:, 1] / z + cy
    return (pc[:, 2] > near) & (u > -margin) & (u < W + margin) \
        & (v > -margin) & (v < H + margin)
