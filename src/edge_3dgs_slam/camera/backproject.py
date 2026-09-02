"""深度反投影：深度图 -> 相机系（或世界系）点云。Phase 2 复用。

Phase 4：backproject_torch 为 GPU 版本（ICP 初值 / 建图加高斯用，消除
numpy round-trip 的 CPU-GPU 同步与全分辨率 CPU 反投影）。
"""
import numpy as np
import torch


def backproject_torch(depth: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU 深度反投影：(H,W) 深度 + (3,3) K → 相机系点云与有效掩码。

    返回 (pts, valid)：pts (H,W,3) 相机系（无效处 NaN），valid (H,W) bool。
    与 numpy 版 backproject 逐位一致（同口径，Phase 3 单元检查验证）。
    """
    H, W = depth.shape
    u = torch.arange(W, dtype=depth.dtype, device=depth.device)
    v = torch.arange(H, dtype=depth.dtype, device=depth.device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts = torch.stack([x, y, z], dim=-1)
    valid = torch.isfinite(z) & (z > 0)
    pts = pts.clone()
    pts[~valid] = torch.nan
    return pts, valid


def backproject(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray | None = None):
    """把深度图反投影为点云。

    参数:
        depth: (H, W) float32，单位米（SyncedFrame.depth）
        K:     (3, 3) 内参矩阵
        T_wc:  (4, 4) 世界->相机位姿；None 时返回相机系点云（Phase 1 验证用）

    返回:
        pts:  (H, W, 3) 点云（无效像素处为 NaN）
        valid: (H, W) bool 有效掩码（深度有限且 > 0）
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)
    valid = np.isfinite(z) & (z > 0)
    pts = pts_cam.copy()
    pts[~valid] = np.nan
    if T_wc is not None:
        # p_world = R^T (p_cam - t)；行向量形式：(p_cam - t) @ R
        R, t = T_wc[:3, :3], T_wc[:3, 3]
        pts[valid] = (pts_cam[valid] - t) @ R
    return pts, valid


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """把 (N, 3) 相机系点投影到像素坐标（验证反投影的逆运算）。"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u = x * K[0, 0] / z + K[0, 2]
    v = y * K[1, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


# 项目世界系约定：x 右、y 前、z 上（z-up）。
# 相机系（x 右 / y 下 / z 前，OpenCV 约定）→ 世界系旋转：
#   world = cam @ CAMERA_TO_ZUP.T
CAMERA_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, -1.0, 0.0]], dtype=np.float64)


def to_zup_frame(pts: np.ndarray) -> np.ndarray:
    """相机系点云转世界系（z-up），便于查看器（CloudCompare 等按 z-up 显示）打开。

    直接打开相机系点云会"立起来/反着"——不是数据错误，是坐标约定不同。
    原点即相机位置，+y 为相机正前方（看向场景）。
    """
    return pts @ CAMERA_TO_ZUP.T
