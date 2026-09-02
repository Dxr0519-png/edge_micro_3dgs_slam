"""SE(3) 位姿工具（文档 §5：统一实现 se3_exp / se3_log）。

位姿约定（与 SplaTAM 一致）：
- 位姿矩阵为 **w2c（世界→相机）**，即 `p_cam = T @ p_world`。
- 四元数统一 wxyz 顺序。
- `se3_exp(delta)` 做**相机系左扰动**：`T_new = se3_exp(delta) @ T`，
  其中 delta = [ω(3), v(3)]，ω 为相机系旋转、v 为相机系平移（米/弧度）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ 四元数
def build_rotation(q: torch.Tensor) -> torch.Tensor:
    """四元数 (N,4) wxyz → 旋转矩阵 (N,3,3)。裁剪自 SplaTAM utils/slam_external.py。"""
    q = F.normalize(q)
    rot = torch.zeros((q.size(0), 3, 3), device=q.device, dtype=q.dtype)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def quat_mult(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """四元数相乘 (N,4) × (N,4)，wxyz。"""
    q1 = F.normalize(q1)
    q2 = F.normalize(q2)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 (B,3,3) → 四元数 (B,4) wxyz。裁剪自 SplaTAM utils/slam_helpers.py。"""
    return _rotmat_to_quat(M)


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(B,3,3) → (B,4) wxyz（Shepperd 法）。"""
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    case = tr > 0
    if case.any():
        s = torch.sqrt(tr[case] + 1.0) * 2
        q[case, 0] = 0.25 * s
        q[case, 1] = (R[case, 2, 1] - R[case, 1, 2]) / s
        q[case, 2] = (R[case, 0, 2] - R[case, 2, 0]) / s
        q[case, 3] = (R[case, 1, 0] - R[case, 0, 1]) / s
    for i, j, k in ((1, 2, 0), (2, 0, 1), (0, 1, 2)):
        m = (R[:, i, i] > R[:, j, j]) & (R[:, i, i] > R[:, k, k]) & ~case
        if m.any():
            s = torch.sqrt(R[m, i, i] - R[m, j, j] - R[m, k, k] + 1.0) * 2
            q[m, i + 1] = 0.25 * s
            q[m, 0] = (R[m, k, j] - R[m, j, k]) / s
            q[m, j + 1] = (R[m, j, i] + R[m, i, j]) / s
            q[m, k + 1] = (R[m, k, i] + R[m, i, k]) / s
    return F.normalize(q + 1e-12)


# ------------------------------------------------------------------ se(3)
def _skew(w: torch.Tensor) -> torch.Tensor:
    """(3,) → 反对称矩阵 (3,3)。in-place 赋值保留 autograd 链。"""
    out = torch.zeros(3, 3, device=w.device, dtype=w.dtype)
    out[0, 1] = -w[2]; out[0, 2] = w[1]
    out[1, 0] = w[2]; out[1, 2] = -w[0]
    out[2, 0] = -w[1]; out[2, 1] = w[0]
    return out


def se3_exp(delta: torch.Tensor) -> torch.Tensor:
    """se(3) 左扰动指数映射：(6,) → (4,4)。

    delta = [ω(3), v(3)]：ω 为旋转向量（弧度），v 为相机系平移。
    连续公式（θ→0 用泰勒系数 + epsilon 分母，无梯度断点）：
        R = I + (sinθ/θ) [ω]× + ((1-cosθ)/θ²) [ω]×²
        J = I + ((1-cosθ)/θ²) [ω]× + ((θ-sinθ)/θ³) [ω]×²
    """
    w, v = delta[:3], delta[3:]
    th = w.norm()
    th_s = th + 1e-12                        # epsilon 分母：避免 0/0，保持光滑梯度
    wx = _skew(w)
    wx2 = wx @ wx
    sin_ov = torch.sin(th_s) / th_s
    one_cos = (1 - torch.cos(th_s)) / th_s ** 2
    th_sin = (th_s - torch.sin(th_s)) / th_s ** 3
    T = torch.eye(4, device=delta.device, dtype=delta.dtype)
    T[:3, :3] = torch.eye(3, device=w.device, dtype=w.dtype) + sin_ov * wx + one_cos * wx2
    T[:3, 3] = (torch.eye(3, device=w.device, dtype=w.dtype) + one_cos * wx + th_sin * wx2) @ v
    return T


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """se(3) 对数映射：(4,4) → (6,)。se3_exp 的逆（左扰动约定）。"""
    R, t = T[:3, :3], T[:3, 3]
    cos_th = torch.clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0)
    th = torch.acos(cos_th)
    if th < 1e-8:
        return torch.cat([torch.zeros(3, device=T.device), t])
    w = th / (2 * torch.sin(th)) * torch.stack([
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    wx = _skew(w)
    J_inv = torch.eye(3, device=T.device, dtype=T.dtype) - 0.5 * wx \
        + (1.0 / th ** 2 - (1 + cos_th) / (2 * th * torch.sin(th))) * (wx @ wx)
    v = J_inv @ t
    return torch.cat([w, v])


def invert_pose(T: torch.Tensor) -> torch.Tensor:
    """(4,4) 位姿求逆。"""
    R, t = T[:3, :3], T[:3, 3]
    out = torch.eye(4, device=T.device, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out
