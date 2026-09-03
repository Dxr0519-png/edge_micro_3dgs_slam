"""Phase 4 ICP 粗对齐初值：帧到帧 / 帧到模型 point-to-plane ICP，替换匀速外推初值。

背景（docs/03 实测结论）：低迭代跟踪发散根源是**初值漂移累积**而非迭代数
不足（真值初值下 3it 仅 3.14cm vs 匀速初值 12.3cm@100帧、81.6cm@200帧）。
ICP 用深度几何做粗对齐，把光栅化迭代从 6 次降到 1-2 次而不牺牲质量——
10 FPS = 100ms/帧预算下唯一能省 130-220ms/帧的杠杆。

**实现选择（2026-08-28 实测）**：
1. **帧到模型为主**：模型 depth 渲染在上一处理帧位姿（锁内 ~15ms@160×120，
   锁外 GN ~8ms）。建图模型的深度质量实测：覆盖 97-100%、z 差中位 2.3cm
   （Replica office0，60k+ 高斯）——帧到模型用模型做**全局锚**，消除帧到帧
   对齐噪声的随机游走（实测 1-2cm/步 × 50 帧累积到 6-10cm 后 ICP 门槛连锁
   失效 → 发散）。早期放弃 f2m 是因为合成探针模型深度差 145cm（33% 覆盖率
   特例），真实建图模型不适用该结论。
2. 纯 numpy CPU 实现 GN——Jetson 实测每微小 CUDA kernel 启动 ~0.2ms，
   GPU 版 GN 循环 67ms 中真正算力仅 ~5ms；numpy 向量化算子 ~30µs/个。

流程（帧到模型，每处理帧）：
1. 锁内：模型 depth 无梯度渲染（T_prev 视角，复用 render_prepare 共享转换）
2. 锁外：当前帧点云 + GN×6 求解模型系左扰动（目标 = 模型 depth）
3. 门槛：coarse-to-fine 内点门（0.10→0.04m）、末端 ratio、修正量不信门
4. 退化 → 回退 T_guess（恒速初值）

位姿约定：w2c（世界→相机），与 track/se3.py 一致；左扰动 T_new = se3_exp(δ)@T。
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..gaussian.render import render, render_prepare
from ..utils.frame_utils import downsample_frame
from ..utils.se3 import invert_pose

_ICP_W, _ICP_H = 160, 120        # ICP 分辨率（反投影点云规模 19,200）
_STRIDE = 2                      # 点云 2×2 抽样（4,800 点；表面平滑，对应质量不降）
_INLIER_D_M = 0.10               # 初始对应点深度差截断（米；粗对齐阶段）
_RATIO_MIN = 0.25                # 对应占比低于此回退恒速初值
_GN_ITERS = 6
# coarse-to-fine 内点门（米）：粗 10cm → 精 4cm。实测平墙段 ICP 会在 10cm 门
# 内滑入错误局部最优（ratio 0.92 但平移误差 15cm，Replica t=12）；收紧后
# 该位姿 z 差 >4cm 塌缩 → 回退初值 → 调度器下一帧强制处理恢复。
_INLIER_SCHEDULE = (0.10, 0.08, 0.06, 0.05, 0.04, 0.04)
# 末端 ratio 门槛：帧到帧用 0.8（最紧 4cm 门口径，好对齐 0.9+）；帧到模型
# 模型深度噪声 2.3cm 中位 → 4cm 门 ratio 上限 ~0.85，用 10cm 口径 0.85
_RATIO_ACCEPT_F2F = 0.8
_RATIO_ACCEPT_F2M = 0.85
_F2M_FINAL_GATE = 0.10           # 帧到模型末端判定用 10cm 门（模型深度噪声容忍）
# 修正量不信门（度）：ICP 相对 CV 初值的旋转修正 > 此值 → 回退 CV。
# 好 ICP 修正通常 <2°（CV 初值 ~1-1.8°）；>2.5° 的"大修正"实测是旋转爆炸
# （Replica t=13：CV 2.79° → ICP 6.06°，ratio 0.56 仍被旧门放行）
_DISTRUST_ROT_DEG = 2.5
_MAX_ROT_STEP_RAD = np.deg2rad(10.0)   # 单步旋转过大视为退化
_DAMP = 1e-6                     # Gauss-Newton 阻尼（正规方程正则）


def _inv4(T: np.ndarray) -> np.ndarray:
    """(4,4) 位姿求逆（numpy，SE(3) 闭式）。"""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _se3_exp_np(delta: np.ndarray) -> np.ndarray:
    """se(3) 左扰动指数映射 (6,) → (4,4)（numpy，Rodrigues 闭式）。"""
    w, v = delta[:3], delta[3:]
    th = np.linalg.norm(w)
    wx = np.array([[0.0, -w[2], w[1]],
                   [w[2], 0.0, -w[0]],
                   [-w[1], w[0], 0.0]])
    if th < 1e-10:
        R = np.eye(3)
        V = np.eye(3)
    else:
        R = np.eye(3) + np.sin(th) / th * wx + (1 - np.cos(th)) / th ** 2 * (wx @ wx)
        V = np.eye(3) + (1 - np.cos(th)) / th ** 2 * wx + (th - np.sin(th)) / th ** 3 * (wx @ wx)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def _se3_log_np(T: np.ndarray) -> np.ndarray:
    """se(3) 对数映射 (4,4) → (6,)（numpy，左扰动约定；旋转部分用于不信门）。"""
    R, t = T[:3, :3], T[:3, 3]
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-8:
        return np.concatenate([np.zeros(3), t])
    w = th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2],
                                          R[0, 2] - R[2, 0],
                                          R[1, 0] - R[0, 1]])
    wx = np.array([[0.0, -w[2], w[1]],
                   [w[2], 0.0, -w[0]],
                   [-w[1], w[0], 0.0]])
    J_inv = np.eye(3) - 0.5 * wx + (1.0 / th ** 2 - (1 + cos_th) / (2 * th * np.sin(th))) * (wx @ wx)
    return np.concatenate([w, J_inv @ t])


def se3_motion_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """位姿 A→B 的 se(3) 增量 (6,)（numpy，2026-09-02 ICP 运动门控用）。

    backend 每处理帧用上一帧位姿与恒速预测位姿的差判断"相机是否在动"——
    静态/慢速段恒速初值已足够准，ICP（~80ms/帧，实测含失败重试）纯属浪费。
    """
    return _se3_log_np(_inv4(A) @ B)


def _depth_normals(d_target: np.ndarray, K: np.ndarray) -> np.ndarray:
    """目标 depth 的中心差分数向：(H,W,3) 单位法向（朝相机，z 分量恒正）。"""
    H, W = d_target.shape
    fy, fx = K[1, 1], K[0, 0]
    du = np.zeros_like(d_target)
    dv = np.zeros_like(d_target)
    du[:, 1:-1] = (d_target[:, 2:] - d_target[:, :-2]) / 2.0
    dv[1:-1, :] = (d_target[2:, :] - d_target[:-2, :]) / 2.0
    n = np.stack([-fy * du, -fx * dv, np.ones_like(d_target)], axis=-1)
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norms, 1e-8)


def _backproject_subsampled(depth_t: np.ndarray, K_use: np.ndarray) -> np.ndarray:
    """相机系点云，2×2 抽样（先整图反投影再抽点——先抽深度会×2 错位，实测坑）。"""
    fx, fy, cx, cy = K_use[0, 0], K_use[1, 1], K_use[0, 2], K_use[1, 2]
    s = _STRIDE
    uu, vv = np.meshgrid(np.arange(depth_t.shape[1], dtype=np.float32),
                         np.arange(depth_t.shape[0], dtype=np.float32), indexing="xy")
    valid = np.isfinite(depth_t) & (depth_t > 0)
    pts = np.stack([(uu - cx) * depth_t / fx, (vv - cy) * depth_t / fy,
                    depth_t], axis=-1)
    return pts[::s, ::s][valid[::s, ::s]].reshape(-1, 3)      # (N/4,3)


def _gn_align(pts_flat: np.ndarray, d_target: np.ndarray, T_prev: np.ndarray,
              T_guess: np.ndarray, K_use: np.ndarray,
              gn_iters: int, inlier_d: float, ratio_min: float,
              ratio_accept: float, final_gate: float,
              distrust_rot_deg: float) -> tuple[np.ndarray, dict]:
    """point-to-plane Gauss-Newton 核心（帧到帧 / 帧到模型共用，纯 numpy CPU）。

    T_rel = T_prev @ inv(T_guess)（当前帧点 → 目标相机系）；返回 (T_out, stats)。
    """
    stats = {"ratio": 0.0, "iter_used": 0, "fallback": False}
    Hm, Wm = d_target.shape
    fx, fy, cx, cy = K_use[0, 0], K_use[1, 1], K_use[0, 2], K_use[1, 2]
    if pts_flat.shape[0] < 100 or not (d_target > 0).any():
        stats["fallback"] = True
        return T_guess, stats
    n_target = _depth_normals(d_target, K_use)                # (H,W,3)

    T_rel = T_prev @ _inv4(T_guess)
    ratio = 0.0
    for it in range(gn_iters):
        d_gate = inlier_d if it >= len(_INLIER_SCHEDULE) else _INLIER_SCHEDULE[it]
        pts_m = (T_rel[:3, :3] @ pts_flat.T + T_rel[:3, 3:4]).T   # (N,3) 目标系
        z = pts_m[:, 2]
        if not (z > 1e-4).any():
            break
        u = pts_m[:, 0] * fx / z + cx
        v = pts_m[:, 1] * fy / z + cy
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        in_img = (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < Hm) & (z > 0)
        ui_c = np.clip(ui, 0, Wm - 1)
        vi_c = np.clip(vi, 0, Hm - 1)
        d_sam = d_target[vi_c, ui_c]                          # 对应目标深度
        corr = in_img & (d_sam > 0) & (np.abs(z - d_sam) < d_gate)
        n_corr = int(corr.sum())
        ratio = n_corr / max(pts_flat.shape[0], 1)
        if n_corr < 50 or ratio < ratio_min:
            # 中途塌陷 = 初值差/大转角——GN 结果不可信，标记 fallback
            # （原实现只 break 不标记，发散初值被当成功喂给光栅化器，实测
            # Replica 转角 t=39 ratio 0.38 → ICP 4.63° 发散 → 全序列雪崩）
            stats["fallback"] = True
            break
        qx = d_sam[corr] * (u[corr] - cx) / fx
        qy = d_sam[corr] * (v[corr] - cy) / fy
        q = np.stack([qx, qy, d_sam[corr]], axis=-1)
        x = pts_m[corr]
        nq = n_target[vi_c[corr], ui_c[corr]]
        e = np.sum(nq * (x - q), axis=-1)                     # 标量残差
        J = np.concatenate([np.cross(x, nq, axis=-1), nq], axis=-1)   # (m,6)
        A = J.T @ J + _DAMP * np.eye(6, dtype=np.float64)
        delta = np.linalg.solve(A, -(J.T @ e))                # (6,)
        if np.linalg.norm(delta[:3]) > _MAX_ROT_STEP_RAD:     # 退化保护
            stats["fallback"] = True
            break
        T_rel = _se3_exp_np(delta) @ T_rel
        stats["iter_used"] += 1
    stats["ratio"] = ratio

    # 末端 ratio 门槛（目标深度噪声口径：f2f 4cm / f2m 10cm）
    if stats["fallback"] or ratio < ratio_accept:
        stats["fallback"] = True
        return T_guess, stats
    T_out = _inv4(T_rel) @ T_prev
    # 修正量不信门：ICP 相对 CV 的旋转/平移修正过大 → 回退 CV
    # （旋转 > 2.5° 实测是旋转爆炸；平移 > 12cm 是平墙浅盆地——ratio 高但
    #   位姿错 15cm 的局部最优，0.7 门下会混入，必须单独拦截）
    w = _se3_log_np(_inv4(T_guess) @ T_out)
    if np.linalg.norm(w[:3]) > np.deg2rad(distrust_rot_deg) \
            or np.linalg.norm(w[3:]) > 0.12:
        stats["fallback"] = True
        return T_guess, stats
    return T_out, stats


def icp_init(frame, depth_prev: np.ndarray, T_prev: np.ndarray, T_guess: np.ndarray,
             K: np.ndarray, W: int = _ICP_W, H: int = _ICP_H,
             gn_iters: int = _GN_ITERS, inlier_d: float = _INLIER_D_M,
             ratio_min: float = _RATIO_MIN,
             return_stats: bool = False):
    """帧到帧 ICP（备用路径）：目标 = 上一处理帧的传感器深度。纯 CPU，可锁外执行。

    主路径为 icp_init_model（帧到模型锚定，消除随机游走）；本函数保留用于
    对比/探针/模型不可用时的回退。
    """
    stats = {"ratio": 0.0, "iter_used": 0, "fallback": False, "ms": 0.0}
    t0 = time.perf_counter()
    fd = downsample_frame(frame, W, H)
    depth_t = fd.depth.astype(np.float32)
    d_prev = np.asarray(depth_prev, dtype=np.float32)
    K_use = fd.K if fd.K is not None else K
    pts_flat = _backproject_subsampled(depth_t, K_use)
    T_out, gstats = _gn_align(pts_flat, d_prev, T_prev, T_guess, K_use,
                              gn_iters, inlier_d, ratio_min,
                              _RATIO_ACCEPT_F2F, _INLIER_SCHEDULE[-1],
                              _DISTRUST_ROT_DEG)
    stats.update(gstats)
    stats["ms"] = (time.perf_counter() - t0) * 1e3
    return (T_out, stats) if return_stats else T_out


def icp_render_model_depth(model, T_prev: np.ndarray, K: np.ndarray,
                           W: int = _ICP_W, H: int = _ICP_H,
                           prep: dict | None = None) -> np.ndarray:
    """锁内调用：模型 depth 无梯度渲染（T_prev 视角），返回 (H,W) numpy。

    prep（render_prepare 输出）与 track 共享存储层浮点转换。
    """
    T_prev_t = torch.as_tensor(np.asarray(T_prev, dtype=np.float32)).cuda().float()
    K_t = torch.as_tensor(np.asarray(K, dtype=np.float32))
    with torch.no_grad():
        if prep is None:
            prep = render_prepare(model, gaussians_grad=False)
        _, d_model, _, _, _ = render(model, T_prev_t, K_t, W, H,
                                     gaussians_grad=False, camera_grad=False,
                                     needs_depth=True, prep=prep)
    return d_model.squeeze(0).cpu().numpy()


def icp_init_model(frame, d_model: np.ndarray, T_prev: np.ndarray, T_guess: np.ndarray,
                   K: np.ndarray, W: int = _ICP_W, H: int = _ICP_H,
                   gn_iters: int = _GN_ITERS, inlier_d: float = _INLIER_D_M,
                   ratio_min: float = _RATIO_MIN,
                   return_stats: bool = False):
    """帧到模型 ICP（主路径）：目标 = 模型 depth 渲染（T_prev 视角）。

    模型是**全局锚**——帧到帧的 1-2cm 对齐噪声在链条上随机游走（50 帧累积
    到 6-10cm 后 ICP 门槛连锁失效 → 发散，实测）；帧到模型对齐误差被模型
    的绝对位置约束，无随机游走。锁外执行（d_model 由调用方在锁内渲染）。
    """
    stats = {"ratio": 0.0, "iter_used": 0, "fallback": False, "ms": 0.0}
    t0 = time.perf_counter()
    fd = downsample_frame(frame, W, H)
    depth_t = fd.depth.astype(np.float32)
    K_use = fd.K if fd.K is not None else K
    pts_flat = _backproject_subsampled(depth_t, K_use)
    T_out, gstats = _gn_align(pts_flat, np.asarray(d_model, dtype=np.float32),
                              T_prev, T_guess, K_use,
                              gn_iters, inlier_d, ratio_min,
                              _RATIO_ACCEPT_F2M, _F2M_FINAL_GATE,
                              _DISTRUST_ROT_DEG)
    stats.update(gstats)
    stats["ms"] = (time.perf_counter() - t0) * 1e3
    return (T_out, stats) if return_stats else T_out
