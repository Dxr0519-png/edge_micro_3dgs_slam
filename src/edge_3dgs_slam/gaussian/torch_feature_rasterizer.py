"""纯 PyTorch 慢速特征光栅化（Phase 6 §2 降级方案先行）。

与 inria kernel（diff_gaussian_rasterization）**同一套**投影/深度排序/α 权重公式，
把颜色通道换成 D 维特征通道：
    F_2d(pixel) = Σ_i f_i · α_i · Π_{j<i}(1−α_j) + T_final · bg

全原生 autograd op（matmul/exp/cumsum/gather/scatter）——dF/df 与 dF/dα 自动正确。

用途（docs/06 §2）：
- 验证特征 splat 数学正确性（与 CUDA 版特征图互验 < 1e-3）；
- D>3 时无 CUDA kernel 的降级路径（慢，仅小规模验证；蒸馏走 render_precomp 快速路径）。

算法（避免 O(H·W·N)）：逐高斯 footprint（3σ 轴对齐盒）展开像素条 + 全局
(px_idx, depth) 排序 + 段内前缀积合成，总代价 = Σ footprint 像素数。
"""
from __future__ import annotations

import numpy as np
import torch

from .model import FEATURE_KEY, GaussianModel
from .render import transform_to_frame

# 以下常量与 SplaTAM 携带的 diff-gaussian-rasterization-w-depth fork 逐项对齐
# （third_party/SplaTAM/diff-gaussian-rasterization-w-depth.git，互验校准结论）：
NEAR = 0.2           # in_frustum：p_view.z <= 0.2 剔除（注意不是 0.01！）
COV_LOWPASS = 0.3    # computeCov2D 低通：cov[0][0] += 0.3 / cov[1][1] += 0.3（至少 1px 宽）
CLAMP_FOV = 1.3      # computeCov2D 位置截断：txtz 截到 ±1.3·tanfov 后再算雅可比
ALPHA_MAX = 0.99     # α 上限 clamp（forward.cu render 同款）
ALPHA_MIN = 1.0 / 255.0   # 微小 α 剔除阈值（render 同款：无像素贡献）


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz 四元数 → (N,3,3) 旋转矩阵。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.zeros(N, 3, 3, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rasterize_feature(gaussians: GaussianModel, w2c, K, H: int, W: int,
                      features: torch.Tensor | None = None,
                      chunk: int = 4096,
                      bg: torch.Tensor | None = None,
                      debug_entries: bool = False) -> torch.Tensor:
    """慢速特征 splat：(D, H, W) 特征图（与 render_precomp 同输入语义）。

    参数:
        gaussians: 含 FEATURE_KEY 的 GaussianModel（无则必须传 features）
        w2c:       (4,4) 世界→相机（numpy 或 cuda tensor，与 render 同约定）
        K:         (3,3) 内参
        H, W:      输出分辨率
        features:  (N, D) 覆盖 params[FEATURE_KEY]（独立梯度追踪时用）
        chunk:     逐 chunk 处理的高斯数（控制峰值显存）
        bg:        (D,) 背景特征（默认 0）
    """
    if features is None:
        if FEATURE_KEY not in gaussians.params:
            raise ValueError("模型无 feature 通道：先 add_feature_dim 或传 features")
        features = gaussians.params[FEATURE_KEY]
    device = gaussians.means3D.device
    features = features.to(device).float()
    D = features.shape[1]

    if not isinstance(w2c, torch.Tensor):
        w2c = torch.as_tensor(np.asarray(w2c, dtype=np.float32)).cuda().float()

    K = torch.as_tensor(np.asarray(K, dtype=np.float64), device=device).double()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tanfovx = W / (2.0 * fx)                      # setup_camera 同款
    tanfovy = H / (2.0 * fy)
    limx = CLAMP_FOV * tanfovx
    limy = CLAMP_FOV * tanfovy

    t = transform_to_frame(gaussians, w2c, gaussians_grad=True, camera_grad=True)
    pts = t["means3D"]                                   # (N,3) 相机系
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    inv_z = 1.0 / z

    # ---- 投影：kernel 的精确 fp32 计算序（互验校准）----
    # point_image = ndc2Pix(p_proj)：p_hom = 转置存储的 projmatrix·p（列主序约定），
    # p_w = 1/(z+1e-7)，ndc2Pix = ((v+1)·W−1)·0.5。代数上 = fx·x/z + cx − 0.5，
    # 但 fp32 逐位不同——α 在 1/255 边界处 ±5e-6 的差异决定保留/剔除（g2322 案例）。
    p_hom_x = (2.0 * fx / W) * x + (-(W - 2.0 * cx) / W) * z
    p_hom_y = (2.0 * fy / H) * y + (-(H - 2.0 * cy) / H) * z
    p_w = 1.0 / (z + 1e-7)
    u = ((p_hom_x * p_w + 1.0) * W - 1.0) * 0.5
    v = ((p_hom_y * p_w + 1.0) * H - 1.0) * 0.5

    # ---- 2D 协方差 Σ2d = J·cov3d_cam·Jᵀ（fork computeCov2D 同款）----
    # ⚠️ 校准结论（单高斯 α 剖面暴力搜索，max|Δα|=0.0000）：是 J·C·Jᵀ 而非 Jᵀ·C·J
    # （glm 列主序约定使 2x2 块为该形式）；J 用「位置截断后」的坐标
    # （txtz 截到 ±1.3·tanfov 再乘回 z，出视锥高斯显著）；低通 cov += 0.3（至少 1px 宽）。
    tx = torch.clamp(x * inv_z, -limx, limx) * z
    ty = torch.clamp(y * inv_z, -limy, limy) * z
    J = torch.stack([
        torch.stack([fx * inv_z, torch.zeros_like(inv_z), -fx * tx * inv_z * inv_z], dim=-1),
        torch.stack([torch.zeros_like(inv_z), fy * inv_z, -fy * ty * inv_z * inv_z], dim=-1),
    ], dim=1)                                            # (N,2,3)
    if gaussians.is_isotropic:
        s2 = gaussians.scales() ** 2                      # (N,1)
        cov3d = torch.zeros(pts.shape[0], 3, 3, device=device)
        cov3d[:, 0, 0] = s2[:, 0]; cov3d[:, 1, 1] = s2[:, 0]; cov3d[:, 2, 2] = s2[:, 0]
    else:
        R = _quat_to_rotmat(t["unnorm_rotations"])        # (N,3,3) 相机系
        s = gaussians.scales()                            # (N,3)
        cov3d = R * s[:, None, :] ** 2 @ R.transpose(1, 2)
    cov2d = J @ cov3d @ J.transpose(1, 2)                 # (N,2,2)
    xx = cov2d[:, 0, 0] + COV_LOWPASS
    xy = cov2d[:, 0, 1]
    yy = cov2d[:, 1, 1] + COV_LOWPASS

    # ---- footprint：kernel getRect 同款（互验校准结论）----
    # 半径 = ceil(3·sqrt(λ1_LP))——2D 协方差最大特征值（**含低通 +0.3**），有 ceil
    # （即 vendored 源码 getRect 语义；此前 point_image 用错导致边界案例矛盾）。
    # 候选集是 **tile 级**（16×16 块）轴对齐范围（C 截断除 16），像素级盒不行：
    # tile 边界处 kernel 直接不渲染（g9661 tile4 整列截断 / g2291 tile12 整列保留）。
    mid = 0.5 * (xx + yy)
    det2d = xx * yy - xy * xy
    lam1 = mid + torch.sqrt(torch.clamp(mid * mid - det2d, min=0.1))
    r3 = torch.ceil(3.0 * torch.sqrt(lam1))
    grid_x = (W + 15) // 16
    grid_y = (H + 15) // 16
    tx0 = torch.trunc((u - r3) / 16.0).clamp(min=0, max=grid_x).long()
    tx1 = torch.trunc((u + r3 + 15.0) / 16.0).clamp(min=0, max=grid_x).long()
    ty0 = torch.trunc((v - r3) / 16.0).clamp(min=0, max=grid_y).long()
    ty1 = torch.trunc((v + r3 + 15.0) / 16.0).clamp(min=0, max=grid_y).long()
    x0 = (tx0 * 16).clamp(max=W - 1); x1 = (tx1 * 16 - 1).clamp(max=W - 1)
    y0 = (ty0 * 16).clamp(max=H - 1); y1 = (ty1 * 16 - 1).clamp(max=H - 1)
    n_px = (x1 - x0 + 1) * (y1 - y0 + 1)
    valid = (z > NEAR) & (n_px > 0) & torch.isfinite(u) & torch.isfinite(v)

    # ---- conic（Σ2d⁻¹，kernel 精确序：det_inv = 1/det 后乘，非除法）----
    det = xx * yy - xy * xy
    det_inv = 1.0 / det
    conic = torch.stack([yy * det_inv, -xy * det_inv, xx * det_inv], dim=-1)   # (N,3)

    opac = torch.sigmoid(gaussians.params["logit_opacities"]).reshape(-1)  # (N,)（单高斯 (1,1)→(1,) 不 squeeze 成标量）

    px_all: list[torch.Tensor] = []
    depth_all: list[torch.Tensor] = []
    alpha_all: list[torch.Tensor] = []
    feat_all: list[torch.Tensor] = []
    gid_all: list[torch.Tensor] = []

    n = pts.shape[0]
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        m = valid[sl]
        if not m.any():
            continue
        x0c, x1c = x0[sl][m], x1[sl][m]
        y0c, y1c = y0[sl][m], y1[sl][m]
        c = (x1c - x0c + 1) * (y1c - y0c + 1)             # 每高斯 footprint 像素数
        total = int(c.sum())
        if total == 0:
            continue
        g_idx = torch.arange(len(c), device=device).repeat_interleave(c)   # 条目→高斯
        off = torch.arange(total, device=device) - (torch.cumsum(c, 0) - c).repeat_interleave(c)
        w_rect = (x1c - x0c + 1)[g_idx]                    # 每条目所属高斯的 rect 宽
        dx = off % w_rect
        dy = off // w_rect
        px = (x0c[g_idx] + dx) + (y0c[g_idx] + dy) * W     # 行优先线性像素索引

        u_g = u[sl][m][g_idx]; v_g = v[sl][m][g_idx]
        # pixf = (x, y)（render kernel 无 +0.5）；d = point_image − pixf（kernel 序）
        ddx = u_g - (px % W).float()
        ddy = v_g - (px // W).float()
        con = conic[sl][m][g_idx]                         # (P,3)
        # power（kernel 序）：-0.5f*(con.x*dx*dx + con.z*dy*dy) - con.y*dx*dy
        power = -0.5 * (con[:, 0] * ddx * ddx + con[:, 2] * ddy * ddy) \
            - con[:, 1] * ddx * ddy
        alpha = torch.clamp(opac[sl][m][g_idx] * torch.exp(power), max=ALPHA_MAX)
        keep = (power <= 0.0) & (alpha >= ALPHA_MIN)
        if not keep.any():
            continue
        px_all.append(px[keep])
        depth_all.append(z[sl][m][g_idx][keep])
        alpha_all.append(alpha[keep])
        feat_all.append(features[sl][m][g_idx][keep])
        gid_all.append((start + torch.where(m)[0])[g_idx][keep])

    if not px_all:
        return torch.zeros(D, H, W, device=device)

    px_s_all = torch.cat(px_all)
    depth_s_all = torch.cat(depth_all)
    alpha_s_all = torch.cat(alpha_all)
    feat_s_all = torch.cat(feat_all)

    # ---- 全局 (px_idx, depth) 排序 + 段内 α 合成（前缀积）----
    # 两次稳定排序实现精确两级排序（torch 2.11 无 lexsort）：
    # 先按 depth 升序（稳定），再按 px 升序（稳定，组内保持 depth 序）。
    order_d = torch.argsort(depth_s_all, stable=True)
    order = order_d[torch.argsort(px_s_all[order_d], stable=True)]
    px_s = px_s_all[order]
    alpha_s = alpha_s_all[order]
    feat_s = feat_s_all[order]

    seg_start = torch.ones_like(px_s, dtype=torch.bool)
    seg_start[1:] = px_s[1:] != px_s[:-1]

    # ⚠️ T 的段内独占前缀必须在 **fp64** 下用 log 和差计算：
    # fp32 log 版（全局 cumsum ~−3e4 量级相减）抵消误差 ~0.002 绝对 → T 接近 1 时
    # 相对误差 2%；fp32 乘积版（全局 cumprod）长序列下溢为 0 → 0/0=NaN（互验暴露）。
    # fp64 抵消误差 ~1e-11，exp 后转回 fp32，与 kernel 逐条顺序乘积差 ~1e-6 相对。
    log1m = torch.log1p(-alpha_s.double())
    cum = torch.cumsum(log1m, 0)
    excl = cum - log1m                                    # 全局逐项前缀
    seg_id = torch.cumsum(seg_start.long(), 0) - 1        # 每条目所属段（0 起）
    seg_tot = torch.zeros(int(seg_id.max()) + 1, dtype=torch.float64, device=device)
    seg_tot = seg_tot.index_add(0, seg_id, log1m)         # 每段 Σ log(1−α)
    seg_cum = torch.cumsum(seg_tot, 0) - seg_tot          # 段级独占前缀（前面段的 Σ）
    logT = excl - seg_cum[seg_id]                         # 段内独占前缀 = Π_{j<i}(1−α_j)
    T = torch.exp(logT).float()
    w = alpha_s * T

    # kernel 早停语义（互验校准）：C 累加前检查 test_T = T·(1−α) < 1e-4 →
    # 该条目被跳过且像素迭代终止（其贡献 α·T·f 可达 ~0.01·f，必须复刻）。
    # T·(1−α) 为**段内** inclusive 前缀，段内单调递减 ⇒ 阈值选出早停点及其后全部条目。
    keep = (T * (1.0 - alpha_s)) >= 1e-4

    n_px_total = int(px_s.max()) + 1
    F_px = torch.zeros(n_px_total, D, device=device)
    F_px = F_px.index_add(0, px_s[keep], w[keep].unsqueeze(1) * feat_s[keep])

    logT_px = torch.zeros(n_px_total, dtype=torch.float64, device=device)
    logT_px = logT_px.index_add(0, px_s[keep], log1m[keep])   # 只计保留条目
    T_final = torch.exp(logT_px).float()                  # Π(1−α) 保留条目
    if bg is not None:
        F_px = F_px + T_final.unsqueeze(1) * bg.to(device).float()

    img = torch.zeros(D, H * W, device=device)
    px_unique = torch.unique(px_s)                        # 排序去重
    img = img.index_copy(1, px_unique, F_px[px_unique].T)
    img = img.view(D, H, W)
    if debug_entries:
        # 调试（仅诊断用，慢）：返回 (img, 排序后条目 gid, px, alpha, T, keep)
        return img, torch.cat(gid_all)[order], px_s, alpha_s, T, keep
    return img
