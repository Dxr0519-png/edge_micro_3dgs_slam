"""§6 Mapping：关键帧建图（加高斯 → 属性优化 → 剪枝）。

文档 §6 的 map_keyframe() 三步：
1) 反投影新增高斯：silhouette < 阈值（无高斯覆盖）的像素处 add_gaussians()；
2) 属性优化：xyz/rot/scale/opacity/color，loss = photometric + ssim + depth
   + λ_iso·‖scale_i − mean(scale)‖₂（各向同性约束，防尺度发散）；
3) 剪枝：opacity < 0.005 或 scale 超 scene_radius 者 prune()。

学习率沿用 SplaTAM mapping 配置（means3D 1e-4 / rgb 2.5e-3 / rot 1e-3 /
opacity 5e-2 / scale 1e-3）。
"""
from __future__ import annotations

import numpy as np
import torch

from ..camera import SyncedFrame, backproject, backproject_torch
from ..gaussian.frustum import frustum_visible
from ..gaussian.model import GaussianModel, PARAM_KEYS, coverage_mask, enforce_capacity
from ..gaussian.render import render
from ..gaussian.ssim import calc_ssim
from ..utils.frame_utils import downsample_frame
from .tracking import _to_cuda

# SplaTAM mapping 学习率
_LRS = {"means3D": 1e-4, "rgb_colors": 2.5e-3, "unnorm_rotations": 1e-3,
        "logit_opacities": 5e-2, "log_scales": 1e-3}
_LAMBDA_ISO = 0.001     # 各向同性约束权重（0.01 会把近/远不同尺度的高斯强制统一，
                        # 远处表面覆盖稀疏 → 渲染 depth 系统性偏浅；0.001 保留约束意图）
_SIL_THRES = 0.5        # 加高斯阈值：silhouette 低于此的像素补高斯


def _add_new_gaussians(frame: SyncedFrame, model: GaussianModel, T_wc: np.ndarray,
                       sil: torch.Tensor, max_new: int = 20000,
                       density_r: float | None = None,
                       seed_opacity: float = 0.5,
                       seed_scale_factor: float = 2.0) -> tuple[int, int]:
    """在 silhouette < 阈值（无高斯覆盖）且深度有效的像素处反投影新增高斯。

    返回 (added, density_blocked)：density_r 给定时（Phase 3 §2 密度判据），
    半径 r 内已有近邻高斯的候选点不新增（防同一区域反复堆高斯）。

    Phase 4：全 GPU 化——原实现 sil/coverage 两次 .cpu().numpy() 往返 + numpy
    全分辨率反投影（每关键帧 10-20ms 且 2 处 CPU-GPU 同步）；现 backproject_torch
    + GPU 掩码索引，候选点全程 GPU（add_gaussians 的 create_from_points 接受
    cuda 张量），仅保留 1 处 valid.any() 同步。
    """
    K_t = torch.as_tensor(np.asarray(frame.K, dtype=np.float32))
    depth_t = torch.from_numpy(frame.depth.astype(np.float32)).cuda().contiguous()
    no_cover = sil.detach() < _SIL_THRES
    valid = (depth_t > 0) & no_cover
    if not bool(valid.any().item()):
        return 0, 0
    pts_cam, _ = backproject_torch(depth_t, K_t)          # (H,W,3) 相机系
    T_wc_t = torch.as_tensor(np.asarray(T_wc, dtype=np.float32)).cuda().float()
    # ⚠️ w2c 约定：p_world = Rᵀ(p_cam − t)（numpy 版 backproject 同口径）。
    # 直接用 R 会种错世界位置（实测 Replica baseline ATE 6.5→82cm、高斯数翻倍）
    pts_world = ((pts_cam[valid] - T_wc_t[:3, 3]) @ T_wc_t[:3, :3])     # (N,3)
    rgb_t = torch.from_numpy(frame.rgb.astype(np.float32) / 255.0).cuda()
    new_colors = rgb_t[valid]
    depth_z = depth_t[valid]
    n_blocked = 0
    if density_r is not None and pts_world.shape[0] > 0:
        covered = coverage_mask(pts_world, model.means3D.detach(), density_r)
        n_blocked = int(covered.sum().item())
        keep = ~covered
        if not bool(keep.any().item()):
            return 0, n_blocked
        pts_world, new_colors, depth_z = pts_world[keep], new_colors[keep], depth_z[keep]
    if pts_world.shape[0] > max_new:
        keep = torch.randperm(pts_world.shape[0], device="cuda")[:max_new]
        pts_world, new_colors, depth_z = pts_world[keep], new_colors[keep], depth_z[keep]
    focal = (frame.K[0, 0] + frame.K[1, 1]) / 2.0
    # 2026-09-02：seed_scale_factor 可调小播种尺度（默认 2.0 与 init 一致；
    # 更小尺度 → 更少重叠 → 透叠轻）
    scales = seed_scale_factor * depth_z / focal
    return model.add_gaussians(pts_world, new_colors, scales=scales,
                               opacity=seed_opacity), n_blocked


def _iso_loss(model: GaussianModel) -> torch.Tensor:
    """各向同性约束：log_scale 偏离均值越远惩罚越大。"""
    ls = model.params["log_scales"]
    return torch.mean((ls - ls.mean()) ** 2)


def map_keyframe(frame: SyncedFrame, gaussians: GaussianModel, T_wc: np.ndarray,
                 iters: int = 50, add_new: bool = True, prune: bool = True,
                 density_r: float | None = None, capacity_max: int | None = None,
                 cull: bool = False, window: list | None = None,
                 max_new: int = 20000, opt_W: int | None = None,
                 opt_H: int | None = None, window_rotate: bool = False,
                 rotate_n: int = 2, map_tier: str = "full",
                 seed_W: int | None = None, seed_H: int | None = None,
                 depth_every: int = 1, seed_opacity: float = 0.5,
                 seed_scale_factor: float = 2.0,
                 prune_opacity: float = 0.005) -> dict:
    """对一帧关键帧执行建图优化（原地修改 gaussians）。

    参数:
        frame:     SyncedFrame
        gaussians: 高斯模型（原地更新）
        T_wc:      (4,4) 该帧的（已追踪出的）世界→相机位姿
        iters:     属性优化迭代数
        density_r: Phase 3 §2 密度判据半径（米）；None 关闭。候选点在 r 内已有
                   近邻高斯则不新增
        capacity_max: Phase 3 §2 容量上限；**只在优化循环结束后**执行（此时优化器
                   已释放，无 Adam 状态与索引错位问题——时序约束，勿挪进迭代内）
        cull:      Phase 3 §4 视锥剔除——优化只对可见高斯反传，不可见冻结
        window:    Phase 3 §5 关键帧滑动窗口 `list[(SyncedFrame, T_wc)]`（含当前帧）。
                   窗口内所有帧参与每轮 loss（旧关键帧出窗即冻结）；add_new/prune
                   只针对最新帧。None 时单帧（Phase 2 行为）
        opt_W/H:   Phase 3 性能档——优化循环内窗口帧的**降采样渲染分辨率**（None =
                   原生）。光栅化与像素数成正比，320×240 档省 ~4 倍。**add_new 的
                   silhouette 渲染保持原生分辨率**（高斯播种质量不降）。
        window_rotate: Phase 3 性能档——每迭代只轮转渲染窗口 rotate_n 帧（确定性
                   轮序）而非全窗口（window=5 时渲染量 5→2）。每帧单关键帧内被
                   访问 ≥3 次（iters=8×2/5≈3.2）时质量与全窗口近似（探针验证）。
                   与 opt_W/H 组合：每关键帧渲染量 ≈ iters×rotate_n 次低分辨率渲染。
        seed_W/H:  2026-09-02 队列压力降档——播种（add_new 的 silhouette 渲染 +
                   反投影）在降采样分辨率执行（默认 None = 原生 640×480，@200k
                   量级 ~150-250ms/关键帧）。低分辨率播种会漏薄结构，仅在队列积压
                   档使用，空闲档恢复原生。
        depth_every: 2026-09-02 建图迭代 depth 隔趟——每 N 迭代只渲染一次
                   depth/silhouette 第二趟（loss 的 depth 项缺席趟跳过，sil 只在
                   播种渲染，不依赖迭代趟）；省 ~1/3 光栅化前向+反传。

    返回:
        统计 dict：新增/密度阻挡/剪枝/容量淘汰数、窗口大小、最终 loss
    """
    # 窗口帧预处理（每帧固定 T，mask/图像只转换一次，避免每迭代重复）；
    # opt_W/H 给定时一次性降采样（Phase 3 §3 语义：K 逐轴缩放已修复）
    win = [(frame, T_wc)] if window is None else list(window)
    prep = []
    for f, T_f in win:
        f_opt = downsample_frame(f, opt_W, opt_H) if (opt_W and opt_H) else f
        rgb_t, depth_t, K_f = _to_cuda(f_opt)
        H, W = depth_t.shape[1:3]
        T_f_t = torch.as_tensor(np.asarray(T_f, dtype=np.float32)).cuda().float()
        vis_f = None
        if cull:
            vis_f = frustum_visible(gaussians.means3D.detach(), T_f_t.detach(), K_f,
                                    H, W, scales=gaussians.scales().detach())
        prep.append((rgb_t, depth_t, K_f, T_f_t, (depth_t > 0).detach(), vis_f))
    # 播种帧：默认原生分辨率（薄结构不漏播）；seed_W/H 给定时（队列压力档）
    # 降采样播种（K 随 downsample_frame 缩放，sil/反投影同分辨率自洽）
    seed_frame = downsample_frame(frame, seed_W, seed_H) if (seed_W and seed_H) else frame
    sil_frame_T = torch.as_tensor(np.asarray(T_wc, dtype=np.float32)).cuda().float()
    sil_frame_K = torch.as_tensor(np.asarray(seed_frame.K, dtype=np.float32))
    sil_H, sil_W = seed_frame.depth.shape

    # 1) 反投影新增高斯（用当前模型渲染 silhouette 找无覆盖像素；只对最新帧，
    #    默认原生分辨率——低分辨率下薄结构像素消失会漏播高斯，播种质量不降）
    added = n_blocked = 0
    at_capacity = bool(
        capacity_max is not None and gaussians.num_gaussians >= capacity_max)
    if add_new and not at_capacity:
        # 2026-09-02：容量已满时跳过播种——silhouette 必全覆盖（无位置可加），
        # 原生分辨率全模型渲染纯属浪费（@200k 量级 150-250ms/关键帧，实测是
        # 建图每 KF 成本最大单项；200k 封顶的稳态下每个 KF 都在白付这笔钱）
        with torch.no_grad():
            _, _, sil, _, _ = render(gaussians, sil_frame_T, sil_frame_K, sil_W,
                                     sil_H, gaussians_grad=False, camera_grad=False)
        added, n_blocked = _add_new_gaussians(seed_frame, gaussians, T_wc, sil,
                                              max_new=max_new, density_r=density_r,
                                              seed_opacity=seed_opacity,
                                              seed_scale_factor=seed_scale_factor)
        # 新增后窗口帧的 vis 掩码可能变化（新高斯未必可见）——重建
        if cull:
            prep = []
            for f, T_f in win:
                f_opt = downsample_frame(f, opt_W, opt_H) if (opt_W and opt_H) else f
                rgb_t, depth_t, K_f = _to_cuda(f_opt)
                H, W = depth_t.shape[1:3]
                T_f_t = torch.as_tensor(np.asarray(T_f, dtype=np.float32)).cuda().float()
                vis_f = frustum_visible(gaussians.means3D.detach(), T_f_t.detach(),
                                        K_f, H, W, scales=gaussians.scales().detach())
                prep.append((rgb_t, depth_t, K_f, T_f_t, (depth_t > 0).detach(), vis_f))

    # 2) 属性优化（cull 时只对每帧可见高斯反传）
    # Phase 3 §3：存储层 half 时，优化期间转回 float32——Adam 的 state（exp_avg/
    # exp_avg_sq）是参数同 dtype，half 下 fp32 量级梯度平方直接上溢 → 参数 NaN
    # （实测）。存储层在优化间隙保持 half 省显存，计算层恒 float，互不冲突。
    # Phase 4 map_tier="addonly"：跳过属性优化（只加高斯+剪枝）——队列积压时
    # 的降档路径，map 每关键帧 ~150ms → ~60-90ms（PSNR 折损由队列空闲补 polish
    # 与多视角播种缓解，探针 --map-tier 验证）。
    was_half = gaussians.is_half_storage
    scene_radius = gaussians.variables.get("scene_radius", 5.0)
    loss = None
    if map_tier != "addonly":
        if was_half:
            gaussians.float_storage()
        opt = torch.optim.Adam(
            [{"params": [gaussians.params[k]], "lr": _LRS[k]} for k in PARAM_KEYS])
        n_win = len(prep)
        for k in range(iters):
            total = None
            need_d = (depth_every <= 1) or (k % depth_every == 0)   # depth 隔趟
            if window_rotate and n_win > rotate_n:
                # 确定性轮序：迭代 k 选 [k*rotate_n, k*rotate_n+rotate_n) 环绕的帧集
                sel = [(k * rotate_n + j) % n_win for j in range(rotate_n)]
                frame_iter = [prep[i] for i in sel]
            else:
                frame_iter = prep
            for rgb_t, depth_t, K_f, T_f_t, depth_mask, vis_f in frame_iter:
                H, W = depth_t.shape[1:3]
                im, depth, sil, _, _ = render(gaussians, T_f_t, K_f, W, H,
                                              gaussians_grad=True, camera_grad=False,
                                              mask=vis_f, needs_depth=need_d)
                l_rgb = 0.8 * torch.abs(im - rgb_t).mean()
                l_ssim = 0.2 * (1.0 - calc_ssim(im.unsqueeze(0), rgb_t.unsqueeze(0)))
                l = 0.5 * (l_rgb + l_ssim) + _LAMBDA_ISO * _iso_loss(gaussians)
                if need_d:
                    l = l + 1.0 * (torch.abs(depth - depth_t)[depth_mask].mean()
                                   if depth_mask.any() else torch.abs(depth - depth_t).mean())
                total = l if total is None else total + l
            loss = total / len(frame_iter)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if was_half:
            gaussians.half_storage()           # 优化结束，存储层回到 half（§3）

    # 3) 剪枝（opacity < 0.005 或 scale 超 scene_radius 10%）→ 容量淘汰
    pruned = capped = 0
    if prune:
        n_before = gaussians.num_gaussians
        gaussians.prune(opacity_threshold=prune_opacity, big_scale=0.1 * scene_radius)
        pruned = n_before - gaussians.num_gaussians
    if capacity_max is not None:
        n_before = gaussians.num_gaussians
        enforce_capacity(gaussians, capacity_max)
        capped = n_before - gaussians.num_gaussians

    return {"added": added, "density_blocked": n_blocked, "pruned": pruned,
            "capped": capped, "window_size": len(prep),
            "num_gaussians": gaussians.num_gaussians,
            "at_capacity": at_capacity,
            "loss": loss.item() if loss is not None else None}
