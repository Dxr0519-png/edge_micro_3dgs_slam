"""§5 Tracking：固定高斯，只优化相机位姿（se(3) 左扰动）。

文档 §5 的 track() 实现：
- 位姿参数化为 6 维扰动 delta（se(3)），左扰动 `T_cur = se3_exp(torch.cat([d_rot, d_tr])) @ T`；
- loss = 0.8*l1(rgb) + 0.2*(1-ssim(rgb)) + 1.0*l1(depth)，depth>0 处 mask；
- Adam 优化 delta，返回最优位姿。
"""
from __future__ import annotations

import numpy as np
import torch

from ..camera import SyncedFrame
from ..gaussian.frustum import frustum_visible
from ..gaussian.model import GaussianModel
from ..gaussian.render import render, render_prepare
from ..gaussian.ssim import calc_ssim
from ..utils.frame_utils import downsample_frame
from ..utils.se3 import invert_pose, se3_exp, se3_log


def _to_cuda(frame: SyncedFrame):
    rgb_t = torch.from_numpy(frame.rgb.astype(np.float32) / 255.0).cuda().permute(2, 0, 1).contiguous()
    depth_t = torch.from_numpy(frame.depth.astype(np.float32)).cuda().unsqueeze(0).contiguous()
    K = torch.as_tensor(frame.K, dtype=torch.float32)
    return rgb_t, depth_t, K


def track(frame: SyncedFrame, gaussians: GaussianModel, T_wc_init: np.ndarray,
          iters: int = 8, lr: float = 1e-2, use_ssim: bool = True,
          depth_weight: float = 1.0, sil_thres: float = 0.5,
          rot_lr_ratio: float = 0.2, cull: bool = False,
          res_schedule: list | None = None, depth_every: int = 1,
          early_stop: bool = False, early_stop_patience: int = 2,
          early_stop_min_iters: int = 2, early_stop_tol: float = 1e-4,
          adaptive_max: int | None = None,
          fail_rot_deg: float = 45.0, fail_trans_m: float = 1.0,
          iters_log: list | None = None) -> torch.Tensor:
    """追踪一帧的相机位姿（世界→相机）。

    参数:
        frame:      SyncedFrame
        gaussians:  已建图的高斯模型（本函数内冻结，无梯度）
        T_wc_init:  (4,4) 初始位姿（上一帧结果 / 匀速外推）
        iters:      优化迭代数（res_schedule 给定时为单阶段默认值，被 schedule 覆盖）
        lr:         Adam 学习率（平移分量；旋转分量 = lr × rot_lr_ratio）
        sil_thres:  silhouette 阈值——只对模型已覆盖（sil > 阈值）的像素计入 loss。
                    模型未覆盖区域（新视角）的 GT depth 无法被当前模型解释，
                    不加 mask 会把位姿推向错误位置（SplaTAM 同款机制）。
        rot_lr_ratio: 旋转/平移学习率比。旋转梯度量级远大于平移（投影几何），
                    统一 lr 会过冲振荡；SplaTAM 用 4e-4/2e-3 = 0.2。
        res_schedule: Phase 3 性能档——`list[(W, H, n_iter)]` 分阶段分辨率（粗到细，
                    低分辨率粗对齐 → 高分辨率精修）。None = 单阶段（frame 原分辨率
                    × iters，即旧行为）。Adam delta/state 跨阶段共享；lr 衰减按
                    **全局总迭代数**（多阶段下按单阶段边界会过早衰减）；
                    best 回退在**阶段内**独立（跨分辨率 loss 数值不可比）。
        depth_every: Phase 3 性能档——每 N 迭代渲染 depth/sil 一趟并计入 loss；
                    中间迭代只渲染 RGB（省 ~1/2 光栅化前向+反传），sil 掩码缓存
                    复用（位姿步长小，掩码 1~2 迭代内基本不变）。1 = 每迭代都有
                    depth（旧行为）。
        early_stop: Phase 3 性能档——loss 相对下降率 < tol 连续 patience 次则
                    提前停止（至少 min_iters 迭代；轻运动帧 1~2 迭代即收敛）。
        adaptive_max: Phase 3 性能档——**自适应迭代上限**：基础 iters 迭代后，
                    若公共 loss 相对历史最优无改善（< tol）则停止；难帧（未收敛）
                    自动扩展迭代直到收敛或 adaptive_max（防长序列误差累积发散：
                    实测 3 迭代在 Replica 200 帧长序列 ATE 81.6cm vs 100 帧
                    12.3cm，难帧补迭代后误差不累积）。None 关闭（旧行为）。

    返回:
        T_wc (4,4) cuda tensor（世界→相机）

    best 回退口径：阶段内**所有迭代都有的 loss 项**——depth_every=1 时含 depth
    （与旧实现逐位一致）；depth_every>1 时只用 RGB+SSIM 公共部分（跳过 depth 的
    迭代 loss 数值尺度小，直接比较会误判"更优"，docs/03 §9 实测坑）。
    """
    stages = res_schedule if res_schedule is not None \
        else [(frame.rgb.shape[1], frame.rgb.shape[0], iters)]
    total_iters = sum(n for _, _, n in stages)
    T = torch.as_tensor(np.asarray(T_wc_init, dtype=np.float32)).cuda().float()
    d_rot = torch.zeros(3, device="cuda", requires_grad=True)
    d_tr = torch.zeros(3, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [d_rot], "lr": lr * rot_lr_ratio},
        {"params": [d_tr], "lr": lr},
    ])

    best = None
    global_i = 0
    steps = 0                     # 实际执行迭代数（2026-09-02 iters_log 用）
    for W, H, n_iter in stages:
        fd = downsample_frame(frame, W, H) if res_schedule is not None else frame
        rgb_t, depth_t, K = _to_cuda(fd)
        depth_mask = (depth_t > 0).detach()
        # 阶段内 best 口径：depth_every=1 → 每迭代都有 depth，用全 loss（旧行为）；
        # depth_every>1 → 跳过迭代无 depth 项，统一用 RGB+SSIM 公共部分
        use_full_loss = (depth_every == 1)
        sil_mask = None                 # depth 趟跳过迭代的掩码缓存
        prev_loss = None
        stagnant = 0
        best_common = torch.tensor(float("inf"), device="cuda")   # 公共 loss 历史最优
        stage_track = []                # [(measure, pose)] GPU 收集，阶段末一次 argmin
        prep = render_prepare(gaussians, gaussians_grad=False)  # Phase 4 迭代间复用
        j = 0
        while j < n_iter or (adaptive_max is not None and j < adaptive_max):
            # lr 衰减：最后 1/3 全局迭代每步 ×0.85，突破 Adam 最小步长限制
            if global_i >= total_iters * 2 // 3:
                for g in opt.param_groups:
                    g["lr"] = g["lr"] * 0.85
            T_cur = se3_exp(torch.cat([d_rot, d_tr])) @ T    # 左扰动
            # Phase 3 §4：cull 时每迭代按当前位姿重算视锥掩码（高斯本就 detach，
            # mask 只压缩光栅化工作量，不影响梯度结构）
            gmask = None
            if cull:
                gmask = frustum_visible(gaussians.means3D.detach(), T_cur.detach(), K,
                                        H, W, scales=gaussians.scales().detach())
            need_depth = (j % depth_every == 0)
            im, depth, sil, _, _ = render(gaussians, T_cur, K, W, H,
                                          gaussians_grad=False, camera_grad=True,
                                          mask=gmask, needs_depth=need_depth, prep=prep)
            if need_depth:
                # 模型覆盖掩码：silhouette > 阈值 且 GT 深度有效
                sil_mask = (depth_mask & (sil > sil_thres).unsqueeze(0)).detach()
            mask = sil_mask if sil_mask is not None else depth_mask
            # RGB 损失（mean：数值稳定；SplaTAM 用 sum 但 Adam 下等价）
            l_rgb = torch.abs(im - rgb_t)[mask.expand_as(im)].mean() \
                if mask.any() else torch.abs(im - rgb_t).mean()
            common = 0.8 * l_rgb
            if use_ssim:
                common = common + 0.2 * (1.0 - calc_ssim(im.unsqueeze(0), rgb_t.unsqueeze(0)))
            loss = common
            # 深度损失（mask 掉无效深度与未覆盖区域；只在 depth 趟计入）
            if need_depth and mask.any():
                loss = loss + depth_weight * torch.abs(depth - depth_t)[mask].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            with torch.no_grad():
                # Phase 4：best 跟踪 GPU 侧收集，阶段末一次 argmin（原每迭代
                # .item() 同步 2 次；loss 数值可比，argmin 与运行最小等价）
                measure = loss if use_full_loss else common
                stage_track.append(
                    (measure.detach().clone(),
                     (se3_exp(torch.cat([d_rot, d_tr])) @ T).detach().clone()))
                # 控制流（adaptive/early stop）：GPU 比较，每迭代至多 1 次同步
                should_break = False
                if j >= n_iter:
                    # 自适应扩展区：公共 loss 相对历史最优无改善 → 停
                    should_break = bool(
                        (common >= best_common * (1 - early_stop_tol)).item())
                    best_common = torch.minimum(best_common, common.detach())
                if not should_break and early_stop and j >= early_stop_min_iters \
                        and prev_loss is not None:
                    # early stop：相对下降率 < tol 连续 patience 次（同阶段口径）
                    rel = float(((prev_loss - loss).abs() / prev_loss).item())
                    if rel < early_stop_tol:
                        stagnant += 1
                        if stagnant >= early_stop_patience:
                            should_break = True
                    else:
                        stagnant = 0
                prev_loss = loss.detach()
            if should_break:
                break
            j += 1
            global_i += 1
        if stage_track:
            measures = torch.stack([m for m, _ in stage_track])
            best_i = int(torch.argmin(measures).item())      # 阶段末单次同步取 best
            best = stage_track[best_i][1]
    with torch.no_grad():
        if best is None:
            if iters_log is not None:
                iters_log.append(steps)
            return (se3_exp(torch.cat([d_rot, d_tr])) @ T).detach()
        # 失败检测：估计与初值差异过大说明优化落入远距离局部极小（常见于
        # silhouette mask 太松被"翻转视角"exploit），视为跟踪失败并回退初值
        # （标准 SLAM 鲁棒机制）。
        # Phase 4 收紧：ICP 精修初值下修正量正常 <2°，>8° 几乎必错（实测
        # Replica t=68 稀疏模型新视角假极小 +9° 旋转，45°/1m 旧阈值放行导致
        # 全序列发散）；默认保持旧行为，fps10 系配置收紧。
        d = se3_log(best @ invert_pose(T))
        if not bool(torch.isfinite(d).all().item()) \
                or d[:3].norm() > float(np.deg2rad(fail_rot_deg)) \
                or d[3:].norm() > fail_trans_m:
            if iters_log is not None:
                iters_log.append(steps)
            return T.detach()
    if iters_log is not None:
        iters_log.append(steps)
    return best
