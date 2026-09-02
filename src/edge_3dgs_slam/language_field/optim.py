"""Phase 6 §3 语言特征蒸馏：掩码级监督优化高斯潜变量（几何冻结）。

流程（docs/06 §3）：
    F_2d = render_precomp(colors_precomp=features)   # (3,H,W)，D=3 快速路径
    V_2d = ae.dec(F_2d)                              # 解码回 512 维
    掩码级损失：对每个 MobileSAM 掩码区域，V_2d 均值与该掩码 clip_vec 对齐
    L = Σ_m [ L1(V_mean_m, v_gt_m) + λ_cos·(1 − cos(V_mean_m, v_gt_m)) ] / |M|
    只优化 gaussians.params["features"]（几何/颜色冻结——坑 2：防破坏已收敛几何）。

- 渲染走 inria kernel 快速路径（render_precomp，~10-20ms/帧）——**绝不走慢速 splat**；
- 解码只算掩码 union 像素（~全图 10-20%），分块防显存尖峰；
- AE 冻结（eval + requires_grad_(False)）——坑 3：AE 收敛后再蒸馏，防解码漂移。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..gaussian import FEATURE_KEY, GaussianModel, freeze_geometry, render_precomp
from ..gaussian.render import render


@dataclass
class FrameSupervision:
    """单帧掩码级监督数据（几何冻结蒸馏的输入）。"""
    w2c: np.ndarray            # (4,4) 世界→相机
    K: np.ndarray              # (3,3) 内参（与渲染同分辨率）
    H: int
    W: int
    segs: list[np.ndarray]     # 每掩码 (H,W) bool segmentation
    clip_vecs: list[np.ndarray]  # 每掩码 (512,) float32 L2 归一化
    weights: list[float] = field(default_factory=list)   # 掩码权重（空=等权）
    rgb: np.ndarray | None = None   # (H,W,3) uint8（预对齐门 PSNR 用，可选）


def distill_features(gaussians: GaussianModel, ae: torch.nn.Module,
                     frames: list[FrameSupervision],
                     iters: int = 200, lr: float = 1e-3, lambda_cos: float = 0.5,
                     device: str = "cuda", dec_chunk: int = 50_000,
                     verbose: bool = True) -> dict:
    """掩码级监督蒸馏：只优化 gaussians.feature（几何冻结）。

    参数:
        gaussians: 含 FEATURE_KEY 的 GaussianModel（蒸馏前先 add_feature_dim）
        ae: LangAE（已 eval + requires_grad_(False)；dec 把 D 维潜变量解码回 512）
        frames: 每帧的掩码级监督（w2c/K/H/W/segs/clip_vecs）
        iters: 迭代轮数（全批次）
        lr: Adam 学习率（只对 features）
        lambda_cos: 余弦损失权重
    返回: {'loss_final', 'cos_train_mean', 'iters', 'changed_norm'}
    """
    assert FEATURE_KEY in gaussians.params, "先 add_feature_dim 再蒸馏"
    d = gaussians.params[FEATURE_KEY].shape[1]
    assert d == 3, f"快速路径要求 D=3（当前 {d}）；D>3 需慢速 splat（未实现蒸馏路径）"

    freeze_geometry(gaussians, freeze=True)          # 坑 2：几何/颜色冻结
    ae = ae.to(device).eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam([gaussians.params[FEATURE_KEY]], lr=lr)

    # 预计算每帧：掩码像素（union）+ 归约索引
    prep = []
    for fr in frames:
        H, W = fr.H, fr.W
        union = np.zeros(H * W, dtype=bool)
        for seg in fr.segs:
            union[seg.reshape(-1)] = True
        uidx = np.where(union)[0]
        # 每掩码在 union 内的像素索引（用于段归约）
        seg_in_union = []
        for seg in fr.segs:
            mask_u = seg.reshape(-1)[uidx]
            seg_in_union.append(np.where(mask_u)[0])
        prep.append({
            "w2c": torch.as_tensor(fr.w2c, dtype=torch.float32, device=device),
            "K": np.asarray(fr.K, np.float64),
            "H": H, "W": W,
            "uidx": torch.as_tensor(uidx, dtype=torch.long, device=device),
            "seg_in_union": seg_in_union,
            "v_gt": torch.as_tensor(np.stack(fr.clip_vecs), dtype=torch.float32, device=device),
            "weights": torch.as_tensor(fr.weights or [1.0] * len(fr.segs),
                                       dtype=torch.float32, device=device),
        })

    loss_final = cos_mean = float("nan")
    for it in range(iters):
        opt.zero_grad()
        total = 0.0
        for p in prep:
            # F_2d (3,H,W)：inria kernel 快速路径（几何冻结 → gaussians_grad=False，
            # 梯度经 grad_colors_precomp 回流到 features）
            F2d = render_precomp(gaussians, p["w2c"], p["K"], p["W"], p["H"],
                                 colors_precomp=gaussians.params[FEATURE_KEY],
                                 gaussians_grad=False, camera_grad=False)
            F_u = F2d.permute(1, 2, 0).reshape(-1, 3)[p["uidx"]]     # (|U|,3)
            # 解码分块（防显存尖峰）；梯度经冻结的 dec 参数回流
            V_u = torch.cat([ae.dec(F_u[s:s + dec_chunk])
                             for s in range(0, F_u.shape[0], dec_chunk)], dim=0)
            V_u = F.normalize(V_u, dim=-1)
            # 掩码级：V_mean_m = V_u[seg_m].mean(0)
            for m, (siu, v_gt, wgt) in enumerate(
                    zip(p["seg_in_union"], p["v_gt"], p["weights"])):
                if len(siu) == 0:
                    continue
                Vm = V_u[siu].mean(0)
                total += wgt * ((Vm - v_gt).abs().mean()
                                + lambda_cos * (1.0 - F.cosine_similarity(Vm, v_gt, dim=-1)))
        loss = total / sum(len(fr.segs) for fr in frames)
        loss.backward()
        opt.step()
        loss_final = float(loss.item())
        if verbose and (it + 1) % 50 == 0 or it + 1 == iters:
            cos_mean = _cos_train(gaussians, ae, prep)
            print(f"  iters {it+1:4d} | loss {loss_final:.4f} | cos_train {cos_mean:.4f}")

    return {"loss_final": loss_final, "cos_train_mean": cos_mean,
            "iters": iters}


def _cos_train(gaussians: GaussianModel, ae: torch.nn.Module,
               prep: list[dict]) -> float:
    """训练帧掩码内 V_2d 均值 vs clip_vec 余弦（验收① 的训练帧口径）。"""
    with torch.inference_mode():
        cos_sum, n = 0.0, 0
        for p in prep:
            F2d = render_precomp(gaussians, p["w2c"], p["K"], p["W"], p["H"],
                                 colors_precomp=gaussians.params[FEATURE_KEY].detach(),
                                 gaussians_grad=False, camera_grad=False)
            F_u = F2d.permute(1, 2, 0).reshape(-1, 3)[p["uidx"]]
            V_u = F.normalize(ae.dec(F_u), dim=-1)
            for siu, v_gt in zip(p["seg_in_union"], p["v_gt"]):
                if len(siu) == 0:
                    continue
                Vm = V_u[siu].mean(0)
                cos_sum += float(F.cosine_similarity(Vm, v_gt, dim=-1).item())
                n += 1
        return cos_sum / max(n, 1)


def mask_cosine_eval(gaussians: GaussianModel, ae: torch.nn.Module,
                     frame: FrameSupervision) -> tuple[float, list[float]]:
    """单帧掩码内解码特征 vs clip_vec 余弦（验收①/留出帧口径）。"""
    with torch.inference_mode():
        F2d = render_precomp(gaussians,
                             torch.as_tensor(frame.w2c, dtype=torch.float32,
                                             device=gaussians.means3D.device),
                             np.asarray(frame.K, np.float64), frame.W, frame.H,
                             colors_precomp=gaussians.params[FEATURE_KEY].detach(),
                             gaussians_grad=False, camera_grad=False)
        Fp = F2d.permute(1, 2, 0)                       # (H,W,3)
        cos_list = []
        for seg, v_gt in zip(frame.segs, frame.clip_vecs):
            if not seg.any():
                continue
            Vm = ae.dec(Fp[seg].reshape(-1, 3)).mean(0)
            Vm = F.normalize(Vm, dim=-1)
            vg = F.normalize(torch.as_tensor(v_gt, dtype=torch.float32,
                                             device=Vm.device), dim=-1)
            cos_list.append(float(F.cosine_similarity(Vm, vg, dim=-1).item()))
        return float(np.mean(cos_list)) if cos_list else 0.0, cos_list


def prealign_check(gaussians: GaussianModel, frame: FrameSupervision,
                   min_psnr: float = 15.0) -> tuple[bool, float]:
    """预对齐门：渲染 frame 0 RGB vs 原图 PSNR（防位姿/模型错位白跑）。"""
    rgb_gt = frame.rgb if hasattr(frame, "rgb") else None
    if rgb_gt is None:
        return True, float("nan")
    im, _, _, _, _ = render(gaussians,
                            torch.as_tensor(frame.w2c, dtype=torch.float32,
                                            device=gaussians.means3D.device),
                            np.asarray(frame.K, np.float64), frame.W, frame.H,
                            needs_depth=False, gaussians_grad=False)
    gt = torch.as_tensor(rgb_gt, device=im.device).float().permute(2, 0, 1) / 255.0
    mse = ((im - gt) ** 2).mean().item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-8))
    return psnr > min_psnr, psnr
