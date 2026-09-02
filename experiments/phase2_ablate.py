#!/usr/bin/env python3
"""Phase 2 消融抽查：验证文档2§10.9 的 iso_loss 权重声明。

文档声称（§10.9）：λ_iso=0.01 → 远处表面覆盖稀疏、渲染 depth 系统性偏浅
（实测远处误差 110cm）；λ_iso=0.001 且 σ×2 → 首帧 depth RMSE 89→20cm。

本脚本在合成序列上以 λ_iso=0.01 重建地图，统计远处（GT depth > 2.5m）
像素的渲染 depth 误差，与文档声称的 110cm 对比（±30% 内算吻合）。
不修改被测源码：mapping._LAMBDA_ISO 是模块级全局，monkey-patch 即可。

用法：
    python3 experiments/phase2_ablate.py [--lambda-iso 0.01] [--frames 240]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import numpy as np
import torch

import edge_3dgs_slam.slam.mapping as mapping_mod
from edge_3dgs_slam.gaussian import render
from edge_3dgs_slam.slam import build_map

DATA = Path("data/outputs/phase2/synth_scene.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-iso", type=float, default=0.01)
    ap.add_argument("--frames", type=int, default=240)
    args = ap.parse_args()

    print(f"λ_iso 消融: {mapping_mod._LAMBDA_ISO} → {args.lambda_iso}（monkey-patch 模块级全局）")
    mapping_mod._LAMBDA_ISO = args.lambda_iso

    d = np.load(DATA)
    rgb, depth, K, poses = d["rgb"], d["depth"], d["K"], d["poses"]
    n = min(args.frames, len(rgb))
    rgb, depth, poses = rgb[:n], depth[:n], poses[:n]
    print(f"[数据] {n} 帧 {rgb.shape[1]}x{rgb.shape[2]}")

    model = build_map(rgb, depth, K, poses, keyframe_every=5, map_iters=60, verbose=False)
    print(f"[建图] {model.num_gaussians} 高斯")

    # 全序列采样渲染，统计远处（GT depth > 2.5m）像素的渲染 depth 误差
    gt_far, est_far = [], []
    with torch.no_grad():
        for t in range(0, n, 5):
            dep_t = torch.from_numpy(depth[t]).cuda()
            gt_t = torch.from_numpy(depth[t]).cuda()
            _, dep_r, _, _, _ = render(model, poses[t], K, rgb.shape[2], rgb.shape[1],
                                       gaussians_grad=False, camera_grad=False)
            m = gt_t > 2.5
            if m.any():
                gt_far.append(gt_t[m]); est_far.append(dep_r.squeeze(0)[m])
    gt_c = torch.cat(gt_far); est_c = torch.cat(est_far)
    err_far = torch.abs(est_c - gt_c)
    print(f"[远处] GT depth>2.5m 像素: {gt_c.numel()}, "
          f"渲染 depth 误差 mean={err_far.mean().item()*100:.1f} cm, "
          f"median={err_far.median().item()*100:.1f} cm, "
          f"渲染系统性偏浅量={((gt_c - est_c).mean()).item()*100:.1f} cm")
    print(f"[对照] 文档§10.9 声称 λ=0.01 → 远处误差 110cm（±30% 内算吻合）")
    ok = 0.7 * 110 <= err_far.mean().item() * 100 <= 1.3 * 110
    print(f"[判定] {'吻合 ✅' if ok else '不吻合 ❌（如实记录，该数字来源版本不可考）'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
