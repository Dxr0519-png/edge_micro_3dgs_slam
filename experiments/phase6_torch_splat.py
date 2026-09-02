#!/usr/bin/env python3
"""Phase 6 §2 特征光栅化验证：纯 PyTorch 慢速 splat vs CUDA（inria kernel）。

验证对象: src/edge_3dgs_slam/gaussian/torch_feature_rasterizer.py
通过标准（docs/06 §7 验收④ + 坑 1）:
    1. 数值梯度：dL/df（feature）与 dL/d(logit_opacity) 中心差分（eps=1e-3）
       每元素 (abs < 1e-5 或 rel < 1e-3)（Phase 2 报告教训：小梯度分量放大 rel）
    2. 互验：慢速 splat vs render_precomp（inria kernel, D=3 即 CUDA 参考）
       max|diff| < 1e-3 且 mean|diff| < 1e-4
    3. D=16 自洽：慢速 splat 前 3 通道与 D=3 慢速 splat 逐元素相等（同一 α 路径）
数据: 合成小场景（500 高斯 @120x68）+ Replica office0 probe 前 10k 高斯 + GT w2c frame 0 @300x170
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from edge_3dgs_slam.dataset.replica import ReplicaSequence
from edge_3dgs_slam.gaussian import (GaussianModel, add_feature_dim, load_ply,
                                     rasterize_feature, render_precomp)

OUT = Path("data/outputs/phase6")
# Phase 2 世界系对齐地图（GT 位姿建图，frame 0-200 覆盖 100%、PSNR 26-30dB；
# 注意 phase3 的 probe_model_replica.pt 用单位位姿占位建图，与世界系不对齐，不可用于语义验证）
REPLICA_MAP = Path("data/outputs/phase2_replica/replica_map.ply")
REPLICA_ROOT = Path("third_party/SplaTAM/data/Replica")
EPS = 1e-3

torch.manual_seed(0)


def make_synthetic(n: int = 500) -> GaussianModel:
    """合成场景：相机前方随机位置高斯（世界系 = 相机系，w2c=I）。"""
    rng = np.random.RandomState(0)
    pts = np.stack([
        rng.uniform(-1.0, 1.0, n),          # x
        rng.uniform(-0.6, 0.6, n),          # y
        rng.uniform(1.5, 4.0, n),           # z（相机前）
    ], axis=-1).astype(np.float32)
    rgb = rng.rand(n, 3).astype(np.float32) * 0.8 + 0.1
    feat = rng.randn(n, 3).astype(np.float32)
    return GaussianModel.create_from_points(pts, rgb, features=feat, anisotropic=False)


def run_grad_test() -> bool:
    """判据 1：数值梯度（feature 与 opacity 两条路径）。"""
    print("=" * 62)
    print("§2.1 数值梯度（中心差分 eps=1e-3，合成 500 高斯 @120x68）")
    model = make_synthetic()
    K = np.array([[60.0, 0, 60.0], [0, 60.0, 34.0], [0, 0, 1]], np.float64)
    w2c = torch.eye(4, dtype=torch.float32, device="cuda")
    H, W = 68, 120

    F = rasterize_feature(model, w2c, K, H, W)
    # 取有贡献的输出像素子集（F 非零），L = 通道 0 求和
    nonzero = F[0].abs() > 1e-6
    idx = torch.nonzero(nonzero.view(-1))[:200].squeeze(1)
    if idx.numel() == 0:
        print("  ❌ 无输出像素有贡献（场景/内参配置问题）")
        return False

    def loss_fn():
        Ff = rasterize_feature(model, w2c, K, H, W)
        return Ff[0].view(-1)[idx].sum()

    L = loss_fn()
    g_feat, g_op = torch.autograd.grad(
        L, [model.params["features"], model.params["logit_opacities"]])

    feats = model.params["features"].detach()
    ops = model.params["logit_opacities"].detach()

    # 抽 30 个 feature 元素 + 10 个 opacity 元素做中心差分
    rng = np.random.RandomState(1)
    fi = rng.choice(feats.numel(), 30, replace=False)
    oi = rng.choice(ops.numel(), 10, replace=False)

    bad_f, bad_o = [], []
    for i in fi:
        p, q = divmod(int(i), feats.shape[1])
        v0 = feats[p, q].item()
        feats[p, q] = v0 + EPS
        Lp = loss_fn().item()
        feats[p, q] = v0 - EPS
        Lm = loss_fn().item()
        feats[p, q] = v0
        fd = (Lp - Lm) / (2 * EPS)
        g = g_feat[p, q].item()
        err = abs(fd - g)
        if not (err < 1e-5 or err / (abs(g) + 1e-12) < 1e-3):
            bad_f.append((int(i), g, fd, err))
    for i in oi:
        v0 = ops[i].item()
        ops[i] = v0 + EPS
        Lp = loss_fn().item()
        ops[i] = v0 - EPS
        Lm = loss_fn().item()
        ops[i] = v0
        fd = (Lp - Lm) / (2 * EPS)
        g = g_op[i].item()
        err = abs(fd - g)
        if not (err < 1e-5 or err / (abs(g) + 1e-12) < 1e-3):
            bad_o.append((int(i), g, fd, err))

    rel_f = [abs(g) for g in g_feat.flatten().tolist() if abs(g) > 1e-8]
    print(f"  L（通道 0 前 200 非零像素求和）: {L.item():.4f} | "
          f"feature 梯度非零元素 {len(rel_f)}")
    print(f"  feature 差分: 检查 30 元素，超差 {len(bad_f)} 个")
    for i, g, fd, err in bad_f[:3]:
        print(f"    idx={i} autograd={g:.3e} finite={fd:.3e} err={err:.3e}")
    print(f"  opacity 差分: 检查 10 元素，超差 {len(bad_o)} 个")
    for i, g, fd, err in bad_o[:3]:
        print(f"    idx={i} autograd={g:.3e} finite={fd:.3e} err={err:.3e}")
    ok = not bad_f and not bad_o
    print(f"  §2.1 {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def load_map_subset(n: int = 10_000) -> tuple[GaussianModel, np.ndarray]:
    """replica_map（世界系对齐，各向异性）前 n 个高斯 + Replica office0 frame 0 GT w2c。

    各向异性正好覆盖慢速 splat 的旋转路径（quat→rot + 相机系协方差）。"""
    model = load_ply(str(REPLICA_MAP))
    for k in model.params:
        model.params[k] = torch.nn.Parameter(model.params[k][:n].contiguous())
    seq = ReplicaSequence.from_dir(REPLICA_ROOT, "office0", max_frames=1)
    w2c = seq.poses_w2c[0].astype(np.float32)
    cam = seq.cam
    # 300x170 = 全分辨率 1200x680 的 0.25 缩放
    sc = 0.25
    K = np.array([[cam.fx * sc, 0, cam.cx * sc],
                  [0, cam.fy * sc, cam.cy * sc],
                  [0, 0, 1]], np.float64)
    return model, w2c, K, 170, 300


def run_mutual_test() -> bool:
    """判据 2：慢速 splat vs inria kernel（D=3）互验 < 1e-3。"""
    print("=" * 62)
    print("§2.2 慢速 splat vs inria kernel（D=3，replica_map 前 10k 高斯 @300x170）")
    model, w2c, K, H, W = load_map_subset(10_000)
    add_feature_dim(model, 3)
    with torch.no_grad():
        model.params["features"].normal_(0, 1)
    feat = model.params["features"].detach()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    F_slow = rasterize_feature(model, w2c, K, H, W, features=feat)
    t1.record()
    torch.cuda.synchronize()
    dt_slow = t0.elapsed_time(t1)

    w2c_t = torch.as_tensor(w2c, device="cuda")
    t0.record()
    F_cuda = render_precomp(model, w2c_t, K, W, H, colors_precomp=feat)   # (width, height)
    t1.record()
    torch.cuda.synchronize()
    dt_cuda = t0.elapsed_time(t1)

    diff = (F_slow - F_cuda).abs()
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    covered = float((F_cuda.abs() > 0).float().mean())
    print(f"  慢速 splat {dt_slow:.0f} ms | inria kernel {dt_cuda:.0f} ms | "
          f"非零像素覆盖 {covered*100:.1f}%")
    print(f"  max|diff| = {max_d:.3e} (< 1e-3: {max_d < 1e-3})")
    print(f"  mean|diff| = {mean_d:.3e} (< 1e-4: {mean_d < 1e-4})")
    ok = max_d < 1e-3 and mean_d < 1e-4
    if not ok:
        # 误差热图定位（R10 校准）：按像素位置打印偏差最大的点
        hm = diff.mean(dim=0)
        py, px = torch.where(hm > hm.quantile(0.999))
        print(f"  偏差最大像素（{len(px)} 个）: 示例 (x={px[:5].tolist()}, y={py[:5].tolist()})")
        np.save(OUT / "p6_splat_errmap.npy", hm.cpu().numpy())
        print(f"  [定位] 误差热图已存 {OUT/'p6_splat_errmap.npy'}")
    print(f"  §2.2 {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def run_d16_consistency_test() -> bool:
    """判据 3：D=16 慢速 splat 前 3 通道与 D=3 慢速 splat 逐元素相等。"""
    print("=" * 62)
    print("§2.3 D=16 自洽（前 3 通道与 D=3 慢速 splat 逐元素相等）")
    model, w2c, K, H, W = load_map_subset(2_000)
    add_feature_dim(model, 16)
    with torch.no_grad():
        model.params["features"].normal_(0, 1)
    feat16 = model.params["features"].detach()

    F16 = rasterize_feature(model, w2c, K, H, W, features=feat16)
    F3 = rasterize_feature(model, w2c, K, H, W, features=feat16[:, :3])
    d = float((F16[:3] - F3).abs().max())
    print(f"  D=16 前 3 通道 vs D=3: max|diff| = {d:.3e}（== 0: {d == 0}）")
    ok = d < 1e-6
    print(f"  §2.3 {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = [
        ("§2.1 数值梯度（feature + opacity）", run_grad_test()),
        ("§2.2 慢速 vs inria 互验 <1e-3", run_mutual_test()),
        ("§2.3 D=16 自洽", run_d16_consistency_test()),
    ]
    print("=" * 62)
    print("Phase 6 §2 特征光栅化验证汇总")
    all_ok = True
    for name, passed in checks:
        all_ok &= passed
        print(f"  [{'PASS ✅' if passed else 'FAIL ❌'}] {name}")
    print("=" * 62)
    print("全部 PASS ✅" if all_ok else "存在 FAIL ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
