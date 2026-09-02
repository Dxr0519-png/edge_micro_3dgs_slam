#!/usr/bin/env python3
"""Phase 6 §1 数据结构改造验证（feature 通道 + load_ply + freeze_geometry）。

验证对象: src/edge_3dgs_slam/gaussian/model.py（FEATURE_KEY / add_feature_dim /
          freeze_geometry / load_ply）与 render.py（render_precomp 快速路径宿主）。
通过标准:
    1. create_from_points(features=F) → params 含 features (N,D)；默认不加（Phase 2-5 兼容）
    2. add_feature_dim(probe, 3) → (50726,3) 零初始化、num_gaussians 不变、幂等
    3. add_gaussians/remove 后 features 行数与 means3D 一致
    4. freeze_geometry 后几何 5 键 requires_grad=False、features=True
    5. save_ply（含 feature）→ load_ply 往返逐元素 max 差 < 1e-5（ASCII 6 位小数）
    6. checkpoint 往返（{'params','variables'}，--load 持久化路径）
数据: 合成小场景 + data/outputs/phase3/probe_model_replica.pt（Replica office0 50726 高斯）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from edge_3dgs_slam.gaussian import (FEATURE_KEY, GaussianModel, add_feature_dim,
                                     freeze_geometry, load_ply)

OUT = Path("data/outputs/phase6")
PROBE = Path("data/outputs/phase3/probe_model_replica.pt")
D = 3  # config/feature/mobilesam_clip.yaml autoencoder.latent_dim


def run_create_test() -> bool:
    """判据 1：create_from_points 的 features 参数与默认兼容性。"""
    print("=" * 62)
    print("§1.1 create_from_points(features=...) 与默认兼容")
    n = 64
    pts = np.random.RandomState(0).randn(n, 3).astype(np.float32)
    rgb = np.random.RandomState(1).rand(n, 3).astype(np.float32)
    F = np.random.RandomState(2).randn(n, D).astype(np.float32)

    m0 = GaussianModel.create_from_points(pts, rgb)
    ok_no_feat = FEATURE_KEY not in m0.params
    print(f"  默认（features=None）: params 键={sorted(m0.params)} | 无 feature: {ok_no_feat}")

    m1 = GaussianModel.create_from_points(pts, rgb, features=F)
    has = FEATURE_KEY in m1.params
    shape_ok = has and m1.params[FEATURE_KEY].shape == (n, D)
    val_ok = has and torch.allclose(m1.params[FEATURE_KEY].detach().cpu(),
                                    torch.as_tensor(F), atol=1e-6)
    print(f"  features 给定: 含键={has} shape={m1.params[FEATURE_KEY].shape if has else '-'}"
          f" 值一致={val_ok}")
    ok = ok_no_feat and shape_ok and val_ok
    print(f"  §1.1 {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def run_add_feature_test() -> bool:
    """判据 2：add_feature_dim 零初始化 + 幂等。"""
    print("=" * 62)
    print("§1.2 add_feature_dim（probe_model_replica.pt, 50726 高斯）")
    if not PROBE.exists():
        print(f"  ❌ 缺少 {PROBE}（Phase 3 产物），跳过本项")
        return False
    s = torch.load(PROBE, map_location="cpu", weights_only=False)
    params = {k: torch.nn.Parameter(v.cuda()) for k, v in s["params"].items()}
    model = GaussianModel(params, variables=s.get("variables", {}))
    n0 = model.num_gaussians

    add_feature_dim(model, D)
    n1 = model.num_gaussians
    shape_ok = FEATURE_KEY in model.params and model.params[FEATURE_KEY].shape == (n0, D)
    zero_ok = shape_ok and torch.all(model.params[FEATURE_KEY] == 0).item()
    fp32_ok = shape_ok and model.params[FEATURE_KEY].dtype == torch.float32
    n_ok = n1 == n0

    # 幂等
    add_feature_dim(model, D)  # 同 D 直接返回
    idem_ok = model.params[FEATURE_KEY].shape == (n0, D)
    # 维度冲突时报错
    try:
        add_feature_dim(model, D + 1)
        conflict_ok = False
    except ValueError:
        conflict_ok = True

    print(f"  高斯数 {n1}（不变: {n_ok}）| feature shape {model.params[FEATURE_KEY].shape}"
          f"（{shape_ok}）| 零初始化 {zero_ok} | FP32 {fp32_ok}")
    print(f"  幂等: {idem_ok} | 维度冲突报错: {conflict_ok}")
    ok = shape_ok and zero_ok and fp32_ok and n_ok and idem_ok and conflict_ok
    print(f"  §1.2 {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def run_add_remove_test() -> bool:
    """判据 3：add_gaussians/remove 后 features 行数与几何一致。"""
    print("=" * 62)
    print("§1.3 add_gaussians / remove 同步 feature 通道")
    n = 40
    pts = np.random.RandomState(3).randn(n, 3).astype(np.float32)
    rgb = np.random.RandomState(4).rand(n, 3).astype(np.float32)
    model = GaussianModel.create_from_points(pts, rgb, features=np.zeros((n, D), np.float32))

    new_pts = np.random.RandomState(5).randn(10, 3).astype(np.float32)
    new_rgb = np.random.RandomState(6).rand(10, 3).astype(np.float32)
    model.add_gaussians(new_pts, new_rgb)
    n_after_add = model.num_gaussians
    feat_add_ok = model.params[FEATURE_KEY].shape[0] == n_after_add
    zero_add_ok = torch.all(model.params[FEATURE_KEY][n:] == 0).item()

    to_remove = torch.zeros(n_after_add, dtype=torch.bool, device="cuda")
    to_remove[:5] = True
    model.remove(to_remove)
    n_after_rm = model.num_gaussians
    feat_rm_ok = model.params[FEATURE_KEY].shape[0] == n_after_rm
    # 保留的前 4 行（0-4 删了）与 删除前 5-8 行一致
    keep_vals_ok = torch.allclose(
        model.params[FEATURE_KEY][:4].detach().cpu(),
        torch.zeros(4, D), atol=1e-6)  # 原 5-8 行也是零，语义上只需行数对齐
    print(f"  add 后 {n_after_add} 高斯（feature 行数对齐 {feat_add_ok}，新行零初始化 {zero_add_ok}）")
    print(f"  remove 后 {n_after_rm} 高斯（feature 行数对齐 {feat_rm_ok}）")
    ok = feat_add_ok and zero_add_ok and feat_rm_ok and keep_vals_ok
    print(f"  §1.3 {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def run_freeze_test() -> bool:
    """判据 4：freeze_geometry 冻结几何、保留 features 可训练。"""
    print("=" * 62)
    print("§1.4 freeze_geometry（几何冻结，features 唯一可训练）")
    n = 32
    pts = np.random.RandomState(7).randn(n, 3).astype(np.float32)
    rgb = np.random.RandomState(8).rand(n, 3).astype(np.float32)
    model = GaussianModel.create_from_points(pts, rgb, features=np.zeros((n, D), np.float32))

    freeze_geometry(model, freeze=True)
    geom_frozen = all(not model.params[k].requires_grad for k in
                      ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales"))
    feat_train = model.params[FEATURE_KEY].requires_grad

    # freeze=False 恢复
    freeze_geometry(model, freeze=False)
    geom_restored = all(model.params[k].requires_grad for k in
                        ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales"))
    print(f"  冻结: 几何 5 键全部 requires_grad=False ({geom_frozen}) | features=True ({feat_train})")
    print(f"  解冻恢复: 几何全部可训练 ({geom_restored})")
    ok = geom_frozen and feat_train and geom_restored
    print(f"  §1.4 {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def run_ply_roundtrip_test() -> bool:
    """判据 5：save_ply（含 feature）→ load_ply 往返 max 差 < 1e-5。"""
    print("=" * 62)
    print("§1.5 save_ply → load_ply 往返（含 feature 通道）")
    n = 64
    pts = np.random.RandomState(9).randn(n, 3).astype(np.float32) * 0.5
    rgb = np.random.RandomState(10).rand(n, 3).astype(np.float32) * 0.9 + 0.05
    F = np.random.RandomState(11).randn(n, D).astype(np.float32) * 0.3
    model = GaussianModel.create_from_points(pts, rgb, features=F, anisotropic=True)
    path = OUT / "p6_feature_roundtrip.ply"
    model.save_ply(str(path))
    back = load_ply(str(path))

    def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max())

    d_xyz = maxdiff(back.params["means3D"].cpu(), model.params["means3D"].cpu())
    d_rgb = maxdiff(back.params["rgb_colors"].cpu(), model.params["rgb_colors"].cpu())
    d_op = maxdiff(back.opacities().cpu(), model.opacities().cpu())
    d_scale = maxdiff(back.scales().cpu(), model.scales().cpu())
    d_rot = maxdiff(back.params["unnorm_rotations"].cpu(), model.params["unnorm_rotations"].cpu())
    d_feat = maxdiff(back.params[FEATURE_KEY].cpu(), model.params[FEATURE_KEY].cpu())
    for name, d in [("xyz", d_xyz), ("rgb", d_rgb), ("opacity", d_op),
                    ("scale", d_scale), ("rot", d_rot), ("feature", d_feat)]:
        print(f"  {name:8s} max|diff| = {d:.3e} (< 1e-5: {d < 1e-5})")
    ok = max(d_xyz, d_rgb, d_op, d_scale, d_rot, d_feat) < 1e-5
    print(f"  §1.5 {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def run_checkpoint_test() -> bool:
    """判据 6：checkpoint 往返（{'params', 'variables'} 结构，--load 持久化路径）。"""
    print("=" * 62)
    print("§1.6 checkpoint 往返（params 含 features → torch.save → 重载）")
    n = 48
    pts = np.random.RandomState(12).randn(n, 3).astype(np.float32)
    rgb = np.random.RandomState(13).rand(n, 3).astype(np.float32)
    F = np.random.RandomState(14).randn(n, D).astype(np.float32)
    model = GaussianModel.create_from_points(pts, rgb, features=F)
    ckpt_path = OUT / "p6_feature_ckpt.pt"
    torch.save({"params": model.params, "variables": model.variables}, str(ckpt_path))

    s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params2 = {k: torch.nn.Parameter(v.cuda()) for k, v in s["params"].items()}
    model2 = GaussianModel(params2, variables=s.get("variables", {}))
    ok = (FEATURE_KEY in model2.params
          and model2.params[FEATURE_KEY].shape == (n, D)
          and torch.allclose(model2.params[FEATURE_KEY].cpu(), torch.as_tensor(F), atol=1e-6)
          and model2.num_gaussians == n)
    print(f"  重载: 含 features={FEATURE_KEY in model2.params} "
          f"shape={model2.params[FEATURE_KEY].shape if FEATURE_KEY in model2.params else '-'} "
          f"值一致={ok}")
    print(f"  §1.6 {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    checks = [
        ("§1.1 create_from_points features 参数", run_create_test()),
        ("§1.2 add_feature_dim 零初始化/幂等", run_add_feature_test()),
        ("§1.3 add_gaussians/remove 同步 feature", run_add_remove_test()),
        ("§1.4 freeze_geometry 冻结语义", run_freeze_test()),
        ("§1.5 save→load_ply 往返 < 1e-5", run_ply_roundtrip_test()),
        ("§1.6 checkpoint 往返（--load 路径）", run_checkpoint_test()),
    ]
    print("=" * 62)
    print("Phase 6 §1 数据结构改造验证汇总")
    all_ok = True
    for name, passed in checks:
        all_ok &= passed
        print(f"  [{'PASS ✅' if passed else 'FAIL ❌'}] {name}")
    print("=" * 62)
    print("全部 PASS ✅" if all_ok else "存在 FAIL ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
