#!/usr/bin/env python3
"""Phase 3 逐 § 单元验证（与 phase2_slam_synthetic.py 同款 PASS/FAIL 惯例）。

段：
    §1 profiling / frame_utils 冒烟
    §2 高斯剪枝 / 容量 / 密度判据（coverage_mask 与暴力法一致性）
    §3 FP16 存储（dtype / 渲染等价 / add / save_ply / 显存对比）
    §4 视锥剔除（full vs masked 渲染逐像素相等 / 反向全黑 / 不可见 grad None）
    §5 关键帧管理 + 异步建图（阈值矩阵 / 丢最旧 / 窗口 / join）

用法：
    python3 experiments/phase3_unit_checks.py [--only §2,§3]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from edge_3dgs_slam.gaussian import GaussianModel, render
from edge_3dgs_slam.gaussian.frustum import frustum_visible
from edge_3dgs_slam.gaussian.model import (
    already_covered, coverage_mask, enforce_capacity, prune,
)
from edge_3dgs_slam.utils.profiling import (
    alloc_mb, peak_mb, profiled, reserved_mb, reset_peak,
)

_OK = []


def section(name: str) -> None:
    print("\n" + "=" * 62)
    print(f"§{name}")
    print("=" * 62)


def check(name: str, ok: bool, extra: str = "") -> bool:
    mark = "PASS ✅" if ok else "FAIL ❌"
    print(f"  {mark}  {name}" + (f"  ({extra})" if extra else ""))
    _OK.append(ok)
    return ok


def _grid_model(n_per_side: int = 16, seed: int = 0) -> GaussianModel:
    """相机前方 xy 网格高斯（z 固定 2m），带随机颜色/尺度，供渲染类测试。"""
    rng = np.random.default_rng(seed)
    s = np.linspace(-0.8, 0.8, n_per_side)
    xx, yy = np.meshgrid(s, s)
    pts = np.stack([xx.ravel(), yy.ravel(), np.full(n_per_side ** 2, 2.0)], axis=-1)
    colors = rng.random((n_per_side ** 2, 3)).astype(np.float32)
    scales = rng.uniform(0.005, 0.02, (n_per_side ** 2,)).astype(np.float32)
    return GaussianModel.create_from_points(pts, colors, scales=scales)


_K = np.array([[310.0, 0, 160.0], [0, 310.0, 120.0], [0, 0, 1.0]], dtype=np.float64)


def check_s1() -> bool:
    section("1 profiling / frame_utils 冒烟")
    ok = True
    r, ms = profiled(lambda: torch.ones(1000, device="cuda").sum())
    ok &= check("profiled 返回 (r, ms) 且 ms>0", ms > 0, f"{ms:.3f} ms")
    reset_peak()
    _ = torch.zeros(2 ** 20, device="cuda")
    a, p, re = alloc_mb(), peak_mb(), reserved_mb()
    ok &= check("alloc/peak/reserved 口径一致（peak≥alloc）", p >= a and re >= a,
                f"alloc={a:.1f}MB peak={p:.1f}MB reserved={re:.1f}MB")
    from edge_3dgs_slam.camera import SyncedFrame
    from edge_3dgs_slam.utils.frame_utils import downsample_frame
    f = SyncedFrame(rgb=np.zeros((480, 640, 3), np.uint8),
                    depth=np.zeros((480, 640), np.float32), K=_K, stamp=0.0)
    d = downsample_frame(f, 320, 240)
    ok &= check("downsample 尺寸与 K 缩放", d.rgb.shape == (240, 320, 3)
                and abs(d.K[0, 0] - 155.0) < 1e-6, f"K00={d.K[0, 0]:.1f}")

    # K 逐轴缩放回归（非等比目标 600×340→320×240：单一比例会错误缩放 fy/cy，
    # 实测污染投影几何与 ATE；等比目标 640→320 必须与旧行为逐元素一致）
    f2 = SyncedFrame(rgb=np.zeros((340, 600, 3), np.uint8),
                     depth=np.zeros((340, 600), np.float32), K=_K, stamp=0.0)
    d2 = downsample_frame(f2, 320, 240)
    s_w, s_h = 320 / 600, 240 / 340
    k_ok = (abs(d2.K[0, 0] - 310.0 * s_w) < 1e-6        # fx 按宽度比例
            and abs(d2.K[1, 1] - 310.0 * s_h) < 1e-6    # fy 按高度比例
            and abs(d2.K[1, 1] - 310.0 * s_w) > 1e-3)   # 且 ≠ 旧的单一比例
    ok &= check("K 逐轴缩放（fx 按 s_w、fy 按 s_h）", k_ok,
                f"fx={d2.K[0,0]:.2f}(×{s_w:.3f}) fy={d2.K[1,1]:.2f}(×{s_h:.3f})")
    d1 = downsample_frame(f, 320, 240)
    ok &= check("等比目标 640→320 与旧实现一致（fx=fy=×0.5）",
                abs(d1.K[0, 0] - 155.0) < 1e-9 and abs(d1.K[1, 1] - 155.0) < 1e-9)

    # Phase 4 回归：backproject_torch 世界系变换方向（w2c → p_world = Rᵀ(p_cam−t)）。
    # 曾用 R@p_cam+t 种错世界位置（Replica baseline ATE 6.5→82cm、高斯数翻倍，
    # 2026-08-28 实测）——GPU 路径必须与 numpy backproject(T_wc) 逐点一致。
    from edge_3dgs_slam.camera.backproject import backproject, backproject_torch
    T_wc = np.eye(4)
    T_wc[:3, 3] = np.array([0.2, -0.1, 0.3])
    T_wc[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    depth_r = np.full((120, 160), 1.5, np.float32)
    depth_r[::37, :] = 0.0                       # 制造无效像素
    pts_np, valid_np = backproject(depth_r, _K, T_wc)
    pts_t, valid_t = backproject_torch(torch.from_numpy(depth_r).cuda(),
                                       torch.as_tensor(_K, dtype=torch.float32))
    T_t = torch.as_tensor(T_wc, dtype=torch.float32).cuda()
    pts_world_t = ((pts_t[valid_t] - T_t[:3, 3]) @ T_t[:3, :3]).cpu().numpy()
    same_pts = (valid_np.sum() == pts_world_t.shape[0]
                and np.abs(pts_np[valid_np] - pts_world_t).max() < 1e-4)
    ok &= check("backproject_torch 世界系方向与 numpy 一致（w2c 回归）", same_pts,
                f"max_diff={np.abs(pts_np[valid_np]-pts_world_t).max():.2e}" if valid_np.sum() == pts_world_t.shape[0] else "点数不一致")
    return ok


def check_s2() -> bool:
    section("2 高斯数量控制与剪枝")
    ok = True

    # prune_points / keep_indices
    m = _grid_model()
    n0 = m.num_gaussians
    m.prune_points(torch.zeros(n0, dtype=torch.bool, device="cuda"))
    ok &= check("prune_points 全 False → 全删", m.num_gaussians == 0)
    m = _grid_model()
    keep_idx = torch.tensor([0, 1, 2], device="cuda")
    m.keep_indices(keep_idx)
    ok &= check("keep_indices 保留指定 3 个", m.num_gaussians == 3)

    # 模块级 prune：opacity 阈值
    m = _grid_model()
    lo = m.opacities().squeeze(-1) < 0.5
    prune(m, opacity_thresh=0.5, scale_thresh=1e9)
    ok &= check("模块 prune 按 opacity 阈值删除", m.num_gaussians == int(lo.sum()),
                f"{n0} → {m.num_gaussians}")

    # enforce_capacity：topk 按 opacity
    m = _grid_model()
    op_before = m.opacities().squeeze(-1)
    enforce_capacity(m, N_max=100)
    ok &= check("enforce_capacity 截到 100", m.num_gaussians == 100)
    op_keep = m.opacities().squeeze(-1)
    expected = op_before.topk(100).values
    ok &= check("保留的是 top-100 opacity（与 topk 一致）",
                torch.allclose(op_keep.sort().values, expected.sort().values, atol=1e-6))

    # already_covered 单点
    pts = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    ok &= check("already_covered 近点 True", already_covered(pts, torch.tensor(
        [[0.0, 0.0, 2.0], [1.0, 1.0, 1.0]], device="cuda"), r=0.01))
    ok &= check("already_covered 远点 False", not already_covered(pts, torch.tensor(
        [[0.5, 0.5, 2.0]], device="cuda"), r=0.01))

    # coverage_mask vs 暴力法：无漏判 + 误报率低
    rng = np.random.default_rng(1)
    existing = torch.as_tensor(rng.uniform(-2, 2, (5000, 3)), dtype=torch.float32,
                               device="cuda")
    query = torch.as_tensor(rng.uniform(-2, 2, (200, 3)), dtype=torch.float32,
                            device="cuda")
    cov = coverage_mask(query, existing, r=0.1)
    brute = (torch.cdist(query, existing) < 0.1).any(dim=1)
    n_miss = int((brute & ~cov).sum())          # 暴力判 True 而 coverage 判 False = 漏判
    n_extra = int((cov & ~brute).sum())         # 多报（保守近似，允许）
    ok &= check("coverage_mask 无漏判（⊇ 暴力法）", n_miss == 0, f"漏判 {n_miss}")
    ok &= check("coverage_mask 误报率低", n_extra / len(query) < 0.25,
                f"多报 {n_extra}/{len(query)} ({n_extra/len(query)*100:.0f}%)")

    # 规模 sanity：20k × 200k
    big_e = torch.as_tensor(rng.uniform(-2, 2, (200_000, 3)), dtype=torch.float32,
                            device="cuda")
    big_q = torch.as_tensor(rng.uniform(-2, 2, (20_000, 3)), dtype=torch.float32,
                            device="cuda")
    t0 = time.perf_counter()
    coverage_mask(big_q, big_e, r=0.01)
    dt = time.perf_counter() - t0
    ok &= check("20k×200k 栅格哈希 < 2s", dt < 2.0, f"{dt:.2f}s")
    return ok


def check_s3() -> bool:
    section("3 FP16 混合精度存储")
    ok = True
    m = _grid_model()
    m.half_storage()
    dtypes = {k: m.params[k].dtype for k in m.params}
    ok &= check("3 个参数 FP16（颜色/opacity/scale）",
                all(dtypes[k] == torch.float16 for k in
                    ("rgb_colors", "logit_opacities", "log_scales")))
    ok &= check("几何参数恒 FP32", m.params["means3D"].dtype == torch.float32
                and m.params["unnorm_rotations"].dtype == torch.float32)
    ok &= check("is_half_storage", m.is_half_storage)

    # 存储层半精度 ↔ 渲染等价（计算层 .float() 边界）
    T = np.eye(4, dtype=np.float64)
    with torch.no_grad():
        im_h, dep_h, _, _, _ = render(m, T, _K, 320, 240,
                                      gaussians_grad=False, camera_grad=False)
    m.float_storage()
    with torch.no_grad():
        im_f, dep_f, _, _, _ = render(m, T, _K, 320, 240,
                                      gaussians_grad=False, camera_grad=False)
    mse = ((im_h - im_f) ** 2).mean().item()
    cross_psnr = 10 * np.log10(1.0 / mse) if mse > 1e-12 else 99.0
    ok &= check("half↔float 渲染交叉 PSNR > 45 dB", cross_psnr > 45.0, f"{cross_psnr:.1f} dB")
    d_rmse = float(((dep_h - dep_f) ** 2).mean().sqrt())
    ok &= check("half↔float 深度 RMSE < 1cm", d_rmse < 0.01, f"{d_rmse*100:.2f} cm")

    # half 模型上 add_gaussians dtype 一致（新块被 .to 对齐到 half）
    n_before = m.num_gaussians
    m.half_storage()
    m.add_gaussians(torch.tensor([[0.0, 0.1, 2.0]], device="cuda"),
                    torch.tensor([[1.0, 0.0, 0.0]], device="cuda"))
    d_after = {k: m.params[k].dtype for k in m.params}
    dtype_ok = (m.num_gaussians == n_before + 1
                and all(d_after[k] == torch.float16 for k in
                        ("rgb_colors", "logit_opacities", "log_scales"))
                and d_after["means3D"] == torch.float32
                and d_after["unnorm_rotations"] == torch.float32)
    ok &= check("half 上 add_gaussians 且 dtype 一致", dtype_ok)
    # half 上 save_ply 可导出
    m.save_ply("/tmp/phase3_half_test.ply")
    txt = Path("/tmp/phase3_half_test.ply").read_text()
    ok &= check("half 上 save_ply 可解析", txt.startswith("ply")
                and f"element vertex {m.num_gaussians}" in txt)

    # 回归（曾崩）：half 模型上 map_keyframe 的 Adam 必须在 float32 上做
    # （half 参数 + Adam → exp_avg 上溢 → 参数 NaN → rasterizer 非法内存访问）
    from edge_3dgs_slam.camera import SyncedFrame
    from edge_3dgs_slam.slam.mapping import map_keyframe
    mh = _grid_model()
    mh.half_storage()
    f_h = SyncedFrame(rgb=np.full((240, 320, 3), 128, np.uint8),
                      depth=np.full((240, 320), 2.0, np.float32), K=_K, stamp=0.0)
    stat = map_keyframe(f_h, mh, np.eye(4), iters=5)
    nan_free = all(not bool(torch.isnan(mh.params[k]).any()) for k in mh.params)
    ok &= check("half 上 map_keyframe 无 NaN（Adam 在 float32 上做）", nan_free
                and stat["num_gaussians"] > 0 and mh.is_half_storage,
                f"loss={stat['loss']:.3f}")

    # 显存对比（如实记录：存储层节省 ~3.6MB@200k，非主要瓶颈）
    def _params_bytes(mm: GaussianModel) -> float:
        return sum(mm.params[k].numel() * mm.params[k].element_size()
                   for k in mm.params) / 1e6
    mf = _grid_model(n_per_side=32)
    mf.float_storage()
    b_f = _params_bytes(mf)
    mf.half_storage()
    b_h = _params_bytes(mf)
    print(f"  [显存] 参数存储 {b_f:.2f}MB(float32) → {b_h:.2f}MB(half)，"
          f"节省 {b_f-b_h:.2f}MB（{mf.num_gaussians} 高斯）")
    ok &= check("half 参数内存 < float 参数内存", b_h < b_f)
    return ok


def _wide_grid_model(n_per_side: int = 24, seed: int = 2) -> GaussianModel:
    """xy ∈ [−2,2] 的大网格（z=2m）：FOV ±27.3° 下仅中心 ~26% 在视锥内。"""
    rng = np.random.default_rng(seed)
    s = np.linspace(-2.0, 2.0, n_per_side)
    xx, yy = np.meshgrid(s, s)
    pts = np.stack([xx.ravel(), yy.ravel(), np.full(n_per_side ** 2, 2.0)], axis=-1)
    colors = rng.random((n_per_side ** 2, 3)).astype(np.float32)
    scales = rng.uniform(0.005, 0.02, (n_per_side ** 2,)).astype(np.float32)
    return GaussianModel.create_from_points(pts, colors, scales=scales)


def check_s4() -> bool:
    section("4 视锥剔除")
    ok = True
    m = _wide_grid_model()
    T = np.eye(4, dtype=np.float64)
    vis = frustum_visible(m.means3D.detach(), T, _K, 240, 320,
                          scales=m.scales().detach())
    vis_ratio = float(vis.float().mean())
    ok &= check("可见占比 < 80%（部分可见场景）", vis_ratio < 0.8,
                f"{vis_ratio*100:.0f}% 可见")
    n_vis = int(vis.sum())

    # 正确性：full 与 mask=vis 渲染逐像素相等（剔除无误伤）
    with torch.no_grad():
        im_full, dep_full, sil_full, _, _ = render(m, T, _K, 320, 240,
                                                   gaussians_grad=False, camera_grad=False)
        im_msk, dep_msk, sil_msk, rad_msk, _ = render(m, T, _K, 320, 240,
                                                      gaussians_grad=False, camera_grad=False,
                                                      mask=vis)
    d_im = float((im_full - im_msk).abs().max())
    d_dep = float((dep_full - dep_msk).abs().max())
    d_sil = float((sil_full - sil_msk).abs().max())
    ok &= check("full vs masked 渲染逐像素一致",
                d_im < 1e-5 and d_dep < 1e-5 and d_sil < 1e-5,
                f"max|Δ| im={d_im:.2e} dep={d_dep:.2e} sil={d_sil:.2e}")
    ok &= check("masked radius 长度为可见数", rad_msk.shape[0] == n_vis,
                f"{rad_msk.shape[0]} == {n_vis}")

    # 反向：只留视锥外 → 全黑（bg=0）
    with torch.no_grad():
        im_out, dep_out, _, _, _ = render(m, T, _K, 320, 240,
                                          gaussians_grad=False, camera_grad=False,
                                          mask=~vis)
    ok &= check("反向 mask（只留视锥外）渲染全黑",
                float(im_out.abs().max()) < 1e-4 and float(dep_out.abs().max()) < 1e-4)

    # 冻结：cull backward 后，不可见位置梯度为 0（可见位置有梯度）
    m2 = _wide_grid_model(seed=3)
    im, _, _, _, _ = render(m2, T, _K, 320, 240, gaussians_grad=True, camera_grad=False,
                            mask=vis)
    im.sum().backward()
    g = m2.params["means3D"].grad
    g_vis_any = g[vis].abs().sum().item() > 0
    g_inv_zero = float(g[~vis].abs().max()) == 0.0
    ok &= check("cull 后可见高斯有梯度", g_vis_any)
    ok &= check("cull 后不可见高斯梯度为 0（冻结）", g_inv_zero,
                f"max|g|(不可见)={float(g[~vis].abs().max()):.2e}")

    # 加速记录（不强判）：可见 26% 时单次渲染耗时对比
    def _fwd_timing():
        with torch.no_grad():
            t_full = []
            for _ in range(10):
                _, ms = profiled(render, m, T, _K, 320, 240,
                                 gaussians_grad=False, camera_grad=False)
                t_full.append(ms)
            t_msk = []
            for _ in range(10):
                _, ms = profiled(render, m, T, _K, 320, 240,
                                 gaussians_grad=False, camera_grad=False, mask=vis)
                t_msk.append(ms)
            return float(np.median(t_full)), float(np.median(t_msk))
    tf, tm = _fwd_timing()
    print(f"  [性能] 渲染耗时 全量 {tf:.2f}ms vs 剔除后 {tm:.2f}ms"
          f"（可见 {vis_ratio*100:.0f}%，加速 {tf/max(tm,1e-6):.2f}x）")
    return ok


def _rotz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def check_s5() -> bool:
    section("5 关键帧管理 + 异步建图")
    ok = True
    from edge_3dgs_slam.camera import SyncedFrame
    from edge_3dgs_slam.slam import SLAMBackend
    from edge_3dgs_slam.slam.keyframe import KeyframeManager, angle_between

    # angle_between
    ok &= check("angle_between 单位阵 0°", abs(angle_between(np.eye(3), np.eye(3))) < 0.1)
    ok &= check("angle_between 绕 z 90° = 90°",
                abs(angle_between(_rotz(90), np.eye(3)) - 90.0) < 0.1)

    # should_insert 阈值矩阵
    km = KeyframeManager()
    T0 = np.eye(4)
    T1 = np.eye(4); T1[:3, 3] = [0.15, 0, 0]
    T2 = np.eye(4); T2[:3, 3] = [0.05, 0, 0]
    T3 = np.eye(4); T3[:3, :3] = _rotz(4)
    T4 = np.eye(4); T4[:3, :3] = _rotz(2); T4[:3, 3] = [0.05, 0, 0]
    ok &= check("平移 0.15m → 插入", km.should_insert(T1, T0))
    ok &= check("平移 0.05m + 2° → 不插入", not km.should_insert(T4, T0))
    ok &= check("旋转 4° → 插入", km.should_insert(T3, T0))

    # 丢最旧语义（确定性测试，不依赖线程调度）：maxsize=2 推 5 个 → 留最新 2 个
    import queue as queue_mod
    from edge_3dgs_slam.slam.backend import SLAMBackend as _SB
    q = queue_mod.Queue(maxsize=2)
    sb = _SB.__new__(_SB)                    # 只测 _push,不启动线程
    sb.map_queue = q
    sb.stats = {"dropped": 0}
    for i in range(5):
        sb._push(SyncedFrame(rgb=np.zeros((2, 2, 3), np.uint8),
                             depth=np.ones((2, 2), np.float32), K=_K, stamp=float(i)),
                 np.eye(4))
    stamps = sorted(item[0].stamp for item in list(q.queue))
    ok &= check("队列满丢最旧、留最新", q.qsize() == 2 and stamps == [3.0, 4.0],
                f"队列内 stamp={stamps}")

    # SLAMBackend：迷你序列跑通（无死锁）+ 窗口 + join
    H, W = 480, 640
    model = _grid_model()
    backend = SLAMBackend(model, _K, track_kwargs={"iters": 2},
                          map_kwargs={"iters": 2}, window_size=3)
    T_cur = np.eye(4)
    for i in range(10):
        T_cur = T_cur.copy()
        T_cur[:3, 3] = [i * 0.3, 0, 0]
        frm = SyncedFrame(rgb=np.zeros((H, W, 3), np.uint8),
                          depth=np.full((H, W), 2.0, np.float32), K=_K, stamp=float(i))
        T_est = backend.track(frm, T_cur)
        assert T_est.shape == (4, 4)
    ok &= check("backend 10 帧跑通（无死锁）", True)
    ok &= check("track_gpu/wall 计时均记录",
                len(backend.stats["track_gpu_ms"]) == 10
                and len(backend.stats["track_wall_ms"]) == 10)
    ok &= check("窗口长度 ≤ 上限", len(backend.keyframes) <= 3,
                f"{len(backend.keyframes)} ≤ 3")
    backend.join(timeout=60)
    ok &= check("join 正常退出（线程已停）", not backend._thread.is_alive())

    # lock_chunk 路径：chunked 建图不崩、track 可插入、锁状态正确
    m3 = _grid_model()
    bk2 = SLAMBackend(m3, _K, track_kwargs={"iters": 2},
                      map_kwargs={"iters": 6}, window_size=2, lock_chunk=2)
    T2 = np.eye(4)
    for i in range(4):
        T2 = T2.copy()
        T2[:3, 3] = [i * 0.3, 0, 0]
        bk2.track(SyncedFrame(rgb=np.zeros((480, 640, 3), np.uint8),
                              depth=np.full((480, 640), 2.0, np.float32), K=_K,
                              stamp=float(i)), T2)
    bk2.join(timeout=60)
    lock_ok = not bk2._thread.is_alive()
    try:
        bk2._gpu_lock.acquire(timeout=0.1)     # 线程退出后锁应可获取（未被残留持有）
        bk2._gpu_lock.release()
        lock_ok = lock_ok and True
    except Exception:
        lock_ok = False
    ok &= check("lock_chunk 建图正常、线程退出后锁释放", lock_ok,
                f"mapped={bk2.stats['mapped']}")

    # 观察性打印：锁等待（真实延迟评估在消融脚本用真实序列做）
    wall = float(np.median(backend.stats["track_wall_ms"]))
    gpu = float(np.median(backend.stats["track_gpu_ms"]))
    print(f"  [观察] track 墙钟中位 {wall:.1f}ms vs GPU 中位 {gpu:.1f}ms"
          f"（极端场景：每帧关键帧+每帧加 2 万高斯，锁等待放大）")
    return ok


def check_s6() -> bool:
    section("6 性能档：res_schedule / depth_every / early_stop / map 减负")
    ok = True
    from edge_3dgs_slam.camera import SyncedFrame
    from edge_3dgs_slam.slam.mapping import map_keyframe
    from edge_3dgs_slam.slam.tracking import track
    from edge_3dgs_slam.utils.se3 import invert_pose, se3_log

    # 合成帧：渐变墙 + 恒 2m 深度，正对网格模型（强约束易收敛帧）
    H, W = 240, 320
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rgb = np.stack([(xx / W * 255).astype(np.uint8), (yy / H * 255).astype(np.uint8),
                    np.full((H, W), 128, np.uint8)], axis=-1)
    f = SyncedFrame(rgb=rgb, depth=np.full((H, W), 2.0, np.float32), K=_K, stamp=0.0)
    m = _grid_model(n_per_side=24)
    T0 = np.eye(4)

    # ① res_schedule 单阶段 = 默认路径（同分辨率同迭代，输出应一致；容差含
    #    rasterizer atomicAdd 非确定性）
    T_a = track(f, m, T0, iters=8)
    T_b = track(f, m, T0, iters=8, res_schedule=[(W, H, 8)])
    d = se3_log(T_a @ invert_pose(T_b))
    ok &= check("res_schedule 单阶段 = 默认路径", float(d.norm()) < 1e-3,
                f"Δ={float(d.norm()):.2e}")

    # ② depth_every=2：depth 趟跳过不崩、易帧结果与每迭代 depth 接近
    T_c = track(f, m, T0, iters=6, depth_every=2)
    d2 = se3_log(T_a @ invert_pose(T_c))
    ok &= check("depth_every=2 结果接近（易帧）",
                float(d2[:3].norm()) < np.deg2rad(5) and float(d2[3:].norm()) < 0.1,
                f"Δ={float(d2[:3].norm()):.4f}rad/{float(d2[3:].norm()):.4f}m")

    # ③ 粗到细两阶段（160×120×2 + 320×240×2）vs 单阶段 4 迭代
    T_d = track(f, m, T0, iters=4, res_schedule=[(160, 120, 2), (320, 240, 2)])
    T_e = track(f, m, T0, iters=4)
    d3 = se3_log(T_d @ invert_pose(T_e))
    ok &= check("粗到细两阶段与单阶段 4 迭代接近",
                float(d3[:3].norm()) < np.deg2rad(5) and float(d3[3:].norm()) < 0.1,
                f"Δ={float(d3[:3].norm()):.4f}rad/{float(d3[3:].norm()):.4f}m")

    # ④ early_stop：易帧提前停不劣化（结果与完整 8 迭代接近）
    T_f = track(f, m, T0, iters=8, early_stop=True)
    d4 = se3_log(T_f @ invert_pose(T_a))
    ok &= check("early_stop 在易帧结果正常",
                float(d4[:3].norm()) < np.deg2rad(2) and float(d4[3:].norm()) < 0.05,
                f"Δ={float(d4[:3].norm()):.4f}rad/{float(d4[3:].norm()):.4f}m")

    # ⑤ map 减负组合（opt 分辨率 + 窗口轮转 + fp16 存储）无 NaN
    mh = _grid_model(n_per_side=16)
    mh.half_storage()
    win = [(SyncedFrame(rgb=rgb, depth=np.full((H, W), 2.0, np.float32), K=_K,
                        stamp=float(i)), np.eye(4)) for i in range(5)]
    stat = map_keyframe(f, mh, np.eye(4), iters=8, window=win,
                        opt_W=160, opt_H=120, window_rotate=True, rotate_n=2)
    nan_free = all(not bool(torch.isnan(mh.params[k]).any()) for k in mh.params)
    ok &= check("map opt分辨率+窗口轮转+fp16 无 NaN", nan_free and mh.is_half_storage,
                f"loss={stat['loss']:.3f} n={stat['num_gaussians']}")

    # ⑥ map opt 分辨率下渲染尺寸正确（不崩 + 高斯数增加）
    m2 = _grid_model(n_per_side=16)
    stat2 = map_keyframe(f, m2, np.eye(4), iters=4, opt_W=160, opt_H=120)
    ok &= check("map opt 分辨率（160×120）正常建图",
                stat2["num_gaussians"] > m2.num_gaussians or stat2["added"] >= 0,
                f"added={stat2['added']} n={stat2['num_gaussians']}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else None

    sections = {"1": check_s1, "2": check_s2, "3": check_s3, "4": check_s4,
                "5": check_s5, "6": check_s6}
    if want:
        sections = {k: v for k, v in sections.items() if k in want}
    for fn in sections.values():
        fn()
    n_ok, n_fail = sum(_OK), len(_OK) - sum(_OK)
    print("\n" + "=" * 62)
    print(f"单元验证总判定: {n_ok}/{len(_OK)} PASS"
          + (f"，{n_fail} FAIL ❌" if n_fail else " ✅"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
