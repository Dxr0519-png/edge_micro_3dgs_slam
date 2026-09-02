#!/usr/bin/env python3
"""Phase 3 性能探针：先测量再决策（docs/03 目标 15-30 FPS 的量化依据）。

--mode per-iter  track 单迭代分段分解（分辨率 × needs_depth）→ 校准 15/30 FPS 配置边界
--mode iters     track (分辨率, 迭代数, depth_every) 边际收益矩阵（Step 2 决策门）
--mode map       map 分辨率 / 窗口轮转 / lock_chunk 减负探针（Step 3）

数据：Replica office0（third_party/SplaTAM/data/Replica，frame_scaled(0.5) → 600×340，
与 phase2_replica_eval 同参）+ 合成序列（640×480）对照。
模型：确定性构建 200k 高斯（build_map 40 kf × 20 iters + enforce_capacity），
缓存到 data/outputs/phase3/probe_replica_model.pt（--rebuild-model 强制重建）。

用法：
    python3 experiments/phase3_track_probe.py --mode per-iter [--frames 100]
    python3 experiments/phase3_track_probe.py --mode iters [--frames 100]
    python3 experiments/phase3_track_probe.py --mode map [--frames 100]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import numpy as np
import torch

from edge_3dgs_slam.camera import SyncedFrame
from edge_3dgs_slam.dataset import ReplicaSequence
from edge_3dgs_slam.gaussian import GaussianModel, render
from edge_3dgs_slam.gaussian.model import enforce_capacity
from edge_3dgs_slam.slam import build_map, track
from edge_3dgs_slam.slam.keyframe import KeyframeManager
from edge_3dgs_slam.utils.frame_utils import downsample_frame
from edge_3dgs_slam.utils.profiling import profiled
from edge_3dgs_slam.utils.se3 import se3_exp

REPLICA_ROOT = Path(__file__).resolve().parents[1] / "third_party/SplaTAM/data/Replica"
OUT = Path(__file__).resolve().parents[1] / "data/outputs/phase3"
OUT.mkdir(parents=True, exist_ok=True)
# 探针模型缓存按数据源分开：replica 序列新增慢（~50k），合成序列能到 200k
# （与消融同源的负载代表）——per-iter 分解用 synth 模型测真实光栅化缩放
MODEL_CACHE = OUT / "probe_model.pt"

# track 单迭代分解的分辨率档位
RES_LEVELS = [(320, 240), (224, 168), (160, 120), (128, 96)]
# Replica 600×340 下的等比档位（保持宽高比 30:17）
RES_REPLICA = [(300, 170), (240, 136), (160, 91), (120, 68)]


# ------------------------------------------------------------------ 数据
def load_replica(frames: int) -> tuple:
    """Replica office0 前 frames 帧 @600×340（frame_scaled(0.5)），返回
    (frames列表, K, 真值 w2c 位姿 (n,4,4))。"""
    seq = ReplicaSequence.from_dir(REPLICA_ROOT, "office0", max_frames=frames)
    frames_l = [seq.frame_scaled(t, 0.5) for t in range(frames)]
    return frames_l, seq.K, seq.poses_w2c[:frames]


def load_synth(frames: int) -> tuple:
    """合成序列（640×480）对照（机制验证用）。"""
    d = np.load(OUT / "synth_scene_480.npz")
    n = min(frames, d["rgb"].shape[0])
    return [SyncedFrame(rgb=d["rgb"][i], depth=d["depth"][i], K=d["K"],
                        stamp=float(i)) for i in range(n)], d["K"], d["poses"][:n]


def build_probe_model(frames_l: list, K, rebuild: bool = False,
                      data: str = "replica") -> GaussianModel:
    """确定性高斯模型（build_map + enforce_capacity 200k），带缓存。

    注意：Replica office0 离线真值建图新增慢（prune 后 ~45k 高斯），
    不代表消融的 200k 负载；合成序列（宽 FOV 覆盖变化大）能长到 200k+，
    per-iter 分解用合成序列模型测真实光栅化缩放。
    """
    cache = OUT / f"probe_model_{data}.pt"
    def _vars_to_cuda(vars_d: dict) -> dict:
        out = {}
        for k, v in vars_d.items():
            out[k] = v.cuda() if torch.is_tensor(v) else v
        return out

    if cache.exists() and not rebuild:
        s = torch.load(cache, map_location="cpu")
        m = GaussianModel({k: torch.nn.Parameter(v.cuda()) for k, v in s["params"].items()},
                          _vars_to_cuda(s["variables"]))
        print(f"[模型] 加载缓存 {cache.name}：{m.num_gaussians} 高斯")
        return m
    print(f"[模型] 构建 200k 高斯（{len(frames_l)} 帧 {data}，keyframe_every=5, "
          f"map_iters=20）…")
    t0 = time.perf_counter()
    n = len(frames_l)
    rgb = np.stack([f.rgb for f in frames_l])
    depth = np.stack([f.depth for f in frames_l])
    poses = np.stack([np.eye(4) for _ in range(n)])     # 占位（build_map 用真值位姿）
    model = build_map(rgb, depth, K, poses, keyframe_every=5, map_iters=20)
    enforce_capacity(model, 200_000)
    torch.save({"params": {k: v.detach().cpu() for k, v in model.params.items()},
                "variables": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                              for k, v in model.variables.items()}},
               cache)
    print(f"[模型] 构建完成 {model.num_gaussians} 高斯，耗时 {time.perf_counter()-t0:.0f}s，"
          f"已缓存 {cache.name}")
    return model


# ------------------------------------------------------------------ per-iter 分解
def _iter_timing(model: GaussianModel, fd: SyncedFrame, T, W: int, H: int,
                 needs_depth: bool, n_reps: int = 10) -> dict:
    """track 风格单迭代完整耗时（前向+loss+backward+Adam，与 track 同构）。"""
    rgb_t, depth_t, K = _to_cuda(fd)
    Tb = torch.as_tensor(np.asarray(T, dtype=np.float32)).cuda().float()
    d_rot = torch.zeros(3, device="cuda", requires_grad=True)
    d_tr = torch.zeros(3, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [d_rot], "lr": 2e-3},
        {"params": [d_tr], "lr": 1e-2},
    ])
    depth_mask = (depth_t > 0).detach()

    def one_iter():
        T_cur = se3_exp(torch.cat([d_rot, d_tr])) @ Tb
        im, depth, sil, _, _ = render(model, T_cur, K, W, H,
                                      gaussians_grad=False, camera_grad=True,
                                      needs_depth=needs_depth)
        mask = depth_mask & ((sil > 0.5).unsqueeze(0) if sil is not None
                             else torch.ones_like(depth_mask, dtype=torch.bool))
        mask = mask.detach()
        l = torch.abs(im - rgb_t)[mask.expand_as(im)].mean() \
            if mask.any() else torch.abs(im - rgb_t).mean()
        if sil is not None and mask.any():
            l = l + torch.abs(depth - depth_t)[mask].mean()
        opt.zero_grad()
        l.backward()
        opt.step()

    for _ in range(2):                                # 预热（光栅化冷启动 ~2s）
        one_iter()
    t = []
    for _ in range(n_reps):
        _, ms = profiled(one_iter)
        t.append(ms)
    return {"iter_total": float(np.median(t))}


def _to_cuda(frame: SyncedFrame):
    rgb_t = torch.from_numpy(frame.rgb.astype(np.float32) / 255.0).cuda().permute(2, 0, 1).contiguous()
    depth_t = torch.from_numpy(frame.depth.astype(np.float32)).cuda().unsqueeze(0).contiguous()
    return rgb_t, depth_t, torch.as_tensor(frame.K, dtype=torch.float32)


def _fwd_timing(model: GaussianModel, fd: SyncedFrame, T, W: int, H: int,
                needs_depth: bool, n_reps: int = 10) -> float:
    """no_grad 前向耗时（RGB 趟 / RGB+depth 趟）。"""
    rgb_t, depth_t, K = _to_cuda(fd)
    Tt = torch.as_tensor(np.asarray(T, dtype=np.float32)).cuda().float()
    with torch.no_grad():
        for _ in range(2):
            render(model, Tt, K, W, H, gaussians_grad=False, camera_grad=False,
                   needs_depth=needs_depth)
        t = []
        for _ in range(n_reps):
            _, ms = profiled(render, model, Tt, K, W, H,
                             gaussians_grad=False, camera_grad=False,
                             needs_depth=needs_depth)
            t.append(ms)
    return float(np.median(t))


def _bwd_timing(model: GaussianModel, fd: SyncedFrame, T, W: int, H: int,
                n_reps: int = 10) -> float:
    """带梯度前向 + backward 耗时（RGB+depth 全量，与 track 反传同构）。"""
    rgb_t, depth_t, K = _to_cuda(fd)
    Tt = torch.as_tensor(np.asarray(T, dtype=np.float32)).cuda().float()
    depth_mask = (depth_t > 0).detach()
    for _ in range(2):
        im, depth, sil, _, _ = render(model, Tt, K, W, H,
                                      gaussians_grad=True, camera_grad=True)
        (torch.abs(im - rgb_t).sum()
         + torch.abs(depth - depth_t)[depth_mask].sum()).backward()
        for p in model.params.values():
            p.grad = None
    t = []
    for _ in range(n_reps):
        im, depth, sil, _, _ = render(model, Tt, K, W, H,
                                      gaussians_grad=True, camera_grad=True)
        _, ms = profiled(lambda: (torch.abs(im - rgb_t).sum()
                                  + torch.abs(depth - depth_t)[depth_mask].sum()).backward())
        t.append(ms)
        for p in model.params.values():
            p.grad = None
    return float(np.median(t))


def run_per_iter(model: GaussianModel, frames_l: list, K, args) -> int:
    """Step 1：分辨率 × needs_depth 的单迭代分解表。"""
    f = frames_l[min(5, len(frames_l) - 1)]
    T = np.eye(4, dtype=np.float64)
    print(f"[数据] 帧分辨率 {f.rgb.shape[1]}x{f.rgb.shape[0]}, K00={f.K[0,0]:.1f}")
    rows = []
    for W, H in RES_LEVELS:
        if W > f.rgb.shape[1] or H > f.rgb.shape[0]:
            continue
        fd = downsample_frame(f, W, H)
        t0 = time.perf_counter()
        rgb_t, depth_t, K_t = _to_cuda(fd)
        prep_ms = (time.perf_counter() - t0) * 1e3
        full = _iter_timing(model, fd, T, W, H, needs_depth=True, n_reps=args.reps)
        rgb_only = _iter_timing(model, fd, T, W, H, needs_depth=False, n_reps=args.reps)
        fwd_full = _fwd_timing(model, fd, T, W, H, needs_depth=True, n_reps=args.reps)
        fwd_rgb = _fwd_timing(model, fd, T, W, H, needs_depth=False, n_reps=args.reps)
        bwd = _bwd_timing(model, fd, T, W, H, n_reps=args.reps)
        row = {"res": f"{W}x{H}", "prep": prep_ms, "fwd_rgb": fwd_rgb,
               "fwd_delta": max(fwd_full - fwd_rgb, 0.0), "bwd": bwd,
               "iter_full": full["iter_total"], "iter_rgb": rgb_only["iter_total"]}
        rows.append(row)
        print(f"  {row['res']:8s} prep {row['prep']:6.1f}ms | "
              f"fwd RGB {row['fwd_rgb']:6.1f} + depth {row['fwd_delta']:6.1f} | "
              f"bwd {row['bwd']:6.1f} | 迭代 full {row['iter_full']:6.1f}ms "
              f"(rgb-only {row['iter_rgb']:6.1f})")
    # 像素相关/固定开销拟合：iter_full ≈ a·pixels + b
    px = np.array([r["res"].split("x") for r in rows], dtype=np.float64)
    px = px[:, 0] * px[:, 1]
    y = np.array([r["iter_full"] for r in rows])
    if len(px) >= 3:
        a, b = np.polyfit(px, y, 1)
        print(f"\n[拟合] iter_full ≈ {a:.5f}ms/像素 + {b:.1f}ms 固定开销"
              f"（160×120 预估 {a*19200+b:.0f}ms、128×96 预估 {a*12288+b:.0f}ms）")
        print(f"[拟合] iter_rgb ≈ 像素缩放，固定开销在 RGB-only 下"
              f"（{np.polyfit(px, np.array([r['iter_rgb'] for r in rows]), 1)[1]:.1f}ms）")
    n_fps = {f"{W}x{H}": 1000.0 / r["iter_full"] for (W, H), r in zip(RES_LEVELS, rows)}
    print("[目标换算] 15 FPS=66.7ms、30 FPS=33.3ms；各分辨率单迭代 FPS："
          + " ".join(f"{k}:{v:.1f}" for k, v in n_fps.items()))
    return 0


# ------------------------------------------------------------------ iters 边际探针（Step 2 决策门）
def _ate_c2w(gt_poses, est_poses) -> float:
    """c2w 平移 Umeyama 对齐 RMSE（与消融脚本同口径）。"""
    p = np.linalg.inv(gt_poses)[:, :3, 3].T
    q = np.linalg.inv(est_poses)[:, :3, 3].T
    mu_p, mu_q = p.mean(1, keepdims=True), q.mean(1, keepdims=True)
    W = (p - mu_p) @ (q - mu_q).T
    U, _, Vt = np.linalg.svd(W)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    t = mu_q - R @ mu_p
    aligned = R @ p + t
    return float(np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=0))))


def run_iters(model: GaussianModel, frames_l: list, K, poses: np.ndarray, args) -> int:
    """Step 2：track 配置边际收益矩阵（决策门）。

    在线循环复刻消融（匀速外推 + KeyframeManager + map_keyframe 25 iters +
    density_r/capacity），扫描 容量上限 × 迭代数 × depth_every × 分辨率 输出
    ATE/失败帧/FPS(track GPU 中位)/PSNR(GT 位姿渲染)——定稿 15/30 FPS 配置。
    """
    from edge_3dgs_slam.slam.init import init_from_depth
    from edge_3dgs_slam.slam.mapping import map_keyframe
    from edge_3dgs_slam.utils.profiling import profiled
    from edge_3dgs_slam.utils.se3 import invert_pose, se3_log

    n = len(frames_l)
    track_res = [(int(v.split("x")[0]), int(v.split("x")[1])) for v in args.track_res]
    iters_list = [int(v) for v in args.iters_list.split(",")]
    caps = [int(v) for v in args.capacity.split(",")]
    deps = [int(v) for v in args.depth_every.split(",")]
    print(f"[iters] {n} 帧 × 分辨率 {track_res} × 迭代 {iters_list} × 容量 {caps}"
          f" × depth_every {deps}（真值位姿 ATE 口径）")
    gt_c2w = np.linalg.inv(poses)
    for cap in caps:
        for W, H in track_res:
            for it in iters_list:
                for de in deps:
                    model_i = init_from_depth(frames_l[0], poses[0], stride=2)
                    if args.fp16:
                        model_i.half_storage()
                    kfm = KeyframeManager()
                    est = np.zeros_like(poses)
                    est[0] = poses[0]
                    last_good = last_good_prev = poses[0]
                    last_kf = poses[0]
                    fails = 0
                    track_ms = []
                    for t in range(1, n):
                        if args.init_gt:
                            T_init = poses[t]      # 真值初值（隔离初值漂移因素）
                        else:
                            T_init = last_good @ np.linalg.inv(last_good_prev) @ last_good
                        fd = downsample_frame(frames_l[t], W, H)
                        T_est, ms = profiled(track, fd, model_i, T_init, iters=it,
                                             depth_every=de, sil_thres=args.sil_thres,
                                             lr=args.track_lr,
                                             adaptive_max=args.adaptive_max)
                        track_ms.append(ms)
                        T_np = T_est.detach().cpu().numpy()
                        d = se3_log(T_est @ invert_pose(torch.as_tensor(T_init, dtype=torch.float32).cuda()))
                        if float(d[:3].norm()) > np.deg2rad(45) or float(d[3:].norm()) > 1.0:
                            fails += 1
                        est[t] = T_np
                        if kfm.should_insert(T_np, last_kf):
                            map_keyframe(frames_l[t], model_i, T_np, iters=25,
                                         density_r=0.01, capacity_max=cap)
                            last_kf = T_np
                        last_good_prev, last_good = last_good, T_np
                    ate = _ate_c2w(gt_c2w, np.linalg.inv(est))
                    fps = 1000.0 / float(np.median(track_ms)) if track_ms else 0.0
                    psnr = _psnr_gt(model_i, frames_l[:n], poses[:n])
                    print(f"  cap {cap:6d} | {W}x{H} × {it}it × d{de}: "
                          f"ATE {ate*100:.2f}cm, 失败 {fails}, FPS {fps:.1f}, "
                          f"PSNR {psnr:.2f}dB, 高斯 {model_i.num_gaussians}")
    return 0


def _psnr_gt(model: GaussianModel, frames_l: list, poses: np.ndarray,
             step: int = 5) -> float:
    """GT 位姿下全分辨率渲染 PSNR（只评建图质量，隔离跟踪误差）。"""
    from edge_3dgs_slam.gaussian import render as _render
    pss = []
    with torch.no_grad():
        for t in range(0, len(frames_l), step):
            f = frames_l[t]
            rgb_t = torch.from_numpy(f.rgb.astype(np.float32) / 255).cuda().permute(2, 0, 1)
            im, _, _, _, _ = _render(model, poses[t],
                                     torch.as_tensor(f.K, dtype=torch.float32),
                                     f.rgb.shape[1], f.rgb.shape[0],
                                     gaussians_grad=False, camera_grad=False)
            mse = float(((im - rgb_t) ** 2).mean())
            pss.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    return float(np.mean(pss))


def run_map(model: GaussianModel, frames_l: list, K, poses: np.ndarray, args) -> int:
    """Step 3：map 减负探针（opt 分辨率档 × 窗口轮转 的耗时/质量对比）。"""
    from edge_3dgs_slam.slam.init import init_from_depth
    from edge_3dgs_slam.slam.mapping import map_keyframe

    n = min(len(frames_l), args.frames)
    print(f"[map] {n} 帧 map 减负探针（真值位姿建图，keyframe_every=5, iters=25）")
    for opt_res in ["native", "320x240", "240x136"]:
        kw = {}
        if opt_res != "native":
            ow, oh = map(int, opt_res.split("x"))
            kw = {"opt_W": ow, "opt_H": oh}
        model_i = init_from_depth(frames_l[0], poses[0], stride=2)
        t0 = time.perf_counter()
        n_kf = 0
        for t in range(5, n, 5):
            map_keyframe(frames_l[t], model_i, poses[t], iters=25,
                         density_r=0.01, capacity_max=200_000, **kw)
            torch.cuda.synchronize()          # map_keyframe 内部无同步，须手动
            n_kf += 1
        wall = time.perf_counter() - t0
        psnr = _psnr_gt(model_i, frames_l[:n], poses[:n])
        print(f"  opt {opt_res:8s} × 25 iters: 每关键帧 {wall*1000/n_kf:.0f}ms, "
              f"PSNR {psnr:.2f} dB, 高斯 {model_i.num_gaussians}")

    # 窗口轮转：窗口 5 帧，full 25 iters vs rotate2 10 iters（等效每帧访问 4 次）
    print("[map] 窗口轮转对比（窗口 5，opt 320×240）：")
    for tag, rotate, iters_c in [("full", False, 25), ("rotate2", True, 10)]:
        model_i = init_from_depth(frames_l[0], poses[0], stride=2)
        win = []
        t0 = time.perf_counter()
        n_kf = 0
        for t in range(5, n, 5):
            win.append((frames_l[t], poses[t]))
            if len(win) > 5:
                win = win[-5:]
            map_keyframe(frames_l[t], model_i, poses[t], iters=iters_c,
                         window=list(win), window_rotate=rotate, rotate_n=2,
                         opt_W=320, opt_H=240,
                         density_r=0.01, capacity_max=200_000)
            torch.cuda.synchronize()
            n_kf += 1
        wall = time.perf_counter() - t0
        psnr = _psnr_gt(model_i, frames_l[:n], poses[:n])
        print(f"  窗口 {tag:8s}: 每关键帧 {wall*1000/n_kf:.0f}ms, PSNR {psnr:.2f} dB, "
              f"高斯 {model_i.num_gaussians}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="per-iter", choices=["per-iter", "iters", "map"])
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--data", default="replica", choices=["replica", "synth"])
    ap.add_argument("--rebuild-model", action="store_true")
    ap.add_argument("--track-res", nargs="+", default=["160x120", "320x240"])
    ap.add_argument("--iters-list", default="2,4,8")
    ap.add_argument("--depth-every", default="1")
    ap.add_argument("--capacity", default="200000")
    ap.add_argument("--sil-thres", type=float, default=0.5)
    ap.add_argument("--track-lr", type=float, default=1e-2)
    ap.add_argument("--adaptive-max", type=int, default=None,
                    help="自适应迭代上限（基础 iters 后难帧扩展，None 关闭）")
    ap.add_argument("--init-gt", action="store_true",
                    help="用真值位姿做 track 初值（隔离初值漂移因素）")
    ap.add_argument("--fp16", action="store_true",
                    help="init 后 half_storage（与消融性能行一致）")
    ap.add_argument("--map-res", nargs="+", default=["native"])
    args = ap.parse_args()

    frames_l, K, poses = load_replica(args.frames) if args.data == "replica" \
        else load_synth(args.frames)
    model = build_probe_model(frames_l, K, rebuild=args.rebuild_model, data=args.data)
    model.float_storage()          # 探针统一 FP32（隔离存储层因素，纯测光栅化缩放）
    if args.mode == "per-iter":
        return run_per_iter(model, frames_l, K, args)
    if args.mode == "iters":
        return run_iters(model, frames_l, K, poses, args)
    if args.mode == "map":
        return run_map(model, frames_l, K, poses, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
