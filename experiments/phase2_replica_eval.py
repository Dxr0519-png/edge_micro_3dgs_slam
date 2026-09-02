#!/usr/bin/env python3
"""Phase 2 §7 Replica 真实数据验证（文档2 验证用）。

在真实 Replica 序列（office0）上跑 §4-§8 管线，验证合成序列的结论
在真实数据上成立，并与文档2§9 的"合成序列降级说明"对照：

    §4  init_from_depth  首帧反投影初始化 → 语义闭环自检（渲染 PSNR/depth RMSE）
        PSNR ≥ 15dB 是位姿语义（traj.txt c2w→w2c）正确的权威判据；
    σ×2  分辨率适用性抽查（f∈{2.0,1.0,0.5} 首帧 depth RMSE，不改源码）
    §8  build_map 离线建图 → 导出 PLY
    §8  track_one_frame 在线回放（匀速外推初值 + T_prev 时间一致性检测）
    §7  轨迹 ATE（自算 + evo 交叉验证）+ 单帧跟踪精度 + 关键帧渲染 PSNR

用法：
    python3 experiments/phase2_replica_eval.py \
        [--scene office0] [--frames 200] [--downscale 0.5] [--map-iters 30] \
        [--keyframe-every 5] [--track-iters 8] [--stride 2] [--out data/outputs/phase2_replica]
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import numpy as np
import torch

from edge_3dgs_slam.camera import SyncedFrame, backproject
from edge_3dgs_slam.dataset import ReplicaSequence
from edge_3dgs_slam.gaussian import GaussianModel, render
from edge_3dgs_slam.slam import build_map, track_one_frame
from edge_3dgs_slam.slam.init import init_from_depth

# 指标函数复用（与合成验证同一套，保证口径一致）
from phase2_slam_synthetic import evaluate_ate, pose_error, psnr, write_tum

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "third_party/SplaTAM/data/Replica"


def frame_at(rgb, depth, K, t) -> SyncedFrame:
    return SyncedFrame(rgb=rgb[t], depth=depth[t], K=K, stamp=float(t))


def init_with_scale_factor(frame: SyncedFrame, T_wc: np.ndarray, factor: float,
                           stride: int) -> GaussianModel:
    """init_from_depth 的 σ 系数变体（factor=2.0 即源码行为，不改被测代码）。

    用于验证「×2 是 stride 像素间距补偿、与分辨率无关」的假设：600x340 下
    ×1 是否已够用、×2 是否过度放大。
    """
    H, W = frame.depth.shape
    pts_world, valid = backproject(frame.depth, frame.K, T_wc)
    valid = valid & (frame.depth > 0)
    y_idx, x_idx = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    idx = y_idx.ravel(), x_idx.ravel()
    valid_s = valid[idx]
    pts = pts_world[idx][valid_s]
    colors = (frame.rgb[idx][valid_s] / 255.0).astype(np.float32)
    depth_z = frame.depth[idx][valid_s].astype(np.float64)
    focal = (frame.K[0, 0] + frame.K[1, 1]) / 2.0
    scales = factor * depth_z / focal
    model = GaussianModel.create_from_points(pts, colors, scales=scales, opacity=0.5)
    model.variables["scene_radius"] = float(np.max(depth_z)) / 3.0
    return model


def depth_rmse(depth_r: torch.Tensor, depth_gt: np.ndarray) -> float:
    """depth_r (1,H,W) 相机系 z vs GT（米），只统计 GT 有效像素。"""
    gt = torch.from_numpy(depth_gt).cuda()
    mask = gt > 0
    if not mask.any():
        return float("nan")
    return float(((depth_r.squeeze(0) - gt)[mask] ** 2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office0")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--downscale", type=float, default=0.5)
    ap.add_argument("--map-iters", type=int, default=30)
    ap.add_argument("--keyframe-every", type=int, default=5)
    ap.add_argument("--track-iters", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("data/outputs/phase2_replica"))
    args = ap.parse_args()

    t0 = time.time()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print(f"Replica 真实数据验证: {args.scene} 前 {args.frames} 帧 "
          f"@ {int(1200 * args.downscale)}x{int(680 * args.downscale)} "
          f"(downscale={args.downscale})")
    print("=" * 62)

    # ---- 加载（预取数组，与合成验证同构）----
    seq = ReplicaSequence.from_dir(args.root, args.scene, max_frames=args.frames)
    K_full = seq.K
    rgb = np.empty((args.frames, int(680 * args.downscale), int(1200 * args.downscale), 3), np.uint8)
    depth = np.empty((args.frames, int(680 * args.downscale), int(1200 * args.downscale)), np.float32)
    for t in range(args.frames):
        f = seq.frame_scaled(t, args.downscale)
        rgb[t], depth[t] = f.rgb, f.depth
    K = seq.frame_scaled(0, args.downscale).K
    poses = seq.poses_w2c[:args.frames]
    print(f"[加载] {args.frames} 帧 {rgb.shape[1]}x{rgb.shape[2]}, "
          f"K fx={K[0,0]:.2f} cx={K[0,2]:.2f}, 位姿 w2c（由 traj.txt c2w 求逆）")
    torch.cuda.synchronize()

    # ---- §4 首帧初始化 + 语义闭环自检 ----
    print("\n" + "=" * 62)
    print("§4 首帧深度反投影初始化（语义闭环自检：PSNR ≥ 15dB ⇒ 位姿语义正确）")
    print("=" * 62)
    f0 = frame_at(rgb, depth, K, 0)
    model = init_from_depth(f0, poses[0], stride=args.stride)
    rgb_t = torch.from_numpy(rgb[0].astype(np.float32) / 255).cuda().permute(2, 0, 1)
    with torch.no_grad():
        im, dep, sil, _, _ = render(model, poses[0], K, rgb.shape[2], rgb.shape[1],
                                    gaussians_grad=False, camera_grad=False)
    p0 = psnr(im, rgb_t)
    d0 = depth_rmse(dep, depth[0])
    print(f"  初始高斯数: {model.num_gaussians}")
    print(f"  首帧渲染 PSNR: {p0:.2f} dB, depth RMSE: {d0*100:.2f} cm, 非空占比: {(sil>0).float().mean().item()*100:.1f}%")
    ok_semantic = p0 >= 15.0
    print(f"  语义闭环 {'PASS ✅' if ok_semantic else 'FAIL ❌'}（<15dB = 位姿语义或数据解析错误，见检查清单）")

    # 保存首帧渲染图（GT vs 渲染并排）
    import cv2
    im_np = im.permute(1, 2, 0).cpu().numpy()
    gt_np = rgb[0].astype(np.float32) / 255.0
    side = np.hstack([gt_np, im_np])
    cv2.imwrite(str(out / "first_frame_render.png"), (np.clip(side, 0, 1) * 255)[:, :, ::-1])

    # ---- σ×2 分辨率适用性抽查（f∈{2.0,1.0,0.5}）----
    print("\n" + "=" * 62)
    print("σ 系数抽查（×2 是 stride=2 的 2px 间距补偿，与分辨率无关假设验证）")
    print("=" * 62)
    for f in (2.0, 1.0, 0.5):
        m = init_with_scale_factor(f0, poses[0], f, args.stride)
        with torch.no_grad():
            _, dep_f, _, _, _ = render(m, poses[0], K, rgb.shape[2], rgb.shape[1],
                                       gaussians_grad=False, camera_grad=False)
        print(f"  σ×{f}: 高斯数 {m.num_gaussians}, 首帧 depth RMSE {depth_rmse(dep_f, depth[0])*100:.2f} cm")

    # ---- §8 build_map 离线建图 ----
    print("\n" + "=" * 62)
    print("§8 build_map（离线建图，keyframe_every=5, map_iters=30）")
    print("=" * 62)
    t_map = time.time()
    model = build_map(rgb, depth, K, poses, keyframe_every=args.keyframe_every,
                      map_iters=args.map_iters, stride=args.stride, verbose=False)
    torch.cuda.synchronize()
    print(f"  build_map 完成: {model.num_gaussians} 高斯, 耗时 {time.time()-t_map:.1f}s")
    model.save_ply(str(out / "replica_map.ply"))
    print(f"  已导出 PLY: {out / 'replica_map.ply'}")

    # ---- §8 在线回放追踪（匀速外推 + T_prev 时间一致性检测）----
    print("\n" + "=" * 62)
    print("§8 track_one_frame 在线回放")
    print("=" * 62)
    t_tr = time.time()
    est_poses = np.zeros_like(poses)
    est_poses[0] = poses[0]
    last_good, last_good_prev = est_poses[0], est_poses[0]
    n_rejected = 0
    for t in range(1, len(rgb)):
        T_init = last_good @ np.linalg.inv(last_good_prev) @ last_good
        est = track_one_frame(frame_at(rgb, depth, K, t), model, T_init,
                              iters=args.track_iters, T_prev=last_good)
        if est is None:
            n_rejected += 1
            est_poses[t] = T_init
        else:
            est_poses[t] = est
            last_good_prev = last_good
            last_good = est
    torch.cuda.synchronize()
    print(f"  回放追踪 {len(rgb)} 帧, 拒绝 {n_rejected} 帧, 耗时 {time.time()-t_tr:.1f}s")

    # ---- §7 评估 ----
    print("\n" + "=" * 62)
    print("§7 评估")
    print("=" * 62)
    ate = evaluate_ate(poses, est_poses)
    ate60 = evaluate_ate(poses[:60], est_poses[:60])
    print(f"  全序列 ATE (自算, {len(poses)} 帧): {ate*100:.2f} cm")
    print(f"  稳定段 ATE (前 60 帧): {ate60*100:.2f} cm")
    # 单帧跟踪精度（真值初值，隔离累积）
    errs = []
    for t in range(1, min(60, len(rgb))):
        T_e = track_one_frame(frame_at(rgb, depth, K, t), model, poses[t],
                              iters=args.track_iters, T_prev=poses[t])
        if T_e is not None:
            errs.append(pose_error(T_e, poses[t]))
    e_med = np.median([e[0] for e in errs]) if errs else float("nan")
    print(f"  单帧跟踪精度（真值初值, 前 60 帧中位）: {e_med*100:.2f} cm")
    # 关键帧渲染 PSNR（GT 位姿）
    pss = []
    with torch.no_grad():
        for t in range(0, len(rgb), args.keyframe_every):
            rgb_t = torch.from_numpy(rgb[t].astype(np.float32) / 255).cuda().permute(2, 0, 1)
            im_t, _, _, _, _ = render(model, poses[t], K, rgb.shape[2], rgb.shape[1],
                                      gaussians_grad=False, camera_grad=False)
            pss.append(psnr(im_t, rgb_t))
    print(f"  关键帧渲染 PSNR (GT 位姿, 均值): {np.mean(pss):.2f} dB")
    # evo 交叉验证（用 evo_ape CLI，与文档§7 命令一致；evo≥1.3 的 metrics.ATE
    # API 已移除，脚本内 API 路径不可用——见验证报告发现清单）
    try:
        import subprocess
        gt_f, est_f = out / "gt_traj.tum", out / "est_traj.tum"
        write_tum(gt_f, poses); write_tum(est_f, est_poses)
        r = subprocess.run(
            ["evo_ape", "tum", str(gt_f), str(est_f), "-a", "-va"],
            capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            m = re.search(r"rmse\s+([\d.eE+-]+)", r.stdout)
            if m:
                print(f"  轨迹 ATE (evo_ape -a): {float(m.group(1))*100:.2f} cm")
            else:
                print(f"  evo_ape 输出未匹配 rmse：\n{r.stdout[-300:]}")
        else:
            print(f"  evo_ape 失败（rc={r.returncode}）：{r.stderr[-200:]}")
    except Exception as e:
        print(f"  evo_ape 不可用（{type(e).__name__}: {e}），仅用自算 ATE")

    # ---- 总判定 ----
    ok = ok_semantic and e_med < 0.05 and ate60 < 0.15 and n_rejected <= 10 and ate < 0.60
    print("\n" + "=" * 62)
    print(f"总耗时 {time.time()-t0:.1f}s")
    print(f"Replica {args.scene} 验证: {'全部 PASS ✅' if ok else '存在 FAIL ❌'}")
    print(f"  判据: 语义闭环 PSNR≥15dB, 单帧<5cm, 稳定段 ATE<15cm, 拒绝≤10, 全序列<60cm")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
