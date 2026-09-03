#!/usr/bin/env python3
"""离线批量 3DGS 重建（2026-09-02）：录制 bag → 无实时约束的重建。

流程（对 data/outputs/offline 分块 npz）：
  1. 首帧 init_from_depth 初始化模型
  2. 逐帧跟踪（每 2 帧精跟踪：15 迭代 + IMU 旋转先验 + ICP 初值——不丢帧重活）
  3. 关键帧（>5cm/2°）加入窗口；第一轮：随走随建（add_new + 窗口 25 迭代）
  4. 第二轮：全 KF 窗口抛光（80 迭代×2 轮次 + 剪枝）→ opacity 实心化
  5. 输出 checkpoint + 关键帧 npz + 多视角渲染图

用法:
    python3 experiments/offline_rebuild.py data/outputs/offline data/outputs/offline_out
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch

from edge_3dgs_slam.camera import SyncedFrame
from edge_3dgs_slam.gaussian.model import GaussianModel
from edge_3dgs_slam.gaussian.render import render
from edge_3dgs_slam.slam import init_from_depth
from edge_3dgs_slam.slam.icp_init import icp_init_model, icp_render_model_depth, se3_motion_np
from edge_3dgs_slam.slam.imu_prior import GyroRotationPrior
from edge_3dgs_slam.slam.mapping import map_keyframe
from edge_3dgs_slam.slam.tracking import track
from edge_3dgs_slam.utils.frame_utils import downsample_frame

TRACK_W, TRACK_H = 320, 240     # 跟踪渲染分辨率（质量档）
OPT_W, OPT_H = 320, 240         # 建图优化分辨率
SEED_CAP = 800_000
KF_DT, KF_DROT = 0.05, 2.0      # 关键帧阈值（离线更密）


class ChunkedSeq:
    """分块 npz 序列（逐块驻留防 OOM：同一时刻只持有一块解压数组）。"""

    def __init__(self, root: Path):
        self.chunks = sorted(root.glob("frames_*.npz"))
        self.ranges = []                     # (start, end) 每块帧区间
        n = 0
        for c in self.chunks:
            d = np.load(c)
            m = d["rgb"].shape[0]
            self.ranges.append((n, n + m))
            n += m
            d.close()
        self.n = n
        self.K = np.load(self.chunks[0])["K"]
        self._cur = -1
        self._rgb = None
        self._dep = None

    def _load(self, ci: int):
        if ci != self._cur:
            self._rgb = self._dep = None
            d = np.load(self.chunks[ci])
            self._rgb = d["rgb"]
            self._dep = d["depth"]
            self._cur = ci

    def frame(self, i: int) -> SyncedFrame:
        for ci, (s, e) in enumerate(self.ranges):
            if s <= i < e:
                self._load(ci)
                j = i - s
                return SyncedFrame(
                    rgb=self._rgb[j], depth=self._dep[j].astype(np.float32) / 1000.0,
                    K=self.K, stamp=float(i))
        raise IndexError


def op_stats(model, tag):
    op = torch.sigmoid(model.params["logit_opacities"].detach().float()).cpu().numpy()[:, 0]
    print(f"{tag}: 高斯 {model.num_gaussians} | opacity 中位 {np.median(op):.2f} "
          f"| >0.9 占 {(op > 0.9).mean() * 100:.0f}%", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/outputs/offline_out"))
    ap.add_argument("--track-every", type=int, default=2)
    ap.add_argument("--polish", action="store_true", help="跑全 KF 窗口抛光（默认跳过，walk 即存）")
    ap.add_argument("--no-imu", action="store_true", help="不用 IMU 旋转先验（纯视觉恒速/保持初值）")
    ap.add_argument("--polish-iters", type=int, default=80)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seq = ChunkedSeq(args.inp)
    imu_f = args.inp / "imu.npz"
    imu_list = [tuple(x) for x in np.load(imu_f, allow_pickle=True)["imu"]] \
        if imu_f.exists() else []
    prior = GyroRotationPrior([])                    # 复用积分逻辑，手动喂窗口

    t_start = (imu_list[0][0] - 1.0) if imu_list else 0.0   # 帧时间近似；无 IMU（--no-imu 录制）时以 0 起算
    fps = 29.97
    def frame_stamp(i):
        return t_start + i / fps

    # ---- 1. 首帧初始化 ----
    fr0 = seq.frame(0)
    model = init_from_depth(fr0, np.eye(4), stride=2)
    print(f"init: {model.num_gaussians} 高斯 @{fr0.rgb.shape[1]}x{fr0.rgb.shape[0]}", flush=True)

    T_last = np.eye(4)
    kfs = []                                        # (frame_idx, SyncedFrame(小), T)
    last_kf_T = None
    i_prev = None
    window = []
    t_loop = time.time()

    # ---- 2+3. 逐帧跟踪 + 随走随建 ----
    for i in range(0, seq.n, args.track_every):
        t = frame_stamp(i)
        fr = seq.frame(i)
        fd = downsample_frame(fr, TRACK_W, TRACK_H)
        # IMU 旋转先验（喂入至 t 的样本）
        while imu_list and imu_list[0][0] < t - 0.1:
            imu_list.pop(0)
        keep = [x for x in imu_list if x[0] <= t + 0.1]
        if not keep:
            prior.buf.clear()
        else:
            prior.buf.clear()
            for x in keep:
                prior.buf.append(x)
        T_imu = None
        if not args.no_imu:
            T_imu = prior.init_pose(T_last, frame_stamp(i_prev) if i_prev is not None else t - 0.033,
                                    t, T_last) if i_prev is not None else None
        T_init = T_imu if T_imu is not None else T_last
        # ICP 帧到模型（模型成熟后）
        if model.num_gaussians > 80_000 and i_prev is not None:
            try:
                with torch.no_grad():
                    prep = None
                    d_mod = icp_render_model_depth(model, T_last, fd.K, W=160, H=120)
                T_icp, st = icp_init_model(fr, d_mod, T_last, T_init, seq.K,
                                           W=160, H=120, return_stats=True)
                if not st["fallback"]:
                    T_init = T_icp
            except Exception:
                pass
        T_cuda = track(fd, model, T_init, iters=15, depth_every=1, adaptive_max=25,
                       cull=False, fail_rot_deg=8.0, fail_trans_m=0.3)
        T = T_cuda.detach().cpu().numpy()
        # 失败门：与初值差过大回退初值
        d = se3_motion_np(T_init, T)
        if np.linalg.norm(d[:3]) > np.deg2rad(8):
            T = T_init
        T_last, i_prev = T, i
        if i % 100 == 0:
            print(f"[track] 帧 {i}/{seq.n} 高斯 {model.num_gaussians} "
                  f"| {time.time() - t_loop:.0f}s | KF {len(kfs)}", flush=True)
        # 关键帧判定
        if last_kf_T is None or (np.linalg.norm(T[:3, 3] - last_kf_T[:3, 3]) > KF_DT
                                 or np.degrees(np.arccos(np.clip(
                                     (np.trace(T[:3, :3] @ last_kf_T[:3, :3].T) - 1) / 2,
                                     -1, 1))) > KF_DROT):
            last_kf_T = T
            fk = downsample_frame(fr, 640, 360) if fr.rgb.shape[1] > 640 else fr
            kfs.append((i, fk, T, t))
            window.append((fk, T))
            window = window[-10:]
            # 随走随建：add_new + 窗口优化 25 迭代
            map_keyframe(fk, model, T, iters=25, add_new=True, prune=True,
                         capacity_max=SEED_CAP, window_rotate=True, rotate_n=3,
                         opt_W=OPT_W, opt_H=OPT_H, depth_every=1,
                         window=list(window), max_new=12000,
                         seed_W=640, seed_H=360,
                         seed_opacity=0.75, seed_scale_factor=1.0,
                         prune_opacity=0.03)
            torch.cuda.synchronize()
            if len(kfs) % 10 == 0:
                torch.cuda.empty_cache()
            if len(kfs) % 20 == 0:
                op_stats(model, f"[map] KF {len(kfs)}")
    op_stats(model, "随走随建完成")
    print(f"关键帧 {len(kfs)}，耗时 {time.time() - t_loop:.0f}s", flush=True)

    # ---- 4. 输出 walk 版（先存，抛光可选；2026-09-02 抛光太慢默认跳过）----
    def _save():
        torch.save({"params": {k: p.detach().float().cpu() for k, p in model.params.items()},
                    "variables": {k: v.detach().float().cpu() if isinstance(v, torch.Tensor)
                                  else v for k, v in model.variables.items()}},
                   args.out / "map_offline.pt")
        np.savez_compressed(args.out / "kfs.npz",
                            rgb=np.stack([k[1].rgb for k in kfs]),
                            depth=np.stack([k[1].depth for k in kfs]),
                            poses=np.stack([k[2] for k in kfs]), K=seq.K)
    _save()
    print(f"[walk] 已存 {model.num_gaussians} 高斯 + {len(kfs)} KF → {args.out}，"
          f"耗时 {time.time() - t_loop:.0f}s", flush=True)
    if not args.polish:
        return 0

    # ---- 5. 全 KF 窗口抛光（2 轮，可选 --polish）----
    for rnd in range(2):
        for j in range(len(kfs)):
            _, fk, T, _ = kfs[j]
            w = kfs[max(0, j - 10):j + 1]
            window = [(f, Tf) for _, f, Tf, _ in w]
            map_keyframe(fk, model, T, iters=args.polish_iters, add_new=False,
                         prune=(rnd == 1), capacity_max=SEED_CAP,
                         window_rotate=True, rotate_n=2,
                         opt_W=OPT_W, opt_H=OPT_H, depth_every=1, window=window)
            torch.cuda.synchronize()
            if j % 30 == 0:
                print(f"[polish] 轮 {rnd} KF {j}/{len(kfs)} | {time.time() - t_loop:.0f}s",
                      flush=True)
    op_stats(model, "抛光完成")
    _save()
    print(f"完成 → {args.out}（{model.num_gaussians} 高斯，{len(kfs)} KF，"
          f"总耗时 {time.time() - t_loop:.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
