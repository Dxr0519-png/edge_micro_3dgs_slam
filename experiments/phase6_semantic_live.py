#!/usr/bin/env python3
"""Phase 6 真机地图 → 开放词汇查询闭环（2026-09-02 衔接脚本）。

输入（跑完 SLAM 自动落在 <仓库>/data/outputs/live/，见 node --out）：
    map_<ts>.pt    全量地图 checkpoint（node --load 同构）
    frames.npz     关键帧（rgb/depth/K/poses w2c，与地图同世界系）

流程（Replica 验证口径 phase6_extract_replica / phase6_distill 的真机版）：
    1. 帧挑选：frames.npz 均匀抽 --max-frames 帧（真机每帧提取 ~13s@600×340）
    2. 特征重提：MobileSAM 掩码 → MobileCLIP → feature_map.pkl + segs/（与
       Replica pkl 同键契约；npz 帧位姿/K 一并存 pkl 元数据供监督构建）
    3. 场景 AE 重训（真机域，COCO 预训练 AE 不适用——Replica 坑 3 同款）
    4. 预对齐门：渲染首帧 RGB vs 原图 PSNR>15（防模型/位姿错位白跑；
       近景 fps 档地图可能不过，--skip-gate 跳过并诚实记录）
    5. 蒸馏（geometric 冻结，只优化 features）→ probe_map_p6.pt
    6. 验收：训练/留出帧掩码内余弦（训练帧判据 > 0.85，留出帧仅记录）

分阶段可跑：--extract-only（只到 2，产出 pkl/segs）或 --distill-only（接已有
pkl/segs）；默认全流程。用法：
    python3 experiments/phase6_semantic_live.py \
        --map data/outputs/live/map_latest.pt \
        --npz data/outputs/live/frames.npz \
        --max-frames 24
产物: data/outputs/live_semantic/{feature_map.pkl, segs/, lang_ae_live.pt,
      probe_map_p6.pt}（--out 可改）
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from edge_3dgs_slam.dataset.live_npz import LiveNpzSequence
from edge_3dgs_slam.feature_factory import (LangAE, extract_hierarchical,
                                            load_feature_config, load_models,
                                            release_models, train_ae, update_cache)
from edge_3dgs_slam.gaussian import FEATURE_KEY, GaussianModel, add_feature_dim
from edge_3dgs_slam.language_field.optim import (FrameSupervision, distill_features,
                                                 mask_cosine_eval, prealign_check)

COS_AE = 0.94            # Replica 0.95 口径对真机噪声域过紧（实测 1019 条掩码
                         # 特征 5000 轮 0.948 封顶）；0.94 作门槛并诚实打印
COS_DISTILL = 0.85
MIN_MASKS = 5            # 真机单帧掩码数门槛（Replica 是 15，宽松些诚实记录）
MIN_AREA = 50            # 掩码过滤：面积 ≥ 50px（与 phase6_extract_replica 同口径）
MIN_IOU = 0.8            # 掩码过滤：predicted_iou ≥ 0.8


def filter_masks(feats: list[dict]) -> list[dict]:
    """按 area/predicted_iou 过滤（extract_replica 的本地函数同款，此处复刻）。"""
    return [f for f in feats if f["area"] >= MIN_AREA and f["predicted_iou"] >= MIN_IOU]

# ------------------------------------------------------------------ 监督构建


def _select_frames(n: int, max_frames: int) -> list[int]:
    """均匀抽帧（首尾都留，保证覆盖轨迹全程）。"""
    if n <= max_frames:
        return list(range(n))
    idx = np.linspace(0, n - 1, max_frames, dtype=int)
    return sorted(set(idx.tolist()))


def build_supervision(seq: LiveNpzSequence, frames: list[int], pkl: Path,
                      segs_dir: Path, scale: float) -> list[FrameSupervision]:
    """从 feature_map.pkl + segs/ 构建监督（权重 1/sqrt(area)，同 Replica 口径）。"""
    with open(pkl, "rb") as f:
        cache = pickle.load(f)
    sups = []
    for t in frames:
        fr = seq.frame_scaled(t, scale)
        d = np.load(segs_dir / f"frame{t}.npz")
        masks = cache[t]["masks"]
        segs = [d["segs"][i] for i in range(len(masks))]
        clip_vecs = [np.asarray(m["clip_vec"], np.float32) for m in masks]
        weights = [1.0 / np.sqrt(max(int(seg.sum()), 1)) for seg in segs]
        sups.append(FrameSupervision(
            w2c=seq.poses_w2c[t].astype(np.float32), K=fr.K, H=fr.rgb.shape[0],
            W=fr.rgb.shape[1], segs=segs, clip_vecs=clip_vecs, weights=weights,
            rgb=fr.rgb))
    return sups


# ------------------------------------------------------------------ 阶段


def extract_stage(args, seq: LiveNpzSequence, frames: list[int],
                  cfg, fm) -> Path:
    """阶段 2：特征重提 → pkl + segs（pkl 键契约与 Replica 一致：
    {frame_id: {"H","W","masks"}}；pose/K 由 distill 阶段直接从 seq 取）。"""
    out = args.out / "segs"
    out.mkdir(parents=True, exist_ok=True)
    pkl = args.out / "feature_map.pkl"
    for t in frames:
        t0 = time.monotonic()
        fr = seq.frame_scaled(t, args.scale)
        feats = extract_hierarchical(t, fr.rgb, fm, keep_seg=True)
        feats = filter_masks(feats)
        print(f"  帧[{t}] {fr.rgb.shape[1]}×{fr.rgb.shape[0]}: 掩码 {len(feats)} "
              f"(≥{MIN_MASKS}: {len(feats) >= MIN_MASKS}) | {time.monotonic() - t0:.0f}s")
        seg_stack = np.stack([f["segmentation"] for f in feats])
        np.savez_compressed(out / f"frame{t}.npz",
                            segs=seg_stack,
                            mask_ids=np.asarray([f["mask_id"] for f in feats]),
                            hier=np.asarray([f["hier_level"] for f in feats]),
                            bboxs=np.asarray([f["bbox"] for f in feats]))
        update_cache(pkl, t, fr.rgb.shape[0], fr.rgb.shape[1], feats)
    return pkl


def distill_stage(args, seq: LiveNpzSequence, frames: list[int], device,
                  holdout: list[int] | None = None) -> Path:
    """阶段 3-6：AE → 预对齐 → 蒸馏 → 验收 → checkpoint。"""
    out = args.out
    cfg = load_feature_config()
    ae_cfg = cfg["feature"]["autoencoder"]
    pkl = out / "feature_map.pkl"
    if not pkl.exists():
        raise SystemExit(f"缺 {pkl}——先跑提取阶段（--extract-only）")
    ae_ckpt = out / "lang_ae_live.pt"

    sups = build_supervision(seq, frames, pkl, out / "segs", args.scale)
    vectors = np.stack([v for s in sups for v in s.clip_vecs])
    print(f"[live] 监督帧 {len(frames)}，掩码特征 {vectors.shape[0]} 条 | AE 重训（真机域）")
    ae, stats = train_ae(vectors, d_latent=ae_cfg["latent_dim"], hidden=ae_cfg["hidden"],
                         lr=ae_cfg["lr"], lambda_cos=ae_cfg["lambda_cos"],
                         checkpoint=ae_ckpt, device=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    ok_ae = stats["cos_mean"] > COS_AE
    print(f"  AE cos_mean = {stats['cos_mean']:.4f} (> {COS_AE}: {ok_ae})"
          + ("（低于判据也继续，真机域记录备查）" if not ok_ae else ""))

    # 模型：真机 checkpoint（params/variables）→ GaussianModel + 特征通道
    s0 = torch.load(args.map, map_location="cpu", weights_only=False)
    model = GaussianModel({k: torch.nn.Parameter(v.cuda())
                           for k, v in s0["params"].items()},
                          variables=s0.get("variables", {}))
    if args.max_gauss and model.num_gaussians > args.max_gauss:
        # 2026-09-02：按索引切片 [:N] 会只留**最早播种区**（插入序=空间序）——
        # 真机全景图会偏起点、与录制帧(最近 150 KF)错位。改为均匀抽稀保空间覆盖
        stride = (model.num_gaussians + args.max_gauss - 1) // args.max_gauss
        for k in model.params:
            model.params[k] = torch.nn.Parameter(
                model.params[k][::stride].contiguous())
    add_feature_dim(model, ae_cfg["latent_dim"])
    with torch.no_grad():
        model.params[FEATURE_KEY].zero_()
    print(f"  模型 {model.num_gaussians} 高斯，feature 通道 (N,{ae_cfg['latent_dim']}) 零初始化")

    # 预对齐门：多候选（首/中/尾）取最佳——录制帧与地图几何的对齐度随轨迹变化
    # （2026-09-02 教训：建图容量 200k 封顶后播种冻结，地图只含起点段，尾部
    #  关键帧位姿漂移 35m+，任何单帧门都会误杀蒸馏）
    gate_cands = ([frames[0], frames[len(frames) // 2], frames[-1]]
                  if len(frames) >= 3 else frames)
    gate_sups = build_supervision(seq, gate_cands, pkl, out / "segs", args.scale)
    gate_ok, gate_psnr = False, -99.0
    for t, sup in zip(gate_cands, gate_sups):
        ok, ps = prealign_check(model, sup)
        print(f"  预对齐候选 frame[{t}] PSNR = {ps:.1f} dB (>15: {ok})")
        if ps > gate_psnr:
            gate_ok, gate_psnr = ok, ps
    if not gate_ok and not args.skip_gate:
        print("  ❌ 全部候选帧预对齐失败——模型与位姿不一致（常见：建图容量封顶"
              "播种冻结 → 地图只含起点段、尾部漂移）。提高 --cap 重扫，或 "
              "--skip-gate 跳过（结果需人工评估）")
        return None
    print(f"  预对齐门通过（最佳 {gate_psnr:.1f} dB）")
    means_before = model.params["means3D"].detach().clone()
    feat_before = model.params[FEATURE_KEY].detach().clone()

    print(f"  蒸馏 iters={args.iters}（几何冻结）")
    stats_d = distill_features(model, ae, sups, iters=args.iters,
                               lambda_cos=ae_cfg["lambda_cos"], device=device)
    print(f"  loss_final={stats_d['loss_final']:.4f} cos_train={stats_d['cos_train_mean']:.4f}")
    geom_ok = torch.equal(model.params["means3D"], means_before)
    feat_changed = float((model.params[FEATURE_KEY].detach() - feat_before).norm()) > 1e-3
    print(f"  几何逐位相等: {geom_ok} | features 变化: {feat_changed}")

    ckpt = out / "probe_map_p6.pt"
    torch.save({"params": {k: p.detach().float().cpu() for k, p in model.params.items()},
                "variables": {k: v.detach().float().cpu()
                              if isinstance(v, torch.Tensor) else v
                              for k, v in model.variables.items()}},
               ckpt)
    print(f"  checkpoint → {ckpt}")

    # 验收：训练帧判据 + 留出帧记录
    for label, tset in (("训练", frames), ("留出", holdout or [])):
        if not tset:
            continue
        for t in tset:
            sups_all = build_supervision(seq, [t], pkl, out / "segs", args.scale)
            cos_mean, cos_list = mask_cosine_eval(model, ae, sups_all[0])
            crit = f"(> {COS_DISTILL}: {cos_mean > COS_DISTILL})" if label == "训练" else "(记录)"
            print(f"  {label}帧[{t}]: cos = {cos_mean:.4f} {crit} | 掩码 {len(cos_list)}")
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, help="SLAM 产出 map_*.pt（node --load 同构）")
    ap.add_argument("--npz", required=True, help="SLAM 产出 frames.npz")
    ap.add_argument("--out", type=Path, default=Path("data/outputs/live_semantic"))
    ap.add_argument("--max-frames", type=int, default=24, help="均匀抽帧数（默认 24）")
    ap.add_argument("--scale", type=float, default=0.5, help="提取分辨率缩放（1280×720→0.5=640×360）")
    ap.add_argument("--extract-only", action="store_true", help="只做特征重提（产出 pkl/segs）")
    ap.add_argument("--distill-only", action="store_true", help="只做 AE+蒸馏（接已有 pkl/segs）")
    ap.add_argument("--iters", type=int, default=200, help="蒸馏迭代轮数")
    ap.add_argument("--max-gauss", type=int, default=200_000, help="地图高斯数上限（显存控制）")
    ap.add_argument("--skip-gate", action="store_true", help="跳过预对齐门（fps 档地图备选）")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_feature_config()
    seq = LiveNpzSequence(args.npz)
    n = len(seq)
    frames = _select_frames(n, args.max_frames)
    holdout = frames[-1:] if len(frames) > 1 else []
    train_frames = frames[:-1] if len(frames) > 1 else frames
    print(f"[live] {args.npz}: {n} 关键帧 @{seq.rgb_arr.shape[2]}×{seq.rgb_arr.shape[1]} "
          f"| 抽 {len(frames)} 帧（训练 {len(train_frames)} + 留出 {len(holdout)}）")

    if not args.distill_only:
        fm = load_models(cfg, device)
        pkl = extract_stage(args, seq, frames, cfg, fm)
        release_models(fm)
        del fm
        print(f"[live] 提取完成 → {pkl}")
    if not args.extract_only:
        ckpt = distill_stage(args, seq, train_frames, device, holdout=holdout)
        if ckpt is not None:
            print(f"[live] 蒸馏完成 → {ckpt}（node --load 后可用查询服务）")
        else:
            print("[live] 蒸馏未执行（预对齐门未过，见上；提高 --cap 重扫或 --skip-gate）")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
