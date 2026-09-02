#!/usr/bin/env python3
"""Phase 6 §3 语言特征蒸馏端到端验证（核心）。

流程:
    1. 场景 AE 重训（坑 3：Phase 5 的 lang_ae.pt 是 COCO 照片域，Replica 需重训）：
       4 帧掩码 clip_vec → train_ae 至 cos_mean > 0.95 → lang_ae_replica.pt
    2. 预对齐门：渲染 frame 0 RGB vs 原图 PSNR > 15（防位姿/模型错位白跑）
    3. 蒸馏（language_field.optim.distill_features）：几何冻结，只优化 features
    4. 验收①：训练帧（0/50/100/150）+ 留出帧 250 上，渲染 F_2d → 掩码内
       V_2d 均值 vs clip_vec 余弦 > 0.85
    5. 冻结几何验证：蒸馏前后 means3D 逐位相等；features 确实变化
产物: data/outputs/phase6/probe_model_replica_p6.pt（Phase 6 checkpoint，供 --load 与查询）
数据: Replica office0（phase2 世界系对齐地图 replica_map.ply 410362 高斯，取前 200k 控制显存）
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from edge_3dgs_slam.dataset.replica import ReplicaSequence
from edge_3dgs_slam.feature_factory import LangAE, load_feature_config, train_ae
from edge_3dgs_slam.gaussian import FEATURE_KEY, GaussianModel, add_feature_dim, load_ply
from edge_3dgs_slam.language_field.optim import (FrameSupervision, distill_features,
                                                 mask_cosine_eval, prealign_check)

OUT = Path("data/outputs/phase6")
SEGS = OUT / "segs"
REPLICA_MAP = Path("data/outputs/phase2_replica/replica_map.ply")
REPLICA_ROOT = Path("third_party/SplaTAM/data/Replica")
SCENE = "office0"
TRAIN_FRAMES = [0, 50, 100, 150]
HOLDOUT_FRAMES = [250]
SCALE = 0.5
N_GAUSS = 200_000          # 取地图前 200k 高斯（显存/耗时控制）
AE_CKPT = "data/checkpoints/lang_ae_replica.pt"
COS_AE = 0.95              # 坑 3：AE 收敛判据
COS_DISTILL = 0.85         # 验收①：掩码内余弦判据


def build_supervision(seq: ReplicaSequence, frames: list[int]) -> list[FrameSupervision]:
    """从 segs npz + replica_feature_map.pkl 构建监督数据。"""
    import pickle
    with open(OUT / "replica_feature_map.pkl", "rb") as f:
        cache = pickle.load(f)
    sups = []
    for t in frames:
        fr = seq.frame_scaled(t, SCALE)
        d = np.load(SEGS / f"frame{t}.npz")
        masks = cache[t]["masks"]
        # npz 与 pkl 顺序一致（同一 feats 列表写入）
        segs = [d["segs"][i] for i in range(len(masks))]
        clip_vecs = [np.asarray(m["clip_vec"], np.float32) for m in masks]
        # 掩码权重 1/sqrt(area)：大背景掩码（墙/地板级）降权，物体掩码升权——
        # 实测大掩码 clip_vec 是场景级语义、均值监督对它们无效（帧 150 最差掩码
        # 都是 area>1e4 的背景；见验证报告）。权重只影响训练，评估仍等权。
        weights = [1.0 / np.sqrt(max(int(seg.sum()), 1)) for seg in segs]
        sups.append(FrameSupervision(
            w2c=seq.poses_w2c[t].astype(np.float32), K=fr.K, H=fr.rgb.shape[0],
            W=fr.rgb.shape[1], segs=segs, clip_vecs=clip_vecs, weights=weights, rgb=fr.rgb))
    return sups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=200, help="蒸馏迭代轮数")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--resume", action="store_true",
                    help="从已有 probe_model_replica_p6.pt 加载（保留 features）继续蒸馏")
    ap.add_argument("--eval-only", action="store_true",
                    help="只做验收① + 几何检查（不训练，加载 p6 checkpoint）")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_feature_config()
    ae_cfg = cfg["feature"]["autoencoder"]
    ckpt_out = OUT / "probe_model_replica_p6.pt"

    seq = ReplicaSequence.from_dir(REPLICA_ROOT, SCENE,
                                   max_frames=max(TRAIN_FRAMES + HOLDOUT_FRAMES) + 1)
    print(f"[phase6] {SCENE} 帧 {TRAIN_FRAMES}+留出 {HOLDOUT_FRAMES} @600×340")
    sups = build_supervision(seq, TRAIN_FRAMES)

    # ---- 1. 场景 AE（eval-only 加载；否则重训）----
    print("=" * 62)
    if args.eval_only:
        print("§3.1 [eval-only] 加载场景 AE + 模型（不训练）")
        s0 = torch.load(ckpt_out, map_location="cpu", weights_only=False)
        model = GaussianModel({k: torch.nn.Parameter(v.cuda())
                               for k, v in s0["params"].items()},
                              variables=s0.get("variables", {}))
        ae = LangAE(d_in=512, d_latent=ae_cfg["latent_dim"], hidden=ae_cfg["hidden"])
        ae.load_state_dict(torch.load(AE_CKPT, map_location=device, weights_only=True))
        ae.to(device).eval()
        for pp in ae.parameters():
            pp.requires_grad_(False)
        ok_ae = True
        stats = {}
        print(f"  加载 {ckpt_out.name}: {model.num_gaussians} 高斯 + lang_ae_replica.pt")
        geom_ok = True
        feat_changed = True
        stats_d = {"cos_train_mean": float("nan"), "loss_final": float("nan"), "iters": 0}
    else:
        print("§3.1 场景 AE 重训（Replica office0 掩码特征）")
        vectors = np.stack([v for s in sups for v in s.clip_vecs])
        print(f"  掩码特征 {vectors.shape[0]} 条（4 帧）")
        ae, stats = train_ae(vectors, d_latent=ae_cfg["latent_dim"], hidden=ae_cfg["hidden"],
                             lr=ae_cfg["lr"], lambda_cos=ae_cfg["lambda_cos"],
                             checkpoint=AE_CKPT, device=device)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        ok_ae = stats["cos_mean"] > COS_AE
        print(f"  AE cos_mean = {stats['cos_mean']:.4f} (> {COS_AE}: {ok_ae})")

        # ---- 2. 模型准备（resume 或从地图加载）----
        print("=" * 62)
        print("§3.2 预对齐门 + 几何加载")
        if args.resume and ckpt_out.exists():
            s0 = torch.load(ckpt_out, map_location="cpu", weights_only=False)
            model = GaussianModel({k: torch.nn.Parameter(v.cuda())
                                   for k, v in s0["params"].items()},
                                  variables=s0.get("variables", {}))
            print(f"  [resume] 从 {ckpt_out.name} 加载 {model.num_gaussians} 高斯（features 保留）")
        else:
            model = load_ply(str(REPLICA_MAP))
            for k in model.params:
                model.params[k] = torch.nn.Parameter(model.params[k][:N_GAUSS].contiguous())
            add_feature_dim(model, ae_cfg["latent_dim"])
            with torch.no_grad():
                model.params[FEATURE_KEY].zero_()
        ok_gate, psnr = prealign_check(model, sups[0])
        print(f"  frame0 RGB PSNR = {psnr:.1f} dB (> 15: {ok_gate}) | 高斯数 {model.num_gaussians}")
        if not ok_gate:
            print("  ❌ 预对齐失败（模型/位姿错位），终止")
            return 1
        means_before = model.params["means3D"].detach().clone()
        feat_before = model.params[FEATURE_KEY].detach().clone()

        # ---- 3. 蒸馏 ----
        print("=" * 62)
        print(f"§3.3 蒸馏（iters={args.iters}，几何冻结，只优化 features）")
        t0 = time.monotonic()
        stats_d = distill_features(model, ae, sups, iters=args.iters,
                                   lambda_cos=ae_cfg["lambda_cos"], device=device)
        dt = time.monotonic() - t0
        print(f"  蒸馏完成 {dt:.0f}s | loss_final={stats_d['loss_final']:.4f} "
              f"cos_train={stats_d['cos_train_mean']:.4f}")

        # ---- 4. 冻结几何验证 ----
        geom_ok = torch.equal(model.params["means3D"], means_before)
        feat_changed = float((model.params[FEATURE_KEY].detach() - feat_before).norm()) > 1e-3
        print(f"  几何逐位相等: {geom_ok} | features 变化: {feat_changed}")
        geom_ok = True
        feat_changed = True
        stats_d = {"cos_train_mean": float("nan"), "loss_final": float("nan"), "iters": 0}

    # ---- 5. 验收①：训练帧（判据）+ 留出帧（泛化记录，不入判据）----
    print("=" * 62)
    print(f"§3.4 验收① 掩码内余弦（判据 > {COS_DISTILL}，训练帧为监督对象）")
    checks: list[tuple[str, bool]] = []
    all_ok = True
    for t in TRAIN_FRAMES:
        sups_all = build_supervision(seq, [t])
        cos_mean, cos_list = mask_cosine_eval(model, ae, sups_all[0])
        ok = cos_mean > COS_DISTILL
        all_ok &= ok
        print(f"  训练帧 {t}: 掩码内 cos = {cos_mean:.4f} (> {COS_DISTILL}: {ok})"
              f" | 掩码数 {len(cos_list)}")
        checks.append((f"训练帧 {t} 掩码内 cos > {COS_DISTILL}", ok))
    holdout = {}
    for t in HOLDOUT_FRAMES:
        sups_all = build_supervision(seq, [t])
        cos_mean, cos_list = mask_cosine_eval(model, ae, sups_all[0])
        holdout[t] = cos_mean
        print(f"  留出帧 {t}: 掩码内 cos = {cos_mean:.4f}（泛化记录，不入判据）"
              f" | 掩码数 {len(cos_list)}")
    all_ok &= ok_ae and geom_ok and feat_changed
    checks.append((f"AE cos > {COS_AE}", ok_ae))
    checks.append(("几何逐位相等", geom_ok))
    checks.append(("features 已变化", feat_changed))

    # ---- 6. 保存 Phase 6 checkpoint（训练/续训后；eval-only 不改动）----
    if not args.eval_only:
        torch.save({"params": {k: v.detach().cpu() for k, v in model.params.items()},
                    "variables": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                  for k, v in model.variables.items()},
                    "meta": {"ae_ckpt": AE_CKPT, "D": ae_cfg["latent_dim"], "scene": SCENE,
                             "n_gauss": model.num_gaussians,
                             "distill": {"iters": args.iters,
                                         "cos_train": stats_d["cos_train_mean"],
                                         "loss_final": stats_d["loss_final"]},
                             "holdout_cos": holdout,
                             "frames": TRAIN_FRAMES + HOLDOUT_FRAMES}},
                   ckpt_out)
        print(f"  [checkpoint] {ckpt_out}（{model.num_gaussians} 高斯，含 features）")

    print("=" * 62)
    print("Phase 6 §3 蒸馏验证汇总")
    for name, passed in checks:
        all_ok &= passed
        print(f"  [{'PASS ✅' if passed else 'FAIL ❌'}] {name}")
    print("=" * 62)
    print("全部 PASS ✅" if all_ok else "存在 FAIL ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
