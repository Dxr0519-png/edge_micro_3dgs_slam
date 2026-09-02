#!/usr/bin/env python3
"""Phase 6 §3 监督数据准备：Replica office0 特征重提（掩码级 + segmentation）。

Phase 5 的 feature_map.pkl 是 COCO 照片特征，与 Replica 场景无关；本脚本在
Replica office0 帧 0/50/100/150（@600×340，GT 位姿同世界系）重新提取分层掩码
特征，keep_seg=True 保留 segmentation（掩码级监督所需，只进内存/磁盘不进 pkl）。

通过标准:
    1. 每帧掩码数 ≥ 15（分层 coarse+fine，过滤 area≥50、predicted_iou≥0.8 后）
    2. clip_vec (512,) L2 范数 ≈ 1.0（tol 1e-3）
    3. segmentation 与 rgb 同尺寸 (H,W) bool
    4. pkl 缓存**不含** segmentation 键（cache 契约保持，Phase 6 反序列化兼容）
产物: data/outputs/phase6/replica_feature_map.pkl、segs/frame{0,50,100,150}.npz、
      掩码叠加 PNG
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch

from edge_3dgs_slam.dataset.replica import ReplicaSequence
from edge_3dgs_slam.feature_factory import (extract_hierarchical, load_feature_config,
                                            load_models, release_models, update_cache)

OUT = Path("data/outputs/phase6")
SEGS = OUT / "segs"
REPLICA_ROOT = Path("third_party/SplaTAM/data/Replica")
SCENE = "office0"
FRAMES = [0, 50, 100, 150]
SCALE = 0.5          # 600×340，与 phase2/3 渲染同分辨率
MIN_AREA = 50        # 掩码过滤：面积 ≥ 50px
MIN_IOU = 0.8        # 掩码过滤：predicted_iou ≥ 0.8
MIN_MASKS = 15       # 每帧掩码数下限


def filter_masks(feats: list[dict]) -> list[dict]:
    """按 area/predicted_iou 过滤（分层 fine 掩码也过，去噪）。"""
    return [f for f in feats if f["area"] >= MIN_AREA and f["predicted_iou"] >= MIN_IOU]


def draw_overlay(rgb: np.ndarray, feats: list[dict], path: Path, title: str = "") -> None:
    img = rgb.copy()
    rng = np.random.RandomState(7)
    for f in feats:
        x, y, w, h = f["bbox"]
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img, str(f["mask_id"]), (x, max(y - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color, 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                    2, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  [可视化] {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--frames", type=int, nargs="+", default=FRAMES,
                    help="要提取的帧号（默认 0/50/100/150）")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SEGS.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_feature_config()
    fm = load_models(cfg, device)

    seq = ReplicaSequence.from_dir(REPLICA_ROOT, SCENE, max_frames=max(args.frames) + 1)
    K = seq.frame_scaled(0, SCALE).K
    print(f"[phase6] {SCENE} 帧 {args.frames} @{int(seq.cam.width*SCALE)}x{int(seq.cam.height*SCALE)} "
          f"| K00={K[0,0]:.1f}")

    checks: list[tuple[str, bool]] = []
    all_ok = True
    for t in args.frames:
        t0 = time.monotonic()
        fr = seq.frame_scaled(t, SCALE)
        feats = extract_hierarchical(t, fr.rgb, fm, keep_seg=True)
        feats = filter_masks(feats)
        dt = time.monotonic() - t0

        n = len(feats)
        ok_n = n >= MIN_MASKS
        norms = [float(np.linalg.norm(f["clip_vec"])) for f in feats]
        ok_l2 = all(abs(nrm - 1.0) < 1e-3 for nrm in norms)
        ok_seg = all(f["segmentation"].shape == fr.rgb.shape[:2]
                     and f["segmentation"].dtype == bool for f in feats)
        print(f"帧 {t}: 掩码 {n}（≥{MIN_MASKS}: {ok_n}）| L2≈1.0: {ok_l2} | "
              f"seg 同尺寸 bool: {ok_seg} | 耗时 {dt:.0f}s")
        ok = ok_n and ok_l2 and ok_seg
        all_ok &= ok
        checks.append((f"帧 {t} 掩码/L2/seg", ok))

        # 保存 segmentation（npz，供蒸馏阶段加载；不进 pkl）
        seg_stack = np.stack([f["segmentation"] for f in feats])
        np.savez_compressed(SEGS / f"frame{t}.npz",
                            segs=seg_stack, mask_ids=np.asarray([f["mask_id"] for f in feats]),
                            hier=np.asarray([f["hier_level"] for f in feats]),
                            bboxs=np.asarray([f["bbox"] for f in feats]))
        draw_overlay(fr.rgb, feats, OUT / f"masks_frame{t}.png", title=f"frame {t}: {n} masks")
        # pkl 缓存（不含 segmentation）
        update_cache(OUT / "replica_feature_map.pkl", t, fr.rgb.shape[0], fr.rgb.shape[1], feats)

    # ---- pkl 契约检查：无 segmentation 键 ----
    import pickle
    pkl_path = OUT / "replica_feature_map.pkl"
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)
    has_seg = any("segmentation" in m for fc in cache.values() for m in fc["masks"])
    ok_contract = not has_seg and all(fc["H"] == 340 and fc["W"] == 600 for fc in cache.values())
    print(f"pkl 契约（无 segmentation、H/W=340/600）: {'PASS ✅' if ok_contract else 'FAIL ❌'}")
    checks.append(("pkl 契约", ok_contract))
    all_ok &= ok_contract

    release_models(fm)
    del fm

    print("=" * 62)
    print("Phase 6 §3 特征重提验证汇总")
    for name, passed in checks:
        all_ok &= passed
        print(f"  [{'PASS ✅' if passed else 'FAIL ❌'}] {name}")
    print("=" * 62)
    print("全部 PASS ✅" if all_ok else "存在 FAIL ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
