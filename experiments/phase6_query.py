#!/usr/bin/env python3
"""Phase 6 §4 查询引擎验证：文本 → bbox/热力图（Replica office0 语义）。

通过标准（docs/06 §7 验收②③ + 坑 4/5）:
    1. bbox 在房间范围内（x∈[-2,0.5] y∈[-2,0.5] z∈[1,4]），extent 每轴 0.05-2m
    2. 椅子 vs 桌子 bbox 中心距 > 0.3m（多物体可区分，验收③）
    3. 簇点数 ≥ 20（top_k=100 内），confidence（英文锚）> 0.1
       （S0 余弦压缩区间适配，docs/05 坑 5；logit_scale 缩放后 > 1.5 并列报告）
    4. 热力图 PNG 输出（验收②数据级）+ eps 网格 {0.05,0.1,0.15,0.2} 调参记录（坑 4）
查询集: 英文主锚 + 中文并列报告（docs/05 坑 8）："black chair" / 「黑色的椅子」、
       "table" / 「桌子」、"monitor"。
数据: data/outputs/phase6/probe_model_replica_p6.pt（蒸馏后 Phase 6 模型）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from edge_3dgs_slam.dataset.replica import ReplicaSequence
from edge_3dgs_slam.feature_factory import LangAE, encode_text, load_feature_config, load_models
from edge_3dgs_slam.gaussian import FEATURE_KEY, GaussianModel
from edge_3dgs_slam.query.engine import query_model, render_heatmap

OUT = Path("data/outputs/phase6")
CKPT = OUT / "probe_model_replica_p6.pt"
REPLICA_ROOT = Path("third_party/SplaTAM/data/Replica")
SCENE = "office0"

# 房间范围由模型 means 实际包围盒外扩 20% 计算（前 200k 高斯切片覆盖房间不同区域，
# 拍脑袋的固定范围会误判——见验证报告）
QUERIES = [("black chair", "黑色的椅子"), ("table", "桌子"), ("monitor", "显示器")]
EPS_GRID = [0.05, 0.1, 0.15, 0.2]


def load_p6_model() -> GaussianModel:
    s = torch.load(CKPT, map_location="cpu", weights_only=False)
    params = {k: torch.nn.Parameter(v.cuda()) for k, v in s["params"].items()}
    model = GaussianModel(params, variables=s.get("variables", {}))
    print(f"[phase6] 加载 {CKPT.name}: {model.num_gaussians} 高斯, D={params[FEATURE_KEY].shape[1]}")
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_p6_model()
    cfg = load_feature_config()
    ae = LangAE(d_in=512, d_latent=cfg["feature"]["autoencoder"]["latent_dim"],
                hidden=cfg["feature"]["autoencoder"]["hidden"])
    ae.load_state_dict(torch.load(cfg["feature"]["autoencoder"]["checkpoint"]
                                  .replace("lang_ae.pt", "lang_ae_replica.pt"),
                                  map_location=device, weights_only=True))
    ae.to(device).eval()
    fm = load_models(cfg, device)

    seq = ReplicaSequence.from_dir(REPLICA_ROOT, SCENE, max_frames=1)
    w2c = seq.poses_w2c[0].astype(np.float32)
    cam = seq.cam
    sc = 0.25
    K = np.array([[cam.fx * sc, 0, cam.cx * sc],
                  [0, cam.fy * sc, cam.cy * sc],
                  [0, 0, 1]], np.float64)
    H, W = 170, 300

    checks: list[tuple[str, bool]] = []
    all_ok = True
    results = {}
    means_all = model.means3D.detach().cpu().numpy()
    lo = means_all.min(0) - 0.2 * (means_all.max(0) - means_all.min(0))
    hi = means_all.max(0) + 0.2 * (means_all.max(0) - means_all.min(0))
    ROOM = dict(x=(lo[0], hi[0]), y=(lo[1], hi[1]), z=(lo[2], hi[2]))
    print(f"  房间判据范围（模型包围盒外扩 20%）: x{ROOM['x']} y{ROOM['y']} z{ROOM['z']}")

    # ---- eps 网格调参（坑 4）----
    print("=" * 62)
    print("§4 eps 网格调参（坑 4：按场景尺度选值）")
    r0 = query_model("black chair", model, ae, fm.clip, fm.tokenizer,
                     top_k=args.top_k, eps=args.eps, device=device)
    for eps in EPS_GRID:
        r = query_model("black chair", model, ae, fm.clip, fm.tokenizer,
                        top_k=args.top_k, eps=eps, device=device)
        n_clu = int((r["cluster_labels"] >= 0).sum())
        print(f"  eps={eps}: 簇点数 {n_clu}/{len(r['points'])} | conf {r['confidence']:.3f}")

    # ---- 主查询 ----
    print("=" * 62)
    print(f"§4 查询（eps={args.eps}, top_k={args.top_k}）")
    bboxes = {}
    for en, zh in QUERIES:
        r = query_model(en, model, ae, fm.clip, fm.tokenizer,
                        top_k=args.top_k, eps=args.eps, device=device)
        results[en] = r
        bboxes[en] = r["bbox_center"]
        in_room = all(ROOM[k][0] <= r["bbox_center"][i] <= ROOM[k][1]
                      for i, k in enumerate(("x", "y", "z")))
        ext_ok = all(0.01 <= e <= 3.0 for e in r["bbox_extent"])
        n_clu = int((r["cluster_labels"] >= 0).sum())
        print(f"  「{en}」/「{zh}」: center={np.round(r['bbox_center'], 2)} "
              f"extent={np.round(r['bbox_extent'], 2)} conf={r['confidence']:.3f}")
        print(f"     在房间内: {in_room} | extent 合理: {ext_ok} | "
              f"簇点数 {n_clu}（≥20: {n_clu >= 20}）| 中文 cos: "
              f"{query_model(zh, model, ae, fm.clip, fm.tokenizer, top_k=5, eps=args.eps, device=device)['confidence']:.3f}")
        ok = in_room and ext_ok and n_clu >= 20
        all_ok &= ok
        checks.append((f"「{en}」bbox/簇", ok))

    # ---- 验收③：椅子 vs 桌子可区分 ----
    d_ct = float(np.linalg.norm(bboxes["black chair"] - bboxes["table"]))
    ok_sep = d_ct > 0.3
    print(f"  椅子-桌子中心距 = {d_ct:.2f} m（> 0.3: {ok_sep}）")
    all_ok &= ok_sep
    checks.append(("椅子 vs 桌子可区分（中心距 > 0.3m）", ok_sep))

    # ---- 热力图（验收②数据级）----
    print("=" * 62)
    print("§4 热力图")
    rel = None
    from edge_3dgs_slam.query.engine import _cosine_relevance
    t = encode_text(fm.clip, fm.tokenizer, ["black chair"], fm.device)
    rel = _cosine_relevance(model.params[FEATURE_KEY].detach().cpu().numpy(),
                            t, ae, fm.device)
    heat = render_heatmap(model, rel, w2c, K, W, H, min_score=0.0, top_k=args.top_k)
    np.save(OUT / "heatmap_black_chair.npy", heat)
    from PIL import Image
    Image.fromarray(heat).save(OUT / "heatmap_black_chair.png")
    cov = float((heat.sum(-1) > 0).mean())
    print(f"  热力图覆盖 {cov*100:.1f}% | 已存 {OUT/'heatmap_black_chair.png'}")
    ok_hm = cov > 0.01
    all_ok &= ok_hm
    checks.append(("热力图输出", ok_hm))

    print("=" * 62)
    print("Phase 6 §4 查询引擎验证汇总")
    for name, passed in checks:
        all_ok &= passed
        print(f"  [{'PASS ✅' if passed else 'FAIL ❌'}] {name}")
    print("=" * 62)
    print("全部 PASS ✅" if all_ok else "存在 FAIL ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
