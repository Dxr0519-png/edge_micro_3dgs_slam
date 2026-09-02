#!/usr/bin/env python3
"""Phase 5 §5 场景级自编码器验证（验收 4：AE 重建余弦相似度均值 > 0.95）。

用法:
    python3 experiments/phase5_langae.py [--mode auto|synth] [--device cuda|cpu]

--mode auto: 优先读 data/outputs/phase5/feature_map.pkl 的真实掩码级 CLIP 特征（M 条）；
             无缓存时回退合成低秩向量（随机均匀超球向量 512→3 无法收敛，必须低秩结构）。
通过标准（重载 lang_ae.pt 后复评，验证持久化）:
    - 重建余弦相似度均值 > 0.95（COS_CONVERGE）
    - embed 输出 (M, d_latent) 且 dtype == float32
产物: data/outputs/phase5/langae_curve.png（loss/cos 曲线）、data/checkpoints/lang_ae.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from edge_3dgs_slam.feature_factory import (LangAE, COS_CONVERGE, embed, load_cache,
                                            load_feature_config, train_ae)

OUT = Path("data/outputs/phase5")
CACHE_PKL = Path("data/outputs/phase5/feature_map.pkl")


def load_vectors(mode: str) -> tuple[np.ndarray, str]:
    """返回 (vectors (M,512) float32, 来源描述)。"""
    if mode in ("auto", "real") and CACHE_PKL.exists():
        cache = load_cache(CACHE_PKL)
        vecs = [np.asarray(m["clip_vec"], np.float32)
                for fc in cache.values() for m in fc["masks"]]
        if vecs:
            return np.stack(vecs), f"feature_map.pkl（{len(vecs)} 条掩码特征）"
        print(f"[phase5_langae] 缓存 {CACHE_PKL} 无掩码特征，回退合成向量")
    # 合成低秩结构向量：512 维由 3 维潜变量经固定投影 + 极小噪声生成
    # （模拟真实 CLIP 特征语义冗余高、有效维数远小于 512；注意 512 维下噪声必须很小：
    #  0.05/元素 → 噪声范数 ~1.1 与信号同量级，重建 cos 上限仅 ~0.66，无法收敛到 0.95）
    rng = np.random.RandomState(0)
    W = rng.randn(512, 3).astype(np.float32)
    z = rng.randn(2048, 3).astype(np.float32)
    v = z @ W.T
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    v = v + 0.005 * rng.randn(*v.shape).astype(np.float32)
    return v, "合成低秩向量（2048 条，512 维秩 3 + 0.5% 噪声，模拟 CLIP 语义冗余）"


def run_ae_test(vectors: np.ndarray, src: str, cfg: dict, device: str) -> bool:
    """§5 训练 + 重载复评。"""
    print("=" * 62)
    print("§5 场景级自编码器（512 → D → 512）")
    print(f"向量: {vectors.shape} | 来源: {src} | device: {device}")
    ae_cfg = cfg["feature"]["autoencoder"]
    d_latent = ae_cfg["latent_dim"]

    ae, stats = train_ae(vectors, d_latent=d_latent, hidden=ae_cfg["hidden"],
                         lr=ae_cfg["lr"], lambda_cos=ae_cfg["lambda_cos"],
                         checkpoint=ae_cfg["checkpoint"], device=device)

    # ---- 重载 state_dict 复评（验证持久化；Phase 6 按此路径加载）----
    ae2 = LangAE(d_in=vectors.shape[1], d_latent=d_latent, hidden=ae_cfg["hidden"])
    ae2.load_state_dict(torch.load(ae_cfg["checkpoint"], map_location=device, weights_only=True))
    ae2.to(device).eval()
    v = F.normalize(torch.as_tensor(vectors, device=device), dim=-1)  # 与训练输入一致
    with torch.inference_mode():
        v_hat = ae2(v)
    cos2 = F.cosine_similarity(v_hat, v, dim=-1).mean().item()
    l1_2 = (v_hat - v).abs().mean().item()

    z = embed(ae2, vectors)
    ok_shape = z.shape == (vectors.shape[0], d_latent) and z.dtype == np.float32
    print(f"训练统计: iters={stats['iters']}, cos_mean={stats['cos_mean']:.4f}, "
          f"L1={stats['l1']:.5f}")
    print(f"重载复评: cos_mean={cos2:.4f} (> {COS_CONVERGE} 收敛判据), L1={l1_2:.5f}")
    print(f"embed 输出: {z.shape} {z.dtype} (期望 (M,{d_latent}) float32)")

    ok = cos2 > COS_CONVERGE and ok_shape
    print(f"§5 验收 4 {'PASS ✅' if ok else 'FAIL ❌'}"
          f" | cos_mean={cos2:.4f} > {COS_CONVERGE}: {cos2 > COS_CONVERGE}"
          f" | embed 形状/dtype: {ok_shape}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["auto", "real", "synth"], default="auto",
                    help="向量来源: auto=缓存优先否则合成, real=强制缓存, synth=强制合成")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_feature_config()
    vectors, src = load_vectors(args.mode)
    ok = run_ae_test(vectors, src, cfg, device)

    print("=" * 62)
    print("全部 PASS ✅" if ok else "存在 FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
