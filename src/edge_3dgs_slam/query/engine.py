"""Phase 6 §4 空间查询引擎：文本 → 3D BBox / 热力图。

流程（docs/06 §4）：
    t = encode_text(clip, tokenizer, [text])      # (1,512) L2
    V = ae.dec(gaussians.feature)                 # (N,512) L2
    relevance = V @ t.T                           # (N,) 余弦
    idx = topk(top_k) → DBSCAN 聚类 → 最大簇 → AABB/OBB bbox + confidence

- 双入口：query()（numpy 快照：ROS 服务）与 query_model()（实验：torch 模型）；
- 输出结构与 QueryResult.msg 对齐（bbox_center/extent/rotation 9 元素行展开）；
- 热力图：relevance → viridis，复用 inria kernel 前向渲染（colors_precomp 换色）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..feature_factory import encode_text
from ..gaussian import GaussianModel, render_precomp

LOGIT_SCALE_DEFAULT = 4.21    # MobileCLIP-S0 温度（S0 余弦压缩区间适配，docs/05 坑 5）


def _cosine_relevance(features: np.ndarray, t: torch.Tensor, ae, device: str) -> torch.Tensor:
    """(N,D) features → (N,512) → 与文本向量余弦 (N,)。"""
    z = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.inference_mode():
        V = F.normalize(ae.dec(z), dim=-1)
    return (V @ t.T).squeeze(-1)


def query(text: str, means: np.ndarray, features: np.ndarray, ae, clip, tokenizer,
          top_k: int = 100, min_score: float = 0.0, eps: float = 0.15,
          min_samples: int = 5, device: str = "cuda",
          cluster_method: str = "obb") -> dict:
    """文本查询：top-k 相关高斯 → DBSCAN 聚类 → bbox/confidence。

    参数:
        means: (N,3) 世界系坐标 numpy
        features: (N,D) 语言潜变量 numpy（经 ae.dec 解码回 512 维）
        ae: LangAE（device 上，eval）
        clip/tokenizer: MobileCLIP 文本编码
        top_k: 候选高斯数
        min_score: 相似度下限（低于此值的高斯不参与聚类）
        eps/min_samples: DBSCAN（场景尺度调参，坑 4；office0 室内米级用 0.1-0.2）
        cluster_method: "obb"（PCA 主轴）或 "aabb"
    返回: {query, bbox_center(3,), bbox_extent(3,), bbox_rotation(3,3),
           confidence, points(K,3), scores(K,), eps, cluster_labels}
    """
    t = encode_text(clip, tokenizer, [text], device)
    relevance = _cosine_relevance(features, t, ae, device)
    k = min(top_k, len(relevance))
    idx = torch.topk(relevance, k).indices.cpu().numpy()
    scores = relevance[idx].cpu().numpy()

    keep = scores >= min_score
    idx, scores = idx[keep], scores[keep]
    pts = means[idx]
    if len(pts) == 0:
        return {"query": text, "bbox_center": np.zeros(3), "bbox_extent": np.zeros(3),
                "bbox_rotation": np.eye(3), "confidence": 0.0, "points": pts,
                "scores": scores, "eps": eps, "cluster_labels": np.array([]),
                "major_cluster": -1}

    # DBSCAN 聚类（无簇时全部点兜底，防碎裂）
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    if (labels >= 0).sum() > 0:
        counts = np.bincount(labels[labels >= 0])
        major = int(np.argmax(counts))
        sel = labels == major
    else:
        major, sel = -1, np.ones(len(pts), dtype=bool)

    center, extent, rot = cluster_bbox(pts[sel], method=cluster_method)
    return {
        "query": text,
        "bbox_center": center, "bbox_extent": extent, "bbox_rotation": rot,
        "confidence": float(scores[sel].mean()) if sel.any() else 0.0,
        "points": pts[sel], "scores": scores[sel],
        "eps": eps, "cluster_labels": labels, "major_cluster": major,
    }


def query_model(text: str, gaussians: GaussianModel, ae, clip, tokenizer, **kw) -> dict:
    """实验入口：直接取模型 means/features。"""
    means = gaussians.means3D.detach().cpu().numpy()
    features = gaussians.params["features"].detach().cpu().numpy()
    return query(text, means, features, ae, clip, tokenizer, **kw)


def cluster_bbox(points: np.ndarray, method: str = "obb") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """簇点 → (center(3,), extent(3,), rot(3,3))。obb = PCA 主轴系。"""
    if method == "aabb" or len(points) < 3:
        mn, mx = points.min(0), points.max(0)
        return (mn + mx) / 2, mx - mn, np.eye(3)
    pts_c = points - points.mean(0)
    _, _, vt = np.linalg.svd(pts_c, full_matrices=False)
    rot = vt.T                                    # 主轴（列）
    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1
    proj = pts_c @ rot
    extent = proj.max(0) - proj.min(0)
    center = points.mean(0) + rot @ ((proj.max(0) + proj.min(0)) / 2)
    return center, extent, rot


def render_heatmap(gaussians: GaussianModel, relevance: torch.Tensor, w2c, K, W: int, H: int,
                   min_score: float = 0.0, top_k: int = 100) -> np.ndarray:
    """relevance → viridis 配色，top-k 外/低分置黑，前向渲染 → (H,W,3) uint8。

    复用 inria kernel（render_precomp colors_precomp 换色）——热力图即语义高亮渲染。
    """
    import matplotlib.cm as cm
    rel = relevance.detach()
    n = len(rel)
    k = min(top_k, n)
    idx = torch.topk(rel, k).indices
    colors = torch.zeros(n, 3, device=rel.device)
    r = rel[idx]
    rn = (r - r.min()) / (r.max() - r.min() + 1e-8)
    viridis = torch.as_tensor(cm.viridis(rn.cpu().numpy())[:, :3],
                              dtype=torch.float32, device=rel.device)
    keep = r >= min_score
    colors[idx[keep]] = viridis[keep]
    img = render_precomp(gaussians, torch.as_tensor(w2c, dtype=torch.float32,
                                                    device=rel.device),
                         np.asarray(K, np.float64), W, H,
                         colors_precomp=colors, gaussians_grad=False, camera_grad=False)
    return (img.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
