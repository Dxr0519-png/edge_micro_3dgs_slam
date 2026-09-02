"""Phase 5 场景级自编码器：512 维 CLIP 特征 → D 维潜变量（LangSplat 思想）。

对应 docs/05 §5：MLP 512→256→64→16→D（enc）/ D→16→64→256→512（dec），
激活 ReLU（末层线性）；损失 = L1 均值 + lambda_cos × (1 − cos 均值)；
收敛判据：重建余弦相似度均值 > 0.95；权重存 data/checkpoints/lang_ae.pt。
Phase 6 用它把高斯潜变量解码回 512 维做监督（V_2d = ae.dec(...)）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 收敛判据（文档 §5）
COS_CONVERGE = 0.95


class MLP(nn.Module):
    """多层感知机：中间层 ReLU，末层线性（文档 §5：enc/dec 末层均无激活）。"""

    def __init__(self, dims: Sequence[int], act: type[nn.Module] = nn.ReLU):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LangAE(nn.Module):
    """512 → D → 512 场景级自编码器。"""

    def __init__(self, d_in: int = 512, d_latent: int = 3,
                 hidden: Sequence[int] = (256, 64, 16)):
        super().__init__()
        h = list(hidden)
        self.enc = MLP([d_in, *h, d_latent])
        self.dec = MLP([d_latent, *reversed(h), d_in])

    def encode(self, v: torch.Tensor) -> torch.Tensor:
        return self.enc(v)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(v))


def train_ae(vectors: np.ndarray, d_latent: int = 3, hidden: Sequence[int] = (256, 64, 16),
             lr: float = 1e-3, lambda_cos: float = 0.5, max_iters: int = 5000,
             checkpoint: str | Path = "data/checkpoints/lang_ae.pt", device: str = "cuda",
             print_every: int = 500, seed: int = 0) -> tuple[LangAE, dict]:
    """训练场景级 AE。vectors: (M,512) 本场景所有掩码级 CLIP 特征。

    输入按行 L2 归一化（与特征提取端一致）；loss = |v̂−v|₁ + lambda_cos·(1−cos(v̂,v))；
    收敛判据 cos_mean > 0.95；state_dict 存 checkpoint。返回 (model, stats)。
    """
    torch.manual_seed(seed)
    v = torch.as_tensor(np.asarray(vectors, np.float32), device=device)
    v = F.normalize(v, dim=-1)
    ae = LangAE(d_in=v.shape[1], d_latent=d_latent, hidden=hidden).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)

    iters, cos_mean, l1, loss = 0, -1.0, float("nan"), float("nan")
    for it in range(max_iters):
        v_hat = ae(v)
        l1 = (v_hat - v).abs().mean()
        cos_mean = F.cosine_similarity(v_hat, v, dim=-1).mean()
        loss = l1 + lambda_cos * (1.0 - cos_mean)
        opt.zero_grad()
        loss.backward()
        opt.step()
        iters = it + 1
        if (it + 1) % print_every == 0 or it + 1 == max_iters:
            print(f"  iters {it + 1:5d} | loss {loss.item():.4f} | "
                  f"L1 {l1.item():.5f} | cos_mean {cos_mean.item():.4f}")
        if cos_mean.item() > COS_CONVERGE:
            break

    cp = Path(checkpoint)
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), str(cp))
    stats = {"iters": iters, "cos_mean": float(cos_mean.item()), "l1": float(l1.item()),
             "loss": float(loss.item()), "latent_dim": d_latent, "checkpoint": str(cp)}
    print(f"[feature_factory] AE 训练完成: iters={iters}, cos_mean={stats['cos_mean']:.4f}, "
          f"checkpoint={cp}")
    return ae, stats


def embed(model: LangAE, vectors: np.ndarray) -> np.ndarray:
    """(M,512) → (M,D) float32 numpy（Phase 6 高斯潜变量）。"""
    dev = next(model.parameters()).device
    v = torch.as_tensor(np.asarray(vectors, np.float32), device=dev)
    with torch.inference_mode():
        z = model.encode(v)
    return z.cpu().numpy().astype(np.float32)


def train_ae_from_cache(cache_path: str | Path, **kw) -> tuple[LangAE, dict]:
    """从 feature_map.pkl 取全部掩码 clip_vec 训练（真实场景特征）。"""
    from .cache import load_cache
    cache = load_cache(cache_path)
    vecs = [np.asarray(m["clip_vec"], np.float32)
            for fc in cache.values() for m in fc["masks"]]
    if not vecs:
        raise ValueError(f"缓存 {cache_path} 无掩码特征（先运行特征提取）")
    return train_ae(np.stack(vecs), **kw)
