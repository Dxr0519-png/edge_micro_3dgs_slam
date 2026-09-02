"""Phase 5 特征图缓存：掩码级（非逐像素）pickle 存储。

对应 docs/05 §3；格式契约见 IMPLEMENTATION_PLAN §5.2：
    {frame_id: {"H": H, "W": W, "masks": [{"mask_id", "bbox": [x,y,w,h],
      "clip_vec": ndarray(512,) float32 L2归一化, "hier_level"}]}}
纯 dict/list/ndarray，Phase 6 任意环境可 pickle.load（反序列化验收）。
"""
from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MaskEntry:
    """一条掩码级特征记录。"""
    mask_id: int
    bbox: list[int]           # [x, y, w, h]
    clip_vec: np.ndarray      # (512,) float32, L2 归一化
    hier_level: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"mask_id": self.mask_id, "bbox": list(self.bbox),
                "clip_vec": np.asarray(self.clip_vec, np.float32),
                "hier_level": self.hier_level}

    @classmethod
    def from_dict(cls, d: dict) -> "MaskEntry":
        return cls(mask_id=int(d["mask_id"]), bbox=[int(v) for v in d["bbox"]],
                   clip_vec=np.asarray(d["clip_vec"], np.float32),
                   hier_level=int(d.get("hier_level", 0)))


@dataclass
class FrameCache:
    """单帧掩码级缓存（H/W + 掩码列表）。"""
    H: int
    W: int
    masks: list[MaskEntry]

    def to_dict(self) -> dict[str, Any]:
        return {"H": int(self.H), "W": int(self.W),
                "masks": [m.to_dict() for m in self.masks]}

    @classmethod
    def from_dict(cls, d: dict) -> "FrameCache":
        return cls(H=int(d["H"]), W=int(d["W"]),
                   masks=[MaskEntry.from_dict(m) for m in d["masks"]])


def save_cache(cache: dict[int, dict], path: str | Path) -> None:
    """写缓存（按 frame_id 组织）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def load_cache(path: str | Path) -> dict[int, dict]:
    """读缓存；不存在返回空 dict。"""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def masks_to_matrix(frame_cache: dict) -> np.ndarray:
    """取单帧全部掩码 clip_vec → (M,512) float32（供 AE 训练/文本查询）。"""
    return np.asarray([m["clip_vec"] for m in frame_cache["masks"]], np.float32)


def update_cache(path: str | Path, frame_id: int, H: int, W: int, feats: list[dict]) -> dict:
    """读-改-写，按 frame_id 幂等。feats 来自 pipeline.extract/extract_hierarchical。"""
    cache = load_cache(path)
    cache[int(frame_id)] = {"H": int(H), "W": int(W), "masks": [
        {"mask_id": int(f["mask_id"]), "bbox": [int(v) for v in f["bbox"]],
         "clip_vec": np.asarray(f["clip_vec"], np.float32),
         "hier_level": int(f["hier_level"])} for f in feats]}
    save_cache(cache, path)
    return cache
