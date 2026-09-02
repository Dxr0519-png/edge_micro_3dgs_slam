"""Phase 5 特征提取流水线：万物掩码 → 外接框 crop → MobileCLIP 特征。

对应 docs/05 §2（extract）、§4（分层掩码）、§6（关键帧接口 / FP16）。
约定：mask dict 来自 SAM generate()，含 segmentation/bbox[x,y,w,h]/area/
      predicted_iou/stability_score/crop_box/point_coords。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models import FeatureModels


def crop_resize(rgb: np.ndarray, bbox: list | tuple, size: int = 224,
                seg: np.ndarray | None = None, fill: int = 0) -> np.ndarray:
    """SAM bbox=[x,y,w,h]（可为 float）→ clamp 到图内 → resize → (size,size,3) uint8。

    seg 给定时（掩码 (H,W) bool），bbox 内掩码外区域填充为 fill（0=黑）——
    显著提升 crop 语义纯度（万物掩码 bbox 常含大量背景，稀释 MobileCLIP 特征）。
    """
    H, W = rgb.shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = rgb  # 空 crop（退化 bbox）回退整图
    if seg is not None:
        mask = seg[y0:y1, x0:x1]
        if mask.shape == crop.shape[:2]:
            crop = crop.copy()
            crop[~mask] = fill
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def clip_encode_image(clip, preprocess, crop: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """crop (224,224,3) uint8 → 官方 preprocess（PIL 输入）→ CUDA → FP16 encode → (1,512) L2 归一化。

    §8 坑 2：必须走 preprocess（Resize 224 + CenterCrop + ToTensor），否则相似度空间崩。
    """
    img = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
        v = clip.encode_image(img)
    v = v.float()                      # autocast 输出 fp16；归一化在 fp32 做更稳
    return F.normalize(v, dim=-1)      # (1,512)


def encode_text(model, tokenizer, texts: list[str], device: str | None = None) -> torch.Tensor:
    """文本 → tokenizer → FP16 encode_text → (N,512) L2 归一化。

    §8 坑 3：文本侧必须与图像用同一个模型（同一 tokenizer / 编码器）。
    """
    dev = device or next(model.parameters()).device
    tok = tokenizer(texts).to(dev)
    with torch.inference_mode(), torch.autocast(device_type=str(dev), dtype=torch.float16):
        t = model.encode_text(tok)
    t = t.float()
    return F.normalize(t, dim=-1)      # (N,512)


def _extract_with(frame_id: int, rgb: np.ndarray, mask_gen, fm: FeatureModels,
                  hier_level: int, keep_seg: bool = False) -> list[dict]:
    """用给定 mask_gen 生成掩码并编码特征；每掩码一条记录。

    keep_seg=True（Phase 6 §3）：feat dict 附 segmentation (H,W) bool——
    掩码级监督需要像素级掩码。**只进内存不进 pkl 缓存**（cache 契约不含
    segmentation，见 cache.py 注释）；默认 False 保持 Phase 5 行为不变。
    """
    with torch.inference_mode(), torch.autocast(**fm.autocast_kwargs):
        masks = mask_gen.generate(rgb)
    feats = []
    for i, m in enumerate(masks):
        crop = crop_resize(rgb, m["bbox"], fm.input_size, seg=m.get("segmentation"))
        v = clip_encode_image(fm.clip, fm.preprocess, crop, fm.device)
        feat = {
            "mask_id": i,
            "bbox": [int(round(x)) for x in m["bbox"]],
            "clip_vec": v[0].float().cpu().numpy(),   # (512,) float32 L2 归一化（入缓存转 float32）
            "area": int(m["area"]),
            "predicted_iou": float(m["predicted_iou"]),
            "stability_score": float(m["stability_score"]),
            "hier_level": hier_level,
        }
        if keep_seg:
            feat["segmentation"] = m["segmentation"]
        feats.append(feat)
    return feats


def extract(frame_id: int, rgb: np.ndarray, fm: FeatureModels, hier_level: int = 0,
            keep_seg: bool = False) -> list[dict]:
    """单帧万物掩码 + MobileCLIP 特征（默认档：16 点 / pred_iou_thresh=0.86）。"""
    return _extract_with(frame_id, rgb, fm.mask_gen, fm, hier_level=hier_level,
                         keep_seg=keep_seg)


def extract_hierarchical(frame_id: int, rgb: np.ndarray, fm: FeatureModels,
                         keep_seg: bool = False) -> list[dict]:
    """§4 分层掩码（LangSplat 思想）：coarse（8 点 → hier_level=0，大物体/背景）
    + fine（32 点 → hier_level=1，小物体/细节）两档合并。重叠保留，Phase 6 反投影按层级加权。"""
    feats = _extract_with(frame_id, rgb, fm.mask_gen_coarse, fm, hier_level=0, keep_seg=keep_seg)
    feats += _extract_with(frame_id, rgb, fm.mask_gen_fine, fm, hier_level=1, keep_seg=keep_seg)
    return feats


def extract_keyframes(frames: dict[int, np.ndarray], fm: FeatureModels,
                      every_n: int = 10) -> dict[int, list[dict]]:
    """§6：只对关键帧提取（非每帧），进一步省算力。frames: {frame_id: rgb (H,W,3) uint8}。"""
    out: dict[int, list[dict]] = {}
    for fid, rgb in frames.items():
        if fid % every_n == 0:
            out[fid] = extract(fid, rgb, fm)
    return out
