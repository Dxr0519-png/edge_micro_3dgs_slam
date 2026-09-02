"""Phase 5 模型加载：MobileSAM 万物掩码 + MobileCLIP-S0（apple 官方包）。

对应 docs/05 §1（就绪检查后加载）、§6（FP16 推理 / 用完释放显存）。
模型名与权重路径全部来自 config，替换模型零代码改动。
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# MobileSAM 权重下载（就绪检查惯例；原文档 main 分支 URL 404，实测 master 可用）
MOBILE_SAM_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
# MobileCLIP-S0 原版权重（HF 直连不可达时经 hf-mirror.com 镜像下载）
MOBILECLIP_S0_URL = "https://hf-mirror.com/apple/MobileCLIP-S0/resolve/main/mobileclip_s0.pt"


@dataclass
class FeatureModels:
    """特征工厂的全部模型实例（SAM 三档掩码生成器 + CLIP 双编码器）。"""
    sam: Any
    mask_gen: Any            # 默认档（points_per_side=16, pred_iou_thresh=0.86）
    mask_gen_coarse: Any     # 分层 coarse（8 点 → 大物体/背景）
    mask_gen_fine: Any       # 分层 fine（32 点 → 小物体/细节）
    clip: Any
    preprocess: Any
    tokenizer: Any
    device: str = "cuda"
    dtype: torch.dtype = torch.float16   # §6：FP16 推理
    input_size: int = 224                # 外接框 crop 尺寸（来自 config）

    @property
    def autocast_kwargs(self) -> dict:
        return {"device_type": self.device, "dtype": self.dtype}


def load_models(cfg: dict, device: str = "cuda", load_sam: bool = True) -> FeatureModels:
    """加载 MobileSAM + MobileCLIP-S0。

    SAM checkpoint 缺失时抛 FileNotFoundError 并给出修正后的下载命令。
    load_sam=False（Phase 6 查询服务）：只加载 CLIP+tokenizer（文本编码所需），
    省 SAM 常驻 ~200MB 显存；sam/mask_gen 置 None，特征提取路径不可用。
    """
    s_cfg, c_cfg = cfg["feature"]["sam"], cfg["feature"]["clip"]

    # ---- MobileSAM（ViT-Tiny 蒸馏，约 40MB）----
    sam = mask_gen = mask_gen_coarse = mask_gen_fine = None
    if load_sam:
        ckpt = Path(s_cfg["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(
                f"MobileSAM 权重不存在: {ckpt}\n"
                f"下载（实测可用 URL）: wget -O {ckpt} {MOBILE_SAM_URL}")
        from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry["vit_t"](checkpoint=str(ckpt))
        sam.eval().to(device)
        mask_gen = SamAutomaticMaskGenerator(
            sam, points_per_side=s_cfg["points_per_side"],
            pred_iou_thresh=s_cfg["pred_iou_thresh"])
        mask_gen_coarse = SamAutomaticMaskGenerator(
            sam, points_per_side=s_cfg["hierarchical"]["coarse_points_per_side"])
        mask_gen_fine = SamAutomaticMaskGenerator(
            sam, points_per_side=s_cfg["hierarchical"]["fine_points_per_side"])

    # ---- MobileCLIP-S0（apple 官方包，512 维，文本/图像同模型）----
    c_ckpt = Path(c_cfg["pretrained"])
    if not c_ckpt.exists():
        raise FileNotFoundError(
            f"MobileCLIP 权重不存在: {c_ckpt}\n"
            f"下载（HF 直连不可达，实测镜像可用）: wget -O {c_ckpt} {MOBILECLIP_S0_URL}")
    import mobileclip
    # apple 包模型名格式为小写下划线（mobileclip_s0），config 保留可读名 MobileCLIP-S0
    model_name = c_cfg["name"].lower().replace("-", "_")
    clip, _, preprocess = mobileclip.create_model_and_transforms(
        model_name, pretrained=str(c_ckpt), device=device)
    tokenizer = mobileclip.get_tokenizer(model_name)

    fm = FeatureModels(
        sam=sam, mask_gen=mask_gen, mask_gen_coarse=mask_gen_coarse,
        mask_gen_fine=mask_gen_fine, clip=clip, preprocess=preprocess,
        tokenizer=tokenizer, device=device, input_size=c_cfg["input_size"])

    n_sam = (sum(p.numel() for p in sam.parameters()) / 1e6) if sam is not None else 0.0
    n_clip = sum(p.numel() for p in clip.parameters()) / 1e6
    print(f"[feature_factory] 加载完成: MobileSAM {n_sam:.1f}M 参数"
          f"{'（load_sam=False 跳过）' if sam is None else ''}, "
          f"{c_cfg['name']} {n_clip:.1f}M 参数, device={device}")
    if device.startswith("cuda"):
        mem = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"[feature_factory] 模型显存占用约 {mem:.0f} MB")
    return fm


def release_models(fm: FeatureModels) -> None:
    """§6：用完释放 GPU 显存（与 Tracking 分时复用）。

    注意：本函数清空模型持有的 CUDA 缓存；调用方随后 `del fm` 才能让模型对象
    真正可回收（del 局部参数不改变调用方引用）。
    """
    for obj in (fm.sam, fm.mask_gen, fm.mask_gen_coarse, fm.mask_gen_fine,
                fm.clip, fm.preprocess, fm.tokenizer):
        if hasattr(obj, "to") and hasattr(obj, "parameters"):
            obj.cpu()   # 先移回 CPU，确保 CUDA 侧引用解除
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[feature_factory] 模型已置 CPU 并 empty_cache，调用方请再 del fm")
