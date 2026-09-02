"""feature_factory 配置：yaml 优先，缺失/不可用时回退模块默认常量。对应 Phase 5。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 默认配置（与 config/feature/mobilesam_clip.yaml 结构一致；yaml 缺失时回退）
# 注：MobileCLIP-S0 用 apple 官方包 mobileclip 加载（open_clip 不支持 S0），
#     pretrained 为本地权重路径（HF 直连不可达时经 hf-mirror.com 下载）。
DEFAULT_FEATURE_CONFIG: dict[str, Any] = {
    "feature": {
        "sam": {
            "name": "MobileSAM",
            "checkpoint": "data/checkpoints/mobile_sam.pt",
            "points_per_side": 16,
            "pred_iou_thresh": 0.86,
            "hierarchical": {"coarse_points_per_side": 8, "fine_points_per_side": 32},
        },
        "clip": {
            "name": "MobileCLIP-S0",            # apple 官方包 mobileclip 的模型名（mobileclip_s0）
            "pretrained": "data/checkpoints/mobileclip_s0.pt",
            "dim": 512,                         # 输出维度
            "input_size": 224,
        },
        "autoencoder": {
            "latent_dim": 3,                    # 512 -> D（可配 16）
            "hidden": [256, 64, 16],
            "lr": 1.0e-3,
            "lambda_cos": 0.5,
            "checkpoint": "data/checkpoints/lang_ae.pt",
        },
        "cache": {
            "format": "mask_level",             # 掩码级（非逐像素，省显存）
            "path": "data/outputs/phase5/feature_map.pkl",
        },
    }
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归深合并：override 非 dict 值直接覆盖，dict 值递归合并。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_feature_config(path: str | Path | None = None) -> dict[str, Any]:
    """读 yaml（默认 config/feature/mobilesam_clip.yaml），按 'feature' 键深合并进默认值。

    yaml 模块/文件缺失 → 回退默认常量并警告一行（不 raise），保证模块可直接使用。
    """
    cfg_path = Path(path) if path else Path("config/feature/mobilesam_clip.yaml")
    cfg = DEFAULT_FEATURE_CONFIG
    if not cfg_path.exists():
        print(f"[feature_factory] 警告: 配置 {cfg_path} 不存在，使用默认常量")
        return cfg
    try:
        with open(cfg_path) as f:
            loaded = yaml.safe_load(f) or {}
        if "feature" in loaded and isinstance(loaded["feature"], dict):
            cfg = _deep_merge(cfg, {"feature": loaded["feature"]})
    except Exception as e:
        print(f"[feature_factory] 警告: 配置 {cfg_path} 解析失败（{e}），使用默认常量")
    return cfg
