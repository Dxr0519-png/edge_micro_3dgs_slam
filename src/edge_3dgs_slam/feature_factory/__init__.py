"""feature_factory 子模块：边缘端 2D 开放词汇特征工厂。

MobileSAM 万物掩码 + MobileCLIP-S0 特征，输出掩码级语义特征缓存。对应 Phase 5。
"""
from .autoencoder import COS_CONVERGE, MLP, LangAE, embed, train_ae, train_ae_from_cache
from .cache import FrameCache, MaskEntry, load_cache, masks_to_matrix, save_cache, update_cache
from .config import DEFAULT_FEATURE_CONFIG, load_feature_config
from .models import FeatureModels, load_models, release_models
from .pipeline import (clip_encode_image, crop_resize, encode_text, extract,
                       extract_hierarchical, extract_keyframes)

__all__ = [
    "load_feature_config", "DEFAULT_FEATURE_CONFIG",
    "FeatureModels", "load_models", "release_models",
    "crop_resize", "clip_encode_image", "encode_text",
    "extract", "extract_hierarchical", "extract_keyframes",
    "MaskEntry", "FrameCache", "save_cache", "load_cache", "masks_to_matrix", "update_cache",
    "MLP", "LangAE", "train_ae", "embed", "train_ae_from_cache", "COS_CONVERGE",
]
