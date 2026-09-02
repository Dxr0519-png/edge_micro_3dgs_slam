"""gaussian 子模块：高斯模型（参数化/增删/保存）+ 可微光栅化封装。

对应 Phase 2 的可微光栅化核心机制（裁剪自 SplaTAM + inria 3DGS）；
Phase 6 §1 追加语言特征通道（FEATURE_KEY/add_feature_dim/freeze_geometry/load_ply）
与特征渲染快速路径 render_precomp。
"""
from .model import (FEATURE_KEY, GaussianModel, add_feature_dim, freeze_geometry,
                    load_ply)
from .render import render, render_precomp, setup_camera, transform_to_frame
from .ssim import calc_ssim
from .torch_feature_rasterizer import rasterize_feature

__all__ = [
    "GaussianModel", "render", "render_precomp", "setup_camera", "transform_to_frame",
    "calc_ssim", "FEATURE_KEY", "add_feature_dim", "freeze_geometry", "load_ply",
    "rasterize_feature",
]
