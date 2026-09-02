"""LangSplat language-gaussian-rasterization wrapper（Phase 6 §2 CUDA 路径）。

来源: third_party/LangSplat/submodules/langsplat-rasterization（diff-gaussian-
rasterization 的 fork，包名与已装的 inria 版冲突——因此以 --target 构建后复制
_C*.so 到此 wrapper 包，**不污染 site-packages 的 inria kernel**）。

与 inria 版的差异: rasterize_gaussians 多一个 `language_feature_precomp (N,3)`
参数，输出多一个 `out_language_feature (3,H,W)` 特征图（同一套投影/深度排序/
α 权重，把颜色通道换成 3 维语言特征——LangSplat 的 D=3 语义，与本项目默认
latent_dim=3 一致；D=16 需改 fork 的 config.h NUM_CHANNELS_language_feature
后重编，见 phase6_langsplat_check.py 的 degraded 记录）。

用法（与 render_precomp 同输入语义，means 相机系 + viewmatrix=I）:
    settings = GaussianRasterizationSettings(..., include_feature=True)
    out = rasterize_gaussians(means3D, means2D, sh, colors_precomp,
                              language_feature_precomp, opacities, scales,
                              rotations, None, settings)
    rgb, feat3 = out[0], out[1]                      # (3,H,W) 各
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item
                      for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D, means2D, sh, colors_precomp, language_feature_precomp,
    opacities, scales, rotations, cov3Ds_precomp, raster_settings,
):
    """fork 接口：多 language_feature_precomp (N,3)，返回 (color, language_feature, radii)。"""
    return _RasterizeGaussians.apply(
        means3D, means2D, sh, colors_precomp, language_feature_precomp,
        opacities, scales, rotations, cov3Ds_precomp, raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, sh, colors_precomp, language_feature_precomp,
                opacities, scales, rotations, cov3Ds_precomp, raster_settings):
        args = (
            raster_settings.bg, means3D, colors_precomp, language_feature_precomp,
            opacities, scales, rotations, raster_settings.scale_modifier, cov3Ds_precomp,
            raster_settings.viewmatrix, raster_settings.projmatrix,
            raster_settings.tanfovx, raster_settings.tanfovy,
            raster_settings.image_height, raster_settings.image_width,
            sh, raster_settings.sh_degree, raster_settings.campos,
            raster_settings.prefiltered, raster_settings.debug,
            raster_settings.include_feature,
        )
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, language_feature, radii, geomBuffer, \
                    binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                raise ex
        else:
            num_rendered, color, language_feature, radii, geomBuffer, \
                binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, language_feature_precomp, means3D,
                              scales, rotations, cov3Ds_precomp, radii, sh,
                              geomBuffer, binningBuffer, imgBuffer)
        return color, language_feature, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_language_feature, _):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (colors_precomp, language_feature_precomp, means3D, scales, rotations,
         cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer,
         imgBuffer) = ctx.saved_tensors
        args = (
            raster_settings.bg, means3D, radii, colors_precomp,
            language_feature_precomp, scales, rotations, raster_settings.scale_modifier,
            cov3Ds_precomp, raster_settings.viewmatrix, raster_settings.projmatrix,
            raster_settings.tanfovx, raster_settings.tanfovy,
            grad_out_color, grad_out_language_feature,
            sh, raster_settings.sh_degree, raster_settings.campos,
            geomBuffer, num_rendered, binningBuffer, imgBuffer,
            raster_settings.debug, raster_settings.include_feature,
        )
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, grad_means2D, grad_means3D, grad_colors_precomp, \
                    grad_language_feature_precomp, grad_opacities, grad_scales, \
                    grad_rotations, grad_cov3Ds_precomp = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                raise ex
        else:
            num_rendered, grad_means2D, grad_means3D, grad_colors_precomp, \
                grad_language_feature_precomp, grad_opacities, grad_scales, \
                grad_rotations, grad_cov3Ds_precomp = _C.rasterize_gaussians_backward(*args)
        grads = (grad_means3D, grad_means2D, None, grad_colors_precomp,
                 grad_language_feature_precomp, grad_opacities, grad_scales,
                 grad_rotations, grad_cov3Ds_precomp, None)
        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    include_feature: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
        return visible

    def forward(self, means3D, means2D, sh, colors_precomp, language_feature_precomp,
                opacities, scales, rotations, cov3Ds_precomp=None):
        return rasterize_gaussians(
            means3D, means2D, sh, colors_precomp, language_feature_precomp,
            opacities, scales, rotations, cov3Ds_precomp, self.raster_settings)


__all__ = ["GaussianRasterizationSettings", "GaussianRasterizer", "rasterize_gaussians"]
