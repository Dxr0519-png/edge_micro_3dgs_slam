"""SSIM 计算（裁剪自 SplaTAM utils/slam_external.py，无额外依赖）。

SplaTAM 的 calc_ssim 来自 inria gaussian-splatting，标准 11x11 高斯窗口实现。
"""
import torch
import torch.nn.functional as F


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor(
        [__import__("math").exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


_WINDOW_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def calc_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
              size_average: bool = True) -> torch.Tensor:
    """SSIM（与 SplaTAM 一致）。

    输入: (3, H, W) 的 0~1 浮点图（与 rgb 渲染输出一致），要求 channel 在前。

    Phase 4：窗口张量按 (window_size, channel) 模块级缓存——track 每迭代调用
    一次，重建 11×11 窗口的 CPU 计算与 .cuda() 拷贝是纯重复开销（实测 ~1ms）。
    """
    (_, channel, _, _) = img1.size()
    key = (window_size, channel)
    window = _WINDOW_CACHE.get(key)
    if window is None or window.device != img1.device:
        window = _create_window(window_size, channel).cuda(img1.device)
        _WINDOW_CACHE[key] = window
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
