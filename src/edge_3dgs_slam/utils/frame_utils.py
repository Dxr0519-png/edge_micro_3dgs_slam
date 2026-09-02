"""Phase 3 §6 帧工具：Tracking/Mapping 分辨率分档。

Tracking 用降采样（320×240）提速，Mapping 用全分辨率（640×480）保质量。
"""
from __future__ import annotations

import cv2

from ..camera import SyncedFrame


def downsample_frame(frame: SyncedFrame, out_W: int = 320, out_H: int = 240) -> SyncedFrame:
    """Tracking 降采样帧。

    RGB 双线性插值；depth 最近邻（插值会伪造深度边缘，物理上不存在）；
    K 按逐轴比例缩放（fx/cx 按宽度比例、fy/cy 按高度比例——非等比目标
    （如 600×340→320×240）下单一比例会错误缩放 fy/cy，投影几何不一致，
    实测污染 ATE 口径；等比目标行为与旧实现逐元素一致）。
    """
    s_w = out_W / frame.rgb.shape[1]
    s_h = out_H / frame.rgb.shape[0]
    K_s = frame.K.copy()
    K_s[0] *= s_w
    K_s[1] *= s_h
    return SyncedFrame(
        rgb=cv2.resize(frame.rgb, (out_W, out_H), interpolation=cv2.INTER_LINEAR),
        depth=cv2.resize(frame.depth, (out_W, out_H), interpolation=cv2.INTER_NEAREST),
        K=K_s,
        stamp=frame.stamp,
    )
