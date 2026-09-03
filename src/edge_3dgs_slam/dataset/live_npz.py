"""LiveNpzSequence：LiveRecorder 产物 frames.npz → ReplicaSequence 同接口序列。

2026-09-02 语义场衔接：真机 SLAM 记录的关键帧 npz（rgb/depth/poses w2c/K，见
ws_src/edge_3dgs_ros/edge_3dgs_ros/live_recorder.py 契约）对 Phase 6 提取/蒸馏
暴露与 ReplicaSequence 一致的接口（frame / frame_scaled / poses_w2c / cam），
使 phase6_semantic_live.py 无需改动即可复用下游监督构建逻辑。位姿与地图天然
同世界系（都是 SLAM 输出），省去 Replica 的坐标对齐。
"""
from __future__ import annotations

import numpy as np

from ..camera import SyncedFrame
from ..utils.frame_utils import downsample_frame


class LiveNpzSequence:
    """npz → 按帧索引访问的序列（np.load 惰性，不整包拷贝）。"""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)                       # npz 惰性（不关文件，进程级长存）
        self.rgb_arr = d["rgb"]                     # (N,H,W,3) uint8
        self.depth_arr = d["depth"]                 # (N,H,W) float32 米
        self.poses = d["poses"]                     # (N,4,4) w2c
        self.K = np.asarray(d["K"], np.float64)
        self.t = d["t"] if "t" in d else None
        n = self.rgb_arr.shape[0]
        self.cam = type("Cam", (), {"width": self.rgb_arr.shape[2],
                                    "height": self.rgb_arr.shape[1]})()

    def __len__(self) -> int:
        return self.rgb_arr.shape[0]

    def frame(self, t: int) -> SyncedFrame:
        """第 t 帧全分辨率 SyncedFrame。"""
        return SyncedFrame(rgb=self.rgb_arr[t], depth=self.depth_arr[t],
                           K=self.K, stamp=float(self.t[t]) if self.t is not None else 0.0)

    def frame_scaled(self, t: int, scale: float = 0.5) -> SyncedFrame:
        """第 t 帧降采样（K 同步缩放，与 ReplicaSequence.frame_scaled 语义一致）。"""
        return downsample_frame(
            self.frame(t),
            out_W=int(round(self.rgb_arr.shape[2] * scale)),
            out_H=int(round(self.rgb_arr.shape[1] * scale)))

    @property
    def poses_w2c(self) -> np.ndarray:
        return self.poses
