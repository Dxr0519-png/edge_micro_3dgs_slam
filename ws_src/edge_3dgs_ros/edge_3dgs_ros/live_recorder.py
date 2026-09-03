"""LiveRecorder：真机 SLAM 关键帧记录器（2026-09-02 语义场衔接）。

用途：跑完 SLAM 后用建出的图做开放词汇查询，需要"图 + 若干带位姿的帧"喂给
Phase 6 特征提取。记录对象 = 建图用的关键帧（backend on_keyframe 钩子，锁外、
~ms 级开销，不影响主路径）。

产物 npz 与回放/提取工具同契约（experiments/phase4_replay_publisher.py 同款）：
    rgb    (N,H,W,3) uint8
    depth  (N,H,W) float32 米（无效像素 0，沿用相机层语义）
    poses  (N,4,4) float64 世界→相机（w2c，项目统一约定，与地图同世界系）
    K      (3,3) float64
    t      (N,) float 秒（相机时间戳）

环形保留最近 max_frames 帧（关键帧已自带视角多样性；语义提取按索引挑几十帧
即可，13s/帧 @Jetson 提取成本）。
"""
from __future__ import annotations

import time

import numpy as np


class LiveRecorder:
    def __init__(self, max_frames: int = 150):
        self.max_frames = max_frames
        self.rgbs: list[np.ndarray] = []
        self.depths: list[np.ndarray] = []
        self.poses: list[np.ndarray] = []
        self.stamps: list[float] = []
        self.K: np.ndarray | None = None
        self.n_recorded = 0          # 累计入队（含被环形淘汰的）
        self.n_dropped = 0           # 因超上限被淘汰数

    def add(self, frame, T_wc: np.ndarray):
        """关键帧回调（backend on_keyframe）：rgb/depth 拷贝 + 位姿入环形缓冲。"""
        self.n_recorded += 1
        if self.K is None:
            self.K = np.asarray(frame.K, np.float64).copy()
        self.rgbs.append(np.asarray(frame.rgb, np.uint8).copy())
        self.depths.append(np.asarray(frame.depth, np.float32).copy())
        self.poses.append(np.asarray(T_wc, np.float64).copy())
        self.stamps.append(float(getattr(frame, "stamp", time.time())))
        if len(self.rgbs) > self.max_frames:
            self.rgbs.pop(0)
            self.depths.pop(0)
            self.poses.pop(0)
            self.stamps.pop(0)
            self.n_dropped += 1

    def save(self, path: str) -> dict:
        """落盘 npz（np.savez_compressed：rgb/depth 压缩，~0.5-1MB/帧）。"""
        n = len(self.rgbs)
        if n == 0 or self.K is None:
            return {"frames": 0}
        np.savez_compressed(
            path,
            rgb=np.stack(self.rgbs),
            depth=np.stack(self.depths),
            poses=np.stack(self.poses),
            K=self.K,
            t=np.asarray(self.stamps, np.float64),
        )
        return {"frames": n, "dropped": self.n_dropped,
                "path": path, "recorded": self.n_recorded}
