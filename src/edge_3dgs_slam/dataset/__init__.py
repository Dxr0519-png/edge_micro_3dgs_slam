"""数据集加载器（真实数据评估用）。

- `ReplicaSequence`：Replica 数据集（SplaTAM 官方格式，traj.txt 为 c2w，加载器统一转 w2c）
"""
from .replica import DEPTH_SCALE, ReplicaSequence, load_cam_params, load_poses

__all__ = ["ReplicaSequence", "load_cam_params", "load_poses", "DEPTH_SCALE"]
