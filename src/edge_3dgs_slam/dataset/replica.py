"""Replica 数据集加载器（SplaTAM 官方下载格式）。

数据布局（third_party/SplaTAM/data/Replica/，来自官方 bash_scripts/download_replica.sh）：
    Replica/<scene>/traj.txt          2000 行 × 16 列，每行 reshape(4,4) 为 **c2w**（官方语义）
    Replica/<scene>/results/          frame%06d.jpg（RGB）+ depth%06d.png（uint16）
    Replica/cam_params.json           w/h/fx/fy/cx/cy/scale（depth 除以 scale → 米）

位姿约定：本项目的 slam 管线（init_from_depth / track / build_map / track_one_frame）
统一使用 **w2c（世界→相机）**，故加载器读入 c2w 后求逆转 w2c（与 SplaTAM
datasets/gradslam_datasets/replica.py 的 c2w 语义 + 管线内求逆一致）。

其他约定：
- RGB 图 cv2.imread 得到 BGR，必须 cvtColor 转 RGB（SyncedFrame 约定 RGB）。
- 深度 scale=6553.5（uint16 → 米），无效像素为 0。
- 降采样由 `frame_scaled()` 提供（复用 utils.frame_utils.downsample_frame：
  RGB 双线性 + depth 最近邻 + K 同比例缩放）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..camera import CameraIntrinsics, SyncedFrame
from ..utils.frame_utils import downsample_frame

DEPTH_SCALE = 6553.5          # cam_params.json 的 scale：uint16 → 米


def load_cam_params(path: Path) -> CameraIntrinsics:
    """cam_params.json → CameraIntrinsics（字段在 "camera" 键下）。"""
    data = json.loads(Path(path).read_text())
    c = data["camera"]
    return CameraIntrinsics(
        fx=float(c["fx"]), fy=float(c["fy"]),
        cx=float(c["cx"]), cy=float(c["cy"]),
        width=int(c["w"]), height=int(c["h"]),
    )


def load_poses(path: Path, n: int, as_w2c: bool = True) -> np.ndarray:
    """traj.txt 前 n 行 → (n, 4, 4) 位姿。

    traj.txt 每行 16 个 float，行优先 reshape(4,4) = **c2w**（官方语义，SplaTAM
    replica.py 亦按 c2w 读取）；as_w2c=True 时求逆转 w2c（本项目管线约定）。
    """
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            vals = list(map(float, line.split()))
            if len(vals) != 16:
                raise ValueError(f"{path}:{i+1} 期望 16 列，实际 {len(vals)}")
            rows.append(np.array(vals).reshape(4, 4))
    poses = np.stack(rows)                        # (n,4,4) c2w
    if as_w2c:
        poses = np.linalg.inv(poses)
    return poses


@dataclass
class ReplicaSequence:
    """Replica 单场景序列（位姿已统一为 w2c）。"""

    root: Path                  # .../Replica/
    sequence: str               # "office0"
    cam: CameraIntrinsics       # 全分辨率内参（fx=600 fy=600 cx=599.5 cy=339.5 @1200x680）
    depth_scale: float = DEPTH_SCALE
    poses_w2c: np.ndarray = None  # (T,4,4) float64
    n_frames: int = 0

    @classmethod
    def from_dir(cls, root, sequence: str, max_frames: int | None = None) -> "ReplicaSequence":
        root = Path(root)
        seq_dir = root / sequence
        cam = load_cam_params(root / "cam_params.json")
        n = len(list(seq_dir.glob("results/frame*.jpg"))) if max_frames is None else max_frames
        poses = load_poses(seq_dir / "traj.txt", n, as_w2c=True)
        return cls(root=root, sequence=sequence, cam=cam, poses_w2c=poses, n_frames=n)

    @property
    def seq_dir(self) -> Path:
        return self.root / self.sequence

    @property
    def K(self) -> np.ndarray:
        """全分辨率 3x3 内参。"""
        return self.cam.K()

    def __len__(self) -> int:
        return self.n_frames

    def frame(self, t: int) -> SyncedFrame:
        """第 t 帧全分辨率 SyncedFrame（rgb RGB uint8 / depth 米 / K 全分辨率）。"""
        return SyncedFrame(
            rgb=_read_rgb(self.seq_dir / "results" / f"frame{t:06d}.jpg"),
            depth=_read_depth(self.seq_dir / "results" / f"depth{t:06d}.png", self.depth_scale),
            K=self.K,
            stamp=float(t),
        )

    def frame_scaled(self, t: int, scale: float = 0.5) -> SyncedFrame:
        """第 t 帧降采样 SyncedFrame（K 同比例缩放，语义不变）。

        scale=0.5 → 600x340，K'=[[300,0,299.75],[0,300,169.75],[0,0,1]]。
        """
        return downsample_frame(self.frame(t), out_W=int(round(self.cam.width * scale)),
                                out_H=int(round(self.cam.height * scale)))


def _read_rgb(path: Path) -> np.ndarray:
    """jpg → (H,W,3) uint8 RGB（cv2 默认 BGR，需转换；SyncedFrame 约定 RGB）。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_depth(path: Path, scale: float) -> np.ndarray:
    """depth png (uint16) → (H,W) float32 米；0 为无效像素。"""
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    return (d.astype(np.float32) / scale)
