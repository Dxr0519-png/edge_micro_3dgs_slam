#!/usr/bin/env python3
"""Phase 1 可视化：点云效果查看。

无头渲染（默认，保存 PNG，VSCode/浏览器直接看）：
    python3 experiments/phase1_view_pointcloud.py
交互窗口（需图形界面，如 Jetson 桌面 :1）：
    DISPLAY=:1 python3 experiments/phase1_view_pointcloud.py --window

输出：data/outputs/phase1/view_grid.png（2x2：原图 / 深度 / 点云重投影 / 侧视点云）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # 无头渲染，不依赖 DISPLAY
import matplotlib.pyplot as plt
from matplotlib import font_manager
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_3dgs_slam.camera import CAMERA_TO_ZUP

# 注册系统 CJK 字体（标题含中文）
for f in font_manager.findSystemFonts():
    if "CJK" in f or "uming" in f:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/outputs/phase1"
PLY = OUT / "frame_aligned.ply"


def load_ply(path: Path):
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)        # 0-1
    return pts, colors


def render_from_camera(pts, colors, K, H=480, W=640):
    """把点云按相机位姿重投影渲染成 RGB 图（z-buffer 近处覆盖），与真图对比验证对齐。"""
    z = pts[:, 2]
    u = (pts[:, 0] * K[0, 0] / z + K[0, 2]).astype(np.int64)
    v = (pts[:, 1] * K[1, 1] / z + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    u, v, z, c = u[inside], v[inside], z[inside], colors[inside]
    idx = v * W + u
    zbuf = np.full(H * W, np.inf)
    np.minimum.at(zbuf, idx, z)            # 每像素最近深度
    keep = z == zbuf[idx]
    img = np.zeros((H * W, 3))
    img[idx[keep]] = c[keep]
    return (img.reshape(H, W, 3) * 255).astype(np.uint8)


def main():
    # ply 存的是世界系（z-up，x 右 / y 前 / z 上）；渲染需转回相机系
    pts_world, colors = load_ply(PLY)
    pts_cam = pts_world @ CAMERA_TO_ZUP      # 旋转矩阵正交，转置即逆
    import cv2  # frame_rgb.png 由 cv2 以 BGR 存储，读回时转 RGB
    rgb = cv2.cvtColor(cv2.imread(str(OUT / "frame_rgb.png")), cv2.COLOR_BGR2RGB)
    depth = np.load(OUT / "frame_depth.npy")

    if "--window" in sys.argv:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name="Phase1 point cloud (z-up)")
        return

    # 相机系 K（与验证实验一致：aligned depth 已注册到颜色系）
    K = np.array([[607.6136, 0, 331.1368],
                  [0, 607.7604, 236.0332],
                  [0, 0, 1]])
    reproj = render_from_camera(pts_cam, colors, K)

    # 侧视角（世界系，绕 Y 轴旋转 50° + 俯仰 25°）
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pts_world), size=min(60000, len(pts_world)), replace=False)
    c, s = np.cos(np.radians(50)), np.sin(np.radians(50))
    rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = np.cos(np.radians(25)), np.sin(np.radians(25))
    rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    pts_side = pts_world[sel] @ rot_y.T @ rot_x.T
    col_side = colors[sel]

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax[0, 0].imshow(rgb);           ax[0, 0].set_title("1. 原图 RGB（D435i）")
    ax[0, 1].imshow(depth, cmap="viridis", vmin=0.2, vmax=3.8)
    ax[0, 1].set_title("2. 深度图（米，viridis）")
    ax[1, 0].imshow(reproj);        ax[1, 0].set_title("3. 点云重投影渲染（应与 1 一致 → 对齐证明）")
    ax[1, 1].scatter(pts_side[:, 0], pts_side[:, 1], c=col_side, s=0.4)
    ax[1, 1].set_title("4. 点云侧视（RGB 着色，60k 采样）")
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Phase 1 · 点云验证（{len(pts_world)} 点）", fontsize=14)
    fig.tight_layout()
    out = OUT / "view_grid.png"
    fig.savefig(out, dpi=110)
    print(f"已保存: {out}")
    print(f"点云点数: {len(pts_world)}")


if __name__ == "__main__":
    main()
