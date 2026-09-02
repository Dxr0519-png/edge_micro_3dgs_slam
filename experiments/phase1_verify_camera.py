#!/usr/bin/env python3
"""Phase 1 验证实验：D435i 同步 → 反投影 → .ply → 回投误差/对齐检查。

运行前提：相机节点已启动（docs/01 §4 路线 B）：
    ros2 launch realsense2_camera rs_launch.py camera_namespace:=/ align_depth.enable:=true ...
用法：
    source /opt/ros/humble/setup.bash
    python3 experiments/phase1_verify_camera.py [目标帧数=30]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # 保证 import src/edge_3dgs_slam

import numpy as np
import rclpy

from edge_3dgs_slam.camera import D435iReader, SyncedFrame, backproject, project, to_zup_frame


class StopCollection(Exception):
    pass


class VerifyReader(D435iReader):
    def __init__(self, node, max_frames=30):
        super().__init__(node)
        self.frames: list[SyncedFrame] = []
        self.max_frames = max_frames

    def on_frame(self, frame):
        self.frames.append(frame)
        if len(self.frames) >= self.max_frames:
            self.node.get_logger().info(f"已收集 {len(self.frames)} 帧，停止采集")
            raise StopCollection


def main() -> int:
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rclpy.init()
    node = rclpy.create_node("phase1_verify")
    reader = VerifyReader(node, n_target)

    t0 = time.time()
    try:
        while time.time() - t0 < 30 and len(reader.frames) < n_target:
            rclpy.spin_once(node, timeout_sec=0.1)
    except StopCollection:
        pass
    node.destroy_node()
    rclpy.shutdown()

    frames = reader.frames
    if len(frames) < 5:
        print(f"FAIL: 只收到 {len(frames)} 帧 —— 相机节点是否在运行？")
        return 1

    # [1] 时间戳连续性（期望 ~33ms @30fps）
    stamps = np.array([f.stamp for f in frames])
    dt_ms = np.diff(stamps) * 1e3
    print(f"[1] 同步帧 {len(frames)} 帧；时间戳间隔 中位 {np.median(dt_ms):.1f} ms"
          f"（期望 ~33ms）")

    # [2] 形状与深度统计（取中间帧验证）
    f = frames[len(frames) // 2]
    H, W = f.depth.shape
    valid = np.isfinite(f.depth) & (f.depth > 0)
    z = f.depth[valid]
    print(f"[2] RGB {f.rgb.shape}（H,W,3 uint8）| Depth {H}x{W} float32；"
          f"有效像素 {valid.mean() * 100:.1f}%")
    if len(z):
        print(f"    深度范围 {z.min():.3f}–{z.max():.3f} m，均值 {z.mean():.3f} m")

    # [3] 反投影 + 回投（K 与数学自洽性；aligned_depth 由相机注册到颜色系，
    #     回投误差应 ≈ 0）
    pts, vmask = backproject(f.depth, f.K)
    pts_v = pts[vmask]
    uv = project(pts_v, f.K)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    orig = np.stack([uu[vmask], vv[vmask]], axis=-1).astype(np.float64)
    err = np.linalg.norm(uv - orig, axis=-1)
    print(f"[3] 反投影 {len(pts_v)} 点；回投误差 均值 {err.mean():.4f} px，"
          f"p99 {np.percentile(err, 99):.3f} px（期望 <0.1）")

    # [4] 存 .ply（带颜色，世界系 z-up：x 右 / y 前 / z 上，查看器打开即正立）并加载校验
    import open3d as o3d
    out_dir = Path(__file__).resolve().parents[1] / "data/outputs/phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "frame_aligned.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_zup_frame(pts_v))
    pcd.colors = o3d.utility.Vector3dVector(f.rgb[vmask].astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    pcd2 = o3d.io.read_point_cloud(str(ply_path))
    n_ply = len(pcd2.points)
    ply_ok = n_ply == len(pts_v) and len(pcd2.colors) == n_ply
    print(f"[4] .ply 保存+加载校验: {ply_path}（{n_ply} 点，含颜色）"
          f"{'OK' if ply_ok else 'FAIL: 点数不一致'}")

    # [5] 记录一帧 RGB/Depth 快照
    import cv2
    snap = out_dir / "frame_rgb.png"
    cv2.imwrite(str(snap), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
    np.save(out_dir / "frame_depth.npy", f.depth)

    # [6] 判定
    checks = [
        ("帧数 >= 5", len(frames) >= 5),
        ("间隔 ~33ms", 20 <= np.median(dt_ms) <= 45),
        ("回投误差均值 < 0.1px", err.mean() < 0.1),
        ("有效像素 > 5%", valid.mean() > 0.05),
        (".ply 校验通过", ply_ok),
    ]
    for name, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    print("==> Phase 1 camera 模块验证" + ("通过" if all(ok for _, ok in checks) else "有失败项"))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
