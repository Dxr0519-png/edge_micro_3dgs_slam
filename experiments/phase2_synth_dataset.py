#!/usr/bin/env python3
"""Phase 2 §7 合成 RGB-D 序列生成器（Replica 降级方案）。

程序化场景 + 独立几何光栅化（z-buffer 平面/球/盒求交 + Lambert 着色），
生成带真值位姿的 RGB-D 序列。**真值不依赖被测的可微光栅化器**，
可严格验证 Tracking/Mapping 的正确性。

场景（世界系 = 首帧相机系，x 右 / y 下 / z 前）：
    一面远墙 + 地板 + 三面侧墙 + 若干彩色球/盒，相机沿弧线轨迹环绕场景。
    深度范围 1.5~8m，模拟 D435i 有效量程。

用法：
    python3 experiments/phase2_synth_dataset.py [--frames 60] [--out ...]

输出 npz：
    rgb (T,H,W,3) uint8 | depth (T,H,W) float32 米 | K (3,3) | poses (T,4,4) w2c
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

# ---------------------------------------------------------------- 场景图元
class Plane:
    """n·p = d 无限平面（n 指向可见侧）。"""
    def __init__(self, n, d, color, checker=False):
        self.n = np.asarray(n, np.float64); self.d = float(d)
        self.color = np.asarray(color, np.float64)
        self.checker = checker

    def intersect(self, o, r):
        """返回 (t, normal)；无交为 inf。o/r: (H,W,3)。"""
        denom = np.einsum("hwj,j->hw", r, self.n)
        t = (self.d - np.einsum("hwj,j->hw", o, self.n)) / denom
        t[(denom > -1e-12) & (denom < 1e-12)] = np.inf    # 平行无交
        t[denom > 1e-12] = np.inf                         # 只取正面（n 朝向相机侧）
        t[t < 0] = np.inf                                 # 相机后方的交点
        return t, np.broadcast_to(self.n, o.shape)


class Sphere:
    def __init__(self, c, radius, color):
        self.c = np.asarray(c, np.float64); self.r = float(radius)
        self.color = np.asarray(color, np.float64)

    def intersect(self, o, r):
        oc = o - self.c
        b = np.einsum("hwj,hwj->hw", oc, r)
        cc = np.einsum("hwj,hwj->hw", oc, oc) - self.r ** 2
        disc = b * b - cc
        t = np.full(o.shape[:2], np.inf)
        m = disc > 0
        sq = np.sqrt(disc[m])
        t1, t2 = -b[m] - sq, -b[m] + sq
        hit = (t1 > 0) | (t2 > 0)
        t[m] = np.where(t1 > 0, t1, t2)
        n = np.zeros(o.shape)
        n[m] = (o[m] + t[m, None] * r[m] - self.c) / self.r
        return t, n


class Box:
    """轴对齐盒（slab 求交），颜色带简单面差异。"""
    def __init__(self, lo, hi, color):
        self.lo = np.asarray(lo, np.float64); self.hi = np.asarray(hi, np.float64)
        self.color = np.asarray(color, np.float64)

    def intersect(self, o, r):
        H, W = o.shape[:2]
        inv = 1.0 / r
        t1 = (self.lo - o) * inv
        t2 = (self.hi - o) * inv
        tmin = np.minimum(t1, t2); tmax = np.maximum(t1, t2)
        t_enter = tmin.max(axis=-1)
        t_exit = tmax.min(axis=-1)
        t = np.full((H, W), np.inf)
        m = (t_enter < t_exit) & (t_exit > 0)
        t[m] = np.where(t_enter[m] > 0, t_enter[m], t_exit[m])
        # 命中面法线：由进入面决定
        n = np.zeros(o.shape)
        if m.any():
            idx = tmin[m].argmax(axis=-1)          # 最晚进入的面 = 第一个交点面
            for j in range(3):
                side = idx == j
                if side.any():
                    enter_lo = t1[m][side, j] > t2[m][side, j]   # 从 hi 面进入
                    n[m][side, j] = np.where(enter_lo, 1.0, -1.0)
        return t, n


def make_scene():
    """房间 + 家具。返回图元列表与轨迹圆心/半径。"""
    room = [
        # 法线统一指向房间内部（单面渲染，可见侧朝内），d = n·平面位置。
        # 地板/远墙加棋盘纹理（纯色平面 Lambert 无明暗变化 → 图像完全无纹理，
        # tracking 的 loss 地形平坦，单帧精度受限；棋盘模拟真实场景纹理）。
        Plane((0, 0, -1), -8.0, (0.82, 0.80, 0.76), checker=True),   # 远墙 z=8
        Plane((0, -1, 0), -1.2, (0.55, 0.52, 0.48), checker=True),   # 地板 y=1.2
        Plane((0, 1, 0), -1.2, (0.75, 0.72, 0.68)),     # 天花板 y=-1.2
        Plane((-1, 0, 0), -2.5, (0.70, 0.42, 0.40)),    # 右墙 x=2.5
        Plane((1, 0, 0), -2.5, (0.42, 0.52, 0.70)),     # 左墙 x=-2.5
        Plane((0, 0, 1), 2.0, (0.60, 0.66, 0.72)),      # 近墙 z=2
    ]
    objects = [
        Sphere((0.6, 0.55, 4.5), 0.55, (0.85, 0.25, 0.20)),
        Sphere((-0.7, 0.35, 5.2), 0.35, (0.20, 0.65, 0.35)),
        Box((-1.5, -0.45, 3.8), (-0.9, 0.55, 4.6), (0.25, 0.40, 0.80)),
        Box((0.2, 0.35, 6.2), (1.0, 1.15, 7.0), (0.90, 0.70, 0.20)),
        Sphere((0.0, -0.15, 6.8), 0.45, (0.60, 0.35, 0.75)),
    ]
    return room + objects


# ---------------------------------------------------------------- 相机轨迹
def lookat(cam_pos, target, up=(0, -1, 0)):
    """构造 c2w（OpenCV 约定：x 右 y 下 z 前），相机朝向 target。"""
    cam_pos = np.asarray(cam_pos, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    z = target - cam_pos; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], axis=1)
    c2w[:3, 3] = cam_pos
    return c2w


def trajectory(n_frames, center=(0.5, 0.3, 5.0), radius=1.3):
    """沿环绕弧线生成位姿（世界系 = 首帧相机系），相机在场景侧面环绕中心，
    全程朝向中心看（场景物体 z∈[3.8, 7]，不会被轨迹遮挡）。返回 w2c 序列。

    注意：lookat 退化的条件是**前向 ∥ up**，即前向的 x 与 z 分量同时为 0。
    因此相机路径 x 不得等于 center.x 且 z 不得固定（本实现 z 随弧线升降，
    前向 z 分量恒非零 → 永不退化；曾因 z 固定 + 路径过 center 正上方产生
    ~180° 位姿翻转）。
    """
    poses = np.zeros((n_frames, 4, 4))
    for t in range(n_frames):
        ang = np.pi * (0.15 + 0.7 * t / max(n_frames - 1, 1))   # 半圈 27°~153°
        cam_pos = (center[0] + radius * np.cos(ang),
                   center[1] - 0.35 + 0.35 * np.sin(ang * 2),
                   center[2] + 0.5 * np.sin(ang))              # z 升降，避免 lookat 退化
        c2w = lookat(cam_pos, center)
        poses[t] = np.linalg.inv(c2w)
    return poses


# ---------------------------------------------------------------- 渲染
def render_frame(prims, c2w, K, H, W, ambient=0.35, light_dir=(0.4, -0.6, 0.6)):
    """几何光栅化单帧：z-buffer 求交 + Lambert 着色。

    深度语义（重要）：输出的是**相机系 z**（前向深度），不是光线欧氏距离。
    求交得到的是光线参数 t（欧氏距离），相机系 z = t · cosθ = t / |d_raw|
    （θ 为像素与光轴的夹角，d_raw = ((u-cx)/fx, (v-cy)/fy, 1) 未归一化方向）。
    此前版本直接输出 t，边缘像素（半 FOV 27°）偏大 12%，3.5m 处错 ~40cm，
    导致 GT 深度与 backproject/光栅化器渲染（均用相机系 z）系统性不一致。
    """
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d_raw = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, dtype=np.float64)], axis=-1)
    d_norm = np.linalg.norm(d_raw, axis=-1, keepdims=True)
    d = d_raw / d_norm                       # 单位光线方向（相机系）
    R = c2w[:3, :3]; t = c2w[:3, 3]
    r = d @ R.T                              # 世界系光线方向
    o = np.broadcast_to(t, d.shape)          # 相机位置 = c2w 平移

    best_t = np.full((H, W), np.inf)
    best_color = np.zeros((H, W, 3))
    best_n = np.zeros((H, W, 3))
    for p in prims:
        t_hit, n_hit = p.intersect(o, r)
        m = t_hit < best_t
        if m.any():
            best_t[m] = t_hit[m]
            col = p.color
            if getattr(p, "checker", False):
                hit = o[m] + t_hit[m, None] * r[m]         # 命中点世界坐标
                cell = np.floor(hit[:, 0] / 0.4) + np.floor(hit[:, 1] / 0.4)
                col = p.color * np.where((cell % 2 == 0)[:, None], 1.0, 0.45)
            best_color[m] = col
            best_n[m] = n_hit[m]

    ld = np.asarray(light_dir, np.float64); ld /= np.linalg.norm(ld)
    shade = np.clip(np.einsum("hwj,j->hw", best_n, ld), 0, 1)
    color = np.clip(best_color * (ambient + (1 - ambient) * shade[..., None]), 0, 1)
    depth = best_t / d_norm.squeeze(-1)      # 相机系 z = t · cosθ
    depth[~np.isfinite(depth)] = 0.0
    return color, depth


def generate(n_frames=60, H=240, W=320, seed=0, out=None):
    fx, fy = 310.0, 310.0
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    prims = make_scene()
    poses = trajectory(n_frames)                     # w2c

    rgb = np.empty((n_frames, H, W, 3), np.uint8)
    depth = np.empty((n_frames, H, W), np.float32)
    for t in range(n_frames):
        c2w = np.linalg.inv(poses[t])
        color, d = render_frame(prims, c2w, K, H, W)
        rgb[t] = (color * 255).astype(np.uint8)
        depth[t] = d.astype(np.float32)
        if t % 10 == 0:
            print(f"  [synth] 帧 {t}: 有效深度占比 {(d > 0).mean() * 100:.0f}%")

    out = Path(out if out else "data/outputs/phase2/synth_scene.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rgb=rgb, depth=depth, K=K, poses=poses)
    print(f"[synth] 已保存 {n_frames} 帧 → {out} (rgb {rgb.shape}, depth {depth.shape})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", type=str, default="data/outputs/phase2/synth_scene.npz")
    args = ap.parse_args()
    generate(n_frames=args.frames, out=args.out)
