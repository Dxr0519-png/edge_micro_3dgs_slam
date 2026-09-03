"""IMU 陀螺旋转先验（2026-09-02 v1）：跟踪初值旋转部分用陀螺积分替代恒速猜测。

动机（真机实测教训）：快速转头时恒速猜测会错 1-5°+（2 迭代光栅化跟踪起点差
→ 局部极小 → 漂移/双影）；陀螺给出物理真旋转，把跟踪起点拉回正确附近。
视觉每帧仍做完整对齐（光度+深度），**IMU 只提供初值**——长期漂移被视觉修正
拉回，不会积累（这是与纯积分位姿的本质区别）。

标定来源（semantic_ws kalibr 实测 2026-09-01，
realsense_stereo_imu_config_kalibr版备份.yaml）：
    R_IMU_CAM：IMU→左目(infra1) 旋转。3DGS 用 color 相机——同模组刚性安装，
    左目与 color 旋转差 <1°，直接沿用（作先验足够；未来可换 kalibr color 版）。
    TD_IMU：t_imu = t_cam + td（kalibr 锁定，estimate_td 已关）。
输入 IMU 应为**校正后**话题 /imu_corrected（imu_corrector.py：kalibr scale/
轴对齐 + g-灵敏度 + 尖峰滤波）——直接积分 raw /camera/imu 会把 ~1° 轴误差
带进先验。

约定与推导（w2c，左扰动，与 tracking/se3.py 一致）：
    相机绕本体轴 δ 旋转 ⇒ 世界点在相机系坐标反向转：R_wc' = Exp(-δ) @ R_wc
    （δ = 本体系旋转向量，转到 color 系：δ_c = R_cam_imu @ δ_imu，R_cam_imu =
    R_IMU_CAMᵀ）。平移沿用恒速外推的相机系平移分量。
    符号经真机验证后锁定（若校正旋角反增则翻号，见 node 仪表）。
"""
from __future__ import annotations

import numpy as np

# kalibr 2026-09-01：IMU→infra1（≈color，刚性模组）
R_IMU_CAM = np.array([
    [0.99997593, -0.00422398, -0.00550499],
    [0.00412065,  0.99981763, -0.01864727],
    [0.00558275,  0.01862414,  0.99981097],
])
TD_IMU = 0.002279            # t_imu = t_cam + td
R_CAM_IMU = R_IMU_CAM.T
MAX_GAP = 0.5                # 帧间 IMU 缺口超 0.5s 回退恒速（buffer 覆盖 2s）


def _exp_so3(w: np.ndarray) -> np.ndarray:
    """so(3) 指数映射（Rodrigues），返回旋转矩阵。"""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    wx = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    return np.eye(3) + np.sin(th) / th * wx + (1 - np.cos(th)) / th ** 2 * (wx @ wx)


class GyroRotationPrior:
    """从 D435iReader.imu_buffer（升序 (stamp, accel, gyro)）积分帧间旋转。

    integrate_body(t0, t1)：相机系旋转向量 δ_c（rad）——T_init 旋转增量 = -δ_c
    （左扰动 w2c 约定）；采样不足/缺口超限返回 None（调用方回退恒速）。
    """

    def __init__(self, imu_buffer):
        self.buf = imu_buffer        # deque[(stamp, accel(3), gyro(3))]

    def integrate_cam(self, t0: float, t1: float):
        """积分 [t0+td, t1+td] 窗口陀螺 → color 系旋转向量 δ_c (3,) rad。"""
        if t1 - t0 <= 1e-6 or t1 - t0 > MAX_GAP:
            return None
        ta, tb = t0 + TD_IMU, t1 + TD_IMU
        # 取窗口样本（buffer 升序，找跨窗口的首尾）
        arr = list(self.buf)
        if not arr:
            return None
        i0 = i1 = None
        for i, (st, _, g) in enumerate(arr):
            if i0 is None and st >= ta - 1e-4:
                i0 = i
            if st <= tb + 1e-4:
                i1 = i
        if i0 is None or i1 is None or i1 < i0:
            return None
        # 首样本前的部分用首样本角速度补（近似），逐段积分
        w_prev = arr[i0][2]
        t_prev = max(ta, arr[i0][0])
        R_acc = np.eye(3)
        for i in range(i0, i1 + 1):
            st, _, g = arr[i]
            dt = st - t_prev
            if dt > 0:
                R_acc = R_acc @ _exp_so3(np.asarray(w_prev, np.float64) * dt)
            w_prev, t_prev = g, st
        if tb > t_prev:            # 末端补
            R_acc = R_acc @ _exp_so3(np.asarray(w_prev, np.float64) * (tb - t_prev))
        # R_acc = 本体系旋转的累积（body 右乘）→ 旋转向量
        th = float(np.arccos(np.clip((np.trace(R_acc) - 1.0) / 2.0, -1.0, 1.0)))
        if th < 1e-9:
            return np.zeros(3)
        wx = (R_acc - R_acc.T) / (2 * np.sin(th)) * th
        w = np.array([wx[2, 1], wx[0, 2], wx[1, 0]])     # so3 log
        # IMU 系 → color 系
        return R_CAM_IMU @ w

    def init_pose(self, T_last: np.ndarray, t0: float, t1: float,
                  T_cv: np.ndarray | None = None) -> np.ndarray | None:
        """用陀螺旋转替换恒速初值 T_cv 的旋转部分（平移保留），无 IMU 返回 None。

        T_last: 最近处理帧位姿（w2c）；T_cv: 恒速外推初值（可 None → 平移 0）。
        """
        d_c = self.integrate_cam(t0, t1)
        if d_c is None:
            return None
        R_new = _exp_so3(-d_c) @ T_last[:3, :3]          # w2c 左扰动 -δ_c
        if T_cv is None:
            t_new = T_last[:3, 3]
        else:
            # 平移保持恒速的相机系增量（左扰动平移分量）：从 T_cv 相对 T_last 取
            R_l, t_l = T_last[:3, :3], T_last[:3, 3]
            d_se3 = np.eye(4)
            d_se3[:3, :3] = T_cv[:3, :3] @ R_l.T
            d_se3[:3, 3] = T_cv[:3, 3] - d_se3[:3, :3] @ t_l
            t_new = d_se3[:3, :3] @ t_l + d_se3[:3, 3]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_new, t_new
        return T
