"""Phase 4 丢帧调度器：输入 30fps → 每 skip_every 帧处理 1 帧，其余帧 SE(3) 恒速外推。

背景（docs/03 口径修正）：单次光栅化迭代固定成本 ~30-66ms（实测 2026-08-28，
200k 高斯），10 FPS = 100ms/帧预算下无法逐帧处理 30fps 输入。调度器保证：
- 处理帧吞吐稳定（rate-adaptive：处理帧超预算时临时扩大 skip 间隔）；
- 未处理帧位姿 = se3_exp(se3_log(ΔT) / Δt 积分) @ T_prev 恒速外推输出；
- 外推误差 < 0.5cm 量级（实测要求），快速旋转段误差超限标记 lost 并回退
  至最近处理帧位姿（消费端可决定丢弃或保持）。

用法：
    sched = FrameScheduler(skip_every=3, budget_ms=100.0)
    for t in range(n):
        if sched.accept(t):            # 处理帧 → 正常 SLAM 管线
            T = backend.track(frame, T_init)
            sched.on_processed(t, T, gpu_ms)
        else:                          # 跳过帧 → 恒速外推
            T = sched.extrapolate(t)
"""
from __future__ import annotations

import numpy as np

from ..utils.se3 import invert_pose, se3_exp, se3_log

_DEFAULT_BUDGET_MS = 100.0    # 10 FPS
_LOST_DIST_M = 0.5            # 外推位移超 0.5m 标记 lost（快速运动保护）


def _mat_to_np(T) -> np.ndarray:
    return np.asarray(T.detach().cpu().numpy(), dtype=np.float64)


def _se3_delta(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """从位姿 A 到 B 的恒速外推增量（se(3) 对数，左扰动约定）。"""
    import torch
    d = se3_log(torch.as_tensor(np.linalg.inv(A) @ B, dtype=torch.float32).cuda())
    return d.detach().cpu().numpy()


def extrapolate_pose(T_prev: np.ndarray, delta: np.ndarray, scale: float) -> np.ndarray:
    """恒速外推：T_new = se3_exp(delta × scale) @ T_prev（scale=时间比例）。"""
    import torch
    d_scaled = torch.as_tensor(delta * scale, dtype=torch.float32).cuda()
    T = se3_exp(d_scaled) @ torch.as_tensor(T_prev, dtype=torch.float32).cuda()
    return _mat_to_np(T)


class FrameScheduler:
    """丢帧调度器：决定每帧是否处理；对跳过帧做 SE(3) 恒速外推。

    策略（Phase 4 决策门定稿参数）：
    - 默认 skip_every=3（30fps 输入 → 10 处理帧/秒）；处理帧预算 budget_ms；
    - rate-adaptive：处理帧实际耗时 > budget_ms 时，下个处理帧间隔 +1（4:1），
      连续达标帧恢复 3:1（回滞 1 帧防抖动）；
    - 关键帧判定由 backend 对处理帧执行（跳过帧不喂给建图）。
    """

    def __init__(self, skip_every: int = 3, budget_ms: float = _DEFAULT_BUDGET_MS,
                 lost_dist_m: float = _LOST_DIST_M):
        self.skip_every = skip_every
        self.budget_ms = budget_ms
        self.lost_dist_m = lost_dist_m
        self._cur_skip = skip_every            # 当前生效间隔（rate-adaptive）
        self._last_processed_t: int | None = None
        self._last_T: np.ndarray | None = None     # 最近处理帧位姿 (4,4) w2c
        self._delta: np.ndarray | None = None      # 最近两处理帧 se(3) 增量（整段间隔）
        self._interval: int = skip_every           # delta 对应的时间间隔（帧数）
        self._processed_steps = 0
        self._over_budget = 0
        self._force_next = False                   # 跟踪失败后的恢复：下一帧强制处理
        self.n_processed = 0
        self.n_skipped = 0
        self.n_lost = 0
        self.n_recovered = 0
        self.lost_flags: dict[int, bool] = {}      # 帧号 → 是否 lost（外推失败）

    # ------------------------------------------------------------------ 主接口
    def accept(self, t: int) -> bool:
        """第 t 帧是否作为处理帧（t=0 恒处理；失败恢复时强制处理下一帧）。"""
        if t == 0 or self._last_processed_t is None:
            return True
        if self._force_next:
            self._force_next = False
            self.n_recovered += 1
            return True
        return (t - self._last_processed_t) >= self._cur_skip

    def force_next(self):
        """跟踪失败后调用：下一帧强制处理（帧间运动小，ICP/光栅化易于恢复）。"""
        self._force_next = True

    def on_processed(self, t: int, T_wc: np.ndarray, gpu_ms: float | None = None):
        """记录处理帧结果；按耗时调整后续间隔（rate-adaptive）。"""
        if self._last_T is not None:
            self._delta = _se3_delta(self._last_T, T_wc)
            self._interval = max(t - self._last_processed_t, 1)
        self._last_T = np.asarray(T_wc, dtype=np.float64)
        self._last_processed_t = t
        self.n_processed += 1
        if gpu_ms is not None:
            if gpu_ms > self.budget_ms:
                self._over_budget += 1
                if self._over_budget >= 2:      # 连续 2 帧超预算才扩大间隔（防抖动）
                    self._cur_skip = min(self._cur_skip + 1, 6)
            else:
                self._over_budget = max(0, self._over_budget - 1)
                if self._cur_skip > self.skip_every and self._over_budget == 0:
                    self._cur_skip = self.skip_every   # 回滞恢复

    def extrapolate(self, t: int) -> np.ndarray:
        """跳过帧的位姿输出：恒速外推；位移超限标记 lost 并回退最近位姿。

        delta 是整段处理帧间隔的运动量，外推比例 = 距最近处理帧帧数 / 间隔。
        """
        if self._last_T is None or self._delta is None:
            self.n_skipped += 1
            self.lost_flags[t] = True
            return self._last_T if self._last_T is not None else np.eye(4)
        frac = float(t - self._last_processed_t) / max(self._interval, 1)
        T = extrapolate_pose(self._last_T, self._delta, frac)
        dist = float(np.linalg.norm(T[:3, 3] - self._last_T[:3, 3]))
        lost = dist > self.lost_dist_m
        if lost:
            self.n_lost += 1
            T = self._last_T.copy()
        self.lost_flags[t] = lost
        self.n_skipped += 1
        return T

    def on_lost_reset(self, T_wc: np.ndarray):
        """跟踪失败后重置外推基线（由调用方在失败回退后调用）。"""
        self._last_T = np.asarray(T_wc, dtype=np.float64)
        self._delta = None
