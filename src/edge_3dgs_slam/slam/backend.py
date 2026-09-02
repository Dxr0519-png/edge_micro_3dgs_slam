"""Phase 3 §5 异步建图后端：Tracking 实时回调永不阻塞，Mapping 由 daemon 线程按关键帧节奏消费。

GPU 串行化（docs/03 §9 坑 1）：单把 self._gpu_lock 串行化所有 GPU 调用；两线程共用
默认 CUDA stream，锁 + stream 语义保证 launch 串行、模型状态无竞态。
可选 lock_chunk：map 属性优化每 N 迭代释放一次锁，让 tracking 在长建图期间可插入
（实测追踪延迟超标时启用）。

丢帧策略（文档 §5）：队列满则丢最旧，优先保 Tracking。
滑动窗口：deque(maxlen=window_size)，出窗的关键帧不再参与建图（冻结）。
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from ..camera import SyncedFrame
from ..gaussian.model import GaussianModel
from ..utils.frame_utils import downsample_frame
from ..utils.profiling import profiled
from .icp_init import icp_init, icp_init_model, icp_render_model_depth
from .keyframe import KeyframeManager
from .mapping import map_keyframe
from .tracking import track


class SLAMBackend:
    def __init__(self, model: GaussianModel, K, track_kwargs: dict | None = None,
                 map_kwargs: dict | None = None,
                 dt_thresh: float = 0.1, dth_thresh_deg: float = 3.0,
                 queue_maxsize: int = 8, window_size: int = 5,
                 track_W: int = 320, track_H: int = 240,
                 lock_chunk: int | None = None, lock_sleep_ms: float = 0.02,
                 init_mode: str = "const", icp_W: int = 160, icp_H: int = 120):
        """异步 SLAM 后端。

        参数:
            model: 高斯模型（track/map 原地更新）
            K:     内参（track 降采样帧的 K 由 downsample_frame 缩放，无需在此处理）
            track_kwargs: 传给 track() 的 kwargs（iters/lr/cull/res_schedule 等）。
                    **含 res_schedule 时传全分辨率帧**——分阶段降采样由 track 内部
                    完成（多分辨率档位下 backend 的固定 320×240 降采样会破坏低档位）；
                    无 res_schedule 时走原路径（先降采样再 track，行为不变）
            map_kwargs:   传给 map_keyframe() 的 kwargs（iters/density_r/capacity_max/cull 等）
            queue_maxsize: Mapping 队列上限（文档 §5：8）
            window_size:   关键帧滑动窗口大小（文档 §5：5~8）
            track_W/H:     Tracking 降采样分辨率（文档 §6）
            lock_chunk:    map 优化每 N 迭代释放一次锁；None 不释放（整段持锁）
            lock_sleep_ms: chunk 释放窗口（文档 §9 坑 1：≥20ms 防 worker 抢回饿死
                    track；map 总持锁时间降下来后可调小到 10ms 让 track 插入更频繁）
        """
        self.model = model
        self.K = np.asarray(K, np.float64)
        self.track_kwargs = track_kwargs or {}
        self.map_kwargs = map_kwargs or {}
        self.kf_manager = KeyframeManager(dt_thresh, dth_thresh_deg)
        self.map_queue = queue.Queue(maxsize=queue_maxsize)
        self.keyframes = __import__("collections").deque(maxlen=window_size)
        self.track_W, self.track_H = track_W, track_H
        self.lock_chunk = lock_chunk
        self.lock_sleep_ms = lock_sleep_ms
        self._gpu_lock = threading.Lock()
        self._stop = threading.Event()
        self.last_keyframe_T = None
        self.init_mode = init_mode
        self.icp_W, self.icp_H = icp_W, icp_H
        self._icp_f2m_min = 0              # 0=纯 f2m（混合切换实测有害：f2f 前段正常但 80k 切 f2m 破坏链条，bd8-bd10）
        self._prev_depth_icp = None        # 上一处理帧深度（ICP 分辨率，帧到帧目标）
        self._last_T_np = None             # 上一处理帧位姿（ICP 的 T_prev）
        self._failure_event = False        # 跟踪失败事件（ICP 退化/大转角）
        self.stats = {"track_wall_ms": [], "track_gpu_ms": [], "dropped": 0, "mapped": 0,
                      "icp_ms": [], "icp_fallback": 0}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ 实时回调
    def track(self, frame: SyncedFrame, T_wc_init: np.ndarray) -> np.ndarray:
        """实时回调（主线程）：降采样追踪 → 关键帧判定 → 入队。永不阻塞（除 GPU 锁等待）。

        res_schedule 在 track_kwargs 时传全分辨率帧（分阶段降采样在 track 内部）；
        否则先降采样到 track_W/H（原路径）。

        Phase 4 修正：降采样（CPU cv2）移到 GPU 锁**外**——锁内 CPU 工作
        会串行化浪费 map 的 GPU 空闲（实测坑，docs/03 口径修正）。
        """
        wall_t0 = time.perf_counter()
        if "res_schedule" in self.track_kwargs:
            fd_frame = frame
        else:
            fd_frame = downsample_frame(frame, self.track_W, self.track_H)
        # Phase 4 ICP 粗对齐初值（混合路径）：
        # - 模型稀疏期（<80k 高斯，首帧初始化+少量关键帧）：帧到帧——1 视角
        #   模型的 depth 渲染与传感器系统性偏差大，f2m 会失败（实测 t=3 起
        #   连续失败）；帧到帧在稀疏期可靠（前 34 帧保持 2-3cm）。
        # - 模型稠密后：帧到模型——模型是全局锚，消除帧到帧对齐噪声的随机
        #   游走（1-2cm/步 × 50 帧累积到 6-10cm 后门槛连锁失效，实测）。
        # 退化/对应率塌陷 → 置失败事件：调度器下一帧强制处理（小运动恢复）
        if self.init_mode == "icp" and self._last_T_np is not None:
            t_icp = time.perf_counter()
            if self.model.num_gaussians < self._icp_f2m_min:
                T_init, st = icp_init(frame, self._prev_depth_icp, self._last_T_np,
                                      T_wc_init, self.K, W=self.icp_W,
                                      H=self.icp_H, return_stats=True)
            else:
                fd_icp = downsample_frame(frame, self.icp_W, self.icp_H)
                with self._gpu_lock:
                    d_model = icp_render_model_depth(self.model, self._last_T_np,
                                                     fd_icp.K, prep=None)
                T_init, st = icp_init_model(frame, d_model, self._last_T_np,
                                            T_wc_init, self.K, W=self.icp_W,
                                            H=self.icp_H, return_stats=True)
            self.stats["icp_ms"].append((time.perf_counter() - t_icp) * 1e3)
            if st["fallback"]:
                self.stats["icp_fallback"] += 1
            if st["fallback"] or st["ratio"] < 0.5:
                self._failure_event = True
        else:
            T_init = T_wc_init
        with self._gpu_lock:
            T_cuda, gpu_ms = profiled(track, fd_frame, self.model, T_init,
                                      **self.track_kwargs)
            self.stats["track_gpu_ms"].append(gpu_ms)
        T_np = T_cuda.detach().cpu().numpy()
        self.stats["track_wall_ms"].append((time.perf_counter() - wall_t0) * 1e3)
        # 帧到帧目标（稀疏期）：上一处理帧深度与位姿
        self._prev_depth_icp = downsample_frame(frame, self.icp_W, self.icp_H).depth
        self._last_T_np = T_np

        if self.last_keyframe_T is None or \
                self.kf_manager.should_insert(T_np, self.last_keyframe_T):
            self._push(frame, T_np)
            self.last_keyframe_T = T_np
        return T_np

    def _push(self, frame: SyncedFrame, T_np: np.ndarray):
        """入队：队列满则丢最旧（FIFO 头部）再入队，极端满丢本次。"""
        try:
            self.map_queue.put_nowait((frame, T_np))
        except queue.Full:
            try:
                self.map_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.map_queue.put_nowait((frame, T_np))
            except queue.Full:
                self.stats["dropped"] += 1

    # ------------------------------------------------------------------ worker
    def _loop(self):
        """worker 线程：消费关键帧队列，按滑动窗口执行建图。"""
        while not self._stop.is_set():
            try:
                frame, T = self.map_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self.keyframes.append((frame, T))
            if self.lock_chunk is None:
                with self._gpu_lock:
                    map_keyframe(frame, self.model, T,
                                 window=list(self.keyframes), **self.map_kwargs)
            else:
                self._map_keyframe_chunked(frame, T)   # 内部自管锁
            self.stats["mapped"] += 1

    def _map_keyframe_chunked(self, frame, T):
        """lock_chunk 模式：每 N 迭代释放一次锁，让 tracking 可插入。

        注意不能用 `with self._gpu_lock` 包裹（块退出时会二次 release）。
        释放后 sleep 1ms 给等待者（track）获锁机会，避免立即抢回。
        """
        base = {k: v for k, v in self.map_kwargs.items()}
        iters = base.pop("iters", 50)
        from .mapping import map_keyframe as _mk
        self._gpu_lock.acquire()
        try:
            n = 0
            while n < iters:
                chunk = min(self.lock_chunk, iters - n)
                _mk(frame, self.model, T, iters=chunk, window=list(self.keyframes), **base)
                n += chunk
                if n < iters:
                    self._gpu_lock.release()
                    # 释放窗口要 ≥ 线程调度延迟（实测 1ms 窗口 worker 立即抢回，
                    # track 被饿死，FPS 0.3）；默认 20ms 足够 track 获锁且总开销
                    # 可忽略——map 总持锁时间降下来后可调小（lock_sleep_ms）
                    time.sleep(self.lock_sleep_ms)
                    self._gpu_lock.acquire()
        finally:
            self._gpu_lock.release()

    def take_failure(self) -> bool:
        """取走并清除跟踪失败事件（ICP 退化/对应率塌陷，供调度器强制下一帧恢复）。"""
        v = self._failure_event
        self._failure_event = False
        return v

    def join(self, timeout: float = 120.0):
        """实验收尾：等队列清空 + 当前 map 完成 → 停线程。

        必须调用（否则 daemon 线程可能在建图中途随进程退出，CUDA 调用被打断）。
        """
        while not self.map_queue.empty():
            time.sleep(0.1)
        self._stop.set()
        self._thread.join(timeout)

    def ensure_feature_dim(self, d: int) -> bool:
        """Phase 6：在 _gpu_lock 内给模型补语言特征通道（在线模式查询前调用）。

        零初始化 (N,D)——在线特征未蒸馏，查询返回低置信但不崩；
        蒸馏是离线流程（Replica 验证），文档诚实注明在线查询需先蒸馏。
        """
        from ..gaussian import add_feature_dim
        with self._gpu_lock:
            add_feature_dim(self.model, d)
            return "features" in self.model.params

    def snapshot_model(self, max_points: int | None = None) -> dict | None:
        """锁内把模型参数拷成 numpy（Phase 4 供 ROS 发布 / Phase 6 查询服务复用）。

        锁内只做 .cpu() 拷贝（~5-15ms @200k），消息构建必须放锁外调用方。
        均匀抽稀到 max_points（stride 采样），None/未超限返回全量。

        返回 dict{means, rgb, opacity, scales, rot}（均 (N,*) numpy）或 None（空模型）。
        """
        with self._gpu_lock:                      # 与 track/map 的唯一 GPU 序列化点
            n = self.model.num_gaussians
            if n == 0:
                return None
            m = self.model
            out = {
                "means":   m.means3D.detach().float().cpu().numpy(),      # (N,3) 世界系
                "rgb":     m.params["rgb_colors"].detach().float().cpu().numpy(),
                "opacity": m.opacities().detach().float().cpu().numpy(),  # (N,1)
                "scales":  m.scales().detach().float().cpu().numpy(),     # (N,3) 米
                "rot":     m.rotations().detach().float().cpu().numpy(),  # (N,4) wxyz
            }
            # Phase 6：模型含语言特征通道时一并快照（查询服务复用）；无则不含
            if "features" in m.params:
                out["features"] = m.params["features"].detach().float().cpu().numpy()
        if max_points and n > max_points:
            stride = (n + max_points - 1) // max_points
            out = {k: v[::stride] for k, v in out.items()}
        return out
