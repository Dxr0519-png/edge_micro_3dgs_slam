"""Edge-3DGS-SLAM 主节点（docs/04 §2）：订阅 RGB-D → Tracking + 异步 Mapping → 发布位姿/高斯点云。

架构（Phase 4 定稿）：
- `Edge3DGSSlamNode(D435iReader)`：Phase 1 的同步订阅子类化复用，on_frame 每帧回调；
- `FrameScheduler` 丢帧（fps 档 2:1）+ SE(3) 恒速外推输出（每输入帧都发 tf/odom，30Hz 位姿流）；
- `SLAMBackend` 异步后端：track 轻量回调 + Mapping worker 线程（自带 _gpu_lock 串行化）；
- 高斯点云发布走独立定时器线程（1Hz），锁内快照 + 锁外建消息，绝不阻塞 sync 回调；
- `--load` 模式：加载现成 checkpoint 只 track 不建图（map_kwargs 置 no-op）；
  `--tier fps|quality` 双档参数抄 experiments/phase3_perf_ablation.py 定稿值（docs/03 §11）。

用法（docs/04 §5）：
    source /opt/ros/humble/setup.bash && source install/setup.bash
    ros2 run edge_3dgs_ros edge_3dgs_slam_node [--load ../data/outputs/phase3/probe_model_replica.pt] [--tier fps|quality]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from edge_3dgs_slam.camera import D435iReader, SyncedFrame
from edge_3dgs_slam.gaussian import GaussianModel
from edge_3dgs_slam.slam import SLAMBackend, init_from_depth
from edge_3dgs_slam.slam.imu_prior import GyroRotationPrior
from edge_3dgs_slam.slam.scheduler import FrameScheduler

from .cloud_publisher import CloudPublisher
from .live_recorder import LiveRecorder
from .query_service import QueryService
from .config import load_ros2_params
from .tf_publisher import TFPublisher

# 2026-09-02 语义场衔接：默认输出目录 = 仓库根 data/outputs/live（与 cwd 无关）。
# 运行的是 install 副本（ws_src/install/.../site-packages/...）时层级深，不能数
# parents——向上找同时含 src/ 与 ws_src/ 的目录即仓库根。
def _find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "src").is_dir() and (p / "ws_src").is_dir():
            return p
    return start


_REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
_DEFAULT_OUT = _REPO_ROOT / "data" / "outputs" / "live"

# ------------------------------------------------------------------ 双档参数
# 逐项核对 experiments/phase3_perf_ablation.py CONFIGS 定稿值（2026-08-28）：
# fps10 = 速度包络档（2:1 丢帧 + ICP 初值 + 2 迭代 d2 + light map + 收紧失败门）；
# track160/maplight = 质量锚档（6it d2 ad8 + full map 8it）。kf_dt/dth、lock_chunk 同源。
# 2026-09-02 真机提速修订（目标 >5 处理帧/s，见 docs/03 §11 口径）：
#   · adaptive_max 8→4：真机噪声下自适应几乎每帧扩到 8 次迭代（~350ms/帧，实测
#     /odom 2Hz 主因之一）；4 次把单帧最坏 ~160-200ms；
#   · budget_ms 100→200：100ms 预算单帧永远超（143ms+），调度器早已偷偷退到
#     4:1 丢帧——预算即真实帧成本时恢复 2:1；
#   · map iters 3→2 + lock_chunk 2→1：建图每关键帧减负、持锁窗口减半（backend
#     chunked 已去重播种重复）。lock_chunk 按档：quality 8 迭代用 2（原值，
#     每 chunk 迭代太多 Adam momentum 丢失反而伤质量），fps 短迭代用 1。
#   · adaptive_max 4→3→2：真机 loss"缓慢改善"判据无法区分噪声与收敛，自适应
#     恒扩满（run6 4.0it / run7 3.0it 实测）——真机场景每迭代 ~50-90ms（fill-
#     bound，近景大 splat），3 迭代 ~270ms 仍超 200ms 预算 → fps 档固定 2 迭代
#     （~120-180ms），skip 保持 2:1 处理率 ~4.5-5/s；难帧兜底由 ICP 门控
#     （watchdog 3 帧）+ 失败强制恢复承担，长序列质量锚仍走 quality 档。
#   · map depth_every=2：建图优化迭代 depth 隔趟（iters=2 → 1 趟 full + 1 趟
#     RGB-only，每 KF 光栅化 -25%）；容量 200k 封顶后播种原生渲染由 backend
#     at_capacity 跳过（稳态每 KF 再省 ~150-250ms）。
#   · 队列积压降档在 backend._loop（≥5 跳过整帧 / ≥3 纯播种低分辨率档）。
TIERS = {
    "fps": dict(
        skip_every=2, budget_ms=200.0, init_mode="icp",
        # 2026-09-02：lock_chunk=None——chunked 每段重建 Adam（动量清零）导致
        # opacity 永不收敛（实测 >0.9 占 0%，半透明叠影主因）；整段单 Adam
        lock_chunk=None,
        track_W=224, track_H=168,
        track_kwargs=dict(iters=2, depth_every=2, adaptive_max=2, cull=False,
                          early_stop_tol=1e-3,
                          fail_rot_deg=8.0, fail_trans_m=0.25),
        map_kwargs=dict(iters=2, density_r=0.01, capacity_max=200_000, cull=False,
                        map_tier="light", window_rotate=True, rotate_n=2,
                        opt_W=224, opt_H=168, depth_every=2),
    ),
    # quality（2026-09-02 真机建图修正）：fps 档 2 迭代跟踪在连续手持运动下每帧
    # 微晃累积 → 几何双影（实测深度偏差 19-44cm、渲染不可看）——建图质量必须走
    # 本档。真机修正：map 优化分辨率 320×240（原为 None=1280×720 原生，真机
    # 每 KF 数秒级 map 跟不上关键帧率会积压丢帧）＋ iters 8→6；跟踪 6it@320×240。
    # 用法：建图用 --tier quality（扫描放慢：旋转 ≤5°/s）；--tier fps 只保位姿输出率。
    "quality": dict(
        skip_every=1, budget_ms=300.0, init_mode="const",
        # 2026-09-02：lock_chunk=None（单 Adam 整段优化，透明度收敛——见 fps 档注）
        lock_chunk=None,
        track_W=320, track_H=240,
        track_kwargs=dict(iters=6, depth_every=2, adaptive_max=8, cull=False),
        # 2026-09-02 重影调优：map iters 6→10——20s 慢扫 PSNR 17-25dB 但仍有
        # 叠影（透明度/尺度未充分收敛）；10 迭代 × 窗口轮转 2 帧 @320×240，
        # 每 KF ~2-3s（慢扫 1KF/3-5s 可消化）
        map_kwargs=dict(iters=10, density_r=0.01, capacity_max=200_000, cull=False,
                        map_tier="full", window_rotate=True, rotate_n=2,
                        opt_W=320, opt_H=240),
    ),
}
# --load 模式禁用建图：map worker 消费队列时 iters=0 + 不新增/不剪枝（安全 no-op，
# 见 mapping.map_keyframe：add_new/prune 关闭后循环为空）。
_NO_MAP_KWARGS = dict(iters=0, add_new=False, prune=False)


def _load_checkpoint(path: str) -> GaussianModel:
    """加载 phase3 缓存格式 checkpoint（{'params', 'variables'} → CUDA）。"""
    s = torch.load(path, map_location="cpu", weights_only=False)
    params = {k: torch.nn.Parameter(v.cuda()) for k, v in s["params"].items()}
    return GaussianModel(params, variables=s.get("variables", {}))


class Edge3DGSSlamNode(D435iReader, Node):
    """多重继承：D435iReader 提供同步订阅+on_frame 回调，Node 提供 ROS 上下文。

    MRO = [Edge3DGSSlamNode, D435iReader, Node, object]——rclpy Node 的
    handle 等 property 必须经 MRO 可达（不能只调 Node.__init__ 而类不继承 Node）。
    """
    def __init__(self, args):
        self.cfg = load_ros2_params(args.params)
        Node.__init__(self, "edge_3dgs_slam_node")
        # 每个参数 declare_parameter（默认来自 cfg），兼容 --ros-args --params-file 覆盖
        for k, v in self.cfg.items():
            self.declare_parameter(k, v)
        D435iReader.__init__(
            self, self,
            slop=float(self.cfg["sync_slop_sec"]),
            queue_size=int(self.cfg["sync_queue_size"]),
            color_topic=self.cfg["rgb_topic"], depth_topic=self.cfg["depth_topic"],
            info_topic=self.cfg["color_info_topic"], imu_topic=self.cfg["imu_topic"])

        self._tier = TIERS[args.tier]
        self._load_path = args.load
        self._cap = args.cap                # --cap 容量覆盖（on_frame 首帧建 backend 用）
        self._seed_max = args.seed_max      # --seed-max 每 KF 播种上限（防早冻结）
        self._backend = None                # 惰性：K 要等 camera_info，模型要等首帧
        self._sched = FrameScheduler(skip_every=self._tier["skip_every"],
                                     budget_ms=self._tier["budget_ms"])
        self._frame_idx = 0
        self._last_T: np.ndarray | None = None
        self._prev_T: np.ndarray | None = None
        self._last_stamp: float | None = None
        self._model = None
        self._logged: set = set()
        self._perf_n = 0                  # 处理帧计数（每 30 帧打一次 perf 日志）
        self._perf_t0 = None

        # ---- 2026-09-02 IMU 陀螺旋转先验（初值旋转换陀螺积分，视觉仍做全量修正）----
        self._proc_T: np.ndarray | None = None     # 最近处理帧位姿（w2c）
        self._proc_stamp: float | None = None      # 最近处理帧时间戳
        self._corr_imu_deg: list = []              # 用 IMU 初值时的跟踪残差旋角
        self._corr_cv_deg: list = []               # 恒速初值时的跟踪残差旋角
        self._imu_hits = 0

        # ---- 2026-09-02 IMU 旋转先验实例（缓冲空/话题缺失时自动回退恒速）----
        self._imu_prior = GyroRotationPrior(self.imu_buffer)

        # ---- 2026-09-02 语义场衔接：关键帧记录 + 地图自动存盘 ----
        self._out_dir = Path(args.out) if args.out else _DEFAULT_OUT
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._recorder = LiveRecorder(max_frames=int(args.record_max)) \
            if int(args.record_max) > 0 else None
        self._autosave_sec = int(args.autosave_sec)
        self._autosave_group = ReentrantCallbackGroup()   # 存盘不与 sync 同组
        if self._autosave_sec > 0:
            self.create_timer(self._autosave_sec, self._autosave_timer,
                              callback_group=self._autosave_group)
        self.get_logger().info(
            f"产出目录 {self._out_dir}"
            + (f"，关键帧记录 ≤{args.record_max} 帧" if self._recorder else "，记录关闭")
            + (f"，地图自动存盘每 {self._autosave_sec}s" if self._autosave_sec > 0
               else "，自动存盘关闭"))

        self._tf_pub = TFPublisher(self, world_frame=self.cfg["pose_tf_frame"],
                                   odom_frame=self.cfg["odom_frame"],
                                   camera_frame=self.cfg["camera_frame"],
                                   publish_odom=bool(self.cfg["publish_odom"]))
        self._cloud_pub = CloudPublisher(
            self, publish_gaussian_map=bool(self.cfg["publish_gaussian_map"]),
            world_frame=self.cfg["pose_tf_frame"],
            max_points=int(self.cfg["cloud_max_points"]))
        # ⚠️ cloud 定时器放独立 Reentrant 回调组：GaussianCloud 构建 ~600-800ms
        # （Python 逐元素循环），若与 sync 回调同组（默认 MutuallyExclusive）会
        # 串行阻塞 on_frame → tf 饿死到 ~1Hz（实测坑，docs/04 §7 回填）。
        self._cloud_group = ReentrantCallbackGroup()
        _cloud_hz = float(self.cfg["cloud_publish_hz"])
        if _cloud_hz > 0:
            self.create_timer(1.0 / _cloud_hz, self._publish_cloud_timer,
                              callback_group=self._cloud_group)

        # ---- Phase 6 §5 语义查询服务（独立回调组：查询重活不阻塞 sync 回调）----
        from edge_3dgs_msgs.srv import Query
        self._query_group = ReentrantCallbackGroup()
        self._query_svc = QueryService(
            self._backend,
            # 2026-09-02：真机地图语义化用 lang_ae_live.pt（域不同——默认的
            # lang_ae_replica.pt 是 Replica 域，AE 不匹配 → 查询结果错乱，坑）
            ae_ckpt=args.ae,
            default_top_k=int(self.cfg.get("default_top_k", 100)),
            default_min_score=float(self.cfg.get("default_min_score", 0.0)),
            default_eps=float(self.cfg.get("default_eps", 0.15)))
        self.create_service(Query, str(self.cfg.get("query_service", "/semantic_query/query")),
                            self._on_query, callback_group=self._query_group)

        # --load 模式：启动即加载模型（不依赖首帧），文件缺失 fail fast
        if self._load_path:
            p = Path(self._load_path)
            if not p.exists():
                self.get_logger().error(f"--load 模型不存在: {p}（fail fast）")
                sys.exit(2)
            self._model = _load_checkpoint(str(p))
            self.get_logger().info(
                f"已加载模型 {p}：{self._model.num_gaussians} 高斯"
                f"（--load 模式只 track 不建图）")
        else:
            self.get_logger().info("完整 SLAM 模式：首帧 init_from_depth 初始化")

        self.get_logger().info(
            f"节点就绪，tier={args.tier}（skip_every={self._tier['skip_every']}），"
            f"等待相机数据（{self.cfg['rgb_topic']} + {self.cfg['depth_topic']}）...")

    # ------------------------------------------------------------------ 回调
    def on_frame(self, frame: SyncedFrame):
        self._frame_idx += 1
        t = self._frame_idx

        if self._backend is None:           # 首帧：模型 + 后端初始化（唯一慢回调）
            model = self._model if self._model is not None else \
                init_from_depth(frame, np.eye(4), stride=int(self.cfg["init_stride"]))
            tw, th = self._tier["track_W"], self._tier["track_H"]
            map_kw = dict(self._tier["map_kwargs"])
            if self._load_path:
                map_kw = dict(_NO_MAP_KWARGS)   # --load：只 track 不建图
            elif self._cap > 0:                 # --cap 覆盖容量（全景扫描防冻结）
                map_kw["capacity_max"] = self._cap
            if self._seed_max > 0:              # --seed-max 控制每 KF 播种量上限
                map_kw["max_new"] = self._seed_max
            self._backend = SLAMBackend(
                model, frame.K,
                track_kwargs=dict(self._tier["track_kwargs"]),
                map_kwargs=map_kw,
                dt_thresh=0.1, dth_thresh_deg=3.0,   # fps10 定稿 kf_dt/kf_dth
                lock_chunk=self._tier["lock_chunk"], lock_sleep_ms=0.01,
                track_W=tw, track_H=th,
                init_mode=self._tier["init_mode"],
                on_keyframe=self._recorder.add if self._recorder else None)
            self._last_T = self._prev_T = np.eye(4)
            self.get_logger().info(
                f"首帧初始化：{model.num_gaussians} 高斯，SLAMBackend(tier, "
                f"track_W×H={tw}×{th}) 就绪")
            self._tf_pub.publish_pose(np.eye(4), frame.stamp)

        if self._sched.accept(t):           # 处理帧：track
            T_init = self._last_T
            T_cv = None
            if self._prev_T is not None and self._last_T is not None:
                # 恒速初值：last @ inv(prev) @ last（与 phase3 消融口径一致）
                T_cv = self._last_T @ np.linalg.inv(self._prev_T) @ self._last_T
                T_init = T_cv
            # 2026-09-02 IMU 陀螺旋转先验：快速转头恒速猜测会错 1-5°+（双影主因
            # 之一）；陀螺积分给物理真旋转。IMU 只供初值，视觉仍全量修正。
            imu_used = False
            if self._imu_prior is not None and self._proc_T is not None \
                    and self._proc_stamp is not None:
                T_imu = self._imu_prior.init_pose(
                    self._proc_T, self._proc_stamp, frame.stamp, T_cv)
                if T_imu is not None:
                    T_init = T_imu
                    imu_used = True
                    self._imu_hits += 1
            T = self._backend.track(frame, T_init)
            # 初值残差仪表（跟踪修正旋角）：IMU vs 恒速初值分开统计，快转段
            # IMU 初值残差应明显更小——符号/外参错误会表现为残差反增
            dR = T[:3, :3] @ T_init[:3, :3].T
            deg = float(np.degrees(np.arccos(
                np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
            (self._corr_imu_deg if imu_used else self._corr_cv_deg).append(deg)
            gpu_ms = (self._backend.stats["track_gpu_ms"][-1]
                      if self._backend.stats["track_gpu_ms"] else None)
            self._sched.on_processed(t, T, gpu_ms)
            if self._backend.take_failure():      # 跟踪失败：下一帧强制处理 + 本帧回退
                self._sched.force_next()
                self._sched.on_lost_reset(self._last_T)
                T = self._last_T.copy()           # 保持位姿流连续性
                self._log_once("track failure → 回退最近好位姿，下一帧强制恢复")
            self._prev_T, self._last_T = self._last_T, T
            self._proc_T, self._proc_stamp = T.copy(), float(frame.stamp)
            self._perf_n += 1
            if self._perf_n % 30 == 0:
                self._log_perf()
        else:                               # 跳过帧：SE(3) 恒速外推输出
            T = self._sched.extrapolate(t)
            if self._sched.lost_flags.get(t):
                self._log_once("extrapolation lost（位移超限）→ 保持最近位姿")

        self._last_stamp = frame.stamp
        self._tf_pub.publish_pose(T, frame.stamp)   # 处理帧与外推帧都发（30Hz 位姿流）

    # ------------------------------------------------------------------ 查询服务
    def _on_query(self, request, response):
        """语义查询：文本 → bbox/confidence/points（QueryService 快照 + 锁外查询）。

        Query.srv 是嵌套结构：response.result（QueryResult msg）承载全部字段。"""
        r = self._query_svc.handle(request, backend=self._backend)
        res = response.result
        if r is None:
            res.confidence = -1.0
            return response
        res.request = request.request
        c, e, rot = r["bbox_center"], r["bbox_extent"], r["bbox_rotation"]
        res.bbox_center = [float(v) for v in c]
        res.bbox_extent = [float(v) for v in e]
        res.bbox_rotation = [float(v) for row in rot for v in row]
        res.confidence = float(r["confidence"])
        pts, scores = r["points"], r["scores"]
        n = min(len(pts), 2000)                     # 上限防服务响应过大
        from edge_3dgs_msgs.msg import SemanticPoint
        res.points = [SemanticPoint(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                                    score=float(s), label=str(request.request.query))
                      for p, s in zip(pts[:n], scores[:n])]
        self.get_logger().info(
            f"[query] 「{request.request.query}」 conf={res.confidence:.3f} "
            f"bbox={[round(v, 2) for v in res.bbox_center]} 点 {len(res.points)}")
        return response

    def _publish_cloud_timer(self):
        """1Hz 定时器：锁内快照 → 锁外建消息发布。

        PointCloud2 每次发（numpy tobytes 毫秒级）；GaussianCloud（逐元素构建
        ~600-800ms@25k，50k 实测 1.5-2s）低频发——实测其 Python 循环即使独立
        回调组也饿死 sync 回调，tf 掉到 ~1Hz（docs/04 §7 坑）。
        2026-09-02：5→10→20s——Python 逐元素循环占 GIL 且抢 CPU，与 track 的
        python enqueue 争 CPU（真机 gpu_ms 墙钟口径被撑大，run8 实测构建
        1540-2122ms/次）。
        """
        if self._backend is None or self._last_stamp is None:
            return
        snap = self._backend.snapshot_model(max_points=int(self.cfg["cloud_max_points"]))
        if snap is not None:
            t0 = Clock().now()
            self._cloud_tick = getattr(self, "_cloud_tick", 0) + 1
            heavy = self._cloud_tick % 20 == 0       # 每 20s 一次 GaussianCloud
            self._cloud_pub.publish(snap, self._last_stamp, gaussian_cloud=heavy)
            ms = (Clock().now() - t0).nanoseconds / 1e6
            if heavy and ms > 50:      # 只记录慢发布（构建耗时实测进验证报告）
                self._log_once(f"cloud 发布 {len(snap['means'])} 高斯，构建 {ms:.0f}ms")

    def _log_once(self, msg: str):
        if msg and msg not in self._logged:
            self._logged.add(msg)
            self.get_logger().info(msg)

    # ------------------------------------------------------------------ 2026-09-02 语义场衔接
    def _autosave_timer(self):
        """周期自动存图（map_autosave.pt 覆盖写）——长跑中途被杀不丢图。"""
        if self._backend is None:
            return
        try:
            self._backend.save_checkpoint(str(self._out_dir / "map_autosave.pt"))
        except Exception:
            pass

    def save_outputs(self):
        """收尾：地图 checkpoint + 关键帧记录（main() finally 调用）。

        2026-09-02 次序/原子性修正（实测教训：退出时第二下 Ctrl-C/关终端中断
        np.savez_compressed → frames.npz 直接损坏且后续 map 保存被跳过）：
          1) 地图先存（快、关键）；
          2) frames.npz 写临时文件 → os.replace 原子改名（中断最多丢新文件，
             不损坏已存在的上一份）。
        产物（同一世界系，可直接喂 Phase 6 提取/蒸馏）：
            frames.npz          rgb/depth/K/poses(w2c)，与回放工具同契约
            map_<ts>.pt         {'params','variables'}，node --load 可读
            map_latest.pt       固定路径副本（脚本用）
        """
        if self._backend is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            p = self._out_dir / f"map_{ts}.pt"
            self._backend.save_checkpoint(str(p))
            self._backend.save_checkpoint(str(self._out_dir / "map_latest.pt"))
            self.get_logger().info(
                f"[out] 地图已存 {self._backend.model.num_gaussians} 高斯 → {p}")
        if self._recorder is not None and self._backend is not None:
            # ⚠️ np.savez_compressed 会给无 .npz 后缀的路径自动补后缀——tmp 必须
            # 以 .npz 结尾，否则 os.replace 找不到源（实测 frames.npz.tmp.npz 坑）
            tmp = self._out_dir / "frames.tmp.npz"
            r = self._recorder.save(str(tmp))
            if r.get("frames", 0) > 0:
                import os
                os.replace(tmp, self._out_dir / "frames.npz")   # 原子改名
            self.get_logger().info(
                f"[out] 关键帧已存 {r.get('frames', 0)} 帧"
                f"（累计 {r.get('recorded', 0)}，淘汰 {r.get('dropped', 0)}）"
                f" → {self._out_dir / 'frames.npz'}")

    def _log_perf(self):
        """每 30 个处理帧打一次性能分解（真机调优仪表，2026-09-02 加）。

        输出：近 30 帧 track wall/gpu/icp 均值、map 每关键帧耗时、队列深度、
        丢帧计数、调度器当前间隔——区分 300ms/帧 花在跟踪迭代 / ICP / 等建图锁。
        """
        s = self._backend.stats
        now = time.perf_counter()
        rate = 30.0 / (now - self._perf_t0) if self._perf_t0 else float("nan")
        self._perf_t0 = now

        def avg(key, k=30):
            v = s[key]
            vv = v[-k:]
            return sum(vv) / len(vv) if vv else float("nan")

        self.get_logger().info(
            f"[perf] 处理帧率≈{rate:.1f}/s | track wall {avg('track_wall_ms'):.0f}ms "
            f"gpu {avg('track_gpu_ms'):.0f}ms ({avg('track_iters'):.1f}it) "
            f"| icp {avg('icp_ms'):.0f}ms "
            f"(fallback {s['icp_fallback']}, 门控跳过 {self._backend.icp_skipped}) "
            f"| map/关键帧 {avg('map_ms'):.0f}ms "
            f"(mapped {s['mapped']}, 压档跳过 {s['map_skipped']}, "
            f"播种跳过 {s['seed_skipped']}, dropped {s['dropped']}) | "
            f"队列深度 {self._backend.map_queue.qsize()} | "
            f"skip {self._sched._cur_skip} (lost {self._sched.n_lost}) "
            f"| 高斯 {self._backend.model.num_gaussians}")

        def deg_avg(lst, k=30):
            v = lst[-k:]
            return sum(v) / len(v) if v else float("nan")

        self.get_logger().info(
            f"[imu] 先验命中 {self._imu_hits} 帧 | 跟踪残差旋角 "
            f"IMU初值 {deg_avg(self._corr_imu_deg):.2f}° vs "
            f"恒速初值 {deg_avg(self._corr_cv_deg):.2f}°"
            f"（IMU < 恒速 = 生效；反增 = 符号/外参错）")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Edge-3DGS-SLAM ROS2 节点（docs/04）")
    p.add_argument("--load", default=None, metavar="PT",
                   help="加载现成模型 checkpoint（只 track 不建图），默认完整 SLAM")
    p.add_argument("--tier", choices=list(TIERS), default="fps",
                   help="性能档位（默认 fps；quality=质量锚）")
    p.add_argument("--params", default=None, metavar="YAML",
                   help="ros2 参数文件（默认 config/ros2/params.yaml）")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="产出目录（地图 checkpoint + 关键帧 npz；"
                        "默认 <仓库>/data/outputs/live）")
    p.add_argument("--record-max", type=int, default=150,
                   help="关键帧记录上限，环形保留最近 N 帧（0=关闭；默认 150）")
    p.add_argument("--ae", default=None, metavar="PT",
                   help="查询用场景 AE checkpoint（真机语义化用 data/outputs/"
                        "live_semantic/lang_ae_live.pt；缺省=Replica 域 AE）")
    p.add_argument("--cap", type=int, default=0, metavar="N",
                   help="高斯容量上限覆盖（0=用档位默认 200k）。2026-09-02 实测："
                        "200k 在前 ~67 KF 封顶、播种冻结 → 地图只含起点段、漂移；"
                        "quality 档原生 720p 播种每 KF 可达 2 万高斯，400k 也 ~10 "
                        "KF 就满——整屋扫描建议 --cap 1000000 配 --seed-max。")
    p.add_argument("--seed-max", type=int, default=0, metavar="N",
                   help="每关键帧播种高斯上限（0=档位默认 20000）。调小可拉长"
                        "容量寿命、降每 KF 播种渲染耗时（quality 720p 播种是每 "
                        "KF 最大单项成本之一）；面积播种均匀性由密度门兜底。")
    p.add_argument("--autosave-sec", type=int, default=120,
                   help="地图自动存盘间隔秒，覆盖写 map_autosave.pt（0=关闭；默认 120）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rclpy.init(args=None)
    node = Edge3DGSSlamNode(args)
    # 2026-09-02：4 线程——IMU 独立回调组需要空闲线程（图像 sync 占用默认组）
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node._backend is not None:
            node._backend.join(timeout=60.0)    # 等 map 队列清空，防 CUDA 中断
        try:
            node.save_outputs()                 # 2026-09-02：地图 + 关键帧落盘
        except (Exception, KeyboardInterrupt) as e:
            # KeyboardInterrupt：退出时第二下 Ctrl-C 不再中断保存（实测损坏 frames.npz）
            node.get_logger().error(f"[out] 保存产出失败: {e}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
