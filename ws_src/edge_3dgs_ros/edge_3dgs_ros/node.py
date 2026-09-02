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
from edge_3dgs_slam.slam.scheduler import FrameScheduler

from .cloud_publisher import CloudPublisher
from .query_service import QueryService
from .config import load_ros2_params
from .tf_publisher import TFPublisher

# ------------------------------------------------------------------ 双档参数
# 逐项核对 experiments/phase3_perf_ablation.py CONFIGS 定稿值（2026-08-28）：
# fps10 = 速度包络档（2:1 丢帧 + ICP 初值 + 2 迭代 d2 + ad8 + light map + 收紧失败门）；
# track160/maplight = 质量锚档（6it d2 ad8 + full map 8it）。kf_dt/dth、lock_chunk 同源。
TIERS = {
    "fps": dict(
        skip_every=2, budget_ms=100.0, init_mode="icp",
        track_W=224, track_H=168,
        track_kwargs=dict(iters=2, depth_every=2, adaptive_max=8, cull=False,
                          fail_rot_deg=8.0, fail_trans_m=0.25),
        map_kwargs=dict(iters=3, density_r=0.01, capacity_max=200_000, cull=False,
                        map_tier="light", window_rotate=True, rotate_n=2,
                        opt_W=224, opt_H=168),
    ),
    "quality": dict(
        skip_every=1, budget_ms=300.0, init_mode="const",
        track_W=320, track_H=240,
        track_kwargs=dict(iters=6, depth_every=2, adaptive_max=8, cull=False),
        map_kwargs=dict(iters=8, density_r=0.01, capacity_max=200_000, cull=False,
                        map_tier="full", window_rotate=True, rotate_n=2),
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
        self._backend = None                # 惰性：K 要等 camera_info，模型要等首帧
        self._sched = FrameScheduler(skip_every=self._tier["skip_every"],
                                     budget_ms=self._tier["budget_ms"])
        self._frame_idx = 0
        self._last_T: np.ndarray | None = None
        self._prev_T: np.ndarray | None = None
        self._last_stamp: float | None = None
        self._model = None
        self._logged: set = set()

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
            self._backend = SLAMBackend(
                model, frame.K,
                track_kwargs=dict(self._tier["track_kwargs"]),
                map_kwargs=map_kw,
                dt_thresh=0.1, dth_thresh_deg=3.0,   # fps10 定稿 kf_dt/kf_dth
                lock_chunk=2, lock_sleep_ms=0.01,
                track_W=tw, track_H=th,
                init_mode=self._tier["init_mode"])
            self._last_T = self._prev_T = np.eye(4)
            self.get_logger().info(
                f"首帧初始化：{model.num_gaussians} 高斯，SLAMBackend(tier, "
                f"track_W×H={tw}×{th}) 就绪")
            self._tf_pub.publish_pose(np.eye(4), frame.stamp)

        if self._sched.accept(t):           # 处理帧：track
            T_init = self._last_T
            if self._prev_T is not None and self._last_T is not None:
                # 恒速初值：last @ inv(prev) @ last（与 phase3 消融口径一致）
                T_init = self._last_T @ np.linalg.inv(self._prev_T) @ self._last_T
            T = self._backend.track(frame, T_init)
            gpu_ms = (self._backend.stats["track_gpu_ms"][-1]
                      if self._backend.stats["track_gpu_ms"] else None)
            self._sched.on_processed(t, T, gpu_ms)
            if self._backend.take_failure():      # 跟踪失败：下一帧强制处理 + 本帧回退
                self._sched.force_next()
                self._sched.on_lost_reset(self._last_T)
                T = self._last_T.copy()           # 保持位姿流连续性
                self._log_once("track failure → 回退最近好位姿，下一帧强制恢复")
            self._prev_T, self._last_T = self._last_T, T
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
        ~600-800ms）每 5 次发一次——实测其 Python 循环即使独立回调组也饿死
        sync 回调，tf 掉到 ~1Hz（docs/04 §7 坑）。
        """
        if self._backend is None or self._last_stamp is None:
            return
        snap = self._backend.snapshot_model(max_points=int(self.cfg["cloud_max_points"]))
        if snap is not None:
            t0 = Clock().now()
            self._cloud_tick = getattr(self, "_cloud_tick", 0) + 1
            heavy = self._cloud_tick % 5 == 0        # 每 5s 一次 GaussianCloud
            self._cloud_pub.publish(snap, self._last_stamp, gaussian_cloud=heavy)
            ms = (Clock().now() - t0).nanoseconds / 1e6
            if heavy and ms > 50:      # 只记录慢发布（构建耗时实测进验证报告）
                self._log_once(f"cloud 发布 {len(snap['means'])} 高斯，构建 {ms:.0f}ms")

    def _log_once(self, msg: str):
        if msg and msg not in self._logged:
            self._logged.add(msg)
            self.get_logger().info(msg)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Edge-3DGS-SLAM ROS2 节点（docs/04）")
    p.add_argument("--load", default=None, metavar="PT",
                   help="加载现成模型 checkpoint（只 track 不建图），默认完整 SLAM")
    p.add_argument("--tier", choices=list(TIERS), default="fps",
                   help="性能档位（默认 fps；quality=质量锚）")
    p.add_argument("--params", default=None, metavar="YAML",
                   help="ros2 参数文件（默认 config/ros2/params.yaml）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rclpy.init(args=None)
    node = Edge3DGSSlamNode(args)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node._backend is not None:
            node._backend.join(timeout=60.0)    # 等 map 队列清空，防 CUDA 中断
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
