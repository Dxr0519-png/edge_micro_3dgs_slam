#!/usr/bin/env python3
"""Phase 4 回放验证工具：Replica / 合成 npz → 与 D435iReader 默认话题一致的话题流。

docs/04 §6 数据约定：接口/回放测试用 Replica 或合成序列回放为话题流
（离线评估数据 → 节点输入，可复现、无硬件依赖；真机最终验证走 D435i 自采）。

话题（与 src/edge_3dgs_slam/camera/d435i_reader.py DEFAULT_TOPICS 一致）：
    /camera/color/image_raw               (sensor_msgs/Image, rgb8)
    /camera/aligned_depth_to_color/image_raw (16UC1, 深度**毫米** —— D435iReader 内部 ×0.001)
    /camera/color/camera_info             (CameraIntrinsics.from_camera_info 只读 k/width/height)
可选：
    /camera/gt_pose                       (PoseStamped, map 系，= inv(w2c)，供 ATE 粗验)

时间戳：t0 + k/hz，严格单调递增，两图同一 stamp（ApproximateTimeSynchronizer slop 0.05 无压力）。
节奏：--rate realtime（默认，定时器 1/hz）| --rate max（立即逐帧发）；--loop 播完重头。

用法：
    python3 experiments/phase4_replay_publisher.py --source replica --scene office0 \
        --frames 200 --scale 0.5 --hz 15 --loop
    python3 experiments/phase4_replay_publisher.py --source synth \
        --npz data/outputs/phase3/synth_scene_480.npz --hz 15
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import CameraInfo, Image

from edge_3dgs_slam.dataset.replica import ReplicaSequence


def _time_from_sec(s: float) -> Time:
    sec = int(s)
    return Time(sec=sec, nanosec=int(round((s - sec) * 1e9)))


def _make_image(encoding: str, stamp: Time, frame_id: str,
                data: np.ndarray, step: int) -> Image:
    img = Image()
    img.header.stamp = stamp
    img.header.frame_id = frame_id
    img.height, img.width = data.shape[:2]
    img.encoding = encoding
    img.is_bigendian = False
    img.step = step
    img.data = data.tobytes()
    return img


def _make_camera_info(stamp: Time, frame_id: str, K: np.ndarray, H: int, W: int) -> CameraInfo:
    """构造最小 CameraInfo：from_camera_info 只读 k[0]/k[4]/k[2]/k[5] + width/height。"""
    ci = CameraInfo()
    ci.header.stamp = stamp
    ci.header.frame_id = frame_id
    ci.height, ci.width = H, W
    ci.distortion_model = "plumb_bob"
    ci.k = [float(K[0, 0]), 0.0, float(K[0, 2]),
            0.0, float(K[1, 1]), float(K[1, 2]),
            0.0, 0.0, 1.0]
    ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    ci.binning_x = ci.binning_y = 0
    return ci


def _pose_msg(stamp: Time, T_wc: np.ndarray) -> PoseStamped:
    """w2c (4,4) → PoseStamped（map 系，相机位姿 = inv(T_wc)）。"""
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    R_wc = R.T
    t_wc = -R_wc @ t
    # 旋转矩阵 → 四元数（xyz 序；w2c 含微量数值误差，规范化处理）
    q = np.zeros(4)
    tr = np.trace(R_wc)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[0], q[1], q[2], q[3] = (R_wc[2, 1] - R_wc[1, 2]) / s, (R_wc[0, 2] - R_wc[2, 0]) / s, \
                                 (R_wc[1, 0] - R_wc[0, 1]) / s, 0.25 * s
    else:
        i = int(np.argmax(np.diag(R_wc)))
        j = (i + 1) % 3
        k = (j + 1) % 3
        s = np.sqrt(1.0 + R_wc[i, i] - R_wc[j, j] - R_wc[k, k]) * 2
        q[i], q[j], q[k] = 0.25 * s, (R_wc[j, i] + R_wc[i, j]) / s, (R_wc[k, i] + R_wc[i, k]) / s
        q[3] = (R_wc[k, j] - R_wc[j, k]) / s
    q /= np.linalg.norm(q)
    p = PoseStamped()
    p.header.stamp = stamp
    p.header.frame_id = "map"
    p.pose.position.x = float(t_wc[0])
    p.pose.position.y = float(t_wc[1])
    p.pose.position.z = float(t_wc[2])
    p.pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]),
                                    z=float(q[2]), w=float(q[3]))
    return p


class ReplayPublisher:
    """把内存中的 RGB-D 帧序列按固定节奏发成话题流。"""

    def __init__(self, node, rgb_frames, depth_frames, K, poses_w2c,
                 hz: float, loop: bool, rate: str = "realtime", gt: bool = True):
        self.node = node
        self.rgb = rgb_frames
        self.depth = depth_frames
        self.K = np.asarray(K, np.float64)
        self.poses = poses_w2c
        self.hz = hz
        self.loop = loop
        self.rate = rate
        self.k = 0
        self.t0 = node.get_clock().now().nanoseconds / 1e9
        self.pub_rgb = node.create_publisher(Image, "/camera/color/image_raw", 10)
        self.pub_depth = node.create_publisher(
            Image, "/camera/aligned_depth_to_color/image_raw", 10)
        self.pub_info = node.create_publisher(CameraInfo, "/camera/color/camera_info", 10)
        self.pub_gt = (node.create_publisher(PoseStamped, "/camera/gt_pose", 10)
                       if gt and len(poses_w2c) else None)

    def _stamp(self) -> Time:
        return _time_from_sec(self.t0 + self.k / self.hz)

    def _publish_one(self) -> None:
        if self.k >= len(self.rgb):
            if self.loop:
                self.k = 0
                # ⚠️ 每轮重置 t0：否则时间戳停在首轮启动时刻，与监听器（rviz2/
                # view_frames）当前时间的差持续拉大 → tf 缓存判定 TF_OLD_DATA
                # 丢弃全部变换（实测坑，docs/04 §7 回填）。
                self.t0 = self.node.get_clock().now().nanoseconds / 1e9
                self.node.get_logger().info("回放完成，--loop 重新开始")
            else:
                self.node.get_logger().info("回放完成（--loop 可循环）")
                raise StopIteration
        stamp = self._stamp()
        H, W = self.rgb[self.k].shape[:2]
        frame_id = "camera"
        # 深度米 → 16UC1 毫米（D435iReader 内部 ×0.001 还原成米——单位必须匹配）
        depth_mm = np.clip(self.depth[self.k] * 1000.0, 0, 65535).astype(np.uint16)
        self.pub_rgb.publish(_make_image("rgb8", stamp, frame_id,
                                         self.rgb[self.k], step=W * 3))
        self.pub_depth.publish(_make_image("16UC1", stamp, frame_id,
                                           depth_mm, step=W * 2))
        self.pub_info.publish(_make_camera_info(stamp, frame_id, self.K, H, W))
        if self.pub_gt is not None:
            self.pub_gt.publish(_pose_msg(stamp, self.poses[self.k]))
        self.k += 1

    def spin(self) -> None:
        if self.rate == "max":          # 立即逐帧发（不加节流）
            try:
                while rclpy.ok():
                    try:
                        self._publish_one()
                    except StopIteration:
                        break
                    rclpy.spin_once(self.node, timeout_sec=0.01)
            finally:
                rclpy.shutdown()
            return
        # realtime：定时器按 1/hz 驱动
        self._timer = self.node.create_timer(1.0 / self.hz, self._timer_cb)
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    def _timer_cb(self) -> None:
        try:
            self._publish_one()
        except StopIteration:
            self.node.get_logger().info("回放完成，退出")
            self.node.destroy_timer(self._timer)
            rclpy.shutdown()


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4 回放：Replica/合成 → RGB-D 话题流")
    p.add_argument("--source", choices=["replica", "synth"], default="replica")
    p.add_argument("--root", default="third_party/SplaTAM/data/Replica")
    p.add_argument("--scene", default="office0")
    p.add_argument("--scale", type=float, default=0.5, help="Replica 降采样比例（K 同缩放）")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--npz", default="data/outputs/phase3/synth_scene_480.npz")
    p.add_argument("--hz", type=float, default=15.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--rate", choices=["realtime", "max"], default="realtime")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.source == "replica":
        seq = ReplicaSequence.from_dir(root / args.root, args.scene, args.frames)
        rgb_frames, depth_frames = [], []
        for t in range(args.start, min(args.frames + args.start, len(seq))):
            f = seq.frame_scaled(t, args.scale)
            rgb_frames.append(f.rgb)
            depth_frames.append(f.depth)
        K = seq.frame_scaled(0, args.scale).K if rgb_frames else seq.K
        poses = seq.poses_w2c[args.start:args.frames + args.start]
        print(f"[replay] Replica {args.scene} ×{args.scale}：{len(rgb_frames)} 帧，"
              f"K={K.tolist()}", flush=True)
    else:
        d = np.load(root / args.npz)
        # ⚠️ 先取数组引用再切片：`d["rgb"][i] for i in range(n)` 中每次访问
        # `d["rgb"]` 都会**重新解压整个数组**（npz 惰性加载，NpzFile 缓存不命中），
        # 60 帧时残留 60×55MB + 60×74MB ≈ 7.7GB 内存 → DDS SHM 分配失败，
        # publish 报 "publisher's context is invalid"（实测坑，docs/04 §7 回填）。
        rgb_arr = d["rgb"]
        depth_arr = d["depth"]
        n = args.frames
        rgb_frames = [rgb_arr[i] for i in range(n)]       # 视图，共享底层数组
        depth_frames = [depth_arr[i] for i in range(n)]
        K = d["K"]
        poses = d["poses"][:n]
        print(f"[replay] 合成 {n} 帧 @{d['rgb'].shape[1:3]}，K={K.tolist()}", flush=True)

    rclpy.init()
    node = rclpy.create_node("phase4_replay_publisher")
    node.get_logger().info(
        f"回放 {len(rgb_frames)} 帧 @{args.hz}Hz（rate={args.rate}, loop={args.loop}）→ "
        "/camera/color/image_raw + /camera/aligned_depth_to_color/image_raw + camera_info")
    rp = ReplayPublisher(node, rgb_frames, depth_frames, K, poses,
                         hz=args.hz, loop=args.loop, rate=args.rate)
    rp.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
