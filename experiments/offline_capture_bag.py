#!/usr/bin/env python3
"""离线重建第一步：ros2 bag → npz 分块（rgb uint8 / depth uint16mm / K / imu）。

用法:
    python3 experiments/offline_capture_bag.py ~/room_scan data/outputs/offline
产出: data/outputs/offline/frames_%04d.npz（每块 ~100 帧，键与 LiveNpzSequence
兼容 + 独立 imu 数组文件）——离线重建脚本按序读取。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import cv2
import numpy as np
from rosbags.highlevel import AnyReader          # 需 pip rosbags（semantic_ws 已装）
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_HUMBLE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/outputs/offline"))
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    topics = {
        "/camera/color/image_raw": "rgb",
        "/camera/aligned_depth_to_color/image_raw": "depth",
        "/camera/color/camera_info": "info",
        "/camera/imu": "imu",
    }
    buf = {"rgb": [], "depth": [], "info": None, "imu": [], "stamp": []}
    K = None
    n_written = 0
    with AnyReader([args.bag]) as r:
        conns = {c.topic: c for c in r.connections}
        for conn, ts, raw in r.messages():
            name = topics.get(conn.topic)
            if name is None:
                continue
            if conn.topic == "/camera/color/camera_info":
                msg = TS.deserialize_cdr(raw, conn.msgtype)
                K = np.array([[msg.k[0], 0, msg.k[2]],
                              [0, msg.k[4], msg.k[5]], [0, 0, 1]], np.float64)
                continue
            if conn.topic == "/camera/imu":
                msg = TS.deserialize_cdr(raw, conn.msgtype)
                buf["imu"].append((ts * 1e-9,
                                   np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                                             msg.linear_acceleration.z]),
                                   np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                                             msg.angular_velocity.z])))
                continue
            msg = TS.deserialize_cdr(raw, conn.msgtype)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if conn.topic.endswith("image_raw") and "/color/" in conn.topic:
                rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
                buf["rgb"].append(rgb.copy())
                buf["stamp"].append(stamp)
            else:
                d16 = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
                buf["depth"].append(d16.copy())
            if len(buf["rgb"]) >= args.chunk:
                if K is None:
                    raise SystemExit("未收到 camera_info")
                np.savez_compressed(args.out / f"frames_{n_written:04d}.npz",
                                    rgb=np.stack(buf.pop("rgb")),
                                    depth=np.stack(buf.pop("depth")), K=K)
                buf["rgb"], buf["depth"] = [], []
                n_written += 1
                print(f"  块 {n_written} 完成（{args.chunk} 帧）", flush=True)
    if buf["rgb"] and K is not None:
        np.savez_compressed(args.out / f"frames_{n_written:04d}.npz",
                            rgb=np.stack(buf["rgb"]), depth=np.stack(buf["depth"]), K=K)
        n_written += 1
    np.savez(args.out / "imu.npz",
             imu=np.asarray(buf["imu"], dtype=object))   # (stamp, accel, gyro)
    print(f"完成：{n_written} 块 + imu {len(buf['imu'])} 条 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
