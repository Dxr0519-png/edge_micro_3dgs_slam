"""ROS2 节点参数加载：默认 dict 与 config/ros2/params.yaml 深合并（Phase 4）。

仿 feature_factory/config.py 的 load_feature_config 模式：yaml 缺失/解析失败
回退默认值并打一行警告（不 raise），节点对每个键再 declare_parameter，
因此 --ros-args --params-file 仍可覆盖（双通道兼容）。
"""
from __future__ import annotations

from pathlib import Path

import yaml

# 与 config/ros2/params.yaml 逐键一致的默认值；Phase 4 新增键（cloud_* 等）
DEFAULT_ROS2_PARAMS = {
    # 订阅话题
    "rgb_topic": "/camera/color/image_raw",
    "depth_topic": "/camera/aligned_depth_to_color/image_raw",
    "color_info_topic": "/camera/color/camera_info",
    "depth_info_topic": "/camera/aligned_depth_to_color/camera_info",
    "imu_topic": "/imu",
    # 发布
    "pose_tf_frame": "map",
    "publish_odom": True,
    "publish_gaussian_map": True,
    "publish_rendered_image": False,
    # 同步
    "sync_queue_size": 10,
    "sync_slop_sec": 0.05,
    # 查询（Phase 6）
    "query_service": "/semantic_query/query",
    "default_top_k": 100,
    "default_min_score": 0.3,
    # ---- Phase 4 新增 ----
    "cloud_publish_hz": 1.0,      # 高斯点云发布频率（Hz），1Hz×50k≈2.6MB/s
    "cloud_max_points": 50000,    # 单帧高斯上限（均匀抽稀；200k 全量 msg 构建 ~1-3s）
    "init_stride": 2,             # 首帧反投影初始化步长
    "camera_frame": "camera",
    "odom_frame": "odom",
}


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(16):
        if (p / "src" / "edge_3dgs_slam").is_dir():
            return p
        p = p.parent
    return Path.cwd()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_ros2_params(path: str | Path | None = None) -> dict:
    """加载 ros2 节点参数：默认值 + config/ros2/params.yaml（可被 --params 覆盖）。

    返回平铺 dict（yaml 顶层键忽略；参数在 node.ros__parameters 下）。
    """
    cfg = dict(DEFAULT_ROS2_PARAMS)
    if path is None:
        path = _repo_root() / "config" / "ros2" / "params.yaml"
    path = Path(path)
    if not path.exists():
        print(f"[config] 参数文件 {path} 不存在，使用默认参数", flush=True)
        return cfg
    try:
        data = yaml.safe_load(path.read_text()) or {}
        # 兼容两种布局：顶层直接是参数 / 包名.ros__parameters 包裹
        merged = {}
        for v in data.values():
            if isinstance(v, dict) and "ros__parameters" in v:
                merged = _deep_merge(merged, v["ros__parameters"])
            elif isinstance(v, dict):
                merged = _deep_merge(merged, v)
        cfg = _deep_merge(cfg, merged)
    except Exception as e:                      # yaml 解析失败 → 默认值 + 警告
        print(f"[config] 参数文件 {path} 解析失败（{e}），使用默认参数", flush=True)
    return cfg
