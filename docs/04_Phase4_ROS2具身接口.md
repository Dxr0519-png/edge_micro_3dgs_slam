# 04 · Phase 4 系统封装与 ROS 2 具身接口

> 目标：把优化后的 SLAM 管线封装为 ROS2 Humble Node，订阅 RGB-D/IMU，发布位姿与高斯点云。
> 依赖：Phase 3 的优化管线。消息定义已在骨架 `ws_src/edge_3dgs_msgs` 生成。

## 1. 先编译消息包（先就绪检查）

> ✅ **已实现（2026-08-29）**：宿主机 ROS2 Humble 直接构建，5 个消息全部可查询
> （`colcon build` 12.4s；`ros2 interface show` 输出与 .msg 定义一致，见验证报告 §1）。

```bash
# ① ROS2 环境就绪检查（apt 已装则跳过，未装按 docker/Dockerfile §2 补）
ls /opt/ros/humble/setup.bash && source /opt/ros/humble/setup.bash

# ② 编译：已构建（消息可查询）则跳过，未构建才执行
cd ws_src
ros2 interface show edge_3dgs_msgs/msg/GaussianCloud >/dev/null 2>&1 \
    || colcon build --packages-select edge_3dgs_msgs
source install/setup.bash

# ③ 验证
ros2 interface show edge_3dgs_msgs/msg/GaussianCloud
```

## 2. 节点骨架（`ws_src/edge_3dgs_ros/`）

> ✅ **已实现（2026-08-29）**，实际文件结构：

```
edge_3dgs_ros/
├── setup.py                   # ament_python 打包；entry point: edge_3dgs_slam_node
├── setup.cfg                  # script_dir / install_scripts
├── package.xml                # 已生成（exec_depend 齐全，未改动）
├── CMakeLists.txt             # 保留骨架；注释说明"勿启用 install()，由 setup.py 安装"
├── resource/edge_3dgs_ros     # ament index 空标记
└── edge_3dgs_ros/
    ├── __init__.py            # sys.path 引导：上溯找 src/edge_3dgs_slam 仓库根
    ├── config.py              # load_ros2_params()：默认值 + config/ros2/params.yaml 深合并
    ├── node.py                # Edge3DGSSlamNode：订阅 + 调度 + track + 发布编排
    ├── tf_publisher.py        # 静态 map→odom + 动态 odom→camera + /odom
    └── cloud_publisher.py     # /gaussian_map（GaussianCloud）+ /gaussian_map_pc2（PointCloud2）
```

**与骨架的差异**（实现时按真实 API 修正，均回填）：

1. **`Edge3DGSSlamNode(D435iReader, Node)` 多重继承**而非组合：`D435iReader` 的
   `on_frame` 回调需子类重写；且 rclpy 的 `Node.handle` 等 property 必须经 MRO
   可达（只调 `Node.__init__` 而类不继承 Node 会 `AttributeError: 'handle'`）。
2. **首帧初始化惰性**：K 等 `camera_info`（reader 自动丢帧），模型等首帧——
   `--load` 模式启动即加载 checkpoint（fail fast）；完整模式首帧
   `init_from_depth(frame, eye(4), stride=2)` 后建 `SLAMBackend`。
3. **`--load` 模式禁用建图**：`map_kwargs=dict(iters=0, add_new=False, prune=False)`
   （map worker 安全 no-op），只 track。
4. **双档 `--tier fps|quality`**：参数逐项抄 `experiments/phase3_perf_ablation.py`
   定稿值（fps10 = skip 2 + ICP + 2it d2 ad8 + light map；track160/maplight = 6it
   + full map 8it），docs/03 §11 双档结论落地；动态切档为后续项。
5. **帧调度**：`FrameScheduler` 丢帧 + SE(3) 恒速外推，**处理帧与外推帧都发
   tf/odom**（30Hz 位姿流）；track 失败 → `force_next()` + 本帧回退最近好位姿
   保持连续性；cloud 定时器独立 `ReentrantCallbackGroup`（见 §4/§7）。
6. **tf 语义**：发布 `inv(T_wc)` 而非 `T_wc`（见 §3 修正）。

## 3. 发布位姿（`tf_publisher.py`）

> ✅ **已实现（2026-08-29）**：[tf_publisher.py](../ws_src/edge_3dgs_ros/edge_3dgs_ros/tf_publisher.py)
> **语义修正**：全项目 `T_wc` 为**世界→相机**（w2c，`p_cam = T @ p_world`，见
> `src/edge_3dgs_slam/utils/se3.py`），相机在世界系中的位姿 = **inv(T_wc) =
> [Rᵀ | -Rᵀt]**。原骨架直接把 `T_wc[:3,3]/T_wc[:3,:3]` 当 tf 发布是错误的
> （相机坐标会落在 `-Rᵀt` 处）。落地为 REP-105 两段：
> - **map→odom**：静态单位阵（`/tf_static`，latched，初始化时发一次）；
> - **odom→camera**：动态，`inv(T_wc)`，每帧发布（含外推帧）；
> - **/odom**（nav_msgs/Odometry）：`frame_id=odom, child_frame_id=camera`，
>   位置/朝向同 odom→camera；twist 全 0（Phase 4 不估速度，VIO 接管位留）。
> 四元数转换用**纯 numpy**（`_matrix_to_quat_np`）：真机建图占 GPU 时 torch 版
> 转换会排队拉高 on_frame 延迟（实测坑，§7）。

```python
def publish_pose(self, T_wc, stamp):
    # T_wc 是 w2c（p_cam = T @ p_world）→ 相机世界位姿 = inv(T_wc) = [Rᵀ | -Rᵀt]
    R = T_wc[:3,:3]; t = T_wc[:3,3]
    R_wc, t_wc = R.T, -R.T @ t
    # 1) 动态 tf odom->camera（inv(T_wc)）
    # 2) /odom：位置=t_wc，朝向=四元数(R_wc)，twist 全 0
    # map->odom 静态单位阵在初始化时发一次（/tf_static）
```

## 4. 发布高斯点云（`cloud_publisher.py`）

> ✅ **已实现（2026-08-29）**：[cloud_publisher.py](../ws_src/edge_3dgs_ros/edge_3dgs_ros/cloud_publisher.py)
> **按真实 API 实现**：骨架的 `gaussians.n/xyz/scale/rot` 属性与 `GaussianModel`
> 不符——实际为 `means3D` / `opacities()` / `scales()` / `rotations()` /
> `params["rgb_colors"]`（CUDA 张量，经 `SLAMBackend.snapshot_model()` 锁内拷成
> numpy 后锁外建消息）。
>
> **发布策略（实测定稿）**：
> - **`/gaussian_map_pc2`（PointCloud2，rviz2 直接显示）每 1 秒发**：numpy
>   结构化数组 `tobytes()` 毫秒级构建，字段 x/y/z + intensity=opacity + rgb
>   （packed uint32，20B/点），与 docs/00 §7 降级口径一致；
> - **`/gaussian_map`（GaussianCloud，Phase 6 API 契约）每 5 秒发**：Python
>   逐元素构建实测 **25k 高斯 ≈ 580-800ms、50k ≈ 1.2-2.0s**——即便放独立
>   Reentrant 回调组仍会饿死 sync 回调（on_frame 延迟 ~570ms，tf 掉到 ~1Hz），
>   故必须低频（实测坑，§7）；`feature=[]` 留空（Phase 6 语言潜变量）；
> - 快照 `snapshot_model(max_points=50k)` 均匀抽稀：锁内 .cpu() 拷贝（5-15ms），
>   锁外建消息；`cloud_max_points` / `cloud_publish_hz` 可配置（config/ros2/params.yaml）。

```python
def publish(self, snap, stamp, gaussian_cloud=True):
    # snap: SLAMBackend.snapshot_model() 的锁内 numpy 快照
    # PointCloud2（每次发，毫秒级）：
    #   dtype = [("x","<f4"),("y","<f4"),("z","<f4"),("intensity","<f4"),("rgb","<u4")]
    # GaussianCloud（每 5 秒发一次，~600ms 构建）：
    #   Gaussian(x,y,z, opacity, scale_xyz, qxyzw, rgb, feature=[])  # Phase 6 填 feature
```

## 5. 启动与可视化

> ✅ **已实现（2026-08-29）**，真机启动序列（source 顺序关键，验证报告 §1）：

```bash
# ① 环境：/opt/ros/humble → semantic_ws（realsense2_camera 所在）→ 本项目 ws_src（最后，overlay）
source /opt/ros/humble/setup.bash
source /home/zc/semantic_ws/semantic_vins_fusion/install/setup.bash
source ws_src/install/setup.bash

# ② 起相机 + SLAM 节点（camera_namespace:=/ 必须：旧版 realsense-ros 缺了会 /camera/camera/ 双前缀）
ros2 launch realsense2_camera rs_launch.py camera_namespace:=/ align_depth.enable:=true &
ros2 run edge_3dgs_ros edge_3dgs_slam_node            # 在线实验模式（首帧初始化+建图，实测 3–9 Hz，见验证报告 §5）
ros2 run edge_3dgs_ros edge_3dgs_slam_node -- --load data/outputs/phase3/probe_model_replica.pt \
    --tier fps                                        # --load 只 track；--tier fps|quality 双档

# ③ 查看（回放验证另起终端跑 experiments/phase4_replay_publisher.py，见验证报告 §3）
ros2 topic hz /odom /tf /gaussian_map_pc2
ros2 run tf2_tools view_frames        # 看 tf 树（map→odom→camera）
rviz2                                 # Add → PointCloud2 /gaussian_map_pc2, Fixed Frame=map
```

> 无头环境（无 X 授权）下 rviz2 等价验证：`experiments/phase4_replay_publisher.py`
> + 数据级渲染脚本，见验证报告 §4（点云范围/覆盖统计 + tf 树 PDF）。

## 6. 验收清单

**验证数据约定**（docs/00 §7 统一口径）：接口/回放测试用 **Replica 或合成序列回放**为话题流
（离线评估数据 → 节点输入，可复现、无硬件依赖）；真机最终验证走 **D435i 自采**（Phase 1 数据流）。
两阶段都要做：回放验证接口正确性，真机验证采集/录制与节点稳定性（在线 SLAM 实时性仅作实验记录——quality 档 ~9.1 Hz / fps 档 ~3–5 Hz、稳定 10 FPS 不可达，结论见验证报告 §5，交付口径为离线重建）。

- [x] `colcon build` 通过，消息接口可 `ros2 interface show`。
      （2026-08-29：msgs 12.4s / ros 2.9s 构建 0 错误；5 个 msg 全部可查询）
- [x] 订阅 ros2 bag 或真机，`/tf` 与 `/gaussian_map` 稳定发布。
      （回放 Replica/合成 + 真机 D435i 两路验证，实测数据见验证报告）
- [x] `rviz2` 中看到高斯点云与轨迹，位姿无跳变。
      （无头环境：数据级等价验证——点云渲染 + tf 树 PDF + 位姿连续性 0 跳变，
      详见验证报告 §4/§3；rviz2 GUI 人工操作项待用户环境）

## 7. 常见坑

1. **回调里做重活**：Tracking 必须轻量，Mapping 丢 worker 线程，否则 DDS 掉消息。
   ✅ 已按此实现（cloud 构建在独立定时器，锁外建消息）。
2. **`numpy<2` ABI**：`cv_bridge` 与 opencv-python 冲突时锁 numpy 版本。
   ✅ 实测 numpy 1.26.4 + cv_bridge 正常。
3. **tf 时间戳**：用 `frame.header.stamp`，别用 `node.get_clock().now()`，避免 rviz 报 tf 抖动。
   ✅ 已按此实现；**补充（实测）**：回放工具的 `--loop` 必须每轮重置 t0——否则时间戳
   停在首轮启动时刻，与监听器当前时间的差持续拉大 → tf 缓存判定 `TF_OLD_DATA`
   丢弃全部变换。
4. **ROS_DOMAIN_ID**：宿主机本地直连无需特殊配置；多机通信时各机 `ROS_DOMAIN_ID` 保持一致。
   ✅ 未配置（默认 0），本地验证通过。

**Phase 4 实测新增坑**（2026-08-29，均已在实现中规避并验证）：

5. **tf 语义：发布 `inv(T_wc)`**：全项目 T_wc 为 w2c，相机世界位姿 = [Rᵀ | -Rᵀt]。
   骨架直接发 T_wc 会把相机放在 `-Rᵀt` 处（§3 修正）。
6. **GaussianCloud 逐元素构建 ~600ms/25k 高斯**：即使放独立 Reentrant 回调组仍会
   饿死 sync 回调（on_frame 延迟 ~570ms，tf 掉到 ~1Hz）。定稿：PointCloud2 1Hz
   （numpy tobytes 毫秒级）+ GaussianCloud 5s 一次。
7. **四元数转换用 numpy 而非 torch**：真机在线建图占 GPU 时，每帧 2 次 CUDA 四元数
   转换排队，on_frame 延迟 +1.4s（tf ~0.6Hz）。纯 numpy 版修复后 on_frame ≈ track 耗时。
8. **回放深度单位**：Replica/npz 深度是米，发布话题必须 ×1000 转 16UC1 毫米
   （D435iReader 内部再 ×0.001），单位不匹配 SLAM 直接废。
9. **npz 惰性解压陷阱**：`[d["rgb"][i] for i in range(n)]` 每次访问 `d["rgb"]` 都
   重新解压整个数组（NpzFile 缓存不命中）——60 帧时残留 60×55MB + 60×74MB ≈
   **7.7GB 内存** → DDS SHM 分配失败，publish 报 "publisher's context is invalid"。
   必须先取数组引用再切片：`rgb_arr = d["rgb"]; [rgb_arr[i] for i in range(n)]`。
10. **rclpy Node 必须经 MRO 可达**：`Edge3DGSSlamNode(D435iReader, Node)` 多重继承
    （只调 `Node.__init__` 而类不继承 Node 会 `AttributeError: 'handle'`）。
