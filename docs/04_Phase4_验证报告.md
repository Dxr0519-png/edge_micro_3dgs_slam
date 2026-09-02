# 04 · Phase 4 验证报告（2026-08-29）

> 按 [docs/04_Phase4_ROS2具身接口.md](04_Phase4_ROS2具身接口.md) §1-§6 顺序实现并逐一验证。
> 环境：宿主机 Jetson Orin NX Super（16GB，JetPack 6.2）+ ROS2 Humble（/opt/ros/humble），
> 非 docker。所有数字标注数据来源 + 分辨率 + 帧数（docs/00 §7 报告规范）。

## 1. 执行记录

| # | 步骤 | 命令 | 结果 |
|---|---|---|---|
| S1 | 消息包编译 | `cd ws_src && colcon build --packages-select edge_3dgs_msgs` | ✅ 0 错误（12.4s） |
| S1 | 消息接口 | `ros2 interface show edge_3dgs_msgs/msg/{Gaussian,GaussianCloud,QueryRequest,QueryResult,SemanticPoint}` | ✅ 5 个全部可查询，字段与 .msg 一致 |
| S2 | 节点包构建 | `colcon build --symlink-install --packages-select edge_3dgs_ros` | ✅ 0 错误（~3s） |
| S2 | 入口注册 | `ros2 pkg executables edge_3dgs_ros` | ✅ 列出 edge_3dgs_slam_node |
| S2 | 冒烟 | `ros2 run ... edge_3dgs_slam_node -- --load probe_model_replica.pt`（75s 无相机） | ✅ 模型加载 50726 高斯，等待相机数据，无 traceback |
| S3 | 回放验证 | Replica office0 200 帧（600×340）+ 合成 60 帧（480×640）→ 话题流 | ✅ 验证矩阵 §3 |
| S4 | 可视化 | view_frames + 点云渲染 | ✅ §4 |
| S5 | 真机验证 | realsense2_camera + 完整 SLAM 节点 | ✅ §5 |

## 2. 逐条核对表（docs/04 §1-§7 vs 实现）

| 文档声明 | 实现 | 核对 |
|---|---|---|
| §1 消息包可 `ros2 interface show` | ws_src/edge_3dgs_msgs 已构建 | ✅ |
| §2 节点骨架 8 文件 | ws_src/edge_3dgs_ros 全部落地 | ✅（与骨架差异见 §6 文档修订） |
| §3 发布位姿 | tf_publisher.py，**inv(T_wc) 语义修正** + map→odom→camera 两段 + /odom | ✅（文档已修正） |
| §4 发布高斯点云 | cloud_publisher.py，GaussianCloud + PointCloud2 双发 | ✅（API 差异 + 发布策略已回填） |
| §5 启动命令 | 真机验证跑通（source 顺序 + camera_namespace:=/） | ✅ |
| §6 验收 3 项 | 全部通过（见 §6） | ✅ |
| §7 坑 1-4 | 逐坑核对 + 新增 6 个实测坑（§7 回填） | ✅ |

## 3. 回放验证（接口正确性，docs/04 §3+§4）

**数据**：② Replica office0 前 200 帧 @0.5 scale（600×340，K'=[[300,0,299.75],[0,300,169.75]]）
+ ① 合成 60 帧（480×640，K=[[310,0,320],[0,310,240]]）；均 15Hz --loop 话题流。

**节点**：`--load probe_model_replica.pt`（Replica 模型 50726 高斯）/ `probe_model_synth.pt`
（134519 高斯），`--tier fps|quality`。

| 检查 | 命令 | PASS 判据 | 实测 |
|---|---|---|---|
| tf/odom 频率 | `ros2 topic hz /tf /odom` | 随 track 耗时自适应（诚实口径） | Replica quality: **9.1Hz**；fps: 5.8Hz；合成: 2.6Hz（134k 高斯） |
| 云频率 | `ros2 topic hz /gaussian_map_pc2` | ~1Hz | ✅ 0.95-1.0Hz |
| GaussianCloud 内容 | `ros2 topic echo /gaussian_map --once` | frame_id=map，字段齐全 | ✅ 25k 高斯，x/y/z/opacity/scale/rot/rgb 正常 |
| odom 内容 | `ros2 topic echo /odom --once` | 无 NaN、范数<100m、frame_id 正确 | ✅ frame=odom/child=camera，位置 -0.01m 级 |
| tf 树 | `ros2 run tf2_tools view_frames` | map→odom→camera | ✅ 三帧存在，camera 变换 15.9Hz（Replica quality），无 TF_OLD_DATA |
| 位姿连续性 | 订阅 /odom 统计 | 相邻帧平移<0.5m、旋转<45°，lost<10% | ✅ Replica quality: 4.4cm/0.8° 0 跳变；合成: 10.3cm/1.3° 0 跳变；NaN=0 |
| 时间戳 | 节点日志 | 无 message_filters/时间戳告警 | ✅（修复 loop t0 重置后） |

**注**：fps 档（2it）在 Replica 回放上输出接近静止（track 收敛不足，docs/03 §11
已诚实记录 fps10 档 ATE 25cm 与 2 迭代收敛限制）；接口正确性由 quality 档
（6it，位姿跟随 1.13m 运动、无跳变）证明。**fps 档静止是算法质量特性而非接口缺陷**。

## 4. 可视化验证（docs/04 §5，无头环境等价验证）

宿主机无 X 授权（rviz2 无法启动）→ 数据级等价验证：

- **tf 树**：view_frames 生成 frames.pdf（data/outputs/phase4/tf_tree_replica.pdf），
  FrameGraph 显示 `odom: parent=map` + `camera: parent=odom`（rate 15.9Hz，无
  TF_OLD_DATA）。
- **点云渲染**：订阅 /gaussian_map_pc2（44840 点，Replica office0）→ PIL 2D 投影
  渲染（data/outputs/phase4/cloud_render.png）：xyz 范围 x[-3.54,2.77] y[-0.91,1.78]
  z[0.44,4.64]（Replica 室内 3-5m 尺度合理），非背景像素覆盖 33%。
- **rviz2 GUI 人工项**：待用户有显示环境时复验（Add → PointCloud2
  /gaussian_map_pc2, Fixed Frame=map）。

## 5. 真机验证（docs/04 §6 阶段②，实时性）

**环境**：D435i（libusb 8086:0b3a）已连接；`ros2 launch realsense2_camera rs_launch.py
camera_namespace:=/ align_depth.enable:=true`（25Hz 数据流）。

**节点**：完整 SLAM 模式（无 --load），`--tier fps`。首帧 init_from_depth →
**104974 高斯**，在线建图增长至 148k（50 秒）。

| 检查 | 实测 | 结果 |
|---|---|---|
| 话题稳定 | /odom 1.8-2.1Hz 稳定（std 74ms；track 280-350ms/帧 的诚实频率，docs/03 §11 口径：本机稳定 10FPS 不可达，速度档 ~3-5Hz 为现实） | ✅ |
| 位姿无跳变 | 50 秒：track failure 1 次、extrapolation lost 1 次（均启动初期，回退最近好位姿保持连续性）；订阅统计 NaN=0、0 跳变 | ✅ |
| DDS/异常 | 全程无 traceback / RCLError / DDS 掉消息告警 | ✅ |
| 资源 | 内存峰值 5.3GB/15GB（节点 ~1.2GB + realsense + torch） | ✅ |
| 建图 | 104k → 148k 高斯在线增长；cloud 发布 49.5k 高斯（构建 ~2s/5s 低频） | ✅ |

**如实标注**：真机验证为**相机静止场景**（接口/实时性/稳定性）；运动跟踪正确性由
回放验证 quality 档覆盖（Replica 4.4cm 帧间增量跟随）。

## 6. 验收清单勾选

- [x] `colcon build` 通过，消息接口可 `ros2 interface show`。
- [x] 订阅回放（Replica/合成）与真机，`/tf` 与 `/gaussian_map` 稳定发布。
- [x] 看到高斯点云与轨迹，位姿无跳变（无头环境数据级等价验证 + 人工项待复验）。

## 7. 发现的问题清单（全部已修复并回填 docs/04 §7）

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | tf 语义错误风险 | T_wc 是 w2c，骨架直接发布导致相机位姿错位 | 发布 inv(T_wc)，docs/04 §3 修正 |
| 2 | cloud 构建饿死 on_frame（tf 1.5Hz） | GaussianCloud 逐元素构建 580-800ms，默认回调组串行 | Reentrant 组 + GaussianCloud 降频 5s + PointCloud2 1Hz |
| 3 | 真机 on_frame +1.4s | torch 四元数转换在建图占 GPU 时排队 | 纯 numpy 四元数（_matrix_to_quat_np） |
| 4 | TF_OLD_DATA | 回放 --loop 不重置 t0，时间戳与监听器差距拉大 | 每轮 loop 重置 t0 |
| 5 | publish "context is invalid" | npz 惰性解压，60 帧列表推导残留 7.7GB 内存 → DDS SHM 失败 | 先取数组引用再切片 |
| 6 | Node 'handle' AttributeError | rclpy Node 需经 MRO 可达 | `(D435iReader, Node)` 多重继承 |
| 7 | GaussianModel 无 rgb_colors 属性 | params 是 dict，非 attribute | `m.params["rgb_colors"]` |
| 8 | CameraInfo k 需 float | 整数混入 k 列表断言失败 | 显式 float() |

## 8. 已执行的文档/配置修订

- docs/04 §1-§7：✅ 标记 + 实现细节回填；**§3 tf 语义修正**（inv(T_wc)）；
  §4 发布策略定稿；§6 验收勾选；§7 新增 6 个实测坑。
- `config/ros2/params.yaml`：新增 Phase 4 键（cloud_publish_hz/cloud_max_points/
  init_stride/camera_frame/odom_frame），由 `edge_3dgs_ros/edge_3dgs_ros/config.py`
  加载（默认值 + yaml 深合并，--ros-args 可覆盖）。
- src 改动仅 1 处：`slam/backend.py` 新增 `snapshot_model()`（锁内 numpy 快照，
  Phase 6 查询服务复用）。

## 附录：输出文件

- `data/outputs/phase4/tf_tree_replica.pdf` — tf 树（map→odom→camera）
- `data/outputs/phase4/cloud_render.png` — 高斯点云 2D 投影渲染（44840 点）
- `ws_src/edge_3dgs_ros/edge_3dgs_ros/{node,tf_publisher,cloud_publisher,config}.py`
- `experiments/phase4_replay_publisher.py` — 回放工具（Replica/npz → 话题流）
