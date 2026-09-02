# 02 · Phase 2 SplaTAM 代码精读指南

> 配套文档：[02_Phase2_RGBD_3DGS_SLAM基线.md](02_Phase2_RGBD_3DGS_SLAM基线.md)。代码在 [third_party/SplaTAM/](../third_party/SplaTAM/)（CVPR 2024 重构版，commit `da6bbcd`）。
> 目标：逐讲精读后，能回答"SplaTAM 每一行优化在做什么、为什么"，并能动手裁剪到 Jetson。
> 前置：已跑通 Phase 1 数据流；对 SE(3)/相机投影有直觉。总用时约 2–3 个工作日（每天 4–6 小时）。
> 💡 本文所有 `文件:行号` 均为可点击链接，点击可直接跳转到对应代码位置。

## 0. 怎么用这份指南

- 共 6 讲，**按数据流顺序**读：世界系 → 渲染 → Tracking → Mapping → 主循环 →（选读）光栅化。不要按文件顺序读。
- 每讲结构：**必读**（函数 + 行号）→ **精读要点**（读的时候盯住什么）→ **自测题**（读完合上代码能答）。
- 每讲读完，往你的笔记里填一节（模板见 §9）。**不会答的自测题 = 该讲没读完，回头再看**。
- 行号以本地 [third_party/SplaTAM/](../third_party/SplaTAM/) 为准；函数级阅读用 `grep -n "^def "` 自查。

## 1. 全局地图：一个文件速览 + 调用关系

**调用关系图**（点击函数名跳转）：

- 入口：[`rgbd_slam(config)`](../third_party/SplaTAM/scripts/splatam.py#L455)（[scripts/splatam.py](../third_party/SplaTAM/scripts/splatam.py)，由 [configs/replica/splatam.py](../third_party/SplaTAM/configs/replica/splatam.py) 驱动）
  - **初始化**：[`initialize_first_timestep`](../third_party/SplaTAM/scripts/splatam.py#L169)
    - [`get_pointcloud`](../third_party/SplaTAM/scripts/splatam.py#L67) → [`initialize_params`](../third_party/SplaTAM/scripts/splatam.py#L120) → [`setup_camera`](../third_party/SplaTAM/utils/recon_helpers.py#L4)
  - **Tracking 循环**：[`splatam.py:676`](../third_party/SplaTAM/scripts/splatam.py#L676)
    - [`get_loss`](../third_party/SplaTAM/scripts/splatam.py#L214) → [`transform_to_frame`](../third_party/SplaTAM/utils/slam_helpers.py#L252) → [`transformed_params2rendervar`](../third_party/SplaTAM/utils/slam_helpers.py#L124) → `diff_gaussian_rasterization`（CUDA 黑盒，§8）
  - **Mapping 循环**：[`splatam.py:776`](../third_party/SplaTAM/scripts/splatam.py#L776)
    - [`add_new_gaussians`](../third_party/SplaTAM/scripts/splatam.py#L378) → [`keyframe_selection_overlap`](../third_party/SplaTAM/utils/keyframe_selection.py#L40) → [`prune_gaussians`](../third_party/SplaTAM/utils/slam_external.py#L167) / [`densify`](../third_party/SplaTAM/utils/slam_external.py#L191)

| 文件 | 行数 | 精读优先级 | 职责 |
|---|---|---|---|
| [scripts/splatam.py](../third_party/SplaTAM/scripts/splatam.py) | 1013 | ★★★ 核心 | 主循环 + 损失 + 初始化 + 加高斯，几乎所有逻辑 |
| [utils/slam_helpers.py](../third_party/SplaTAM/utils/slam_helpers.py) | 303 | ★★★ | 位姿变换、高斯→渲染参数、损失函数（**与 gs_helpers.py 重复，只读这份**） |
| [utils/slam_external.py](../third_party/SplaTAM/utils/slam_external.py) | 287 | ★★☆ | 3DGS 版权代码：SSIM、densify、prune、参数拼接/删除 |
| [utils/keyframe_selection.py](../third_party/SplaTAM/utils/keyframe_selection.py) | 95 | ★★☆ | 关键帧重叠度选择 |
| [utils/recon_helpers.py](../third_party/SplaTAM/utils/recon_helpers.py) | 27 | ★★★ | `setup_camera`：rasterizer 的相机设置（小但关键） |
| [utils/graphics_utils.py](../third_party/SplaTAM/utils/graphics_utils.py) | 76 | ★☆☆ | 投影矩阵、fov↔focal（对照用，可扫读） |
| [utils/neighbor_search.py](../third_party/SplaTAM/utils/neighbor_search.py) | 35 | ★★☆ | KNN 邻居搜索（新高斯初始化用） |
| [utils/common_utils.py](../third_party/SplaTAM/utils/common_utils.py) | 73 | ★☆☆ | checkpoint 保存、随机种子 |
| `datasets/gradslam_datasets/` | — | ☆☆☆ 跳过 | 数据集加载，你们用 D435i 数据流，不读 |
| [diff-gaussian-rasterization-w-depth.git/](../third_party/SplaTAM/diff-gaussian-rasterization-w-depth.git/) | ~600 | ★☆☆ 选读 | 可微光栅化 CUDA，§8 只读接口 |

**读代码前先立两个心锚**（整份代码围绕它们展开）：
1. `params` 是 dict of tensors：`means3D/quats→unnorm_rotations/scales→log_scales/opacity→logit_opacities/color→rgb_colors` + **每帧一份**的 `cam_unnorm_rots (4,N)` / `cam_trans (3,N)`。所有可学习量都在这一个 dict 里，Tracking/Mapping 只是切换其中谁的 `requires_grad`。
2. 相机位姿是世界系 **w2c**（世界→相机），相对第一帧。rasterizer 只认 OpenGL 风格相机（§3）。

---

## 2. 第 1 讲：世界系、相机与高斯参数（45–60 min）

**必读**（共 ~90 行，全读）：
- [`utils/recon_helpers.py`](../third_party/SplaTAM/utils/recon_helpers.py) 全文（27 行）
- [`scripts/splatam.py:67-117`](../third_party/SplaTAM/scripts/splatam.py#L67-L117) `get_pointcloud`、[`:120-157`](../third_party/SplaTAM/scripts/splatam.py#L120-L157) `initialize_params`、[`:169-211`](../third_party/SplaTAM/scripts/splatam.py#L169-L211) `initialize_first_timestep`
- 对照：[`utils/graphics_utils.py:51-76`](../third_party/SplaTAM/utils/graphics_utils.py#L51-L76)（投影矩阵公式）

**精读要点**：
1. **setup_camera 的 OpenGL 约定**（[recon_helpers.py:4-27](../third_party/SplaTAM/utils/recon_helpers.py#L4-L27)）：rasterizer 要 `viewmatrix`（w2c 转置）、`projmatrix`（w2c ⊙ opengl_proj 的"列向量投影矩阵"）、`tanfovx/y`、`campos`。它内部用的是 OpenGL 风格（x 右 y 上 z 朝后、NDC 列主序）。**你们后面接 D435i 内参走的就是这里**——换成自己的 K 只需要构造同样的 `Camera`。
2. **get_pointcloud**（[splatam.py:67](../third_party/SplaTAM/scripts/splatam.py#L67)）：`xx=(x−CX)/FX` 归一化像素坐标 → `pts_cam=(xx·d, yy·d, d)` 相机系 → 乘 `c2w=inv(w2c)` 进世界系。`mean_sq_dist_method="projective"` 时尺度按 `(d/f)²` 初始化（远处高斯更大，[splatam.py:97-100](../third_party/SplaTAM/scripts/splatam.py#L97-L100)）。
3. **initialize_params**（[splatam.py:120](../third_party/SplaTAM/scripts/splatam.py#L120)）——读完要能默写：
   - 旋转初始化为**单位四元数** `[1,0,0,0]`；opacity 初始 `logit=0`（即 σ(0)=0.5）
   - `isotropic` 时 `log_scales` 形状 **(N,1)**（渲染时 tile 成 3 列，[slam_helpers.py:126-127](../third_party/SplaTAM/utils/slam_helpers.py#L126-L127)）——这是全代码 `log_scales.shape[1]==1` 判断的由来
   - `cam_unnorm_rots` 形状 (4, num_frames)、`cam_trans` (3, num_frames)，**全部初始化为第 0 帧位姿**（[splatam.py:140-143](../third_party/SplaTAM/scripts/splatam.py#L140-L143)）→ 所有帧位姿都是相对第一帧的，Tracking 只改自己那一列
   - `variables` 四个成员：`max_2D_radius`、`means2D_gradient_accum`、`denom`（densify 统计用）、`timestep`（每个高斯"出生"的帧号，裁剪新高斯时记）
4. `initialize_first_timestep`（[splatam.py:169](../third_party/SplaTAM/scripts/splatam.py#L169)）：拿第一帧反投影全部有效深度像素 → 初始点云 → 高斯化。**地图的起点就是第一帧的像素**。

**自测题**：
- w2c 和 c2w 在代码里分别是哪几个变量？GT 位姿为什么要 `torch.linalg.inv`（[splatam.py:180](../third_party/SplaTAM/scripts/splatam.py#L180)、[`:647`](../third_party/SplaTAM/scripts/splatam.py#L647)）？
- 为什么 `log_scales` 各向同性时存 (N,1) 而不是 (N,3)？渲染前在哪补成 3 列？
- 第 5 帧的相机参数初始值是什么？"相对第一帧"具体体现在哪行？
- `variables['timestep']` 有什么用？搜一下它在哪被使用。

---

## 3. 第 2 讲：渲染管线（60–90 min）

**必读**：
- [`utils/slam_helpers.py:124-139`](../third_party/SplaTAM/utils/slam_helpers.py#L124-L139) `transformed_params2rendervar`、[`:196-249`](../third_party/SplaTAM/utils/slam_helpers.py#L196-L249) `get_depth_and_silhouette` + `transformed_params2depthplussilhouette`
- [`utils/slam_helpers.py:252-303`](../third_party/SplaTAM/utils/slam_helpers.py#L252-L303) `transform_to_frame`（**本讲核心**）
- [`utils/slam_helpers.py:21-29`](../third_party/SplaTAM/utils/slam_helpers.py#L21-L29) `quat_mult`
- 扫读：[`utils/slam_helpers.py:1-19`](../third_party/SplaTAM/utils/slam_helpers.py#L1-L19) 损失函数、[`:106-122`](../third_party/SplaTAM/utils/slam_helpers.py#L106-L122) `params2rendervar`（无 transform 版本，对比着看）

**精读要点**：
1. **transform_to_frame 的梯度开关**（[slam_helpers.py:252-303](../third_party/SplaTAM/utils/slam_helpers.py#L252-L303)）：三个参数 `(params, time_idx, gaussians_grad, camera_grad)`。Tracking 调 `(gaussians_grad=False, camera_grad=True)`，Mapping 反过来——**这就是整个 SLAM 里"谁在动"的开关**。主循环里对应的调用在 [splatam.py:222-240](../third_party/SplaTAM/scripts/splatam.py#L222-L240)。
2. **位姿变换**：`rel_w2c[:3,:3]=build_rotation(cam_rot)` + `translation`；高斯中心 `pts4 @ rel_w2c`（[slam_helpers.py:292-295](../third_party/SplaTAM/utils/slam_helpers.py#L292-L295)）。**各向异性高斯要把旋转也转到相机系**：`quat_mult(cam_rot, norm_rots)`（[slam_helpers.py:297-300](../third_party/SplaTAM/utils/slam_helpers.py#L297-L300)）——注意是**左乘**（相机旋转在前）。各向同性时跳过。
3. **一次光栅化同时出深度+不透明度+不确定性**（SplaTAM 的核心 trick，[slam_helpers.py:196-213](../third_party/SplaTAM/utils/slam_helpers.py#L196-L213)）：把每个高斯的"颜色"设为 `(z, 1, z²)`，α 混合后第一通道是深度期望 `Σαz/Σα`、第二通道是 silhouette `Σα`、第三通道是 `Σαz²/Σα`；在 get_loss 里 `uncertainty = depth_sq − depth²`（[splatam.py:257-259](../third_party/SplaTAM/scripts/splatam.py#L257-L259)）——**方差是免费附赠的**。
4. **为什么每次损失要渲染两遍**（[splatam.py:248-253](../third_party/SplaTAM/scripts/splatam.py#L248-L253)）：一遍 RGB（`transformed_params2rendervar`，颜色为真 RGB），一遍 depth+sil（颜色为 z,1,z²）。两遍前向 = 主要算力开销，裁剪时这里是第一热点。
5. `means2D.retain_grad()`（[splatam.py:248](../third_party/SplaTAM/scripts/splatam.py#L248)）：光栅化输出的像素坐标梯度被保留，供 densify 的 2D 梯度累积用（[slam_external.py:100](../third_party/SplaTAM/utils/slam_external.py#L100)）。
6. rasterizer 接口：`Renderer(raster_settings=cam)(**rendervar)`，输入是预变换后的高斯参数 dict，输出 `(image, radius, depth)`。**读到这里只需知道"喂什么吐什么"**，CUDA 内部见 §8。

**自测题**：
- Tracking 和 Mapping 调 `transform_to_frame` 时四个参数分别是什么？为什么这么设？
- silhouette 通道为什么渲染成常数 1？混合后它等于什么？
- 不确定性 `depth_sq − depth²` 来自哪两个通道？它在代码里被怎么用（搜 `uncertainty`）？
- 为什么各向同性高斯不需要转旋转？
- 渲染两遍时哪些变量是共享的、哪些是每次重算的？

---

## 4. 第 3 讲：损失函数与 Tracking（60–90 min）

**必读**：
- [`scripts/splatam.py:214-349`](../third_party/SplaTAM/scripts/splatam.py#L214-L349) `get_loss`（**本讲核心**，含 [274-291](../third_party/SplaTAM/scripts/splatam.py#L274-L291) 的损失主体与 [292-337](../third_party/SplaTAM/scripts/splatam.py#L292-L337) 的可视化）
- [`scripts/splatam.py:160-166`](../third_party/SplaTAM/scripts/splatam.py#L160-L166) `initialize_optimizer`、[`:423-443`](../third_party/SplaTAM/scripts/splatam.py#L423-L443) `initialize_camera_pose`
- [`scripts/splatam.py:676-758`](../third_party/SplaTAM/scripts/splatam.py#L676-L758) Tracking 循环
- 对照配置：[configs/replica/splatam.py](../third_party/SplaTAM/configs/replica/splatam.py) 的 `tracking:` 段（lrs、loss_weights、sil_thres）

**精读要点**：
1. **mask 的三层构造**（[splatam.py:261-272](../third_party/SplaTAM/scripts/splatam.py#L261-L272)）：
   - 有效深度：`gt_depth > 0`（D435i 无效像素=0，你们数据流同理）
   - 离群截断（可选 `ignore_outlier_depth_loss`）：`|gt−render| < 10·median`（[splatam.py:263-266](../third_party/SplaTAM/scripts/splatam.py#L263-L266)）
   - Tracking 专属：`& presence_sil_mask`（silhouette > sil_thres，只算"已有地图覆盖"的像素 → **新看到的区域不参与位姿优化**，这是 SLAM 鲁棒性的关键）
2. **Tracking vs Mapping 的损失形式不同**：
   - Tracking：depth 用 **sum**、RGB 用 **sum**（带 mask，[splatam.py:278](../third_party/SplaTAM/scripts/splatam.py#L278)、[`:283-286`](../third_party/SplaTAM/scripts/splatam.py#L283-L286)）——位姿对每个像素的小误差都敏感
   - Mapping：depth 用 **mean**、RGB 用 `0.8·L1 + 0.2·(1−SSIM)`（[splatam.py:280](../third_party/SplaTAM/scripts/splatam.py#L280)、[`:290`](../third_party/SplaTAM/scripts/splatam.py#L290)）——SSIM 保纹理
3. **initialize_optimizer**（[splatam.py:160-166](../third_party/SplaTAM/scripts/splatam.py#L160-L166)）：所有参数组进一个 Adam，但 `lrs` 决定谁有学习率。Tracking 的 lrs 里高斯参数全 0、只有 `cam_unnorm_rots=4e-4, cam_trans=2e-3`；Mapping 反之（相机 0、高斯非 0）。**"只动位姿/只动地图"是靠学习率 0 实现的，不是靠参数隔离**。
4. **best-candidate 机制**（[splatam.py:682-711](../third_party/SplaTAM/scripts/splatam.py#L682-L711)、[`:741-744`](../third_party/SplaTAM/scripts/splatam.py#L741-L744)）：Tracking 每轮记录损失最小的候选位姿，迭代结束才写回 → 中途发散不污染结果。
5. **initialize_camera_pose**（[splatam.py:423](../third_party/SplaTAM/scripts/splatam.py#L423)）：`forward_prop=True` 时**恒速外推**：`t_new = t1 + (t1 − t2)`，旋转在四元数空间同样外推后归一化；否则复制上一帧。
6. 可选 `use_depth_loss_thres`：depth 损失没降下来就把迭代数翻倍重跑一轮（[splatam.py:727-738](../third_party/SplaTAM/scripts/splatam.py#L727-L738)）。Replica 默认关。

**自测题**：
- Tracking 时 mask 里 `presence_sil_mask` 的作用是什么？如果关掉会发生什么（想想相机看向新区域时）？
- 为什么 Tracking 用 sum、Mapping 用 mean？调换会有什么影响？
- `cam_unnorm_rots` 为什么在 transform 前要 `F.normalize`？初始值 [1,0,0,0] 是单位四元数，为什么不能直接用？
- best-candidate 和"最后一步的结果"哪个被写回？为什么？
- Tracking 的 lrs 里 `means3D=0.0` 意味着什么？优化器里这个参数组还在吗？

---

## 5. 第 4 讲：地图增长与 Mapping（90–120 min）

**必读**：
- [`scripts/splatam.py:350-375`](../third_party/SplaTAM/scripts/splatam.py#L350-L375) `initialize_new_params`、[`:378-422`](../third_party/SplaTAM/scripts/splatam.py#L378-L422) `add_new_gaussians`（**核心**）
- [`scripts/splatam.py:776-909`](../third_party/SplaTAM/scripts/splatam.py#L776-L909) Mapping 循环（加高斯、选关键帧、建图迭代）
- [`utils/keyframe_selection.py`](../third_party/SplaTAM/utils/keyframe_selection.py) 全文（95 行）
- [`utils/neighbor_search.py`](../third_party/SplaTAM/utils/neighbor_search.py) 全文（35 行）
- [`utils/slam_external.py:100-161`](../third_party/SplaTAM/utils/slam_external.py#L100-L161)（梯度累积、参数拼接/删除）+ [`:167-188`](../third_party/SplaTAM/utils/slam_external.py#L167-L188) `prune_gaussians` + [`:191-243`](../third_party/SplaTAM/utils/slam_external.py#L191-L243) `densify`
- 对照配置：[configs/replica/splatam.py](../third_party/SplaTAM/configs/replica/splatam.py) 的 `mapping:` 段的 `sil_thres=0.5`、`pruning_dict`、`densify_dict`

**精读要点**：
1. **新高斯从哪来——"非存在" mask**（[splatam.py:380-393](../third_party/SplaTAM/scripts/splatam.py#L380-L393)）：渲染 silhouette 后，`(silhouette < sil_thres)` 是地图没覆盖的像素；**加一个深度条件**：`render_depth > gt_depth` 且误差 > 50·median → 捕捉"新出现的前景物体"（之前被遮挡、现在露出来了）。两者取或。
2. **新高斯初始化**（[splatam.py:394-423](../third_party/SplaTAM/scripts/splatam.py#L394-L423)）：mask 像素反投影成点云 → `initialize_new_params`（单位四元数、opacity 0.5、projective 尺度）→ `torch.cat` 拼进 params → **重置** `means2D_gradient_accum/denom/max_2D_radius` 为新长度 → `timestep` 记当前帧。注意：颜色直接用像素 RGB（`new_pt_cld[:,3:6]`），不是 KNN 继承——KNN（`calculate_neighbors`）在**尺度**初始化里用（追 `mean3_sq_dist` 的 `knn` 分支）。
3. **关键帧选择**（[keyframe_selection.py:40-95](../third_party/SplaTAM/utils/keyframe_selection.py#L40-L95)）：当前帧深度随机采 1600 像素 → 反投影 → 投影到每个关键帧视角 → 落在画面内（留 20px 边）的比例排序 → 取重叠最大的 k 个。主循环里窗口 = `mapping_window_size−2` 个重叠帧 + 上一关键帧 + 当前帧（[splatam.py:808-817](../third_party/SplaTAM/scripts/splatam.py#L808-L817)）。
4. **建图迭代**（[splatam.py:828-891](../third_party/SplaTAM/scripts/splatam.py#L828-L891)）：每步**从窗口随机抽一帧**（关键帧或当前帧）算 `get_loss(mapping=True)`；`torch.no_grad()` 里先 `prune` 再 `densify` 再 `optimizer.step()`。
5. **prune_gaussians**（[slam_external.py:167](../third_party/SplaTAM/utils/slam_external.py#L167)）：按 `σ(logit_opacities) < removal_opacity_threshold` 裁掉半透明高斯；`iter ≥ remove_big_after` 时再加"尺度 > 0.1·scene_radius"的大点。删除走 `remove_points`（[slam_external.py:139](../third_party/SplaTAM/utils/slam_external.py#L139)）——它同步删优化器里对应参数行。
6. **densify**（[slam_external.py:191](../third_party/SplaTAM/utils/slam_external.py#L191)）：3DGS 原版机制——`means2D_gradient_accum/denom` 是"每个高斯在屏幕上的梯度均值"；超阈值且**小**的 clone，超阈值且**大**的 split（`num_to_split_into` 份，沿旋转轴加噪声，尺度除 `0.8n`，[slam_external.py:198-217](../third_party/SplaTAM/utils/slam_external.py#L198-L217)）。SplaTAM 默认在 Mapping 里**关掉**它（`use_gaussian_splatting_densification=False`），靠 §5.1 的 add_new_gaussians 生长——读代码时留意这个取舍。
7. `variables['scene_radius'] = max(depth)/3`（[splatam.py:206](../third_party/SplaTAM/scripts/splatam.py#L206)）：prune/densify 的"多大算大"阈值，初始化时从第一帧深度估出。

**自测题**：
- 新增高斯的三种判定（silhouette 阈值、深度差、mask 取或）分别防什么情况？
- 为什么加完高斯要重置 `means2D_gradient_accum` 等三个 variables？
- 关键帧选择为什么用"投影落在画面内的比例"而不是深度重合度？比例 >0 的还随机打乱（[keyframe_selection.py:93-94](../third_party/SplaTAM/utils/keyframe_selection.py#L93-L94)），为什么？
- clone 和 split 的判定条件分别是什么？`num_to_split_into` 后尺度为什么除 `0.8·n`？
- 为什么 `optimizer.step()` 在 `torch.no_grad()` 里，而 loss.backward() 在外面？
- Mapping 迭代里 `iter_time_idx` 为什么每步都可能不同？它影响什么（想 transform_to_frame 的参数）？

---

## 6. 第 5 讲：主循环串讲 rgbd_slam（90–120 min）

**必读**：[`scripts/splatam.py:455-971`](../third_party/SplaTAM/scripts/splatam.py#L455-L971) 整段。这是验收讲——前面 4 讲的内容在这里全部汇合。

**精读要点**：照下面的流程表逐段对，确认每段与前面各讲的对应关系：

| 阶段 | 行号 | 对应前文 | 备注 |
|---|---|---|---|
| 数据集/分辨率检查 | [486-557](../third_party/SplaTAM/scripts/splatam.py#L486-L557) | — | `seperate_densification_res/tracking_res`：建图和跟踪可用不同分辨率（低分辨率加速） |
| 第一帧初始化 | [553-565](../third_party/SplaTAM/scripts/splatam.py#L553-L565) | 第 1 讲 | `initialize_first_timestep` |
| checkpoint 恢复 | [604-638](../third_party/SplaTAM/scripts/splatam.py#L604-L638) | — | 恢复 params + keyframe 列表，断点续跑 |
| 帧数据加载 | [643-657](../third_party/SplaTAM/scripts/splatam.py#L643-L657) | — | `gt_w2c=inv(gt_pose)`；`curr_data` 字典是 get_loss 的输入契约 |
| 位姿初始化 | [672-674](../third_party/SplaTAM/scripts/splatam.py#L672-L674) | 第 3 讲·5 | 恒速外推 |
| **Tracking** | [676-758](../third_party/SplaTAM/scripts/splatam.py#L676-L758) | 第 3 讲 | best-candidate 收尾写回 [741-744](../third_party/SplaTAM/scripts/splatam.py#L741-L744)；GT 位姿模式 [745-754](../third_party/SplaTAM/scripts/splatam.py#L745-L754) |
| 加高斯 | [779-798](../third_party/SplaTAM/scripts/splatam.py#L779-L798) | 第 4 讲·1-2 | 每 `map_every` 帧 |
| 选关键帧 | [800-819](../third_party/SplaTAM/scripts/splatam.py#L800-L819) | 第 4 讲·3 | 打印 `Selected Keyframes` 可观察窗口变化 |
| **Mapping** | [824-891](../third_party/SplaTAM/scripts/splatam.py#L824-L891) | 第 4 讲·4-6 | 随机抽帧 + prune/densify |
| 关键帧入库 | [911-925](../third_party/SplaTAM/scripts/splatam.py#L911-L925) | — | 每 `keyframe_every` 帧，存 `{id, est_w2c, color, depth}` |
| checkpoint | [927-931](../third_party/SplaTAM/scripts/splatam.py#L927-L931) | — | `params{t}.npz` + `keyframe_time_indices{t}.npy` |
| 耗时统计 | [939-959](../third_party/SplaTAM/scripts/splatam.py#L939-L959) | — | **Jetson 性能基线从这里取** |
| 最终评估 | [961-971](../third_party/SplaTAM/scripts/splatam.py#L961-L971) | — | `eval()`：ATE/PSNR |

**收尾任务**（读完必做，验证是否真懂）：画出你自己的调用图——拿 [src/edge_3dgs_slam/slam/tracking.py](../src/edge_3dgs_slam/slam/tracking.py) 和 [mapping.py](../src/edge_3dgs_slam/slam/mapping.py) 的骨架，对照这张表标出：SplaTAM 的哪个函数对应你 src 里的哪个方法、哪些逻辑你裁剪时不打算要（wandb、可视化、GT 位姿模式、checkpoint）。

**自测题**：
- `map_every=1`、`keyframe_every=5`、`mapping_window_size=24` 时，第 30 帧的建图窗口是哪几帧？（答：24−2=22 个重叠帧 + 关键帧 25/30 中在窗口内的 + 当前帧）
- 中断后 `load_checkpoint=True` 续跑，keyframe 列表从哪来？为什么必须一起存？
- 把 `num_iters_mapping` 设 0 会发生什么？这为什么是调试时的第一个旋钮？
- 想让 Tracking 也快（Jetson 上），你会先动 `num_iters`、`map_every`、还是分辨率？为什么？

---

## 7. 第 6 讲（选读，≤45 min）：光栅化 CUDA——只读接口，不啃内核

**必读**（只读 3 处）：
- [diff-gaussian-rasterization-w-depth.git/setup.py](../third_party/SplaTAM/diff-gaussian-rasterization-w-depth.git/setup.py)：看打包了什么、架构标志
- [diff-gaussian-rasterization-w-depth.git/diff_gaussian_rasterization/__init__.py](../third_party/SplaTAM/diff-gaussian-rasterization-w-depth.git/diff_gaussian_rasterization/__init__.py)：`GaussianRasterizationSettings` / `GaussianRasterizer` 的完整签名（**对照 [recon_helpers.py](../third_party/SplaTAM/utils/recon_helpers.py) 的 Camera 构造**）
- [rasterize_points.cu](../third_party/SplaTAM/diff-gaussian-rasterization-w-depth.git/rasterize_points.cu) 里 `forward` 的大致骨架：`computeGaussianCovariance3D → computeCov2D → computeColorFromSH → checkInsideFrustum → alpha compositing` 的顺序即可，不用逐行

**要点**：它是"给每高斯 (均值, 协方差, 颜色, 不透明度, 2D 屏幕坐标) → 输出逐像素 α 混合结果 + 对每个输入的梯度"的稠密 CUDA 内核。SplaTAM 的改动点是把渲染**像素的颜色**从 SH 换成"每高斯自带的 3 通道值"（RGB 或 z/1/z²）——所以 get_loss 能用同一内核渲染两种图。**裁剪时不需要动它**；只有要提速（比如低分辨率渲染）才需要碰。

---

## 8. 精读笔记模板（每讲填一节）

```markdown
### 第 X 讲：<主题>　（日期 / 用时）
**一句话总结**：<如果只能记一句>
**关键机制**（3-5 条，每条 ≤1 行）：
- ...
**行号速查**：
- <机制> → [third_party/SplaTAM/<文件>:<行号>](../third_party/SplaTAM/<文件>#L<行号>)  ← 填笔记时直接写链接
**不懂/存疑**：
- <问题>（问 Claude 或查论文）
**裁剪点**：
- <这个机制在我们 Jetson 裁剪里要保留/砍掉/改参数>
```

## 9. 裁剪点速查表（读完代码后填这张表）

| 配置项 | Replica 默认 | 位置 | 裁剪时考虑 |
|---|---|---|---|
| `tracking.num_iters` | 40 | [configs/replica/splatam.py](../third_party/SplaTAM/configs/replica/splatam.py) | 降它 = 直接降延迟 |
| `mapping.num_iters` | 60 | 同上 | 同上 |
| `map_every` | 1 | 同上 | 隔帧建图 = 一半算力 |
| `mapping_window_size` | 24 | 同上 | 窗口内关键帧数，控制内存/算力 |
| `keyframe_every` | 5 | 同上 | 关键帧密度 |
| `sil_thres`（加高斯） | 0.5 | mapping 段 | 越高越激进加高斯 |
| `sil_thres`（Tracking mask） | 0.99 | tracking 段 | 越低越信任新区域 |
| `loss_weights` | im=0.5, depth=1.0 | tracking/mapping 段 | D435i 深度噪声大时可调 |
| `pruning_dict` / `densify_dict` | 见配置 | mapping 段 | 与 `num_iters` 联动（stop_after 等） |
| `desired_image_height/width` | 680×1200 | data 段 | 低分辨率渲染是最大杠杆 |
| `gaussian_distribution` | isotropic | 顶层 | anisotropic 质量更好但更慢 |

## 10. 验收清单（全部完成 = 精读通过）

- [ ] 能默写 `params` / `variables` 全部键及其形状
- [ ] 能说清 Tracking/Mapping 交替时"谁 requires_grad、谁 lr=0"
- [ ] 能画出 get_loss 里 mask 的构造与三处损失的流向
- [ ] 能讲出 z/1/z² 三通道一次渲染出深度+silhouette+方差
- [ ] 能讲出加高斯的判定条件与 prune/densify 的触发条件
- [ ] 能默写主循环每帧的四阶段顺序与各自行号范围
- [ ] §6 收尾任务完成：src 骨架与 SplaTAM 的对应表已画
- [ ] 笔记 6 节齐，裁剪点速查表已填
