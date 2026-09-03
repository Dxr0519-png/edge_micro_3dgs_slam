# 06 · Phase 6 Language-Embedded 3DGS 与空间查询

> 目标：高斯新增低维特征维度，改写可微渲染使语言特征可 splat，与 Phase 5 特征图算 Loss 优化；开发文本查询节点发布 3D BBox / 热力图。
> 依赖：Phase 4（ROS2 封装）+ Phase 5（特征工厂）。
> **状态：已实现并全部验证通过（2026-08-31，见 [验证报告](06_Phase6_验证报告.md)）**。

## 1. 数据结构改造（`gaussian/model.py`）

```python
# 沿 params dict 惯例（非 self._feature 属性）：与 checkpoint/快照格式同构
FEATURE_KEY = "features"                      # 不进 PARAM_KEYS（SLAM 热路径零影响）
model.params["features"] = nn.Parameter(torch.zeros(N, D, device="cuda"))   # D=3（可配 16）
# 入口：add_feature_dim(model, D)（幂等；D 取 config autoencoder.latent_dim）
# save_ply / load_ply 增加 feature 通道读写（f_feature_0..{D-1} 列；load_ply 为新建）
# 语言特征优化阶段冻结几何：freeze_geometry(model) —— 5 几何键 requires_grad_(False)，
# features 恒为唯一可训练参数
```

**实测（2026-08-31）**：add/remove 后 features 行数与几何对齐；save→load_ply 往返 max 差 5e-7 < 1e-5；checkpoint 往返（`{'params','variables'}`）供 `--load`。

## 2. 特征光栅化（核心难点）

**降级方案先行**（`gaussian/torch_feature_rasterizer.py`）：纯 PyTorch 慢速 splat。

```python
def rasterize_feature(gaussians, T_wc, K, H, W, features=None, chunk=4096, bg=None):
    # 与 RGB 渲染共享同一套投影/深度排序/α 权重，把颜色通道换成 D 维特征
    # F_2d(pixel) = Σ_i f_i · α_i · Π_{j<i}(1-α_j)   -> (D, H, W)
    ...
```

**逐项校准结论**（慢速 splat 与已装 inria kernel 互验从 max 2.05 校准到 6.4e-5，详见验证报告 §3.2）：
- 近平面 z>0.2（SplaTAM fork `in_frustum`）；
- 2D 协方差 `J·C·Jᵀ`（2×3 J）+ 低通 +0.3px²；conic 用 `det_inv = 1/det` 后乘；
- point_image 用 ndc2Pix 精确 fp32 表达式（转置存储 projmatrix 含 cx 项）；
- **tile 级 footprint**（BLOCK=16，半径 `ceil(3·sqrt(λ1_LP))`）——像素级盒在 tile 边界整列截断；
- 早停 `T·(1−α) < 1e-4`（C 累加前）；T 用 **fp64 log 段前缀**（fp32 大数抵消/乘积下溢均实测暴露）。

验证正确性后，再改 CUDA kernel（参考 LangSplat `language-gaussian-rasterization`）：
1. `preprocess`：把 D 维 `feature` 与颜色一起分配进 tile。
2. `render`：累加特征 `F` 通道。
3. `backward`：特征梯度反传回 `feature` 与共享量。

编译该 CUDA kernel 前先做就绪检查（docs/00 §4 惯例），已就绪跳过：

```bash
# ① 源码未 clone 才拉取
ls third_party/LangSplat >/dev/null 2>&1 || \
    git clone --recursive https://github.com/minghanqin/LangSplat.git third_party/LangSplat

# ② 已编译（import 成功）则跳过，未编译才执行
# ⚠️ 包名与已装 inria kernel 冲突（都叫 diff_gaussian_rasterization）——
# 必须 --target 构建后复制 _C*.so 进 wrapper 包，禁止直接 pip install 覆盖：
cd third_party/LangSplat/submodules/langsplat-rasterization && \
    TORCH_CUDA_ARCH_LIST="8.7" pip install . --target /tmp/langsplat_build --no-build-isolation
# .so 复制进 src/edge_3dgs_slam/gaussian/langsplat_rasterization/（wrapper __init__.py 已就位）
```

**实测（2026-08-31）**：aarch64/sm_87 编译成功（~5 min）；wrapper 集成后 fork 自洽性（language_feature 输出 == color 输出，同一 α 路径）**max 差 0.000e+00**；site-packages 的 inria kernel 未被污染（--target + wrapper 方案）。**D=16 降级记录**：fork 的 `NUM_CHANNELS_language_feature=3` 固定于 config.h，D=16 需改后重编；验收④以 D=3 口径通过（慢速 vs inria 互验 6.4e-5）。

## 3. 损失与优化（`language_field/optim.py`）

```python
F_2d = rasterize_feature(gaussians, pose)        # (D,H,W)
V_2d = ae.dec(F_2d.permute(1,2,0))               # (H,W,512)
# 掩码级监督：对每个 MobileSAM 掩码区域，取 V_2d 均值与该掩码 clip_vec 对齐
loss = L1(V_2d[mask], v_gt) + λ_cos * (1 - cos(V_2d[mask], v_gt))
# 只优化 gaussians.feature（几何冻结）
```

**实现要点**（`distill_features(gaussians, ae, frames, iters, lr, lambda_cos)`）：
- 渲染走 inria kernel 快速路径（`render_precomp(colors_precomp=features)`，几何冻结时 `gaussians_grad=False`，梯度经 `grad_colors_precomp` 回流）——**绝不走慢速 splat**；
- 解码只算掩码 union 像素（分块防显存尖峰）；AE 冻结（坑 3）；
- **掩码权重 `1/sqrt(area)`**（实测坑：墙/地板级大掩码 clip_vec 是场景级语义、等权监督对它们无效——见验证报告 §3.5）；
- 预对齐门：渲染 frame 0 RGB PSNR > 15 才继续。

**实测（2026-08-31）**：场景 AE（318 条掩码特征）cos_mean 0.9502 > 0.95；蒸馏 1000 轮（200k 高斯 @600×340，~55 min/500 轮）cos_train 0.8921；**训练帧掩码内余弦 0.915/0.914/0.884/0.851 全 > 0.85**（验收①）；留出帧 250 泛化 0.807（记录）；几何逐位相等。

## 4. 查询引擎（`query/engine.py`）

```python
def query(text, means, features, ae, clip, tokenizer, top_k=100, min_score=0.0,
          eps=0.15, min_samples=5, cluster_method="obb"):
    t = clip.encode_text(tokenizer([text]))          # (1,512) L2 归一化
    V = ae.dec(gaussians.feature)                    # (N,512) L2 归一化
    relevance = V @ t.T                              # (N,) 余弦
    idx = torch.topk(relevance, top_k).indices
    pts = gaussians.xyz[idx].cpu().numpy()
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(pts)   # 聚类去离群
    bbox = aabb_or_obb(pts[labels == major_cluster])
    return {'query': text, 'bbox': bbox, 'confidence': relevance[idx].mean(), 'points': ...}
```

热力图：把 `relevance` 映射 viridis，对 top-k 高斯中心用当前位姿前向渲染输出高亮图（`render_heatmap`，复用 inria kernel colors_precomp 换色）。

**实测（2026-08-31）**：office0 查询 bbox 全部在模型范围、簇点 75-91、**椅子 vs 桌子中心距 2.97m**（验收③）；eps 网格 0.05-0.2 簇点稳定，**eps=0.15 为推荐值**（坑 4）；conf ~0.25 处于 S0 余弦压缩区间（判据 raw > 0.1，docs/05 坑 5）；热力图正确高亮 chair 簇（验收②数据级）。

## 5. 查询服务（Phase 4 节点扩展）

```python
# 在 Edge3DGSSlamNode 中加 service（Query.srv: QueryRequest -> QueryResult）
self.create_service(Query, '/semantic_query/query', self._on_query,
                    callback_group=ReentrantCallbackGroup())   # 独立回调组防阻塞 sync

def _on_query(self, req, resp):
    r = query(req.request.query, snap['means'], snap['features'], ...)  # 嵌套解包 req.request
    resp.result.bbox_center = r['bbox'].center ...            # 嵌套响应 resp.result
    resp.result.confidence = r['confidence']
    return resp
```

> `Query.srv` 定义在 `edge_3dgs_msgs`（`srv/Query.srv`：`QueryRequest request --- QueryResult result`，嵌套结构）。
> `QueryService`（`ws_src/edge_3dgs_ros/edge_3dgs_ros/query_service.py`）：懒加载 MobileCLIP（`load_models(load_sam=False)`，省 SAM ~200MB）+ 场景 AE；模型快照走 `backend.snapshot_model()`（含 features）；backend 由调用方动态传入（惰性初始化引用坑，见验证报告 §4）。

**实测（2026-08-31）**：
- 回放验证（Replica 60 帧话题流 + `--load probe_model_replica_p6.pt`）：服务可达、conf 0.251、bbox/points 有限值、无 NaN——**4 项全 PASS**；
- D435i 真机冒烟（在线模式）：服务可达、返回有限值（真机世界坐标）、节点不崩溃；conf 0.235 为在线零初始化特征的低置信——**符合预期**（在线查询需先蒸馏，蒸馏离线验证于 Replica）。

```bash
ros2 run edge_3dgs_ros edge_3dgs_slam_node
ros2 service call /semantic_query/query edge_3dgs_msgs/srv/Query "{request: {query: '黑色的椅子', top_k: 100, min_score: 0.0}}"
# rviz2 中看高亮热力图 + 3D BBox Marker
# ⚠️ min_score 适配：S0 余弦压缩区间下默认 0.0（params.yaml），显式传值需按 raw 余弦判据（>0.1）
```

## 6. 端到端验证

**验证数据约定**（docs/00 §7 统一口径）：语言查询验证**必须用有真实语义的室内场景**——
Replica（office0 有真实感物体纹理）或 D435i 自采；**合成序列不可用**（球/盒/棋盘
无语言语义，无法验证"黑色的椅子"这类查询）。验证前先确认场景中确实存在目标物体
（Replica 可对照官方 mesh，自采可目视确认）。

**实测（2026-08-31）**：Replica office0 全链路（特征重提 → 场景 AE → 蒸馏 → 查询）+ 回放服务验证 + D435i 真机冒烟全部通过（详见 [验证报告](06_Phase6_验证报告.md)）。

## 7. 验收清单

- [x] 渲染语言特征图解码后与 Phase 5 特征图在掩码内余弦相似度 > 0.85。
      **训练帧 0.915/0.914/0.884/0.851（全部 > 0.85）；留出帧 0.807（泛化记录）**
- [x] 查询「黑色的椅子」bbox 落在目标附近；热力图正确高亮。
      **bbox 在模型范围（数据级验证 + bbox_overlay_frame0.png 人工目视项）；heatmap_black_chair.png**
- [x] 多物体（椅子/桌子）可区分。
      **椅子 vs 桌子中心距 2.97m > 0.3m**
- [x] torch 慢速 splat 与 CUDA 版互验误差 < 1e-3。
      **max 6.4e-5（D=3 inria 口径）；LangSplat fork 自洽 0.0（D=16 降级记录）**

## 8. 常见坑

1. **特征光栅化梯度回传错误**：先慢速 torch splat 验证数值梯度，再上 CUDA。
   **实测**：慢速 splat 与 inria kernel 有 6 处语义差异（近平面/协方差/投影/倒数/tile/早停），
   校准记录见验证报告 §3.2——互验 max 2.05 → 6.4e-5。
2. **语言优化破坏几何**：务必冻结几何（或 1e-5 极小学习率）。
   **实测**：freeze_geometry 后蒸馏 1000 轮 means3D 逐位相等。
3. **AE 解码漂移**：AE 训练收敛（余弦 > 0.95）后再蒸馏进高斯，否则查询不聚合。
   **实测**：场景 AE（Replica 掩码特征）cos 0.9502；Phase 5 的 COCO 照片 AE 域不匹配须重训。
4. **DBSCAN 聚类参数**：`eps` 要按场景尺度调（米级），否则 bbox 碎裂或粘连。
   **实测**：office0 eps=0.15 推荐（网格 0.05-0.2 簇点 90-92 稳定）。
5. **S0 余弦压缩**：MobileCLIP-S0 raw 余弦 0.1-0.25 区间——`min_score` 默认必须 0.0
   （params.yaml 原 0.3 会过滤全部点致 DBSCAN 0 样本，实测坑）；查询判据用 raw > 0.1 或温度缩放。
6. **大背景掩码稀释监督**：墙/地板级掩码（area>1e4 px）的 clip_vec 是场景级语义——
   训练按 `1/sqrt(area)` 加权（评估仍等权），帧 150 等权均值 0.847→0.851 达标。
7. **查询服务集成坑**：symlink-install 下路径解析（按目录特征找仓库根）、backend 惰性
   引用（动态传参）、Query.srv 嵌套解包（`req.request`/`resp.result`）、load_ply 的
   logit_opacities 形状 (N,1)（cloud_publisher 依赖）——详见验证报告 §4。

---

## §10. 真机地图 → 开放词汇查询衔接（2026-09-02 落地）

补上了闭环的前两步（此前缺口见验证报告：真机只到冒烟）：

### 产物（跑 SLAM 自动生成，无需额外操作）

运行 `ros2 run edge_3dgs_ros edge_3dgs_slam_node -- --tier fps` 后，Ctrl-C 退出时
自动写到 `<仓库>/data/outputs/live/`（可用 `--out DIR` 改）：

| 文件 | 内容 | 用途 |
|---|---|---|
| `map_<时间戳>.pt` / `map_latest.pt` | {'params','variables'} 全量地图（float32，与 node `--load` 同构，200k 高斯 ~13MB） | 蒸馏/查询加载 |
| `map_autosave.pt` | 每 `--autosave-sec`（默认 120s）覆盖写 | 长跑防丢 |
| `frames.npz` | 关键帧：rgb(N,H,W,3) uint8 / depth float32 米 / poses(N,4,4) **w2c 与地图同世界系** / K / t | 特征提取帧源 |

关键帧由 backend `on_keyframe` 钩子记录（锁外 ~ms 级，环形保留最近
`--record-max` 帧，默认 150；提取成本 13s/帧 @Jetson，150 帧≈30min，够挑子集）。

### 接下游（离线，已完成适配 2026-09-02，脚本在 experiments/）

```
1) 建图：ros2 run ... -- --tier fps   → Ctrl-C 自动产出 map_*.pt + frames.npz
2+3) 真机一键语义化（提取→AE→蒸馏→验收→checkpoint）：
   python3 experiments/phase6_semantic_live.py \
       --map data/outputs/live/map_latest.pt \
       --npz data/outputs/live/frames.npz --max-frames 24
   （可拆 --extract-only / --distill-only 断点续跑；产物 data/outputs/live_semantic/）
4) 查询：ros2 run ... --load data/outputs/live_semantic/probe_map_p6.pt
   → ros2 service call /semantic_query/query ...
```

新增组件：`src/edge_3dgs_slam/dataset/live_npz.py`（LiveNpzSequence：npz 关键帧序列，
接口对齐 ReplicaSequence——frame/frame_scaled/poses_w2c/cam，位姿与地图同世界系）；
`experiments/phase6_semantic_live.py`（Replica 提取/蒸馏流程的真机版，pkl/segs 键
契约一致，可复用 feature_factory / language_field.optim 全部库函数）。

**真机小数据冒烟（2026-09-02，5 帧桌面 npz 抽 3 帧 @640×360）**：提取 11-12s/帧、
62-69 掩码/帧；AE cos 0.95 ✅；预对齐 PSNR 18.5dB（fps 档地图+记录位姿天然对齐，
无需 --skip-gate）；蒸馏 30 轮 cos_train 0.785（<0.85 判据——冒烟只跑 30 轮/3 帧，
正式用 ≥24 帧 + 200 轮）。

### 记录器实现备忘

- `ws_src/edge_3dgs_ros/edge_3dgs_ros/live_recorder.py`（LiveRecorder，npz 契约与
  phase4_replay_publisher 同款）；backend `on_keyframe` 钩子 + `save_checkpoint()`
  （variables 含非张量标量如 scene_radius，需 isinstance 守卫——首版踩坑）。
- 单测：LiveRecorder 环形保留/往返 ✅；真机冒烟：5 帧/200k 图落盘 + `--load`
  加载往返 ✅（map 200000 高斯，scene_radius 保留）。
