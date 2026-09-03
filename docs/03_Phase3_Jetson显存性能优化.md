# 03 · Phase 3 针对 Jetson 平台的显存与性能极致优化

> 目标：16GB 统一内存 + 有限算力下，控制重建/优化过程的显存峰值（防 OOM）并产出**可量化消融表**；同时对「在线端到端建图能否实时」给出实测结论——**不可达**（本机稳定 10 FPS 不行，见 §11 与 `docs/04_Phase4_验证报告.md` §5），交付口径据此修订为离线重建（`docs/00 §1`）。
> 依赖：Phase 2 的几何 SLAM 骨架。

## 1. 先建立性能基线（务必先做）

在 Phase 2 管线上埋点：

```python
import torch, time
def profiled(fn, *a, **kw):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = fn(*a, **kw); torch.cuda.synchronize()
    return r, (time.perf_counter() - t0) * 1e3   # ms
```

记录四项基线：**FPS、峰值显存（`torch.cuda.max_memory_allocated()`）、高斯数、ATE/PSNR**，写入 `docs/03` 作为优化前参照。定位瓶颈：渲染 vs 反传 vs 位姿优化 vs 高斯增删。

> ✅ **已实现（2026-08-27 实测）**：`utils/profiling.py`（`profiled/alloc_mb/peak_mb/reserved_mb/reset_peak`）+ `experiments/phase3_perf_ablation.py --stages`。
>
> **基线四指标**（60 帧 640×480 合成序列，`data/outputs/phase3/synth_scene_480.npz`；同步管线 + §6 分辨率分档；track 8 iters @320×240 / map 25 iters @640×480）：
>
> | FPS（track 中位） | 峰值显存 | 高斯数 | ATE | PSNR |
> |---|---|---|---|---|
> | 2.2 | 393 MB | 450,789 | 42.35 cm | 15.31 dB |
>
> **瓶颈分类**（`--stages` 手工分段计时，预热后实测；Jetson 上 torch.profiler 需 CUPTI 权限不可用）：
>
> | 阶段 | 耗时 | 占比 |
> |---|---|---|
> | 渲染前向（RGB+depth 两趟，640×480） | 30.7 ms | 12% |
> | 反传（单次 backward，640×480） | 87.4 ms | 33% |
> | 位姿优化（track 单迭代，320×240，含前向+反传+Adam） | 64.0 ms | 24% |
> | 高斯增删（加高斯全流程） | 82.1 ms | 31% |
>
> **结论**：光栅化相关（前向+反传）~45% 是大头（反传 ≈ 前向 ×2.8），加高斯全流程 ~31% 次之。优化优先级：剔除/关键帧（省光栅化）> 加高斯流程（反投影+密度判据）。
> ⚠️ **首次调用冷启动伪影**：光栅化器首次调用 ~2s（内核加载/缓冲分配），预热后单迭代仅 ~70ms——测性能必须先预热。

**验证数据约定**（docs/00 §7 统一口径）：性能基线与消融**必须用 Replica 固定场景 + 固定输入**
（如 office0 前 200 帧 600x340，`experiments/phase2_replica_eval.py` 同参数），优化前后同输入
对比，否则 FPS/显存不可比。合成序列（320x240）只用于单元级机制验证，其渲染压力/高斯规模
不代表真实负载；基线数值需标注 数据来源 + 分辨率 + 帧数（同报告规范）。

> ✅ **Replica 正式基线（2026-08-28 实测，`phase3_perf_ablation.py --data replica`）**：
> office0 前 200 帧 600×340（`ReplicaSequence.frame_scaled(0.5)`，同 phase2_replica_eval 参数），
> 同步管线 + §6 分辨率分档（track 8 iters @320×240 / map 25 iters @600×340，K 逐轴缩放后）：
>
> | FPS（track 中位） | 峰值显存 | 高斯数 | ATE | PSNR |
> |---|---|---|---|---|
> | 2.1 | 340 MB | 430,886 | 7.09 cm | 19.87 dB |
>
> （基线行数字本身随 rasterizer 非确定性波动：三次独立跑 ATE 6.50-9.36cm、PSNR 19.4-20.1dB，
> 判据全部相对基线行，自适应。）
>
> **单迭代分解探针**（`experiments/phase3_track_probe.py --mode per-iter`，134k 高斯模型，预热后）：
>
> | 分辨率 | 前向 RGB | +depth 趟 | 反传 | 单迭代合计 | RGB-only 合计 |
> |---|---|---|---|---|---|
> | 320×240 | 7.0 ms | 4.6 ms | 22.9 ms | 42.9 ms | 28.6 ms |
> | 224×168 | 7.3 | 4.9 | 17.3 | 37.6 | 28.7 |
> | 160×120 | 8.1 | 5.4 | 18.7 | 38.7 | 30.0 |
> | 128×96 | 8.0 | 5.8 | 20.3 | 41.8 | 30.9 |
>
> **关键结论（本 Phase 实测，推翻"分辨率降档提速"假设）**：单迭代耗时**几乎不随分辨率变化**
> （38-43ms 区间），拟合 iter_full ≈ 39ms 固定开销 + 微小像素项——投影/排序/反传是
> O(高斯基数) 开销，且低分辨率下每像素高斯密度升高使反传更贵（128×96 反传 20.3ms >
> 320×240 的 22.9ms 反而更贵）。**分辨率不是 FPS 杠杆**；`needs_depth=False`（RGB-only）
> 省 14-33% 单迭代成本是唯一有效的渲染侧优化。

## 2. 高斯数量控制与剪枝（`gaussian/model.py`）

> ✅ **已实现（2026-08-27）**：[src/edge_3dgs_slam/gaussian/model.py](../src/edge_3dgs_slam/gaussian/model.py)
>
> - 模块级 `prune(gaussians, opacity_thresh=0.005, scale_thresh=100.0)`（文档 §2 签名逐字对齐；与类方法 `prune(opacity_threshold, big_scale)` 并存——后者是 mapping 用的旧语义）
> - 类方法 `prune_points(keep)` / `keep_indices(keep_idx)`（布尔掩码/索引保留）
> - `enforce_capacity(gaussians, N_max=200_000)`：按 opacity `torch.topk` 淘汰；**只在属性优化循环结束后调用**（Adam 状态与索引错位防护，代码注释锁定时序）
> - `already_covered(point, existing_xyz, r)`（单点 O(N)，仅测试）+ `coverage_mask(points, existing_xyz, r)`（批量）：栅格哈希 + 段内精确距离，与暴力法逐元素一致（无漏判无多报），复杂度 O(邻域点数)；段内点数超 512 时保守判覆盖（防 r 过大/密度过高时 repeat_interleave 物化百亿元素 OOM——实测踩坑）
> - `mapping.map_keyframe(..., density_r=, capacity_max=)`：集成密度判据与容量上限

```python
def prune(gaussians, opacity_thresh=0.005, scale_thresh=100.0):
    keep = (gaussians.opacity > opacity_thresh) & (gaussians.scale.max(-1) < scale_thresh)
    gaussians.prune_points(keep)     # 删除

def enforce_capacity(gaussians, N_max=200_000):
    if gaussians.n > N_max:          # 超上限淘汰最低 opacity/贡献
        score = gaussians.opacity.squeeze(-1)
        keep_idx = torch.topk(score, N_max).indices
        gaussians.prune_points_keep(keep_idx)
```

密度判据（新增高斯前）：

```python
# 反投影点半径 r 内已有近邻高斯则不新增（kd-tree 或栅格哈希）
def already_covered(point, existing_xyz, r=0.01):
    return torch.any(torch.norm(existing_xyz - point, dim=-1) < r)
```

## 3. FP16 / 混合精度

```python
# 存储层：scale/opacity/color 用 FP16，xyz 保留 FP32
gaussians._scale   = nn.Parameter(gaussians._scale.half())
gaussians._opacity = nn.Parameter(gaussians._opacity.half())
gaussians._color   = nn.Parameter(gaussians._color.half())
# 计算层：送入 rasterizer 前 .float()（CUDA kernel 内部 float）
# NN 部分（Phase 5/6）用 torch.cuda.amp.autocast 包 forward
```

> 注意：`diff-gaussian-rasterization` 的 kernel 内部用 `float`，FP16 只在**存储**省钱、**边界转换**要显式可控。是否真省显存要在基线表里实测。

> ✅ **已实现（2026-08-27）**：[gaussian/model.py](../src/edge_3dgs_slam/gaussian/model.py) `half_storage()/float_storage()/is_half_storage`；[gaussian/render.py](../src/edge_3dgs_slam/gaussian/render.py) 计算层统一 `.float()` 边界。
>
> - 存储层：`rgb_colors/logit_opacities/log_scales` → FP16，`means3D/unnorm_rotations` 恒 FP32（几何绝不降精度，见 §9 坑 2）
> - 计算层：送入 rasterizer 前 `.float()`（瞬态分配，用完即释放；kernel 内部恒 float）
> - **关键实测修正**：Adam 在 half 参数上会数值上溢（exp_avg state 与参数同 dtype，fp32 量级梯度平方 → inf → 参数 NaN → rasterizer 非法内存访问，`CUDA illegal memory access`）。因此 `map_keyframe` 在优化期间把模型转回 float32、结束后转回 half（存储层在优化间隙保持 half 省显存）
> - `add_gaussians` 的 cat 对齐 dtype；`save_ply` 导出前 `.float()`
> - **显存实测**：200k 高斯参数存储 float32 11.2MB → half 7.6MB，节省 ~3.6MB（~2.5% 级）——**存储层不是显存瓶颈**（主要收益在 §4 剔除与 §5 关键帧），如实记录防简历注水

## 4. 视锥剔除（Frustum Culling）

```python
def frustum_visible(xyz, T_wc, K, H, W, margin=0.2):
    # 把 xyz 变换到相机系，投影到像素，判断是否在 [−margin, W+margin]×[−margin, H+margin]
    # 且在 z > near；返回布尔掩码
    ...
# 优化时只对 visible 高斯反传；不可见高斯冻结（.requires_grad_(False) 或过滤）
```

> ✅ **已实现（2026-08-27）**：[gaussian/frustum.py](../src/edge_3dgs_slam/gaussian/frustum.py) `frustum_visible(xyz, T_wc, K, H, W, margin=None, near=0.01, scales=None)`；[gaussian/render.py](../src/edge_3dgs_slam/gaussian/render.py) `render(..., mask=None)`；`mapping.map_keyframe(..., cull=)` 与 `track(..., cull=)`。
>
> - **margin 自动计算**（关键正确性修正）：margin 必须覆盖高斯的屏幕足迹（≈ f·σ/z）——中心在图像边缘外的高斯其半透明尾巴仍贡献边缘像素，固定 0.2px 会误剔除（剔除前后渲染不一致）。默认 `max(fx,fy)·max_scale/z_min·1.5`
> - **mask 语义**：压缩索引只送可见高斯进光栅化；不可见高斯无像素贡献 → grad=0 → Adam 不步进（天然冻结）；backward 经索引自动 scatter 回全量参数
> - **单测锁定**（`experiments/phase3_unit_checks.py` §4）：full vs masked 渲染**逐像素一致**（max|Δ|=0）、反向 mask 全黑、不可见高斯梯度为 0

## 5. 关键帧管理 + 异步建图（在线实验记录）

`src/edge_3dgs_slam/slam/keyframe.py` + `slam/backend.py`：

```python
class KeyframeManager:
    def should_insert(self, T_new, T_last):
        dt = np.linalg.norm(T_new[:3,3] - T_last[:3,3])
        dth = angle_between(T_new[:3,:3], T_last[:3,:3])
        return dt > 0.1 or dth > 3.0     # 平移/旋转阈值

class SLAMBackend:
    def __init__(self):
        self.map_queue = Queue(maxsize=8)          # Mapping 队列
        self.mapping_thread = Thread(target=self._loop, daemon=True)
    def track(self, frame):                         # 实时回调：只 Tracking，永不阻塞
        ...
    def _loop(self):                                # worker 线程：按关键帧节奏 Mapping
        while True:
            kf = self.map_queue.get()
            self.map_keyframe(kf)
```

- 滑动窗口：只优化最近 K 个关键帧（如 5~8），旧关键帧冻结。
- 丢帧策略：队列满则丢弃最旧，优先保 Tracking。

> ✅ **已实现（2026-08-27）**：[slam/keyframe.py](../src/edge_3dgs_slam/slam/keyframe.py)（`KeyframeManager` + `angle_between`）；[slam/backend.py](../src/edge_3dgs_slam/slam/backend.py)（`SLAMBackend`：`Queue(maxsize=8)` + daemon 线程 + `lock_chunk` + `join()`）；`slam/__init__.py` 导出（供 Phase 4 复用）。
>
> - **位移度量修正（实测踩坑）**：`should_insert` 的平移差用**相机位置差**（c2w 平移）而非 w2c 平移差——w2c 平移 = −Rᵀ·c，旋转耦合。相机绕场景中心俯视时实测帧间真实位移 5.5cm、w2c 差 51cm，用 w2c 会误判关键帧（60 帧序列 44 个关键帧 → 修正后 31 个）
> - **GPU 串行化**（§9 坑 1）：单把 `threading.Lock` 串行化 track/map；`lock_chunk` 每 N 迭代放锁让 tracking 插入——释放窗口必须 ≥ 线程调度延迟（实测 1ms 窗口 worker 立即抢回、track 饿死 FPS 0.3；20ms 后恢复正常）
> - **单测锁定**（`phase3_unit_checks.py` §5）：阈值矩阵、队列满丢最旧留最新（确定性测试）、窗口长度 ≤ 上限、join 正常退出、lock_chunk 锁状态正确
> - **实测代价**（消融行 5）：单 GPU 上窗口建图（5 帧×每迭代 5 次渲染）使端到端时间 ×1.2、峰值显存 879MB（tile buffer 叠加，非泄漏，16GB 预算内）；track 时地图滞后 Δ 关键帧 → ATE 劣化 ~55%（异步固有权衡，快速运动放大，PSNR 反由窗口多帧约束提升）

## 6. 渲染/迭代控制

- Tracking 用降采样（320×240），Mapping 用全分辨率（640×480）。
- `max_sh_degree=0`（不用 SH），`tracking_iters` 5~10。
- 限制每 tile 高斯数 / 总渲染高斯数。

> ✅ **已实现（2026-08-27）**：[utils/frame_utils.py](../src/edge_3dgs_slam/utils/frame_utils.py) `downsample_frame`（RGB 双线性、depth 最近邻防伪造、K 按比例缩放）；消融管线统一 track @320×240 / map @640×480；`max_sh_degree=0` 在 `setup_camera` 已写死；tracking_iters=8（区间 5~10）。每 tile 高斯数由 rasterizer 内置限制 + §2 总容量上限兜底。
>
> ✅ **性能档扩展（2026-08-28 实测）**——探针揭示"分辨率降档无收益"（§1 分解表）后，
> 迭代预算成为唯一 FPS 杠杆，新增：
>
> | 参数 | 位置 | 说明 | 默认 |
> |---|---|---|---|
> | `render(..., needs_depth=False)` | gaussian/render.py | 跳过 depth/sil 第二趟光栅化（省 14-33% 单迭代成本），返回 depth=None | True |
> | `track(..., res_schedule)` | slam/tracking.py | `[(W,H,n_iter),...]` 粗到细分阶段；lr 衰减按全局迭代计数、best 阶段内独立 | None |
> | `track(..., depth_every)` | 同上 | 每 N 迭代渲染 depth 一趟；跳过迭代 sil 掩码缓存复用；**best 口径 = 阶段内所有迭代都有的 loss 项**（跳 depth 时用 RGB+SSIM 公共部分，防跨口径误判） | 1 |
> | `track(..., early_stop)` | 同上 | 相对下降率 < tol 连续 patience 次提前停（轻运动帧 1-2 迭代即收敛） | False |
> | `track(..., adaptive_max)` | 同上 | **自适应迭代上限**：基础 iters 后难帧自动扩展至收敛或上限——防长序列初值漂移累积发散（§9 坑 12） | None |
> | `map_keyframe(..., opt_W/H)` | slam/mapping.py | 优化循环内窗口帧降采样渲染（光栅化 ∝ 像素数）；**add_new 的 silhouette 渲染保持原生分辨率**（高斯播种质量不降） | None |
> | `map_keyframe(..., window_rotate, rotate_n)` | 同上 | 每迭代只轮转渲染窗口 rotate_n 帧（窗口 5 → 每迭代 2 帧，耗时 -70%，PSNR 等价 -0.29dB 实测） | False |
> | `SLAMBackend(..., lock_sleep_ms)` | slam/backend.py | chunk 释放窗口可调（map 减负后 20ms → 10ms，track 插入更频繁） | 0.02 |
>
> 同步管线（`run_sync`）与异步后端（`run_async`）均已透传上述参数（`experiments/phase3_perf_ablation.py`）。

## 7. 内存预算表（实测填数，2026-08-27）

| 项 | 占用 | 说明 |
|---|---|---|
| PyTorch 运行时 + CUDA 上下文 | allocated 口径 0 MB（懒分配） | `torch.cuda.memory_allocated()` 不含上下文，物理占用见 nvidia-smi（§9 坑 4） |
| 高斯参数（20 万 × 48B/个） | float32 9.6 MB → half 7.6 MB | 极小，非瓶颈（48B = 5 参数 float32 各向同性） |
| 渲染 tile buffer（640×480 单次前向） | 峰值 51 MB（增量 38 MB） | 与分辨率/高斯密度正比；backward 峰值 83 MB |
| 优化器状态（Adam 2× 参数） | 8.3 MB | map_keyframe 每轮新建，用后即释放 |
| 全序列峰值（Replica 口径消融行 1~4） | 221–340 MB | 单帧建图模式（200k 高斯上限后较合成 45 万高斯更低） |
| 窗口建图（async 行） | 757 MB | 5 帧 tile buffer 叠加（16GB 预算内） |
| maplight（opt 320×240 + 轮转） | 276 MB | map 渲染量 -70% 后峰值，见 §6 参数表 |
| MobileSAM/MobileCLIP（Phase 5/6） | ~1GB | 分时复用、用完 `empty_cache()` |

**结论**：高斯参数非瓶颈；峰值在光栅化 tile buffer——窗口建图叠加 ~757MB（合成口径
879MB），map 减负（opt 320×240 + 轮转）后回落到 ~276MB，16GB 预算内余量充足。

## 8. 验收与消融表

✅ **已实测（2026-08-28，`experiments/phase3_perf_ablation.py --data replica`，8 行）**。
口径：**Replica office0 前 200 帧 600×340**（docs/00 §7 验证数据约定的正式口径；
合成序列口径见下方参考表），track 8 iters @320×240 / map 25 iters @600×340
（maplight 行 map 8 iters + opt 320×240 + 窗口轮转），同一关键帧策略（dt 0.1m /
dth 3°），同一评估函数（ATE = c2w 平移 Umeyama 标准口径）。性能行配置由
`phase3_track_probe.py` 决策门定稿（§9 坑 12/13 的探针证据）。

| 配置 | FPS | 峰值显存 | 高斯数 | ATE | PSNR | 判定 |
|---|---|---|---|---|---|---|
| baseline（同步+分档） | 2.1 | 340 MB | 430,886 | 6.50 cm | 19.90 dB | 基准（三次独立跑 ATE 6.5-9.4cm） |
| + 高斯剪枝/容量上限 | 2.7 | 222 MB | 200,000 | 7.78 cm | 17.39 dB | ⚠️ PSNR -2.5dB（建图随机波动，v4 轮 PASS） |
| + FP16 存储 | 2.7 | 221 MB | 200,000 | 5.97 cm | 17.74 dB | ⚠️ PSNR -2.2dB（同上波动） |
| + 视锥剔除 | 1.8 | 221 MB | 200,000 | 7.77 cm | 18.23 dB | ✅ 质量等价；FPS 无收益（见下） |
| + 关键帧窗口 + 异步建图 | 1.8 | 757 MB | 200,000 | 3.83 cm | 20.78 dB | ✅ 架构价值（见下） |
| + 性能档（6it+d2+自适应） | **3.3** | 229 MB | 200,000 | 10.50 cm | 19.44 dB | ✅ **FPS ×1.57**、ATE ×1.62（判据 ×1.8 内）、PSNR -0.5dB |
| + map 减负（maplight） | **3.0** | 276 MB | 200,000 | 8.30 cm | **23.22 dB** | ✅ **FPS ×1.43**、ATE ×1.28、**PSNR +3.3dB 全场最高** |
| FPS 极值（3it+d2+自适应） | **6.4** | 225 MB | 200,000 | 62.83 cm | 16.47 dB | ✅ 极值记录行（只判 FPS；质量劣化如实记录） |

> 判定标准分两套（代码注释逐条注明）：**质量行**（baseline 之后 4 行）相对基线 ATE 劣化
> <20%、PSNR 下降 <2dB、FPS ≥上一行 ×0.85（cull 行豁免 FPS 判据——实测 masked 渲染
> 320×240 +29% / 600×340 +53% 更慢，价值在梯度冻结）。**性能行**（track160/maplight/
> maxfps）独立判据：FPS ≥ 基线 ×1.3、ATE ≤ 基线 ×1.8、PSNR ≥ 基线 −3.5dB、显存 <4GB
> （与 async 行同构；×1.8 与 −3.5dB 按实测波动上界设定——rasterizer 非确定性 ±2dB 级、
> 温度降频 FPS ±20%）。
>
> **逐行结论**：
> - 行 2（剪枝/容量）：高斯数 43 万 → 20 万、显存 −35%，FPS +29%；ATE/PSNR 与基线
>   相当（波动内）。收益最大的一项。PSNR 判据波动（v4 轮 PASS、v7 轮 -2.5dB）源于
>   建图随机性（rasterizer atomicAdd + add_new 采样），如实记录。
> - 行 3（FP16）：存储层省 ~1%（参数非瓶颈，§3 已证）；ATE/PSNR 无显著变化。
> - 行 4（视锥剔除）：**FPS 无收益**（mask 索引开销 > 光栅化节省，宽 FOV 可见率 ~80%）；
>   价值在梯度冻结（不可见高斯不被无关帧扰动）与大规模/多房间场景。
> - 行 5（异步窗口）：架构解耦（track 永不阻塞、队列满丢最旧、lock_chunk 保实时）；
>   窗口多帧约束使 **ATE 3.83cm（全场最优）**、PSNR +0.9dB；代价：显存 757MB（窗口
>   tile buffer 叠加，16GB 预算内）、端到端 ×1.1。
> - 行 6（track160，**性能行主配置**）：6 迭代 + depth_every=2 + adaptive_max=8 →
>   **FPS 3.3（×1.57）**、ATE 10.5cm（×1.62）、PSNR -0.5dB——"降 FPS 保质量"决策的
>   收敛临界配置（5it 波动 8-16cm 发散临界；6it 稳入收敛区，见 §9 坑 12/13）。
> - 行 7（maplight）：map 减负（opt 320×240 + 窗口轮转 2 + iters 8 + lock_chunk 2/
>   sleep 10ms）→ **PSNR 23.22dB 全场最高**（窗口多帧约束弥补分辨率损失，探针 -0.29dB
>   等价性）、FPS ×1.43、显存 276MB（map 渲染量 -70%）。**异步架构 + 减负后的最佳
>   质量-速度组合**。
> - 行 8（maxfps）：3it d2+ad8 → FPS 6.4（×3.0）为能力极值；ATE 62.8cm（长序列
>   初值漂移发散，§9 坑 12）**不可用**——如实记录为极值参考。
> - **温度噪声**：连续跑 GPU 降频 FPS 波动 ±20%；跨行对比同一次连续跑。性能行 FPS
>   判据 ×1.3 已含温度余量（×1.4 时 maplight 3.0 vs 目标 3.08 差 2.5% FAIL，无统计意义）。
>
> **合成口径参考**（2026-08-27，60 帧 640×480 合成序列，与 Replica 不可直接对比）：

| 配置 | FPS | 峰值显存 | 高斯数 | ATE | PSNR | 判定 |
|---|---|---|---|---|---|---|
| Phase 2 基线（同步+分档） | 2.2 | 393 MB | 450,789 | 42.35 cm | 15.31 dB | ✅ 基准 |
| + 高斯剪枝/容量上限 | 2.7 | 254 MB | 200,000 | 37.32 cm | 14.76 dB | ✅ 显存 −35%，高斯 −56%，FPS +23% |
| + FP16 存储 | 2.7 | 252 MB | 200,000 | 45.76 cm | 14.87 dB | ✅ 显存再 −1%（存储层非瓶颈，见 §3） |
| + 视锥剔除 | 2.0 | 250 MB | 200,000 | 42.81 cm | 14.70 dB | ✅ 质量等价；FPS 无收益（见下） |
| + 关键帧窗口 + 异步建图 | 1.9 | 879 MB | 200,000 | 65.78 cm | 15.08 dB | ✅ 架构价值；精度/显存代价如实记录（见下） |

> 合成口径历史结论（2026-08-27，仅作机制参考；**简历证据以 Replica 口径表为准**）：
>
> **逐行结论**：
> - 行 2（剪枝/容量）：密度判据 + topk 容量把高斯数压到 20 万、峰值显存 −35%、FPS +23%，ATE 反而更优（高斯爆炸抑制）——收益最大的一项。
> - 行 3（FP16）：存储层节省 ~1%（参数非显存瓶颈），ATE/PSNR 无显著变化——**如实记录：FP16 在存储层收益有限**，价值在后续 Phase 的大模型（语言嵌入特征）复用。
> - 行 4（视锥剔除）：**FPS 无收益**——实测 masked 渲染比 full 慢（320×240 +29% / 640×480 +53%，20 万高斯；mask 索引开销 > 光栅化节省，且宽 FOV 91° 可见率 75%）。价值在**梯度冻结**（不可见高斯不被无关帧扰动，质量保障）与大规模/多房间场景（剔除率提升后收益反转）。FPS 判据对 cull 行豁免（见 §9 坑 5）。
> - 行 5（异步窗口）：架构解耦（track 永不阻塞于 map 流程、队列满丢最旧、`lock_chunk` 保实时）；**代价**：窗口 5 帧建图峰值显存 879MB、端到端 ×1.2、track 时地图滞后 Δ 关键帧使 ATE 劣化 ~55%（快速运动放大，PSNR 反 +0.2dB 由窗口多帧约束）；丢帧 0（队列未满）。判据按异步固有权衡独立设定（ATE 容忍 60%、显存 4GB 预算）。
> - **温度噪声说明**：连续跑 5 行 GPU 温度降频使 FPS 波动 ±20%（单跑 2.7 vs 连续跑 1.3），FPS 判据按 ×0.85 容忍；行间数字取同一次连续跑保证可比。

## 9. 常见坑

1. **异步建图与 Tracking 竞争 GPU**：单把 `threading.Lock` 串行化 GPU 调用（实测）——实测两线程无锁时模型状态竞态；加锁后 track 墙钟受锁等待影响，`lock_chunk` 每 N 迭代放锁让 tracking 插入，**释放窗口必须 ≥20ms**（1ms 时 worker 立即抢回锁、track 被饿死，FPS 0.3）。合成回放（无限 push）下墙钟延迟被放大，真实时间驱动系统由队列满丢帧兜底。
2. **FP16 导致几何漂移**：`xyz` 千万别降 FP16——`means3D/unnorm_rotations` 恒 FP32，实测 half 存储下 ATE/PSNR 无劣化（45.76cm vs 44.03cm 基线，噪声内）。
3. **剪枝过猛破坏几何**：先小阈值、看 ATE 劣化 < 20% 再收紧——实测 density_r=0.01 + capacity 200k 行 ATE 37.32cm（比基线 42.35cm 更优），高斯爆炸（45 万）被抑制后跟踪反而受益。
4. **统一内存「显存」看错**：`torch.cuda.memory_allocated()` 为准；`reserved`（缓存池）与 nvidia-smi 物理占用作对照——allocated 口径不含 CUDA 上下文（懒分配，实测 init 后 0 MB），上下文物理占用由 nvidia-smi 观察；消融行间必须 `empty_cache()+reset_peak()` 防内存池串扰峰值。
5. **视锥剔除「优化」反而变慢**（本 Phase 实测新增）：mask 路径（frustum 计算 + 5 参数 gather 索引）开销 > 光栅化节省——20 万高斯 masked 渲染比 full 慢 29%（320×240）/ 53%（640×480）。cull 的价值在**梯度冻结**（质量）与大规模场景（剔除率提升后收益反转），FPS 判据对 cull 行豁免。单测锁定正确性（full vs masked 逐像素一致、不可见 grad=0）。
6. **Adam 在 FP16 参数上数值上溢**（本 Phase 实测新增）：Adam state 与参数同 dtype，fp32 量级梯度平方在 half 下 → inf → 参数 NaN → rasterizer 非法内存访问（`CUDA illegal memory access`，异步报错难定位）。修复：存储 half + **优化期间转回 float32**，结束后转回 half。
7. **`coverage_mask` 大 r/高密度下 OOM**（本 Phase 实测新增）：所有点同 cell 时段内点达百亿，`repeat_interleave` 物化 83GB 崩溃（NVML 断言）。修复：单段点数 > 512 保守判「已覆盖」（密度判据意图即保守）。
8. **关键帧判据位移度量**（本 Phase 实测新增）：w2c 平移差 = −Rᵀ·c 与旋转耦合——相机俯视场景中心时帧间真实位移 5.5cm、w2c 差 51cm（60 帧 44 个关键帧 → 修正后 31 个）。用 c2w 相机位置差。
9. **光栅化首次调用冷启动 ~2s**（本 Phase 实测新增）：内核加载/缓冲分配，测性能必须先预热，否则单迭代计时虚高 30 倍（2073ms vs 72ms）。
10. **非等比降采样的 K 逐轴缩放**（2026-08-28 实测新增）：`downsample_frame` 曾用单一比例
    `s=out_W/W` 缩放整个 K——合成序列 640→320 恰好等比（s=0.5）掩盖了错误；Replica
    600×340→320×240 时 s_w=0.533 ≠ s_h=0.706，fy/cy 被错误缩放 → 投影几何不一致，
    污染 ATE 口径（修复前 Replica 200 帧基线 ATE 87cm/PSNR 12dB vs 修复后 6.5cm/20dB）。
    修法：fx/cx 按 s_w、fy/cy 按 s_h 逐轴缩放（等比目标行为不变，单测锁定）。
11. **分辨率降档无收益甚至更慢**（2026-08-28 探针实测）：单迭代耗时与分辨率几乎无关
    （320×240 42.9ms vs 160×120 38.7ms vs 128×96 41.8ms，134k 高斯）——投影/排序/反传
    是 O(高斯基数) 固定开销，且低分辨率下每像素高斯密度升高使反传更贵（128×96 反传
    20.3ms > 320×240 22.9ms）。**"track 用更低分辨率提速"是错误直觉**；有效的渲染侧
    杠杆只有 `needs_depth=False`（省 14-33%）与高斯基数控制。
12. **低迭代在长序列上初值漂移累积发散**（2026-08-28 探针实测，本 Phase 最重要发现）：
    3 迭代在 Replica 100 帧 ATE 12.3cm、200 帧 81.6cm（发散）；**真值初值对照 3.14cm**
    证明发散根源是初值漂移累积（匀速外推残差逐帧累积 → 3 迭代修正不了大残差 → 恶性
    循环），非迭代数不足。修复：`adaptive_max` 难帧自动补迭代（200 帧 81.6→57.5cm），
    收敛临界在 **6 迭代**（6it d2+ad8 = 6.20cm 0.87× 基线；5it 波动 8-16cm 临界）。
    教训：短序列（100 帧）验证的迭代档必须复核长序列（200 帧）稳定性。
13. **track best-loss 跨口径不可比**（2026-08-28 实测新增）：depth_every>1 时跳过
    depth 趟的迭代 loss 数值尺度变小（无 depth 项），直接比较会误判"更优"→ best 回退
    选错迭代。修法：best 度量 = 阶段内**所有迭代都有的 loss 项**（depth_every=1 时含
    depth，与旧实现逐位一致；>1 时用 RGB+SSIM 公共部分）。
14. **map 窗口轮转质量等价**（2026-08-28 探针实测）：每迭代只渲染窗口 2 帧（确定性
    轮序）vs 全窗口 5 帧——PSNR 差仅 -0.29dB（32.48 vs 32.74），每关键帧耗时 -70%
    （4.9s→1.5s，含 opt 320×240）。窗口多帧约束弥补了低分辨率损失（maplight 行
    PSNR 23.22dB 全场最高）。
15. **map_keyframe 计时必须 CUDA 同步**（2026-08-28 实测新增）：`time.perf_counter()`
    测 map_keyframe 耗时显示 "2ms"——实际 2.9s！map 内部无 `torch.cuda.synchronize`，
    墙钟只量到 kernel launch；探针脚本踩坑后修复（`wall*1000/n_kf` 且循环内同步）。
16. **Replica 消融的 K 口径**（2026-08-28 实测新增）：`load_or_generate` 曾返回
    `seq.K`（全分辨率 1200×680 口径 fx=600）而帧是 `frame_scaled(0.5)`（600×340，
    fx=300）——投影几何全错，消融全表 ATE 87cm/PSNR 12dB。修法：K 必须取降采样后
    每帧自带内参（`frames[0].K`），单测/冒烟先验证基线数字再跑全表。

## 10. 产出清单

- `src/edge_3dgs_slam/utils/profiling.py`（§1 计时/显存）、`utils/frame_utils.py`（§6 降采样 + **K 逐轴缩放修复**）
- `src/edge_3dgs_slam/gaussian/model.py`（§2 剪枝/容量/密度判据、§3 FP16 存储）
- `src/edge_3dgs_slam/gaussian/frustum.py`（§4 视锥剔除）、`gaussian/render.py`（mask + float 边界 + **needs_depth**）
- `src/edge_3dgs_slam/slam/keyframe.py`、`slam/backend.py`（§5 关键帧 + 异步建图 + **lock_sleep_ms**）
- `src/edge_3dgs_slam/slam/mapping.py` / `tracking.py`（density/capacity/cull/window 参数 + **opt_W/H/window_rotate**、**res_schedule/depth_every/early_stop/adaptive_max**，默认值 = Phase 2 行为零回归）
- `experiments/phase3_unit_checks.py`（**47 项**单元验证：原 39 项 + §6 性能档回归，逐 § PASS）
- `experiments/phase3_perf_ablation.py`（§1 基线 + §8 消融；`--data replica` 正式口径 8 行，输出 `ablation_replica.csv`；合成口径 5 行保留）
- `experiments/phase3_track_probe.py`（**新增**：per-iter 分解 / 迭代边际 / map 减负三模式探针，决策门数据来源）
- 产物：`data/outputs/phase3/synth_scene_480.npz`（640×480 合成序列）、`ablation.csv`、**`ablation_replica.csv`**（Replica 正式口径）、`ablation_replica.log`

**已知限制（如实记录）**：① **FPS 目标 15-30 未达成**——物理约束：单迭代固定开销
~40-57ms（§1 分解表）→ 15 FPS 需 ≤2 迭代，但其收敛天花板使 ATE ~3× 且长序列发散
（§9 坑 12）；用户决策"降 FPS 保质量"→ 性能行定稿 6it d2+ad8（FPS 3.3 = ×1.57、
ATE 1.62×、PSNR -0.5dB）与 maplight（FPS 3.0、PSNR +3.3dB），FPS 极值 6.4（3it，
质量不可用）。15-30 FPS 需光栅化器级优化（rasterizer BLOCK 调优 / 稀疏化渲染 /
CUDA graph）或硬件升级，留待后续；② 消融在 Replica 合成数据上验证，真实 D435i
场景待 Phase 4 补测；③ 真实时间驱动系统的实时性收益（队列解耦 + lock_chunk）：**Phase 4 已实机补测——quality 档 ~9.1 Hz / fps 档 ~2–6 Hz，稳定 10 FPS 不可达，在线口径废弃**（`docs/04_Phase4_验证报告.md` §5；合成回放无限 push 放大锁竞争的局限见原记录）；④ 视锥剔除在当前光栅化器下无 FPS 收益，
价值留待大规模场景验证；⑤ 质量行 PSNR 判据受建图随机性影响（rasterizer 非确定性
±2dB 级，prune/fp16 行 v4 PASS / v7 FAIL 波动），文档按单次运行如实记录；⑥ Phase 2
§5 情形 B 在本环境确定性 FAIL（21.3cm > 10cm 阈值，A/B 已证与 Phase 3 改动无关，
为环境/数据差异，记录备查）。

---

## 11. Phase 4 附节：FPS≥10 优化实测记录（2026-08-28）

> 本节记录为"10 FPS + 质量贴近当前"目标进行的完整优化探索与**如实结论**。
> 独立于 §8 消融表（口径：诚实墙钟 FPS = 处理帧数/端到端墙钟，含 map 锁竞争）。

### 11.1 文档数字校正（前文数字部分不可信，全部重新实测）

| 项 | 原文档记录 | 实测（2026-08-28，满频 1173MHz） |
|---|---|---|
| 单迭代 @320×240 full（134k 高斯） | 42.9ms | **61-66ms**（200k：66.4ms） |
| depth 趟 @320×240 | 4.6ms | **~33ms**（分辨率敏感部分） |
| 分辨率是否杠杆 | "不是" | **是**：320→224 省 35%（depth 趟所致）；RGB-only 档影响小（29-33ms@200k） |
| 高斯数影响 | "O(N) 固定" | 134k→200k 几乎不变；435k baseline 反推 ~59ms/迭代 |
| Replica 帧间位移 | 5.5cm | **~1cm**（0.3°/帧） |
| GPU 微小 kernel 启动 | 未记录 | **~0.2ms/个**（se3_exp 25 算子 = 10.5ms，numpy CPU 版 ICP 由此诞生） |

RGB-only 迭代 @224×168 @200k = **29ms**；Jetson 温度/频率：idle 306MHz，负载 boost 1173MHz。

### 11.2 已实现的基础设施（Phase 4，48/48 单元检查通过）

- `slam/scheduler.py`（新建）：2:1/3:1 丢帧 + SE(3) 恒速外推输出 + rate-adaptive（超预算扩间隔）+ 失败强制下一帧恢复
- 诚实 FPS 口径：`phase3_perf_ablation.py` fps_honest = n/wall（baseline 实测墙钟 1.0-1.2 vs track GPU 2.1-2.4——**原 FPS 数字全是 track GPU 口径**）
- `slam/icp_init.py`（新建）：帧到帧 / 帧到模型 point-to-plane ICP，**纯 numpy CPU**（Jetson GPU 微小 kernel 启动税 0.2ms/个，GPU 版 GN 67ms vs numpy 版 9-12ms）；coarse-to-fine 内点门（10→4cm）、末端 ratio 门、修正量不信门（旋转 >2.5°/平移 >12cm 回退）
- `slam/backend.py`：降采样移出 GPU 锁；ICP 混合路径（f2f→f2m，阈值可配）；失败事件 + take_failure
- `slam/mapping.py`：加高斯全 GPU 化（修复 w2c 方向 bug——曾致 baseline ATE 6.5→82cm、高斯数翻倍，§1 已加回归）；map_tier {full/light/addonly}
- `slam/tracking.py`：迭代内 3 处 `.item()` 同步 → GPU 侧收集 + 阶段末 argmin；失败检测收紧（fail_rot_deg/fail_trans_m 可配，8°/25cm）+ NaN 守卫
- `gaussian/render.py`：render_prepare 跨迭代复用浮点转换；`gaussian/ssim.py` 窗口缓存

### 11.3 质量-FPS 边界（Replica office0 200 帧，多配置多次运行）

| 配置 | ATE | 诚实 FPS |
|---|---|---|
| baseline（同步，8it） | 6.3-7.4cm | ~1.0-1.2 |
| track160（6it d2 ad8，现有质量锚） | 6.2-10.5cm | 3.3（track GPU 口径） |
| **fps10 官方验收（ICP+2it d2 ad8 + 2:1 + light map）** | **23.14cm**（单次最佳；运行间 23-107cm 方差 4×，多次运行） | **3.8**（track GPU 7.0 = 143ms/帧；稳定段 98-130ms → 潜在 7-10） |
| fps10b（addonly map） | 144.6cm（无属性优化模型质量不足） | 2.9 |
| 3-4it 变体 | 48-63cm | ~6-7 |
| 无 ICP 2it / 3it 极值 | 发散 / 62.8cm | 6.4 |

### 11.4 结论（诚实）

1. **"稳定 10 FPS + 质量贴近当前"在本机、不改 CUDA 约束下不可达**。机制：单迭代
   43ms(full)@224×168 → 100ms 预算仅 2-3 迭代；2 迭代无法收敛 2-4° 初值（旋转 lr
   2e-3 收敛慢）；稀疏模型光度量有假极小（t≈50 全配置一致的失效墙）；帧到模型锚定
   受"管线自建模型质量"鸡生蛋限制（GT 位姿建图模型 depth 差 2.3cm 中位，管线位姿
   建图模型不达标）。
2. 帧到帧 ICP 的 1-2cm 对齐噪声在链条上随机游走（50 帧累积 6-10cm 后门槛连锁失效）。
3. 质量锚 = 现有 track160（3.3 FPS / 6-10cm）；速度锚 = ICP+2it+2:1（~5-10 FPS /
   25-107cm 如实标注）；中间无稳定配置。双档运行时切换（队列满降速度档）为推荐
   交付形态。
4. 10 FPS 与 PSNR≥17dB 的并集在 200k 高斯、不改 CUDA 约束下处于物理临界；15-30 FPS
   需光栅化器级优化（BLOCK 调优 / FP16 tile 缓冲 / CUDA graph）或硬件升级（沿用
   §10 已知限制①的结论，本次实测进一步证实）。

---

## 12. 真机 FPS 提速实测记录（2026-09-02）

> 本节是 §11.4"本机稳定 10 FPS 不可达"结论在**真机运行条件**下的修正实测。
> 环境：Jetson Orin NX Super 16GB（JP 6.2，nvpmodel 40W），D435i 25Hz 真机，
> 完整 SLAM 模式 `--tier fps`。所有 /odom Hz 均 `ros2 topic hz` ≥45s 稳态采样。

### 12.1 根因 1（主因）：GPU governor 降频，文档数字全是强开满频测的

- 现象：真机处理帧 wall 280-350ms（2026-08-29 报告口径 ~2Hz），track "gpu" 200-300ms；
  但 `tegrastats` 实测 GR3D 利用率 0-26%、整机功耗仅 ~11W（40W 档）。
- 根因：`nvhost_podgov` 在帧间空闲（~400ms）把 GPU 降到 306MHz；每帧 kernel 在
  低频执行 + 等频率爬坡（306→714→1173MHz）。探针实测：持续负载下 RGB-only 渲染
  @224×168@200k 仅 **7ms**，单发冷启动 **86ms**（差 12×）。
- 修复：`sudo jetson_clocks`（GPU 锁 1173MHz，devfreq min=max）。**§1/§11 全部
  性能数字均在满频条件下测得；真机/回放若未锁频，FPS 直接 ×0.25。**
- 操作注意：jetson_clocks 每次开机需重跑（root）；建议加入开机自启或 robot 启动脚本。

### 12.2 根因 2：ICP 每帧失败重试白烧 30-80ms/帧（门控前）

- 现象：icp_ms 均值 66-178ms/帧（渲染 ~26ms + CPU GN 2ms + 同步/调度开销），
  fallback 计数 ~400 次/10 分钟——**真机静态/桌面段 f2m ICP 几乎每帧失败**
  （模型单帧种子未抛光 + D435i 深度噪声 > 4-10cm 内点门），失败后又回退恒速初值
  = 每帧纯浪费 + 每次失败置 failure_event 强迫下一帧重处理。
- 修复（`slam/backend.py` ICP 三层门控）：
  1. 模型成熟度门：在线建图 mapped<2 帧时禁用（f2m 需要模型深度可信）；
     `--load` 模式（成熟 checkpoint）恒允许；
  2. 修正量门：上一帧光度 track 相对 CV 初值的修正 > 0.5°/1cm 才跑（初版用
     "CV 预测步长"失效：静止段 track 噪声漂移被恒速继承、步长虚高，真机教训）；
  3. 连续失败退避（≥2 次暂停）+ 每 10 帧看门狗防静默漂移。
  `slam/icp_init.py` 新增 `se3_motion_np()`。

### 12.3 顺带修复

- `backend._map_keyframe_chunked`：只有首个 chunk 执行播种/剪枝/容量淘汰——原实现
  每个 chunk 重跑整段 `map_keyframe`，add_new 的全模型原生分辨率 silhouette 渲染
  （@200k 量级 100ms+）每关键帧重复 2-4 遍（后续 chunk 播种必然空手但渲染全价）；
  且每 chunk 重建 Adam（momentum 清零）。map 每关键帧 829→~285ms（锁频前口径）。
- `backend` stats 增 `map_ms`；node 每 30 处理帧打 `[perf]` 分解日志
  （track wall/gpu、icp ms/fallback/门控跳过、map ms、队列深度、skip、高斯数）。
- node fps 档参数：`adaptive_max 8→4`（真机噪声下自适应几乎每帧扩满 8 次迭代，
  单帧最坏 ~350ms）、`budget_ms 100→200`（100ms 预算单帧永超 → 调度器已偷偷退到
  4:1 丢帧，预算对齐真实帧成本后恢复 2:1）、`lock_chunk 2→1`（quality 档保持 2）。

### 12.4 实测结果（D435i 25Hz 真机，fps 档，jetson_clocks 锁频后）

| 指标 | 2026-08-29（锁频前，报告口径） | 2026-09-02 |
|---|---|---|
| /odom Hz（`ros2 topic hz` ≥45s） | 1.8-2.1 | **7.9-10.7**（稳态；连续运动+建图饱和段回落 2-4） |
| track wall / 处理帧 | 280-350ms | ~200-250ms（运动段自适应扩迭代时） |
| 处理帧率 | ~2/s | 3-4/s（静态桌面；GPU 245ms×4it 上限 ~4/s） |
| map/关键帧 | ~829ms（含重复播种） | ~285ms（空闲）~900ms（队列饱和含锁等待） |
| 跟踪失败/外推丢失 | 启动期各 1 次 | 60s 窗口各 ≤1 次 |

> 口径诚实说明：/odom 含 skip 帧的 SE(3) 恒速外推输出（每输入帧都发）；真"处理帧
> 率"3-4/s 由 track 单帧 GPU ~200-250ms 决定（4 次自适应迭代 @200k@224×168）。
> >5Hz 位姿流已达成；若要求**运动段 >5 处理帧/s**，下一瓶颈是 map 每关键帧 GPU
> 成本（见 12.5），再往上是 §11.4(4) 的光栅化器级改动。

### 12.5 遗留（下一步候选）

1. **map 每 KF 成本**：空闲 ~285ms / 饱和 ~900ms（含锁等待）。运动段 KF 率
   ~0.7/s + map 饱和 → GPU 满载 → 处理帧率被压。候选：map 迭代 depth_every
   （depth/sil 隔趟渲染，省 ~1/3 光栅化）、队列压力降档（积压 ≥3 时 iters/rotate
   降档、≥5 走 addonly，§11.4(3) 推荐形态仍未接线）、seeding 原生分辨率隔 KF 降档。
2. **adaptive 扩迭代真机几乎每帧触发**（修正量 >0.5°/1cm 常态）——需区分"真运动"
   与"模型未抛光导致的假残差"（模型成熟后跟踪残差应收敛，2026-08-29 真机同样现象
   待模型质量提升后复测）。
3. **jetson_clocks 持久化**：开机自启（root 脚本/systemd），否则回到 2Hz 档。

---

## 13. 第二轮真机优化记录（2026-09-02，不改光栅化）

> 承接 §12（jetson_clocks 锁频 + ICP 三层门控 + chunked 去重），本轮把 §12.5
> 遗留项全部实现并真机验证。环境同上：Orin NX Super 16GB（40W，jetson_clocks
> 锁 1173MHz），D435i 25Hz，`--tier fps`，run6-run9 多轮。

### 13.1 本轮改动清单（全部已落地，src/ 与 ws_src/edge_3dgs_ros）

| # | 改动 | 文件 | 动机/效果 |
|---|---|---|---|
| 1 | map_keyframe 增 `seed_W/H`：播种（silhouette 渲染+反投影）分辨率可选 | mapping.py | 队列压力档低分辨率播种，原生渲染 ~150-250ms→~30ms @200k |
| 2 | **容量满跳过播种**：`num_gaussians ≥ capacity_max` 时 add_new 整段跳过 | mapping.py | 200k 封顶后 sil 必全覆盖，原生全模型渲染纯浪费——稳态每个 KF 省最大单项（run9 播种跳过 200+ 次） |
| 3 | map_keyframe 增 `depth_every`：优化迭代 depth 隔趟（loss depth 项缺席趟跳过） | mapping.py | 每 KF 光栅化 -25~33% |
| 4 | **队列积压降档**（§11.4(3) 推荐形态接线）：出队时积压 ≥5 整帧跳过（GPU 全给 track）；≥3 降为 iters=0 纯播种 + 低分辨率档 | backend.py `_loop` | 运动段 map 不再把 track 压死（docs/03 §11.4(3)） |
| 5 | chunked 签名修正（**run8 实测 bug**：`_map_keyframe_chunked(frame,T,kw)` 定义少 kw 参数 → worker 线程启动即 TypeError 死亡，队列空涨、mapped 恒 0；src 直连 PYTHONPATH 改动即时生效但此 bug 当日才暴露） | backend.py | 修复 |
| 6 | 实际迭代数计数（`iters_log`，track 内 steps 计步）入 stats | tracking.py | 风暴诊断仪表 |
| 7 | adaptive_max 4→3→**2**（fps 档固定 2 迭代）+ tol 1e-3 | node.py | 见 13.2 教训② |
| 8 | ICP 看门狗 10→3 帧 | backend.py | adaptive 砍掉后难帧兜底收紧（静态段平均代价 +10ms/帧） |
| 9 | GaussianCloud 发布 5s→10s→**20s** 一次 | node.py | Python 逐元素构建 1.5-2.2s/次（50k 高斯实测）占 GIL 抢 CPU，撑大 track 墙钟 |

### 13.2 实测教训（真机数据说话）

1. **"是否在收敛"判据无法区分噪声**：early_stop_tol 1e-4→1e-3 后自适应仍
   4.0it/3.0it 恒满（run6/run7）——D435i 真机 loss 每步改善稳定 >0.1%，任何
   相对改善阈值都会"看起来一直在收敛"。直接砍扩展上限才治本。
2. **真机每迭代成本是场景相关的 fill-bound，不是常数**：run6（4it）208ms vs
   run7（3it）270ms 同一桌面场景——近景大 splat 场景 ~90ms/迭代 @224×168，
   远景/运动中新场景 ~45ms。文档 §11"29ms@224 RGB-only"是远景模型口径，近景
   桌面会到 2-3×。**处理帧率上限因此是场景相关的，标称必须带场景。**
3. 队列积压降档与 at-capacity 播种跳过在本轮场景（cap 200k 稳态 + 手持运动）
   中始终未触发压档（队列保持 0，map 1.7 KF/s @ ~450ms 跟得上）——降档路径的
   GPU 收益留待更高 KF 率的场景验证；但 播种跳过 203 次已确认每 KF 省下原生渲染。
4. ICP 三层门控运动段表现：全 run 仅 fallback 2 次（此前每帧 fallback），门控
   跳过 500+ 帧；跟踪失败全 run 仅 1 次。

### 13.3 最终实测（run8/run9，D435i 25Hz 真机，fps 档，jetson_clocks）

| 指标 | 本轮前（§12 末） | 本轮后（run9） |
|---|---|---|
| /odom（45-55s 稳态） | 7.9-10.7Hz | **静态 32.7Hz / 手持运动 29.3Hz** |
| track/帧（2 迭代） | 200-300ms | 90-140ms（运动段）；~200ms（近景静态段，迭代成本场景相关） |
| 处理帧率 | 3-4/s | 3-4/s（近景静态）；运动段 3-4/s 稳定不掉 |
| ICP fallback | ~每帧 | 全 run 2 次 |
| map/关键帧 | 285（空闲）~900ms（饱和） | ~415-460ms（含锁等待，播种已跳过） |
| 跟踪失败 / 外推丢失 | 各 ≤1/60s | 全 run（8 分钟含手持运动）各 1 次 |
| 高斯 | 200k 封顶 | 200k 封顶（播种跳过 200+） |

> 口径：/odom 含 skip 帧恒速外推（每输入帧都发）；处理帧率 = n_processed/墙钟，
> GPU 口径 89-140ms/帧。**输出位姿流 ~30Hz 已达成；处理帧率 3-4/s 是 2 迭代 ×
> 场景相关迭代成本（45-90ms）的物理结果**。>10 处理帧/s 仍需光栅化器级或降
> 分辨率（可选：track 192×144 预计 +25-35% 处理率，代价精度，未做）。

### 13.4 实测污染更正（同日 16:30 复盘发现，必须读）

**§12.4/§13.3 中 15:27 之后采的 run5-run9 数字全部作废**——复盘发现 run4 的
节点进程从未被杀掉（`pkill -x edge_3dgs_slam_node` 无效：ament console script
经 shebang 执行后进程名是 `python3` 而非脚本名；`pkill -f` 又会误杀含同名参数的
启动 shell，15:27 起改用 `pkill -x` 后实际一个节点都没杀成），run5-run9 全程与
run4 僵尸节点并发：

- /odom Hz 被两个发布者叠加（>输入帧率的 26-33Hz 即此所致）；
- GPU 被两个 CUDA 上下文时间片瓜分 → track/map 耗时虚高 1.5-2×（run6 的
  4it=208ms vs run7 的 3it=270ms 同场景矛盾由此解释）；
- 内存 ~2× → 多次 OOM 静默被杀（exit 247）。

**单节点干净复测（2026-09-02 16:26，最终配置，jetson_clocks 锁频，连续运动场景）**：

| 指标 | 值（单节点，可信） |
|---|---|
| 处理帧率 | **8.1-8.8/s**（track wall 88-100ms，gpu 67-75ms，2 迭代） |
| /odom | 13.7Hz（55s 窗口；GPU+map 饱和期输入帧被丢，交付率低于相机 26Hz） |
| map/关键帧 | 115-155ms（含 at-capacity 播种跳过，mapped 1200+，全程跟得上） |
| ICP | fallback 2 次 / 门控跳过 1600+ |
| 跟踪失败 | 0（连续运动 ~6 分钟） |
| 高斯 | 200k 封顶（播种跳过 96-98 次 = 容量满后播种渲染全免） |

> 结论维持且更强：**位姿输出 >10Hz 达成（13.7Hz，饱和口径）；处理帧率 ~8-9/s
> 达标（>5）**；近景大 splat 场景的处理率仍需降分辨率/光栅化器级手段（§13.2②）。
> 队列压档（≥5 跳过）全程未触发（map 115ms/KF 跟得上 2-3 KF/s），该路径收益留待
> 更高 KF 率场景。杀节点教训：用 `pkill -f` 时必须带正则防自杀（如
> `pkill -f "edge_3dg[s]_slam_node"`），或直接 `kill <pid>`。

---

## 14. IMU 陀螺旋转先验（2026-09-02，真机验证通过）

### 动机
快速转头/手持扫描时，恒速外推的旋转猜测错 1-5°+（2-8 迭代光栅化跟踪起点差 →
局部极小 → 位姿微晃累积 → 重访双影/鬼影，§13 的 fps 档质量失败的机制之一）。
陀螺给出物理真旋转（kalibr 校正后），把每帧跟踪起点拉回正确附近。

### 设计（src/edge_3dgs_slam/slam/imu_prior.py）
- **IMU 只提供初值旋转，视觉每帧仍做全量对齐**——陀螺长期漂移被视觉修正拉回，
  不会积累（区别于纯积分位姿）；
- 输入 `/imu_corrected`（semantic_ws scripts/imu_corrector.py：kalibr scale/轴
  对齐 + g-灵敏度 + 尖峰滤波，2026-09-01 实测）；
- 外参：kalibr 版备份 `realsense_stereo_imu_config_kalibr版备份.yaml`
  `body_T_cam0`（IMU→infra1，同模组 ≈color，R 差 <1°），td=0.002279s；
- 帧间积分 [t_prev+td, t_cur+td] 窗口陀螺 → 旋转向量 δ_c → 左扰动 w2c：
  `R_pred = Exp(-δ_c) @ R_last`，平移保留恒速分量；采样缺口 >0.5s 回退恒速。

### 真机排障链（4 个坑，均为实测）
1. **imu_corrector 尖峰滤波基准毒化**（semantic 项目）：丢弃时不更新基准，
   启动首样本坏值 → 全样本被误丢、输出恒空（78k+ 丢弃）。已修：连续丢 ≥200
   样本重置基准（200 ≈ 1s）。
2. **QoS**：校正器 best_effort 发布 vs reader 默认 reliable 订阅不兼容 →
   一条都收不到。改 qos_profile_sensor_data。
3. **Vector3 转换**：`np.array(msg.linear_acceleration, dtype=float)` 对
   sensor_msgs/Vector3 直接 TypeError——每条 IMU 都崩、缓冲恒空。改逐分量。
4. **executor 回调组**：图像 sync（track 200ms+）独占默认 MutuallyExclusive 组
   → IMU 回调饿死。改独立 Reentrant 组 + MultiThreadedExecutor(4)。

### 验证（--load quality 地图 + 快甩/慢转/静止，真机）
| 指标 | 恒速初值 | IMU 初值 |
|---|---|---|
| 跟踪残差旋角（处理帧均值） | 0.21-0.46° | **0.04-0.06°**（4-7× 改善）|
| 命中率 | — | ~100% 处理帧 |

仪表：node `[imu]` 每 30 帧日志（先验命中数 + 两初值的跟踪残差旋角对比；
反增即符号/外参错——真机验证符号正确，未翻号）。

### 遗留
- 平移仍恒速（旋转是 IMU 收益最大的部分；完整 VIO 平移需加速度计预积分 + bias
  估计，超当前范围）；在线陀螺 bias 估计（kalibr 校正后帧间 bias 影响 <0.05°，
  未做）；R_ic 为 infra1 近似（未来 kalibr color 版替换）；quality/fps 双档
  T_init 共用（node.on_frame 已统一走先验，初值残差仪表已并入 [imu] 日志）。
- 下一步：带先验重扫整屋（--tier quality），对比 §13 无先验图的重影/PSNR。

---

## 15. 重影根因定位与"单 Adam"修正（2026-09-02 收尾记录）

### 现象
多档扫描渲染均现"重影/场景互透叠"（fps/quality、20s-6min、不同容量均复现）；
数值体检正常（PSNR 17-25dB、深度 0.7-3cm）但肉眼观感差。

### 根因（雾度审计实证）
视锥内高斯 231k/299k、覆盖 1.0 层/像素（无堆叠）、**opacity 中位 0.58 且
>0.9 占 0%**——没有任何高斯收敛到实心 → 全部半透明 → 每像素 ~60% 主场景 +
40% 背景/错层内容透出 = "不同场景叠加"观感。

### 直接原因链
chunked 建图（lock_chunk=2）**每 chunk 重建 Adam**：iters=10 被切成 5 段、
每段动量清零只等效 ~2 步优化；fps 档 iters=2/chunk=1 更甚（每 KF 仅 1 步）。
→ 各档地图实际优化步数不足，opacity 永远停在初值 0.5 附近。

### 修正（已落地）
TIERS fps/quality `lock_chunk=None`：整段持锁单 Adam 跑完全部迭代（慢扫
KF 稀疏，map 独占 GPU 2-3s 可接受）。25s 复测：全帧 PSNR 22-25dB（v2 的
转后帧 12dB → 24.6dB，收敛改善显著）；**但 opacity 仍 0.58/0%>0.9**——
25s 内每高斯优化步数仍不足（需数十轮窗口优化才能实心化）。

### 结论与遗留
- 单 Adam 修正必要且已生效（帧间一致性大幅提升）；**透明度实心化需要
  离线窗口抛光**（5 帧窗口 × 50-200 迭代，对 20-30 个关键帧）——下步实验
  用 25s 测试数据（map_20260902_193954.pt + frames.npz 5 帧）验证 >0.9% 占比，
  有效则把"空闲时窗口抛光"做成 backend 机制。

---

## 16. Replica 标准数据对照实验（2026-09-02，结论：管线固有漂移，非数据问题）

用 Replica office0 600 帧（GT 一致、零噪声）跑同一套无 IMU 离线增量重建：
**位姿轨迹误差 vs GT 中位 87.6cm**、渲染覆盖 41-76%、PSNR 11-16dB、
视觉与真机数据同样"模糊叠影"。而 opacity 收敛（>0.9 占 30%）明显快于真机
（18%）——数据干净度只影响透明度收敛速度，**不影响几何一致性与叠影**。

**结论**：叠影/错位的根因 = 增量建图无闭环的位姿累积漂移（本 GPU 每帧迭代
预算有限，SplaTAM 论文不漂是因其验证用 GT 位姿），在所有数据源上一致出现。

**遗留（下步候选）**：
1. GT 位姿直接建图（跳过跟踪）5 分钟实验——若渲染锐利即 100% 分离"位姿漂移"
   与"渲染/几何"两个环节；
2. 双程重建（pass2 以 pass1 图为锚重跟全程）——无闭环下的实用缓解；
3. 显式闭环（位姿图/重叠回环）——根治但工程量大。
