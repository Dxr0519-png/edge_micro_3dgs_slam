# 03 · Phase 3 针对 Jetson 平台的显存与性能极致优化

> 目标：16GB 统一内存 + 有限算力下，Tracking 维持 15–30 FPS，防 OOM，产出**可写进简历的量化消融表**。
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

## 2. 高斯数量控制与剪枝（`gaussian/model.py`）

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

## 4. 视锥剔除（Frustum Culling）

```python
def frustum_visible(xyz, T_wc, K, H, W, margin=0.2):
    # 把 xyz 变换到相机系，投影到像素，判断是否在 [−margin, W+margin]×[−margin, H+margin]
    # 且在 z > near；返回布尔掩码
    ...
# 优化时只对 visible 高斯反传；不可见高斯冻结（.requires_grad_(False) 或过滤）
```

## 5. 关键帧管理 + 异步建图（保实时）

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

## 6. 渲染/迭代控制

- Tracking 用降采样（320×240），Mapping 用全分辨率（640×480）。
- `max_sh_degree=0`（不用 SH），`tracking_iters` 5~10。
- 限制每 tile 高斯数 / 总渲染高斯数。

## 7. 内存预算表（模板，实测填数）

| 项 | 占用 | 说明 |
|---|---|---|
| PyTorch 运行时 + CUDA 上下文 | ~1–2GB | 固定开销 |
| 高斯参数（30 万 × ~14 floats） | ~17MB | 极小，非瓶颈 |
| 渲染 tile buffer | ~数百 MB | 与分辨率/高斯密度正比 |
| MobileSAM/MobileCLIP（Phase 5/6） | ~1GB | 分时复用、用完 `empty_cache()` |

## 8. 验收与消融表

做 4 组消融，记录 **FPS / 峰值显存 / 高斯数 / ATE / PSNR**：

| 配置 | FPS | 峰值显存 | 高斯数 | ATE | PSNR |
|---|---|---|---|---|---|
| Phase 2 基线 | | | | | |
| + 高斯剪枝/容量上限 | | | | | |
| + FP16 存储 | | | | | |
| + 视锥剔除 | | | | | |
| + 关键帧窗口 + 异步建图 | | | | | |

> 这张表就是简历里「Jetson 边缘优化」的核心证据。

## 9. 常见坑

1. **异步建图与 Tracking 竞争 GPU**：串行化 GPU 调用（锁/事件），否则互相拖慢。
2. **FP16 导致几何漂移**：`xyz` 千万别降 FP16。
3. **剪枝过猛破坏几何**：先小阈值、看 ATE 劣化 < 20% 再收紧。
4. **统一内存「显存」看错**：Jetson 上 `free -h` 与 `nvidia-smi` 口径不同，以 `torch.cuda.memory_allocated()` 为准。
