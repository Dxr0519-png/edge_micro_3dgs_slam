# 02 · Phase 2 轻量级 RGB-D 3DGS-SLAM 基线构建（SplaTAM 蓝本）

> 目标：fork/裁剪 SplaTAM，跑通 Tracking + Mapping，用 D435i 深度反投影初始化（免 COLMAP），验证 Jetson 上可微光栅化可用。
> 依赖：Phase 1 的 `SyncedFrame` 数据流。核心公式见 `IMPLEMENTATION_PLAN.md` 2.4。

## 1. 拉取并读懂 SplaTAM 代码结构

```bash
cd third_party && git clone https://github.com/spla-tam/SplaTAM.git
cd SplaTAM && tree -L 2
```

必读文件（逐文件读懂，做笔记）：

```
SplaTAM/
├── gaussian_splatting/
│   ├── scene/gaussian_model.py         # GaussianModel：xyz/rot/scale/opacity/color + 增删/保存
│   ├── gaussian_renderer/__init__.py   # render()：调 rasterizer 前向（RGB + depth 两通道）
│   ├── utils/
│   │   ├── slam_backend.py             # 主 SLAM 后端：关键帧、建图循环
│   │   ├── slam_utils.py               # SSIM、深度损失、位姿变换、SE(3) 工具
│   │   └── camera_utils.py             # 内参、图像→相机系、深度后处理
│   └── ...
└── scripts/slam.py                     # 入口：逐帧 Tracking + Mapping
```

## 2. 编译 CUDA 光栅化（第一道坎）

```bash
export TORCH_CUDA_ARCH_LIST="8.7"
cd third_party/SplaTAM
pip install submodules/diff-gaussian-rasterization
# 验证
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer; print('ok')"
```

若失败：
- 检查 `submodules/diff-gaussian-rasterization/setup.py` 是否硬编码 `-march=native` / x86 架构名，去掉；
- GCC 用宿主版本，`nvcc --version` 与 `gcc --version` 匹配；
- 确认 `torch` 是 Jetson aarch64 构建。

> **降级方案**：先写 `src/edge_3dgs_slam/gaussian/torch_rasterizer.py`，用纯 PyTorch 自动求导实现慢速 splat（显式深度排序 + α 混合），先验证数学，再换 CUDA。

## 3. 最小可运行脚本（先跑通，再入库）

`experiments/phase2_render_minimal.py`：

```python
import torch
# 1) 从 ply 或随机初始化 N 个高斯（xyz/rot/scale/opacity/color）
# 2) 给定一个 (R, t, K) 位姿，调 render() 输出合成 RGB 图
# 3) 对 xyz 做数值梯度 vs 解析梯度校验，误差 < 1e-4
```

此脚本通过 = 「可微光栅化在 Jetson 上可用」得证。

## 4. 深度反投影初始化（替换 COLMAP）

首帧/关键帧初始化流程（`src/edge_3dgs_slam/slam/init.py`）：

```python
def init_from_depth(frame: SyncedFrame, T_wc):
    pts_world, valid = backproject(frame.depth, frame.K, T_wc)   # Phase 1 的 backproject
    colors = frame.rgb[valid] / 255.0
    pts = pts_world[valid]
    # 可选：均匀下采样（如 stride=2）控制初始高斯数
    return GaussianModel.create_from_points(pts, colors)          # opacity/scale 小值初始化
```

## 5. Tracking（位姿追踪）

`src/edge_3dgs_slam/slam/tracking.py`：

```python
def track(frame, gaussians, T_wc_init, iters=8):
    T = T_wc_init.clone()
    delta = torch.zeros(6, device='cuda', requires_grad=True)     # se(3)
    opt = torch.optim.Adam([delta], lr=1e-2)
    for _ in range(iters):
        T_cur = se3_exp(delta) @ T                                  # 左扰动
        rgb_r, depth_r = render(gaussians, T_cur, frame.K)
        loss = 0.8 * l1(rgb_r, frame.rgb_t) + 0.2 * (1 - ssim(rgb_r, frame.rgb_t)) \
             + 1.0 * l1(depth_r, frame.depth_t)
        opt.zero_grad(); loss.backward(); opt.step()
    return se3_exp(delta) @ T
```

> 务必先确认 SplaTAM 的位姿约定（世界→相机 or 相机→世界、左乘 or 右乘），在 `utils/se3.py` 里统一实现 `se3_exp`/`se3_log`。

## 6. Mapping（高斯建图）

`src/edge_3dgs_slam/slam/mapping.py`：

```python
def map_keyframe(frame, gaussians, T_wc, iters=50):
    # 1) 反投影新增高斯：在「无高斯覆盖」像素处 add_gaussians()
    # 2) 属性优化：xyz/rot/scale/opacity/color，加 iso_loss
    #    loss = photometric + ssim + depth + λ_iso * Σ‖scale_i - mean(scale)‖₂
    # 3) 剪枝：opacity < 0.005 或 scale 异常者 prune()
    pass
```

## 7. 数据集与评估

- 先用 **Replica**（自带真值位姿）验证正确性，再切 D435i 自采。
- 轨迹评估：`evo_ape tum <gt> <est> -va --plot`。

## 8. 封装统一接口（供 Phase 4/6 复用）

```python
# src/edge_3dgs_slam/slam/__init__.py
def build_map(rgb, depth, K, poses) -> gaussian_model: ...   # 离线建图
def track_one_frame(frame, model, T_init) -> T_wc: ...        # 在线追踪
```

## 9. 本 Phase 产出清单

- [ ] rasterizer 在 sm_87 上编译通过，前向/反向单测通过。
- [ ] Replica 单序列 ATE ~cm 级，能出渲染图 + `.ply`。
- [ ] 数值梯度校验误差 < 1e-4。
- [ ] `build_map` / `track_one_frame` 接口可用。

## 10. 常见坑

1. **rendered depth 全 0 或 NaN**：检查 rasterizer 的 depth 模式开关、`depth` 是否在相机系。
2. **位姿漂移/发散**：learning rate 太大、SSIM 权重不当、或左乘右乘搞反。
3. **高斯爆炸增长**：没做密度判据，同一像素反复加高斯——先补「近邻判据」。
4. **显存超**：本阶段高斯控制在 10 万级；正式优化在 Phase 3。
