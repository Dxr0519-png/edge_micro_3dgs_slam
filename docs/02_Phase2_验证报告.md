# 02 · Phase 2 验证报告（2026-08-27）

> 验证对象：[`docs/02_Phase2_RGBD_3DGS_SLAM基线.md`](02_Phase2_RGBD_3DGS_SLAM基线.md)
> 验证方式：静态核对（文档 vs 代码）+ 动态复现（重跑验证脚本）+ Replica office0 真实数据验证
> 环境：Jetson Orin（aarch64, sm_87），torch 2.11.0 CUDA 12.6，evo 1.37.0，cv2/scipy 可用
> 数据：`third_party/SplaTAM/data/Replica/`（8 场景×2000 帧，Replica.zip 已下载解压）

## 1. 执行记录

| 步骤 | 命令 | 结果 |
|---|---|---|
| 预检 | 8 场景文件数 + traj 行数 + npz/ply sha256 | ✅ 全场景 4000 文件 + 2000 行；npz `d627ac8f…` 全程未变 |
| 复现 A | `experiments/phase2_render_minimal.py`（eps=1e-3 与 1e-2） | ⚠️ 单高斯 rel 判据 FAIL（见 §3） |
| 复现 B | `experiments/phase2_slam_synthetic.py` ×2 次 | ⚠️ §5 情形 B 稳定 FAIL（见 §3） |
| 消融 | `experiments/phase2_ablate.py --lambda-iso 0.01` | ⚠️ 方向吻合，数字不吻合（见 §3） |
| Replica | `experiments/phase2_replica_eval.py --scene office0 --frames 200 --downscale 0.5 --map-iters 30` | ✅ 全部 PASS（见 §4） |

git 基线：`d85a9bf`（工作区另有未提交的 WIP 改动，不涉及被测渲染/建图路径）。

## 2. 逐条核对表（文档 vs 代码静态核对）

| 文档章节 | 声明 | 验证方式 | 结果 | 证据 |
|---|---|---|---|---|
| §1 | SplaTAM CVPR 2024 重构版：无 gaussian_splatting/ 子包，高斯参数为 dict | 目录对比 | ✅ | `third_party/SplaTAM/` git log "CVPR 2024 Updates"；scripts/{splatam,post_splatam_opt,export_ply}.py、utils/{slam_helpers,slam_external,keyframe_selection,recon_helpers,graphics_utils,neighbor_search,common_utils}.py 全对应 |
| §2 | 依赖就绪检查；torch CUDA 12.6；rasterizer 编译 | import + 版本 | ✅ | torch 2.11.0 / CUDA 12.6；`diff_gaussian_rasterization` import OK；docs/00 §4 惯例存在 |
| §4 | `init_from_depth`：反投影 + stride 下采样 + σ=2·z/focal | 代码对照 | ✅ | [init.py:47-50](../src/edge_3dgs_slam/slam/init.py#L47-L50) |
| §5 | `track`：左扰动 se(3)、0.8·l1+0.2·(1-ssim)+1.0·l1_depth、Adam | 代码对照 | ✅ | [tracking.py:67-91](../src/edge_3dgs_slam/slam/tracking.py#L67-L91)（含 sil mask、rot_lr_ratio=0.2、lr 衰减） |
| §6 | `map_keyframe`：加高斯→属性优化+iso_loss→剪枝 opacity<0.005 | 代码对照 | ✅ | [mapping.py](../src/edge_3dgs_slam/slam/mapping.py)（`_LAMBDA_ISO=0.001`、剪枝阈值一致） |
| §8 | `build_map` / `track_one_frame` 统一接口（含 T_prev 时间一致性） | 代码对照 + 调用 | ✅ | [slam/__init__.py](../src/edge_3dgs_slam/slam/__init__.py) |
| §9 | 产出物：240 帧 npz、PLY、tum 轨迹 | 文件检查 | ✅ | `synth_scene.npz`(240×240×320, K fx=310)、`synth_map.ply`、`gt/est_traj.tum` 各 240 行 |
| §10.5 | projmatrix 双重变换 bug：纯透视转置 + means3D 相机系 | 代码对照 | ✅ | [render.py:24-48](../src/edge_3dgs_slam/gaussian/render.py#L24-L48) 注释与实现 |
| §10.7 | se(3) 梯度断点：θ+1e-12 连续公式、in-place 反对称阵 | 代码对照 | ✅ | [se3.py:85-104](../src/edge_3dgs_slam/utils/se3.py#L85-L104)（`_skew` in-place 赋值） |
| §10.9 | iso_loss 权重 λ=0.01 有害 → 0.001 | 消融复测 | ⚠️ | 方向吻合，具体数字见 §3 |
| §10.11 | 合成 depth 语义 z=t·cosθ | 代码对照 | ✅ | [phase2_synth_dataset.py:156-161](../experiments/phase2_synth_dataset.py#L156-L161) 注释 |
| §7 | evo 轨迹评估 | 工具检查 + CLI 实测 | ✅ | evo 1.37.0 + `evo_ape` 可用（脚本内 API 路径失效，见 §5-4） |

## 3. 复现数字对比表（文档声称 vs 本次实测）

### 3.1 phase2_render_minimal.py（§3/§9 梯度校验）

| 指标 | 文档声称 | 本次实测 | 判定 |
|---|---|---|---|
| 单高斯 rel 误差 | 8.3e-5 < 1e-4 | eps=1e-3：**2.66e-4**（FAIL）；eps=1e-2：**1.5e-2**（FAIL） | ❌ 不可复现 |
| 单高斯绝对误差 | — | mean 4.3e-6（梯度本身正确） | ✅ 解析梯度正确 |
| 多高斯网格校验 | — | mean 绝对误差 7.7e-6 < 1e-4 | ✅ PASS |

结论：判据「rel < 1e-4」对**小梯度分量**（y/z ≈ 0.005–0.02）过于敏感，且 `target=torch.rand` 未设种子，逐次运行梯度分量不同 → 结果波动跨越阈值。文档的 8.3e-5 是某次幸运运行的读数，非稳定性质。建议：设固定种子 + 判据改为「abs < 1e-5 或 rel < 1e-4」。

### 3.2 phase2_slam_synthetic.py（§4-§8 端到端，跑 2 次）

| 指标 | 文档声称 | run1 | run2 | 判定 |
|---|---|---|---|---|
| 首帧 PSNR | — | 27.29 dB | 27.29 dB | 稳定 |
| 首帧 depth RMSE | 25 cm | 24.94 cm | 24.94 cm | ✅ 稳定吻合 |
| build_map 高斯数 | "14 万" | 150,111 | 150,024 | ⚠️ 实为 15.0 万 |
| 全序列 ATE（240 帧） | 24.9 cm | 23.41 cm | **15.00 cm** | ⚠️ 同量级但逐次波动 ±30% |
| 稳定段 ATE（前 60 帧） | 5.1 cm | 4.89 cm | 5.05 cm | ✅ 吻合 |
| 单帧跟踪精度 | 2.9 cm | 2.54 cm | 2.54 cm | ✅ 吻合 |
| 全序列渲染 PSNR | 20.9 dB | 21.66 dB | 21.66 dB | ✅ 吻合 |
| §5 情形 A（真值初值） | PASS | 2.05 cm | 1.87 cm | ✅ <5cm 稳定通过 |
| §5 情形 B（扰动 5cm/3°） | **全部 PASS** | **21.32 cm FAIL** | **21.32 cm FAIL** | ❌ **稳定失败**（判据 <10cm） |
| 鲁棒性（翻转注入） | PASS | PASS | PASS | ✅ |
| 拒绝帧数 | — | 1 | 1 | 一致 |

结论：§9 主要数字（RMSE 25、稳定段 5.1、单帧 2.9、PSNR 20.9）均可复现；但：

1. **「§4-§8 全部 PASS」不可复现**：§5 情形 B（扰动初值恢复测试）两次独立运行均收敛到 21.32 cm（判据 <10cm），为当前代码的一致行为。情形 B 的目标帧是未参与建图的帧，模型覆盖不足 + 扰动初值落入局部极小，是判据边缘的敏感测试。
2. **全序列 ATE 逐次波动**（15.0–23.4 cm）：rasterizer backward 的 atomicAdd 非确定性使地图逐次略异（高斯数 150,024–150,162），下游累积指标随之波动。文档的 24.9 cm 是其中一次读数，非稳定值。稳定段（60 帧）与单帧指标波动小，结论可靠。

### 3.3 消融（§10.9 iso_loss）

| 指标 | 文档声称 | 本次实测 | 判定 |
|---|---|---|---|
| λ=0.01 远处误差 | 110 cm | mean **41.6 cm**（median 2.9，系统性偏浅 41.0） | ⚠️ 方向吻合、数字不吻合 |
| λ=0.01 系统性偏浅 | 有 | 有（41 cm） | ✅ 机制确认 |
| λ=0.001（基线，σ×2） | 首帧 RMSE 89→20 cm | 24.94 cm（×2 下） | ✅ |

结论：λ=0.01 确实造成远处深度系统性偏浅（机制成立）；具体数字 110cm 未复现（可能来自更早版本或不同口径/不同 σ 配置），如实记录。

### 3.4 σ 系数（§10.9 隐含的 σ×2 结论）——Replica 真实数据实证

600x340 分辨率下 σ 系数抽查（首帧 depth RMSE）：

| σ 系数 | depth RMSE |
|---|---|
| **×2.0（源码默认）** | **7.77 cm** |
| ×1.0 | 57.95 cm |
| ×0.5 | 125.89 cm |

结论：σ×2 最优结论在**真实数据、600x340 分辨率**下同样成立（合成序列 320x240 下文档消融为 ×1→84cm、×0.5→168cm，趋势一致）。×2 是 stride=2 的 2px 采样间距补偿（像素域条件），与分辨率无关，假设被实证确认。

## 4. Replica office0 真实数据验证（新增，§7 落地）

参数：前 200 帧、downscale 0.5（600x340）、stride 2、keyframe_every 5、map_iters 30、track_iters 8。输出：`data/outputs/phase2_replica/`。

| 环节 | 指标 | 结果 | 备注 |
|---|---|---|---|
| 加载 | traj.txt c2w→w2c、depth/6553.5、BGR→RGB | ✅ | `src/edge_3dgs_slam/dataset/replica.py` |
| §4 首帧 | PSNR 29.57 dB / depth RMSE 7.77 cm / 非空 100% | ✅ | PSNR≥15dB 即位姿语义正确（文档§10.11 语义闭环教训） |
| §8 建图 | 410,362 高斯，217.5s | ✅ | — |
| §8 回放 | 200 帧，拒绝 0 帧，334.9s | ✅ | 时间一致性检测未误报 |
| §7 全序列 ATE | **0.86 cm** | ✅ | 自算（Umeyama 对齐） |
| §7 稳定段 ATE（60 帧） | **1.15 cm** | ✅ | — |
| §7 单帧跟踪精度 | **0.69 cm** | ✅ | GT 初值，前 60 帧中位 |
| §7 关键帧渲染 PSNR | 29.17 dB | ✅ | GT 位姿 |
| §7 evo 交叉验证 | rmse **1.14 cm** | ✅ | `evo_ape tum -a -va`，与自算同量级 |

总耗时 646s（10.8 分钟）。总判定：**全部 PASS**。

对照说明：SplaTAM 论文 office0 全序列 ~1-2 cm 是 2000 帧在线循环（tracking+mapping 交替反馈）结果；本验证为固定模型 + 纯跟踪回放 200 帧段，无 BA 反馈，协议不同但量级相当——管线在真实数据上成立，cm 级可达。合成序列的"降级"缺口（§9）由本次真实数据验证补上。

## 5. 发现的问题清单

| # | 问题 | 严重度 | 建议 |
|---|---|---|---|
| 1 | 文档§9「单高斯 rel 8.3e-5 < 1e-4」不可稳定复现（eps=1e-3 → 2.7e-4 FAIL；eps=1e-2 → 1.5e-2 FAIL）；`torch.rand` 未设种子 | 中 | render_minimal.py 设固定种子；判据改「abs<1e-5 或 rel<1e-4」并注明小梯度分量放大 rel；文档措辞改为量级校验 |
| 2 | 文档§9「§4-§8 全部 PASS」不成立：§5 情形 B（扰动初值 5cm/3°）稳定收敛 21.3cm > 10cm 判据（两次独立运行一致） | 中 | 文档如实标注；情形 B 是模型未覆盖帧的敏感测试，判据或初值扰动幅度需斟酌 |
| 3 | 全序列 ATE 逐次波动 ±30%（15.0 vs 23.4 vs 24.9 cm）：rasterizer atomicAdd 非确定性 → 地图逐次略异 | 低 | 文档数字标注"单次读数"；评估用稳定段/单帧指标为主 |
| 4 | 脚本内 evo 交叉验证路径失效：evo 1.37 已移除 `metrics.ATE`（`AttributeError`）→ 文档§9 数字实际均来自自算 ATE | 低 | 脚本改用 `evo_ape` CLI（已在新脚本落地）；`phase2_slam_synthetic.py` 的 API 路径待修 |
| 5 | `config/slam/splatam_edge.yaml` `lambda_iso: 0.01` 与代码 `_LAMBDA_ISO=0.001` 不一致（文档§10.9 实测 0.001 最优） | 中 | 改为 0.001（已顺手修，见 §6） |
| 6 | `scripts/download_data.sh` 计划路径 `data/raw/replica/` 与实际 `third_party/SplaTAM/data/Replica/` 不符 | 低 | 已顺手更正说明 |
| 7 | `phase2_render_minimal.py` docstring 声明 eps=1e-2，argparse 默认 1e-3 | 低 | 注释与默认值对齐 |
| 8 | 文档§9「14 万高斯」实为 15.0 万（150,162） | 低 | 已改为实测值 |
| 9 | 文档§9「Replica 12.4GB 下载不现实」已过时（数据已下载） | 低 | 已更新（见文档修订） |
| 10 | 文档§9「首帧/关键帧 30+ dB」：首帧复现稳定 27.29 dB（<30），仅建图后关键帧达 30.46 | 低 | 已改为「首帧 ~27、关键帧（建图后）30+」 |

## 6. 已执行的文档/配置修订

- [x] `docs/02_Phase2_RGBD_3DGS_SLAM基线.md`：§7 补 Replica 接入；§9 替换降级说明、回填实测数字、追加验证脚本；§10 追加 3 条新坑
- [x] `config/slam/splatam_edge.yaml`：lambda_iso 0.01 → 0.001（与代码及文档实测一致）
- [x] `scripts/download_data.sh`：路径更正为 `third_party/SplaTAM/data/Replica/`
- [x] 新建 `src/edge_3dgs_slam/dataset/replica.py`、`experiments/phase2_replica_eval.py`、`experiments/phase2_ablate.py`

## 附录：输出文件

```
data/outputs/phase2/            合成复现（重跑覆盖 gt/est_traj.tum，其余不变）
data/outputs/phase2_replica/    gt_traj.tum / est_traj.tum（200 行）
                                replica_map.ply（410,362 高斯）
                                first_frame_render.png（GT | 渲染并排）
/tmp/p2_minimal.log, /tmp/p2_synth.log, /tmp/p2_synth_run2.log, /tmp/p2_ablate.log, /tmp/p2_replica.log
```
