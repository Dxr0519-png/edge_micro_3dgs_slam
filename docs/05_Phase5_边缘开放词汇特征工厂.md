# 05 · Phase 5 边缘端轻量级 2D 开放词汇特征提取（Feature Factory）

> 目标：MobileSAM + MobileCLIP 替代标准 SAM/CLIP，生成降采样 2D 语义特征图缓存，为 Phase 6 的 3D 特征投影做准备。
> 依赖：Phase 1 数据流（可并行于 Phase 2/3 开发）。
> 状态：**已实现并通过全部验收（2026-08-27 实测）**，验证脚本见文末。

## 1. 模型加载（先就绪检查，未就绪才下载/安装）

按就绪检查惯例（`docs/00` §4）：依赖与权重**先检查是否已就绪，未就绪才下载/安装**：

```bash
# ① 依赖未装才装
#    MobileCLIP-S0 用 apple 官方包 mobileclip（open_clip 不支持 S0，见 §8 坑 2）
python3 -c "import mobileclip"  || pip3 install git+https://github.com/apple/ml-mobileclip.git
python3 -c "import mobile_sam"  || pip3 install git+https://github.com/ChaoningZhang/MobileSAM.git
python3 -c "import ultralytics" || pip3 install ultralytics

# ② 权重未下载才下载（已存在则跳过）
#    MobileCLIP-S0 原版权重仅在 HuggingFace；本网络 HF 直连不可达，经 hf-mirror.com 镜像下载
[ -f data/checkpoints/mobileclip_s0.pt ] || \
    wget -O data/checkpoints/mobileclip_s0.pt \
        https://hf-mirror.com/apple/MobileCLIP-S0/resolve/main/mobileclip_s0.pt
#    mobile_sam.pt：注意分支是 master（原文档 main 分支 URL 实测 404）
[ -f data/checkpoints/mobile_sam.pt ] || \
    wget -O data/checkpoints/mobile_sam.pt \
        https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

```python
# MobileSAM（ViT-Tiny 蒸馏，实测 10.1M 参数 / 40MB 权重）
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
sam = sam_model_registry['vit_t'](checkpoint='data/checkpoints/mobile_sam.pt')
sam.eval().cuda()
mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.86)

# MobileCLIP-S0（apple 官方包 mobileclip，512 维，实测 53.8M 参数）
import mobileclip
model, _, preprocess = mobileclip.create_model_and_transforms(
    'mobileclip_s0', pretrained='data/checkpoints/mobileclip_s0.pt', device='cuda')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
```

> `MobileCLIP-S0/S1/S2/B` 均为 512 维，边缘端优先 S0（最快）；文本侧必须用**同一个**模型。
> 实现位于 `feature_factory/models.py`（`load_models` / `release_models`，模型名与权重路径全部来自 config）。

## 2. 流水线（`feature_factory/pipeline.py`，已实现）

实际签名（模型显式注入，便于 §6 显存管理）：

```python
def extract(frame_id, rgb, fm, hier_level=0) -> list[dict]:
    masks = fm.mask_gen.generate(rgb)              # 万物掩码
    feats = []
    for m in masks:
        crop = crop_resize(rgb, m['bbox'], fm.input_size, seg=m['segmentation'])
        v = clip_encode_image(fm.clip, fm.preprocess, crop, fm.device)   # (1,512) FP16
        v = v / v.norm(dim=-1, keepdim=True)       # L2 归一化
        feats.append({'mask_id':..., 'bbox':..., 'clip_vec': v[0].float().cpu().numpy(),
                      'area':..., 'predicted_iou':..., 'stability_score':..., 'hier_level':...})
    return feats
```

- `crop_resize` 支持 `seg` 掩码填充（bbox 内掩码外区域填 0）：万物掩码 bbox 常含大量背景，
  填充后特征语义纯度显著提升（实测椅子 diff 0.044 → 0.076）。
- `clip_vec` 中间态 fp16，**入缓存统一转 float32**（Phase 6 pickle 契约）。
- 配套接口：`encode_text`（文本/图像同模型）、`extract_hierarchical`（§4）、`extract_keyframes`（§6）。

## 3. 特征图缓存（掩码级，非逐像素，已实现）

```python
# 每关键帧存一次，写入 feature_map.pkl（路径以 config feature.cache.path 为准：
# data/outputs/phase5/feature_map.pkl，按项目 outputs/phaseN/ 惯例）
cache = {frame_id: {"H":H, "W":W, "masks":[{"bbox":..., "clip_vec":..., "hier_level":...}]}}
```

> **掩码级而非逐像素**是省显存关键：逐像素 512 维张量 (H,W,512) 巨大，掩码级只存每个物体的一个向量。
> 实现位于 `feature_factory/cache.py`（`save_cache`/`load_cache`/`update_cache`/`masks_to_matrix`）；
> 格式契约见 IMPLEMENTATION_PLAN §5.2，纯 dict/list/ndarray，Phase 6 任意环境可 `pickle.load`。

## 4. 分层掩码（LangSplat 思想，缓解边界歧义，已实现）

```python
# 用不同 points_per_side / pred_iou_thresh 跑两档，给每个掩码打 hier_level
mask_gen_coarse = SamAutomaticMaskGenerator(sam, points_per_side=8)    # 大物体/背景
mask_gen_fine   = SamAutomaticMaskGenerator(sam, points_per_side=32)   # 小物体/细节
```

`extract_hierarchical` 将两档合并，`hier_level`（0=coarse / 1=fine）写入每条缓存记录。
每像素「掩码 ID → 语义层级」归属表由 Phase 6 反投影监督时按层级加权（本 Phase 只标注层级）。

## 5. 场景级自编码器（512 → D 维，已实现）

```python
class LangAE(nn.Module):
    def __init__(self, d_in=512, d_latent=3):
        self.enc = MLP([512, 256, 64, 16, d_latent], act=nn.ReLU)      # 末层无激活
        self.dec = MLP([d_latent, 16, 64, 256, 512], act=nn.ReLU)      # 末层线性

def train_ae(vectors, d_latent=3, hidden=(256,64,16), lr=1e-3, lambda_cos=0.5, ...):
    v_hat = ae.dec(ae.enc(v))
    loss = (v_hat - v).abs().mean() + lambda_cos * (1 - F.cosine_similarity(v_hat, v).mean())
    # 收敛判据：重建余弦相似度均值 > 0.95（COS_CONVERGE）
    torch.save(ae.state_dict(), 'data/checkpoints/lang_ae.pt')
```

> 实现位于 `feature_factory/autoencoder.py`（`MLP`/`LangAE`/`train_ae`/`embed`/`train_ae_from_cache`）。
> 文档伪代码形参不一致（`train_ae(vectors)` 函数体用 `v`/`ae`）已统一为上述签名。

## 6. Jetson 优化（已实现）

- MobileSAM/MobileCLIP 用 **FP16 CUDA 推理**（`torch.autocast`），与 Tracking 分时复用 GPU，用完 `torch.cuda.empty_cache()`。
- 只对**关键帧**提取（`extract_keyframes`，非每帧），进一步省算力。
- `release_models` 把模型置 CPU + `empty_cache()`；注意调用方还需 `del fm`（del 局部参数不回收调用方引用）。
- **进阶（加分项，本 Phase 不实施）**：MobileCLIP 转 ONNX → TensorRT（INT8/FP16 engine）用 DLA 加速，延迟可降一个量级。

## 7. 验收清单（2026-08-27 实测）

**验证数据约定**（docs/00 §7 统一口径）：语义特征验证**必须用真实图像**——COCO 测试图 /
Replica 场景帧 / D435i 自采均可（本 Phase 实测用 COCO 图，`data/raw/phase5_coco128_643.jpg`）；
**合成场景不可用**（球/盒/棋盘无语言语义，MobileSAM/CLIP 输出无意义）。

- [x] 单帧能输出数十个有语义意义的掩码。
      —— 实测 56 掩码（640px 宽单帧，`data/raw/phase5_coco128_643.jpg`，COCO 标注确认含椅子）；
      area min=132 / 中位=1514 / max=53769，pred_iou 均值 0.954（`experiments/phase5_feature_pipeline.py` 验收1）。
- [x] 「椅子」文本与椅子掩码相似度明显高于背景。
      —— 实测「a chair」best=0.226 vs bg=0.149（raw 差 0.076，logit_scale=4.21 温度缩放后 5.1），
      语义排序正确（对照 a cat/a dog/a table 均更低）；中文「椅子」差 0.042（缩放 2.8）也通过。
      **判据适配说明**：MobileCLIP-S0 的 raw cosine 被压缩在 0.1~0.2 区间（S0 固有特性，
      标准 CLIP 的 ">0.3 余弦差" 对 S0 不可达，实测真身区域差 ~0.04~0.08），
      判据改为 **raw 差 > 0.03 且 温度缩放后差 > 2.0 且 语义排序正确**（2026-08-27 用户确认）。
- [x] `feature_map.pkl` 可被 Phase 6 反序列化。
      —— 实测 116KB / 56 掩码条目；子进程（独立 Python 环境）`pickle.load` 通过，
      结构/形状(512,)/dtype(float32)/L2 范数=1.0 全部断言通过（`phase5_feature_pipeline.py` 验收3）。
- [x] AE 重建余弦相似度均值 > 0.95。
      —— 实测真实 MobileCLIP 特征 56 条：251 iters 收敛，重载复评 cos_mean **0.9504**；
      合成低秩 2048 条：cos_mean **0.9624**（`experiments/phase5_langae.py` 验收4）。

## 8. 常见坑（含本 Phase 实测新增）

1. **MobileSAM 掩码质量差**：`pred_iou_thresh` 太低/太高都影响，调参并做分层。
2. **MobileCLIP-S0 不能用 open_clip**（实测新增）：open_clip 3.x 源码注明 "s0 not supported"，
   且 S0 权重仅在 HuggingFace（本网络直连不可达）→ 用 apple 官方包
   `mobileclip`（git+ 安装）+ hf-mirror.com 镜像下载原版权重到本地路径。
3. **MobileCLIP 输入尺寸**：必须走官方 `preprocess`（Resize 224 + CenterCrop + ToTensor，
   **无 ImageNet 归一化**——自行加归一化实测 NaN）。注意其接受 **PIL 输入**，numpy array 直接 TypeError。
4. **文本/图像模型不一致**：query 用 MobileCLIP 文本编码，特征也必须同款 MobileCLIP（同一 tokenizer/编码器）。
5. **S0 raw cosine 被压缩**（实测新增）：相似度落在 0.1~0.2 区间，绝对阈值判据（>0.3）对 S0 不可达；
   用 `logit_scale`（4.21）温度缩放或相对排序做判据（验收 2 已适配）。
6. **pip 依赖连锁破坏环境**（实测新增）：装 open_clip/timm 系列会把 numpy 升到 2.x、opencv 升到 5.x，
   导致系统 scipy（`_ARRAY_API not found`）崩溃 → 钉回 `numpy==1.26.4` + `opencv-python==4.10.0.84`。
7. **mobile_sam.pt URL**（实测新增）：文档原 `raw/main/` URL 404 → 实际用 `raw/master/`（302→200，40MB）。
8. **中文「椅子」响应弱**（实测新增）：MobileCLIP 训练语料英文主导，中文查询区分度明显偏低（0.042 vs 0.076），
   判据用英文主锚（a chair / chair）+ 中文并列报告。
9. **万物掩码 bbox 稀释特征**（实测新增）：bbox 常含大量背景，`crop_resize` 用掩码填充后
   语义区分度显著提升（椅子 diff 0.044 → 0.076）。
10. **显存**：MobileSAM+CLIP 常驻约 254MB（fp32 加载，实测），与 Tracking 分时或用完释放；
    释放需 `release_models` + 调用方 `del fm` + `torch.cuda.empty_cache()`。
11. **Jetson 无 nvidia-smi**（实测新增）：显存统计统一用 `torch.cuda.memory_allocated()`。

## 验证脚本

| 脚本 | 覆盖 | 通过标准 |
| --- | --- | --- |
| `experiments/phase5_feature_pipeline.py` | §1-§4/§6 + 验收 1/2/3 | 掩码 ≥20；椅子显著区分+排序正确；pkl 子进程反序列化；显存释放 <256MB |
| `experiments/phase5_langae.py` | §5 + 验收 4 | 重载 lang_ae.pt 后重建 cos_mean > 0.95；embed 输出 (M,3) float32 |

产出：`data/outputs/phase5/`（feature_map.pkl、masks_overlay/hier_coarse/hier_fine/chair_best_*.png）、
`data/checkpoints/lang_ae.pt`。
