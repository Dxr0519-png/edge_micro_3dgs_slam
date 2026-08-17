# 05 · Phase 5 边缘端轻量级 2D 开放词汇特征提取（Feature Factory）

> 目标：MobileSAM + MobileCLIP 替代标准 SAM/CLIP，生成降采样 2D 语义特征图缓存，为 Phase 6 的 3D 特征投影做准备。
> 依赖：Phase 1 数据流（可并行于 Phase 2/3 开发）。

## 1. 模型加载

```python
# MobileSAM（ViT-Tiny 蒸馏，~10MB）
# 权重: https://github.com/ChaoningZhang/MobileSAM 的 mobile_sam.pt -> data/checkpoints/
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
sam = sam_model_registry['vit_t'](checkpoint='data/checkpoints/mobile_sam.pt')
sam.eval().cuda()
mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.86)

# MobileCLIP（open_clip 预训练，512 维）
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S0', pretrained='datacomp_xl_s13b_b90k')
tokenizer = open_clip.get_tokenizer('MobileCLIP-S0')
model.eval().cuda()
```

> `MobileCLIP-S0/S1/S2/B` 均为 512 维，边缘端优先 S0（最快）；文本侧必须用**同一个**模型。

## 2. 流水线（`feature_factory/pipeline.py`）

```python
def extract(frame_id, rgb):
    masks = mask_gen.generate(rgb)                 # 万物掩码
    feats = []
    for m in masks:
        crop = crop_resize(rgb, m['bbox'], 224)    # 外接框 crop -> 224
        v = clip_encode_image(crop)                # (1,512) FP16
        v = v / v.norm(dim=-1, keepdim=True)       # L2 归一化
        feats.append({'mask_id':..., 'bbox':m['bbox'], 'clip_vec':v})
    return feats
```

## 3. 特征图缓存（掩码级，非逐像素）

```python
# 每关键帧存一次，写入 feature_map.pkl
cache = {frame_id: {"H":H, "W":W, "masks":[{"bbox":..., "clip_vec":..., "hier_level":...}]}}
```

> **掩码级而非逐像素**是省显存关键：逐像素 512 维张量 (H,W,512) 巨大，掩码级只存每个物体的一个向量。

## 4. 分层掩码（LangSplat 思想，缓解边界歧义）

```python
# 用不同 points_per_side / pred_iou_thresh 跑两档，给每个掩码打 hier_level
mask_gen_coarse = SamAutomaticMaskGenerator(sam, points_per_side=8)    # 大物体/背景
mask_gen_fine   = SamAutomaticMaskGenerator(sam, points_per_side=32)   # 小物体/细节
```

每像素维护「掩码 ID → 语义层级」归属表，Phase 6 反投影监督时按层级加权。

## 5. 场景级自编码器（512 → D 维）

```python
class LangAE(nn.Module):
    def __init__(self, d_in=512, d_latent=3):
        self.enc = MLP([512, 256, 64, 16, d_latent], act=nn.ReLU)      # 末层无激活
        self.dec = MLP([d_latent, 16, 64, 256, 512], act=nn.ReLU)      # 末层线性

def train_ae(vectors):   # vectors: (M, 512) 本场景所有掩码级 CLIP 特征
    v_hat = ae.dec(ae.enc(v))
    loss = (v_hat - v).abs().mean() + 0.5 * (1 - F.cosine_similarity(v_hat, v).mean())
    # 收敛判据：重建余弦相似度均值 > 0.95
    torch.save(ae.state_dict(), 'data/checkpoints/lang_ae.pt')
```

## 6. Jetson 优化

- MobileSAM/MobileCLIP 用 **FP16 CUDA 推理**，与 Tracking 分时复用 GPU，用完 `torch.cuda.empty_cache()`。
- 只对**关键帧**提取（非每帧），进一步省算力。
- **进阶（加分项）**：MobileCLIP 转 ONNX → TensorRT（INT8/FP16 engine）用 DLA 加速，延迟可降一个量级。

## 7. 验收清单

- [ ] 单帧能输出数十个有语义意义的掩码。
- [ ] 「椅子」文本与椅子掩码相似度明显高于背景（定量：> 0.3 余弦差）。
- [ ] `feature_map.pkl` 可被 Phase 6 反序列化。
- [ ] AE 重建余弦相似度均值 > 0.95。

## 8. 常见坑

1. **MobileSAM 掩码质量差**：`pred_iou_thresh` 太低/太高都影响，调参并做分层。
2. **MobileCLIP 输入尺寸**：必须走 `preprocess`（224×224 + ImageNet 归一化），否则相似度空间崩。
3. **文本/图像模型不一致**：query 用 MobileCLIP 文本编码，特征也必须同款 MobileCLIP。
4. **显存**：MobileSAM+CLIP 常驻会占 1GB+，与 Tracking 分时或用完释放。
