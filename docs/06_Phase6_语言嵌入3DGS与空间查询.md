# 06 · Phase 6 Language-Embedded 3DGS 与空间查询

> 目标：高斯新增低维特征维度，改写可微渲染使语言特征可 splat，与 Phase 5 特征图算 Loss 优化；开发文本查询节点发布 3D BBox / 热力图。
> 依赖：Phase 4（ROS2 封装）+ Phase 5（特征工厂）。

## 1. 数据结构改造（`gaussian/model.py`）

```python
self._feature = nn.Parameter(torch.zeros(N, D, device='cuda'))   # D=3（可配 16）
# save_ply / load_ply 增加 feature 通道读写
# 语言特征优化阶段冻结几何：xyz/rot/scale/opacity/color.requires_grad_(False)
```

## 2. 特征光栅化（核心难点）

**降级方案先行**（`gaussian/torch_feature_rasterizer.py`）：纯 PyTorch 慢速 splat。

```python
def rasterize_feature(gaussians, T_wc, K, H, W):
    # 与 RGB 渲染共享同一套投影/深度排序/α 权重，把颜色通道换成 D 维特征
    # F_2d(pixel) = Σ_i f_i * α_i * Π_{j<i}(1-α_j)   -> (D, H, W)
    ...
```

验证正确性后，再改 CUDA kernel（参考 LangSplat `language-gaussian-rasterization`）：
1. `preprocess`：把 D 维 `feature` 与颜色一起分配进 tile。
2. `render`：累加特征 `F` 通道。
3. `backward`：特征梯度反传回 `feature` 与共享量。

互验：慢速 torch splat 与 CUDA 版特征图误差 < 1e-3。

## 3. 损失与优化（`language_field/optim.py`）

```python
F_2d = rasterize_feature(gaussians, pose)        # (D,H,W)
V_2d = ae.dec(F_2d.permute(1,2,0))               # (H,W,512)
# 掩码级监督：对每个 MobileSAM 掩码区域，取 V_2d 均值与该掩码 clip_vec 对齐
loss = L1(V_2d[mask], v_gt) + λ_cos * (1 - cos(V_2d[mask], v_gt))
# 只优化 gaussians.feature（几何冻结）
```

## 4. 查询引擎（`query/engine.py`）

```python
def query(text, gaussians, ae, clip, top_k=100):
    t = clip.encode_text(tokenizer([text]))          # (1,512) L2 归一化
    V = ae.dec(gaussians.feature)                    # (N,512) L2 归一化
    relevance = V @ t.T                              # (N,) 余弦
    idx = torch.topk(relevance, top_k).indices
    pts = gaussians.xyz[idx].cpu().numpy()
    labels = DBSCAN(eps=0.05).fit_predict(pts)       # 聚类去离群
    bbox = aabb_or_obb(pts[labels == major_cluster])
    return {'query': text, 'bbox': bbox, 'confidence': relevance[idx].mean(), 'points': ...}
```

热力图：把 `relevance` 映射 viridis，对 top-k 高斯中心用当前位姿前向渲染输出高亮图。

## 5. 查询服务（Phase 4 节点扩展）

```python
# 在 Edge3DGSSlamNode 中加 service（Query.srv: QueryRequest -> QueryResult）
self.create_service(Query, '/semantic_query/query', self._on_query)

def _on_query(self, req, resp):
    r = query(req.query, self.gaussians, self.ae, self.clip, req.top_k)
    resp.bbox_center = r['bbox'].center ...
    resp.confidence = r['confidence']
    return resp
```

> `Query.srv` 定义在 Phase 6 补充进 `edge_3dgs_msgs`（`srv/Query.srv`，用已有 `QueryRequest`/`QueryResult`）。

## 6. 端到端验证

```bash
ros2 run edge_3dgs_ros edge_3dgs_slam_node
ros2 service call /semantic_query/query edge_3dgs_msgs/srv/Query "{request: {query: '黑色的椅子', top_k: 100, min_score: 0.3}}"
# rviz2 中看高亮热力图 + 3D BBox Marker
```

## 7. 验收清单

- [ ] 渲染语言特征图解码后与 Phase 5 特征图在掩码内余弦相似度 > 0.85。
- [ ] 查询「黑色的椅子」bbox 落在目标附近；热力图正确高亮。
- [ ] 多物体（椅子/桌子）可区分。
- [ ] torch 慢速 splat 与 CUDA 版互验误差 < 1e-3。

## 8. 常见坑

1. **特征光栅化梯度回传错误**：先慢速 torch splat 验证数值梯度，再上 CUDA。
2. **语言优化破坏几何**：务必冻结几何（或 1e-5 极小学习率）。
3. **AE 解码漂移**：AE 训练收敛（余弦 > 0.95）后再蒸馏进高斯，否则查询不聚合。
4. **DBSCAN 聚类参数**：`eps` 要按场景尺度调（米级），否则 bbox 碎裂或粘连。
