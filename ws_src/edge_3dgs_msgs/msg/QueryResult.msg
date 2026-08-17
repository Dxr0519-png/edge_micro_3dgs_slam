# 自然语言三维查询结果（Phase 6）
QueryRequest request        # 原始查询（回显）

SemanticPoint[] points      # top-k 相关点（按 score 降序）

float32[3] bbox_center      # 3D BBox 中心 (x,y,z)
float32[3] bbox_extent      # 3D BBox 尺寸 (dx,dy,dz)
float32[9] bbox_rotation    # 3×3 旋转矩阵（按行展开）

float32 confidence          # 查询置信度（如 top-k 平均 score）
