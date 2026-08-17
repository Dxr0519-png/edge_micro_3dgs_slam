# 自然语言三维查询请求（Phase 6）
string query           # 自然语言指令，如 "黑色的椅子"

float32 min_score      # 相似度阈值（低于此值的高斯不返回）

int32 top_k            # 返回的 top-k 候选高斯数
