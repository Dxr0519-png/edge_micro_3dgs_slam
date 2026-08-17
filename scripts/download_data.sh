#!/usr/bin/env bash
# 下载评估数据集（Replica 等）到 data/raw
# 具体 URL 与组织方式在 Phase 2 实现时填充
set -euo pipefail
cd "$(dirname "$0")/.."

echo "TODO(Phase 2): 在此下载 Replica 数据集（自带 RGB-D + 真值位姿 + 内参）"
echo "  数据放入 data/raw/replica/ 并组织成 rgb/ depth/ pose/ intrinsics 结构"
