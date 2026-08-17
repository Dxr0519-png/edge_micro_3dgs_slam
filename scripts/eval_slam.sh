#!/usr/bin/env bash
# 轨迹评估（ATE/RPE）：用 evo 对比真值位姿与估计位姿
# 具体路径在 Phase 2/3 实现时填充
set -euo pipefail
cd "$(dirname "$0")/.."

echo "TODO(Phase 2): 用 evo_ape / evo_rpe 评估，示例："
echo "  evo_ape tum data/raw/replica/<scene>/traj.txt data/outputs/<run>/est_traj.txt -va --plot"
