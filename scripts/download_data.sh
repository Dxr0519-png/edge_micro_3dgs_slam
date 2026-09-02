#!/usr/bin/env bash
# 下载评估数据集（Replica 等）
# 注意：Replica 已下载至 third_party/SplaTAM/data/Replica/
#（SplaTAM bash_scripts/download_replica.sh 的目标位置，8 场景×2000 帧）
# 官方 zip（ETH 镜像）约 12.4 GB；加载器见 src/edge_3dgs_slam/dataset/replica.py
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Replica 数据已在 third_party/SplaTAM/data/Replica/（场景 office0-4/room0-2）"
echo "如需重下：cd third_party/SplaTAM && bash bash_scripts/download_replica.sh"
