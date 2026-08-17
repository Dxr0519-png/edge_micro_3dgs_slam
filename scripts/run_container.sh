#!/usr/bin/env bash
# 启动容器并进入交互式 shell
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose -f docker/docker-compose.yml up -d
echo "==> 进入容器（验证：python -c \"import torch; print(torch.cuda.get_device_capability())\" 应输出 (8, 7)）"
docker compose -f docker/docker-compose.yml exec edge_3dgs_slam bash
