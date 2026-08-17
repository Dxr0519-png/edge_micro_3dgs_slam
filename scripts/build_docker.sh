#!/usr/bin/env bash
# 构建 Edge-3DGS-SLAM 边缘开发容器
# 前置：宿主机（Jetson）已安装 Docker + nvidia-container-toolkit
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 构建镜像 edge_3dgs_slam:humble (L4T + PyTorch-for-Jetson + ROS2 Humble) ..."
docker compose -f docker/docker-compose.yml build

echo "==> 构建完成。启动：./scripts/run_container.sh"
