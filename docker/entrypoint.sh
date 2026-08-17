#!/usr/bin/env bash
set -e

# source ROS 2 环境
if [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

# source 工作空间（若已 colcon build）
if [ -f /workspace/install/setup.bash ]; then
  # shellcheck disable=SC1091
  source /workspace/install/setup.bash
fi

exec "$@"
