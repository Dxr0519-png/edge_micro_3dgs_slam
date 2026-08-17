# syntax=docker/dockerfile:1.4
# =====================================================================
# Edge-3DGS-SLAM 边缘开发容器（aarch64 / L4T）
#   Jetson Orin NX Super（16GB，JetPack 6.2 / L4T R36.4，Ubuntu 22.04）
#   PyTorch for Jetson + ROS2 Humble + Librealsense(RSUSB) + 3DGS/SLAM 依赖
#
# 关键点（与桌面 x86 完全不同）：
#   1. 基础镜像必须 arm64：用 nvcr.io 的 l4t-pytorch（自带 PyTorch-for-Jetson），
#      不能用 osrf/ros:humble-desktop-full（无 arm64）。
#   2. CUDA 工具链由宿主机 JetPack 经 nvidia-container-toolkit 挂载，容器内不装驱动。
#   3. 所有 CUDA 扩展（diff-gaussian-rasterization 等）按 sm_87 编译：
#      TORCH_CUDA_ARCH_LIST="8.7"
#   4. PyTorch 必须来自 NGC 的 Jetson aarch64 构建，不能用 pypi 默认 torch（x86）。
#
# 构建:  docker compose build   （或 ./scripts/build_docker.sh）
# =====================================================================

# l4t-pytorch 自带匹配 JetPack 的 PyTorch-for-Jetson（torch+torchvision 已装好）。
# ⚠️ 下面的 tag 是「占位代表值」，并非一定存在 —— 构建前务必核对实际存在的精确 tag！
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
# 例：JetPack 6.2 / L4T 36.4 -> r36.4.0-pth2.6.0-py3.10（pth 小版本以 NGC 实际为准）
# 核对后回填下面 ARG，避免 docker pull 失败。
ARG L4T_TAG=r36.4.0-pth2.6.0-py3.10
FROM nvcr.io/nvidia/l4t-pytorch:${L4T_TAG}

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
# 让 CUDA 扩展按 Jetson Orin（Ampere sm_87）编译
ENV TORCH_CUDA_ARCH_LIST="8.7"
ENV CUDA_HOME=/usr/local/cuda

# ---------- 1. 基础工具 + 编译依赖 ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget vim tmux htop ca-certificates gnupg lsb-release software-properties-common \
        build-essential cmake pkg-config ninja-build \
        python3-pip python3-dev \
        libeigen3-dev libopencv-dev libboost-all-dev \
        libgl1-mesa-glx libglib2.0-0 \
        libusb-1.0-0-dev libglfw3-dev libgtk-3-dev libssl-dev \
        && rm -rf /var/lib/apt/lists/*

# ---------- 2. ROS 2 Humble（arm64 apt） ----------
# 注：osrf/ros 镜像无 arm64，故在 L4T 基础上手动加 ROS2 apt 源
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-image-transport \
        ros-humble-image-transport-plugins \
        ros-humble-pcl-ros \
        ros-humble-tf2-ros \
        ros-humble-cv-bridge \
        ros-humble-rosbag2 \
        ros-humble-rosbag2-storage-default-plugins \
        && rm -rf /var/lib/apt/lists/*

# ---------- 3. Librealsense SDK（源码编译，RSUSB 后端） ----------
# Jetson 无官方 arm64 deb；RSUSB 用户态后端免内核 patch，D435i 即插即用
RUN cd /opt && \
    git clone --depth 1 https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    mkdir build && cd build && \
    cmake .. \
        -DFORCE_RSUSB_BACKEND=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_GRAPHICAL_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /opt/librealsense

# ---------- 4. realsense2_camera（ROS2 wrapper，源码编译） ----------
RUN mkdir -p /ros2_ws/src && \
    cd /ros2_ws/src && \
    git clone --depth 1 -b ros2-development https://github.com/IntelRealSense/realsense-ros.git && \
    . /opt/ros/humble/setup.sh && \
    cd /ros2_ws && \
    colcon build --symlink-install --packages-select realsense2_camera_msgs realsense2_camera \
        && rm -rf /ros2_ws/src /ros2_ws/build

# ---------- 5. Python：数值/图像 + 3DGS/SLAM 依赖 ----------
# numpy 锁 <2 避免与 cv2 / ROS cv_bridge ABI 冲突
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --ignore-installed "numpy<2" sympy mpmath
RUN pip3 install \
        opencv-python plyfile scipy trimesh open3d munch pyyaml tqdm rich \
        matplotlib evo scikit-learn imageio[ffmpeg]

# ---------- 6. 开放词汇（Phase 5/6，边缘轻量） ----------
RUN pip3 install open-clip-torch segment-anything ultralytics
# MobileSAM：从官方权重加载（在 Phase 5 实现时于 data/checkpoints 缓存 .pt）
# MobileCLIP：open_clip.create_model('MobileCLIP-S0', pretrained='...') 直接可用

# ---------- 7. CUDA 光栅化扩展说明（构建后进入容器执行，见 IMPLEMENTATION_PLAN Phase 2） ----------
# SplaTAM 的 diff-gaussian-rasterization（带 depth 渲染）需针对 sm_87 编译：
#   cd third_party/SplaTAM
#   pip install submodules/diff-gaussian-rasterization
# 注意 setup.py 不得硬编码 x86；若编译失败，降级用纯 PyTorch 慢速 splat（见 Phase 2.5）

# ---------- 8. 工作空间 ----------
WORKDIR /workspace

# ---------- 9. 环境与入口 ----------
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'source /workspace/install/setup.bash 2>/dev/null' >> /root/.bashrc

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
