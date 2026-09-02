"""ament_python 打包配置（Phase 4 实现时补充，见 docs/04 §2）。"""
from setuptools import setup

package_name = "edge_3dgs_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Edge-3DGS-SLAM",
    maintainer_email="you@example.com",
    description="Edge-3DGS-SLAM 的 ROS2 节点（Tracking/Mapping + 位姿/高斯点云发布）",
    license="MIT",
    entry_points={
        "console_scripts": [
            # ros2 run edge_3dgs_ros edge_3dgs_slam_node（docs/04 §5）
            "edge_3dgs_slam_node = edge_3dgs_ros.node:main",
        ],
    },
)
