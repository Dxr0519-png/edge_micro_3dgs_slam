cmake_minimum_required(VERSION 3.8)
project(edge_3dgs_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ament_python 包：实际节点源码 + setup.py 在 Phase 4 实现时补充
find_package(ament_cmake_python REQUIRED)

# Phase 4 时取消注释，安装节点源码与入口脚本：
# install(DIRECTORY edge_3dgs_ros DESTINATION lib/${PROJECT_NAME})
# install(PROGRAMS scripts/edge_3dgs_slam_node.py DESTINATION lib/${PROJECT_NAME})

ament_package()
