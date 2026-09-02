cmake_minimum_required(VERSION 3.8)
project(edge_3dgs_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ament_python 包：源码与安装由 setup.py 负责（Phase 4 已实现）
find_package(ament_cmake_python REQUIRED)

# ⚠️ 勿启用下面两行：ament_python 的安装由 setup.py（data_files + console_scripts）
# 完成，这里再 install() 会与 setup.py 双安装冲突。
# install(DIRECTORY edge_3dgs_ros DESTINATION lib/${PROJECT_NAME})
# install(PROGRAMS scripts/edge_3dgs_slam_node.py DESTINATION lib/${PROJECT_NAME})

ament_package()
