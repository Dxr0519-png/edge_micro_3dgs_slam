"""Edge-3DGS-SLAM 核心算法库。

在 Jetson Orin NX 边缘板上运行的轻量级 RGB-D 3D Gaussian Splatting SLAM，
预留开放词汇 3D 语义场（Language-Embedded 3DGS）演进接口。

子模块：
    camera           D435i 数据读取 / 对齐 / 时间戳同步 / 内参
    slam             Tracking（位姿追踪）+ Mapping（高斯建图）核心
    gaussian         高斯模型 + 可微光栅化封装
    feature_factory  MobileSAM + MobileCLIP 特征工厂（Phase 5）
    language_field   语言特征高斯 + 自编码器 + 特征光栅化（Phase 6）
    query            自然语言三维查询引擎（Phase 6）
    utils            se(3)/评估/可视化等通用工具
"""

__version__ = "0.1.0"
