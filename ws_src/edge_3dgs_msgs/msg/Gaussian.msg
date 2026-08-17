# 单个 3D 高斯基元（用于把高斯场发布到 ROS2 供可视化/查询）
float32 x
float32 y
float32 z             # 中心（世界系）

float32 opacity       # 不透明度（经 sigmoid 激活后 ∈ [0,1]）

float32 scale_x
float32 scale_y
float32 scale_z       # 每轴尺度

float32 qx
float32 qy
float32 qz
float32 qw            # 旋转四元数 (x,y,z,w)

float32 r
float32 g
float32 b             # 颜色（0 阶球谐 / 单色）

float32[] feature     # 语言特征潜变量（Phase 6 启用，可变长度 D 维）
