"""edge_3dgs_ros：Edge-3DGS-SLAM 的 ROS2 节点包（Phase 4）。

sys.path 引导：向上找含 src/edge_3dgs_slam 的仓库根并加入 import 路径，
使本包（colcon --symlink-install 源码态或 install 态）都能直接
`from edge_3dgs_slam import ...` 复用 Phase 1-3 管线。
"""
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    p = Path(__file__).resolve()
    for _ in range(16):      # install 态（site-packages）上溯到仓库根需 8 层，留余量
        if (p / "src" / "edge_3dgs_slam").is_dir():
            sys.path.insert(0, str(p / "src"))
            return
        p = p.parent


_add_src_to_path()
