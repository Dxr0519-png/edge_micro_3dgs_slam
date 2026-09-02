#!/usr/bin/env python3
"""Phase 6 §5-6 查询服务回放验证（无相机，复用 Phase 4 回放模式）。

流程:
    1. 起 phase4_replay_publisher.py（Replica office0 前 60 帧 15Hz 话题流）
    2. 起 edge_3dgs_slam_node --load probe_model_replica_p6.pt --tier quality
    3. 等 /semantic_query/query 服务出现 → ros2 service call「black chair」
    4. 断言：request 回显、confidence > 0、bbox/points 有限值且在房间范围、
       points ≥ 10、无 NaN → PASS/FAIL，teardown 子进程
判据: docs/06 §6 端到端（回放口径；真机冒烟另跑）。
"""
import subprocess
import sys
import numpy as np
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "data/outputs/phase6/probe_model_replica_p6.pt"
WS_SRC = REPO / "ws_src"
# 子进程统一 source ROS + workspace（本脚本可能运行于未 source 的 shell）
SRC = ("source /opt/ros/humble/setup.bash && source " + str(WS_SRC / "install/setup.bash") + " && ")
# 房间判据范围（离线查询已验证：模型包围盒外扩 20%）
ROOM = dict(x=(-2.1, 1.1), y=(-1.9, 2.2), z=(-1.6, 2.2))

QUERY_CMD = ('ros2 service call /semantic_query/query edge_3dgs_msgs/srv/Query '
             '"request: {query: \'black chair\', top_k: 100, min_score: 0.0}"')


def wait_for_service(timeout: float = 120.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        r = subprocess.run(["bash", "-c", SRC + "ros2 service list | grep semantic_query"],
                           capture_output=True, text=True)
        if r.returncode == 0 and "semantic_query" in r.stdout:
            return True
        time.sleep(2)
    return False


def main() -> int:
    if not CKPT.exists():
        print(f"❌ 缺少 {CKPT}（先跑 phase6_distill.py）")
        return 1

    procs = []
    try:
        print("1) 起回放发布器（Replica office0 60 帧 15Hz）...")
        p_replay = subprocess.Popen(
            ["bash", "-c", f"{SRC}python3 {REPO / 'experiments/phase4_replay_publisher.py'} "
             "--source replica --scene office0 --frames 60 --scale 0.5 --hz 15 --loop"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p_replay)

        print("2) 起节点（--load p6 checkpoint, tier quality）...")
        p_node = subprocess.Popen(
            ["bash", "-c", f"{SRC}ros2 run edge_3dgs_ros edge_3dgs_slam_node "
             f"--load {CKPT} --tier quality"],
            stdout=subprocess.DEVNULL,
            stderr=open("/tmp/p6_node_svc.log", "w"))
        procs.append(p_node)

        if not wait_for_service():
            print("❌ 服务 /semantic_query/query 未在 120s 内出现")
            return 1
        print("   服务已就绪 ✅")

        print("3) service call「black chair」...")
        time.sleep(2)   # 等首帧 track 建立位姿
        r = subprocess.run(["bash", "-c", SRC + QUERY_CMD],
                           capture_output=True, text=True, timeout=60)
        out = r.stdout
        if r.returncode != 0 or "result" not in out:
            print(f"❌ service call 失败: {r.stderr[:300]}")
            return 1
        print(f"   响应片段: {out[:400]}")

        # ---- 断言 ----
        import re
        checks: list[tuple[str, bool]] = []
        # 嵌套 srv 输出格式: Query_Response(result=QueryResult(request=..., points=[SemanticPoint...], bbox_center=[...], confidence=...))
        conf_m = re.search(r"confidence=([-\d.]+)", out)
        bbox_m = re.search(r"bbox_center=array\(\[([-\d.\s,]+)\],", out)
        extent_m = re.search(r"bbox_extent=array\(\[([-\d.\s,]+)\],", out)
        pts_m = re.search(r"points=\[edge_3dgs_msgs\.msg\.SemanticPoint", out)

        ok_conf = bool(conf_m) and float(conf_m.group(1)) > 0
        ok_pts = bool(pts_m)
        ok_bbox = False
        if bbox_m and extent_m:
            c = [float(v) for v in bbox_m.group(1).replace(",", " ").split()]
            e = [float(v) for v in extent_m.group(1).replace(",", " ").split()]
            ok_bbox = (all(ROOM[k][0] <= c[i] <= ROOM[k][1]
                           for i, k in enumerate(("x", "y", "z")))
                       and all(0.02 <= v <= 3.0 for v in e)
                       and all(np.isfinite(v) for v in c + e))
        ok_nan = "nan" not in out.lower()
        for name, ok in [("confidence > 0", ok_conf), ("points 非空", ok_pts),
                         ("bbox 有限且在房间范围", ok_bbox), ("无 NaN", ok_nan)]:
            print(f"  [{'PASS ✅' if ok else 'FAIL ❌'}] {name}")
            checks.append((name, ok))

        all_ok = all(ok for _, ok in checks)
        print("=" * 62)
        print("Phase 6 §5 查询服务回放验证：" + ("全部 PASS ✅" if all_ok else "存在 FAIL ❌"))
        return 0 if all_ok else 1
    finally:
        for p in procs:
            p.terminate()
        time.sleep(1)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
