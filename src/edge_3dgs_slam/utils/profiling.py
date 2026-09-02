"""Phase 3 §1 性能剖析工具：CUDA 同步计时 + 显存打点。

Jetson 是统一内存，`free -h` 与 `nvidia-smi` 口径不同，一律以
`torch.cuda.memory_allocated()` 为准（见 docs/03 §9 坑 4）。
"""
from __future__ import annotations

import time

import torch


def profiled(fn, *a, **kw):
    """执行 fn 并返回 (结果, 耗时 ms)。

    前后 torch.cuda.synchronize()，确保 CPU 墙钟即 GPU 真实耗时
    （异步 launch 不阻塞，不加同步会量出虚高的 FPS）。
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    torch.cuda.synchronize()
    return r, (time.perf_counter() - t0) * 1e3


def alloc_mb() -> float:
    """当前已分配（含缓存复用）显存，MB。"""
    return torch.cuda.memory_allocated() / 1e6


def peak_mb() -> float:
    """本进程峰值已分配显存（reset_peak 后重新累计），MB。"""
    return torch.cuda.max_memory_allocated() / 1e6


def reserved_mb() -> float:
    """缓存池保留显存（含未用缓存），MB。"""
    return torch.cuda.memory_reserved() / 1e6


def reset_peak():
    """重置峰值统计。消融行之间必须调用（配合 empty_cache 防内存池串扰）。"""
    torch.cuda.reset_peak_memory_stats()
