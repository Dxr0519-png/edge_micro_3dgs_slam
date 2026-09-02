"""Phase 6 §5 查询服务：语义查询回调封装（懒加载 + 快照 + 结果填充）。

- 懒加载：首次请求时才加载 MobileCLIP（load_sam=False，省 SAM ~200MB）+ 场景 AE；
- 模型来源：SLAMBackend.snapshot_model()（含 features，锁内 numpy 快照）——
  查询重活全部在锁外，不阻塞 sync 回调（独立回调组，见 node.py）；
- 在线模式（无蒸馏）：backend.ensure_feature_dim() 零初始化特征 →
  查询返回低置信但不崩（文档诚实注明在线查询需先蒸馏，蒸馏离线验证于 Replica）。
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from edge_3dgs_slam.feature_factory import LangAE, load_feature_config, load_models
from edge_3dgs_slam.query.engine import query

def _repo_root() -> Path:
    """向上查找仓库根（含 config/ + ws_src/ 的目录）。

    ⚠️ symlink-install 下 __file__ 是 install/ 里的符号链接——parents 索引不可靠
    （实测解析到 ws_src/install/edge_3dgs_ros/），必须按目录特征查找。
    """
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "config").is_dir() and (p / "ws_src").is_dir():
            return p
        p = p.parent
    raise RuntimeError("找不到仓库根（config/ + ws_src/ 目录）")


_REPO_ROOT = _repo_root()


class QueryService:
    """封装文本查询服务的懒加载与执行（线程安全：互斥初始化）。"""

    def __init__(self, backend, feature_cfg_path: str | None = None,
                 ae_ckpt: str | None = None, d: int = 3,
                 default_top_k: int = 100, default_min_score: float = 0.0,
                 default_eps: float = 0.15, device: str = "cuda"):
        self._backend = backend
        self._lock = threading.Lock()
        self._feature_cfg_path = feature_cfg_path or str(_REPO_ROOT / "config/feature/mobilesam_clip.yaml")
        self._ae_ckpt = ae_ckpt or str(_REPO_ROOT / "data/checkpoints/lang_ae_replica.pt")
        self._d = d
        self._top_k = default_top_k
        self._min_score = default_min_score
        self._eps = default_eps
        self._device = device
        self._fm = None          # FeatureModels（CLIP 常驻，查询低频释放反增延迟）
        self._ae = None
        self._initialized = False
        self._mem_mb = 0.0

    def _ensure_models(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            cfg = load_feature_config(self._feature_cfg_path)
            self._fm = load_models(cfg, self._device, load_sam=False)   # 只需 CLIP+tokenizer
            self._ae = LangAE(d_in=512, d_latent=self._d,
                              hidden=cfg["feature"]["autoencoder"]["hidden"])
            self._ae.load_state_dict(
                __import__("torch").load(self._ae_ckpt, map_location=self._device,
                                         weights_only=True))
            self._ae.to(self._device).eval()
            self._initialized = True
            if self._device.startswith("cuda"):
                import torch
                self._mem_mb = torch.cuda.memory_allocated() / 1024 / 1024

    def handle(self, req, backend=None) -> dict:
        """执行查询，返回与 QueryResult 对齐的 dict（None 表示模型不可用）。

        ⚠️ backend 由调用方动态传入：节点 __init__ 时 backend 惰性为 None，
        之后 self._backend 重新赋值不会更新 QueryService 持有的旧引用（实测坑）。
        """
        self._ensure_models()
        backend = backend if backend is not None else self._backend
        if backend is None:
            return None                       # backend 惰性初始化：无数据时不可用
        snap = backend.snapshot_model()
        if snap is None:
            return None
        if "features" not in snap:
            # 在线模式：模型无特征通道 → 锁内补零初始化（低置信但不崩）
            if not backend.ensure_feature_dim(self._d):
                return None
            snap = backend.snapshot_model()
        # Query.srv 是嵌套结构：请求在 req.request（QueryRequest msg）里
        text = req.query if hasattr(req, "query") else req.request.query
        top_k = req.top_k if hasattr(req, "top_k") and req.top_k > 0 else self._top_k
        min_score = (req.min_score if hasattr(req, "min_score") and req.min_score > 0
                     else self._min_score)
        r = query(text, snap["means"], snap["features"], self._ae,
                  self._fm.clip, self._fm.tokenizer,
                  top_k=top_k, min_score=min_score,
                  eps=self._eps, device=self._device)
        return r
