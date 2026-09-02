"""GaussianModel：可微高斯点云模型（裁剪自 SplaTAM，参数组织与其 params dict 一致）。

- 参数全在 CUDA 上（Jetson aarch64），以 torch.nn.Parameter 管理，key 与 SplaTAM 相同，
  便于对照其 `utils/slam_external.py` 的增删剪枝逻辑。
- `create_from_points` 提供文档 §4 的初始化入口（深度反投影 → 高斯）。
- 导出 PLY 为通用 3DGS 字段（f_dc / opacity / scale / rot），可被 CloudCompare/Open3D 打开。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# 参数 key（与 SplaTAM 一致）
PARAM_KEYS = ["means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales"]
# Phase 6 §1 语言特征潜变量 key：不进 PARAM_KEYS（几何语义保持 5 键，SLAM 热路径零影响），
# 模型含此键时 add/remove/导出同步处理；蒸馏阶段冻结几何，此键为唯一可训练参数。
FEATURE_KEY = "features"
# 点相关的变量 key
VAR_POINT_KEYS = ["max_2D_radius", "means2D_gradient_accum", "denom", "timestep"]
# Phase 3 §3 FP16 存储：只降这 3 个（颜色/透明度/尺度），几何（means3D/rotations）恒 FP32
FP16_KEYS = ["rgb_colors", "logit_opacities", "log_scales"]


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


class GaussianModel:
    def __init__(self, params: dict, variables: dict | None = None):
        self.params = params
        self.variables = variables if variables is not None else {}

    # ------------------------------------------------------------------ 创建
    @classmethod
    def create_from_points(cls, pts: np.ndarray | torch.Tensor,
                           colors: np.ndarray | torch.Tensor,
                           scales: np.ndarray | torch.Tensor | None = None,
                           opacity: float = 0.5,
                           anisotropic: bool = False,
                           features: np.ndarray | torch.Tensor | None = None) -> "GaussianModel":
        """从世界系点云初始化高斯。

        参数:
            pts:    (N, 3) 世界系坐标
            colors: (N, 3) RGB，0~1
            scales: (N,) 或 (N, 1) 或 (N, 3) 各向尺度（米）；None 时用 0.01 米小值
            opacity: 初始不透明度（sigmoid 后）
            anisotropic: True 时每轴独立 scale（(N, 3)）
            features: (N, D) 语言特征潜变量（Phase 6 §1）；None（默认）时不创建
                      feature 通道——Phase 2-5 调用方与显存占用完全不变
        """
        device = torch.device("cuda")
        # as_tensor 统一处理 numpy / torch（CUDA）输入；np.asarray 对 cuda tensor 会报错
        pts = torch.as_tensor(pts, dtype=torch.float32, device=device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=device)
        n = pts.shape[0]
        if scales is None:
            log_scales = torch.full((n, 3 if anisotropic else 1), float(np.log(0.01)),
                                    dtype=torch.float32, device=device)
        else:
            s = torch.as_tensor(scales, dtype=torch.float32, device=device).reshape(n, -1)
            if not anisotropic:
                s = s[:, :1]
            log_scales = torch.log(torch.clamp(s, min=1e-6))

        params = {
            "means3D": torch.nn.Parameter(pts.contiguous()),
            "rgb_colors": torch.nn.Parameter(colors.contiguous()),
            "unnorm_rotations": torch.nn.Parameter(
                torch.tile(torch.tensor([1., 0., 0., 0.], device=device), (n, 1))),
            "logit_opacities": torch.nn.Parameter(
                torch.full((n, 1), float(inverse_sigmoid(torch.tensor(opacity))), device=device)),
            "log_scales": torch.nn.Parameter(log_scales.contiguous()),
        }
        if features is not None:
            params[FEATURE_KEY] = torch.nn.Parameter(
                torch.as_tensor(features, dtype=torch.float32, device=device).contiguous())
        variables = {k: torch.zeros(n, dtype=torch.float32, device=device) for k in VAR_POINT_KEYS}
        return cls(params, variables)

    # ------------------------------------------------------------------ 查询
    @property
    def means3D(self) -> torch.Tensor:
        return self.params["means3D"]

    @property
    def num_gaussians(self) -> int:
        return self.params["means3D"].shape[0]

    @property
    def is_isotropic(self) -> bool:
        return self.params["log_scales"].shape[1] == 1

    def opacities(self) -> torch.Tensor:
        """sigmoid 后的不透明度 (N, 1)。"""
        return torch.sigmoid(self.params["logit_opacities"])

    def scales(self) -> torch.Tensor:
        """exp 后的尺度；isotropic 时铺成 (N, 3)。"""
        s = torch.exp(self.params["log_scales"])
        if s.shape[1] == 1:
            s = s.tile(1, 3)
        return s

    def rotations(self) -> torch.Tensor:
        return F.normalize(self.params["unnorm_rotations"])

    # ------------------------------------------------------------------ 增删
    def add_gaussians(self, pts, colors, scales=None):
        """追加新高斯（文档 §6 第 1 步：无高斯覆盖像素处反投影新增）。

        Phase 6 §1：模型含 feature 通道时同步追加零初始化特征（(new_n, D)）——
        特征在蒸馏阶段学习，新增高斯先用 0（语义中性）再蒸馏填充。
        """
        new = GaussianModel.create_from_points(pts, colors, scales, opacity=0.5,
                                               anisotropic=not self.is_isotropic)
        for k in PARAM_KEYS:
            # Phase 3 §3：FP16 存储下必须对齐 dtype（torch.cat 混合 dtype 直接报错）
            self.params[k] = torch.nn.Parameter(
                torch.cat([self.params[k], new.params[k].to(self.params[k].dtype)], dim=0))
        if FEATURE_KEY in self.params:
            self.params[FEATURE_KEY] = torch.nn.Parameter(
                torch.cat([self.params[FEATURE_KEY],
                           torch.zeros(new.num_gaussians, self.params[FEATURE_KEY].shape[1],
                                       dtype=self.params[FEATURE_KEY].dtype,
                                       device=self.params[FEATURE_KEY].device)]))
        n = self.num_gaussians
        for k in VAR_POINT_KEYS:
            self.variables[k] = torch.cat([self.variables[k], new.variables[k]])
        return n

    def remove(self, to_remove: torch.Tensor):
        """按 bool 掩码删除高斯（True 的被删），并同步裁剪优化器无关的变量。

        Phase 6 §1：模型含 feature 通道时同步裁剪（蒸馏/查询与几何共用同一套索引）。"""
        keep = ~to_remove
        for k in PARAM_KEYS:
            self.params[k] = torch.nn.Parameter(self.params[k][keep])
        if FEATURE_KEY in self.params:
            self.params[FEATURE_KEY] = torch.nn.Parameter(self.params[FEATURE_KEY][keep])
        for k in VAR_POINT_KEYS:
            self.variables[k] = self.variables[k][keep]
        return self.num_gaussians

    # ------------------------------------------------------------------ Phase 3 §2/§3 增删与精度
    def prune_points(self, keep: torch.Tensor) -> int:
        """按 keep 布尔掩码保留（True 保留），与 remove(to_remove) 互补（文档 §2 的 prune_points）。"""
        return self.remove(~keep)

    def keep_indices(self, keep_idx: torch.Tensor) -> int:
        """按索引保留指定高斯（enforce_capacity 的 topk 淘汰用，文档 §2 的 prune_points_keep）。"""
        keep = torch.zeros(self.num_gaussians, dtype=torch.bool, device=self.means3D.device)
        keep[keep_idx] = True
        return self.remove(~keep)

    def half_storage(self) -> "GaussianModel":
        """存储层半精度（Phase 3 §3）：rgb_colors/logit_opacities/log_scales → FP16。

        means3D/unnorm_rotations 恒 FP32——几何精度一旦降级会导致累积漂移（docs/03 §9 坑 2）。
        计算层在 render 边界统一 .float()（CUDA kernel 内部 float），上游对 dtype 无感。
        """
        for k in FP16_KEYS:
            self.params[k] = torch.nn.Parameter(self.params[k].data.half())
        return self

    def float_storage(self) -> "GaussianModel":
        """恢复 FP32 存储（half_storage 的逆操作）。"""
        for k in FP16_KEYS:
            self.params[k] = torch.nn.Parameter(self.params[k].data.float())
        return self

    @property
    def is_half_storage(self) -> bool:
        return self.params["rgb_colors"].dtype == torch.float16

    def prune(self, opacity_threshold: float = 0.005, big_scale: float | None = None):
        """剪枝：opacity 低于阈值；big_scale 给定时一并删除尺度超限者。"""
        to_remove = (self.opacities() < opacity_threshold).squeeze()
        if big_scale is not None:
            to_remove = to_remove | (self.scales().max(dim=1).values > big_scale)
        return self.remove(to_remove)

    # ------------------------------------------------------------------ 导出
    def save_ply(self, path: str):
        """导出 3DGS 格式 PLY（ASCII）。字段与 inria 官方一致，Open3D/CloudCompare 可读。

        Phase 6 §1：模型含 feature 通道时，header 追加 `f_feature_0..{D-1}` 列
        （D 维语言潜变量），load_ply 按属性名解析回读。
        """
        with torch.no_grad():
            xyz = self.means3D.detach().float().cpu().numpy()        # .float()：兼容 FP16 存储（§3）
            rgb = np.clip(self.params["rgb_colors"].detach().float().cpu().numpy(), 0, 1)
            op = self.opacities().detach().float().cpu().numpy()
            scale = self.scales().detach().float().cpu().numpy()    # 统一铺成 (N, 3)
            rot = self.rotations().detach().float().cpu().numpy()
            feat = (self.params[FEATURE_KEY].detach().float().cpu().numpy()
                    if FEATURE_KEY in self.params else None)
        n = xyz.shape[0]
        d = feat.shape[1] if feat is not None else 0
        header = ["ply", "format ascii 1.0",
                  f"element vertex {n}",
                  "property float x", "property float y", "property float z",
                  "property float nx", "property float ny", "property float nz",
                  "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
                  "property float opacity",
                  "property float scale_0", "property float scale_1", "property float scale_2",
                  "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3"]
        for j in range(d):
            header.append(f"property float f_feature_{j}")
        header.append("end_header")
        lines = []
        for i in range(n):
            row = (f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                   f"0 0 0 "
                   f"{rgb[i,0]:.5f} {rgb[i,1]:.5f} {rgb[i,2]:.5f} "
                   f"{op[i,0]:.6f} "
                   f"{scale[i,0]:.6f} {scale[i,1]:.6f} {scale[i,2]:.6f} "
                   f"{rot[i,0]:.6f} {rot[i,1]:.6f} {rot[i,2]:.6f} {rot[i,3]:.6f}")
            if feat is not None:
                row += " " + " ".join(f"{v:.6f}" for v in feat[i])
            lines.append(row)
        with open(path, "w") as f:
            f.write("\n".join(header) + "\n" + "\n".join(lines) + "\n")
        return path

    def __len__(self):
        return self.num_gaussians


# ------------------------------------------------------------------ Phase 6 §1 模块级工具（特征通道 / 几何冻结 / PLY 回读）

def add_feature_dim(model: GaussianModel, d: int) -> GaussianModel:
    """给模型加 D 维语言特征潜变量通道（Phase 6 §1 入口）。

    幂等：已有 "features" 且维度一致时直接返回。零初始化 (N, D) FP32——
    特征是被蒸馏优化的量，不进 FP16_KEYS（与几何恒 FP32 同理）。
    D 来源由调用方传 config["feature"]["autoencoder"]["latent_dim"]
    （本模块不 import feature_factory，保持 slam 与 feature_factory 无依赖）。
    """
    if FEATURE_KEY in model.params:
        if model.params[FEATURE_KEY].shape[1] == d:
            return model
        raise ValueError(f"模型已有 {FEATURE_KEY} 维度 {model.params[FEATURE_KEY].shape[1]}，"
                         f"与请求的 {d} 不一致（需重建或蒸馏前对齐）")
    n = model.num_gaussians
    model.params[FEATURE_KEY] = torch.nn.Parameter(
        torch.zeros(n, d, dtype=torch.float32, device=model.means3D.device))
    return model


def freeze_geometry(model: GaussianModel, freeze: bool = True) -> None:
    """冻结/解冻几何与外观参数（Phase 6 §1：语言特征优化阶段几何冻结）。

    freeze=True 时：means3D/rgb_colors/unnorm_rotations/logit_opacities/log_scales
    全部 requires_grad_(False)，features 恒为唯一可训练参数——防止语言蒸馏
    破坏已收敛几何（文档 §8 坑 2）。freeze=False 恢复全参数可训练（如后续重建）。
    """
    for k in PARAM_KEYS:
        model.params[k].requires_grad_(not freeze)
    if FEATURE_KEY in model.params:
        model.params[FEATURE_KEY].requires_grad_(True)


def load_ply(path: str) -> GaussianModel:
    """读回 save_ply 导出的 3DGS PLY（Phase 6 §1：save_ply 的逆操作）。

    从 header 属性名统计 `f_feature_*` 个数得 D（0 = 无特征通道）；按 header
    顺序映射数值列：xyz/normal/f_dc(3)/opacity/scale(3)/rot(4)/f_feature(D)。
    重建 params dict（与 checkpoint 同构）：
    - logit_opacities 经 inverse_sigmoid(clip(op, 1e-6, 1-1e-6)) 还原；
    - log_scales = log(clip(scale, 1e-6))（保存的是 exp 后尺度，统一铺成 (N,3)）；
    - unnorm_rotations 直接装载（PLY 存归一化四元数，rotations() 会再归一化）。
    """
    with open(path) as f:
        header_lines = []
        n_vertex = None
        for line in f:
            line = line.strip()
            header_lines.append(line)
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            if line == "end_header":
                break
    props = [l.split()[2] for l in header_lines if l.startswith("property")]
    d = sum(1 for p in props if p.startswith("f_feature_"))
    n = int(n_vertex)
    data = np.loadtxt(path, skiprows=len(header_lines))
    data = np.asarray(data, np.float64).reshape(n, len(props))
    def col(name):
        return data[:, props.index(name)]
    means3D = np.stack([col("x"), col("y"), col("z")], axis=-1).astype(np.float32)
    rgb = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=-1).astype(np.float32)
    op = np.clip(col("opacity"), 1e-6, 1 - 1e-6).astype(np.float32)
    scale = np.clip(np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=-1),
                    1e-6, None).astype(np.float32)
    rot = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")],
                   axis=-1).astype(np.float32)
    model = GaussianModel.create_from_points(means3D, np.clip(rgb, 0, 1), scale,
                                             opacity=0.5, anisotropic=True)
    dev = model.means3D.device
    model.params["logit_opacities"] = torch.nn.Parameter(
        torch.log(torch.as_tensor(op, device=dev) / (1 - torch.as_tensor(op, device=dev)))
        .reshape(-1, 1))                                   # (N,1)：与 PARAM_KEYS 其余一致
    model.params["unnorm_rotations"] = torch.nn.Parameter(torch.as_tensor(rot, device=dev))
    if d > 0:
        feat = np.stack([col(f"f_feature_{j}") for j in range(d)], axis=-1).astype(np.float32)
        model.params[FEATURE_KEY] = torch.nn.Parameter(torch.as_tensor(feat, device=dev))
    return model


# ------------------------------------------------------------------ Phase 3 §2 模块级工具
# 注意：与类方法 prune() 并存（方法版是 mapping 用的 opacity+big_scale 语义），
# 本模块函数版是文档 §2 的 opacity+scale_thresh 语义，两者都保留，docstring 互指。

def prune(gaussians: GaussianModel, opacity_thresh: float = 0.005,
          scale_thresh: float = 100.0) -> int:
    """文档 §2：keep = (opacity > thresh) & (scale.max < scale_thresh)，删除其余。"""
    keep = (gaussians.opacities().squeeze(-1) > opacity_thresh) \
        & (gaussians.scales().max(dim=-1).values < scale_thresh)
    return gaussians.prune_points(keep)


def enforce_capacity(gaussians: GaussianModel, N_max: int = 200_000) -> int:
    """文档 §2：超上限按 opacity topk 保留前 N_max（淘汰最低 opacity/贡献）。

    ⚠️ 时序约束：只能在属性优化循环**结束后**调用（此时优化器已释放）。
    迭代中途淘汰会与 Adam 状态索引错位（map_keyframe 每轮新建优化器，天然满足）。
    """
    if gaussians.num_gaussians <= N_max:
        return gaussians.num_gaussians
    score = gaussians.opacities().squeeze(-1).float()   # half 存储也转 float 再 topk
    keep_idx = torch.topk(score, N_max).indices
    return gaussians.keep_indices(keep_idx)


def already_covered(point, existing_xyz, r: float = 0.01) -> bool:
    """文档 §2 单点密度判据（O(N) 朴素遍历，仅测试/演示用；批量路径用 coverage_mask）。"""
    return bool(torch.any(torch.norm(existing_xyz - point, dim=-1) < r))


_NEIGH_OFFSETS = torch.tensor(
    [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=torch.long)
# coverage_mask 单段精确验证点数上限（cell 内密度 ≥ 此值时保守判覆盖，防 OOM）
_MAX_SEG = 512


def coverage_mask(points: torch.Tensor, existing_xyz: torch.Tensor,
                  r: float = 0.01) -> torch.Tensor:
    """批量密度判据：(M,3) 查询点 → bool(M,)，True = 半径 r 内已有近邻高斯（不新增）。

    栅格哈希 + 段内精确验证：cell 边长 r 的均匀网格（int64 打包坐标，无碰撞），
    查询点只与 27 邻域 cell 内的 existing 点做精确距离比较。

    精确性：距离 < r 的两点 cell 差每轴 ≤ 1，必然落在 27 邻域内 → 无漏判；
    段内精确距离保证无多报 → 与暴力法逐元素一致，代价 O(邻域点数) 而非 O(N·M)。
    """
    def _cells(xyz: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        cell = (xyz / r).floor().long() - off
        cell = torch.clamp(cell, 0, (1 << 20) - 1)
        return (cell[:, 0] << 40) | (cell[:, 1] << 20) | cell[:, 2]

    off = (existing_xyz / r).floor().long().min(dim=0).values
    ids_e = _cells(existing_xyz, off)
    order = torch.argsort(ids_e)
    sorted_ids = ids_e[order]

    M = points.shape[0]
    q_cell = (points / r).floor().long() - off
    offs = _NEIGH_OFFSETS.to(points.device)                # (27, 3)
    q_neigh = q_cell.unsqueeze(1) + offs.unsqueeze(0)      # (M, 27, 3)
    q_neigh = torch.clamp(q_neigh, 0, (1 << 20) - 1)
    qids = (q_neigh[:, :, 0] << 40) | (q_neigh[:, :, 1] << 20) | q_neigh[:, :, 2]  # (M, 27)

    # 每个 (查询点 i, 邻域 j) 的段 [lo, hi)：sorted_ids[lo:hi] 是该 cell 内的 existing 点
    lo = torch.searchsorted(sorted_ids, qids.reshape(-1), side="left").reshape(M, 27)
    hi = torch.searchsorted(sorted_ids, qids.reshape(-1), side="right").reshape(M, 27)
    seg_len = (hi - lo).reshape(-1)

    # 防爆保护：单段点数超上限（= 该 cell 内密度极高，重复堆积场景）不再做精确距离，
    # 直接保守判"已覆盖"（宁可少加高斯，不可重复堆积——文档 §2 密度判据意图）。
    # 否则 r 过大/密度过高时 repeat_interleave 物化百亿元素直接 OOM。
    heavy = seg_len > _MAX_SEG
    seg_len = seg_len.clamp(max=_MAX_SEG)
    total = int(seg_len.sum())
    hit = torch.zeros(M, dtype=torch.bool, device=points.device)
    if total > 0:
        seg_q = torch.arange(M * 27, device=points.device).repeat_interleave(seg_len)
        starts = torch.cumsum(seg_len, 0) - seg_len
        off_in = torch.arange(total, device=points.device) - starts.repeat_interleave(seg_len)
        pt_idx = order[lo.reshape(-1)[seg_q] + off_in]     # 段内 existing 点索引
        q_idx = seg_q // 27                                # 展开点所属查询点 i
        d = torch.norm(existing_xyz[pt_idx] - points[q_idx], dim=-1)
        hit.index_put_((q_idx[d < r],), torch.tensor(True, device=points.device))
    if heavy.any():
        heavy_pts = heavy.reshape(M, 27).any(dim=1)        # 任一段超限 → 保守判覆盖
        hit = hit | heavy_pts
    return hit
