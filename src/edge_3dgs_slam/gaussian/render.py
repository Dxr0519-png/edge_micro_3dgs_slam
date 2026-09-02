"""可微渲染：GaussianModel + 位姿 + 内参 → RGB/depth 图（裁剪自 SplaTAM）。

- 位姿约定：w2c（世界→相机）(4, 4)，与 SplaTAM 一致（详见 slam/se3.py）。
- depth 语义：相机系 z（米）。
- 每帧两次光栅化前向：RGB 一次；depth/silhouette/depth² 作为颜色通道一次
  （SplaTAM 的 depth+silhouette 渲染技巧）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .model import GaussianModel
from ..utils.se3 import quat_mult


def setup_camera(width: int, height: int, K, w2c=None,
                 near: float = 0.01, far: float = 100.0) -> GaussianRasterizationSettings:
    """构造光栅化设置。

    实测约定（本 fork 沿袭 OpenGL 列主序矩阵存储）：
    - means3D 输入为**相机系**坐标（调用方经 transform_to_frame 变换）；
    - projmatrix 传纯透视矩阵的转置视图，viewmatrix 传单位阵。
    - 与 SplaTAM 原版差异：原版把 w2c 乘进 projmatrix（viewmatrix.bmm(opengl_proj)），
      仅在首帧相对位姿为 I 时自洽（Replica 相对位姿掩盖了该问题）；本实现支持任意 w2c。
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    viewmatrix = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor(
        [[2 * fx / width, 0.0, -(width - 2 * cx) / width, 0.0],
         [0.0, 2 * fy / height, -(height - 2 * cy) / height, 0.0],
         [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
         [0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32, device="cuda").unsqueeze(0).transpose(1, 2)
    return GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=width / (2 * fx),
        tanfovy=height / (2 * fy),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=opengl_proj,
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
        prefiltered=False,
    )


def transform_to_frame(model: GaussianModel, w2c: torch.Tensor,
                       gaussians_grad: bool = True, camera_grad: bool = True):
    """把高斯从世界系变换到相机系（裁剪自 SplaTAM transform_to_frame）。

    返回 dict: means3D (N,3) 相机系、unnorm_rotations（anisotropic 时旋转到相机系）。
    """
    pts = model.params["means3D"] if gaussians_grad else model.params["means3D"].detach()
    unnorm_rots = model.params["unnorm_rotations"] if gaussians_grad \
        else model.params["unnorm_rotations"].detach()
    if not camera_grad:
        w2c = w2c.detach()

    pts_ones = torch.ones(pts.shape[0], 1, device="cuda")
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (w2c @ pts4.T).T[:, :3]

    out = {"means3D": transformed_pts}
    if model.is_isotropic:
        out["unnorm_rotations"] = unnorm_rots
    else:
        cam_q = _rotmat_to_quat(w2c[:3, :3].unsqueeze(0))
        out["unnorm_rotations"] = quat_mult(cam_q, F.normalize(unnorm_rots))
    return out


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) → (B, 4) wxyz 四元数（Shepperd 法，数值稳定）。"""
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    case = tr > 0
    if case.any():
        s = torch.sqrt(tr[case] + 1.0) * 2
        q[case, 0] = 0.25 * s
        q[case, 1] = (R[case, 2, 1] - R[case, 1, 2]) / s
        q[case, 2] = (R[case, 0, 2] - R[case, 2, 0]) / s
        q[case, 3] = (R[case, 1, 0] - R[case, 0, 1]) / s
    for i, j, k in ((1, 2, 0), (2, 0, 1), (0, 1, 2)):
        m = (R[:, i, i] > R[:, j, j]) & (R[:, i, i] > R[:, k, k]) & ~case
        if m.any():
            s = torch.sqrt(R[m, i, i] - R[m, j, j] - R[m, k, k] + 1.0) * 2
            q[m, i + 1] = 0.25 * s
            q[m, 0] = (R[m, k, j] - R[m, j, k]) / s
            q[m, j + 1] = (R[m, j, i] + R[m, i, j]) / s
            q[m, k + 1] = (R[m, k, i] + R[m, i, k]) / s
    # 兜底（数值退化时回退到单位四元数以外的正常分支不会走到这里，仅防御）
    q = F.normalize(q + 1e-12)
    return q


def render_prepare(model: GaussianModel, gaussians_grad: bool = False) -> dict:
    """预计算迭代间不变的部分：颜色/透明度/尺度浮点转换（分辨率无关）。

    Phase 4 性能优化：track 迭代内每次 render 会重复执行 .float() 存储层转换
    （FP16 存储下每趟 4 个瞬态转换，~1-3ms/趟）。迭代间这些值不变（调用方持
    _gpu_lock，模型迭代内不可变），预计算一次跨迭代、跨分辨率复用。

    约束：返回的 prep 只在模型未被修改期间有效（add/remove/prune 后必须重建，
    由调用方持有锁保证——track 调用期间 map 无法改模型）。
    setup_camera 张量分配很小（~0.2ms），保持每次 render 构建，不纳入 prep。
    """
    scales = model.scales()
    colors = model.params["rgb_colors"].float() if gaussians_grad \
        else model.params["rgb_colors"].detach().float()
    opacities = model.opacities().float() if gaussians_grad \
        else model.opacities().detach().float()
    scales_f = scales.float() if gaussians_grad else scales.detach().float()
    return {"colors": colors, "opacities": opacities,
            "scales": scales_f, "gaussians_grad": gaussians_grad,
            "n_gaussians": model.num_gaussians}


def render(model: GaussianModel, w2c, K, width: int, height: int,
           gaussians_grad: bool = True, camera_grad: bool = True,
           mask=None, needs_depth: bool = True, prep: dict | None = None):
    """渲染一帧。返回 (rgb, depth, silhouette, radius, means2D)。

    prep（Phase 4）：render_prepare() 的输出——迭代间复用设置矩阵与
    颜色/透明度/尺度浮点转换（省每次 render 的重复分配与拷贝）。
    仅在调用方保证模型不变时传；grad 模式必须与 prep 构建时一致。

    rgb (3,H,W) 0~1；depth (1,H,W) 相机系 z；silhouette (H,W)；radius (N,)；means2D (N,2)。
    w2c 可为 numpy (4,4) 或 cuda tensor；需要梯度时传 cuda tensor（如 tracking 扰动链）。

    mask（Phase 3 §4 视锥剔除）：(N,) bool，True = 可见。mask 给定时只渲染可见子集——
    不可见高斯无像素贡献 → 无梯度（grad=None）→ Adam 不步进，天然冻结；
    backward 经压缩索引自动 scatter 回全量参数的对应位置。
    返回的 radius/means2D 对应压缩后的子集（现有调用方均忽略，安全）。

    needs_depth（Phase 3 性能档）：False 时跳过 depth/silhouette 第二趟光栅化，
    返回 depth=None、silhouette=None——track 在 depth 趟跳过迭代下省 ~1/2 光栅化
    前向与反传；RGB 前向与 True 完全一致（同输入逐元素相等）。
    """
    if not isinstance(w2c, torch.Tensor):
        w2c = torch.as_tensor(np.asarray(w2c, dtype=np.float32)).cuda().float()
    t = transform_to_frame(model, w2c, gaussians_grad=gaussians_grad, camera_grad=camera_grad)

    # --- RGB 前向 ---
    # Phase 3 §3：FP16 存储下统一 .float() 边界——CUDA kernel 内部恒 float，
    # 存储层 half 只在省常驻显存，计算层转换是瞬态分配（用完即释放）。
    # Phase 4：prep 给定时复用迭代间不变的转换结果（模型不变性由调用方保证）
    cam = setup_camera(width, height, K)
    if prep is not None:
        assert prep["gaussians_grad"] == gaussians_grad, \
            "prep grad 模式与调用不一致（模型可能已变，需重建 prep）"
        scales_f, colors, opacities = prep["scales"], prep["colors"], prep["opacities"]
    else:
        scales = model.scales()
        colors = model.params["rgb_colors"].float() if gaussians_grad \
            else model.params["rgb_colors"].detach().float()
        opacities = model.opacities().float() if gaussians_grad \
            else model.opacities().detach().float()
        scales_f = scales.float() if gaussians_grad else scales.detach().float()
    if mask is not None:
        t = {k: v[mask] for k, v in t.items()}
        colors = colors[mask]
        opacities = opacities[mask]
        scales_f = scales_f[mask]
        n_gauss = int(mask.sum())
    else:
        n_gauss = model.num_gaussians
    means2d_buf = torch.zeros(n_gauss, 3, requires_grad=True, device="cuda")
    rendervar = {
        "means3D": t["means3D"],
        "colors_precomp": colors,
        "rotations": t["unnorm_rotations"],
        "opacities": opacities,
        "scales": scales_f,
        "means2D": means2d_buf,
    }
    im, radius, _ = GaussianRasterizer(raster_settings=cam)(**rendervar)

    if not needs_depth:
        return im, None, None, radius, rendervar["means2D"]

    # --- depth + silhouette 前向（把每个高斯的 z / 1 / z² 当颜色渲染）---
    pts = t["means3D"]
    d_sil = torch.stack([pts[:, 2], torch.ones_like(pts[:, 2]), pts[:, 2] ** 2], dim=-1)
    depth_rendervar = {
        "means3D": pts,
        "colors_precomp": d_sil,
        "rotations": t["unnorm_rotations"],
        "opacities": rendervar["opacities"],
        "scales": rendervar["scales"],
        "means2D": torch.zeros_like(means2d_buf),
    }
    depth_sil, _, _ = GaussianRasterizer(raster_settings=cam)(**depth_rendervar)

    depth = depth_sil[0:1, :, :]
    silhouette = depth_sil[1, :, :]
    return im, depth, silhouette, radius, rendervar["means2D"]


def render_precomp(model: GaussianModel, w2c, K, width: int, height: int,
                   colors_precomp: torch.Tensor, gaussians_grad: bool = False,
                   camera_grad: bool = False, mask=None) -> torch.Tensor:
    """用预计算颜色通道渲染（Phase 6 §1：D=3 特征光栅化快速路径）。

    colors_precomp: (N, 3) 预计算颜色/特征（如 gaussians.params["features"]）——
    inria kernel 对 colors_precomp 与几何共用同一套投影/深度排序/α 权重，
    数学上即特征 splat：F_2d(pixel) = Σ f_i·α_i·Π_{j<i}(1-α_j)。

    gaussians_grad=False / camera_grad=False：蒸馏阶段几何冻结时的省显存设置——
    kernel backward 的 grad_colors_precomp 仍正常回流到 colors_precomp（即 features），
    detach 的几何输入梯度被丢弃，行为正确（几何本就冻结）。
    返回 (3, H, W) 特征图。
    """
    if not isinstance(w2c, torch.Tensor):
        w2c = torch.as_tensor(np.asarray(w2c, dtype=np.float32)).cuda().float()
    t = transform_to_frame(model, w2c, gaussians_grad=gaussians_grad, camera_grad=camera_grad)
    cam = setup_camera(width, height, K)
    if mask is not None:
        t = {k: v[mask] for k, v in t.items()}
        colors = colors_precomp[mask]
        n_gauss = int(mask.sum())
    else:
        colors = colors_precomp
        n_gauss = model.num_gaussians
    rendervar = {
        "means3D": t["means3D"],
        "colors_precomp": colors,
        "rotations": t["unnorm_rotations"],
        "opacities": model.opacities().detach().float() if not gaussians_grad
        else model.opacities().float(),
        "scales": model.scales().detach().float() if not gaussians_grad
        else model.scales().float(),
        "means2D": torch.zeros(n_gauss, 3, requires_grad=True, device="cuda"),
    }
    im, _, _ = GaussianRasterizer(raster_settings=cam)(**rendervar)
    return im
