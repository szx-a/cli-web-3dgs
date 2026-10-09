"""把训练好的 3D 高斯模型转成**三角网格**（mesh）。

背景与选型
----------
用户反馈"点云 GLB 没有实感"，希望得到能进 Blender / 游戏引擎的网格。
调研了三条路线（详见 docs/03 的 G14）：

* **SuGaR**（CVPR 2024，表面对齐 + 网格提取）
  官方 Windows 移植版要求 `pytorch==2.0.1 + pytorch-cuda=11.8 + pytorch3d==0.7.4`。
  本机是 **RTX 5060 / Blackwell sm_120 / CUDA 12.8**：torch 2.0.1 不含 sm_120 内核、
  pytorch3d 也没有对应 wheel，其自带 viewer 还明确写着不支持 Windows。
  走通等于把整个图形栈降级到 2023 年并与驱动冲突 ⇒ **放弃**。

* **2DGS / 2D-surfel-gaussian**（surfel 对齐，几何更准）
  `gsplat` 确实自带 2DGS 光栅化（`RasterizeToPixels2DGS*`，本机已编译），
  但 2DGS 是**另一套训练方式**（surfel + 法线一致性 + 深度畸变正则），
  要重写训练循环；网格提取同样要自己实现 ⇒ 作为后续可选路线，本次不做。

* **TSDF 融合**（几何式，本次采用）
  用**已训练好的 3DGS** 从训练视角渲染**深度图 + RGB**，
  再用 `open3d` 的 `ScalableTSDFVolume` 做截断符号距离场融合，提取三角网格。
  优点：不需要新训练、不引入版本冲突、8GB 显存可跑（逐视角渲染）。
  缺点：网格质量取决于深度质量；高斯渲染的深度对"薄结构/透明物"不够锐利，
        所以得到的网格偏"实体化"，细节不如 SuGaR。

实现要点
--------
* 深度用 gsplat 的 `render_mode="RGB+ED"`（ED = expected depth，比中值深度平滑）。
* 只融合**有足够有效像素**的视角（避免背景/空洞污染 SDF）。
* 体素尺寸按场景尺度自适应（`scene_extent / voxel_div`）。
* 输出 `.obj`（通用，Blender 直接可开）与可选的 `.ply` 网格。
"""

from __future__ import annotations

import numpy as np


def scale_intrinsics(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """把内参从原始分辨率缩放到目标分辨率。

    踩坑：最初想从 `K[0,2]`（主点）反推缩放比例，那是自相矛盾的
    （主点本身就随分辨率变化）。必须显式给出原始尺寸。
    """
    sw, sh = src_wh
    dw, dh = dst_wh
    sx, sy = dw / max(sw, 1), dh / max(sh, 1)
    Ks = np.asarray(K, dtype=np.float64).copy()
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] *= sx
    Ks[1, 2] *= sy
    return Ks


def _scene_extent(c2ws: np.ndarray, points: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """估计场景中心与尺度（用于自适应体素大小与裁剪体积）。"""
    centers = c2ws[:, :3, 3]
    center = centers.mean(axis=0)
    if points is not None and len(points) > 0:
        # 用稀疏点云的范围更贴近"物体"本身（相机中心可能离物体较远）
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo + hi) / 2.0
        extent = float(np.linalg.norm(hi - lo))
    else:
        extent = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    extent = max(extent, 1e-6)
    # 留一点余量，保证整体落在裁剪框内
    return center, extent * 1.35


def render_depths(
    means, quats, scales_real, opacities, shs,
    c2ws: np.ndarray, ks: np.ndarray,
    width: int, height: int, sh_degree: int,
    src_wh: tuple[int, int],
    device: str = "cuda",
    max_views: int = 80,
    background=None,
):
    """从给定相机位姿渲染 (RGB, expected-depth, alpha)。

    src_wh：内参 K 对应的原始图像尺寸，用于正确缩放到 width/height。
    返回 (列表, 实际使用的视角下标)，每项为 (rgb, depth, alpha) 的 float32 数组。
    """
    import torch
    from gsplat import rasterization

    n_views = len(c2ws)
    if max_views and n_views > max_views:
        idx = np.linspace(0, n_views - 1, max_views).round().astype(int)
    else:
        idx = np.arange(n_views)

    bg = torch.zeros(3, device=device) if background is None else background
    out = []
    with torch.no_grad():
        for i in idx:
            viewmat = torch.linalg.inv(torch.as_tensor(c2ws[i], dtype=torch.float32, device=device))
            Ks = scale_intrinsics(ks[i], src_wh, (width, height))
            colors, alphas, _ = rasterization(
                means=means, quats=quats, scales=scales_real,
                opacities=opacities, colors=shs,
                viewmats=viewmat[None],
                Ks=torch.as_tensor(Ks, dtype=torch.float32, device=device)[None],
                width=width, height=height, sh_degree=sh_degree,
                near_plane=0.01, far_plane=1e10, packed=False, render_mode="RGB+ED",
            )
            # ⚠️ 维度坑：gsplat 返回的 colors 形状是 (C, H, W, ch)，**带 batch 维**。
            # 必须先 `colors[0]` 取出 (H, W, ch)，再做通道切分。
            # 我一开始写成 `colors[0][..., :3]`，实际得到 (H, W, H, 3) ——
            # 因为 `[..., :3]` 作用在 3 维数组上会去切**空间维**，
            # 结果 rgb 是四维、depth 是二维，后面 integrator 直接报
            # "array is 2-dimensional, but 3 were indexed"。这个坑靠形状打印才定位到。
            img = colors[0]
            alpha = alphas[0]
            if img.dim() != 3 or img.shape[-1] < 4:
                raise RuntimeError(
                    f"render_mode=RGB+ED 期望 colors[0] 形状 (H,W,4)，实际 {tuple(colors.shape)}")
            rgb = img[..., :3] + bg * (1 - alpha)
            depth = img[..., 3]
            out.append((rgb.clamp(0, 1).cpu().numpy().astype(np.float32),
                        depth.cpu().numpy().astype(np.float32),
                        alphas[0].cpu().numpy().astype(np.float32)))
    return out, idx


def fuse_tsdf(
    rendered, c2ws: np.ndarray, view_idx, ks: np.ndarray,
    width: int, height: int,
    src_wh: tuple[int, int],
    center: np.ndarray, extent: float,
    voxel_size: float = 0.01,
    sdf_trunc_ratio: float = 4.0,
    depth_trunc_ratio: float = 1.2,
    min_alpha: float = 0.35,
    min_valid_ratio: float = 0.04,
    depth_max: float = 0.0,
):
    """把渲染出的深度图融合成三角网格。返回 (open3d TriangleMesh, 统计信息)。

    实现参考社区成熟做法（19reborn/neural-texture-splatting 的 extract_mesh_tsdf.py）：
    用**张量版** `o3d.t.geometry.VoxelBlockGrid` + `compute_unique_block_coordinates`，
    只对**视锥实际覆盖的体素块**做积分。相比旧版 `ScalableTSDFVolume`：

      * 稀疏块存储，内存随"表面附近的块数"增长，而不是随整个立方体边长³ 增长；
      * 有 `compute_unique_block_coordinates`，先算出要更新的块再积分，快得多；
      * 可跑 CPU 也可跑 CUDA（本机 open3d 0.20 的 CUDA 支持取决于构建版本）。

    踩坑记录：
      * `ScalableTSDFVolume.extract_triangle_mesh()` 会固定分配 512³ float32
        （1.50 GiB），本机 15.2GB 内存下直接 MemoryError，且不受 voxel_length 影响；
      * `UniformTSDFVolume` 的 origin 参数要 numpy 向量（open3d 没有 `Point3D` 类型）；
      * open3d 的 `Image` 只接受**连续**缓冲区，非连续数组要 `np.ascontiguousarray`。
    """
    import open3d as o3d
    import open3d.core as o3c

    voxel = max(float(voxel_size), 1e-6)
    sdf_trunc = voxel * sdf_trunc_ratio
    depth_trunc = float(depth_max) if depth_max else (extent * depth_trunc_ratio)

    device = o3c.Device("CPU:0")
    try:
        if o3c.cuda.is_available():
            device = o3c.Device("CUDA:0")
    except Exception:  # noqa: BLE001
        device = o3c.Device("CPU:0")

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel,
        block_resolution=16,
        block_count=100000,
        device=device,
    )

    n_used = 0
    for k, i in enumerate(view_idx):
        rgb, depth, alpha = rendered[k]
        # ⚠️ 维度坑：alpha 是 (H, W, 1)，而 depth 是 (H, W)。
        # 直接 `alpha > t` 会和 depth 广播成 (H, W, H) —— 我踩过这个，
        # valid 的形状变成三维后，掩码语义完全错乱。必须先 squeeze。
        a2 = alpha.reshape(depth.shape[0], depth.shape[1])
        valid = (a2 > min_alpha) & np.isfinite(depth) & (depth > 1e-4) & (depth < depth_trunc)
        if float(valid.mean()) < min_valid_ratio:
            continue
        # 无效像素置 0：社区做法一致（VoxelBlockGrid 把 0 视为"无观测"）
        d = np.ascontiguousarray(np.where(valid, depth, 0.0).astype(np.float32))
        c = np.ascontiguousarray(rgb.astype(np.float32))  # open3d 要求 color 为 float

        Ks = scale_intrinsics(ks[i], src_wh, (width, height))
        intrinsic = o3c.Tensor(Ks, dtype=o3c.float64)
        extrinsic = o3c.Tensor(np.linalg.inv(np.asarray(c2ws[i], dtype=np.float64)),
                               dtype=o3c.float64)
        # 注意：compute_unique_block_coordinates / integrate 要的是
        # `o3d.t.geometry.Image` 对象，不是裸 Tensor（踩过 TypeError）。
        depth_img = o3d.t.geometry.Image(o3c.Tensor(d, dtype=o3c.float32))
        color_img = o3d.t.geometry.Image(o3c.Tensor(c, dtype=o3c.float32))

        frustum = vbg.compute_unique_block_coordinates(
            depth_img, intrinsic, extrinsic, 1.0, depth_trunc)
        vbg.integrate(frustum, depth_img, color_img, intrinsic, extrinsic, 1.0, depth_trunc)
        n_used += 1

    mesh_t = vbg.extract_triangle_mesh()
    try:
        mesh = mesh_t.to_legacy()
    except Exception:  # noqa: BLE001
        mesh = mesh_t
    mesh.compute_vertex_normals()
    return mesh, {"voxel": voxel, "sdf_trunc": sdf_trunc,
                  "depth_trunc": depth_trunc, "views_used": n_used,
                  "device": str(device)}


def clean_mesh(mesh, min_tri_ratio: float = 0.002):
    """去掉离群小连通块（TSDF 边缘常见）。ratio 相对最大连通块的三角形数。"""
    if len(mesh.triangles) == 0:
        return mesh, 0
    try:
        tri_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()
    except Exception:  # noqa: BLE001
        return mesh, 0
    tri_clusters = np.asarray(tri_clusters)
    counts = np.asarray(cluster_n_tri)
    if counts.size == 0:
        return mesh, 0
    keep = counts >= max(1, int(counts.max() * min_tri_ratio))
    keep_ids = set(np.nonzero(keep)[0].tolist())
    mask = np.array([c in keep_ids for c in tri_clusters], dtype=bool)
    removed = int((~mask).sum())
    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, removed


def to_largest_component(mesh):
    """只保留最大连通块（单物体重建最常用）。"""
    if len(mesh.triangles) == 0:
        return mesh, 0
    try:
        tri_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()
    except Exception:  # noqa: BLE001
        return mesh, 0
    tri_clusters = np.asarray(tri_clusters)
    counts = np.asarray(cluster_n_tri)
    if counts.size == 0:
        return mesh, 0
    biggest = int(counts.argmax())
    mask = tri_clusters == biggest
    removed = int((~mask).sum())
    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, removed
