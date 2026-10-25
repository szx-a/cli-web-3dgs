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


def gaussian_axis_normals(quats: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
    """把每个高斯的**旋转后 +z 轴**作为它的表面法线（单位向量）。

    为什么这是本项目**最干净的法线来源**（实测）：
      * 高斯自带法线：邻域一致度 mean_dot **0.787**、翻转率 **0.2%**；
      * 深度梯度法线：翻转率约 **17%**（是"网格碎/毛刺"的直接来源，见 docs/03 G62）。
    两者差两个数量级。而且 `argmin(scales)` 取"最短轴"是**错**的 ——
    在压平模型上只有 19.2% 的高斯最短轴恰好是 z，其余会取到水平轴（见 docs/03 G65）。
    压平（`scatter(-1, argmin, cap)`）压的就是**最小**尺度，所以最短轴与 +z 本是同一根，
    直接取旋转第 3 列既正确又无索引歧义。

    `scales` 参数保留以兼容调用方，当前不使用。
    """
    q = np.asarray(quats, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats 形状应为 [N,4]，实际 {q.shape}")
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.stack([2 * (x * z + w * y),
                  2 * (y * z - w * x),
                  1 - 2 * (x * x + y * y)], axis=1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def transfer_normals_from_gaussians(points: np.ndarray, gauss_xyz: np.ndarray,
                                    gauss_nrm: np.ndarray,
                                    voxel: float, max_dist: float | None = None,
                                    fallback: np.ndarray | None = None,
                                    orient_ref: np.ndarray | None = None):
    """把每个反投影点关联到**最近的高斯**，并用该高斯的法线替换它的法线。

    为什么需要（本轮发现的真问题）：
    导出流程里泊松拿到的法线一直是 `backproject_depth_to_points` 算的**深度梯度法线**，
    实测翻转率 **43%**（`mesh_method=poisson`）。而高斯自带法线只有 **0.2%** ——
    差两个数量级，却没有被送进泊松。

    ⚠️ **朝向必须与既有法线场一致，否则会更糟**（这是我踩的坑）：
    高斯法线是"朝外"的，而深度梯度法线在反投影里已被统一成 **`dot(n, P_cam) > 0`（朝向相机）**。
    两者混在同一个点云里，泊松会收到互相矛盾的方向 —— 实测翻转率**反而升到 45%**
    （对照：同模型 `mesh_method=2dgs` 是 20%）。
    所以传入 `orient_ref`（通常就是原来的深度梯度法线），用它的符号约定对齐高斯法线：
    与参照反向的点先把法线取反。这一步不做，迁移越成功、结果越差。

    做法（体素哈希，O(N)，不用 KDTree —— open3d 的 KDTreeFlann 在百万级重合点上会
    报 `search_knn_vector_3d() error!` 且不说明原因，本项目踩过）：
      1. 以 `3×voxel` 为边长给高斯建哈希桶（桶必须比查询尺度大：
         实测最近高斯距离 p99 = 2.5 体素，±1 桶在 1 倍桶长下只覆盖 ±1 体素，
         曾导致"只迁移 5.3% 的点"，看起来像半径问题、实际是搜索范围不足）；
      2. 每桶记平均位置与**平均法线**（先做符号消歧，否则相反符号互相抵消）；
      3. 每个点查所在桶及相邻 26 桶，取最近桶作为法线来源；
      4. 超过 `max_dist` 的点用 `fallback`（没有则置零）。

    返回 `(normals, stats)`。
    """
    pts = np.asarray(points, dtype=np.float64)
    gx = np.asarray(gauss_xyz, dtype=np.float64)
    gn = np.asarray(gauss_nrm, dtype=np.float64)
    n = len(pts)
    if n == 0 or len(gx) == 0:
        return (np.asarray(fallback) if fallback is not None else np.zeros((n, 3))), {
            "transferred": 0, "dropped": int(n), "transfer_ratio": 0.0}

    # ⚠️ 桶边长取 **3 倍**查询体素，而不是 1 倍。
    # 原因（实测教训）：搜索只覆盖"点所在桶 ± 1 桶"，若桶边长 = 查询体素，
    # 覆盖范围就只有 ±1 体素；而真实最近高斯距离实测 p99 = **2.50 体素**，
    # 于是大头部的近邻**根本没被查到** —— 表现为"只迁移了 5.3% 的点"，
    # 看起来像"半径太小"，实际是**搜索范围不足**。
    # 桶放大 3 倍后，±1 桶即覆盖 ±3 体素，能包住 p99 的 2.50。
    cell = float(voxel) * 3.0
    keys = np.floor(gx / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    ncell = len(uniq)

    # 符号消歧：整体对齐到一个参考方向，避免桶内平均被抵消
    ref = gn.sum(0)
    rl = float(np.linalg.norm(ref))
    if rl > 1e-12:
        s = np.sign(gn @ (ref / rl))
        s[s == 0] = 1.0
        gn = gn * s[:, None]

    acc_p = np.zeros((ncell, 3))
    acc_n = np.zeros((ncell, 3))
    cnt = np.zeros(ncell)
    np.add.at(acc_p, inv, gx)
    np.add.at(acc_n, inv, gn)
    np.add.at(cnt, inv, 1.0)
    cnt = np.maximum(cnt, 1.0)
    cell_p = acc_p / cnt[:, None]
    cell_n = acc_n / cnt[:, None]
    cell_n /= np.maximum(np.linalg.norm(cell_n, axis=1, keepdims=True), 1e-12)

    lut: dict[tuple[int, int, int], int] = {
        (int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2])): i for i in range(ncell)}

    out = np.zeros((n, 3))
    ok = np.zeros(n, dtype=bool)
    pkeys = np.floor(pts / cell).astype(np.int64)
    lim = float(max_dist) if max_dist else float(voxel) * 1.75
    # 记录每个点的最近距离，用于**有依据地**选半径。
    # 不记录的话只能靠猜：我先用 voxel*1.75 只命中 3.9%、voxel*4 只命中 5.3%，
    # 却无法判断"是半径太小"还是"这些点本来就没有近邻高斯"。
    nd = np.full(n, np.inf)
    for i in range(n):
        cx, cy, cz = int(pkeys[i, 0]), int(pkeys[i, 1]), int(pkeys[i, 2])
        best, best_d2 = -1, np.inf
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    j = lut.get((cx + dx, cy + dy, cz + dz))
                    if j is None:
                        continue
                    d2 = float(((cell_p[j] - pts[i]) ** 2).sum())
                    if d2 < best_d2:
                        best_d2, best = d2, j
        if best >= 0:
            nd[i] = np.sqrt(best_d2)
        if best >= 0 and best_d2 <= lim * lim:
            out[i] = cell_n[best]
            ok[i] = True

    dropped = int((~ok).sum())
    if dropped and fallback is not None:
        fb = np.asarray(fallback, dtype=np.float64)
        if len(fb) == n:
            out[~ok] = fb[~ok]

    # ---- 朝向对齐（不做这一步会让结果明显变差，见 docstring）----
    flipped = 0
    if orient_ref is not None:
        rf = np.asarray(orient_ref, dtype=np.float64)
        if rf.shape == out.shape:
            d = (out * rf).sum(1)
            need = ok & (d < 0)
            flipped = int(need.sum())
            out[need] = -out[need]
    fin = nd[np.isfinite(nd)]
    stats = {
        "transferred": int(ok.sum()),
        "dropped": dropped,
        "transfer_ratio": float(ok.mean()) if n else 0.0,
        "voxel": float(voxel),
        "max_dist": lim,
        "gaussians": int(len(gx)),
        "dist_voxels_p50": float(np.median(fin) / voxel) if len(fin) else float("nan"),
        "dist_voxels_p90": float(np.percentile(fin, 90) / voxel) if len(fin) else float("nan"),
        "dist_voxels_p99": float(np.percentile(fin, 99) / voxel) if len(fin) else float("nan"),
        "no_neighbour": int((~np.isfinite(nd)).sum()),
        "flipped_for_orientation": flipped,
    }
    return out, stats


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


def auto_crop_bounds(points: np.ndarray, opacity: np.ndarray,
                     op_thresh: float = 0.3, quantile: float = 1.0,
                     pad_ratio: float = 0.12) -> tuple[np.ndarray, np.ndarray, dict]:
    """自动圈出"主体"的包围盒，用于把 TSDF 聚焦到物体上。

    ## 为什么需要

    COLMAP 会把桌面、书本、背景杂物也重建出来，点云里还有大量低不透明度的
    漂浮高斯。直接用**全部**点的包围盒做 TSDF 会导致
        * 场景"尺度"被放大数倍 -> 体素与截断距离全部失配；
        * 体积里绝大部分是空气，网格覆盖整个房间而不是物体。

    所以先用**不透明度**过滤（漂浮/背景高斯普遍偏低），这一步是有效的。

    ## ⚠️ 为什么**不再**按分位数裁（真实事故）

    原来用 `quantile=0.95/0.98` 再裁一遍两端。实测 `img3-final`（牌坊）：

        不透明度 > 0.3 的高斯：24,629 个，铺满 **5.83 × 2.16 × 2.85**
        但它们的 95% 分位框只有 **1.67 × 1.05 × 0.49**
        最终网格 bbox 只有高斯的 **25% / 22% / 15%**，体积占 **0.85%**

    ⇒ 导出的网格只是中央一小块**扭曲碎片**（用户看到的就是"网络破碎"）。

    **根因**：3DGS 会把容量放在**细节密集**处（门楼雕花），屋檐/翼角面积大但纹理平滑
    ⇒ 分到的**高斯数量很少**。按"点的分位数"裁，永远会先把**稀疏但真实的结构**
    裁掉 —— 这与我在查看器取景上犯的错是同一个（`docs/03` G113 之后又一次）。

    ⇒ 默认 `quantile=1.0`：**只用不透明度过滤，不再按分位数裁**。
    背景污染的主要来源（低不透明度漂浮点）已经被滤掉。
    """
    pts = np.asarray(points, dtype=np.float64)
    op = np.asarray(opacity, dtype=np.float64).reshape(-1)
    if len(pts) == 0:
        return np.zeros(3), np.ones(3), {"mode": "empty"}

    mask = op > op_thresh
    if int(mask.sum()) < max(64, int(0.02 * len(pts))):
        mask = np.ones(len(pts), dtype=bool)  # 点太少就不筛
    sel = pts[mask]

    q = float(quantile)
    if q >= 1.0:
        lo = sel.min(axis=0)
        hi = sel.max(axis=0)
    else:
        lo_q, hi_q = (1.0 - q) / 2.0, 1.0 - (1.0 - q) / 2.0
        lo = np.quantile(sel, lo_q, axis=0)
        hi = np.quantile(sel, hi_q, axis=0)
    size = np.maximum(hi - lo, 1e-6)
    pad = size * float(pad_ratio)
    lo -= pad
    hi += pad
    info = {"mode": "auto", "points_used": int(mask.sum()),
            "points_total": int(len(pts)),
            "box_size": [round(float(v), 4) for v in (hi - lo)]}
    return lo, hi, info


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
    erode_px: int = 1,
    grad_thresh_ratio: float = 0.06,
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
    stats: list[dict] = []
    for k, i in enumerate(view_idx):
        rgb, depth, alpha = rendered[k]
        # 用 filter_depth 抑制 floater（详见该函数注释）：
        # 原始 alpha 阈值会把星芒状漂浮高斯也算成表面，在 TSDF 里长出假几何。
        valid, fstat = filter_depth(
            depth, alpha,
            min_alpha=max(min_alpha, 0.5),
            erode_px=erode_px,
            grad_thresh_ratio=grad_thresh_ratio,
            extent=extent,
        )
        stats.append(fstat)
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
    keep = [s["kept_ratio"] for s in stats]
    return mesh, {"voxel": voxel, "sdf_trunc": sdf_trunc,
                  "depth_trunc": depth_trunc, "views_used": n_used,
                  "device": str(device),
                  "filter_keep_ratio": round(float(np.mean(keep)), 4) if keep else 0.0}


def select_views_by_distance(c2ws: np.ndarray, points: np.ndarray | None = None,
                             mad_scale: float = 3.0, min_keep: int = 8):
    """按"相机到物体中心的距离"剔除异常视角，返回保留下来的下标。

    为什么需要：真实手持环绕素材里，用户常会**贴近物体拍特写**、也会**拉远拍全貌**
    （实测某次：相机距离 0.33 ~ 7.59，差 23 倍）。
    * 贴太近的视角只看到一小块表面，贡献有限却会把包围盒撑大；
    * 拉太远的视角看到大量背景，COLMAP 在那里重建的杂点会把"场景尺度"放大数倍，
      导致 TSDF 的体素与截断距离全部失配、网格支离破碎。

    这里用中位数 + MAD（对离群值稳健）做筛选，保留距离在
    `median ± mad_scale * 1.4826 * MAD` 内的视角。
    """
    centers = np.asarray(c2ws)[:, :3, 3]
    if points is not None and len(points) > 0:
        target = np.asarray(points, dtype=np.float64).mean(axis=0)
    else:
        target = centers.mean(axis=0)
    dist = np.linalg.norm(centers - target, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med))) * 1.4826
    if mad <= 1e-9:
        keep = np.ones(len(dist), dtype=bool)
    else:
        lo, hi = med - mad_scale * mad, med + mad_scale * mad
        keep = (dist >= max(lo, 1e-6)) & (dist <= hi)
    if int(keep.sum()) < min_keep:
        keep = np.ones(len(dist), dtype=bool)
    return keep, {"median_dist": round(med, 4), "mad": round(mad, 4),
                  "kept": int(keep.sum()), "total": int(len(dist))}


def filter_depth(depth, alpha, min_alpha: float = 0.5, erode_px: int = 1,
                 grad_thresh_ratio: float = 0.06, extent: float = 1.0):
    """把"可信任的表面像素"挑出来，抑制 3DGS 的 floater 造成的假几何。

    背景（实测踩到）：3DGS 会学出大量**漂浮高斯（floater）**，
    它们在物体前方形成半透明薄片，渲染出来是星芒状的亮斑。
    若直接拿去融合 TSDF，就会在那里长出一片虚假表面 —— 表现为网格
    "只有内壁、外壁破碎、边缘有尖刺"。

    三个判据（都用纯 numpy/scipy，不引入新依赖）：
      1. **alpha 阈值**：过滤半透明像素（floaters 的不透明度普遍偏低）。
      2. **alpha 腐蚀**：floaters 常是薄结构，腐蚀能整片去掉；
         物体主体的实心区域几乎不受影响。
      3. **深度不连续性**：floaters 与真实表面之间常有深度跳变，
         用局部深度梯度剔除突跳像素。

    返回 (valid_mask, 统计字典)。
    """
    import numpy as np
    from scipy import ndimage

    d = np.asarray(depth, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32).reshape(d.shape)

    base = (a > min_alpha) & np.isfinite(d) & (d > 1e-4)
    n0 = int(base.sum())

    # 2) alpha 腐蚀：只对 base 掩码做，用最小滤波（等价于对掩码腐蚀）
    if erode_px and erode_px > 0:
        mask = ndimage.binary_erosion(base, iterations=int(erode_px))
    else:
        mask = base
    n1 = int(mask.sum())

    # 3) 深度不连续性：在有效区域内比较与邻域中值的差
    if grad_thresh_ratio > 0 and n1 > 0:
        med = ndimage.median_filter(np.where(mask, d, 0.0), size=3)
        diff = np.abs(d - med)
        thresh = float(extent) * float(grad_thresh_ratio)
        mask = mask & (diff < thresh)
    n2 = int(mask.sum())

    return mask, {"px_raw": n0, "px_eroded": n1, "px_final": n2,
                  "kept_ratio": (n2 / n0) if n0 else 0.0}


def filter_by_multiview_consistency(points, normals=None, colors=None,
                                    view_id=None, voxel: float = 0.01,
                                    min_views: int = 2,
                                    normal_dot_min: float = 0.0):
    """按**多视角一致性**剔除只在单个视角出现的反投影点（2DGS 路线的核心过滤）。

    为什么需要：单纯深度反投影会把每个视角的**噪声与深度跳变**都变成点。
    两个具体后果，都是本项目的实测问题：

    1. **表面碎/毛刺**：只在 1 个视角出现的点，多半是深度不连续处的伪影
       （前景边缘"拖尾"到背景上）。它们让泊松在局部生成薄片与尖刺。
    2. **法线翻转 ~35%**：同一个表面被相邻视角各投一份点，两份点的法线
       可能因为噪声而方向相反；体素下采样把它们**平均成接近零**的法线，
       于是表面光照出现大片黑斑。

    判据（体素投票）：
      * 把点按体素量化，同一体素内统计**出现过的不同视角数**
      * 少于 `min_views` 个视角支持的点直接删除（孤立噪点）
      * 同一体素内，若某点法线与该体素**平均法线的夹角余弦 < normal_dot_min**
        则删除（抑制方向矛盾的点）

    这对应 2DGS / Gaussian Opacity Fields 里"用多视角深度一致性约束表面"的做法，
    但作用在**导出阶段**，不需要重训模型。

    返回 (points, normals, colors, stats)。
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0 or voxel <= 0:
        return pts, normals, colors, {"points_in": n, "points_out": n, "kept_ratio": 1.0}

    nrm = np.asarray(normals, dtype=np.float64) if normals is not None else None
    col = np.asarray(colors, dtype=np.float64) if colors is not None else None
    vid = (np.asarray(view_id).reshape(-1) if view_id is not None
           else np.zeros(n, dtype=np.int64))

    keys = np.floor(pts / float(voxel)).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)

    # 每个体素内出现过多少个**不同视角**
    order = np.lexsort((vid, inv))
    inv_s, vid_s = inv[order], vid[order]
    new_group = np.empty(len(inv_s), dtype=bool)
    new_group[0] = True
    new_group[1:] = (inv_s[1:] != inv_s[:-1]) | (vid_s[1:] != vid_s[:-1])
    uniq_per_voxel = np.bincount(inv_s[new_group], minlength=len(counts))
    keep = uniq_per_voxel[inv] >= int(min_views)

    dropped_views = int((~keep).sum())

    # 体素内平均法线（只统计保留下来的点），用于一致性方向过滤
    dropped_normal = 0
    if nrm is not None and normal_dot_min > -1.0:
        acc = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(acc, inv[keep], nrm[keep])
        mean_n = acc[inv]
        ln = np.linalg.norm(mean_n, axis=1)
        good = ln > 1e-12
        dot = np.ones(len(pts), dtype=np.float64)
        dot[good] = (nrm[good] * (mean_n[good] / ln[good, None])).sum(1)
        ok_n = (~keep) | (dot >= float(normal_dot_min))
        dropped_normal = int((keep & ~ok_n).sum())
        keep = keep & ok_n

    out_pts = pts[keep]
    out_nrm = (nrm[keep] if nrm is not None else None)

    # ---- 多视角法线**平均**（社区标准做法：2DGS / Gaussian Opacity Fields 系）----
    #
    # 为什么必须做：上面虽然算了体素平均法线，但**只用于投票，从不输出** ——
    # 保留下来的仍是每个视角各自的**逐点原始法线**（含深度噪声）。
    # 同一个表面被相邻视角各投一份点，两份法线方向略有差异；体素下采样只是
    # 把它们混在一起，噪声照旧传递到泊松，表现为**表面碎/毛刺**。
    #
    # 做法：把同一体素内**一致**的法线取平均，用平均法线替换该体素内所有点。
    # 关键细节是**符号消歧**：法线的正负号本身是任意的（n 与 -n 表示同一平面），
    # 直接相加会让相反的符号互相抵消、得到接近零的法线 —— 这正是历史上
    # "法线翻转 ~35%" 的成因。因此先与体素平均值对齐符号，再平均。
    if nrm is not None and len(out_pts):
        inv_k = inv[keep]
        nk = np.array(out_nrm, dtype=np.float64, copy=True)
        # 先做一次不带符号的平均，用来判断"多数派朝向"（消歧基准）。
        # ⚠️ 不能复用上面的 `dot`：那个变量只在 normal_dot_min > -1.0 时才被赋值，
        #    在这里引用会 UnboundLocalError（实测踩到）。
        acc0 = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(acc0, inv_k, nk)
        ref = acc0[inv_k]
        # 与多数派反向的点先把符号翻过来，再加入平均
        flip = (nk * ref).sum(1) < 0
        if flip.any():
            nk[flip] = -nk[flip]
        acc2 = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(acc2, inv_k, nk)
        cnt2 = np.bincount(inv_k, minlength=len(counts)).astype(np.float64)
        mean2 = acc2 / np.maximum(cnt2, 1.0)[:, None]
        out_nrm = mean2[inv_k]

    out_stats = {
        "points_in": n,
        "points_out": int(keep.sum()),
        "kept_ratio": float(keep.mean()) if n else 0.0,
        "dropped_few_views": dropped_views,
        "dropped_normal_conflict": dropped_normal,
        "voxel": float(voxel),
        "min_views": int(min_views),
        "views_total": int(len(np.unique(vid))),
    }
    return (out_pts,
            out_nrm,
            (col[keep] if col is not None else None),
            out_stats)


def backproject_depth_to_points(rendered, c2ws, view_idx, ks, src_wh,
                                width: int, height: int,
                                depth_stride: int = 2,
                                min_alpha: float = 0.5, erode_px: int = 1,
                                grad_thresh_ratio: float = 0.06,
                                max_depth_ratio: float = 0.0,
                                colors_out: bool = True,
                                return_view_id: bool = False):
    """把多视角渲染出的深度图**反投影**成"紧贴表面"的稠密点云。

    这是整条网格流水线的**核心步骤**（用户给的社区标准流程第 1 步）：
    3DGS 的高斯中心是"分布在表面附近的三维体"，直接拿去重建会得到
    厚壳/鼓包；而深度反投影给出的是**薄薄一层贴在表面上**的点，
    每个点的法线天然就是"从相机指向表面"的方向。

    过滤：
      * `min_alpha` 不透明度阈值
      * `erode_px` 对 alpha 做腐蚀，去掉星芒状漂浮高斯
      * `grad_thresh_ratio` 剔除与邻域中值深度差异过大的像素（深度跳变）
      * `depth_stride` 像素采样步长（控制点数，避免几百万点拖慢泊松）

    返回 (points[N,3], colors[N,3], normals[N,3] 或 None, 统计)；
    `return_view_id=True` 时统计里会多一个 "view_id"（每个点来自哪个视角的索引），
    供 `filter_by_multiview_consistency` 做多视角投票。
    """
    import numpy as np

    pts_all, col_all, nrm_all, vid_all = [], [], [], []
    n_used = 0
    kept = 0
    total = 0
    for k, i in enumerate(view_idx):
        rgb, depth, alpha = rendered[k]
        valid, _ = filter_depth(depth, alpha, min_alpha=max(min_alpha, 0.5),
                                erode_px=erode_px,
                                grad_thresh_ratio=grad_thresh_ratio,
                                extent=max(float(np.nanmax(depth)), 1e-6))
        if max_depth_ratio > 0:
            valid &= depth < max_depth_ratio
        if not valid.any():
            continue
        vs, us = np.nonzero(valid)
        total += len(vs)
        if depth_stride > 1:
            sel = ((vs % depth_stride) == 0) & ((us % depth_stride) == 0)
            vs, us = vs[sel], us[sel]
        if len(vs) == 0:
            continue
        kept += len(vs)

        Ks = scale_intrinsics(ks[i], src_wh, (width, height))
        fx, fy, cx, cy = Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2]
        z = depth[vs, us].astype(np.float64)
        x = (us - cx) / fx * z
        y = (vs - cy) / fy * z
        cam_pts = np.stack([x, y, z], axis=1)

        # ---- 表面法线：**从深度图的梯度求**（社区标准做法，2DGS / GOF 同源）----
        #
        # ⚠️ 这里历史上错过**两次**，两次都直接损害网格质量，务必看清推导：
        #
        # 错误一（已修）：原实现写的是 `cam_n = cam_pts / ||cam_pts||` ——
        #   那是**视线方向（相机->像素的射线）**，不是表面法线。视线方向指向
        #   "穿过表面继续往外"，而且**同一表面点从不同视角看方向完全相反**。
        #   泊松靠法线定朝向，喂这种法线时指示函数无解 -> "填满轮廓的碎面"
        #   （实测轮廓 IoU 0.837 但邻域法线一致仅 0.0022、翻转 50%）。
        #
        # 错误二（本轮修）：改成"深度梯度切向量"后，把 P_v 的 z 分量写成了常数 1。
        #   正确推导：参数化 P(u,v) = z(u,v)·K⁻¹[u,v,1]，记 m = (x/z, y/z, 1)，
        #       P_u = z_u·m + z·(1/fx, 0, 0)
        #       P_v = z_v·m + z·(0, 1/fy, 0)
        #       n   = P_u × P_v
        #   代入**正对相机的平面**（z_u = z_v = 0）应得 -z²·fx·fy·ẑ ∝ [0,0,-1]。
        #   而错误写法会给 [0, 0.447, -0.894]（偏离真实方向 26.6°）——
        #   注意它**不退化**，所以既不会报错也不会被兜底掩盖，只能靠单测发现。
        z_img = depth.astype(np.float64)
        zu_img = np.zeros_like(z_img)
        zv_img = np.zeros_like(z_img)
        zu_img[:, 1:-1] = 0.5 * (z_img[:, 2:] - z_img[:, :-2])
        zv_img[1:-1, :] = 0.5 * (z_img[2:, :] - z_img[:-2, :])
        zu = zu_img[vs, us]
        zv = zv_img[vs, us]
        zc = np.maximum(z, 1e-12)
        m = np.stack([x / zc, y / zc, np.ones_like(z)], axis=1)
        pu = zu[:, None] * m + z[:, None] * np.stack(
            [np.full_like(z, 1.0 / fx), np.zeros_like(z), np.zeros_like(z)], axis=1)
        pv = zv[:, None] * m + z[:, None] * np.stack(
            [np.zeros_like(z), np.full_like(z, 1.0 / fy), np.zeros_like(z)], axis=1)
        cam_n = np.cross(pu, pv)
        nl = np.linalg.norm(cam_n, axis=1, keepdims=True)
        good_n = nl[:, 0] > 1e-12
        cam_n = np.divide(cam_n, np.where(nl > 1e-12, nl, 1.0))
        # 数值退化时（极少）退回视线方向作兜底
        if not good_n.all():
            fallback = cam_pts / np.maximum(np.linalg.norm(cam_pts, axis=1, keepdims=True), 1e-12)
            cam_n[~good_n] = fallback[~good_n]

        # 背面剔除：**可见**表面的法线必然朝向相机（相机在原点、看向 +Z），
        # 因此 `dot(cam_n, cam_pts) > 0` 才是"朝向相机"。
        #
        # ⚠️ 这里曾经写成 `dot(cam_n, -cam_pts) > 0`（方向反了），后果是
        # **每个视角的全部点都被剔除 -> 反投影恒为 0 点 -> 网格导出静默失败**。
        # 实测（两个数据集完全一致，所以不是个例）：
        #   dot(cam_n, -cam_pts) > 0 : 0 / 84509
        #   dot(cam_n, +cam_pts) > 0 : 84509 / 84509
        #   n_z > 0（可见表面应为正）: 92%
        # 注意它**不报错**：`poisson_mesh` 之前的 `len(bp_pts) < 1000` 只是 warn，
        # 所以表现为"能跑完但没有网格产物"。已在导出阶段加了硬失败（见 export.py）。
        facing = np.einsum("ij,ij->i", cam_n, cam_pts) > 0
        if not facing.any():
            continue
        if not facing.all():
            vs, us, z = vs[facing], us[facing], z[facing]
            cam_pts, cam_n = cam_pts[facing], cam_n[facing]
        # 统一朝向相机（保证同一表面在不同视角给出同向法线）
        cam_n = np.where((np.einsum("ij,ij->i", cam_n, cam_pts) > 0)[:, None],
                         cam_n, -cam_n)

        c2w = np.asarray(c2ws[i], dtype=np.float64)
        R, t = c2w[:3, :3], c2w[:3, 3]
        pts_all.append(cam_pts @ R.T + t)
        nrm_all.append(cam_n @ R.T)
        vid_all.append(np.full(len(cam_pts), int(i), dtype=np.int64))
        if colors_out:
            col_all.append((rgb[vs, us] * 255.0).astype(np.float64))
        n_used += 1

    if not pts_all:
        return np.zeros((0, 3)), np.zeros((0, 3)), None, {"views_used": 0}
    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(col_all, axis=0) if col_all else None
    nrms = np.concatenate(nrm_all, axis=0) if nrm_all else None
    stats = {"views_used": n_used, "pixels": total, "points": len(pts)}
    if return_view_id and vid_all:
        stats["view_id"] = np.concatenate(vid_all, axis=0)
    # 同一表面点会被多个视角重复采到：体素下采样前先去掉完全重合的点
    return pts, cols, nrms, stats


def voxel_downsample(points, colors=None, normals=None, voxel: float = 0.01):
    """体素下采样：均匀化点密度，防止局部过密让泊松过拟合。

    社区标准流程第 2 步。深度反投影出来的点密度**严重不均**：
    近处视角每像素一个点、远处几个视角叠加，密度差可达百倍。
    泊松对密度敏感，所以必须先均匀化。

    这里用 numpy 实现（按体素索引取均值），避免为了一个下采样引入 open3d 依赖。
    同体素内的点取坐标/颜色/法线的平均。
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0 or voxel <= 0:
        return pts, colors, normals
    keys = np.floor(pts / float(voxel)).astype(np.int64)
    # 用唯一化 + 反查分组，比 Python 循环快几个数量级
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    out = np.zeros((len(cnt), 3), dtype=np.float64)
    np.add.at(out, inv, pts)
    out /= cnt[:, None]

    def _avg(arr):
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float64)
        acc = np.zeros((len(cnt), a.shape[1]), dtype=np.float64)
        np.add.at(acc, inv, a)
        return acc / cnt[:, None]

    return out, _avg(colors), _avg(normals)


def mesh_topology(mesh) -> dict:
    """网格拓扑体检：边界边 / 非流形边 / **封闭判据** / 自交。

    为什么不用 Open3D 的 `is_watertight()` 作为唯一判据：
    它**还要求无自交**，而泊松/TSDF 结果几乎必然自交 ⇒ 永远 False，失去区分度。
    本项目把"封闭"定义为：**无边界边 + 无非流形边 + 顶点流形**；自交单独报告。
    """
    import numpy as np

    from collections import defaultdict

    tris = np.asarray(mesh.triangles, dtype=np.int64)
    cnt: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            cnt[(min(e), max(e))] += 1
    boundary = sum(1 for n in cnt.values() if n == 1)
    nonman = sum(1 for n in cnt.values() if n > 2)
    try:
        vman = bool(mesh.is_vertex_manifold())
    except Exception:  # noqa: BLE001
        vman = True
    out = {
        "triangles": int(len(tris)),
        "vertices": int(len(np.asarray(mesh.vertices))),
        "boundary_edges": int(boundary),
        "nonmanifold_edges": int(nonman),
        "vertex_manifold": vman,
        "closed": bool(boundary == 0 and nonman == 0 and vman),
    }
    try:
        out["open3d_watertight"] = bool(mesh.is_watertight())
    except Exception:  # noqa: BLE001
        out["open3d_watertight"] = None
    try:
        out["self_intersecting"] = bool(mesh.is_self_intersecting())
    except Exception:  # noqa: BLE001
        out["self_intersecting"] = None
    return out


def fill_holes(mesh, max_loop_edges: int = 4000, log=None):
    """闭合网格上的孔洞（**水密化的关键一步**）。

    ## 为什么需要（2026-10-25）

    实测 IMG5 的 TSDF 网格 `boundary_edge_ratio` 只有 0.0225，但**视觉上仍是"空壳"**：
    边界边少 ≠ 封闭。真正判据是 `is_watertight()`，它要求**每条边恰好被两个面共享**。
    薄壳结构（屋檐/墙板）在 TSDF 里常常留下长条状缺口 —— 面积占比很小，
    但让网格无法作为实体交付（3D 打印、游戏引擎碰撞、体积计算全部失败）。

    Open3D 的 legacy `TriangleMesh` **没有** `fill_holes`（只有张量网格才有），
    所以这里自己实现：

      1. 统计每条边被多少个面使用；**只被 1 个面使用**的是边界边；
      2. 把边界边串成闭环（每个边界顶点通常恰好两条边界边）；
      3. 每个闭环用**以质心为扇心**的三角扇封闭（新增 1 个顶点 + k 个三角形）——
         对"近似平面的小孔"足够好，且能保证拓扑闭合；
      4. 超过 `max_loop_edges` 的巨环**不补**（那通常不是孔洞而是模型边界，
         强行补会糊上一大片假面），只统计并在日志里说明。

    返回 `(mesh, stats)`；stats 记录补了多少环/多少面，便于验收与横向比较。
    """
    import numpy as np
    import open3d as o3d

    stats = {"loops": 0, "filled_loops": 0, "skipped_loops": 0,
             "added_triangles": 0, "boundary_edges_before": 0, "watertight_before": False}
    if len(mesh.triangles) == 0:
        return mesh, stats

    topo_before = mesh_topology(mesh)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    stats["watertight_before"] = bool(topo_before.get("closed"))
    stats["boundary_edges_before"] = int(topo_before["boundary_edges"])

    # ---- 1) 边界边 ----
    from collections import defaultdict

    cnt: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            cnt[(min(e), max(e))] += 1
    boundary = [e for e, n in cnt.items() if n == 1]
    stats["boundary_edges_before"] = len(boundary)
    if not boundary:
        # 没有洞 ⇒ 补洞前后拓扑相同（不能直接 return，否则调用方拿不到 closed 字段）
        stats.update({"boundary_edges_after": 0,
                      "nonmanifold_edges_after": int(topo_before["nonmanifold_edges"]),
                      "closed_after": bool(topo_before["closed"]),
                      "open3d_watertight": topo_before.get("open3d_watertight"),
                      "self_intersecting": topo_before.get("self_intersecting")})
        return mesh, stats

    # 记录每条边界边在**原面里**的方向 (a->b)：补洞三角形必须反向 (b->a)，
    # 否则相邻面共享边同向 ⇒ Open3D `is_watertight()` 仍为 False
    # （实测：边界边已归零、edge/vertex manifold 都 True，却依然 watertight=False，
    #  就是朝向不一致导致的）。
    bdir: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            k = (min(u, v), max(u, v))
            if cnt[k] == 1:
                bdir[k] = (int(u), int(v))

    # ---- 2) 串环 ----
    #
    # 用**标准边界行走**：从任一未使用的边界边出发，每步选一条**未使用**的邻接边，
    # 直到回到起点。第一版用"跳过 prev + 查 visited"的写法，会在度数>2 的顶点处
    # 走出**跨越两个环**的假环，于是扇面与真实面共享边成 3 个面 ⇒ 被后续清理删光
    # （实测：面数 282088→282088，一个都没补上）。这是必须用"边"而不是"顶点"做状态的典型。
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)

    used_edges: set[tuple[int, int]] = set()

    def _ek(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    loops: list[list[int]] = []
    for a, b in boundary:
        if _ek(a, b) in used_edges:
            continue
        loop = [a]
        prev, cur = a, b
        used_edges.add(_ek(a, b))
        closed = False
        while True:
            if cur == a:
                closed = True
                break
            loop.append(cur)
            nxt = None
            for cand in adj[cur]:
                if cand == prev:
                    continue
                if _ek(cur, cand) in used_edges:
                    continue
                nxt = cand
                break
            if nxt is None:
                break
            used_edges.add(_ek(cur, nxt))
            prev, cur = cur, nxt
        if closed and len(loop) >= 3:
            loops.append(loop)
        else:
            stats["skipped_loops"] += 1

    stats["loops"] = len(loops)
    if not loops:
        return mesh, stats

    # ---- 3) 以质心为扇心补面 ----
    new_verts: list[np.ndarray] = []
    new_tris: list[tuple[int, int, int]] = []
    base = len(verts)
    for loop in loops:
        if len(loop) > max_loop_edges:
            stats["skipped_loops"] += 1
            if log:
                log(f"跳过超长边界环（{len(loop)} 条边 > {max_loop_edges}）："
                    f"多半是模型边界，不是孔洞")
            continue
        c = verts[loop].mean(axis=0)
        ci = base + len(new_verts)
        new_verts.append(c)
        for k in range(len(loop)):
            p, q = int(loop[k]), int(loop[(k + 1) % len(loop)])
            # 反向于原面方向：原面是 p->q 则补 (ci, q, p)
            stored = bdir.get((min(p, q), max(p, q)), (p, q))
            if stored == (p, q):
                new_tris.append((ci, q, p))
            else:
                new_tris.append((ci, p, q))
        stats["filled_loops"] += 1

    if not new_tris:
        stats.update({"boundary_edges_after": int(topo_before["boundary_edges"]),
                      "nonmanifold_edges_after": int(topo_before["nonmanifold_edges"]),
                      "closed_after": bool(topo_before["closed"]),
                      "open3d_watertight": topo_before.get("open3d_watertight"),
                      "self_intersecting": topo_before.get("self_intersecting")})
        return mesh, stats

    all_verts = np.vstack([verts, np.asarray(new_verts, dtype=np.float64)]) if new_verts else verts
    all_tris = np.vstack([tris, np.asarray(new_tris, dtype=np.int64)])
    out = mesh
    out.vertices = o3d.utility.Vector3dVector(all_verts)
    out.triangles = o3d.utility.Vector3iVector(all_tris)
    stats["added_triangles"] = len(new_tris)

    # ---- 4) 焊接（只做安全的两种）----
    #
    # ⚠️ 这里**不能**调用 `remove_non_manifold_edges()`：它会把"与已有面共享同一条边"的
    # 新面整片删掉，实测让补洞完全失效（282088 面一个没补上）。补洞的正确判据是
    # **边界边是否归零**，而不是"强行让网格变成流形"。
    try:
        out.remove_duplicated_vertices()
        out.remove_unreferenced_vertices()
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"补洞后的焊接失败（{type(exc).__name__}: {exc}）")
    try:
        out.compute_vertex_normals()
    except Exception:  # noqa: BLE001
        pass
    # 统计补洞后的残余边界（理想为 0）
    tris2 = np.asarray(out.triangles, dtype=np.int64)
    cnt2: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in tris2:
        for e in ((a, b), (b, c), (c, a)):
            cnt2[(min(e), max(e))] += 1
    stats["boundary_edges_after"] = sum(1 for n in cnt2.values() if n == 1)
    stats["nonmanifold_edges_after"] = sum(1 for n in cnt2.values() if n > 2)

    # ⚠️ Open3D 的 `is_watertight()` **还要求无自交**（实测：本网格每条边都被恰好 2 个面共享、
    # edge/vertex manifold 都为 True，只因泊松结果自交就返回 False）。泊松/TSDF 网格几乎必然自交，
    # 所以真正可用于交付判定的是下面这个**封闭判据**：
    #     无边界边 && 无非流形边 && 顶点流形
    # 自交单独报告（消除它需要重网格化，属于另一项工作）。
    try:
        stats["closed_after"] = bool(
            stats["boundary_edges_after"] == 0
            and stats["nonmanifold_edges_after"] == 0
            and out.is_vertex_manifold())
    except Exception:  # noqa: BLE001
        stats["closed_after"] = stats["boundary_edges_after"] == 0
    try:
        stats["open3d_watertight"] = bool(out.is_watertight())
    except Exception:  # noqa: BLE001
        stats["open3d_watertight"] = False
    try:
        stats["self_intersecting"] = bool(out.is_self_intersecting())
    except Exception:  # noqa: BLE001
        stats["self_intersecting"] = None
    return out, stats


def smooth_mesh(mesh, method: str = "taubin", iterations: int = 5):
    """网格平滑：去掉 TSDF/泊松产生的阶梯状噪点。

    Taubin（λ/μ）比拉普拉斯好：拉普拉斯会整体收缩体积，
    Taubin 用一次负向步长抵消收缩，更适合"要保持形状"的重建结果。
    """
    try:
        if method == "taubin":
            return mesh.filter_smooth_taubin(number_of_iterations=int(iterations))
        if method == "laplacian":
            return mesh.filter_smooth_laplacian(number_of_iterations=int(iterations))
    except Exception:  # noqa: BLE001
        pass
    return mesh


def simplify_mesh(mesh, target_triangles: int = 0, voxel: float = 0.0):
    """QEM 二次误差简化：把几百万面降到可用的量级（社区标准流程第 4 步）。

    open3d 用的是 `simplify_quadric_decimation`（QEM），
    能在大幅减少面数的同时较好保留形状。
    """
    n = len(mesh.triangles)
    if target_triangles and n > target_triangles:
        try:
            return mesh.simplify_quadric_decimation(int(target_triangles))
        except Exception:  # noqa: BLE001
            pass
    if voxel and voxel > 0:
        try:
            return mesh.simplify_vertex_clustering(float(voxel))
        except Exception:  # noqa: BLE001
            pass
    return mesh


def drop_unseen_faces(mesh, c2ws, ks, src_whs, min_views: int = 1, log=None):
    """删除"从未被任何相机看到"的网格面 —— 泊松重建的内部垃圾几何。

    ## 为什么需要（实测数据，img3-crop）

    泊松重建会把物体包成"实心"：表面之外，内部也生成了面。
    统计每个面被多少视角看到（投影落在图像内 + 面朝向该相机）：

    | 被看到次数 | 面数 | 占比 |
    |---|---|---|
    | **0** | **15,190** | **25.6%** |
    | 1~2 | 2,022 | 3.4% |
    | 3~9 | 5,958 | 10.0% |
    | 10~49 | 12,304 | 20.7% |
    | >=50 | 23,839 | 40.2% |

    而这些"从未被看到"的面：包围盒 [1.82,1.05,0.84] ≈ 全网格 [1.86,1.08,0.86]，
    到中心距离 0.46 ≈ 0.49 ⇒ **散布在物体内部**（不是背面遮挡）。

    它们的害处：
      1. 在 UV 展开里**白占 texture 空间**却烘不出颜色 ⇒ 纹理黑洞
         （实测 43.7% 的 chart 内部 texel 是黑的）；
      2. 让网格法线混乱、体积虚胖；
      3. 对渲染零贡献（本来就看不见）。

    ## 判据为什么可靠

    ⚠️ 与"落在前景掩码内的比例"那个错误判据的对比：
    掩码判据在**转台**素材上原理错误（物体旋转使同一面在掩码内外来回变），
    我曾据此误删 28.7% 顶点，回原始帧才发现删的是物体细节。
    本判据是"**是否被相机看到过**"，与掩码无关，不受转台/绕行影响。

    ## 参数

    `c2ws` [V,4,4] 相机到世界，`ks` [V,3,3] 内参，`src_whs` [(W,H)] 图像尺寸。
    `min_views` 保留阈值（默认 1：只要被看到过一次就保留）。
    """
    import numpy as np
    import open3d as o3d

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(tris) == 0:
        return mesh, {"faces_before": 0, "faces_kept": 0, "removed": 0}

    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fc = (p0 + p1 + p2) / 3.0
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    # 先统一朝外，否则 facing 判据会把内部面误判成"可见"
    c = verts.mean(0)
    d = fc - c
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    flip = np.einsum("ij,ij->i", fn, d) < 0
    if flip.any():
        tris[flip] = tris[flip][:, [0, 2, 1]]
        fn[flip] = -fn[flip]

    seen = np.zeros(len(tris), dtype=np.int64)
    for i in range(len(c2ws)):
        W, H = int(src_whs[i][0]), int(src_whs[i][1])
        if W <= 0 or H <= 0:
            continue
        w2c = np.linalg.inv(np.asarray(c2ws[i], dtype=np.float64))
        pc = fc @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        ok = z > 1e-6
        u = (pc @ np.asarray(ks[i], dtype=np.float64)[0, :3]) / np.maximum(z, 1e-9)
        v = (pc @ np.asarray(ks[i], dtype=np.float64)[1, :3]) / np.maximum(z, 1e-9)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        cam = np.asarray(c2ws[i], dtype=np.float64)[:3, 3]
        vd = cam - fc
        vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-9)
        ok &= np.einsum("ij,ij->i", fn, vd) > 0.1
        seen += ok.astype(np.int64)

    keep = seen >= max(1, int(min_views))
    n_removed = int((~keep).sum())
    if n_removed == 0:
        return mesh, {"faces_before": int(len(tris)), "faces_kept": int(len(tris)),
                      "removed": 0}
    if log:
        log(f"删除从未被看到的内部面：{n_removed} / {len(tris)}"
            f"（{n_removed/len(tris)*100:.1f}%）—— 它们烘不出纹理、对渲染零贡献")

    used_v = np.zeros(len(verts), dtype=bool)
    used_v[tris[keep].ravel()] = True
    new_index = -np.ones(len(verts), dtype=np.int64)
    new_index[used_v] = np.arange(int(used_v.sum()))
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(verts[used_v])
    out.triangles = o3d.utility.Vector3iVector(new_index[tris[keep]])
    if mesh.has_vertex_colors():
        out.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_colors)[used_v])
    out.compute_vertex_normals()
    return out, {"faces_before": int(len(tris)), "faces_kept": int(keep.sum()),
                 "removed": n_removed}


def postprocess_mesh(mesh, target_triangles: int = 0, smooth_iters: int = 5,
                     smooth_method: str = "taubin", presimplify_ratio: float = 10.0,
                     log=None):
    """网格后处理链：**预简化 -> 平滑 -> 最终 QEM**，返回 (mesh, 步骤记录)。

    ⚠️ 顺序是这个函数的**核心语义**，不要调换：

    原来的顺序是"平滑全体 -> QEM"，在稍大的网格上会卡到不可用。实测：
    泊松 depth=10 产出 **2771 万面 / 1393 万顶点** 时，"平滑（taubin × 5）"
    跑了 **61 分钟仍未完成**、内存 7.9 GB（本机只有 15.2 GB）；
    而同一模型 depth=9（485 万面）全流程只要 2.5 分钟。
    原因是 Taubin/拉普拉斯平滑在 CPU 上对**全体顶点及其邻接**迭代，
    代价随顶点数超线性增长，而 QEM 之后还要再全量遍历一次。

    正确做法（社区通行）：平滑**不要在原始高密度网格上做**。
    先 QEM 把规模压到 `target × presimplify_ratio`，再平滑去噪，最后 QEM 到目标面数。
    这样平滑与最终简化的规模都由目标面数界定，泊松 depth 调高也不会失控。

    `presimplify_ratio <= 1` 等于关闭预简化（保留旧行为，供需要时对照）。

    已实测：QEM（open3d `simplify_quadric_decimation`）与 Taubin 平滑
    **全程保留顶点色与法线**（端到端颜色均值差异 0.0024，法线保持单位化），
    所以预简化不会让预览变灰、也不会破坏光照。
    """
    steps: list[str] = []
    target = int(target_triangles or 0)
    ratio = float(presimplify_ratio or 0.0)

    # ⑤ 预简化：只为把平滑的规模降下来
    if target > 0 and ratio > 1.0 and len(mesh.triangles) > target * ratio:
        before = len(mesh.triangles)
        mesh = simplify_mesh(mesh, target_triangles=int(target * ratio))
        steps.append(f"预简化（QEM，为平滑降规模）：{before} -> {len(mesh.triangles)} 面")

    # ⑥ 平滑（去噪）：此时规模已被界定
    it = int(smooth_iters or 0)
    if it > 0:
        mesh = smooth_mesh(mesh, method=str(smooth_method or "taubin"), iterations=it)
        steps.append(f"平滑（{smooth_method} × {it}）@{len(mesh.triangles)} 面")

    # ⑦ 最终 QEM 到目标面数
    if target > 0 and len(mesh.triangles) > target:
        before = len(mesh.triangles)
        mesh = simplify_mesh(mesh, target_triangles=target)
        steps.append(f"QEM 简化：{before} -> {len(mesh.triangles)} 面")

    # ⑧ 重算法线（**必须**）
    # ⚠️ 实测坑：open3d 的 `simplify_quadric_decimation` 与 Taubin 平滑都会让
    # **顶点法线不再是单位长度**（实测偏差达 0.027，即 2.7%）。
    # 而导出的 GLB 直接把这份法线写给查看器做逐像素光照 —— 非单位法线会让
    # `dot(n, L)` 偏小/偏大，表现为整体偏暗或明暗失真。
    # 这里统一重算并单位化，保证"导出即正确"，不依赖下游各自补救。
    try:
        mesh.compute_vertex_normals()
        steps.append("重算顶点法线（QEM/平滑后单位化）")
    except Exception:  # noqa: BLE001
        pass

    if log is not None:
        for s in steps:
            log(s)
    return mesh, steps


def unwrap_uv(mesh, tex_size: int = 2048):
    """用 xatlas 做 UV 展开，返回 (新网格, 顶点 UV, 索引)。

    UV 展开是纹理烘焙的前提：要把三维表面"摊平"到二维图集上。
    open3d 没有自带展开（没有 `compute_uvatlas`），所以用社区标准的 **xatlas**
    （纯 CPU、无需网络、wheel 直接装）。

    注意 xatlas 会**重新索引顶点**（同一顶点在不同 UV 岛里要拆开），
    所以返回的是新网格 + 新 UV，不要拿旧索引去对应。
    """
    import numpy as np
    import open3d as o3d
    import xatlas

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris = np.asarray(mesh.triangles, dtype=np.uint32)
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError("网格为空，无法做 UV 展开")

    vmapping, new_tris, uvs = xatlas.parametrize(verts, tris)
    if vmapping is None or len(new_tris) == 0:
        raise RuntimeError("xatlas 展开失败")

    new_verts = verts[vmapping]
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(new_verts.astype(np.float64))
    out.triangles = o3d.utility.Vector3iVector(np.asarray(new_tris, dtype=np.int32))
    if mesh.has_vertex_colors():
        cols = np.asarray(mesh.vertex_colors)
        out.vertex_colors = o3d.utility.Vector3dVector(cols[vmapping])
    out.compute_vertex_normals()

    # ⚠️ 两套 UV 约定要分清：
    #   xatlas 返回的是**逐顶点** UV，长度 = 新顶点数
    #   open3d 的 `triangle_uvs` 要的是**逐三角形-角** UV，长度 = 面数 × 3
    # 直接赋值会报 "UV 数量不匹配：113923 != 236712"（我踩过）。
    # 正确做法是按三角形索引把逐顶点 UV 展开成逐角。
    uv = np.asarray(uvs, dtype=np.float64)
    new_tris = np.asarray(new_tris, dtype=np.int64)
    if uv.shape[0] == len(new_verts):
        corner_uv = uv[new_tris].reshape(-1, 2)
    elif uv.shape[0] == len(new_tris) * 3:
        corner_uv = uv.reshape(-1, 2)
    else:
        raise RuntimeError(f"xatlas 返回的 UV 数量无法识别：{uv.shape[0]}"
                           f"（顶点 {len(new_verts)}，面 {len(new_tris)}）")
    out.triangle_uvs = o3d.utility.Vector2dVector(corner_uv)
    out.triangle_material_ids = o3d.utility.IntVector(
        np.zeros(len(new_tris), dtype=np.int32))
    return out, new_tris, corner_uv


def bake_texture_from_views(mesh, uv_tris, uv, images, c2ws, ks, src_whs,
                            tex_size: int = 2048, batch: int = 200,
                            margin_px: int = 2, max_views: int = 60,
                            bake_reject_black: bool = True,
                            masks: list | None = None):
    """从多视角图像给网格烘焙一张漫反射贴图。

    做法（社区常用思路，不依赖可微渲染器）：
      1. xatlas 展开得到 UV 图集
      2. 遍历图集里的三角形，在 UV 空间的包围盒内逐像素填色
      3. 每个像素用重心坐标反推到三维点，投影到**所有视角**，
         在"可见且拍摄方向与法线夹角最小"的视角上采样颜色（最接近正对表面）
      4. 边缘膨胀几次（dilate），避免采样时出现黑缝

    ⚠️ 性能坑：一开始写成"逐三角形 + 逐视角"两层 Python 循环，
    8 万面 × 29 视角要跑十几分钟。现在改成**按批处理**：
    一批三角形（默认 200）先把所有像素铺平成一个大数组，
    再对全部视角做一次**向量化**投影/采样，速度提升一到两个数量级。

    返回 (H,W,3) uint8 贴图 与 统计。
    """
    import numpy as np

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(uv_tris, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 3, 2)
    n_face = len(tris)
    normal = np.asarray(mesh.vertex_normals, dtype=np.float64)

    # 视角太多时均匀抽稀（烘焙不需要全部视角，正对表面才是关键）
    if max_views and len(c2ws) > max_views:
        idx = np.linspace(0, len(c2ws) - 1, max_views).round().astype(int)
    else:
        idx = np.arange(len(c2ws))

    cams = []
    for i in idx:
        c2w = np.asarray(c2ws[i], dtype=np.float64)
        cams.append((np.linalg.inv(c2w), c2w[:3, 3],
                     np.asarray(ks[i], dtype=np.float64),
                     src_whs[i][0], src_whs[i][1],
                     images[i].astype(np.float32) / 255.0))

    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    filled = np.zeros((tex_size, tex_size), dtype=bool)
    # 因采到黑背景而被拒绝的采样数（写进统计，避免行为变化被静默吞掉）
    n_rejected = 0

    # 大三角形优先（小碎面容易被覆盖，且大面贡献的像素多）
    area = np.abs(
        (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1]) -
        (uv[:, 2, 0] - uv[:, 0, 0]) * (uv[:, 1, 1] - uv[:, 0, 1])) * 0.5
    order = np.argsort(-area)
    uv_px = uv * (tex_size - 1)

    for s in range(0, n_face, int(batch)):
        fid = order[s:s + int(batch)]
        P_list, N_list, px_list, py_list = [], [], [], []
        for f in fid:
            t = uv_px[f]
            x0 = max(int(np.floor(t[:, 0].min())) - 1, 0)
            x1 = min(int(np.ceil(t[:, 0].max())) + 1, tex_size - 1)
            y0 = max(int(np.floor(t[:, 1].min())) - 1, 0)
            y1 = min(int(np.ceil(t[:, 1].max())) + 1, tex_size - 1)
            if x1 < x0 or y1 < y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
            a, b, c = t[0], t[1], t[2]
            v0, v1 = b - a, c - a
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < 1e-12:
                continue
            v2 = pts - a
            w1 = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
            w2 = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
            w0 = 1.0 - w1 - w2
            inside = (w0 >= -0.002) & (w1 >= -0.002) & (w2 >= -0.002)
            if not inside.any():
                continue
            w0, w1, w2 = w0[inside], w1[inside], w2[inside]
            i0, i1, i2 = tris[f]
            P_list.append(w0[:, None] * verts[i0] + w1[:, None] * verts[i1] + w2[:, None] * verts[i2])
            nrm = (w0[:, None] * normal[i0] + w1[:, None] * normal[i1] + w2[:, None] * normal[i2])
            N_list.append(nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9))
            px_list.append(pts[inside][:, 0].astype(np.int32))
            py_list.append(pts[inside][:, 1].astype(np.int32))

        if not P_list:
            continue
        P = np.concatenate(P_list, axis=0)
        N = np.concatenate(N_list, axis=0)
        px = np.concatenate(px_list)
        py = np.concatenate(py_list)

        best = np.full(len(P), -1.0)
        col = np.zeros((len(P), 3), dtype=np.float32)
        for ci, (w2c, ctr, K, iw, ih, img) in enumerate(cams):
            vc = P @ w2c[:3, :3].T + w2c[:3, 3]
            z = vc[:, 2]
            ok = z > 1e-6
            if not ok.any():
                continue
            u = (vc @ K[0, :3]) / np.maximum(z, 1e-9)
            v = (vc @ K[1, :3]) / np.maximum(z, 1e-9)
            ok &= (u >= 0) & (u < iw) & (v >= 0) & (v < ih)
            vd = P - ctr
            vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-9)
            facing = -(N * vd).sum(axis=1)
            ok &= facing > 0.15
            if not ok.any():
                continue
            sc = np.where(ok, facing, -1.0)
            take = sc > best
            if not take.any():
                continue
            sy = (img.shape[0] - 1) / max(ih - 1, 1)
            sx = (img.shape[1] - 1) / max(iw - 1, 1)
            ui = np.clip((u[take] * sx).astype(np.int32), 0, img.shape[1] - 1)
            vi = np.clip((v[take] * sy).astype(np.int32), 0, img.shape[0] - 1)

            # ---- 拒绝采到"黑背景"的像素 ----
            #
            # `mask.mode=black` 的训练图里背景被涂成**纯黑**（实测 85.4% 像素为黑）。
            # 网格表面在轮廓边缘的采样点会投影到那片黑上 ⇒ 采到 (0,0,0)
            # ⇒ 纹理出现黑斑。实测 final-v1：**黑色 texel 占 35.99%**。
            #
            # 判据：采样像素三通道最大值 <= 阈值即视为背景，拒绝该视角。
            # 阈值取 6（接近纯黑）：真实物体上的暗部通常 >10，
            # 而涂黑的背景是**精确的 0**。
            # `bake_reject_black` 可关闭（若素材本身就是深色物体、怕误杀）。
            # ⚠️ 这里的判据有优先级：**有掩码时以掩码为准**（原理正确），
            # 无掩码时才退回"像素接近纯黑"的近似判据。
            # 黑像素判据的缺陷：真实物体的暗部（阴影、深色石材）会被误杀；
            # 背景不是纯黑时完全失效。掩码没有这两个问题。
            _mask_arr = masks[ci] if (masks is not None and ci < len(masks)) else None
            if _mask_arr is not None or bake_reject_black:
                # ⚠️ 两个踩过的坑：
                #   1) `vi` / `ui` 的长度是 `take.sum()`（已筛选后的），不是原数组长度。
                #      用原索引访问会报 `IndexError: index 1259 is out of bounds`。
                #   2) **图像已归一化到 0~1**（`images[i].astype(np.float32) / 255.0`），
                #      所以阈值必须用 0~1 尺度。我第一版写 `> 6`（0~255 尺度）
                #      ⇒ 永远 False ⇒ **全部采样被拒、覆盖 0.0%**。
                #      选 6/255 ≈ 0.024 的原因：被涂黑的背景是**精确的 0**，
                #      而真实物体的暗部一般 >10/255。
                idx_take = np.nonzero(take)[0]
                pix = img[vi, ui]                      # 长度 = len(idx_take)
                if _mask_arr is not None:
                    # 掩码可能与图像尺寸不同（去畸变/缩放），先对齐
                    _m = _mask_arr
                    if _m.shape[0] != img.shape[0] or _m.shape[1] != img.shape[1]:
                        from PIL import Image as _PI
                        _m = np.asarray(
                            _PI.fromarray((_m.astype(np.uint8) * 255)).resize(
                                (img.shape[1], img.shape[0]), _PI.NEAREST)) > 127
                    # ⚠️ **掩码与"非黑"必须同时成立，不能只用掩码。**
                    #
                    # 实测教训：只用掩码时 `final-v1` 的黑色 texel 反而从 2.30%
                    # 涨到 **22.03%**。原因：掩码是在**去畸变之前**生成的，
                    # 与 `04_train/data/images`（已去畸变）有几像素错位 ⇒
                    # 被判"在物体内"的像素其实仍是黑背景 ⇒ 复制黑色 ⇒
                    # 后续扩散把黑色铺开。
                    #
                    # 因此：掩码负责"位置对"（原理正确的物体/背景划分），
                    # 非黑负责"内容对"（防止错位采样到黑背景）。两者取交集。
                    sel = _m[vi, ui] & (pix.max(axis=1) > (6.0 / 255.0))
                else:
                    sel = pix.max(axis=1) > (6.0 / 255.0)  # 非黑背景（0~1 尺度）
                if not sel.any():
                    continue
                its = idx_take[sel]
                # 未被采纳的像素**不更新** best，给别的视角机会
                col[its] = pix[sel]
                best[its] = sc[its]
                n_rejected += int((~sel).sum())
            else:
                col[take] = img[vi, ui]
                best[take] = sc[take]

        got = best >= 0
        if got.any():
            tex[py[got], px[got]] = col[got]
            filled[py[got], px[got]] = True

    # ---- 边缘膨胀 + **UV 区域全填充** ----
    #
    # 原实现只膨胀 `margin_px` 圈（默认 2）。但 chart **内部**的未填充孔洞
    # 往往有十几到几十像素宽，2 圈补不到 ⇒ 纹理里留下大片黑斑。
    # 实测 img3-crop：**43.7% 的 chart 内部 texel 是黑的**（模型表面可见黑斑）。
    #
    # 正确做法（生产级烘焙工具的标配）：把 **UV 覆盖区域全部填满** ——
    # 反复用"已填充像素的邻居均值"扩散，直到 UV 区域内没有未填充像素。
    # 未填充像素在 UV 区域外（padding）保持黑色，那是无害的。
    #
    # 第一步：光栅化 UV 覆盖区域（用三角形填充，与采样口径一致）
    uv_cov = np.zeros((tex_size, tex_size), dtype=bool)
    try:
        from PIL import Image as _PI, ImageDraw as _ID

        _img = _PI.new("L", (tex_size, tex_size), 0)
        _dr = _ID.Draw(_img)
        _uvp = uv * (tex_size - 1)
        for _f in range(n_face):
            _tri = _uvp[_f]
            _dr.polygon([(float(x), float(y)) for x, y in _tri], fill=255)
        uv_cov = np.asarray(_img) > 127
    except Exception:
        uv_cov = filled.copy()

    # 第二步：反复扩散，直到 UV 覆盖区域内全部填充（上限防死循环）
    for _ in range(max(1, int(margin_px)) + 512):
        todo = uv_cov & ~filled
        if not todo.any():
            break
        acc = np.zeros_like(tex)
        cnt = np.zeros(filled.shape, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sh_t = np.roll(np.roll(tex, dy, axis=0), dx, axis=1)
                sh_f = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
                acc += sh_t * sh_f[..., None]
                cnt += sh_f
        fill_now = (~filled) & (cnt > 0)
        if not fill_now.any():
            break
        tex[fill_now] = acc[fill_now] / cnt[fill_now][:, None]
        filled[fill_now] = True

    return ((np.clip(tex, 0, 1) * 255).astype(np.uint8),
            {"tex_size": tex_size, "coverage": float(filled.mean()),
             "views_used": len(cams),
             "samples_rejected_black": int(n_rejected),
             "mask_used": bool(masks is not None)})


def render_mesh_previews(mesh, out_dir, size: int = 512, views=None) -> list[dict]:
    """用 open3d 的离屏渲染器把网格渲成几张预览图。

    为什么要做：用户拿到一个 `model_mesh.obj` 其实无法判断质量好坏
    （要开 Blender 才知道）。点云那边我们已经有了真实渲染预览，
    网格这边同样需要，否则"网格导出"等于交了个黑盒。

    open3d 0.20 的 `OffscreenRenderer` 走 Filament（本机解析到 Vulkan 后端、
    使用 RTX 5060），无需窗口、可在无 GUI 环境跑。
    """
    import numpy as np
    import open3d as o3d
    from PIL import Image

    if views is None:
        # (名字, 方位角°, 仰角°)
        views = [("front", 0.0, 12.0), ("side", 90.0, 12.0),
                 ("top", 0.0, 70.0), ("iso", 45.0, 28.0)]

    bb = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bb.get_center(), dtype=np.float64)
    ext = np.asarray(bb.get_extent(), dtype=np.float64)
    # 取景：以"包围盒最大边长"为准（而不是对角线），再留 1.6 倍余量。
    # 踩坑：早期用 `对角线 * 1.25` 且 fov=60，相机离物体太远，渲出来是一片纯白。
    radius = float(np.max(ext)) * 1.6 + 1e-6

    renderer = o3d.visualization.rendering.OffscreenRenderer(size, size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    # ⚠️ `defaultLit` 需要**场景里有光源**，OffscreenRenderer 默认没有，
    # 不加灯会渲出一片纯白（我踩过：怎么调相机都没用，因为问题不在相机）。
    try:
        renderer.scene.scene.add_directional_light(
            "sun", [0.6, 0.6, -1.0], [1.0, 1.0, 1.0], 2.0)
        renderer.scene.scene.add_directional_light(
            "fill", [-0.6, 0.3, -0.5], [0.7, 0.7, 0.7], 1.0)
    except Exception:  # noqa: BLE001
        pass

    out: list[dict] = []
    for name, az_deg, el_deg in views:
        az, el = np.radians(az_deg), np.radians(el_deg)
        eye = center + np.array([
            radius * np.cos(el) * np.cos(az),
            radius * np.cos(el) * np.sin(az),
            radius * np.sin(el),
        ])
        renderer.setup_camera(60.0, center.tolist(), eye.tolist(), [0.0, 0.0, 1.0])
        img = renderer.render_to_image()
        arr = np.asarray(img)
        # 有效性自检：本机的 Filament 离屏渲染在无窗口环境下会返回**全黑**图
        # （实测 min=max=0）。与其交付一张全黑图让人误判"网格是空的"，
        # 不如检测出来并如实跳过 —— 并保留已经算好的网格统计。
        if arr.size == 0 or (int(arr.max()) - int(arr.min())) < 8:
            return []
        p = out_dir / f"preview_mesh_{name}.png"
        Image.fromarray(arr).save(p)
        kb = round(p.stat().st_size / 1024, 1)
        out.append({"kind": "preview_mesh", "path": p.name, "size_kb": kb,
                    "size_mb": round(kb / 1024, 3),
                    "desc": f"网格预览 {name} 视角（离屏渲染）"})
    return out


def poisson_mesh(points, colors, normals=None, depth: int = 9,
                 density_quantile: float = 0.0,
                 knn: int = 30, statistical_nb: int = 20, statistical_std: float = 2.0,
                 orient: bool = True, voxel_downsample_size: float = 0.0,
                 trim_distance_sigma: float = 2.5):
    """泊松表面重建：直接从高斯点云得到**闭合实体网格**。

    为什么需要这条路线（用户提醒后补上）：
    TSDF 融合依赖多视角深度一致性，对"视锥没覆盖到"和"薄壳开口"很敏感
    （实测碗类物体只能出一层内壁、边缘破碎）。而泊松重建是**全局隐式函数拟合**：

      * 只用点 + 法线，不要求深度图一致，对视角缺口更宽容；
      * 输出天然**闭合**（水密）的表面，"点云没有实感"这个诉求正好对上；
      * 是社区从点云出网格的标准做法（open3d / MeshLab 同款流程）。

    流程（社区标准）：
      1. 统计离群点剔除（去掉远处的漂浮高斯）
      2. 法线估计 + 一致定向（这一步决定成败：法线朝向乱了，泊松会出翻边/破洞）
      3. `create_from_point_cloud_poisson`（depth 越大越细，内存/时间指数上升）
      4. 按**密度分位数**裁掉低密度区域（泊松会在无数据处生成"鼓包"）
      5. 裁到主包围盒，必要时只留最大连通块

    法线优先用调用方给的（3DGS 的旋转四元数里就含朝向信息），
    没有则用 `estimate_normals` + `orient_normals_consistent_tangent_plane`。
    """
    import numpy as np
    import open3d as o3d

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 64:
        raise ValueError(f"点太少（{len(pts)}），无法做泊松重建")

    # 2) 体素下采样（可选）：均匀化密度，防止局部过密让泊松过拟合。
    #    深度反投影出来的点密度差异很大，这一步在"反投影路线"里很重要。
    if voxel_downsample_size and voxel_downsample_size > 0:
        pts, colors, normals = voxel_downsample(pts, colors, normals,
                                                 voxel=float(voxel_downsample_size))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        c = np.asarray(colors)
        if c.max() > 1.0:
            c = c / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(c.astype(np.float64), 0, 1))

    if normals is not None:
        n = np.asarray(normals, dtype=np.float64)
        if len(n) == len(pts):
            pcd.normals = o3d.utility.Vector3dVector(n)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    if orient and len(pcd.points) > 0:
        # 泊松对**法线朝向一致性**极敏感；深度反投影给的法线本来就大致朝外，
        # 但在凹面/薄壳处仍会翻转，所以再跑一次一致定向。
        try:
            pcd.orient_normals_consistent_tangent_plane(k=int(knn))
        except Exception:  # noqa: BLE001
            pass

    # 1) 统计离群点
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=int(statistical_nb), std_ratio=float(statistical_std))

    # 3) 泊松
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth))
    dens = np.asarray(densities)
    n_raw = len(mesh.triangles)

    # 4) 裁剪 A：**按到输入点的距离**剔除泊松"桥接"出来的面（社区标准做法）。
    #
    # 为什么必须加这一步：泊松是"求一个隐函数"，它在**没有观测数据的地方**
    # 也会给出解 —— 典型表现就是把两个本来不相邻的表面**桥接**起来，
    # 于是"该空的地方是实的"。用户实测反馈的正是这个现象。
    # 密度分位数（下面那条）治不了它：那是**全局**阈值，一刀切，
    # 会连同观测较少的侧面一起裁掉（实测 0.03 就把 IoU 从 0.68 砍到 0.32）。
    #
    # 正确判据是**局部的**：泊松的三角形顶点如果离最近的输入点很远，
    # 就说明那块面没有数据支撑，是凭空外推出来的。
    # 阈值取"输入点平均最近邻间距"的若干倍（自适应尺度，不用手填绝对距离）。
    # 4) 裁剪 A：**按到输入点的距离**剔除泊松"桥接"出来的面（社区标准做法）。
    #
    # 为什么必须加这一步：泊松是"求一个隐函数"，它在**没有观测数据的地方**
    # 也会给出解 —— 典型表现就是把两个本来不相邻的表面**桥接**起来，
    # 于是"该空的地方是实的"。用户实测反馈的正是这个现象。
    # 密度分位数（下面那条）治不了它：那是**全局**阈值，一刀切，
    # 会连同观测较少的侧面一起裁掉（实测 0.03 就把 IoU 从 0.68 砍到 0.32）。
    #
    # 正确判据是**局部的**：泊松的三角形顶点如果离最近的输入点很远，
    # 就说明那块面没有数据支撑，是凭空外推出来的。
    # 阈值取"输入点平均最近邻间距"的若干倍（自适应尺度，不用手填绝对距离）。
    trim_removed = 0
    if trim_distance_sigma and trim_distance_sigma > 0 and len(pcd.points) > 0:
        try:
            src = np.asarray(pcd.points, dtype=np.float64)
            # 输入点的典型间距（子采样估算，别在上百万点上算全量）
            step = max(1, len(src) // 2000)
            sub = src[::step][:2000]
            kd_all = o3d.geometry.KDTreeFlann(pcd)
            nn = []
            for p in sub:
                cnt, _, d2 = kd_all.search_knn_vector_3d(p, 2)
                if cnt >= 2:
                    nn.append(np.sqrt(max(d2[1], 0.0)))
            spacing = float(np.median(nn)) if nn else 0.0
            if spacing > 0:
                thr = spacing * float(trim_distance_sigma)
                verts = np.asarray(mesh.vertices, dtype=np.float64)
                keep = None
                # ⚠️ 必须**向量化**：逐顶点 `search_knn_vector_3d` 在 40 万顶点上
                # 慢到不可用（我在别处踩过同类坑）。scipy 的 cKDTree 批量查询是秒级：
                # 实测 20 万点建树 + 40 万次查询 0.08 秒。
                try:
                    from scipy.spatial import cKDTree

                    dist, _ = cKDTree(src).query(verts, k=1, workers=-1)
                    keep = dist <= thr
                except Exception:  # noqa: BLE001
                    pass
                if keep is not None and len(keep) == len(mesh.vertices):
                    trim_removed = int((~keep).sum())
                    if trim_removed:
                        mesh.remove_vertices_by_mask(~keep)
                        # 密度数组与顶点一一对应，按同一掩码裁剪
                        # （长度不一致时就不再参与后面的密度裁剪，避免尺寸错位）
                        dens = dens[keep] if dens.size == len(keep) else np.array([])
        except Exception:  # noqa: BLE001
            trim_removed = 0

    # 5) 裁剪 B：密度分位数（低密度 = 没有观测数据的凭空外推）。
    #    ⚠️ 阈值必须很小 —— 实测这是**最敏感**的参数：
    #       density_quantile=0.03 会把轮廓 IoU 从 0.683 砸到 0.315（砍半），
    #       因为它连同大量**真实表面**一起裁掉了。
    #       所以默认 0.001；确实存在大鼓包时才用 0.005 级别的小值。
    if 0.0 < density_quantile < 1.0 and dens.size and dens.size == len(mesh.vertices):
        thr = float(np.quantile(dens, density_quantile))
        mesh.remove_vertices_by_mask(dens < thr)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    return mesh, {"method": "poisson", "depth": int(depth),
                  "points_in": int(len(pts)), "points_used": int(len(pcd.points)),
                  "trim_sigma": float(trim_distance_sigma),
                  "trim_removed_vertices": int(trim_removed),
                  "density_thresh": (round(float(np.quantile(dens, density_quantile)), 6)
                                     if dens.size else None)}


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


def sanitize_mesh(mesh, log=None):
    """安全清理：去退化面 / 重复三角形 / 重复顶点 / 未引用顶点。

    ## 为什么这一步必要

    泊松重建 + QEM 简化之后，网格里会留下：
      * **零面积三角形** —— 法线算出来是 NaN，光栅化时直接出黑点/闪点；
      * **重复顶点**（同一位置多份）—— 顶点数与文件体积虚高，
        而且"按位置焊接"的工具（Blender、Blender 的 Merge by Distance）会重算拓扑；
      * **重复三角形**（同一组三个顶点出现两次）—— 双面渲染时 z-fighting。

    实测 `img3-final` 的 `model_mesh.ply`：退化面 1 个、重复顶点 0、
    最大长宽比 **577.7**（`img3-crop` 只有 17.4）。

    ## 为什么**不**顺手删细长三角形

    删掉长宽比超阈值的面会**在网格上开一个洞**（周围没有替代面），
    贴图烘焙后表现为一个可见的洞 —— 比留一个细长面更糟。
    所以这里只删"面积≈0"的退化面（它们本来就不贡献面积），
    细长面只统计、报告，不删。

    返回 `(mesh, stats)`，stats 里记清楚删了多少，便于验收核对。
    """
    import numpy as np

    stats = {"degenerate_removed": 0, "zero_area_removed": 0,
             "duplicate_tri_removed": 0,
             "duplicate_vert_removed": 0, "unreferenced_removed": 0,
             "tris_before": int(len(mesh.triangles)),
             "verts_before": int(len(mesh.vertices))}
    if len(mesh.triangles) == 0:
        return mesh, stats

    try:
        before = len(mesh.triangles)
        mesh.remove_degenerate_triangles()
        stats["degenerate_removed"] = before - len(mesh.triangles)
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"去掉退化三角形失败（{type(exc).__name__}: {exc}）")

    # ---- 几何上的零面积面（open3d **不**处理这一种）----
    #
    # ⚠️ 用合成网格（三点共线）测试时才发现：
    # `remove_degenerate_triangles()` 只删**索引重复**的面（如 (a,a,b)），
    # 对"三个不同顶点但几何共线、面积为 0"的面它删 **0** 个。
    # 而零面积面会让面法线变成 NaN ⇒ 光栅化出现黑点/闪点，必须自己按面积清一遍。
    #
    # 阈值取**相对**面积 `med * 1e-12`：足以区分"真零面积"（浮点上通常正好 0）
    # 与"细长面"（约 med*1e-4 量级），从而**不会误删细长面**
    # （删细长面会在网格上开洞，贴图后可见）。
    try:
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        if len(f):
            a0, a1, a2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            area = 0.5 * np.linalg.norm(np.cross(a1 - a0, a2 - a0), axis=1)
            med = float(np.median(area)) if len(area) else 0.0
            eps = max(med * 1e-12, 1e-18)
            bad = area < eps
            n_bad = int(bad.sum())
            if n_bad:
                mesh.remove_triangles_by_mask(bad)
                mesh.remove_unreferenced_vertices()
                stats["zero_area_removed"] = n_bad
                if log:
                    log(f"去掉几何零面积面 {n_bad} 个（open3d 不会删这一类）")
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"零面积面清理失败（{type(exc).__name__}: {exc}）")

    try:
        before = len(mesh.triangles)
        mesh.remove_duplicated_triangles()
        stats["duplicate_tri_removed"] = before - len(mesh.triangles)
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"去掉重复三角形失败（{type(exc).__name__}: {exc}）")

    try:
        before = len(mesh.vertices)
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        stats["duplicate_vert_removed"] = before - len(mesh.vertices)
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"合并重复顶点失败（{type(exc).__name__}: {exc}）")

    try:
        before = len(mesh.vertices)
        mesh.remove_unreferenced_vertices()
        stats["unreferenced_removed"] = before - len(mesh.vertices)
    except Exception:  # noqa: BLE001
        pass

    mesh.compute_vertex_normals()
    stats["tris_after"] = int(len(mesh.triangles))
    stats["verts_after"] = int(len(mesh.vertices))
    if log:
        log("网格清理：退化面 -{}, 重复面 -{}, 顶点 {} → {}".format(
            stats["degenerate_removed"], stats["duplicate_tri_removed"],
            stats["verts_before"], stats["verts_after"]))
    return mesh, stats
