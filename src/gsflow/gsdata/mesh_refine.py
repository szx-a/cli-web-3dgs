"""网格精修：把"薄壳/塌陷"的网格按多视角**期望深度（ED）**拉回正确表面。

## 为什么需要（2026-10-25）

实测 IMG5（俯视素材、193/193 注册、掩码内 SSIM 0.9654）导出的 TSDF 网格仍然是"封闭的薄壳"：
屋顶塌成一片、墙是分离的薄板。三条路线（poisson / 2dgs / tsdf）都这样 ⇒ 问题不在提取算法，
而在**提取之后没有任何一步把网格对齐到"高斯渲染出来的表面"**。

高斯渲染的 ED 深度是"这个视角下表面在哪儿"最可靠的观测；网格只是它的近似。
所以精修就是：**让网格的深度去贴合 ED 深度**。

## 算法（每次迭代）

1. 取若干视角，把网格顶点投影到该视角；
2. 双线性采样该视角的 ED 深度；
3. 残差 `r = d_render − d_vertex`（沿相机 z）；只在
   `|r| < tol*z` 且 `alpha > min_alpha` 的顶点上采信（否则是被遮挡/背景）；
4. 沿**顶点法线**位移 `step * r * cos(θ)`，θ 是视线与法线夹角（避免掠射角放大误差）；
5. 每 `smooth_every` 次迭代做一次 Taubin 平滑（保体积，抑制拉出尖刺）；
6. 记录每轮的**中位残差**；连续 `patience` 轮不改善就停（并记录停止原因）。

整个过程纯 numpy + Open3D，不需要 GPU 训练循环，也不需要新依赖。

## 与"端到端可微网格"的区别（如实说明）

SuGaR / 2DGS 那类方法是**训练时就产出网格**（可微光栅化把梯度传回顶点）。
本模块是**训练后**的几何拟合：用的是已收敛的高斯 ED 深度作为目标，因此
  * 优点：不动训练、对所有网格路线都适用、失败也不会影响高斯产物；
  * 限制：只能修正"深度方向"的误差，补不出完全没被观测到的结构（如未拍到的屋顶内侧）。
"""

from __future__ import annotations

import numpy as np


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """在 `img[H,W]` 上按浮点像素坐标做双线性采样（越界返回 0）。"""
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx = xs - x0
    wy = ys - y0
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    v = (img[y0c, x0c] * (1 - wx) * (1 - wy)
         + img[y0c, x1c] * wx * (1 - wy)
         + img[y1c, x0c] * (1 - wx) * wy
         + img[y1c, x1c] * wx * wy)
    return np.where(valid, v, 0.0)


def refine_mesh_to_depths(
    mesh,
    cams,
    rendered,
    view_idx,
    ks: np.ndarray,
    src_wh: tuple[int, int],
    iters: int = 200,
    step: float = 0.6,
    max_step_ratio: float = 0.01,
    tol_ratio: float = 0.15,
    tol_abs_ratio: float = 0.02,
    min_alpha: float = 0.5,
    views_per_iter: int = 6,
    smooth_every: int = 10,
    patience: int = 5,
    log=None,
):
    """把网格顶点沿法线拉向多视角 ED 深度（见模块 docstring 的算法）。

    参数
    ----
    mesh : open3d.geometry.TriangleMesh
    cams : list[(c2w(4,4), K(3,3), w, h, name)] —— 与 `export` 里一致
    rendered : list[(rgb, depth, alpha)] —— `render_depths` 的返回值
    view_idx : np.ndarray —— `render_depths` 返回的"实际使用了哪些视角"
    ks : np.ndarray —— 每个视角的原始内参
    src_wh : (w, h) —— 内参对应的原始图像尺寸（用于缩放 K）

    返回 `(mesh, stats)`；stats 含每轮中位残差、停止原因、被采信顶点比例。
    """
    import open3d as o3d

    from .mesh_export import scale_intrinsics

    stats: dict = {"iters": 0, "residual_median": [], "stop": "", "used_fraction": []}
    if len(mesh.triangles) == 0 or not rendered or not cams:
        stats["stop"] = "empty"
        return mesh, stats

    out = o3d.geometry.TriangleMesh(mesh)
    verts = np.asarray(out.vertices, dtype=np.float64).copy()
    rng = np.random.default_rng(0)

    # 预计算每个视角的 K（缩放到渲染分辨率）
    view_cache = []
    for j, vi in enumerate(np.asarray(view_idx).tolist()):
        if j >= len(rendered):
            break
        _, depth, alpha = rendered[j]
        # ⚠️ `render_depths` 返回的 alpha 形状是 (H,W,1)（gsplat 输出带通道维）。
        # 不 squeeze 的话 `_bilinear` 返回 (N,1)，随后
        # `ok & (a_r > min_alpha) & (d_r > 0)` 会在 (N,) 与 (N,1) 之间广播成 (N,N)，
        # 再拿它索引法线就报 `IndexError: too many indices for array`
        # （实测在 img5-quality 的导出里踩到，整个精修因此被降级跳过）。
        depth = np.asarray(depth, dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        c2w = np.asarray(cams[vi][0], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        K = scale_intrinsics(np.asarray(ks[vi], dtype=np.float64), src_wh,
                             (depth.shape[1], depth.shape[0]))
        view_cache.append((w2c, K, depth, alpha))
    if not view_cache:
        stats["stop"] = "no_views"
        return out, stats

    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    verts_best = verts.copy()
    damp = float(step)
    tol_abs = max(tol_abs_ratio * diag, 1e-9)
    best = float("inf")
    bad_rounds = 0
    for it in range(int(iters)):
        out.compute_vertex_normals()
        normals = np.asarray(out.vertex_normals, dtype=np.float64)

        pick = rng.choice(len(view_cache), size=min(views_per_iter, len(view_cache)), replace=False)
        disp_sum = np.zeros(len(verts))
        disp_cnt = np.zeros(len(verts))
        residuals: list[np.ndarray] = []

        for pi in pick:
            w2c, K, depth, alpha = view_cache[int(pi)]
            p = verts @ w2c[:3, :3].T + w2c[:3, 3]          # 相机系
            z = p[:, 2]
            ok = z > 1e-6
            if not ok.any():
                continue
            u = K[0, 0] * p[:, 0] / np.where(ok, z, 1.0) + K[0, 2]
            v = K[1, 1] * p[:, 1] / np.where(ok, z, 1.0) + K[1, 2]
            d_r = _bilinear(depth, u, v)
            a_r = _bilinear(alpha, u, v)
            m = ok & (a_r > min_alpha) & (d_r > 0)
            if not m.any():
                continue
            r = d_r[m] - z[m]
            # 采信判据：**鲁棒尺度**（ICP 常见做法）+ 相对/绝对容差。
            #
            # 只用"相对容差"会在网格离真值较远时**一个顶点都不采信**（实测：盒面到球面
            # 残差 1.4、容差 1.15 ⇒ 精修 0 轮）。改用残差中位数的 3 倍做尺度，
            # 整体偏厚/偏薄能被逐步拉回，被遮挡的另一层仍会被相对容差挡掉。
            s = float(np.median(np.abs(r))) if r.size else 0.0
            shrink = 1.0 - 0.5 * (it / max(int(iters), 1))
            tol = np.maximum(tol_ratio * np.abs(d_r[m]),
                             max(tol_abs * shrink, 3.0 * s))
            keep = np.abs(r) <= tol
            if not keep.any():
                continue
            idx = np.where(m)[0][keep]
            r = r[keep]
            # 沿法线：把深度残差投影到法线上（掠射角不能放大）
            ray = p[idx] / np.linalg.norm(p[idx], axis=1, keepdims=True).clip(min=1e-12)
            cos = np.abs((normals[idx] * ray).sum(axis=1)).clip(0.05, 1.0)
            disp = np.zeros(len(verts))
            disp[idx] = r * cos
            disp_sum += disp
            disp_cnt += (np.abs(disp) > 0)
            residuals.append(np.abs(r))

        if not residuals:
            stats["stop"] = "no_valid_residual"
            break

        med = float(np.median(np.concatenate(residuals)))
        stats["residual_median"].append(med)
        stats["used_fraction"].append(float((disp_cnt > 0).mean()))
        stats["iters"] = it + 1

        # ---- ICP 式回溯：残差变差就**回退到最好的顶点**并把步长减半 ----
        #
        # 只用"固定步长 + 固定上限"会越过表面来回震荡（实测：盒面到球面误差 0.039 →
        # 0.187，越修越糟）。回溯 + 阻尼减半保证每轮都单调不增，这是 ICP 的标准做法。
        #
        # ⚠️ 回退后**本轮必须跳过位移**（`continue`）：第一版回退完仍继续施加同一个坏位移，
        # 结果残差从 0.0643 涨到 0.0789 —— "回退了"但状态又被打坏，等于没回退。
        if med < best * 0.999:
            best = med
            verts_best = verts.copy()
            bad_rounds = 0
        else:
            bad_rounds += 1
            damp *= 0.5
            verts = verts_best.copy()
            out.vertices = o3d.utility.Vector3dVector(verts)
            if bad_rounds >= patience or damp < 0.05:
                stats["stop"] = "converged"
                break
            continue

        has = disp_cnt > 0
        if has.any():
            delta = np.zeros(len(verts))
            delta[has] = disp_sum[has] / disp_cnt[has]
            # 单步位移上限：防止个别错误观测把顶点甩出去
            scale = max_step_ratio * float(np.linalg.norm(verts.max(0) - verts.min(0)))
            delta = np.clip(delta * damp, -scale, scale)
            verts = verts + normals * delta[:, None]
            out.vertices = o3d.utility.Vector3dVector(verts)

        if smooth_every and (it + 1) % smooth_every == 0:
            try:
                out = out.filter_smooth_taubin(number_of_iterations=1)
                verts = np.asarray(out.vertices, dtype=np.float64).copy()
            except Exception:  # noqa: BLE001
                pass
        if log and (it + 1) % 20 == 0:
            log(f"  网格精修 {it + 1}/{iters}：中位深度残差 {med:.5f}")

    if not stats["stop"]:
        stats["stop"] = "max_iters"
    # 收尾：几何**一定是最好那一版**（回溯时已恢复），但残差列表的最后一个样本
    # 可能是"触发回退的那次坏测量"。所以单独记 best，报告用 best，避免看起来"越修越差"。
    stats["residual_best"] = float(best) if best != float("inf") else None
    if verts_best is not None:
        out.vertices = o3d.utility.Vector3dVector(verts_best)
    out.compute_vertex_normals()
    if log and stats["residual_median"]:
        log(f"网格精修结束：{stats['iters']} 轮，中位深度残差 "
            f"{stats['residual_median'][0]:.5f} → {best:.5f}，停止原因={stats['stop']}")
    return out, stats
