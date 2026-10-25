"""网格精修的数据契约测试：**alpha 带通道维 (H,W,1) 时不能崩**。

## 背景（2026-10-25，img5-quality 导出实测）

`render_depths` 返回的 alpha 是 `alphas[0]`，形状 **(H, W, 1)**（gsplat 输出带通道维）。
第一版精修直接把它放进 `_bilinear`，得到 (N,1)，随后

    m = ok & (a_r > min_alpha) & (d_r > 0)

在 (N,) 与 (N,1) 之间**广播成 (N,N)**，再用它索引法线就抛
`IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed`。
外层 try 把它降级成一条 WARN，于是"精修被跳过"而没人注意 —— 与项目里多次出现的
"静默降级"是同一类问题，所以这里用真实形状把它钉住。
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from gsflow.gsdata.mesh_refine import refine_mesh_to_depths

W = H = 48
FX = 60.0


def _camera(dist: float = 3.0):
    c = np.array([0.0, 0.0, dist])
    look = np.array([0.0, 0.0, -1.0])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -look, c
    K = np.array([[FX, 0.0, W / 2.0], [0.0, FX, H / 2.0], [0.0, 0.0, 1.0]])
    return c2w, K


def _scene_with_channel_dim():
    """平面网格 + 6 个视角；深度/alpha **都带通道维**，复现 render_depths 的形状。"""
    plane = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 0.02)
    plane.translate((-0.5, -0.5, -0.01))
    plane.compute_vertex_normals()

    cams, rendered, idx = [], [], []
    for axis in range(3):
        for sign in (1, -1):
            dist = 3.0
            c = np.zeros(3)
            c[axis] = sign * dist
            look = -c / np.linalg.norm(c)
            up0 = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
            right = np.cross(up0, look)
            right /= np.linalg.norm(right)
            up = np.cross(look, right)
            c2w = np.eye(4)
            c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, look, c
            K = np.array([[FX, 0.0, W / 2.0], [0.0, FX, H / 2.0], [0.0, 0.0, 1.0]])

            ys, xs = np.mgrid[0:H, 0:W]
            dirs = np.stack([(xs - W / 2.0) / FX, (ys - H / 2.0) / FX, np.ones_like(xs, float)], -1)
            d_cam = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
            d_world = d_cam @ c2w[:3, :3].T
            # 平面 z=0：t = -c_z / d_z
            t = np.where(d_world[..., 2] < -1e-6, -c[2] / d_world[..., 2], 0.0)
            cos = (d_world * look).sum(-1)
            depth = np.where(t > 0, t * np.clip(cos, 1e-6, 1.0), 0.0).astype(np.float32)
            alpha = (t > 0).astype(np.float32)
            # ⚠️ 关键：加通道维
            depth = depth[..., None]
            alpha = alpha[..., None]
            cams.append((c2w, K, W, H, f"v{axis}{sign}"))
            rendered.append((np.zeros((H, W, 3), np.float32), depth, alpha))
            idx.append(len(cams) - 1)
    return plane, cams, rendered, np.array(idx, dtype=int), np.stack([c[1] for c in cams])


def test_refine_accepts_channel_dim_alpha() -> None:
    plane, cams, rendered, idx, ks = _scene_with_channel_dim()
    out, st = refine_mesh_to_depths(plane, cams, rendered, idx, ks, (W, H), iters=5)
    assert st["stop"] not in ("no_valid_residual", "no_views"), st
    assert st["residual_median"], "带通道维的 alpha 必须能被正常消费"


def test_refine_survives_non_square_depth() -> None:
    """非方形渲染分辨率下 K 缩放必须按 (w,h) 而不是 (h,w)。"""
    plane, cams, rendered, idx, ks = _scene_with_channel_dim()
    out, st = refine_mesh_to_depths(plane, cams, rendered, idx, ks, (W, H), iters=3)
    assert len(out.triangles) == len(plane.triangles)
