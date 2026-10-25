"""网格精修的回归测试：**残差必须真的下降**。

用合成场景：真实表面是单位球，但输入网格是一个"偏大/形状不对"的盒子（外接立方体）。
精修若正确，顶点应被拉向球面 ⇒ 中位深度残差下降、且顶点与球面的距离变小。

这个测试的动机来自本项目反复出现的模式：**看起来接上了正则，其实没有生效**
（法线正则写成恒等式、`lpips_every=16` 让梯度打架、补洞被 `remove_non_manifold_edges` 删光）。
所以这里必须断言"指标真的变好"，而不是"函数跑完没报错"。
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from gsflow.gsdata.mesh_refine import refine_mesh_to_depths

W = H = 64
FX = 70.0


def _view(axis: int, sign: int):
    """构造 **z 朝前**（grasp/COLMAP 约定）的相机，并渲染单位球的 ED 深度。

    ⚠️ 第一版写成 OpenGL 约定（+z 朝后），而 `refine_mesh_to_depths` 假定 `z>0` 在相机前方
    ⇒ 所有顶点都被判为"在相机后面"，残差为空、精修 0 轮。相机约定必须与消费方一致。
    """
    dist = 3.0
    c = np.zeros(3)
    c[axis] = sign * dist
    look = -c / np.linalg.norm(c)                 # 从相机指向原点
    up = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
    right = np.cross(up, look)
    right /= np.linalg.norm(right)
    up2 = np.cross(look, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up2, look, c
    K = np.array([[FX, 0.0, W / 2.0], [0.0, FX, H / 2.0], [0.0, 0.0, 1.0]])

    ys, xs = np.mgrid[0:H, 0:W]
    dirs = np.stack([(xs - W / 2.0) / FX, (ys - H / 2.0) / FX, np.ones_like(xs, float)], -1)
    d_cam = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    d_world = d_cam @ c2w[:3, :3].T               # 单位方向，z 朝前
    b = (d_world * c).sum(-1)
    disc = b * b - (c @ c - 1.0)
    t = np.where(disc > 0, -b - np.sqrt(np.clip(disc, 0.0, None)), 0.0)
    cos = (d_world * look).sum(-1)                # 射线与相机 z 轴的夹角余弦
    # 精修比较的是**相机 z 坐标**，所以深度必须是 t*cos，而不是射线长度 t
    depth = np.where(disc > 0, t * np.clip(cos, 1e-6, 1.0), 0.0).astype(np.float32)
    alpha = (disc > 0).astype(np.float32)
    return (c2w, K, W, H, f"v{axis}{sign}"), (np.zeros((H, W, 3), np.float32), depth, alpha)


def _scene():
    box = o3d.geometry.TriangleMesh.create_box(1.2, 1.2, 1.2)
    box.translate((-0.6, -0.6, -0.6))
    box.compute_vertex_normals()
    cams, rendered, idx = [], [], []
    for axis in range(3):
        for sign in (1, -1):
            cam, r = _view(axis, sign)
            cams.append(cam)
            rendered.append(r)
            idx.append(len(cams) - 1)
    ks = np.stack([c[1] for c in cams])
    return box, cams, rendered, np.array(idx, dtype=int), ks


def test_refinement_reduces_depth_residual() -> None:
    box, cams, rendered, idx, ks = _scene()
    out, st = refine_mesh_to_depths(
        box, cams, rendered, idx, ks, (W, H), iters=40, views_per_iter=6, smooth_every=0)
    assert st["stop"] not in ("no_valid_residual", "no_views"), st
    assert st["residual_median"], "没有采信到任何观测"
    first, last = st["residual_median"][0], st["residual_median"][-1]
    assert last < first * 0.95, f"中位深度残差没有下降：{first:.4f} -> {last:.4f}"
    assert st["used_fraction"][-1] > 0.5, f"被采信的顶点比例过低：{st['used_fraction'][-1]:.2f}"


def test_refinement_moves_vertices_towards_the_true_surface() -> None:
    box, cams, rendered, idx, ks = _scene()
    v0 = np.asarray(box.vertices)
    out, _ = refine_mesh_to_depths(
        box, cams, rendered, idx, ks, (W, H), iters=40, views_per_iter=6, smooth_every=0)
    v1 = np.asarray(out.vertices)
    # 盒子顶点到单位球面 |1 - |v|| 的平均距离应下降
    e0 = float(np.abs(np.linalg.norm(v0, axis=1) - 1.0).mean())
    e1 = float(np.abs(np.linalg.norm(v1, axis=1) - 1.0).mean())
    assert e1 < e0, f"顶点没有靠近真实表面：{e0:.4f} -> {e1:.4f}"


def test_refinement_keeps_topology() -> None:
    box, cams, rendered, idx, ks = _scene()
    n0 = len(box.triangles)
    out, _ = refine_mesh_to_depths(box, cams, rendered, idx, ks, (W, H), iters=10)
    assert len(out.triangles) == n0, "精修不应改变拓扑（只移动顶点）"
