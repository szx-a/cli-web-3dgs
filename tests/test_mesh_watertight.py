"""孔洞填充（水密化）的回归测试。

## 判据说明（2026-10-25 实测）

Open3D 的 `is_watertight()` **同时要求**：无边界边、边/顶点流形、**且无自交**。
泊松/TSDF 重建的网格几乎必然自交，所以它在本项目里几乎永远是 False ——
直接拿它当"封闭"判据会得出"永远不封闭"的错误结论。

本项目改用**封闭判据**（`fill_holes` 的 `stats["closed_after"]`）：
    无边界边 && 无非流形边 && 顶点流形
自交单独报告（消除它需要重网格化，属于另一项工作）。
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from gsflow.gsdata.mesh_export import fill_holes


def _sphere_with_hole(n_remove: int = 1) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=12)
    t = np.asarray(s.triangles)
    m = o3d.geometry.TriangleMesh()
    m.vertices = s.vertices
    m.triangles = o3d.utility.Vector3iVector(np.delete(t, list(range(5, 5 + n_remove)), axis=0))
    return m


def test_fill_holes_closes_a_single_triangle_hole() -> None:
    m = _sphere_with_hole(1)
    assert not m.is_watertight()
    out, st = fill_holes(m)
    assert st["boundary_edges_before"] == 3
    assert st["boundary_edges_after"] == 0
    assert st["closed_after"] is True, st
    assert st["filled_loops"] == 1
    assert st["added_triangles"] == 3


def test_fill_holes_handles_multiple_holes() -> None:
    m = _sphere_with_hole(3)
    out, st = fill_holes(m)
    assert st["boundary_edges_after"] == 0, st
    assert st["closed_after"] is True, st
    assert st["filled_loops"] >= 1


def test_fill_holes_skips_oversized_loops() -> None:
    """巨环多半是"模型边界"而不是孔洞：不该强行糊一大片假面。"""
    m = _sphere_with_hole(1)
    out, st = fill_holes(m, max_loop_edges=1)   # 3 条边的环 > 1 ⇒ 跳过
    assert st["filled_loops"] == 0
    assert st["skipped_loops"] >= 1
    assert st["boundary_edges_after"] == st["boundary_edges_before"]


def test_no_hole_mesh_is_left_alone() -> None:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=8)
    out, st = fill_holes(s)
    assert st["boundary_edges_before"] == 0
    assert st["filled_loops"] == 0
    assert st["added_triangles"] == 0
    assert st["closed_after"] is True
