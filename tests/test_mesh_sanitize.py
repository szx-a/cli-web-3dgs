"""回归测试：网格清理的安全性，以及"零面积面 vs 细长面"的正确区分。

## 背景（一次真实的误报）

我给 `img3-final` 加网格体检时，把"面积小于中位数 0.1% 的面"叫"退化三角形"，
于是报告里出现"退化面 1 个"。但在加清理步骤时发现：
`open3d.remove_degenerate_triangles()` 删了 **0** 个面。

原因是我混淆了两个不同的概念：

  * **零面积面（degenerate）**：面积**严格为 0**，法线是 NaN ⇒ 必须清理；
  * **细长面（sliver）**：面积小但**非零**，长宽比可以到几百 ⇒ 能正常渲染，
    **删掉会在网格上开洞**（贴图后可见），所以只能报告、不能删。

这个测试把这条区分固定下来，免得以后又把两者混为一谈。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

o3d = pytest.importorskip("open3d")


def _mesh(verts, tris):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    return m


def _unit_cube():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                  [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                  [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], dtype=np.int32)
    return v, f


def _areas(m):
    v = np.asarray(m.vertices)
    f = np.asarray(m.triangles)
    if len(f) == 0:
        return np.array([])
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def test_zero_area_face_removed() -> None:
    """真·零面积面必须被清理掉（法线 NaN 的根源）。"""
    from gsflow.gsdata.mesh_export import sanitize_mesh

    v, f = _unit_cube()
    # 加一个三点共线（面积=0）的退化面
    v = np.vstack([v, [[2.0, 0.0, 0.0]]])
    f = np.vstack([f, [[0, 1, 8]]])
    m = _mesh(v, f)
    assert (np.abs(_areas(m)) < 1e-15).any(), "构造的退化面不存在，测试无效"

    m2, stats = sanitize_mesh(m)
    areas = _areas(m2)
    assert (areas > 0).all(), f"仍有零面积面：{areas.min()}"
    # open3d 的 remove_degenerate_triangles() 只管**索引重复**，几何共线的面
    # 由我们自己的按面积清理负责，所以计数要看 zero_area_removed。
    assert stats["degenerate_removed"] + stats["zero_area_removed"] >= 1, stats


def test_duplicate_triangle_removed() -> None:
    """重复三角形（同三点出现两次）会被删掉（否则双面渲染 z-fighting）。"""
    from gsflow.gsdata.mesh_export import sanitize_mesh

    v, f = _unit_cube()
    f2 = np.vstack([f, f[:1]])          # 第一面重复一次
    m = _mesh(v, f2)
    m2, stats = sanitize_mesh(m)
    assert len(m2.triangles) == len(f), \
        f"重复面未删除：{len(m2.triangles)} vs {len(f)}"
    assert stats["duplicate_tri_removed"] == 1


def test_sliver_is_preserved() -> None:
    """细长面（面积小但非零）**不能**被删 —— 删掉会开洞。"""
    from gsflow.gsdata.mesh_export import sanitize_mesh

    base_v, base_f = _unit_cube()
    # 细长面：底边 1.0，高 1e-4 ⇒ 面积 5e-5（相对立方体面 0.5 极小但非零）
    extra = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.5, 1e-4, -1.0]])
    v = np.vstack([base_v, extra])
    n = len(base_v)
    f = np.vstack([base_f, [[n, n + 1, n + 2]]])
    m = _mesh(v, f)
    areas = _areas(m)
    assert (areas > 0).all(), "该细长面被构造为零面积了，测试无效"
    assert areas.min() < 1e-4, "细长面不够细，测试无效"

    m2, _stats = sanitize_mesh(m)
    assert len(m2.triangles) == len(f), \
        f"细长面被误删（会开洞）：{len(m2.triangles)} vs {len(f)}"
    assert _areas(m2).min() > 0


def test_sanitize_preserves_geometry() -> None:
    """清理不得改变表面积/包围盒 —— 这是"安全"的定义。"""
    from gsflow.gsdata.mesh_export import sanitize_mesh

    v, f = _unit_cube()
    m = _mesh(v, f)
    a0 = float(_areas(m).sum())
    bb0 = np.asarray(m.vertices).max(0) - np.asarray(m.vertices).min(0)
    m2, _ = sanitize_mesh(m)
    a1 = float(_areas(m2).sum())
    bb1 = np.asarray(m2.vertices).max(0) - np.asarray(m2.vertices).min(0)
    assert abs(a1 - a0) < 1e-9, f"表面积变了：{a0} → {a1}"
    assert np.allclose(bb0, bb1), "包围盒变了"


def test_mesh_quality_metric_distinguishes_sliver_from_zero() -> None:
    """`mesh_quality.mesh_metrics` 必须把两者分开报，而不是合成一个"退化"数。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import mesh_quality as MQ

    # 立方体 + 一个零面积面 + 一个细长面
    v, f = _unit_cube()
    v = np.vstack([v, [[2.0, 0.0, 0.0]], [[0.0, 0.0, -1.0]],
                   [[1.0, 0.0, -1.0]], [[0.5, 1e-4, -1.0]]])
    n = 8
    f = np.vstack([f, [[0, 1, n]], [[n + 1, n + 2, n + 3]]])
    m = _mesh(v, f)
    m.compute_triangle_normals()
    r = MQ.mesh_metrics(m)
    assert r["zero_area"] >= 1, f"零面积面没统计到：{r}"
    assert r["sliver"] >= 1, f"细长面没统计到：{r}"
    # 旧字段必须仍然存在（下游/文档引用过 degenerate）
    assert r["degenerate"] == r["zero_area"]
