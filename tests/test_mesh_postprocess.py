"""网格后处理链的顺序与属性保留测试。

锁定两件曾经真实出问题、且**只看结果看不出来**的事：

1. **顺序必须是"预简化 -> 平滑 -> 最终 QEM"**。
   旧顺序是"平滑全体 -> QEM"：泊松 depth=10 产出 2771 万面时，
   平滑跑了 61 分钟仍未完成（内存 7.9 GB，本机只有 15.2 GB）。
   顺序一旦被改回去，功能测试仍然会全绿（只是变得极慢），所以必须在这里钉住。

2. **QEM 与平滑必须保留顶点色与法线**。
   否则网格导出会变成灰白、光照失效 —— 这正是用户反馈过的问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sphere_with_attrs(n: int = 40):
    import open3d as o3d

    m = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=n)
    m.compute_vertex_normals()
    v = np.asarray(m.vertices)
    cols = np.clip((v - v.min(0)) / np.maximum(v.max(0) - v.min(0), 1e-9), 0, 1)
    m.vertex_colors = o3d.utility.Vector3dVector(cols)
    return m


def test_postprocess_order_is_presimplify_then_smooth_then_final() -> None:
    """步骤记录必须体现"先预简化、再平滑、最后 QEM"的顺序。"""
    from gsflow.gsdata.mesh_export import postprocess_mesh

    m = _sphere_with_attrs(60)
    target = 200                      # 远小于原始面数，确保三步都会触发
    mesh, steps = postprocess_mesh(
        m, target_triangles=target, smooth_iters=2, presimplify_ratio=10.0)

    kinds = []
    for s in steps:
        if s.startswith("预简化"):
            kinds.append("pre")
        elif s.startswith("平滑"):
            kinds.append("smooth")
        elif s.startswith("QEM 简化"):
            kinds.append("final")
    assert kinds == ["pre", "smooth", "final"], f"后处理顺序不对：{steps}"

    # 预简化的目标应是 target×ratio，平滑必须在"已降规模"之后
    assert len(mesh.triangles) <= target + 2, f"最终面数 {len(mesh.triangles)} 超过目标 {target}"
    pre = [s for s in steps if s.startswith("预简化")][0]
    pre_faces = int(pre.split("->")[1].split("面")[0].strip())
    assert pre_faces <= target * 10, f"预简化后仍有 {pre_faces} 面，没有起到降规模的作用"


def test_postprocess_ratio_one_disables_presimplify() -> None:
    """`presimplify_ratio <= 1` 应关闭预简化（保留旧行为供对照）。"""
    from gsflow.gsdata.mesh_export import postprocess_mesh

    m = _sphere_with_attrs(40)
    _, steps = postprocess_mesh(m, target_triangles=100, smooth_iters=1,
                                presimplify_ratio=1.0)
    assert not any(s.startswith("预简化") for s in steps), steps
    assert any(s.startswith("平滑") for s in steps), steps


def test_qem_and_smoothing_preserve_colors_and_normals() -> None:
    """QEM 与平滑必须**保留顶点色**，且法线必须被**重新单位化**。

    后者是实测坑：open3d 的 `simplify_quadric_decimation` 与 Taubin 平滑都会让
    法线不再是单位长度（实测偏差 0.027），而 GLB 把这份法线直接用于逐像素光照，
    非单位法线会导致明暗失真。`postprocess_mesh` 末尾必须重算。
    """
    from gsflow.gsdata.mesh_export import postprocess_mesh

    m = _sphere_with_attrs(40)
    c0 = np.asarray(m.vertex_colors).mean(0)

    mesh, steps = postprocess_mesh(m, target_triangles=150, smooth_iters=2,
                                   presimplify_ratio=5.0)
    assert mesh.has_vertex_colors(), "QEM/平滑后顶点色丢失"
    assert mesh.has_vertex_normals(), "QEM/平滑后法线丢失"

    c1 = np.asarray(mesh.vertex_colors).mean(0)
    assert np.abs(c0 - c1).max() < 0.05, f"顶点色被显著改变：{c0} -> {c1}"

    n = np.asarray(mesh.vertex_normals)
    dev = np.abs(np.linalg.norm(n, axis=1) - 1.0).max()
    assert dev < 1e-6, f"法线未单位化（最大偏差 {dev:.4f}），导出后光照会失真"
    # 必须显式记录这一步，便于在日志里确认它真的跑了
    assert any("法线" in s for s in steps), f"没有重算法线的步骤记录：{steps}"


@pytest.mark.parametrize("ratio,expect_pre", [(10.0, True), (1.0, False)])
def test_presimplify_avoids_smoothing_huge_meshes(ratio: float, expect_pre: bool) -> None:
    """核心回归：网格远大于 target 时，**平滑之前**规模必须已经被降下来。

    这条直接对应"depth=10 卡 61 分钟"：只要预简化存在，
    平滑面对的顶点数就不会随泊松 depth 失控。
    """
    from gsflow.gsdata.mesh_export import postprocess_mesh

    m = _sphere_with_attrs(70)
    n_src = len(m.triangles)
    if n_src <= 100 * ratio:
        pytest.skip("样例网格不够大，无法检验预简化触发")

    _, steps = postprocess_mesh(m, target_triangles=100, smooth_iters=1,
                                presimplify_ratio=ratio)
    has_pre = any(s.startswith("预简化") for s in steps)
    assert has_pre is expect_pre, steps

    # 平滑那一步记录的规模，必须已经 <= target×ratio
    smooth_step = [s for s in steps if s.startswith("平滑")][0]
    faces_at_smooth = int(smooth_step.split("@")[1].split("面")[0].strip())
    if expect_pre:
        assert faces_at_smooth <= 100 * ratio, (
            f"平滑时仍有 {faces_at_smooth} 面（应已被预简化到 <= {100 * ratio}）")
    else:
        assert faces_at_smooth == n_src


# --------------------------------------------------------------- 2dgs 一致性投票
def test_multiview_consistency_drops_single_view_points() -> None:
    """一致性投票必须删掉"只被 1 个视角看到"的点（伪影），保留多视角支持的点。

    这是 `mesh_method=2dgs` 的核心机制，对应本项目的实测问题：
    单视角点会让泊松生成毛刺/薄片，并使法线在体素下采样后互相抵消（翻转 ~35%）。
    """
    from gsflow.gsdata.mesh_export import filter_by_multiview_consistency

    rng = np.random.default_rng(0)
    # 100 个被 3 个视角共同看到的点（真实表面），挤在一起构成表面
    surface = rng.normal(0, 0.001, size=(100, 3))
    vid_surface = np.tile([0, 1, 2], 34)[:100].astype(np.int64)
    # 50 个只被视角 9 看到、且远离表面的孤立点（伪影）
    ghost = rng.normal(5.0, 0.001, size=(50, 3))
    vid_ghost = np.full(50, 9, dtype=np.int64)

    pts = np.vstack([surface, ghost])
    vid = np.concatenate([vid_surface, vid_ghost])
    nrm = np.tile(np.array([0.0, 1.0, 0.0]), (len(pts), 1))

    out, out_n, _c, st = filter_by_multiview_consistency(
        pts, normals=nrm, view_id=vid, voxel=0.02, min_views=2)

    assert st["points_in"] == 150
    assert st["dropped_few_views"] == 50, st
    assert len(out) == 100, f"应保留 100 个表面点，实际 {len(out)}"
    assert out[:, 1].max() < 1.0, "孤立的单视角点没有被删掉"
    assert out_n is not None and len(out_n) == len(out)

    # min_views=1 等于关闭该过滤
    out1, _, _, st1 = filter_by_multiview_consistency(
        pts, normals=nrm, view_id=vid, voxel=0.02, min_views=1)
    assert len(out1) == 150 and st1["dropped_few_views"] == 0


def test_multiview_consistency_prefers_larger_voxel() -> None:
    """投票体素越大保留越多点 —— 这解释了实测里 1.5x 删掉 52% 点的现象。

    回归依据（img2-opfix，80 视角）：1.5x 保留 48.0%、IoU 0.684；
    3.0x 保留 84.0%、IoU 0.815。所以默认取 3.0。
    """
    from gsflow.gsdata.mesh_export import filter_by_multiview_consistency

    rng = np.random.default_rng(1)
    # 表面点略有噪声：视角不同 -> 采样位置略有偏移
    pts, vid = [], []
    for v in range(4):
        p = rng.normal(0, 0.004, size=(200, 3))
        pts.append(p)
        vid.append(np.full(200, v, dtype=np.int64))
    pts = np.vstack(pts)
    vid = np.concatenate(vid)

    small, _, _, st_s = filter_by_multiview_consistency(
        pts, view_id=vid, voxel=0.004, min_views=2)
    big, _, _, st_b = filter_by_multiview_consistency(
        pts, view_id=vid, voxel=0.02, min_views=2)
    assert len(big) >= len(small), (
        f"体素变大后保留数不应减少：小 {len(small)} vs 大 {len(big)}")
    assert st_b["kept_ratio"] >= st_s["kept_ratio"]


def test_multiview_consistency_keeps_everything_without_voxel() -> None:
    """voxel<=0（未启用）时应原样返回，不应静默删点。"""
    from gsflow.gsdata.mesh_export import filter_by_multiview_consistency

    pts = np.zeros((10, 3))
    out, _, _, st = filter_by_multiview_consistency(pts, voxel=0.0, min_views=2)
    assert len(out) == 10 and st["points_out"] == 10
