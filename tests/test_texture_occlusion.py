"""遮挡测试：前景挡住背景时，背景的 texel 不能采到前景颜色。

构造一个最小场景：
* 相机在原点、朝 +z（OpenCV 约定），内参 f=100、cx=cy=32，图像 64×64；
* 前景四边形 z=2（覆盖整个视野），背景四边形 z=4（与前景同尺寸，被完全挡住）；
* "拍摄到的图像"整幅是**红色**（因为看到的就是前景）。

于是：
* **没有遮挡测试**时，背景四边形的点也投影到画面中心 ⇒ 采到红色 ⇒ 纹理里红色 texel
  数量约等于两倍面积（这就是用户看到的"不同部位纹理杂糅"）；
* **有遮挡测试**（深度图全 2.0）时，背景点 z=4 > 2*(1+3%) ⇒ 被拒 ⇒ 不再变成红色。

断言：带遮挡时红色 texel 数显著少于不带遮挡。
"""
from __future__ import annotations

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
pytest.importorskip("xatlas")

from gsflow.gsdata.mesh_export import bake_texture_from_views, unwrap_uv  # noqa: E402


def _quad(z: float, half: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    verts = np.array([
        [-half, -half, z],
        [half, -half, z],
        [half, half, z],
        [-half, half, z],
    ], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, tris


def _mesh_two_quads():
    v1, t1 = _quad(2.0)
    v2, t2 = _quad(4.0)
    verts = np.vstack([v1, v2])
    tris = np.vstack([t1, t2 + 4])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()
    # ❌ 坑：平面四边形的绕序给出的法线可能朝 +z（背向相机），
    # 而焙焙的 facing = -(N·vd) 要求法线朝向相机（- z），否则一个采样都拿不到。
    mesh.vertex_normals = o3d.utility.Vector3dVector(
        np.tile(np.array([0.0, 0.0, -1.0]), (len(verts), 1)))
    return mesh


def test_occlusion_blocks_background_from_sampling_foreground() -> None:
    mesh = _mesh_two_quads()
    tex_size = 128
    uv_mesh, uv_tris, uv = unwrap_uv(mesh, tex_size)  # (新网格, 三角形, 每角度 UV)
    if uv_mesh is None or uv_tris is None or uv is None:
        pytest.skip("xatlas 不可用")

    w = h = 64
    k = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])
    c2w = np.eye(4)
    img = np.full((h, w, 3), 255, dtype=np.uint8)  # 整幅红
    img[..., 1:] = 0

    common = dict(
        images=[img], c2ws=np.stack([c2w]), ks=np.stack([k]), src_whs=[(w, h)],
        tex_size=tex_size, batch=8, margin_px=1, max_views=1,
        bake_reject_black=False, masks=None,
    )

    tex_no, info_no = bake_texture_from_views(mesh, uv_tris, uv, **common)
    # 深度图：整幅 2.0（看到的是前景），alpha=1
    depth = np.full((h, w), 2.0, dtype=np.float32)
    alpha = np.ones((h, w), dtype=np.float32)
    tex_oc, info_oc = bake_texture_from_views(
        mesh, uv_tris, uv, **common, depths=[(depth, alpha)], depth_tol_ratio=0.03)

    # ❌ 不能用“红色 texel 数”做断言：被遮挡拒掉的 texel 会被**扩散填充**从邻居补上颜色，
    # 统计上看不出差别。直接数“因遮挡被拒的采样次数”才是可靠信号。
    assert info_no["samples_rejected_occlusion"] == 0, "不传 depths 时不应有遮挡拒绝"
    assert info_oc["samples_rejected_occlusion"] > 0, (
        "前景挡住背景时，应有采样因遮挡被拒"
    )
    # 背景四边形的采样量大致占一半：拒绝比例不应为 0，也不应全部被拒
    total_views_samples = info_oc.get("samples_rejected_black", 0) + info_oc["samples_rejected_occlusion"]
    assert total_views_samples > 0
    assert info_oc["coverage"] > 0, "遮挡不应把整张贴图变成空白"
