"""验证 `to_largest_component` / `clean_mesh` 是否会打乱面绕序（法线朝向）。

## 背景

导出链末尾我加了 `_orient_normals_outward(mesh)`（实测能把朝外比例从 85.1% 提到 100%），
但**导出到磁盘的 model_mesh.ply 仍只有 85.4%**。说明 `_finish_mesh` 里还有一步改了绕序。

`_finish_mesh` 开头会做：
  * `mesh_largest_only=True` ⇒ `to_largest_component(mesh)`，内部调用
    `remove_triangles_by_mask()` + `remove_unreferenced_vertices()` + `compute_vertex_normals()`
  * 否则 `clean_mesh(mesh)`

本脚本在**同一份网格**上逐步测朝外比例，定位是哪一步把它打乱的。

用法: scripts\\gpu_python.cmd scripts/diag_orientation_pipeline.py --job img3-crop
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def outward(mesh) -> float:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    t = np.asarray(mesh.triangles, dtype=np.int64)
    if len(t) == 0:
        return 0.0
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    c = v.mean(0)
    fc = (p0 + p1 + p2) / 3.0 - c
    fc /= np.maximum(np.linalg.norm(fc, axis=1, keepdims=True), 1e-12)
    return float((np.einsum("ij,ij->i", fn, fc) > 0).mean())


def orient(mesh):
    import open3d as o3d

    v = np.asarray(mesh.vertices, dtype=np.float64)
    t = np.asarray(mesh.triangles, dtype=np.int64).copy()
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    c = v.mean(0)
    fc = (p0 + p1 + p2) / 3.0 - c
    fc /= np.maximum(np.linalg.norm(fc, axis=1, keepdims=True), 1e-12)
    flip = np.einsum("ij,ij->i", fn, fc) < 0
    t[flip] = t[flip][:, [0, 2, 1]]
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(v)
    out.triangles = o3d.utility.Vector3iVector(t)
    if mesh.has_vertex_colors():
        out.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
    out.compute_vertex_normals()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d

    from gsflow.gsdata.mesh_export import clean_mesh, to_largest_component

    job = ROOT / "workspace" / args.job
    src = job / "05_export" / "model_mesh.ply"
    m0 = o3d.io.read_triangle_mesh(str(src))
    print(f"读取 {src.name}: {len(m0.vertices)} 顶点 / {len(m0.triangles)} 面")
    print(f"  ① 读取后朝外比例 = {outward(m0)*100:.1f}%")

    m1 = orient(o3d.geometry.TriangleMesh(m0))
    print(f"  ② 定向后朝外比例 = {outward(m1)*100:.1f}%")

    m2, rm = to_largest_component(o3d.geometry.TriangleMesh(m1))
    print(f"  ③ to_largest_component 后 = {outward(m2)*100:.1f}%（删 {rm} 面）")

    m3, _ = clean_mesh(o3d.geometry.TriangleMesh(m1))
    print(f"  ④ clean_mesh 后         = {outward(m3)*100:.1f}%")

    # 写盘再读回
    tmp = job / "05_export" / "_orient_probe.ply"
    o3d.io.write_triangle_mesh(str(tmp), m1, write_vertex_colors=True)
    m4 = o3d.io.read_triangle_mesh(str(tmp))
    print(f"  ⑤ 写盘再读回后          = {outward(m4)*100:.1f}%（说明 IO 是否改绕序）")
    tmp.unlink(missing_ok=True)

    print("\n判读：哪一步把 100% 降下来，就要在**那一步之后**再做定向。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
