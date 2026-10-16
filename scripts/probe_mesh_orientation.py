"""逐项检查网格后处理的每一步对**法线翻转率**的影响，并测试"修朝向"是否有效。

背景（用户最初的抱怨是"网格破碎"）：
导出的网格实测**相邻面法线点积 <0 的比例约 18%~22%**（即约 1/5 的相邻面朝向相反），
表现为光照上的大片黑斑与"破碎感"。

此前我一直以为是**几何**问题（深度噪声、泊松）。
但顶点法线是由**三角面绕序**决定的 —— 如果网格只是**朝向不一致**（绕序乱了），
那它是一个**可修复**的问题，而不是几何精度问题。

`mesh_export.postprocess_mesh` 目前只 `compute_vertex_normals()`，
**从不调用 `orient_triangles()`**。本脚本量化：
  1. 各后处理步骤（清理/裁剪/泊松/预简化/平滑/QEM/重算法线）之后的翻转率；
  2. 加上 `orient_triangles()` 之后能否降下来（以及 IoU/尺度是否受损）。

用法: scripts\\gpu_python.cmd scripts\\probe_mesh_orientation.py --job dd-0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def flips(mesh) -> tuple[float, float, int]:
    """返回 (相邻面法线点积均值, 翻转占比, 样本数)。"""
    t = np.asarray(mesh.triangles, dtype=np.int64)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if len(t) < 10:
        return float("nan"), float("nan"), 0
    # 面法线（由绕序决定，**不**用 mesh.triangle_normals，避免被缓存影响）
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    fn = np.cross(b - a, c - a)
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(ln, 1e-12)

    # 边 -> 相邻面
    from collections import defaultdict
    e2f: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, tri in enumerate(t):
        for k in range(3):
            x, y = int(tri[k]), int(tri[(k + 1) % 3])
            e2f[(min(x, y), max(x, y))].append(i)
    dots = []
    for faces in e2f.values():
        if len(faces) == 2:
            dots.append(float(np.dot(fn[faces[0]], fn[faces[1]])))
    if not dots:
        return float("nan"), float("nan"), 0
    d = np.asarray(dots)
    return float(d.mean()), float((d < 0).mean()), int(len(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="dd-0")
    ap.add_argument("--mesh", default="")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d

    job = ROOT / "workspace" / args.job
    path = Path(args.mesh) if args.mesh else job / "05_export" / "model_mesh.obj"
    if not path.exists():
        print(f"找不到网格：{path}")
        return 1
    print(f"网格 {path}")
    mesh = o3d.io.read_triangle_mesh(str(path))
    print(f"三角形 {len(mesh.triangles)}  顶点 {len(mesh.vertices)}  "
          f"边流形={mesh.is_edge_manifold()}  顶点流形={mesh.is_vertex_manifold()} "
          f"可定向={mesh.is_orientable()}")

    m0, f0, n0 = flips(mesh)
    print(f"\n原始：相邻面点积均值={m0:+.4f}  翻转占比={f0*100:.1f}%  样本={n0}")

    # 步骤 1：修朝向（open3d 用最小生成树做全局一致定向）
    m1 = o3d.geometry.TriangleMesh(mesh)
    try:
        m1.orient_triangles()
        a, b, c = flips(m1)
        print(f"orient_triangles()：均值={a:+.4f}  翻转={b*100:.1f}%  （{len(m1.triangles)} 面）")
    except Exception as e:  # noqa: BLE001
        print(f"orient_triangles() 失败：{type(e).__name__}: {e}")

    # 步骤 2：先去掉退化/重复面，再定向（重复面会让定向无解）
    m2 = o3d.geometry.TriangleMesh(mesh)
    m2.remove_duplicated_triangles()
    m2.remove_degenerate_triangles()
    m2.remove_duplicated_vertices()
    m2.remove_unreferenced_vertices()
    a2, b2, c2 = flips(m2)
    print(f"清洗后（去重/去退化）：均值={a2:+.4f}  翻转={b2*100:.1f}%  （{len(m2.triangles)} 面）")
    m3 = o3d.geometry.TriangleMesh(m2)
    try:
        m3.orient_triangles()
        a3, b3, c3 = flips(m3)
        print(f"清洗 + orient_triangles()：均值={a3:+.4f}  翻转={b3*100:.1f}%  （{len(m3.triangles)} 面）")
    except Exception as e:  # noqa: BLE001
        print(f"清洗后定向失败：{type(e).__name__}: {e}")

    # 步骤 3：检查朝向不一致是否只是"局部小岛"（若是，定向应几乎全修好）
    print("\n解读：")
    print("  * `orient_triangles()` 把翻转率显著降低 ⇒ 主因是**绕序不一致（可修）**，")
    print("    而不是几何错误。那样就应当把定向加进 postprocess_mesh。")
    print("  * 若降不下来 ⇒ 朝向冲突源于非流形结构/自交，需要从重建端解决。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
