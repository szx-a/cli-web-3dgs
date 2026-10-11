"""确认 QEM 简化（open3d `simplify_quadric_decimation`）**保留顶点色与法线**。

为什么必须验证：我在导出流程里把 QEM 提前到了平滑之前（预简化），
如果简化会丢掉顶点色/法线，预览会变成灰白、光照也会失效 —— 这正是用户反馈过的问题。
官方文档没有明确承诺保留属性，所以这里用实测钉住。

用法：env\\venv\\Scripts\\python.exe scripts\\check_qem_preserves_attrs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import open3d as o3d

    from gsflow.gsdata.mesh_export import simplify_mesh, smooth_mesh

    # 造一个有颜色、有法线的球
    src = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=60)
    src.compute_vertex_normals()
    v = np.asarray(src.vertices)
    cols = np.clip((v - v.min(0)) / np.maximum(v.max(0) - v.min(0), 1e-9), 0, 1)
    src.vertex_colors = o3d.utility.Vector3dVector(cols)

    print(f"原始：{len(src.triangles)} 面 / {len(src.vertices)} 顶点 "
          f"有颜色={src.has_vertex_colors()} 有法线={src.has_vertex_normals()}")

    # 1) QEM 简化
    s = simplify_mesh(src, target_triangles=len(src.triangles) // 10)
    print(f"QEM 后：{len(s.triangles)} 面 / {len(s.vertices)} 顶点 "
          f"有颜色={s.has_vertex_colors()} 有法线={s.has_vertex_normals()}")
    if s.has_vertex_colors():
        c0 = np.asarray(src.vertex_colors).mean(0)
        c1 = np.asarray(s.vertex_colors).mean(0)
        print(f"  颜色均值 原 {c0.round(3)} -> 后 {c1.round(3)}  "
              f"差异 {np.abs(c0 - c1).max():.4f}")
    if s.has_vertex_normals():
        n = np.asarray(s.vertex_normals)
        unit = np.abs(np.linalg.norm(n, axis=1) - 1.0).max()
        print(f"  法线是否已单位化（最大偏差 {unit:.2e}）")

    # 2) 平滑后再看（顺序：预简化 -> 平滑 -> 最终 QEM）
    sm = smooth_mesh(s, method="taubin", iterations=5)
    print(f"平滑后：{len(sm.triangles)} 面 有颜色={sm.has_vertex_colors()} "
          f"有法线={sm.has_vertex_normals()}")
    f = simplify_mesh(sm, target_triangles=len(sm.triangles) // 2)
    print(f"最终 QEM：{len(f.triangles)} 面 有颜色={f.has_vertex_colors()} "
          f"有法线={f.has_vertex_normals()}")
    if f.has_vertex_colors():
        c2 = np.asarray(f.vertex_colors).mean(0)
        c0 = np.asarray(src.vertex_colors).mean(0)
        print(f"  端到端颜色均值 {c0.round(3)} -> {c2.round(3)}  "
              f"差异 {np.abs(c0 - c2).max():.4f}")

    ok = (s.has_vertex_colors() and s.has_vertex_normals()
          and sm.has_vertex_colors() and f.has_vertex_colors() and f.has_vertex_normals())
    print("\n结论:", "QEM/平滑全程保留颜色与法线" if ok
          else "**有属性丢失** —— 预简化顺序不可行，需要改成先保存属性再回填")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
