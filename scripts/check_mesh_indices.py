"""检查网格/GLB 的索引有效性，定位"索引越界"的来源。

用法：env\\venv\\Scripts\\python.exe scripts\\check_mesh_indices.py --mesh <file.obj>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    args = ap.parse_args()

    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(Path(args.mesh)))
    t = np.asarray(m.triangles)
    v = np.asarray(m.vertices)
    print(f"文件: {args.mesh}")
    print(f"顶点 {len(v)}  三角形 {len(t)}")
    print(f"索引 dtype={t.dtype}  min={t.min()}  max={t.max()}")
    bad = int((t >= len(v)).sum())
    print(f"越界索引数: {bad}  ({100.0*bad/max(t.size,1):.4f}%)")
    if bad:
        rows = np.unique(np.nonzero(t >= len(v))[0])[:5]
        print("含越界索引的三角形（前 5 行）:")
        for r in rows:
            print("   ", t[r].tolist())
    print("前 3 个三角形:", t[:3].tolist())
    if m.has_triangle_uvs():
        uv = np.asarray(m.triangle_uvs)
        print(f"triangle_uvs: {uv.shape}  (期望 {(len(t)*3, 2)})  "
              f"min={uv.min():.3f} max={uv.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
