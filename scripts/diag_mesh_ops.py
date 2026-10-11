"""分项计时：读网格 / QEM 简化 / xatlas 展开，各自到底多慢。

背景：`diag_xatlas_scale.py` 用子进程扫描时**所有面数都超时**（连 500 面都 >90s），
唯一说得通的解释是"超时发生在 UV 展开之前"—— 也就是 **QEM 简化或读取本身**就慢。
而这两个操作在导出流程里都在用，所以必须查清（不只影响贴图，也影响网格质量）。

用法：
  env\\venv\\Scripts\\python.exe scripts\\diag_mesh_ops.py
  env\\venv\\Scripts\\python.exe scripts\\diag_mesh_ops.py --ply <path> --faces 80000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def mark(label: str, t0: float) -> float:
    now = time.time()
    print(f"  {label:<34} {now - t0:8.2f}s", flush=True)
    return now


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=str(ROOT / "workspace" / "img2-01" / "05_export" / "model_mesh.ply"))
    ap.add_argument("--faces", type=int, default=80000)
    args = ap.parse_args()

    import open3d as o3d

    t = time.time()
    m = o3d.io.read_triangle_mesh(args.ply)
    n0 = len(m.triangles)
    t = mark(f"读取 PLY（{n0} 面）", t)

    t = time.time()
    m2 = m.simplify_quadric_decimation(min(args.faces, n0))
    t = mark(f"QEM 简化 -> {len(m2.triangles)} 面", t)

    try:
        import xatlas

        v = np.asarray(m2.vertices, dtype=np.float32)
        tri = np.asarray(m2.triangles, dtype=np.uint32)
        t = time.time()
        vm, nt, uv = xatlas.parametrize(v, tri)
        t = mark(f"xatlas 展开（{len(tri)} 面）", t)
    except Exception as exc:  # noqa: BLE001
        print(f"  xatlas 失败: {type(exc).__name__}: {exc}")

    print("\n判读：哪一项是大头，就优化哪一项。")
    print("若 QEM 很慢，说明网格后处理本身是瓶颈（不只影响贴图）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
