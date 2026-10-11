"""在真实网格上测 xatlas 的分阶段耗时：分 chart vs 打包。

已知事实：
  * 合成球体 9,940 面 -> `parametrize` 0.61s
  * 真实网格 79,808 面 -> `parametrize` **813.92s**（差 3 个数量级）
  * `bruteForce` / `max_iterations` / `resolution` 在球体上**毫无影响**
=> 瓶颈不是面数，也不是那几个打包开关，而是**这团网格产生出的 chart 结构**。

xatlas 的 generate 分两步：先 `compute_charts`（按法线/曲率切分），再 `pack_charts`。
本脚本用 `Atlas` API 把两者分开计时，并打印 chart 数 —— 这是定位的关键数字。

用法：env\\venv\\Scripts\\python.exe scripts\\diag_xatlas_charts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_sphere(n_lat: int, n_lon: int):
    verts = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            verts.append([np.sin(theta) * np.cos(phi), np.cos(theta),
                          np.sin(theta) * np.sin(phi)])
    verts = np.asarray(verts, dtype=np.float32)
    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            tris.append([a, c, b])
            tris.append([b, c, d])
    return verts, np.asarray(tris, dtype=np.uint32)


def measure(label: str, verts: np.ndarray, tris: np.ndarray, **opts) -> float:
    import xatlas

    co = xatlas.ChartOptions()
    for k, v in opts.get("chart", {}).items():
        setattr(co, k, v)
    po = xatlas.PackOptions()
    po.resolution = opts.get("res", 1024)
    for k, v in opts.get("pack", {}).items():
        setattr(po, k, v)
    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, tris)
    t0 = time.time()
    atlas.generate(chart_options=co, pack_options=po)
    dt = time.time() - t0
    n_chart = atlas.chart_count           # 属性，不是方法
    info = atlas.get_mesh(0)
    nv = len(info[0]) if info and info[0] is not None else -1
    print(f"{label:38s} {dt:8.2f}s  charts={n_chart:6d}  "
          f"新顶点 {nv:7d}  ({nv/max(len(verts),1):.2f}x)")
    return dt


def main() -> int:
    import open3d as o3d

    # 合成对照
    sv, st = make_sphere(71, 70)
    print("--- 合成球体（对照）---")
    measure("球体 9940 面", sv, st)

    # 真实网格
    ply = ROOT / "workspace" / "cmp-resetfix" / "05_export" / "model_mesh.ply"
    print(f"\n--- 真实网格 {ply.name} ---")
    m = o3d.io.read_triangle_mesh(str(ply))
    rv = np.asarray(m.vertices, dtype=np.float64).astype(np.float32)
    rt = np.asarray(m.triangles, dtype=np.uint32)
    print(f"面 {len(rt)}  顶点 {len(rv)}")
    measure("真实网格（默认选项，可能很慢）", rv, rt)

    print("\n--- 真实网格：限制 chart 大小 ---")
    # max_chart_area 用 texels 表示；调小可强制切成更多但更简单的 chart（未必更快）
    for area in (0, 256, 1024):
        try:
            measure(f"max_chart_area={area}", rv, rt, chart={"max_chart_area": area})
        except Exception as exc:  # noqa: BLE001
            print(f"max_chart_area={area} 失败：{type(exc).__name__}: {exc}")

    print("\n--- 真实网格：调法线相关权重（影响切分出的 chart 数）---")
    for ndw in (0.1, 1.0, 4.0):
        try:
            measure(f"normal_deviation_weight={ndw}", rv, rt,
                    chart={"normal_deviation_weight": ndw})
        except Exception as exc:  # noqa: BLE001
            print(f"ndw={ndw} 失败：{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
