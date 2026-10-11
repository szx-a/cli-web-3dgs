"""在**合成球体**上对比 xatlas 各配置的 UV 展开耗时（可控、快速）。

为什么要合成网格：用真实网格做对比时，"QEM 简化到 5000 面"本身就是重活
（40 万面 -> 5 千面要十几分钟），把测量对象和测量成本搅在一起。
合成球体的面数/拓扑完全可控，能干净地回答"哪个配置快"。

背景（真实网格实测）：`xatlas.parametrize` 在 79,808 面上要 **813.92 秒**，
是纹理烘焙的真正瓶颈（而"像素 × 视角"的投影只要 ~2 秒）。

用法：env\\venv\\Scripts\\python.exe scripts\\diag_xatlas_options.py --faces 10000
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def make_sphere(n_lat: int, n_lon: int):
    """生成球体网格（经纬度参数化），返回 (verts, tris)。"""
    verts = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            verts.append([np.sin(theta) * np.cos(phi),
                          np.cos(theta),
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


def tri_area(uv: np.ndarray, tris: np.ndarray) -> float:
    a, b, c = uv[tris][:, 0], uv[tris][:, 1], uv[tris][:, 2]
    return float((np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                         - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) * 0.5).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", type=int, default=10000)
    args = ap.parse_args()

    import xatlas

    # 由目标面数反推经纬度分段：面数 ≈ 2 * n_lat * n_lon
    n_lon = max(8, int(np.sqrt(args.faces / 2)))
    n_lat = max(4, args.faces // (2 * n_lon))
    verts, tris = make_sphere(n_lat, n_lon)
    print(f"合成球体：面 {len(tris)}  顶点 {len(verts)}  "
          f"(lat={n_lat}, lon={n_lon})")

    print("\n=== xatlas 能力 ===")
    print("ChartOptions:", [a for a in dir(xatlas.ChartOptions()) if not a.startswith("_")])
    print("PackOptions :", [a for a in dir(xatlas.PackOptions()) if not a.startswith("_")])

    t = time.time()
    vm, nt, uv = xatlas.parametrize(verts, tris)
    uv = np.asarray(uv, dtype=np.float64)
    print(f"\nA) parametrize（默认）            {time.time()-t:7.2f}s  "
          f"新顶点 {len(vm)}  UV面积 {tri_area(uv, np.asarray(nt)):.0f}")

    def run_atlas(label, chart_kw=None, pack_kw=None, res=1024, verts_=None, tris_=None):
        co = xatlas.ChartOptions()
        for k, v in (chart_kw or {}).items():
            setattr(co, k, v)
        po = xatlas.PackOptions()
        po.resolution = res
        for k, v in (pack_kw or {}).items():
            setattr(po, k, v)
        atlas = xatlas.Atlas()
        atlas.add_mesh(verts_ if verts_ is not None else verts,
                       tris_ if tris_ is not None else tris)
        t0 = time.time()
        atlas.generate(chart_options=co, pack_options=po)
        d = time.time() - t0
        # 注意：chart_count 是**属性**不是方法（踩过 TypeError）
        n_chart = atlas.chart_count
        info = atlas.get_mesh(0)
        nv = len(info[0]) if info and info[0] is not None else -1
        print(f"{label:34s} {d:7.2f}s  charts={n_chart:6d}  新顶点 {nv:7d}")
        return d

    print()
    t_default = run_atlas("B) Atlas（默认选项）")
    t_nobrute = run_atlas("C) + bruteForce=False", pack_kw={"bruteForce": False})
    t_fewiter = run_atlas("D) + max_iterations=1", chart_kw={"max_iterations": 1})
    t_both = run_atlas("E) + bruteForce=False + max_iter=1",
                       chart_kw={"max_iterations": 1}, pack_kw={"bruteForce": False})
    t_lowres = run_atlas("F) resolution=512 + bruteForce=False",
                         pack_kw={"bruteForce": False}, res=512)

    print("\n=== 结论 ===")
    base = max(t_default, 1e-6)
    for name, d in (("bruteForce=False", t_nobrute), ("max_iterations=1", t_fewiter),
                    ("both", t_both), ("res=512+nobrute", t_lowres)):
        print(f"  {name:22s} {d:7.2f}s  （相对默认 {base/d:.1f}x 加速）")

    # ---- 关键对照：真实网格（chart 数极多）----
    # 合成球体 9940 面只要 0.64s，而真实网格 79808 面要 813.92s —— 差三个数量级，
    # 说明瓶颈不是面数，而是**这团网格产生出的 chart 数量/形状**。
    print("\n=== 真实网格对照（同上配置） ===")
    try:
        import open3d as o3d

        real_ply = Path(__file__).resolve().parents[1] / \
            "workspace" / "cmp-resetfix" / "05_export" / "model_mesh.ply"
        mv = o3d.io.read_triangle_mesh(str(real_ply))
        rv = np.asarray(mv.vertices, dtype=np.float32)
        rt = np.asarray(mv.triangles, dtype=np.uint32)
        print(f"真实网格：面 {len(rt)}  顶点 {len(rv)}")
        # 只测"分 chart"这一步（跳过打包），看 chart 数是否是瓶颈
        atlas = xatlas.Atlas()
        atlas.add_mesh(rv, rt)
        t0 = time.time()
        atlas.generate(chart_options=xatlas.ChartOptions(),
                       pack_options=xatlas.PackOptions())
        print(f"H) 真实网格 generate（默认）    {time.time()-t0:7.2f}s  "
              f"charts={atlas.chart_count}")
    except Exception as exc:  # noqa: BLE001
        print(f"真实网格对照跳过：{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
