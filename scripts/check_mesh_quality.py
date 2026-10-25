"""网格体检：客观量化"破不破"，不依赖渲染。

## 为什么需要

`render_textured_mesh.py` 能出图，但**渲染结果会受材质/绕序/后端影响**
（实测同一相机同一材质：Poisson 网格渲得出、TSDF 网格全黑），
不能只凭"图黑不黑"判断网格好坏。所以补一套**纯几何**指标：

| 指标 | 含义 | 好网格的期望 |
|---|---|---|
| `boundary_edge_ratio` | 边界边 / 总边（"开缝"程度）| 封闭物体应 **接近 0** |
| `non_manifold_ratio` | 非流形边占比（三面共边）| **0** |
| `watertight` | 是否闭合 | **True** |
| `components` | 连通块数量 | 1（或个位数）|
| `area_over_hull` | 面积 / 凸包面积 | 越小越"实"，>1 说明有内部冗余面 |
| `volume_over_hull` | 体积 / 凸包体积 | 接近 1 说明是实心体而不是壳 |
| `z_thickness_ratio` | 最薄方向厚度 / 最长边 | <0.05 说明是一张"片" |

用法:
    python scripts/check_mesh_quality.py --obj a.obj b.obj ...
    python scripts/check_mesh_quality.py --job img4-v1      # 查 05_export 里的网格
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _metrics(path: Path) -> dict:
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(path))
    out = {"file": path.name, "triangles": len(m.triangles), "vertices": len(m.vertices)}
    if len(m.triangles) == 0:
        return out
    bb = m.get_axis_aligned_bounding_box()
    ext = np.asarray(bb.get_extent(), dtype=float)
    out["bbox"] = np.round(ext, 3).tolist()
    out["z_thickness_ratio"] = float(ext.min() / max(ext.max(), 1e-9))

    edges = np.asarray(m.edges) if hasattr(m, "edges") else None
    try:
        m.compute_adjacency_list()
    except Exception:  # noqa: BLE001
        pass
    # 边界边 / 非流形：按"每条边被多少个三角形共享"统计
    tri = np.asarray(m.triangles, dtype=np.int64)
    from collections import Counter

    cnt: Counter = Counter()
    for a, b, c in tri:
        for e in ((a, b), (b, c), (c, a)):
            cnt[(min(e), max(e))] += 1
    total_e = max(len(cnt), 1)
    boundary = sum(1 for v in cnt.values() if v == 1)
    nonman = sum(1 for v in cnt.values() if v > 2)
    out["boundary_edge_ratio"] = boundary / total_e
    out["non_manifold_ratio"] = nonman / total_e
    out["watertight"] = bool(m.is_watertight()) if hasattr(m, "is_watertight") else None

    try:
        labels = np.asarray(m.cluster_connected_triangles()[0])
        out["components"] = int(labels.max() + 1) if labels.size else 0
    except Exception:  # noqa: BLE001
        out["components"] = None

    area = float(m.get_surface_area())
    out["area"] = area
    hull = m.compute_convex_hull()[0] if hasattr(m, "compute_convex_hull") else None
    if hull is not None and len(hull.triangles):
        h_area = float(hull.get_surface_area())
        out["area_over_hull"] = area / max(h_area, 1e-9)
        try:
            out["volume_over_hull"] = float(m.get_volume()) / max(float(hull.get_volume()), 1e-9)
        except Exception:  # noqa: BLE001
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", nargs="*", default=[])
    ap.add_argument("--job", default="")
    args = ap.parse_args()

    paths = [Path(p) for p in args.obj]
    if args.job and not paths:
        d = ROOT / "workspace" / args.job / "05_export"
        paths = sorted(p for p in d.glob("model_mesh*.obj"))

    if not paths:
        print("没有输入")
        return 1
    print(f"=== 网格体检：{len(paths)} 个 ===")
    rows = []
    for p in paths:
        if not p.exists():
            print(f"  缺失：{p}")
            continue
        rows.append(_metrics(p))
    for r in rows:
        print(f"\n  {r['file']}")
        for k, v in r.items():
            if k == "file":
                continue
            if isinstance(v, float):
                print(f"    {k:<22} {v:.4f}")
            else:
                print(f"    {k:<22} {v}")
    # 横比
    if len(rows) > 1:
        print("\n  横比（越小越好：boundary / non_manifold；越接近 1 越好：volume_over_hull）")
        for r in rows:
            print("    %-22s boundary=%.4f nonman=%.4f watertight=%s vol/hull=%s" % (
                r["file"], r.get("boundary_edge_ratio", float("nan")),
                r.get("non_manifold_ratio", float("nan")), r.get("watertight"),
                ("%.3f" % r["volume_over_hull"]) if "volume_over_hull" in r else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
