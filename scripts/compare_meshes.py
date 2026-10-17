"""对比两份网格（滤背景前后）的结构与质量，验证"背景泄漏已剔除"。

判据（每一条都能对应一个可观测现象）：
  * **面数变化**：滤掉的应是背景残留（实测 59,365 → 42,323，−28.7%）；
  * **法线翻转率**：只保留物体后应当**下降**（背景那块是独立的小平面，法线混乱）；
  * **包围盒**：滤掉背景后 x/z 跨度应收窄（绿牌子/杂物盒在物体左侧）；
  * **连通分量数**：滤后应更少（背景残留是独立小块）。

用法: scripts\\gpu_python.cmd scripts\\compare_meshes.py --a model_mesh.ply --b model_mesh_clean.ply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def stats(path: Path) -> dict:
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(path))
    v = np.asarray(m.vertices)
    t = np.asarray(m.triangles)
    out = {"file": path.name, "vertices": len(v), "faces": len(t)}
    if len(v):
        out["bbox"] = np.round(v.max(0) - v.min(0), 3).tolist()
        out["center"] = np.round(v.mean(0), 3).tolist()
    # 法线翻转：相邻三角形法线点积 < 0 的比例
    if len(t) > 1:
        m.compute_vertex_normals()
        m.compute_triangle_normals()
        tn = np.asarray(m.triangle_normals)
        # 按共享顶点找相邻面
        from collections import defaultdict
        edge2tri = defaultdict(list)
        for ti, (a, b, c) in enumerate(t):
            for e in ((min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))):
                edge2tri[e].append(ti)
        dots = []
        for e, ts in edge2tri.items():
            if len(ts) == 2:
                dots.append(float(np.dot(tn[ts[0]], tn[ts[1]])))
        if dots:
            d = np.asarray(dots)
            out["mean_dot"] = round(float(d.mean()), 4)
            out["flip_ratio"] = round(float((d < 0).mean()), 4)
            out["n_pairs"] = len(d)
    # 连通分量
    try:
        _, n_comp, _ = m.cluster_connected_triangles()
        out["components"] = int(len(np.asarray(n_comp)))
    except Exception:  # noqa: BLE001
        out["components"] = -1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    pa, pb = Path(args.a), Path(args.b)
    for p in (pa, pb):
        if not p.exists():
            print(f"找不到 {p}")
            return 1

    ra, rb = stats(pa), stats(pb)
    print(f"=== 网格对比 ===\n")
    keys = ["file", "vertices", "faces", "mean_dot", "flip_ratio", "n_pairs",
            "components", "bbox", "center"]
    print(f"  {'指标':<14}{'A':>26}{'B':>26}")
    for k in keys:
        va, vb = ra.get(k, "-"), rb.get(k, "-")
        print(f"  {k:<14}{str(va):>26}{str(vb):>26}")

    fa, fb = ra["faces"], rb["faces"]
    if fa:
        print(f"\n  面数变化：{fa} → {fb}（{(fb-fa)/fa*100:+.1f}%）")
    if "flip_ratio" in ra and "flip_ratio" in rb:
        d = rb["flip_ratio"] - ra["flip_ratio"]
        print(f"  法线翻转率：{ra['flip_ratio']*100:.2f}% → {rb['flip_ratio']*100:.2f}%"
              f"（{d*100:+.2f} pp）")
        print(f"  ⇒ {'改善' if d < 0 else '变差' if d > 0 else '不变'}")
    if ra.get("components", -1) >= 0 and rb.get("components", -1) >= 0:
        print(f"  连通分量：{ra['components']} → {rb['components']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
