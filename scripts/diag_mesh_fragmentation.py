"""量化网格的"碎片化/杂乱"程度，用于判断质量瓶颈在哪一步。

背景：导出的网格在查看器里看起来是"碎渣、没有结构"。要区分两种可能：
  A) **连通块很多** —— 泊松/裁剪把表面切碎了
  B) **连通块只有一个，但表面起伏剧烈、法线乱** —— 深度反投影的点太噪

指标：
  * 连通块数量、最大块占比      -> 判断 A
  * 面积分布（面积≈0 的碎三角） -> 判断 A
  * 邻域顶点法线夹角余弦        -> 判断 B（越小越乱，<0 表示翻转）
  * 网格到输入点云的距离        -> 判断表面是否"飘"在点云外

用法：
  env\\venv\\Scripts\\python.exe scripts\\diag_mesh_fragmentation.py --job img2-04 [--job img2-reg]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def analyze(job: str) -> None:
    import open3d as o3d

    p = ROOT / "workspace" / job / "05_export" / "model_mesh.ply"
    if not p.exists():
        print(f"{job}: 无 model_mesh.ply")
        return
    m = o3d.io.read_triangle_mesh(str(p))
    ntri = len(m.triangles)
    v = np.asarray(m.vertices)
    n = np.asarray(m.vertex_normals)
    print(f"=== {job} ===")
    print(f"  顶点 {len(v)}  三角形 {ntri}")

    # A) 连通块
    lab, cnt, area = m.cluster_connected_triangles()
    lab = np.asarray(lab)
    area = np.asarray(area)
    sizes = np.bincount(lab) if len(lab) else np.array([0])
    big = int(sizes.max()) if len(sizes) else 0
    print(f"  连通块 {len(sizes)} 个；最大块 {big} 面（占 {big / max(1, ntri) * 100:.1f}%）")
    tiny = int((area < 1e-8).sum())
    print(f"  面积 < 1e-8 的退化三角 {tiny} 个（{tiny / max(1, ntri) * 100:.1f}%）")
    print(f"  三角形总面积 {area.sum():.4f}；平均面积 {area.mean():.3e}")

    # B) 邻域法线一致性
    kd = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v)))
    cos: list[float] = []
    step = max(1, len(v) // 800)
    for i in range(0, len(v), step):
        _, idx, _ = kd.search_knn_vector_3d(v[i], 8)
        for j in idx[1:]:
            cos.append(float(np.dot(n[i], n[j])))
    cos_a = np.asarray(cos)
    print(f"  邻域法线一致 {cos_a.mean():.4f}；翻转(<0) 比例 {(cos_a < 0).mean() * 100:.1f}%")

    # C) 与输入点云的距离
    ply = None
    tr = ROOT / "workspace" / job / "04_train"
    if tr.exists():
        cands = sorted((tr / "point_cloud").glob("iteration_*/point_cloud.ply"))
        if cands:
            ply = cands[-1]
    if ply is not None:
        from plyfile import PlyData

        d = PlyData.read(str(ply)).elements[0].data
        xyz = np.stack([np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["z"])], 1).astype(np.float64)
        # 训练 PLY 是 +Y 上、导出网格做过 y/z 翻转
        xyz = xyz.copy()
        xyz[:, 1] *= -1
        xyz[:, 2] *= -1
        spacing = float(np.median(o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(xyz)).compute_nearest_neighbor_distance()))
        kd2 = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)))
        dists = []
        for i in range(0, len(v), max(1, len(v) // 3000)):
            _, _, d2 = kd2.search_knn_vector_3d(v[i], 1)
            dists.append(np.sqrt(d2[0]))
        dists_a = np.asarray(dists)
        thr = spacing * 5
        print(f"  输入点间距中位数 {spacing:.4f}")
        print(f"  网格顶点到点云距离：中位数 {np.median(dists_a):.4f}  "
              f"90 分位 {np.percentile(dists_a, 90):.4f}  最大 {dists_a.max():.4f}")
        print(f"  **距离 > 5×点间距 的顶点比例 {(dists_a > thr).mean() * 100:.1f}%**（越高越说明表面飘离点云）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True)
    args = ap.parse_args()
    for j in args.job:
        analyze(j)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
