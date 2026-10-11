"""对比"3DGS 点云"与"导出的网格"的尺度/分布，判断网格是不是把点云糊成一团。

动机（用户反馈"该空的地方是实的、三角面明显且锐利"）：
如果网格的包围盒明显**小于**点云、且大量网格顶点离最近点云很远，
那说明**输入点云本身不成立**（不是裁剪问题）——泊松在没有数据的地方
用隐函数"填"出了面，于是镂空物体的洞被填实。

输出：
  * 点云 / 网格的包围盒与尺度对比
  * 网格顶点到最近点云的距离分布（中位数 / p90 / p99）
  * 距离 > 若干倍点间距的顶点占比（这些就是"凭空填出来的面"）

用法：
  env\\venv\\Scripts\\python.exe scripts\\diag_mesh_vs_points.py --job img2-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    import open3d as o3d
    from scipy.spatial import cKDTree

    from gsflow.stages.export import ExportStage

    job = ROOT / "workspace" / args.job
    g = ExportStage()._read_ply(ExportStage._find_ply(job), max_points=0)
    pts = np.asarray(g["xyz"], dtype=np.float64)
    op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
    print(f"3DGS 点云：{len(pts)} 个高斯")
    print(f"  包围盒 min={np.round(pts.min(0),3)} max={np.round(pts.max(0),3)}")
    print(f"  尺度   {np.round(pts.max(0)-pts.min(0),3)}")
    for q in (0.5, 0.9):
        m = op >= np.quantile(op, q)
        p = pts[m]
        print(f"  不透明度 >= p{int(q*100)}（{m.sum()} 个）: "
              f"min={np.round(p.min(0),3)} max={np.round(p.max(0),3)}")

    mesh_p = job / "05_export" / "model_mesh.ply"
    if not mesh_p.exists():
        print(f"\n没有 {mesh_p.name}，跳过网格对比")
        return 0
    m = o3d.io.read_triangle_mesh(str(mesh_p))
    mv = np.asarray(m.vertices, dtype=np.float64)
    print(f"\n网格：{len(m.triangles)} 面 / {len(mv)} 顶点")
    print(f"  包围盒 min={np.round(mv.min(0),3)} max={np.round(mv.max(0),3)}")
    print(f"  尺度   {np.round(mv.max(0)-mv.min(0),3)}")

    # 点云典型间距
    step = max(1, len(pts) // 3000)
    sub = pts[::step][:3000]
    kd = cKDTree(pts)
    d_sub, _ = kd.query(sub, k=2, workers=-1)
    spacing = float(np.median(d_sub[:, 1]))
    print(f"\n点云典型间距（最近邻中位数）= {spacing:.5f}")

    dist, _ = cKDTree(pts).query(mv, k=1, workers=-1)
    for q in (0.5, 0.9, 0.99):
        print(f"  网格顶点到最近点云距离 p{int(q*100)} = {np.quantile(dist, q):.5f} "
              f"({np.quantile(dist, q)/max(spacing,1e-9):.1f} × 间距)")
    for k in (2, 3, 5, 10):
        frac = float((dist > k * spacing).mean())
        print(f"  距离 > {k:>2}× 间距的顶点占比 = {frac*100:5.1f}%"
              f"   ← 这些是没有数据支撑的'填面'")
    print("\n判读：若 p90 就有好几倍间距、且 >5× 的占比可观，"
          "说明网格大量区域**没有点云支撑**，问题在输入数据/位姿，不在裁剪。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
