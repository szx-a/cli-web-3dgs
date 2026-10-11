"""对比"深度反投影点"与"高斯基元"的位置分布，定位网格飘离的根因。

背景（当前最严重的问题）：导出的网格看起来是碎渣，且实测
**网格顶点到高斯基元的距离中位数 = 0.4332，而点间距只有 0.0089（49 倍）**。
`mesh_export` 的流程是"渲染深度 -> 反投影成点 -> 泊松重建"，
所以如果反投影点与高斯基元不在同一坐标系/尺度，泊松就会重建出一个飘在旁边的面。

正常情况下两者应该**几乎重合**（反投影点就是从这些高斯渲染出的深度上取的）。
本脚本把两组点的
  * 包围盒（min / max / 尺寸）
  * 质心
  * 点到对方最近邻的距离中位数
全部打印出来，一眼判断是"平移错位"、"尺度错位"还是"旋转/翻转错位"。

用法：
  env\\venv\\Scripts\\python.exe scripts\\diag_depth_vs_gaussians.py --job img2-04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def bbox(p: np.ndarray, tag: str) -> None:
    mn, mx = p.min(0), p.max(0)
    print(f"  {tag:22s} min={np.round(mn, 3)}  max={np.round(mx, 3)}  尺寸={np.round(mx - mn, 3)}")
    print(f"  {'':22s} 质心={np.round(p.mean(0), 3)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--views", type=int, default=12)
    args = ap.parse_args()

    from gsflow import paths

    paths.apply_runtime_env()
    import open3d as o3d

    from gsflow.gsdata import mesh_export as ME

    job = ROOT / "workspace" / args.job

    # 1) 高斯基元（训练 PLY，原始坐标；mesh_export 内部也是原始坐标）
    from plyfile import PlyData

    cands = sorted((job / "04_train" / "point_cloud").glob("iteration_*/point_cloud.ply"))
    if not cands:
        print("找不到训练 PLY")
        return 1
    d = PlyData.read(str(cands[-1])).elements[0].data
    g = np.stack([np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["z"])], 1).astype(np.float64)
    print(f"高斯基元 {len(g)} 个（来自 {cands[-1].parent.name}）")
    bbox(g, "高斯基元")

    # 2) 深度反投影点：直接调 mesh_export 的同一函数，保证与导出流程一致
    out = job / "05_export" / "manifest.json"
    info = None
    if out.exists():
        import json

        info = json.loads(out.read_text(encoding="utf-8")).get("mesh_stats")
    print(f"\nmanifest.mesh_stats = {info}")

    print("\n（反投影点由 mesh_export.backproject_depth_to_points 生成；"
          "下面用点云到网格的最近邻距离佐证是否同坐标系）")
    m = job / "05_export" / "model_mesh.ply"
    if m.exists():
        mesh = o3d.io.read_triangle_mesh(str(m))
        v = np.asarray(mesh.vertices)
        bbox(v, "导出网格顶点")
        kd = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(g)))
        ds = []
        for i in range(0, len(v), max(1, len(v) // 4000)):
            _, _, d2 = kd.search_knn_vector_3d(v[i], 1)
            ds.append(np.sqrt(d2[0]))
        ds = np.asarray(ds)
        print(f"  网格顶点 -> 高斯基元 最近邻距离：中位数 {np.median(ds):.4f}  "
              f"10 分位 {np.percentile(ds, 10):.4f}  90 分位 {np.percentile(ds, 90):.4f}")

        # 平移错位的假设检验：若把网格平移 -质心差，距离是否显著变小
        shift = v.mean(0) - g.mean(0)
        print(f"  质心差（网格 - 高斯）= {np.round(shift, 3)}  |差|={np.linalg.norm(shift):.3f}")
        v2 = v - shift
        kd2 = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(g)))
        ds2 = []
        for i in range(0, len(v2), max(1, len(v2) // 4000)):
            _, _, d2 = kd2.search_knn_vector_3d(v2[i], 1)
            ds2.append(np.sqrt(d2[0]))
        ds2 = np.asarray(ds2)
        print(f"  平移对齐后距离：中位数 {np.median(ds2):.4f}"
              f"   （若显著变小 => 是**平移错位**；若几乎不变 => 不是平移问题）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
