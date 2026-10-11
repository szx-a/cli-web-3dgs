"""测量纹理烘焙各阶段耗时，并查看 xatlas 能否按 chart 烘焙。

动机：40 万面的网格上"UV 展开 + 烘焙"会卡住 >20 分钟（8 万面时约 2~3 分钟），
是当前"带贴图导出"在大网格上的瓶颈。要修就得先知道时间花在哪：
  ① xatlas 展开本身
  ② 逐三角形的"像素反推"（numpy，按批）
  ③ 每个像素 × 每个视角的投影/采样（视角数是大头）
  ④ 边缘膨胀

用法：scripts\\gpu_python.cmd scripts\\diag_bake_profile.py --job cmp-resetfix --faces 80000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="cmp-resetfix")
    ap.add_argument("--faces", type=int, default=80000)
    ap.add_argument("--tex", type=int, default=1024)
    ap.add_argument("--views", type=int, default=24)
    args = ap.parse_args()

    from gsflow import paths

    paths.apply_runtime_env()
    import open3d as o3d
    import xatlas

    # 1) xatlas 的能力
    print("=== xatlas 能力 ===")
    print("顶层:", [n for n in dir(xatlas) if not n.startswith("_")])
    if hasattr(xatlas, "Atlas"):
        a = xatlas.Atlas()
        print("Atlas 方法:", [n for n in dir(a) if not n.startswith("_")])

    # 2) 载入网格并简化到目标面数
    job = ROOT / "workspace" / args.job
    src = job / "05_export" / "model_mesh.ply"
    if not src.exists():
        src = job / "05_export" / "model_mesh.obj"
    print(f"\n载入 {src.name}")
    mesh = o3d.io.read_triangle_mesh(str(src))
    if len(mesh.triangles) > args.faces:
        mesh = mesh.simplify_quadric_decimation(args.faces)
    mesh.compute_vertex_normals()
    n_tri = len(mesh.triangles)
    print(f"面数 {n_tri}  顶点 {len(mesh.vertices)}")

    # 3) xatlas 展开耗时
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris = np.asarray(mesh.triangles, dtype=np.uint32)
    t = time.time()
    vmapping, new_tris, uvs = xatlas.parametrize(verts, tris)
    print(f"\nxatlas.parametrize: {time.time()-t:.2f}s  "
          f"新顶点 {len(vmapping)}  新面 {len(new_tris)}  UV {np.asarray(uvs).shape}")

    # 4) 逐三角形"像素反推"的规模估计
    new_tris = np.asarray(new_tris, dtype=np.int64)
    uv = np.asarray(uvs, dtype=np.float64)
    if uv.shape[0] == len(vmapping):
        corner = uv[new_tris]
    else:
        corner = uv.reshape(-1, 3, 2)
    tex = args.tex
    upx = corner * (tex - 1)
    a = upx[:, 0]
    b = upx[:, 1]
    c = upx[:, 2]
    area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) * 0.5
    est_px = float(area.sum())
    print(f"\nUV 三角形总面积 = {est_px:.0f} 像素"
          f"（@ {tex}²，理论上限 {tex*tex}）")
    print(f"平均每面 {est_px/max(len(new_tris),1):.1f} 像素")

    # 5) 真正的大头：像素 × 视角 的投影/采样
    #    当前实现按批(200 面)处理，每批把这些像素铺平后对全部视角做一次向量化投影
    n_px = int(est_px)
    n_views = args.views
    print(f"\n=== 投影/采样规模 ===")
    print(f"像素总数 {n_px}  ×  视角 {n_views}  =  {n_px*n_views/1e6:.1f} M 次投影")
    # 实测一次 1M 次投影的耗时
    P = np.random.randn(1_000_000, 3).astype(np.float64)
    w2c = np.eye(4)
    K = np.eye(3)
    t = time.time()
    vc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = vc[:, 2]
    u = (vc @ K[0, :3]) / np.maximum(z, 1e-9)
    v = (vc @ K[1, :3]) / np.maximum(z, 1e-9)
    ok = (z > 1e-6) & (u >= 0) & (u < 1000) & (v >= 0) & (v < 1000)
    dt = time.time() - t
    print(f"1M 像素 × 1 视角投影 = {dt:.3f}s  ->  预计 {n_px*n_views/1e6*dt/60:.1f} 分钟")

    print("\n=== 结论线索 ===")
    print("若上面预计时间与实测吻合，瓶颈就是「像素 × 视角」的投影循环；")
    print("改进方向：a) 按 chart 而非逐三角形核销  b) 限制每像素候选视角数"
          "  c) 降低参与烘焙的视角数  d) 用 GPU 做投影")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
