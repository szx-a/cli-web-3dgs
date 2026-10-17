"""检查"前景掩码一致性"判据在**转台**素材上是否有效 —— 我怀疑它原理上就错了。

## 问题的由来

我按"顶点投影落进前景掩码的比例"给网格顶点分类，得到：
  * 纯背景（比例≈0）24.3%
  * 混合（0.05~0.5）6.6%
  * 纯前景（≈1）59.1%
并推断"31% 是背景几何"，准备据此过滤。

但**预览数据的形态否定了这个解释**：
被删顶点的包围盒 `[1.87, 0.98, 0.86]` 几乎等于全网格 `[1.87, 1.08, 0.86]`，
中心也几乎重合 ⇒ 被删的**不是外围外壳，而是散布在整个物体里的**。

## 原理上的缺陷

转台拍法下相机固定、物体旋转。掩码是**相机固定坐标下的前景区域**。
对某个物体表面点：
  * 当它转到**正面**时，投影落在掩码内 ⇒ 计入"前景"；
  * 当它转到**背面**时，它可能：
      (a) 被物体自身遮挡（看不见），
      (b) 或仍然可见但投影在**轮廓之外**（例如它是侧面，转到某个角度露出）。
  ⇒ **"落在掩码内"不能区分"物体点"与"背景点"**，
     因为背景点也可能投影进掩码（掩码是按物体分出来的矩形/像素区域，
     背景的东西若恰好在掩码区域里也会被算进去）。

更根本的问题：**这不是一个"物体 vs 背景"的分类问题**，
而是"这个面**有没有被任何相机看到过**"的问题。
没被看到的面，纹理必然黑 —— 但那不等于它该被删除
（它可能是物体真实的一部分，只是当前视角看不到）。

## 正确的判据

区分两件事：
  1. **是否被看到过**：对每个面计算"有多少视角能看见它"（投影在图像内 + 面朝相机）。
     0 个视角看到 ⇒ 纹理必然黑（无法修复，只能补拍或接受）。
  2. **是否属于物体**：应由**语义分割/连通性**判断，而不是投影一致性。

本脚本量化 1，并检验"未被看到的面"是否与"纹理黑洞"在空间上对应。

用法: scripts\\gpu_python.cmd scripts\\check_visibility_vs_holes.py --job img3-crop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--views", type=int, default=0)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d

    from gsflow.trainers.gsplat_trainer import load_colmap_data

    job = ROOT / "workspace" / args.job
    ply = job / "05_export" / "model_mesh.ply"
    mesh = o3d.io.read_triangle_mesh(str(ply))
    mesh.compute_triangle_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    tn = np.asarray(mesh.triangle_normals, dtype=np.float64)
    print(f"网格 {len(verts)} 顶点 / {len(tris)} 面")

    d = load_colmap_data(job / "04_train" / "data", downscale=1,
                         white_background=False, flip_camera=False)
    c2ws = np.asarray(d["c2ws"], dtype=np.float64)
    ks = np.asarray(d["ks"], dtype=np.float64)
    names = list(d["names"])
    from PIL import Image
    # 图像尺寸（用于判断投影是否落在图像内）
    sizes = []
    for n in names:
        p = job / "04_train" / "data" / "images" / n
        if p.exists():
            with Image.open(p) as im:
                sizes.append(im.size)      # (W,H)
        else:
            sizes.append((0, 0))
    V = len(names)
    use = list(range(V))
    if args.views and V > args.views:
        step = V / args.views
        use = [int(k * step) for k in range(args.views)]
    print(f"使用 {len(use)} 个视角")

    # 面心与法线
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fc = (p0 + p1 + p2) / 3.0

    seen = np.zeros(len(tris), dtype=np.int64)
    for i in use:
        w2c = np.linalg.inv(c2ws[i])
        W, H = sizes[i]
        if W == 0:
            continue
        pc = fc @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        ok = z > 1e-6
        u = (pc @ ks[i][0, :3]) / np.maximum(z, 1e-9)
        v = (pc @ ks[i][1, :3]) / np.maximum(z, 1e-9)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # 面朝相机：法线与"面心->相机"同向
        cam_c = c2ws[i][:3, 3]
        vd = cam_c - fc
        vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-9)
        facing = np.einsum("ij,ij->i", tn, vd)
        ok &= facing > 0.1
        seen += ok.astype(np.int64)

    print(f"\n=== 面被多少个视角看到 ===")
    for lo, hi, label in ((0, 0, "0（从未被看到）"), (1, 2, "1~2"), (3, 9, "3~9"),
                          (10, 49, "10~49"), (50, 10**9, ">=50")):
        if lo == 0 and hi == 0:
            n = int((seen == 0).sum())
        else:
            n = int(((seen >= lo) & (seen <= hi)).sum())
        print(f"  {label:<18} {n:>7} 面（{n/len(tris)*100:5.1f}%）")

    never = seen == 0
    print(f"\n  从未被看到的面 = {int(never.sum())}（{never.mean()*100:.1f}%）")
    if never.any():
        nv = fc[never]
        print(f"  它们的位置包围盒 {np.round(nv.max(0)-nv.min(0), 2).tolist()}"
              f"（全网格 {np.round(fc.max(0)-fc.min(0), 2).tolist()}）")
        print(f"  它们到中心距离：均值 {np.linalg.norm(nv - fc.mean(0), axis=1).mean():.2f}"
              f"，全网格 {np.linalg.norm(fc - fc.mean(0), axis=1).mean():.2f}")
        print("\n  判读：若这些面**集中在内部或背面**，说明是遮挡/未覆盖（合理）；")
        print("        若散布在整个包围盒里，说明法线朝向有问题（facing 判据失效）。")

    rep = {"faces": len(tris), "never_seen": int(never.sum()),
           "never_seen_frac": round(float(never.mean()), 4),
           "median_views": int(np.median(seen))}
    (job / "05_export" / "mesh_visibility_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
