"""验证纹理 54% 黑洞的真因：**网格法线朝内**导致烘焙时所有视角都被 facing 剔除。

## 推理链

`bake_texture_from_views` 里：
```python
vd     = P - cam_center          # 表面点 -> 相机
facing = -(N * vd).sum(axis=1)   # N=表面法线
ok    &= facing > 0.15           # 只接受"相机朝向表面正面"的视角
```
若网格法线**朝内**，则 `facing` 对任何视角都是负的 ⇒ `ok` 全 False
⇒ 该 texel 永远填不上 ⇒ 保持黑色。

实测 img3-crop：**53.96% 的 UV 覆盖 texel 是黑的** —— 与"部分区域法线朝内"吻合。

## 本脚本判据

对网格每条边检查相邻面的法线是否一致朝外：
  * 用**有符号体积**判断整体朝向：`sum((v0 × v1) · v2) / 6`，
    封闭网格若法线朝外则该值为正；朝内则为负；
  * 统计"面法线与'从物体中心指向该面'的点积 > 0"的比例
    （朝外的面应占多数）。

若确认法线朝内，修法是烘焙前 **翻转法线**（或用 `mesh.is_watertight()`
与体积符号自动判定）。

用法: scripts\\gpu_python.cmd scripts\\check_normal_orientation.py --job img3-crop
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
    ap.add_argument("--ply", default="")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d

    job = ROOT / "workspace" / args.job
    ply = Path(args.ply) if args.ply else (job / "05_export" / "model_mesh.ply")
    mesh = o3d.io.read_triangle_mesh(str(ply))
    print(f"网格 {ply.name}: {len(mesh.vertices)} 顶点 / {len(mesh.triangles)} 面")
    print(f"  is_watertight = {mesh.is_watertight()}")
    print(f"  is_orientable = {mesh.is_orientable()}")

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    tn = np.asarray(mesh.triangle_normals)

    # ① 有符号体积（仅对封闭网格有意义）
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    vol = float(np.einsum("ij,ij->i", np.cross(p0, p1), p2).sum() / 6.0)
    print(f"\n  ① 有符号体积 = {vol:.6f}"
          f"（封闭网格：正=法线朝外，负=朝内）")

    # ② 面法线 vs "从中心指向该面"
    c = v.mean(0)
    fc = (p0 + p1 + p2) / 3.0 - c
    fc /= np.maximum(np.linalg.norm(fc, axis=1, keepdims=True), 1e-9)
    dot = np.einsum("ij,ij->i", tn, fc)
    frac_out = float((dot > 0).mean())
    print(f"  ② 面法线与'中心→面'同向的比例 = {frac_out*100:.1f}%"
          f"（朝外应占多数，>50%）")

    # ③ 顶点法线的同样统计
    vn = np.asarray(mesh.vertex_normals)
    vc = v - c
    vc /= np.maximum(np.linalg.norm(vc, axis=1, keepdims=True), 1e-9)
    vdot = np.einsum("ij,ij->i", vn, vc)
    frac_vout = float((vdot > 0).mean())
    print(f"  ③ 顶点法线与'中心→顶点'同向的比例 = {frac_vout*100:.1f}%")

    print("\n=== 判读 ===")
    inward = (frac_out < 0.5) or (vol < 0 and mesh.is_watertight())
    if inward:
        print(f"  ⇒ **法线朝内**（面法线朝外比例仅 {frac_out*100:.1f}%）")
        print("     烘焙时 `facing = -(N·vd) > 0.15` 会几乎全部失败 ⇒ 纹理大量黑色。")
        print("     修法：烘焙前 `mesh.orient_triangles()` 或用体积符号判定后翻转。")
    else:
        print(f"  ⇒ 法线整体朝外（{frac_out*100:.1f}%）")
        print("     纹理黑洞另有原因：可能是 UV 覆盖量大但**可见性判据过严**")
        print("     （facing > 0.15 对薄片/镂空结构会排除大部分视角）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
