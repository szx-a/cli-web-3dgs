"""诊断 UV 碎片化的真因：**网格法线噪声**，并验证"平滑法线"能否改善。

## 背景

img3-crop 的纹理图集有上千个碎块。`xatlas.parametrize` 在 0.0.11 里
**只接受 `(positions, indices, normals, uvs)`**，不暴露 ChartOptions，
所以唯一能影响切分的输入就是 **`normals`**。

xatlas 的切分规则：**法线夹角超过阈值的地方切**。如果网格法线带噪声
（泊松重建 + 简化的典型问题），表面会被切得极碎。

## 本脚本做什么

对同一网格跑四种法线输入，比较 UV 质量（岛数、占用率、前 10 块占比）：
  1. **无 normals**（当前实现，xatlas 自己算）
  2. **顶点法线**（open3d `compute_vertex_normals`）
  3. **平滑后的顶点法线**（先做 Taubin 平滑再算法线）
  4. **二次平滑法线**（更强平滑）

判据：若"平滑法线"显著减少岛数 ⇒ 碎片化的真因是法线噪声，
修法是**在 unwrap 前平滑法线**（而不是调 xatlas 参数，因为调不到）。

用法: scripts\\gpu_python.cmd scripts\\diag_uv_fragmentation.py --job img3-crop
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def uv_stats(uv: np.ndarray, tris: np.ndarray, tex: int = 512) -> dict:
    """光栅化 UV 占用图（画满三角形），统计连通块。"""
    img = np.zeros((tex, tex), dtype=np.uint8)
    step = max(1, len(tris) // 60000)
    for t in tris[::step]:
        try:
            p = np.clip(uv[t] * (tex - 1), 0, tex - 1).astype(np.int32)
        except Exception:  # noqa: BLE001
            continue
        # 用三角形边界 + 重心近似填充
        xs, ys = p[:, 0], p[:, 1]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if x1 - x0 < 400 and y1 - y0 < 400:
            img[y0:y1 + 1, x0:x1 + 1] |= 1
        else:
            img[ys, xs] = 1
    occ = float((img > 0).mean())
    try:
        from scipy import ndimage
        lab, n = ndimage.label(img > 0)
        sizes = np.bincount(lab.ravel())[1:]
        n_big = int((sizes > 30).sum())
        top10 = float(np.sort(sizes)[::-1][:10].sum() / max(sizes.sum(), 1))
        return {"occ": occ, "islands": int(n), "big": n_big, "top10": top10}
    except ImportError:
        return {"occ": occ, "islands": -1, "big": -1, "top10": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--ply", default="")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d
    import xatlas

    from gsflow.gsdata.mesh_export import smooth_mesh

    job = ROOT / "workspace" / args.job
    ply = Path(args.ply) if args.ply else (job / "05_export" / "model_mesh.ply")
    base = o3d.io.read_triangle_mesh(str(ply))
    verts = np.asarray(base.vertices, dtype=np.float32)
    tris = np.asarray(base.triangles, dtype=np.uint32)
    print(f"网格 {ply.name}: {len(verts)} 顶点 / {len(tris)} 面")

    def try_unwrap(label: str, normals) -> dict | None:
        try:
            if normals is None:
                vm, nt, uvs = xatlas.parametrize(verts, tris)
            else:
                vm, nt, uvs = xatlas.parametrize(
                    verts, tris, np.asarray(normals, dtype=np.float32))
            st = uv_stats(np.asarray(uvs, dtype=np.float64),
                          np.asarray(nt, dtype=np.int64))
            print(f"  {label:<26}{st['islands']:>8}{st['occ']*100:>9.1f}%"
                  f"{st['big']:>9}{st['top10']*100:>10.1f}%")
            return st
        except Exception as e:  # noqa: BLE001
            print(f"  {label:<26}  失败 {type(e).__name__}: {e}")
            return None

    print(f"\n  {'输入法线':<26}{'岛数':>8}{'占用率':>9}{'大块数':>9}{'前10占比':>10}")
    print("  " + "-" * 62)

    r0 = try_unwrap("① 无 normals（现状）", None)

    base.compute_vertex_normals()
    vn = np.asarray(base.vertex_normals, dtype=np.float32)
    r1 = try_unwrap("② 顶点法线", vn)

    # 平滑后再算法线
    for iters, tag in ((5, "③ 平滑5次后法线"), (15, "④ 平滑15次后法线")):
        try:
            sm = smooth_mesh(o3d.geometry.TriangleMesh(base), method="taubin",
                             iterations=iters)
            sm.compute_vertex_normals()
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:<26}  平滑失败 {type(e).__name__}: {e}")
            continue
        # ⚠️ smooth_mesh 不改变顶点数与顺序（Taubin 只移动顶点），可直接复用 tris
        vn2 = np.asarray(sm.vertex_normals, dtype=np.float32)
        if vn2.shape[0] != verts.shape[0]:
            print(f"  {tag:<26}  顶点数变了（{vn2.shape[0]} != {verts.shape[0]}），跳过")
            continue
        try_unwrap(tag, vn2)

    print("\n判读：若平滑法线显著减少岛数 ⇒ 碎片化的真因是**法线噪声**，")
    print("      修法是在 unwrap 前平滑法线（xatlas 0.0.11 调不到 ChartOptions）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
