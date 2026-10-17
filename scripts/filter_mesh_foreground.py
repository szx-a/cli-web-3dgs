"""按"前景掩码一致性"过滤网格 —— **先预览再应用**，避免误删物体细节。

## 为什么需要（数据支撑）

img3-crop 用 `mask.mode=keep`（转台必须保留背景作为"相机没动"的证据），
代价是网格里混入了背景几何。实测：
  * **纯背景顶点 24.3%**（可见时从不落在前景掩码内）；
  * **混合 6.6%**（0.05~0.5）；
  * 纯前景/物体 **59.1%**（≈1）。
  ⇒ 约 **31%** 的几何是背景。

后果：烘焙只从相机视角采样颜色，背景面没有任何相机**正面**看到
⇒ 对应 UV texel 保持黑色 ⇒ 实测 **43.7% 的 chart 内部是黑的**（模型上大片黑斑）。

## 上次的教训（必须内置）

上一轮我看到渲染里有"绿牌子、灰台阶"就判定为背景泄漏，写了过滤脚本删掉 28.7% 顶点 ——
**回原始帧后发现那是牌坊本身的细节**（712/712 绿色像素都在前景掩码内）。
所以本脚本**默认只预览**，必须显式 `--apply` 才写文件，并且：

  1. 输出**多种阈值**下的保留量，让人判断哪个合理；
  2. 报告**被删顶点的空间分布**（包围盒），若被删的是"外围薄壳"多半是背景，
     若删掉的是"内部大块"那可能是误删；
  3. 保留 `--keep-ratio` 可调（默认 0.5）。

用法:
  scripts\\gpu_python.cmd scripts\\filter_mesh_foreground.py --job img3-crop            # 预览
  scripts\\gpu_python.cmd scripts\\filter_mesh_foreground.py --job img3-crop --apply    # 应用
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
    ap.add_argument("--views", type=int, default=0, help="0=全部视角")
    ap.add_argument("--apply", action="store_true", help="写出过滤后的网格")
    ap.add_argument("--keep-ratio", type=float, default=0.5,
                    help="保留'可见时落在前景掩码内比例' >= 该值的顶点")
    ap.add_argument("--ply", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d
    from PIL import Image

    from gsflow.gsdata.mesh_export import to_largest_component
    from gsflow.trainers.gsplat_trainer import load_colmap_data

    job = ROOT / "workspace" / args.job
    ply = Path(args.ply) if args.ply else (job / "05_export" / "model_mesh.ply")
    mesh = o3d.io.read_triangle_mesh(str(ply))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    print(f"网格 {ply.name}: {len(verts)} 顶点 / {len(tris)} 面")

    d = load_colmap_data(job / "04_train" / "data", downscale=1,
                         white_background=False, flip_camera=False)
    c2ws = np.asarray(d["c2ws"], dtype=np.float64)
    ks = np.asarray(d["ks"], dtype=np.float64)
    names = list(d["names"])
    mask_dir = job / "02_masks"
    masks = {}
    for i, n in enumerate(names):
        stem = Path(n).stem
        for c in (mask_dir / f"{stem}.png", mask_dir / f"{n}.png"):
            if c.exists():
                with Image.open(c) as im:
                    masks[i] = np.asarray(im.convert("L")) > 127
                break
    use = sorted(masks)
    if args.views and len(use) > args.views:
        step = len(use) / args.views
        use = [use[int(k * step)] for k in range(args.views)]

    vh = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    inside = np.zeros(len(verts))
    visible = np.zeros(len(verts))
    for i in use:
        m = masks[i]
        H, W = m.shape
        pc = (np.linalg.inv(c2ws[i]) @ vh.T).T[:, :3]
        z = pc[:, 2]
        ok = z > 1e-6
        uv = (ks[i] @ pc.T).T[:, :2] / np.maximum(z, 1e-9)[:, None]
        u = np.round(uv[:, 0]).astype(np.int64)
        v = np.round(uv[:, 1]).astype(np.int64)
        ink = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(len(verts), dtype=bool)
        idx = np.nonzero(ink)[0]
        if idx.size:
            hit[idx] = m[v[idx], u[idx]]
        inside += hit
        visible += ink

    ratio = np.where(visible > 0, inside / np.maximum(visible, 1), 1.0)
    # 从未可见的顶点**保留**（它们不是"背景"的证据不足，宁可保留）

    print(f"\n=== 不同阈值下的保留量（预览，不做任何修改）===")
    print(f"  {'阈值':<8}{'保留顶点':>10}{'保留面':>10}{'删掉顶点的包围盒':>30}")
    for thr in (0.2, 0.3, 0.5, 0.7, 0.9):
        keep = ratio >= thr
        kept_tris = keep[tris].all(axis=1)
        drop = ~keep
        if drop.any():
            dv = verts[drop]
            bb = np.round(dv.max(0) - dv.min(0), 2).tolist()
            bbs = str(bb)
        else:
            bbs = "（无删除）"
        print(f"  {thr:<8}{int(keep.sum()):>10}{int(kept_tris.sum()):>10}{bbs:>30}")

    keep = ratio >= args.keep_ratio
    kept_tris = keep[tris].all(axis=1)
    new_index = -np.ones(len(keep), dtype=np.int64)
    new_index[keep] = np.arange(int(keep.sum()))
    drop = ~keep
    print(f"\n选定阈值 {args.keep_ratio}："
          f"保留 {int(keep.sum())}/{len(verts)} 顶点、"
          f"{int(kept_tris.sum())}/{len(tris)} 面")
    if drop.any():
        dv = verts[drop]
        amax = verts.max(0)
        amin = verts.min(0)
        inside_frac = float(((dv >= amin) & (dv <= amax)).all(axis=1).mean())
        print(f"  被删顶点包围盒 {np.round(dv.max(0)-dv.min(0),2).tolist()}"
              f"（全网格 {np.round(amax-amin,2).tolist()}）")
        print(f"  被删顶点中心 {np.round(dv.mean(0),2).tolist()}"
              f"，全网格中心 {np.round(verts.mean(0),2).tolist()}")
        print(f"  ⇒ 若被删的是'外围',删除合理；若集中在物体中心区域，需警惕误删")

    if not args.apply:
        print("\n（预览模式，未写文件。确认合理后加 --apply）")
        return 0

    out_mesh = o3d.geometry.TriangleMesh()
    out_mesh.vertices = o3d.utility.Vector3dVector(verts[keep])
    out_mesh.triangles = o3d.utility.Vector3iVector(new_index[tris[kept_tris]])
    if mesh.has_vertex_colors():
        out_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_colors)[keep])
    out_mesh.compute_vertex_normals()
    out_mesh, removed = to_largest_component(out_mesh)
    out = Path(args.out) if args.out else (job / "05_export" / "model_mesh_fg.ply")
    o3d.io.write_triangle_mesh(str(out), out_mesh, write_vertex_colors=True)
    print(f"\n已写入 {out}（{len(out_mesh.vertices)} 顶点 / "
          f"{len(out_mesh.triangles)} 面，剔除小连通块 {removed} 面）")
    (out.parent / "mesh_fg_filter_report.json").write_text(json.dumps({
        "src": str(ply), "keep_ratio": args.keep_ratio, "views": len(use),
        "kept_vertices": int(keep.sum()), "kept_faces": int(kept_tris.sum()),
        "final_faces": int(len(out_mesh.triangles)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
