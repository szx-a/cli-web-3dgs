"""按**多视角掩码一致性**裁掉网格里混进来的背景顶点（转台模式的必要后处理）。

## 为什么需要

转台模式用 `mask.mode=keep` 保留背景（背景是"相机没动"的证据），
但代价是**背景也会被重建**。导出时若只按密度/分位裁剪，
背景物体（绿牌子、杂物箱）会留在网格里 ⇒ 渲染图中出现**真值里没有的东西**。
实测 img3-crop 的 view 200：渲染出现绿色牌子与灰色盒子，真值中是白墙。

## 判据

对一个**真实的物体表面点**：从任何能看到它的视角投影过去，都应落在**前景掩码内**。
对一个**背景点**：它只在一个或少数视角"恰好"落在掩码内，换个视角就露馅。

做法：
  1. 把网格顶点逐个投影到抽样的训练视角；
  2. 统计"投影落在掩码内的视角比例"；
  3. 保留比例 >= `--min-ratio` 的顶点，以及三顶点都保留的面；
  4. 顺带做"小连通块剔除"（背景残留通常是独立的小块）。

用法（需要 open3d，走 GPU 包装）:
  scripts\\gpu_python.cmd scripts\\filter_mesh_by_masks.py --job img3-crop --views 40
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
    ap.add_argument("--views", type=int, default=40, help="抽样多少视角（0=全部）")
    ap.add_argument("--min-ratio", type=float, default=0.6,
                    help="顶点需在多少比例的视角里落在掩码内（默认 0.6）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--in-ply", default="", help="输入网格（默认 05_export/model_mesh.ply）")
    ap.add_argument("--out", default="", help="输出（默认 05_export/model_mesh_clean.ply）")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d
    from PIL import Image

    from gsflow.gsdata.mesh_export import to_largest_component
    from gsflow.trainers.gsplat_trainer import load_colmap_data

    job = ROOT / "workspace" / args.job
    src = Path(args.in_ply) if args.in_ply else (job / "05_export" / "model_mesh.ply")
    if not src.exists():
        print(f"找不到 {src}")
        return 1

    mesh = o3d.io.read_triangle_mesh(str(src))
    if len(mesh.vertices) == 0:
        print(f"{src} 里没有顶点")
        return 1
    verts = np.asarray(mesh.vertices)

    data_dir = job / "04_train" / "data"
    d = load_colmap_data(data_dir, downscale=1, white_background=False, flip_camera=False)
    c2ws = np.asarray(d["c2ws"], dtype=np.float64)
    ks = np.asarray(d["ks"], dtype=np.float64)
    names = list(d["names"])

    mask_dir = job / "02_masks"
    masks: dict[int, np.ndarray] = {}
    for i, n in enumerate(names):
        stem = Path(n).stem
        for c in (mask_dir / f"{stem}.png", mask_dir / f"{n}.png"):
            if c.exists():
                with Image.open(c) as im:
                    masks[i] = np.asarray(im.convert("L")) > 127
                break

    print(f"网格：{len(verts)} 顶点 / {len(mesh.triangles)} 面；"
          f"可用掩码视角 {len(masks)}/{len(names)}")
    if not masks:
        print("没有掩码，无法按掩码筛选")
        return 1

    use = sorted(masks.keys())
    if args.views and len(use) > args.views:
        step = len(use) / args.views
        use = [use[int(k * step)] for k in range(args.views)]

    vh = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)   # [N,4]
    inside = np.zeros(len(verts), dtype=np.float64)
    counted = 0
    for i in use:
        m = masks[i]
        H, W = m.shape
        pc = (np.linalg.inv(c2ws[i]) @ vh.T).T[:, :3]
        z = pc[:, 2]
        ok = z > 1e-6
        uv = (ks[i] @ pc.T).T[:, :2] / np.maximum(z, 1e-9)[:, None]
        u = np.round(uv[:, 0]).astype(np.int64)
        v = np.round(uv[:, 1]).astype(np.int64)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(len(verts), dtype=bool)
        idx = np.nonzero(ok)[0]
        if idx.size:
            hit[idx] = m[v[idx], u[idx]]
        inside += hit
        counted += 1

    ratio = inside / max(counted, 1)
    keep_v = ratio >= args.min_ratio

    print(f"\n投影到 {counted} 个视角：")
    print(f"  顶点掩码内比例：p10={np.quantile(ratio,0.1):.2f} "
          f"p50={np.median(ratio):.2f} p90={np.quantile(ratio,0.9):.2f}")
    print(f"  保留顶点 {int(keep_v.sum())}/{len(verts)}（{keep_v.mean()*100:.1f}%）")

    if args.dry_run:
        print("\n（dry-run，未写文件）")
        return 0

    # 保留顶点 → 重建网格（三顶点都保留的面）
    tris = np.asarray(mesh.triangles)
    keep_f = keep_v[tris].all(axis=1)
    new_index = -np.ones(len(keep_v), dtype=np.int64)
    new_index[keep_v] = np.arange(int(keep_v.sum()))
    new_tris = new_index[tris[keep_f]]

    out_mesh = o3d.geometry.TriangleMesh()
    out_mesh.vertices = o3d.utility.Vector3dVector(verts[keep_v])
    out_mesh.triangles = o3d.utility.Vector3iVector(new_tris)
    if mesh.has_vertex_colors():
        out_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_colors)[keep_v])
    out_mesh.compute_vertex_normals()

    n_before = len(out_mesh.triangles)
    out_mesh, removed = to_largest_component(out_mesh)
    print(f"  保留面 {len(new_tris)}，剔除小连通块 {removed} 个三角形"
          f"（→ {len(out_mesh.triangles)}）")

    out = Path(args.out) if args.out else (job / "05_export" / "model_mesh_clean.ply")
    o3d.io.write_triangle_mesh(str(out), out_mesh, write_vertex_colors=True)
    print(f"\n已写入 {out}（{len(out_mesh.vertices)} 顶点 / {len(out_mesh.triangles)} 面）")
    (out.parent / "mesh_filter_report.json").write_text(json.dumps({
        "src": str(src), "views": counted, "min_ratio": args.min_ratio,
        "kept_vertices": int(keep_v.sum()), "kept_faces": int(new_tris.size // 3),
        "removed_small_components": int(removed),
        "final_faces": int(len(out_mesh.triangles)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
