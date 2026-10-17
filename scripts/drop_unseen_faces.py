"""删除"从未被任何视角看到"的网格面 —— 泊松重建的内部垃圾几何。

## 数据支撑（本轮诊断）

img3-crop 网格（泊松重建，`mask.mode=keep`）：

| 面被多少视角看到 | 面数 | 占比 |
|---|---|---|
| **0（从未被看到）** | **15,190** | **25.6%** |
| 1~2 | 2,022 | 3.4% |
| 3~9 | 5,958 | 10.0% |
| 10~49 | 12,304 | 20.7% |
| >=50 | 23,839 | 40.2% |

而"从未被看到"的面：
  * 包围盒 `[1.82, 1.05, 0.84]` ≈ 全网格 `[1.86, 1.08, 0.86]`
  * 到中心距离 0.46 ≈ 全网格 0.49
  ⇒ **散布在整个物体内部**，不是"背面遮挡"。

这是泊松重建的典型产物：**内部自交面**（物体被包了一层壳，内部也生成了面）。
它们：
  1. 让网格看起来"胖"、法线混乱；
  2. 在 UV 展开里**白占 texture 空间**，却**烘不出颜色** ⇒ 纹理黑洞
     （实测 43.7% 的 chart 内部是黑的 —— 与 25.6% 内部面 + 边界/遮挡吻合）；
  3. 对渲染无贡献（本来就是内部）。

## 判据（硬事实，非启发式）

一个面在某个视角"被看到" = 投影落在图像内 **且** 面法线朝向该相机。
统计所有视角后，**被看到次数 = 0** 的面可以直接删除：
它们在外观上不可见，删除不改变渲染结果。

⚠️ 与上一轮误删的对比：上轮我用"落在前景掩码内的比例"当判据，
那个判据在转台素材上**原理就错**（物体旋转使同一面在掩码内外来回变）。
本判据是"是否被相机看到过"，与掩码无关，因此不受转台/绕行影响。

用法:
  scripts\\gpu_python.cmd scripts\\drop_unseen_faces.py --job img3-crop            # 预览
  scripts\\gpu_python.cmd scripts\\drop_unseen_faces.py --job img3-crop --apply    # 应用
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
    ap.add_argument("--min-views", type=int, default=1,
                    help="保留'被至少这么多视角看到'的面（默认 1）")
    ap.add_argument("--apply", action="store_true")
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

    # ---- 统一朝外（否则 facing 判据会把内部面当成"可见"）----
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    c = verts.mean(0)
    fc = (p0 + p1 + p2) / 3.0
    fc_dir = fc - c
    fc_dir /= np.maximum(np.linalg.norm(fc_dir, axis=1, keepdims=True), 1e-12)
    flip = np.einsum("ij,ij->i", fn, fc_dir) < 0
    if flip.any():
        tris[flip] = tris[flip][:, [0, 2, 1]]
        fn[flip] = -fn[flip]
        print(f"  先统一朝向：翻转 {int(flip.sum())} 个面的绕序")

    d = load_colmap_data(job / "04_train" / "data", downscale=1,
                         white_background=False, flip_camera=False)
    c2ws = np.asarray(d["c2ws"], dtype=np.float64)
    ks = np.asarray(d["ks"], dtype=np.float64)
    names = list(d["names"])
    sizes = []
    for n in names:
        p = job / "04_train" / "data" / "images" / n
        if p.exists():
            with Image.open(p) as im:
                sizes.append(im.size)
        else:
            sizes.append((0, 0))
    V = len(names)
    use = list(range(V))
    if args.views and V > args.views:
        step = V / args.views
        use = [int(k * step) for k in range(args.views)]

    seen = np.zeros(len(tris), dtype=np.int64)
    for i in use:
        W, H = sizes[i]
        if W == 0:
            continue
        w2c = np.linalg.inv(c2ws[i])
        pc = fc @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        ok = z > 1e-6
        u = (pc @ ks[i][0, :3]) / np.maximum(z, 1e-9)
        v = (pc @ ks[i][1, :3]) / np.maximum(z, 1e-9)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        cam_c = c2ws[i][:3, 3]
        vd = cam_c - fc
        vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-9)
        ok &= np.einsum("ij,ij->i", fn, vd) > 0.1
        seen += ok.astype(np.int64)

    keep_f = seen >= max(1, args.min_views)
    print(f"\n  被 >= {max(1, args.min_views)} 个视角看到的面："
          f"{int(keep_f.sum())}/{len(tris)}（{keep_f.mean()*100:.1f}%）")
    print(f"  将删除：{int((~keep_f).sum())} 面（{(~keep_f).mean()*100:.1f}%）")

    if not args.apply:
        print("\n（预览模式。确认后加 --apply）")
        return 0

    # 重建网格（只保留被看到的面）
    used_v = np.zeros(len(verts), dtype=bool)
    used_v[tris[keep_f].ravel()] = True
    new_index = -np.ones(len(verts), dtype=np.int64)
    new_index[used_v] = np.arange(int(used_v.sum()))
    out_mesh = o3d.geometry.TriangleMesh()
    out_mesh.vertices = o3d.utility.Vector3dVector(verts[used_v])
    out_mesh.triangles = o3d.utility.Vector3iVector(new_index[tris[keep_f]])
    if mesh.has_vertex_colors():
        out_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_colors)[used_v])
    out_mesh.compute_vertex_normals()
    out_mesh, removed = to_largest_component(out_mesh)
    out = Path(args.out) if args.out else (job / "05_export" / "model_mesh_visible.ply")
    o3d.io.write_triangle_mesh(str(out), out_mesh, write_vertex_colors=True)
    print(f"\n已写入 {out}（{len(out_mesh.vertices)} 顶点 / "
          f"{len(out_mesh.triangles)} 面，剔除小连通块 {removed} 面）")
    (out.parent / "drop_unseen_report.json").write_text(json.dumps({
        "src": str(ply), "views": len(use), "min_views": max(1, args.min_views),
        "faces_before": int(len(tris)), "faces_kept": int(keep_f.sum()),
        "faces_after_component": int(len(out_mesh.triangles)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
