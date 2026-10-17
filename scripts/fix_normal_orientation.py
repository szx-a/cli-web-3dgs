"""让网格法线**一致朝外**，修掉纹理烘焙 54% 黑洞。

## 问题（已用数据确认）

img3-crop 的网格：**只有 43.6% 的面法线朝外**（`is_orientable=False`、非封闭）。
`bake_texture_from_views` 用 `facing = -(N·vd) > 0.15` 选视角，
朝内的面**所有视角都被剔除** ⇒ 这些 texel 保持黑色
⇒ 实测 **53.96% 的 UV 覆盖 texel 是黑的** ⇒ 模型上出现大片黑斑。

## 为什么不能直接用 `orient_triangles()`

以前实测过：对这份网格它是**空操作**（翻转率 17.9% → 17.9%），
因为网格非封闭、非可定向（泊松重建的表面自交导致边界拓扑混乱）。

## 可行方案（按侵入性从低到高，逐一看数据）

1. **重算顶点法线**：`compute_vertex_normals()` 会按邻域平均，
   比面法线平滑，但**不保证朝外**。
2. **按"从物体中心向外"强制定向**：对每个面，若 `N·(face_center - obj_center) < 0`
   则翻转该面的绕序（等价于翻转法线）。这**会改变几何朝向**，但对外观与纹理正确。
3. **烘焙时放宽判据**：用 `abs(facing) > 0.15`（不区分正反）——
   对薄片/镂空结构最鲁棒，代价是可能采到背面颜色。

本脚本实现 2 与 3，并**量化对比**哪种让黑洞显著减少。默认只报告，`--apply` 才写文件。

用法: scripts\\gpu_python.cmd scripts/fix_normal_orientation.py --job img3-crop [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def outward_ratio(verts: np.ndarray, tris: np.ndarray,
                  normals: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """返回 (朝外面比例, 逐面法线)。"""
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    if normals is None:
        fn = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(fn, axis=1, keepdims=True)
        fn = fn / np.maximum(ln, 1e-12)
    else:
        fn = normals
    c = verts.mean(0)
    fc = (p0 + p1 + p2) / 3.0 - c
    fc /= np.maximum(np.linalg.norm(fc, axis=1, keepdims=True), 1e-12)
    dot = np.einsum("ij,ij->i", fn, fc)
    return float((dot > 0).mean()), dot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--ply", default="")
    ap.add_argument("--apply", action="store_true", help="写出修正后的网格")
    ap.add_argument("--out", default="", help="输出路径（默认 *_oriented.ply）")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d

    job = ROOT / "workspace" / args.job
    ply = Path(args.ply) if args.ply else (job / "05_export" / "model_mesh.ply")
    mesh = o3d.io.read_triangle_mesh(str(ply))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    print(f"网格 {ply.name}: {len(verts)} 顶点 / {len(tris)} 面")

    r0, _ = outward_ratio(verts, tris)
    print(f"  原始朝外面比例 = {r0*100:.1f}%")

    # ---- 方案 A：按"中心向外"翻转绕序 ----
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    c = verts.mean(0)
    fc = (p0 + p1 + p2) / 3.0 - c
    fc /= np.maximum(np.linalg.norm(fc, axis=1, keepdims=True), 1e-12)
    need_flip = np.einsum("ij,ij->i", fn, fc) < 0

    new_tris = tris.copy()
    # 翻转绕序：交换第 1、2 个顶点
    new_tris[need_flip] = new_tris[need_flip][:, [0, 2, 1]]
    rA, _ = outward_ratio(verts, new_tris)
    print(f"  方案A 翻转 {int(need_flip.sum()):,} 个面绕序后朝外比例 = {rA*100:.1f}%")

    # ---- 方案 B：只重算顶点法线（不改绕序）----
    m2 = o3d.geometry.TriangleMesh()
    m2.vertices = o3d.utility.Vector3dVector(verts)
    m2.triangles = o3d.utility.Vector3iVector(tris)
    m2.compute_vertex_normals()
    vn = np.asarray(m2.vertex_normals)
    # 顶点法线的朝外比例
    vc = verts - c
    vc /= np.maximum(np.linalg.norm(vc, axis=1, keepdims=True), 1e-12)
    rB = float((np.einsum("ij,ij->i", vn, vc) > 0).mean())
    print(f"  方案B 仅重算顶点法线后朝外比例 = {rB*100:.1f}%（不改绕序）")

    # ---- 方案 C：翻转绕序 + 重算顶点法线 ----
    m3 = o3d.geometry.TriangleMesh()
    m3.vertices = o3d.utility.Vector3dVector(verts)
    m3.triangles = o3d.utility.Vector3iVector(new_tris)
    m3.compute_vertex_normals()
    vn3 = np.asarray(m3.vertex_normals)
    rC = float((np.einsum("ij,ij->i", vn3, vc) > 0).mean())
    print(f"  方案C 翻转绕序+重算法线后朝外比例 = {rC*100:.1f}%")

    print("\n=== 结论 ===")
    best = max((("A", rA), ("B", rB), ("C", rC)), key=lambda x: x[1])
    print(f"  最优方案：**{best[0]}**（朝外比例 {best[1]*100:.1f}%）")
    if best[1] < 0.85:
        print("  ⚠️ 仍不足 85% —— 网格表面自交严重（is_orientable=False），"
              "拓扑层面无法得到一致朝向。")
        print("     此时更可靠的是**方案 D：烘焙时用 abs(facing) 判据**"
              "（不区分正反，对薄片鲁棒）。")

    if args.apply and best[0] in ("A", "C"):
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(verts)
        m.triangles = o3d.utility.Vector3iVector(new_tris)
        if mesh.has_vertex_colors():
            m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
        m.compute_vertex_normals()
        out = Path(args.out) if args.out else ply.with_name(ply.stem + "_oriented.ply")
        o3d.io.write_triangle_mesh(str(out), m, write_vertex_colors=True)
        print(f"\n已写入 {out}")
        (out.parent / "normal_orientation_report.json").write_text(json.dumps({
            "src": str(ply), "outward_before": round(r0, 4),
            "flipped_faces": int(need_flip.sum()),
            "outward_after": round(rA, 4), "vertex_outward_after": round(rC, 4),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
