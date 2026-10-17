"""在**现有**网格产物上评估 `sanitize_mesh` 的效果（不重跑导出、不用 GPU）。

## 为什么这样做

`img3-final` 的 `model_mesh.ply` 里有 1 个退化三角形（面积≈0）、0 个重复顶点，
最大长宽比 577.7。我想在导出流程里加一步"安全清理"，但**必须先量出它到底改了什么**，
而不是加了就宣称更好：

  * 删掉多少面、多少顶点；
  * 表面积/包围盒有没有变化（应该几乎不变 —— 退化面本来不贡献面积）；
  * 朝外比例有没有被破坏（清理会重排索引，法线朝向可能被弄乱）。

只有这些都确认无副作用，才把 `sanitize_mesh` 接进 `_finish_mesh`。

用法: python scripts/eval_mesh_sanitize.py --job img3-final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def measure(mesh) -> dict:
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    center = v.mean(axis=0)
    n = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    outward = (((a + b + c) / 3.0 - center) * n).sum(axis=1)
    return {
        "verts": int(len(v)), "tris": int(len(f)),
        "zero_area": int((area <= 1e-12).sum()),
        "sliver": int(((area < max(float(np.median(area)) * 1e-3, 1e-15))
                       & (area > 1e-12)).sum()),
        "surface_area": float(area.sum()),
        "outward_ratio": float((outward > 0).mean()),
        "bbox": (v.max(0) - v.min(0)).round(6).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img3-final")
    args = ap.parse_args()

    import open3d as o3d

    from gsflow.gsdata.mesh_export import sanitize_mesh

    p = ROOT / "workspace" / args.job / "05_export" / "model_mesh.ply"
    if not p.exists():
        print(f"找不到 {p}")
        return 1
    print(f"=== 清理前后对比：{p.name}（{args.job}）===")
    before_mesh = o3d.io.read_triangle_mesh(str(p))
    before_mesh.compute_triangle_normals()
    b = measure(before_mesh)
    print(f"  清理前: 顶点 {b['verts']:,}  面 {b['tris']:,}  "
          f"零面积 {b['zero_area']} 细长 {b['sliver']}  表面积 {b['surface_area']:.6f}  "
          f"朝外 {b['outward_ratio']*100:.2f}%")

    mesh2 = o3d.io.read_triangle_mesh(str(p))
    mesh2.compute_triangle_normals()
    mesh2, stats = sanitize_mesh(mesh2, log=print)
    mesh2.compute_triangle_normals()
    a = measure(mesh2)
    print(f"  清理后: 顶点 {a['verts']:,}  面 {a['tris']:,}  "
          f"零面积 {a['zero_area']} 细长 {a['sliver']}  表面积 {a['surface_area']:.6f}  "
          f"朝外 {a['outward_ratio']*100:.2f}%")

    print("\n=== 判定 ===")
    checks = [
        ("删掉的面很少（<1%）", (stats["tris_before"] - stats["tris_after"])
         / max(stats["tris_before"], 1) < 0.01),
        # ⚠️ 不要断言"退化面归零"：`sanitize_mesh` 删的是 open3d 定义的
        # **零面积面**，而这里 `measure()` 统计的是"面积<中位数 0.1%"的**细长面**
        # —— 两者不是一回事（细长面非零、删了会开洞）。我第一版就是这么误判的。
        ("没有零面积面（真退化）", a.get("zero_area", 0) == 0),
        ("细长面数量未被改变（不该动它）", a.get("sliver", 0) == b.get("sliver", 0)),
        ("表面积几乎不变（<0.5%）",
         abs(a["surface_area"] - b["surface_area"]) / max(b["surface_area"], 1e-12) < 0.005),
        ("包围盒不变", a["bbox"] == b["bbox"]),
        ("朝外比例未被破坏（仍 100%）", a["outward_ratio"] > 0.999),
    ]
    bad = 0
    for label, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")
        if not ok:
            bad += 1
    print(f"\n结论: {'清理安全，可接入导出流程' if bad == 0 else f'**{bad} 项不通过，不要接**'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
