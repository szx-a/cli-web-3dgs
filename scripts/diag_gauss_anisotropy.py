"""诊断"细丝化 / 背面混乱 / 网格乱成一团"的几何根因。

## 用户报告的现象

> 我这里渲染器里有明显的**细丝化**，模型**背面混乱**，用 webgl 看的网格导出几乎**乱作一团**。
> 我们的问题是：渲染器一样，为什么看到的会不一样？哪一步训练流程出了问题？

## 为什么"预览好、自由视角差"

预览图（`preview_*.png`）用的是**训练/留出相机**附近的视角。3DGS 的每个高斯是一个
**各向异性椭球** —— 如果某个高斯被优化成"一根针"（一个轴极长、另两轴极短），
它在**训练视角**下投影出来仍然像一片正常的面，但一旦转到别的角度，就变成一根细丝。
所以：

    训练视角像面 → 自由视角是丝   ← 这就是"细丝化"的成因

## 量化什么

1. **各向异性**：`max(scale)/min(scale)`；针状高斯占比，以及它们承担的视觉权重；
2. **针状高斯的位置**：是不是集中在某个区域（背面/外壳）；
3. **网格碎片度**：连通分量数量、最大分量占比 —— "乱作一团"= 几千个小块；
4. **导出用的是哪个检查点**、不透明度清理删了多少。

用法: python scripts/diag_gauss_anisotropy.py --job img3-final
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
    ap.add_argument("--job", action="append", default=[])
    args = ap.parse_args()
    jobs = args.job or ["img3-final"]

    from gsflow.gsdata.shio import read_gaussians_full

    print("=== 高斯各向异性 / 针状结构 / 网格碎片度 ===")
    for job in jobs:
        jd = ROOT / "workspace" / job
        plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
        print(f"\n[{job}]")
        if not plys:
            print("  没有训练产物")
            continue
        # 导出用的是 metrics 里的 best_ply；这里两个都看
        mpath = jd / "04_train" / "metrics.json"
        best = None
        if mpath.exists():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            best = m.get("best_ply")
            print(f"  metrics: best_ply={Path(best).parent.name if best else None}"
                  f"  best_psnr={m.get('best_psnr')}  final_psnr={m.get('psnr')}")
        use = Path(best) if best and Path(best).exists() else plys[-1]
        print(f"  分析模型: {use.parent.name}/point_cloud.ply")
        g = read_gaussians_full(use)
        sc = np.asarray(g["scales"], dtype=np.float64)
        op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
        xyz = np.asarray(g["xyz"], dtype=np.float64)

        smin = sc.min(axis=1)
        smax = sc.max(axis=1)
        aniso = smax / np.maximum(smin, 1e-12)
        w = op * (smax ** 2)
        w = w / w.sum() if w.sum() > 0 else w

        print(f"  高斯总数        : {len(xyz):,}")
        print(f"  各向异性 中位/p95/p99: {np.median(aniso):.1f} / "
              f"{np.percentile(aniso, 95):.1f} / {np.percentile(aniso, 99):.1f}")
        for thr in (10, 50, 100, 500):
            sel = aniso > thr
            print(f"  aniso>{thr:<3}        : {int(sel.sum()):>7,} 个（{sel.mean()*100:5.2f}%）"
                  f"  占视觉权重 {w[sel].sum()*100:5.1f}%")
        # "针"= 各向异性极高且最长轴不小
        needle = (aniso > 100) & (smax > np.percentile(smax, 50))
        print(f"  针状(aniso>100 且 smax>中位): {int(needle.sum()):,} 个，"
              f"占视觉权重 {w[needle].sum()*100:.1f}%")
        if needle.sum() > 10:
            c_all = xyz.mean(0)
            c_needle = xyz[needle].mean(0)
            print(f"    针状中心 {np.round(c_needle, 2).tolist()}  vs 全模型中心 "
                  f"{np.round(c_all, 2).tolist()}")
            r_all = np.linalg.norm(xyz - c_all, axis=1)
            r_needle = np.linalg.norm(xyz[needle] - c_all, axis=1)
            print(f"    到中心半径 中位: 针状 {np.median(r_needle):.2f} vs 全体 "
                  f"{np.median(r_all):.2f}（p95: {np.percentile(r_needle,95):.2f} vs "
                  f"{np.percentile(r_all,95):.2f}）")

        # ---- 网格碎片度 ----
        mesh = jd / "05_export" / "model_mesh.ply"
        if mesh.exists():
            try:
                import open3d as o3d

                mm = o3d.io.read_triangle_mesh(str(mesh))
                tris = np.asarray(mm.triangles)
                if len(tris):
                    import collections
                    parent = np.arange(len(mm.vertices), dtype=np.int64)

                    def find(a: int) -> int:
                        while parent[a] != a:
                            parent[a] = parent[parent[a]]
                            a = parent[a]
                        return a

                    for t in tris:
                        ra, rb, rc = find(int(t[0])), find(int(t[1])), find(int(t[2]))
                        for other in (rb, rc):
                            if other != ra:
                                parent[other] = ra
                    roots = np.array([find(i) for i in range(len(parent))])
                    used = np.unique(roots[tris.reshape(-1)])
                    counts = collections.Counter(roots[tris[:, 0]].tolist())
                    sizes = sorted(counts.values(), reverse=True)
                    total = sum(sizes)
                    print(f"  网格             : {len(tris):,} 面，"
                          f"**连通分量 {len(sizes):,} 个**，最大占 {sizes[0]/total*100:.1f}%"
                          f"（前 5 大合计 {sum(sizes[:5])/total*100:.1f}%）")
                    small = sum(1 for s in sizes if s < 50)
                    print(f"    小于 50 面的碎块: {small:,} 个（{small/len(sizes)*100:.1f}% 的分量）")
            except Exception as exc:  # noqa: BLE001
                print(f"  网格分析失败: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
