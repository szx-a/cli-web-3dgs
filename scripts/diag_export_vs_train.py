"""对比"预览渲染用的模型"与"查看器加载的模型"是否同一个。

## 为什么要查（用户的直问）

> 我们的问题是：渲染器一样，为什么看到的会不一样？哪一步的训练流程出了问题？

一个此前没验过的可能：**预览图和查看器根本不是同一个文件**。

* `preview_*.png` 由导出的 `_render_previews_splat()` 生成，它读的是 `04_train` 的 PLY；
* 查看器（`splat` 模式）加载的是 `05_export/model_gaussians.ply`（`ply_full` 格式复制）。

如果导出阶段做了任何**剪枝 / 改写 / 排序 / 量化**（例如不透明度清理、`ply_full` 不是纯拷贝），
那么"预览好看、查看器难看"就有了无需怀疑渲染器的解释。

## 判据

数量相同只能说明"没删"，还不够 —— 还要比
**逐高斯的位置/尺度/不透明度是否一致**（抽样比对各字段），以及**各向异性分布**是否相同。

用法: python scripts/diag_export_vs_train.py --job img3-final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def stats(g: dict) -> dict:
    sc = np.asarray(g["scales"], dtype=np.float64)
    op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    smax = sc.max(1)
    aniso = smax / np.maximum(sc.min(1), 1e-12)
    w = op * smax ** 2
    w = w / w.sum() if w.sum() > 0 else w
    return {
        "n": len(xyz),
        "op_median": float(np.median(op)),
        "op_mean": float(op.mean()),
        "scale_median": float(np.median(smax)),
        "scale_max": float(smax.max()),
        "aniso_median": float(np.median(aniso)),
        "aniso_p99": float(np.percentile(aniso, 99)),
        "needle_weight": float(w[aniso > 100].sum()),
        "bbox": (xyz.max(0) - xyz.min(0)).round(3).tolist(),
        "center": xyz.mean(0).round(3).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img3-final")
    args = ap.parse_args()

    from gsflow.gsdata.shio import read_gaussians_full

    jd = ROOT / "workspace" / args.job
    train_plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
    exp_ply = jd / "05_export" / "model_gaussians.ply"
    if not train_plys or not exp_ply.exists():
        print("缺少训练或导出 PLY")
        return 1

    # 导出阶段用的是 metrics.best_ply（可能是更早的检查点），先看它指谁
    import json

    mp = jd / "04_train" / "metrics.json"
    best = None
    if mp.exists():
        best = json.loads(mp.read_text(encoding="utf-8")).get("best_ply")
    use = Path(best) if best and Path(best).exists() else train_plys[-1]

    print(f"=== {args.job}：预览用的模型 vs 查看器加载的模型 ===")
    print(f"  训练 PLY  : {use.parent.name}/point_cloud.ply")
    print(f"  导出 PLY  : {exp_ply.name}（{exp_ply.stat().st_size/1e6:.1f} MB）")

    a = read_gaussians_full(use)
    b = read_gaussians_full(exp_ply)
    sa, sb = stats(a), stats(b)
    keys = ["n", "op_median", "op_mean", "scale_median", "scale_max",
            "aniso_median", "aniso_p99", "needle_weight", "bbox", "center"]
    print(f"\n  {'指标':<14}{'训练(预览用)':>22}{'导出(查看器)':>22}")
    for k in keys:
        va, vb = sa[k], sb[k]
        if isinstance(va, float):
            print(f"  {k:<14}{va:>22.4f}{vb:>22.4f}")
        else:
            print(f"  {k:<14}{str(va):>22}{str(vb):>22}")

    same_count = sa["n"] == sb["n"]
    print(f"\n  数量一致: {same_count}（{sa['n']:,} vs {sb['n']:,}）")
    if same_count and sa["n"] > 0:
        # 逐字段比对（按文件顺序；两份都是同一来源，顺序应一致）
        for field in ("xyz", "scales", "opacity", "quats"):
            A = np.asarray(a[field], dtype=np.float64)
            B = np.asarray(b[field], dtype=np.float64)
            if A.shape != B.shape:
                print(f"  {field:<8} 形状不同 {A.shape} vs {B.shape} ⇒ **不是同一个模型**")
                continue
            d = np.abs(A - B).max()
            print(f"  {field:<8} 最大逐元素差 {d:.6g}"
                  + ("  ⇒ 一致" if d == 0 else "  ⇒ **被改写过**"))
    # 排序是否一致（read_gaussians_full 按文件顺序；若导出重排过，字段差会很大）
    return 0


if __name__ == "__main__":
    sys.exit(main())
