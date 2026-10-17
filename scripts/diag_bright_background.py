"""检查模型里是否有一大片**亮背景**（会让查看器看起来"曝光过度"）。

## 背景

用户报告："现在我看到的东西就像曝光一样"。

一个具体可查的假设：IMG3 是**转台 + 保留背景**（`mask.mode=keep`）拍摄的，
而 `preview_iso.png` 里背景是**浅灰白**。如果训练把大面积亮背景也拟合成高斯，
那在参考查看器里这些亮高斯会铺满屏幕 ⇒ 观感就是"过曝/发白"。

本脚本量化：

  * 颜色亮度分布（中位 / p95）；
  * 亮度 > 0.8 与 < 0.25 的高斯各占多少**数量**和多少**视觉权重**；
  * 亮背景的空间范围与中心 —— 判断它是不是"在物体后面的一层壳"。

用法: python scripts/diag_bright_background.py --job img3-final
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
    ap.add_argument("--job", action="append", default=[])
    args = ap.parse_args()
    jobs = args.job or ["img3-final", "img3-crop"]

    from gsflow.gsdata.shio import read_gaussians_full

    print("=== 亮背景检查（解释“像曝光一样”的观感）===")
    for job in jobs:
        jd = ROOT / "workspace" / job
        plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
        print(f"\n[{job}]")
        if not plys:
            print("  没有训练产物")
            continue
        g = read_gaussians_full(plys[-1])
        rgb = np.asarray(g["rgb"], dtype=np.float64)
        op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
        sc = np.asarray(g["scales"], dtype=np.float64)
        xyz = np.asarray(g["xyz"], dtype=np.float64)
        lum = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]) / 255.0
        w = op * (sc.max(1) ** 2)
        w = w / w.sum() if w.sum() > 0 else w

        print(f"  颜色亮度 中位 {np.median(lum):.3f}  p95 "
              f"{np.percentile(lum, 95):.3f}  max {lum.max():.3f}")
        for label, m in (("亮度>0.8（亮）", lum > 0.8),
                         ("亮度<0.25（暗）", lum < 0.25)):
            print(f"  {label:<16}: {int(m.sum()):>7,} 个（{m.mean()*100:5.1f}%）"
                  f"  占视觉权重 {w[m].sum()*100:5.1f}%")
        bright = lum > 0.8
        if bright.sum() > 10:
            b = xyz[bright]
            print(f"  亮背景 bbox     : {np.round(b.max(0) - b.min(0), 2).tolist()}"
                  f"  中心 {np.round(b.mean(0), 2).tolist()}")
        print(f"  全部   bbox     : {np.round(xyz.max(0) - xyz.min(0), 2).tolist()}"
              f"  中心 {np.round(xyz.mean(0), 2).tolist()}")
        share = w[bright].sum() * 100 if bright.sum() else 0.0
        if share > 25:
            print(f"  ⇒ **亮高斯占视觉权重 {share:.0f}%，很可能就是“过曝”观感的来源**")
        else:
            print(f"  ⇒ 亮高斯只占视觉权重 {share:.0f}%，不是主因；过曝更可能来自渲染端")
    return 0


if __name__ == "__main__":
    sys.exit(main())
