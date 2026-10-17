"""检查 `.splat` 的取景半径该用"包围盒"还是"稳健分位数"。

## 为什么必须查

`viewer.js` 的 `reset()` 用 `dist = obj.radius * 2.8` 取景。
点云路径的 `radius` 来自包围盒对角线的一半 —— 但**高斯点云常有远处离群点**
（泊松/训练残留、半透明浮点），一个离群点就能把包围盒撑大好几倍，
于是相机被迫拉到很远，物体在画面里变成一小块 ⇒ **用户会觉得"比预览图差很多"**。

实测 img3-crop：`.splat` 的包围盒半径 **4.76**，而网格包围盒只有约 2 —— 差了 2.4 倍。

本脚本量化"包围盒半径"与"稳健半径（95 分位距中心距离）"的差距，
用来决定查看器该用哪个。

用法: python scripts/diag_splat_extent.py --job img3-crop
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
    ap.add_argument("--job", default="img3-crop")
    args = ap.parse_args()

    from gsflow.gsdata.shio import read_gaussians_full

    job = ROOT / "workspace" / args.job
    plys = sorted((job / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
    if not plys:
        print("找不到点云")
        return 1
    g = read_gaussians_full(plys[-1])
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    print(f"高斯 {len(xyz)}")

    mn, mx = xyz.min(0), xyz.max(0)
    ext = mx - mn
    cen_bbox = (mn + mx) / 2
    rad_bbox = float(np.linalg.norm(ext) / 2)
    print(f"\n  包围盒 extent = {np.round(ext, 3).tolist()}")
    print(f"  包围盒中心    = {np.round(cen_bbox, 3).tolist()}")
    print(f"  包围盒半径    = {rad_bbox:.3f}")

    # 稳健：中位数中心 + 分位数距离
    cen_med = np.median(xyz, axis=0)
    d = np.linalg.norm(xyz - cen_med, axis=1)
    print(f"\n  中位数中心    = {np.round(cen_med, 3).tolist()}")
    for q in (0.50, 0.90, 0.95, 0.99, 1.00):
        print(f"  距离 {q*100:5.1f}% 分位 = {np.quantile(d, q):.3f}")

    rad_q95 = float(np.quantile(d, 0.95))
    rad_q99 = float(np.quantile(d, 0.99))
    print(f"\n  包围盒半径 / 95分位 = {rad_bbox/max(rad_q95,1e-9):.2f}×")
    print(f"  包围盒半径 / 99分位 = {rad_bbox/max(rad_q99,1e-9):.2f}×")

    # 离群点占比：距中位数中心 > 95 分位 * 2 的点
    out = int((d > rad_q95 * 2).sum())
    print(f"  明显离群（> 2×95分位） = {out} 个（{out/len(xyz)*100:.2f}%）")

    print("\n=== 建议 ===")
    if rad_bbox > rad_q95 * 1.5:
        print(f"  ⇒ 包围盒把半径撑大了 {rad_bbox/rad_q95:.1f} 倍（有离群点），")
        print(f"     查看器取景应改用**稳健半径**（95 分位 {rad_q95:.3f}），"
              f"否则相机会拉太远、模型显得很小")
    else:
        print("  ⇒ 包围盒与稳健半径接近，用哪个都行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
