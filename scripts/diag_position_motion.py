"""测量"位置到底动没动"：跨检查点配对高斯，统计位移 vs 尺度变化。

## 用户的问题（2026-10-24）

> 会不会是位置几乎不动 ⇒ 只能靠拉长/放大高斯去凑像素导致细丝化？

这个假设可以**直接测**：把 5000 步与 30000 步的高斯做最近邻配对，
比较"位置移动量"与"尺度增长量"。若位置几乎不动而尺度显著变大，
假设成立；若位置也在动，则假设不成立（此时"针"另有来源）。

输出：
* 配对比例、位移中位数/90 分位（米，及占场景尺度的百分比）；
* 尺度（最长轴）中位数变化；
* 位移与尺度增长的比值。

用法: python scripts/diag_position_motion.py --job img3-final --a 5000 --b 30000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(ply: Path):
    from gsflow.gsdata.shio import read_gaussians_full

    g = read_gaussians_full(ply)
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    sc = np.asarray(g["scales"], dtype=np.float64)
    return xyz, sc.max(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img3-final")
    ap.add_argument("--a", type=int, default=5000)
    ap.add_argument("--b", type=int, default=30000)
    ap.add_argument("--radius", type=float, default=0.05,
                    help="配对半径（米）。超过则视为不同高斯")
    args = ap.parse_args()

    from scipy.spatial import cKDTree

    jd = ROOT / "workspace" / args.job / "04_train" / "point_cloud"
    pa = jd / f"iteration_{args.a}" / "point_cloud.ply"
    pb = jd / f"iteration_{args.b}" / "point_cloud.ply"
    if not (pa.exists() and pb.exists()):
        print(f"缺少检查点：{pa.name} / {pb.name}")
        return 1
    xa, sa = _load(pa)
    xb, sb = _load(pb)
    print(f"=== 位置运动测量：{args.job}  {args.a} -> {args.b} ===")
    print(f"  高斯数 {len(xa):,} -> {len(xb):,}")
    print(f"  尺度(最长轴) 中位 {np.median(sa):.4f} -> {np.median(sb):.4f} m"
          f"  (×{np.median(sb)/max(np.median(sa),1e-9):.2f})")

    tree = cKDTree(xb)
    dist, idx = tree.query(xa, k=1, distance_upper_bound=args.radius)
    ok = np.isfinite(dist)
    frac = ok.mean()
    if not ok.any():
        print("  没有配对成功的高斯（半径太小？）")
        return 1
    d = dist[ok]
    scene = float(np.linalg.norm(np.percentile(xb, 95, axis=0) -
                                 np.percentile(xb, 5, axis=0)))
    print(f"  配对成功 {ok.sum():,}/{len(xa):,}（{frac*100:.1f}%），配对半径 {args.radius} m")
    print(f"  位移中位 {np.median(d):.4f} m   90 分位 {np.percentile(d, 90):.4f} m"
          f"   最大 {d.max():.3f} m")
    print(f"  场景对角线约 {scene:.2f} m ⇒ 位移中位占 **{np.median(d)/max(scene,1e-9)*100:.2f}%**")
    # 未配对的高斯（新增/消失）比例
    moved = d > 0.005
    print(f"  位移 >5mm 的占比 {moved.mean()*100:.1f}%")
    grow = np.median(sb[idx[ok]]) / np.maximum(np.median(sa[ok]), 1e-9)
    print(f"  配对高斯的尺度增长 中位 {np.median(sb[idx[ok]]):.4f} m（比 {args.a} 步 ×{grow:.2f}）")
    print()
    if np.median(d) / max(scene, 1e-9) < 0.005 and grow > 1.2:
        print("  判读：**位置基本不动、尺度明显增长 ⇒ 用户的假设成立**：")
        print("        优化器确实靠【放大高斯】而不是【移动高斯】去拟合像素。")
    elif np.median(d) / max(scene, 1e-9) < 0.005:
        print("  判读：位置基本不动，但尺度也没有明显增长 ⇒ 更像只调颜色/不透明度。")
    else:
        print("  判读：位置在明显移动 ⇒ 【位置不动导致撑大】的解释**不成立**，")
        print("        细丝另有来源（例如掩码偏大导致的轮廓拟合，见 check_mask_tightness）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
