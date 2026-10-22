"""亮高斯的空间位置：是"背景壳"还是"白色牌坊本体"？

## 为什么要分清

用户说"像曝光一样"。我先测出 **13.5% 的亮高斯占了 62.8% 视觉权重**，
于是推断"背景被训练进去了"。但这个推断**没被验证**：

* 牌坊本体是**白色/浅灰石材**（`preview_iso.png` 里整体偏亮）；
* 若亮高斯其实就在物体上，那"过曝"就只是**物体本来就白**，
  而不是背景污染 —— 两者对应的修法完全不同（前者要改损失，
  后者要改查看器的色调/背景）。

判据：**到物体中心的半径分布**。转台拍摄时背景在图像里静止，
会被重建为一层**环绕物体的大半径壳**；而物体表面的高斯半径应显著更小。

用法: python scripts/diag_bright_shell.py --job img3-final
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
    ap.add_argument("--bins", type=int, default=8)
    args = ap.parse_args()

    from gsflow.gsdata.shio import read_gaussians_full

    print("=== 亮高斯是「背景壳」还是「物体本体」 ===")
    for job in (args.job or ["img3-final"]):
        jd = ROOT / "workspace" / job
        plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
        print(f"\n[{job}]")
        if not plys:
            print("  没有训练产物")
            continue
        g = read_gaussians_full(plys[-1])
        xyz = np.asarray(g["xyz"], dtype=np.float64)
        rgb = np.asarray(g["rgb"], dtype=np.float64)
        op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
        sc = np.asarray(g["scales"], dtype=np.float64)
        lum = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]) / 255.0
        w = op * (sc.max(1) ** 2)
        w = w / w.sum()

        center = xyz.mean(axis=0)
        r = np.linalg.norm(xyz - center, axis=1)
        bright = lum > 0.8
        dark = lum < 0.25

        print(f"  到质心半径 中位: 亮 {np.median(r[bright]):.2f} / "
              f"暗 {np.median(r[dark]):.2f} / 全部 {np.median(r):.2f}")
        print(f"  半径 p90      : 亮 {np.percentile(r[bright], 90):.2f} / "
              f"暗 {np.percentile(r[dark], 90):.2f}")

        # 分半径区间看"亮 vs 暗"的视觉权重比例
        edges = np.linspace(0, float(np.percentile(r, 99.5)), args.bins + 1)
        print(f"  {'半径区间':<18}{'亮权重':>9}{'暗权重':>9}{'亮/暗':>8}  判读")
        for i in range(args.bins):
            m = (r >= edges[i]) & (r < edges[i + 1])
            if m.sum() < 50:
                continue
            wb = w[m & bright].sum()
            wd = w[m & dark].sum()
            ratio = wb / max(wd, 1e-9)
            note = ""
            if i >= args.bins - 2 and wb > wd * 2:
                note = "外层几乎全是亮高斯 ⇒ **背景壳**"
            print(f"  [{edges[i]:5.2f},{edges[i+1]:5.2f})   "
                  f"{wb*100:7.1f}% {wd*100:8.1f}% {ratio:7.2f}  {note}")

        # 决定性判据：外层（半径后 2 个区间）亮权重占比 vs 内层
        outer = r >= edges[-3]
        inner = r < edges[2]
        if inner.sum() and outer.sum():
            wb_in = w[inner & bright].sum() / max(w[inner].sum(), 1e-9)
            wb_out = w[outer & bright].sum() / max(w[outer].sum(), 1e-9)
            print(f"\n  内层亮权重占比 {wb_in*100:.1f}%   外层亮权重占比 {wb_out*100:.1f}%")
            if wb_out > wb_in * 1.5 and wb_out > 0.6:
                print("  => 外层显著更亮 => 亮高斯主要是**背景壳**（应抑制背景拟合）")
            else:
                print("  => 内外亮度接近 => 亮高斯主要是**物体白石材**"
                      "（观感上的过曝更可能是查看器色调/背景设置）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
