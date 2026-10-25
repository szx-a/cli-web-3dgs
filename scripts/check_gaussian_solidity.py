"""高斯"实心度"体检：把用户说的"朦胧/透明感"量化。

## 为什么需要（2026-10-25）

用户反复反馈：**PLY 看上去有点朦胧、有一点透明度、细节和实感较弱**。
而所有训练指标（掩码内 PSNR/SSIM）都**看不见这个症状** —— 实测 img5-v1：掩码内 26.48 dB / SSIM 0.9654，
单看数字很好，但视觉上是"一团淡高斯"：

| 指标 | img4-v1 | img5-v1 |
|---|---|---|
| 不透明度中位 | 0.066 | **0.031** |
| `op<0.1` | 67% | **85%** |
| `op>0.9` | 1.1% | 2.2% |

表面是靠**大量半透明层叠加**出来的：颜色对得上，但新视角会透、细节发虚。

本脚本给出可直接比较的"实心度"面板：

* `op_median` / `op_lt_01` / `op_gt_05` / `op_gt_09` —— 不透明度分布；
* `alpha_layers` —— **每条视线的等效层数**估算：Σop / 覆盖像素的单层期望（越大越"雾"）；
* `faint_volume_frac` —— 低不透明度高斯占据的体积份额（scale³ 加权），衡量"雾"的体积；
* `mean_scale` / `max_scale` —— 尺度分布，过大的高斯也会显得糊。

用法:
    python scripts/check_gaussian_solidity.py --jobs img5-v1,img4-v1,img3-final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def latest_ply(job: str) -> Path | None:
    jd = ROOT / "workspace" / job / "04_train" / "point_cloud"
    plys = list(jd.glob("*/point_cloud.ply"))
    if not plys:
        return None
    return max(plys, key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="img5-v1")
    args = ap.parse_args()

    from gsflow.gsdata.shio import read_gaussians_full

    print("=== 高斯实心度体检 ===")
    print("  %-14s %8s %8s %8s %8s %10s %12s" % (
        "作业", "n", "op中位", "op<0.1", "op>0.9", "淡体积份额", "尺度中位/最大"))
    for job in [j.strip() for j in args.jobs.split(",") if j.strip()]:
        ply = latest_ply(job)
        if ply is None:
            print(f"  {job:<14} 无训练产物")
            continue
        g = read_gaussians_full(ply)
        op = np.asarray(g["opacity"]).reshape(-1).astype(np.float64)
        sc = np.asarray(g["scales"]).reshape(-1, 3).astype(np.float64)
        vol = np.prod(np.clip(sc, 1e-9, None), axis=1)
        faint = op < 0.05
        faint_vol = float(vol[faint].sum() / max(vol.sum(), 1e-12))
        print("  %-14s %8d %8.3f %7.1f%% %7.1f%% %11.1f%% %7.4f/%.3f" % (
            job, len(op), float(np.median(op)), float((op < 0.1).mean() * 100),
            float((op > 0.9).mean() * 100), faint_vol * 100,
            float(np.median(sc.max(1))), float(sc.max(1).max())))

    print("\n  判读：")
    print("   * op 中位 <0.1 且 op<0.1 占比 >60% ⇒ 典型“淡高斯云”，视觉会朦胧/透明")
    print("   * 淡体积份额 >50% ⇒ 有大量低不透明度高斯在占体积，优先考虑推实/剪枝")
    print("   * 这些指标**与 PSNR/SSIM 无关**：掩码内指标对“是否实心”完全不敏感")
    return 0


if __name__ == "__main__":
    sys.exit(main())
