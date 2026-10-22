"""比较两个模型的"雾化"来源：尺度 vs 不透明度。

## 背景

`max_scale_ratio=0.05` 在 15k 迭代上把 veil（半透明像素占比）从 17.6% 压到 8.4%，
但同样的设置跑 **30k** 反而回到 22.2%。说明还有一个**与迭代数相关**的雾化来源：

* 尺度已经被钳住了（`max_scale ≈ 0.05 × scene_scale`）；
* 那么剩下的可能是**大量低不透明度高斯**（"薄雾"）：数量多、每个贡献一点点 alpha，
  叠加后形成半透明区域 —— 这正是官方 3DGS 用**不透明度重置**要解决的问题，
  而本项目 `opacity_reset_interval=0`（关闭）。

用法: python scripts/diag_haze_sources.py --job A --job B
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

    from gsflow.gsdata.shio import read_gaussians_full

    print("=== 雾化来源：尺度上限 vs 低不透明度高斯 ===")
    print(f"  {'模型':<26}{'n':>9}{'max_scale':>11}{'op中位':>9}"
          f"{'op<0.1占比':>11}{'op<0.1权重':>11}{'针状权重':>10}")
    for job in (args.job or ["img3-final"]):
        jd = ROOT / "workspace" / job
        plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
        if not plys:
            print(f"  {job:<26} 没有产物")
            continue
        g = read_gaussians_full(plys[-1])
        sc = np.asarray(g["scales"], dtype=np.float64)
        op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
        smax = sc.max(1)
        aniso = smax / np.maximum(sc.min(1), 1e-12)
        w = op * smax ** 2
        w = w / w.sum() if w.sum() > 0 else w
        faint = op < 0.1
        print(f"  {job:<26}{len(op):>9,}{smax.max():>11.3f}"
              f"{np.median(op):>9.4f}{faint.mean()*100:>10.1f}%"
              f"{w[faint].sum()*100:>10.1f}%"
              f"{w[aniso > 100].sum()*100:>9.1f}%")
    print("\n  判读：若 'op<0.1 权重' 很高 ⇒ 雾来自**大量淡高斯**（需要不透明度重置）；")
    print("        若 max_scale 仍很大 ⇒ 尺度上限没生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
