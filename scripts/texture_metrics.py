"""报告纹理质量指标（黑色 texel 比例 + 细节量），用于跨作业对比。

功率：`python scripts/texture_metrics.py --job img3-crop --job final-v1`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[])
    args = ap.parse_args()
    jobs = args.job or ["img3-crop", "final-v1"]

    from PIL import Image

    print(f"  {'作业':<14}{'尺寸':>10}{'黑色texel':>11}{'细节量':>9}{'覆盖率':>9}")
    print("  " + "-" * 56)
    for j in jobs:
        p = ROOT / "workspace" / j / "05_export" / "model_mesh_texture.png"
        if not p.exists():
            print(f"  {j:<14}   （无纹理）")
            continue
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGB"))
        black = float((a.max(axis=2) <= 10).mean()) * 100
        g = a.mean(axis=2)
        gy, gx = np.gradient(g)
        detail = float(np.hypot(gx, gy).mean())
        nonempty = float((a.max(axis=2) > 10).mean()) * 100
        print(f"  {j:<14}{str(a.shape[:2]):>10}{black:>10.2f}%{detail:>9.2f}"
              f"{nonempty:>8.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
