"""在**全量真实掩码**上验证 `largest_component` 的尺寸守卫不会误伤。

背景：`keep_largest` 对"物体很小的帧"会把多块切成一块，可能阉割物体。
加了尺寸守卫（最大块 < 全图 min_keep_ratio 时原样返回）后，
必须用真实数据确认：**该保留的帧没被切**、**该去噪的帧确实简化了**。

用法：
  env\\venv\\Scripts\\python.exe scripts\\check_mask_guard.py --job real-final
  env\\venv\\Scripts\\python.exe scripts\\check_mask_guard.py --job img2-01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--guard", type=float, default=0.01)
    args = ap.parse_args()

    from PIL import Image
    from scipy import ndimage

    from gsflow.imageops import largest_component

    mdir = ROOT / "workspace" / args.job / "02_masks"
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    cand = [p for p in mdir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files = sorted({p.name.split(".")[0]: p for p in cand}.values())
    print(f"{args.job}: {len(files)} 张掩码，守卫阈值 = {args.guard*100:.1f}% 全图面积")

    n_multi = n_simplified = n_guarded = 0
    lost_px = 0.0
    t0 = time.time()
    for p in files:
        m = np.asarray(Image.open(p).convert("L")) > 127
        if not m.any():
            continue
        struct = ndimage.generate_binary_structure(2, 1)
        _, k = ndimage.label(m, structure=struct)
        out = largest_component(m, min_keep_ratio=args.guard)
        if k > 1:
            n_multi += 1
            if out.sum() < m.sum():
                n_simplified += 1
                lost_px += 1.0 - out.sum() / max(m.sum(), 1)
            else:
                n_guarded += 1
    dt = time.time() - t0
    print(f"  多块帧          : {n_multi}")
    print(f"  其中被简化      : {n_simplified}"
          + (f"（平均丢掉 {lost_px/max(n_simplified,1)*100:.1f}% 前景像素）" if n_simplified else ""))
    print(f"  其中被守卫保护  : {n_guarded}  ← 这些帧物体较小，切了会缺一块")
    print(f"  总耗时 {dt:.1f}s（平均 {dt/max(len(files),1)*1000:.0f} ms/张）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
