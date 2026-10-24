"""把一份作业数据里的掩码**腐蚀**若干像素，生成"紧掩码"数据集。

## 为什么（2026-10-24）

`check_mask_tightness.py` 实测：掩码内靠边 4px 环的颜色亮度 **0.89~0.90**，
与掩码外背景 **0.91** 几乎一致，而掩码内部只有 0.43~0.51 ⇒ **掩码整体偏大**，
那一圈其实是背景。训练会把背景学成物体，渲染出**轮廓白边**。

更糟的是：掩码内指标（PSNR/SSIM）**在奖励这种行为** —— 把背景涂成"物体"能得到更高分。
所以必须同时修正**掩码本身**与**评估口径**。

用法:
    python scripts/make_tight_masks.py --src img3-final --dst img3-tight --erode 4
之后用 `ab_densify_rasterize.py --data-from img3-tight` 训练。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="img3-final")
    ap.add_argument("--dst", default="img3-tight")
    ap.add_argument("--erode", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from PIL import Image
    from scipy import ndimage as nd

    src = ROOT / "workspace" / args.src / "04_train" / "data"
    dst = ROOT / "workspace" / args.dst / "04_train" / "data"
    if not src.exists():
        print(f"源数据不存在：{src}")
        return 1
    if dst.exists() and not args.force:
        print(f"目标已存在（用 --force 覆盖）：{dst}")
        return 1
    dst.mkdir(parents=True, exist_ok=True)

    # images / sparse：直接复制（图像原样，不改）
    for sub in ("images", "sparse"):
        s, d = src / sub, dst / sub
        if s.exists():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)

    # masks：腐蚀
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    (dst / "masks_orig").mkdir(parents=True, exist_ok=True)
    masks = sorted((src / "masks").glob("*.png"))
    kept, before, after = 0, 0, 0
    for mp in masks:
        with Image.open(mp) as mi:
            arr = np.asarray(mi.convert("L")) > 127
        if not arr.any():
            continue
        er = nd.binary_erosion(arr, iterations=args.erode)
        if not er.any():
            continue
        before += int(arr.sum())
        after += int(er.sum())
        Image.fromarray((er.astype(np.uint8) * 255)).save(dst / "masks" / mp.name)
        shutil.copy2(mp, dst / "masks_orig" / mp.name)
        kept += 1
    print(f"  {args.src} -> {args.dst}：腐蚀 {args.erode}px，处理 {kept}/{len(masks)} 张")
    print(f"  前景像素 {before} -> {after}（保留 {after/max(before,1)*100:.1f}%）")
    print(f"  数据目录：{dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
