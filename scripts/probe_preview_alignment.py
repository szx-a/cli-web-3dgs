"""量化"渲染 vs 真值"的**内容位置与尺度**是否一致。

动机：`preview_*.png` 里左（渲染）与右（真值）看着尺度不同（渲染的牌坊填满画面，
真值只有顶部一块），而内参检查是匹配的。肉眼判断容易出错，这里直接算：
  * 各自非黑像素的包围盒（相对画面高度的上下边界）
  * 质心位置
  * 覆盖率
若两者包围盒差异很大，说明相机/裁剪不一致；若接近，说明只是观感差异。

用法: python scripts/probe_preview_alignment.py --job dens15k-v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def bbox_stats(arr: np.ndarray, thresh: float = 0.06) -> dict:
    """返回非黑内容的包围盒（归一化到 [0,1]）与质心、覆盖率。"""
    g = arr.max(axis=2) if arr.ndim == 3 else arr
    m = g > thresh
    if not m.any():
        return {"empty": True}
    ys, xs = np.nonzero(m)
    h, w = m.shape
    return {
        "empty": False,
        "top": float(ys.min() / h), "bottom": float(ys.max() / h),
        "left": float(xs.min() / w), "right": float(xs.max() / w),
        "height_frac": float((ys.max() - ys.min() + 1) / h),
        "width_frac": float((xs.max() - xs.min() + 1) / w),
        "centroid": [float(xs.mean() / w), float(ys.mean() / h)],
        "coverage": float(m.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--preview", default="preview_front.png")
    args = ap.parse_args()

    job = ROOT / "workspace" / args.job
    prev = job / "05_export" / args.preview
    if not prev.exists():
        print(f"找不到 {prev}")
        return 1
    with Image.open(prev) as im:
        a = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    H, W, _ = a.shape
    print(f"{args.preview}: {W}x{H}")
    # 预览是左右拼接（左渲染 / 右真值）
    half = W // 2
    left, right = a[:, :half], a[:, half:]
    sl, sr = bbox_stats(left), bbox_stats(right)
    for name, s in (("左（渲染）", sl), ("右（真值）", sr)):
        if s.get("empty"):
            print(f"  {name}: 全黑")
            continue
        print(f"  {name}: 上下 [{s['top']:.3f}, {s['bottom']:.3f}] 高度占比 {s['height_frac']:.3f}  "
              f"左右 [{s['left']:.3f}, {s['right']:.3f}] 宽度占比 {s['width_frac']:.3f}")
        print(f"          质心 ({s['centroid'][0]:.3f}, {s['centroid'][1]:.3f})  覆盖率 {s['coverage']:.3f}")

    if not sl.get("empty") and not sr.get("empty"):
        dh = abs(sl["height_frac"] - sr["height_frac"])
        dc = abs(sl["centroid"][1] - sr["centroid"][1])
        print(f"\n高度占比差 {dh:.3f}   质心纵向差 {dc:.3f}")
        scale_ratio = sl["height_frac"] / max(sr["height_frac"], 1e-6)
        print(f"渲染/真值 的高度比 = {scale_ratio:.3f}（1.0 表示尺度一致）")
        if dh > 0.15 or dc > 0.10:
            print("**渲染与真值的取景明显不一致 -> 相机或裁剪有问题**")
            return 1
        print("渲染与真值取景基本一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
