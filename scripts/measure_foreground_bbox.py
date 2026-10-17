"""测量前景（掩码）在全部帧上的**联合包围盒**，为"裁掉空白"提供依据。

动机（IMG3 的核心问题）：
  576×1280 的竖屏画面里，牌坊只占 **8.5%** 的像素 ⇒ 有效前景像素仅约 62k/帧，
  而 IMG2（720×720、前景 38.9%）有约 200k。**每帧的"信息量"差 3 倍**。

正确做法（社区对"物体在画面里很小"的通行处理）：
  **先按掩码联合包围盒裁掉四周空白**，再训练。
  这样同样的网络容量/gaussians 预算全部落在物体上，等效分辨率提升数倍。

本脚本输出：
  1. 全部帧掩码的**联合包围盒**（并集，保证任何一帧都不被裁掉）；
  2. 建议裁剪参数（留 5% 边距），以及裁剪后的尺寸与前景占比；
  3. 逐帧前景包围盒的波动（判断是否某些帧物体位置跳变）。

用法: scripts\\gpu_python.cmd scripts\\measure_foreground_bbox.py --job img3-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--pad", type=float, default=0.05, help="边距（相对包围盒边长）")
    ap.add_argument("--out", default="", help="把建议参数写成 json")
    args = ap.parse_args()

    from PIL import Image

    job = ROOT / "workspace" / args.job
    mask_dir = job / "02_masks"
    img_dir = job / "01_images_raw"
    if not mask_dir.exists():
        print(f"没有掩码目录 {mask_dir}")
        return 1

    mask_files = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() == ".png")
    # 过滤掉 COLMAP 命名的那种（00001.jpg.png），只保留 {stem}.png
    mask_files = [p for p in mask_files if p.stem.count(".") == 0]
    print(f"{len(mask_files)} 个掩码")

    x0s, y0s, x1s, y1s, fracs = [], [], [], [], []
    W = H = None
    for p in mask_files:
        try:
            with Image.open(p) as im:
                m = np.asarray(im.convert("L")) > 127
        except Exception:  # noqa: BLE001
            continue
        if W is None:
            H, W = m.shape
        if m.shape != (H, W):
            continue
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        x0s.append(int(xs.min())); x1s.append(int(xs.max()))
        y0s.append(int(ys.min())); y1s.append(int(ys.max()))
        fracs.append(float(m.mean()))

    if not x0s:
        print("没有有效掩码")
        return 1

    X0, X1 = min(x0s), max(x1s)     # 并集：任何一帧都不裁掉
    Y0, Y1 = min(y0s), max(y1s)
    bw, bh = X1 - X0 + 1, Y1 - Y0 + 1
    print(f"\n图像尺寸 {W}x{H}")
    print(f"联合包围盒 x[{X0},{X1}] y[{Y0},{Y1}]  = {bw}x{bh}"
          f"（占画面 {bw/W*100:.1f}% × {bh/H*100:.1f}% = {bw*bh/(W*H)*100:.1f}%）")
    print(f"逐帧包围盒波动：x0 {min(x0s)}~{max(x0s)}  x1 {min(x1s)}~{max(x1s)}"
          f"  y0 {min(y0s)}~{max(y0s)}  y1 {min(y1s)}~{max(y1s)}")
    print(f"前景占比：均值 {np.mean(fracs)*100:.2f}%  范围 {min(fracs)*100:.2f}%~{max(fracs)*100:.2f}%")

    # 加边距
    px = int(bw * args.pad) + 4
    py = int(bh * args.pad) + 4
    cx0, cy0 = max(0, X0 - px), max(0, Y0 - py)
    cx1, cy1 = min(W - 1, X1 + px), min(H - 1, Y1 + py)
    cw, ch = cx1 - cx0 + 1, cy1 - cy0 + 1
    new_frac = (bw * bh) / (cw * ch)
    print(f"\n建议裁剪（边距 {args.pad*100:.0f}%）：")
    print(f"  crop = {cx0},{cy0},{cw},{ch}   尺寸 {cw}x{ch}")
    print(f"  裁剪后前景占比 ≈ {new_frac*100:.2f}%（原 {np.mean(fracs)*100:.2f}%）"
          f" ⇒ 提升 {new_frac/max(np.mean(fracs),1e-9):.1f} 倍")
    print(f"  等效像素：每帧前景 ≈ {int(bw*bh):,}（原 {int(np.mean(fracs)*W*H):,}）")

    sug = {
        "job": args.job, "image_size": [W, H],
        "bbox": [X0, Y0, X1, Y1],
        "crop": [cx0, cy0, cw, ch],
        "fg_frac_before": round(float(np.mean(fracs)), 4),
        "fg_frac_after": round(float(new_frac), 4),
        "bbox_jitter": {"x0": [min(x0s), max(x0s)], "x1": [min(x1s), max(x1s)],
                        "y0": [min(y0s), max(y0s)], "y1": [min(y1s), max(y1s)]},
    }
    out = Path(args.out) if args.out else (job / "bbox_suggestion.json")
    out.write_text(json.dumps(sug, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
