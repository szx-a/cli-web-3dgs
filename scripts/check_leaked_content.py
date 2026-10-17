"""检查"绿牌子/杂物"是否属于物体本身（真值），还是重建出的背景泄漏。

背景：img3-crop 渲染出绿色牌子与灰色盒子。有两种可能：
  A. **物体本身就有**（如"文明单位"类的绿牌）⇒ 不是 bug，是正确重建；
  B. **背景泄漏**（房间里的东西被重建进了物体网格）⇒ 需要判定为错误。

判据：**在原始帧（未清理）里找这些颜色**，并看它们是否落在**前景掩码内**。
  * 在掩码内 ⇒ 属于物体 ⇒ A（不是泄漏）；
  * 在掩码外 ⇒ 属于背景 ⇒ B（是泄漏）。

用法: python scripts/check_leaked_content.py --job img3-crop --frames 200
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
    ap.add_argument("--job", required=True)
    ap.add_argument("--frames", default="0,100,200,300", help="逗号分隔帧号")
    args = ap.parse_args()

    from PIL import Image

    job = ROOT / "workspace" / args.job
    img_dir = job / "01_images_raw"
    mask_dir = job / "02_masks"
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    out_dir = ROOT / "env" / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fi in [int(x) for x in args.frames.split(",") if x.strip()]:
        if fi >= len(files):
            continue
        p = files[fi]
        with Image.open(p) as im0:
            im = im0.convert("RGB")
            a = np.asarray(im)
        # 掩码
        m = None
        if mask_dir.exists():
            for c in (mask_dir / (p.stem + ".png"), mask_dir / (p.name + ".png")):
                if c.exists():
                    with Image.open(c) as mm:
                        m = np.asarray(mm.convert("L")) > 127
                    break
        if m is not None and m.shape != a.shape[:2]:
            m = None

        # 绿色（牌子）判定：G 明显高于 R 和 B
        g = ((a[:, :, 1].astype(int) - a[:, :, 0].astype(int) > 18)
             & (a[:, :, 1].astype(int) - a[:, :, 2].astype(int) > 18))
        n_g = int(g.sum())
        print(f"\n=== 帧 {fi}（{p.name}）尺寸 {im.size} ===")
        print(f"  绿色像素 {n_g}（{n_g/(a.shape[0]*a.shape[1])*100:.3f}%）")
        if n_g > 20:
            ys, xs = np.nonzero(g)
            print(f"    位置 x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
            if m is not None:
                inside = int((g & m).sum())
                print(f"    落在**前景掩码内**的绿色像素 {inside}/{n_g}"
                      f"（{inside/n_g*100:.1f}%）")
                verdict = ("**属于物体**（不是背景泄漏）" if inside / n_g > 0.8
                           else "**大部分在掩码外 ⇒ 是背景泄漏**")
                print(f"    ⇒ {verdict}")
            else:
                print("    （没有掩码，无法判断归属）")
        im.save(out_dir / f"leak_check_{fi}.png")
        print(f"  已保存 {out_dir / f'leak_check_{fi}.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
