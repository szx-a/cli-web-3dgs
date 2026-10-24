"""把掩码轮廓叠在真值上并放大，肉眼判断"掩码贴不贴"。

数字指标（`rim_ratio`）说掩码偏大 4~5px，但那依赖"边缘带亮度≈背景"的假设。
这里直接出图：真值 + 红色掩码轮廓，4 倍放大，看轮廓是压在物体边缘上，还是飘在背景里。

用法: python scripts/visualize_mask_overlay.py --job img3-final --views 3
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
    ap.add_argument("--job", default="img3-final")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--out", default="env/tmp/mask_overlay.png")
    args = ap.parse_args()

    from PIL import Image
    from scipy import ndimage as nd

    jd = ROOT / "workspace" / args.job
    data = Path(args.data_dir) if args.data_dir else jd / "04_train" / "data"
    names = sorted(p.name for p in (data / "images").glob("*")
                   if p.suffix.lower() in (".jpg", ".png"))
    picks = np.linspace(0, len(names) - 1, args.views).astype(int).tolist()
    tiles = []
    for k in picks:
        n = names[k]
        with Image.open(data / "images" / n) as gi:
            img = np.asarray(gi.convert("RGB"), dtype=np.float32) / 255.0
        mp = data / "masks" / (n + ".png")
        if not mp.exists():
            mp = data / "masks" / (Path(n).stem + ".png")
        with Image.open(mp) as mi:
            m = np.asarray(mi.convert("L").resize((img.shape[1], img.shape[0]),
                                                  Image.NEAREST)) > 127
        edge = m & ~nd.binary_erosion(m, iterations=1)
        out = img.copy()
        out[edge] = [1.0, 0.1, 0.1]            # 轮廓涂红
        # 裁出物体包围盒附近并放大
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        y0, y1 = max(ys.min() - 6, 0), min(ys.max() + 6, img.shape[0])
        x0, x1 = max(xs.min() - 6, 0), min(xs.max() + 6, img.shape[1])
        crop = out[y0:y1, x0:x1]
        big = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (crop.shape[1] * args.zoom, crop.shape[0] * args.zoom), Image.NEAREST),
            dtype=np.float32) / 255.0
        tiles.append(big)
    if not tiles:
        print("没有可用的图")
        return 1
    h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        pad = np.zeros((h, t.shape[1], 3), dtype=np.float32)
        pad[:t.shape[0]] = t
        padded.append(pad)
    sheet = np.concatenate(padded, axis=1)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(out_path)
    print(f"  已写出 {out_path}（{args.views} 张 ×{args.zoom} 放大，红线=掩码轮廓）")
    print("  判读：红线若落在物体边缘之外（背景里）⇒ 掩码偏大；压在边缘上 ⇒ 掩码正确。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
