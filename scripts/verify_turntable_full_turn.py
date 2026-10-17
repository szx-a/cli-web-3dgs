"""验证转台"是否转满 360°" —— 用图像相似度，不依赖任何相机坐标。

## 为什么必须换判据

我此前用"COLMAP 解出的相机方位角覆盖"来判断，得出"IMG3 只覆盖 180°"，
而用户明确说"360 度都拍到了"。**我的判据前提就错了**：

转台拍法下**相机不动、物体在转**，所以 COLMAP 解出的"相机位置"只是
"物体相对相机的旋转"的等价表示。它反映的是**物体转过多少角度**，
而不是"相机走过多少位置"。用方位角覆盖去推断"有没有拍到背面"是不可靠的。

## 正确判据

转台转满一圈 ⇔ **最后一帧 ≈ 第一帧**（物体转回原方位）。
更强的判据：对每个帧 i，求它与**所有其他帧**的图像距离，
**最小距离应出现在"相差约 360°"的位置**（即首尾附近），而不是别处。

本脚本用前景区域的距离（掩码内平均绝对差），并输出：
  1. 首帧 vs 末帧的距离（转满一圈 ⇒ 很小）；
  2. 首帧 vs 每一帧的距离曲线上的**最小值位置**（应接近末尾）；
  3. 若最小距离出现在中间某帧 ⇒ 转台只转了不到一圈（来回或半圈）；
  4. 用"首帧附近 N 帧"与"末帧附近 N 帧"的平均距离判断是否闭合。

用法: python scripts/verify_turntable_full_turn.py --job img3-01
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
    ap.add_argument("--samples", type=int, default=60, help="均匀采样多少帧做对比")
    args = ap.parse_args()

    from PIL import Image

    job = ROOT / "workspace" / args.job
    img_dir = job / "01_images_raw"
    mask_dir = job / "02_masks"
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if len(files) < 10:
        print("帧太少")
        return 1
    print(f"{len(files)} 帧")

    def load(p: Path):
        with Image.open(p) as im:
            g = np.asarray(im.convert("L"), dtype=np.float32)
        m = None
        if mask_dir.exists():
            for c in (mask_dir / (p.stem + ".png"), mask_dir / (p.name + ".png")):
                if c.exists():
                    with Image.open(c) as im:
                        m = np.asarray(im.convert("L")) > 127
                    break
        if m is not None and m.shape != g.shape:
            from PIL import Image as PILImage
            m = np.asarray(PILImage.fromarray((m.astype(np.uint8) * 255)).resize(
                (g.shape[1], g.shape[0]), PILImage.NEAREST)) > 127
        if m is None:
            m = g > 12
        return g, m

    # 采样（含首尾）
    idx = sorted(set([0, len(files) - 1] + [
        int(i * (len(files) - 1) / (args.samples - 1)) for i in range(args.samples)]))
    cache = {}
    for i in idx:
        cache[i] = load(files[i])

    g0, m0 = cache[0]
    first_last = None
    curve = []
    for i in idx:
        g, m = cache[i]
        h = min(g.shape[0], g.shape[1] and g.shape[0])
        H = min(g0.shape[0], g.shape[0]); W = min(g0.shape[1], g.shape[1])
        a, b = g0[:H, :W], g[:H, :W]
        mm = m0[:H, :W] & m[:H, :W]
        if mm.sum() < 100:
            mm = np.ones((H, W), dtype=bool)
        d = float(np.abs(a - b)[mm].mean())
        curve.append((i, d))
        if i == len(files) - 1:
            first_last = d

    print("\n=== 首帧与其他帧的前景差异（越小越像）===")
    step = max(1, len(curve) // 12)
    for i, d in curve[::step]:
        pct = i / (len(files) - 1) * 100
        bar = "#" * int(min(d, 60) / 2)
        print(f"  帧 {i:>4}（{pct:5.1f}%）距离 {d:6.2f}  {bar}")
    print(f"  帧 {len(files)-1:>4}（100.0%）距离 {first_last:6.2f}")

    dists = np.array([d for _, d in curve])
    fidx = np.array([i for i, _ in curve])
    imin = int(np.argmin(dists))
    print(f"\n  最小值出现在帧 {fidx[imin]}"
          f"（{fidx[imin]/(len(files)-1)*100:.1f}% 处），距离 {dists[imin]:.2f}")
    print(f"  首帧-末帧距离 {first_last:.2f}")
    print(f"  距离中位数 {np.median(dists):.2f}，最大 {dists.max():.2f}")

    # 闭合判据：末帧距离应显著小于"中位数"，且接近最小值
    med = float(np.median(dists))
    closed = first_last < med * 0.35 and first_last <= dists[imin] * 3.0
    near_end = fidx[imin] >= len(files) * 0.75

    print("\n=== 判读 ===")
    if closed and near_end:
        print(f"  ⇒ **转台转满了一圈**：末帧与首帧距离 {first_last:.2f}"
              f"（中位数 {med:.2f} 的 {first_last/med*100:.0f}%），")
        print(f"     且最小距离出现在 {fidx[imin]/(len(files)-1)*100:.0f}% 处（接近末尾）。")
        print("     ⇒ 360° 确实都拍到了。我此前用'相机方位角覆盖'得出的'只覆盖 180°'是**错误判据**。")
    elif near_end:
        print(f"  ⇒ 最小距离在末尾附近，但闭合程度不足"
              f"（末帧/中位数 = {first_last/max(med,1e-9)*100:.0f}%）")
        print("     可能是转台略多于/少于一圈，或光照有变化。")
    else:
        print(f"  ⇒ 最小距离出现在 {fidx[imin]/(len(files)-1)*100:.0f}% 处（不在末尾）")
        print("     ⇒ 转台没有转满一圈（可能是来回摆动或半圈）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
