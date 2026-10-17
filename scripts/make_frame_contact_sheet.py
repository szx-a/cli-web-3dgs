"""把 IMG3 的抽帧做成"联系表"（contact sheet），用肉眼判断相机是否真的绕了一圈。

为什么需要：`inspect_dataset.py` 用**相机中心坐标**算方位角覆盖，
得出"只覆盖 247.8°、缺 112.2°"。但用户说"我拍的确实是一圈"。
两种可能：
  A. 工具错（坐标解析/方位轴选错/COLMAP 模型选错）；
  B. 数据确实不是完整一圈。
**看图像本身**是最直接的仲裁 —— 若图像里能看到物体各个面，就是绕了一圈。

本脚本输出：
  1. 12 张等间隔采样拼成一张图（看视角变化）；
  2. 每张图的前景占比（看物体是否居中/大小一致）；
  3. 相邻帧的图像差（判断是否连续移动，还是跳切）。

用法: python scripts/make_frame_contact_sheet.py --job img3-01 --n 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--thumb", type=int, default=210)
    args = ap.parse_args()

    job = ROOT / "workspace" / args.job
    d = job / "01_images_raw"
    # ⚠️ 只取图像文件：目录里还有 `_frames.json` 等元数据（COLMAP 会把它们当图像读而报错）
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not files:
        print(f"{d} 下没有图像")
        return 1
    print(f"找到 {len(files)} 张图像：{files[0].name} … {files[-1].name}")

    idx = [int(i * (len(files) - 1) / max(args.n - 1, 1)) for i in range(args.n)]
    rows = (len(idx) + args.cols - 1) // args.cols

    thumbs = []
    fgs = []
    for i in idx:
        p = files[i]
        with Image.open(p) as im0:
            im = im0.convert("RGB")
            im.thumbnail((args.thumb, args.thumb))
            arr = np.asarray(im0.convert("RGB"))
            fg = float((arr.max(axis=2) > 12).mean())
        thumbs.append((i, p.name, im.copy()))
        fgs.append(fg)

    W = max(t[2].width for t in thumbs)
    H = max(t[2].height for t in thumbs)
    sheet = Image.new("RGB", (args.cols * W, rows * (H + 18)), (25, 25, 25))
    dr = ImageDraw.Draw(sheet)
    for k, (i, name, im) in enumerate(thumbs):
        x = (k % args.cols) * W
        y = (k // args.cols) * (H + 18)
        sheet.paste(im, (x, y + 18))
        dr.text((x + 2, y + 4), f"#{i} {name} fg={fgs[k]*100:.0f}%", fill=(255, 255, 0))

    out = job / f"contact_sheet_{args.n}.png"
    sheet.save(out)
    print(f"已生成 {out}  尺寸 {sheet.size}")

    # 相邻采样帧的图像差异（判断视角变化幅度）
    print("\n相邻采样帧的差异（平均绝对差，0~255）：")
    for k in range(1, len(thumbs)):
        a = np.asarray(thumbs[k - 1][2].convert("L"), dtype=np.int16)
        b = np.asarray(thumbs[k][2].convert("L"), dtype=np.int16)
        if a.shape != b.shape:
            print(f"  #{idx[k-1]} -> #{idx[k]}：尺寸不同，跳过")
            continue
        print(f"  #{idx[k-1]:>3} -> #{idx[k]:>3}：{np.abs(a-b).mean():6.2f}")

    print(f"\n前景占比：均值 {np.mean(fgs)*100:.1f}%  范围 {min(fgs)*100:.1f}%~{max(fgs)*100:.1f}%")
    print("\n判读：若相邻帧差异都很小且图像看起来在绕圈，说明相机确实在连续移动；")
    print("      若某些区间图像差异骤增，说明那里有跳切或快速移动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
