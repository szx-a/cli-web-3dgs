"""检查前若干帧是否"几乎不动"，并输出关键帧拼图 —— 排查帧序异常。

背景：`verify_turntable_full_turn.py` 发现"帧 0→33 距离仅 1.34（几乎相同），
帧 33→66 骤增到 36"。这不正常，可能是：
  A. 视频开头物体确实静止了约 1.4 秒（转台还没启动）；
  B. 抽帧顺序/时间戳有问题（例如抽到了重复帧）；
  C. 我的距离度量被掩码影响（前景区域重合度低）。

本脚本逐帧打印相邻帧差异 + 输出关键帧拼图，用来区分 A/B/C。

用法: python scripts/diagnose_frame_start.py --job img3-01 --n 45
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
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--sheet", default="env/tmp/key_frames.png")
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    job = ROOT / "workspace" / args.job
    d = job / "01_images_raw"
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"{len(files)} 帧")

    # 1) 前 n 帧的相邻差异
    print(f"\n=== 前 {args.n} 帧的相邻帧差异（全图，0~255）===")
    prev = None
    diffs = []
    for i in range(min(args.n, len(files))):
        with Image.open(files[i]) as im:
            g = np.asarray(im.convert("L"), dtype=np.float32)
        if prev is not None:
            v = float(np.abs(g - prev).mean())
            diffs.append((i, v))
            bar = "#" * int(min(v, 40))
            print(f"  {i-1:>3} -> {i:<3} {v:6.3f}  {bar}")
        prev = g

    zero = [i for i, v in diffs if v < 0.5]
    print(f"\n  差异 < 0.5 的相邻帧对：{zero[:12]}{' ...' if len(zero) > 12 else ''}")
    print(f"  共 {len(zero)} 对（若很多，说明开头确实静止或抽到重复帧）")

    # 2) 关键帧拼图
    picks = [0, args.n // 3, (2 * args.n) // 3, args.n,
             len(files) // 4, len(files) // 2, (3 * len(files)) // 4, len(files) - 1]
    picks = sorted(set(p for p in picks if 0 <= p < len(files)))
    ims = []
    for i in picks:
        with Image.open(files[i]) as im0:
            im = im0.convert("RGB")
            im.thumbnail((240, 240))
            ims.append((i, im.copy()))
    W = max(x[1].width for x in ims)
    H = max(x[1].height for x in ims)
    sheet = Image.new("RGB", (len(ims) * W, H + 20), (25, 25, 25))
    dr = ImageDraw.Draw(sheet)
    for k, (i, im) in enumerate(ims):
        sheet.paste(im, (k * W, 20))
        dr.text((k * W + 2, 4), f"#{i}", fill=(255, 255, 0))
    out = ROOT / args.sheet
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\n已生成关键帧拼图 {out}  尺寸 {sheet.size}")
    print(f"  帧号：{picks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
