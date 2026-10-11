"""统计掩码的前景占比与连通性，判断这次拍摄的"可用像素覆盖率"是否够。

动机：IMG2（牌楼）重建糊、网格软。已排除"裁剪没做"（掩码只有 1 个连通块），
所以怀疑是**数据侧**问题。这里量化两个关键量：

  1. 前景占比的**分布**（中位数 / 极差）—— 占比小且波动大，意味着
     每帧贡献的物体像素少、且视角间不一致，训练与位姿都会更难。
  2. 前景的**连通块数**（>1 说明掩码把物体拆散了，会干扰 COLMAP）。

用法：
  env\\venv\\Scripts\\python.exe scripts\\diag_mask_coverage.py --job img2-01
  env\\venv\\Scripts\\python.exe scripts\\diag_mask_coverage.py --job img2-01 --compare
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
    ap.add_argument("--masks", default="", help="掩码目录，默认自动找 02_masks")
    args = ap.parse_args()

    from PIL import Image
    from scipy import ndimage

    job = ROOT / "workspace" / args.job
    mdir = Path(args.masks) if args.masks else job / "02_masks"
    if not mdir.exists():
        print(f"没有掩码目录 {mdir}")
        return 1
    # 两种命名都可能存在（image.jpg.png / image.png），去重；
    # 并**按扩展名过滤** —— 掩码目录里还有 manifest.json，直接 Image.open 会炸
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    cand = [p for p in mdir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files = sorted({p.name.split(".")[0]: p for p in cand}.values())
    print(f"掩码 {len(files)} 张（{mdir}）")

    fracs, ncomp, areas = [], [], []
    for p in files:
        m = np.asarray(Image.open(p).convert("L")) > 127
        if m.size == 0:
            continue
        fracs.append(float(m.mean()))
        _, k = ndimage.label(m, structure=ndimage.generate_binary_structure(2, 1))
        ncomp.append(int(k))
        areas.append(int(m.sum()))
    f = np.asarray(fracs)
    print(f"\n前景占比：min={f.min()*100:.1f}%  p25={np.quantile(f,0.25)*100:.1f}%  "
          f"p50={np.median(f)*100:.1f}%  p75={np.quantile(f,0.75)*100:.1f}%  max={f.max()*100:.1f}%")
    print(f"          均值={f.mean()*100:.1f}%   极差={(f.max()-f.min())*100:.1f} 个百分点")
    nc = np.asarray(ncomp)
    print(f"连通块数：min={nc.min()}  p50={int(np.median(nc))}  max={nc.max()}  "
          f"（>1 的帧占 {float((nc>1).mean())*100:.1f}%）")
    print(f"前景像素数：中位数 {int(np.median(areas)):,}  "
          f"（这是每帧真正参与重建的像素量）")

    print("\n判读：")
    print("  * 前景中位数 < 10 万像素   -> 每帧有效观测太少，重建必然偏软")
    print("  * 前景占比极差 > 25 个百分点 -> 视角间尺度/覆盖严重不一致")
    print("  * 连通块 > 1 的比例高       -> 掩码把物体拆散，COLMAP 位姿会被干扰")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
