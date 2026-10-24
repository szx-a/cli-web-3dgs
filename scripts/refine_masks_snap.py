"""掩码**边界吸附**：在边界带内按颜色归属重新判定，替代"统一腐蚀"。

## 为什么（2026-10-24）

实测本数据集掩码整体偏大 4~5px：掩码内靠边 4px 环的亮度 0.89，几乎等于掩码外背景 0.91，
而掩码内部只有 0.5（`check_mask_tightness.py`，`rim_ratio` 0.95~0.98）。
训练把这圈背景当物体 ⇒ 渲染出**轮廓白边**（`rim_bright` 0.295）。

* 统一腐蚀 4px 能压掉白边（0.037），但**内部 PSNR −2.2 dB**：薄结构被削、轮廓整体内缩；
* 本脚本改做**吸附**：只在边界带（±`--band` px）里，用**内部/外部颜色模型**重新判定
  每个像素属于哪边，再取最大连通域、填洞。

预期：白边消失，同时掩码面积几乎不缩（比腐蚀保留更多前景）。

用法:
    python scripts/refine_masks_snap.py --src img3-final --dst img3-snap --band 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="img3-final")
    ap.add_argument("--dst", default="img3-snap")
    ap.add_argument("--band", type=int, default=6, help="边界带宽度（像素）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import shutil

    import numpy as np
    from PIL import Image
    from scipy import ndimage as nd

    src = ROOT / "workspace" / args.src / "04_train" / "data"
    dst = ROOT / "workspace" / args.dst / "04_train" / "data"
    if not src.exists():
        print(f"源数据不存在：{src}")
        return 1
    if dst.exists() and not args.force:
        print(f"目标已存在（--force 覆盖）：{dst}")
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "sparse"):
        s, d = src / sub, dst / sub
        if s.exists():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    (dst / "masks_orig").mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in (src / "images").glob("*")
                   if p.suffix.lower() in (".jpg", ".png"))
    tot_orig = tot_snap = tot_erode = 0
    n_ok = 0
    moved = []
    for n in names:
        mp = src / "masks" / (n + ".png")
        if not mp.exists():
            mp = src / "masks" / (Path(n).stem + ".png")
        if not mp.exists():
            continue
        with Image.open(src / "images" / n) as gi:
            img = np.asarray(gi.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(mp) as mi:
            m = np.asarray(mi.convert("L").resize((img.shape[1], img.shape[0]),
                                                  Image.NEAREST)) > 127
        if not m.any():
            continue

        # 内部/外部训练样本：腐蚀/膨胀 3px，避开边界带
        core = nd.binary_erosion(m, iterations=3)
        far = nd.binary_dilation(m, iterations=max(args.band, 3)) & ~nd.binary_dilation(m, iterations=2)
        if not (core.any() and far.any()):
            core, far = m, ~m
        fi = img[core].reshape(-1, 3)
        fo = img[far].reshape(-1, 3)
        # 稳健统计（中位 + MAD），避免高光/阴影拉偏
        mi_, mo_ = np.median(fi, axis=0), np.median(fo, axis=0)
        si_ = np.median(np.abs(fi - mi_), axis=0) * 1.4826 + 0.03
        so_ = np.median(np.abs(fo - mo_), axis=0) * 1.4826 + 0.03

        # 只改边界带
        inner = nd.binary_erosion(m, iterations=args.band)
        outer = nd.binary_dilation(m, iterations=args.band)
        band = outer & ~inner

        di = (((img - mi_) / si_) ** 2).sum(2)
        do = (((img - mo_) / so_) ** 2).sum(2)
        snap = m.copy()
        snap[band] = di[band] < do[band]

        # 后处理：开运算去毛刺 + 最大连通域 + 填洞
        snap = nd.binary_opening(snap, iterations=1)
        lab, k = nd.label(snap)
        if k > 0:
            sizes = nd.sum(snap, lab, range(1, k + 1))
            snap = lab == (int(np.argmax(sizes)) + 1)
        snap = nd.binary_fill_holes(snap)
        if not snap.any():
            snap = m

        orig_px = int(m.sum())
        snap_px = int(snap.sum())
        er_px = int(nd.binary_erosion(m, iterations=4).sum())
        tot_orig += orig_px
        tot_snap += snap_px
        tot_erode += er_px
        moved.append((snap_px - orig_px) / max(orig_px, 1))
        n_ok += 1

        Image.fromarray((snap.astype(np.uint8) * 255)).save(dst / "masks" / Path(mp).name)
        shutil.copy2(mp, dst / "masks_orig" / Path(mp).name)

    print(f"  {args.src} -> {args.dst}：吸附 {n_ok} 张（带 {args.band}px）")
    print(f"  前景像素：原 {tot_orig:,}  吸附后 {tot_snap:,}（{tot_snap/max(tot_orig,1)*100:.1f}%）"
          f"  对比腐蚀4px {tot_erode:,}（{tot_erode/max(tot_orig,1)*100:.1f}%）")
    if moved:
        mv = np.array(moved) if (np := __import__("numpy")) else None
        print(f"  单张面积变化：中位 {np.median(mv)*100:+.1f}%  最小 {mv.min()*100:+.1f}%"
              f"  最大 {mv.max()*100:+.1f}%")
    print(f"  输出：{dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
