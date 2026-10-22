"""用**时间方差**剔除掩码里误当成前景的**静止背景**（仅适用于转台拍摄）。

## 为什么需要（实测）

`img3-final` 的掩码前景占 57.7%，但把 60 帧缩到 152×96 算**逐像素时间标准差**后：

```
静止像素（std < 3）占 53.1%
其中被判为前景的：占前景的 20.8%   ← ≈ 12% 整幅画面
```

也就是**约五分之一的"物体"其实是背景**。转台拍摄时相机固定、物体旋转 ⇒
背景在图像里**完全静止**（std≈0），所以静止像素一定是背景。

被误标的背景会带来两个后果（都能在渲染里看到）：
1. 模型在那里学出**假几何**（黑壳、离群高斯）；
2. 颜色被烘到网格/贴图上 —— 实测图集里那块**矩形饱和绿**就是背景里的一个
   绿色物体（每帧固定在 x107..133, y180..199，掩码 100% 判成前景）。

## 判据的适用条件（重要）

* **只在 `capture.mode == turntable` 时正确**：那时物体在转、背景静止。
  环绕拍摄（相机绕物体走）时背景也在变，本判据不适用。
* 阈值 `std < 3`（0~255 灰度）对应"几乎逐像素相同"。太大会误删物体上
  颜色均匀且缓慢转动的部分；太小会漏掉有轻微噪声的背景。

## 用法

    python scripts/fix_masks_static_bg.py --job img3-final --dry-run
    python scripts/fix_masks_static_bg.py --job img3-final --apply

`--apply` 会先把原掩码备份到 `04_train/data/masks_orig/`，再就地覆盖 `masks/`。
训练与导出只读 `masks/`，所以**不需要重跑 COLMAP**（但它当时的特征提取用的是旧掩码，
如需完全一致请连 COLMAP 一起重跑）。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img3-final")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--threshold", type=float, default=3.0, help="时间标准差阈值（灰度）")
    ap.add_argument("--probe-w", type=int, default=228, help="计算方差时的缩放宽度")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    from PIL import Image

    data = Path(args.data_dir) if args.data_dir else \
        ROOT / "workspace" / args.job / "04_train" / "data"
    img_dir, mask_dir = data / "images", data / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        print(f"找不到 images/ 或 masks/：{data}")
        return 1

    names = sorted({p.stem for p in img_dir.glob("*") if p.suffix.lower() in
                    (".jpg", ".jpeg", ".png", ".webp")})
    if not names:
        print("没有图像")
        return 1
    first = next(img_dir.glob(names[0] + ".*"))
    with Image.open(first) as im:
        W0, H0 = im.size
    W = min(args.probe_w, W0)
    H = max(1, int(W * H0 / W0))

    # ---- 逐像素时间标准差（小尺寸上算，再放大回掩码尺寸）----
    stack = []
    for n in names:
        cand = next((p for p in img_dir.glob(n + ".*")), None)
        if cand is None:
            continue
        with Image.open(cand) as im:
            stack.append(np.asarray(im.convert("L").resize((W, H), Image.BILINEAR),
                                    dtype=np.float32))
    if len(stack) < 8:
        print(f"帧数太少（{len(stack)}），判据不可靠，退出")
        return 1
    std = np.stack(stack).std(axis=0)
    static_small = std < args.threshold
    print(f"=== {args.job}：静止背景剔除 ===")
    print(f"  取样 {len(stack)} 帧，判定尺寸 {W}x{H}，阈值 std < {args.threshold}")
    print(f"  静止像素占 {static_small.mean()*100:.1f}%")

    changed, removed_total, fg_total = 0, 0, 0
    out_dir = mask_dir
    if args.apply:
        backup = data / "masks_orig"
        if not backup.exists():
            shutil.copytree(mask_dir, backup)
            print(f"  已备份原掩码到 {backup.name}/")
        out_dir = mask_dir

    for n in names:
        cand = next((p for p in mask_dir.glob(n + ".png")), None)
        if cand is None:
            continue
        with Image.open(cand) as m:
            a = np.asarray(m.convert("L"))
        fg = a > 127
        if not fg.any():
            continue
        static = np.asarray(
            Image.fromarray((static_small.astype(np.uint8) * 255)).resize(
                (a.shape[1], a.shape[0]), Image.NEAREST)) > 127
        new_fg = fg & ~static
        removed = int(fg.sum() - new_fg.sum())
        removed_total += removed
        fg_total += int(fg.sum())
        if removed:
            changed += 1
            if args.apply:
                Image.fromarray((new_fg.astype(np.uint8) * 255)).save(out_dir / cand.name)
                # 同时更新 `name.jpg.png` 这种双命名，避免两套掩码不一致
                alt = out_dir / (n + ".jpg.png")
                if alt.exists():
                    Image.fromarray((new_fg.astype(np.uint8) * 255)).save(alt)

    print(f"  受影响的掩码: {changed}/{len(names)}")
    if fg_total:
        print(f"  前景像素减少: {removed_total:,} / {fg_total:,}"
              f"（**{removed_total/max(fg_total,1)*100:.1f}%**）")
    if args.dry_run or not args.apply:
        print("\n  （dry-run：未写盘。确认后用 --apply 生效）")
        return 0
    print(f"\n  已就地更新 {out_dir}")
    print("  下一步：`gsflow run --job-id <job> --stages train,export --force train`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
