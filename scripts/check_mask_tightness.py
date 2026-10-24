"""检查掩码"是否偏大"：掩码边缘那一圈到底是不是背景。

## 为什么需要（2026-10-24 实测）

用户反复说模型有"轮廓白边/发光"。查下来不是渲染问题，而是**掩码偏大**：

| 图 | 掩码内部亮度 | 掩码内靠边 4px 环 | 掩码外 4px |
|---|---|---|---|
| 00001 | 0.514 | **0.892** | 0.912 |
| 00082 | 0.434 | **0.897** | 0.911 |

掩码边缘那一圈亮度几乎等于**掩码外背景**（0.89 vs 0.91），说明它其实是背景。
训练把这一圈当物体 ⇒ 模型学出**沿轮廓的亮边**。这个坑在指标上完全看不出来
（掩码内 PSNR 26.97 dB 很好看），只有肉眼能发现。

判据：`rim_ratio = (rim - interior) / (outer - interior)`
* < 0.25 → 掩码贴合（边缘是物体）
* > 0.5  → **掩码偏大**，边缘是背景，会学出白边
* 中间 → 边界略软（真实物体边缘常见），可接受但建议腐蚀 1~2px 训练

用法: python scripts/check_mask_tightness.py --job img3-final --samples 12
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
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--band", type=int, default=4)
    args = ap.parse_args()

    from PIL import Image
    from scipy import ndimage as nd

    jd = ROOT / "workspace" / args.job
    data = Path(args.data_dir) if args.data_dir else jd / "04_train" / "data"
    img_dir, mask_dir = data / "images", data / "masks"
    names = sorted(p.name for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".png"))
    if not names:
        print("没有训练图像")
        return 1
    picks = np.linspace(0, len(names) - 1, min(args.samples, len(names))).astype(int)

    print(f"=== 掩码贴合度检查：{args.job}（{len(picks)} 张，带宽 {args.band}px）===")
    print(f"  {'图像':<14}{'内部':>8}{'边缘环':>8}{'外部':>8}{'rim_ratio':>11}  判读")
    ratios = []
    for k in picks:
        n = names[k]
        mp = mask_dir / (n + ".png")
        if not mp.exists():
            mp = mask_dir / (Path(n).stem + ".png")
        if not mp.exists():
            continue
        with Image.open(img_dir / n) as gi:
            gt = np.asarray(gi.convert("L"), dtype=np.float64) / 255.0
        with Image.open(mp) as mi:
            m = np.asarray(mi.convert("L").resize((gt.shape[1], gt.shape[0]),
                                                  Image.NEAREST)) > 127
        if not m.any():
            continue
        inner = nd.binary_erosion(m, iterations=args.band)
        rim = m & ~inner
        outer = nd.binary_dilation(m, iterations=args.band) & ~m
        if not (inner.any() and rim.any() and outer.any()):
            continue
        i_m, r_m, o_m = gt[inner].mean(), gt[rim].mean(), gt[outer].mean()
        denom = max(abs(o_m - i_m), 1e-6)
        ratio = (r_m - i_m) / denom
        ratios.append(ratio)
        verdict = ("**偏大**" if ratio > 0.5 else
                   "略软" if ratio > 0.25 else "贴合")
        print(f"  {n:<14}{i_m:>8.3f}{r_m:>8.3f}{o_m:>8.3f}{ratio:>11.2f}  {verdict}")

    if not ratios:
        print("  没有可评估的样本")
        return 1
    r = float(np.mean(ratios))
    print(f"\n  平均 rim_ratio = {r:.2f}")
    if r > 0.5:
        print("  判读：**掩码整体偏大**（边缘一圈是背景）。建议：")
        print("    * 训练损失用 `train.mask_erode_px≈3~5` 的腐蚀掩码；")
        print("    * 或在掩码阶段收紧阈值/用更强的分割模型重新出掩码。")
    elif r > 0.25:
        print("  判读：掩码边界偏软（真实边缘常见）。轻微腐蚀 1~2px 可减少轮廓高光。")
    else:
        print("  判读：掩码贴合良好，白边问题不来自掩码。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
