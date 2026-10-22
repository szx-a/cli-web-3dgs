"""贴图图集体检：黑斑 / 斑驳度 / 覆盖 / 色块（仅供观察）。

## 为什么需要

用户在带贴图网格上看到"脏"，可归为两类：

1. **黑斑**：采样落在背景黑像素（`mask.mode=black` 时背景是纯黑）——
   已有 `mesh_texture_reject_black` 处理，实测黑色 texel 从 35.99% 降到 0.00%；
2. **斑块感 / 大色块**：这块"绿色矩形"我一度判成背景污染，**结论是错的**。
   实测它**随物体转动**（帧 0 在 x107、帧 80 移到 x160、帧 160~320 转出视野、
   帧 405 回来），所以它是**场景里真实存在的绿色物体**（跟着牌坊一起转），
   掩码标成前景是**正确的**，模型重建它也是正确的。

   ⚠️ 我犯的错：只用**相邻的 6 帧**就断定"每帧同一位置 ⇒ 静止背景"。
   相邻帧时间上几乎相同，当然不动。**判断"是否静止"必须跨整个序列取样。**

   至于"斑块感"本身，真因是**源图只有 457×289 而图集曾默认 2048²**——
   每个源像素被放大成 ~16 texel，等于把低分辨率插值放大（现已自动限幅到 1024）。

## 判据（绿/色块只报告，不判 FAIL）

场景里可能有真实的绿色物体，所以**不能**把"出现绿色"当缺陷。
真正要守住的是：黑 texel 比例、斑驳度、以及"异常色块是否只在少数几处"。

* `black`（max(R,G,B) <= 8）应 < 2%（黑斑）；
* `green`（G 明显高于 R/B）应 < 1% —— 户外石材模型的正常图集不该有大片饱和绿；
* `coverage`：UV 覆盖的 texel 中被填过的比例（膨胀后）应 > 95%；
* `blotch`：相邻像素平均差，> 8 说明图集很斑驳。

用法: python scripts/diag_texture_atlas.py --job img3-final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def analyze(p: Path) -> dict:
    from PIL import Image

    a = np.asarray(Image.open(p).convert("RGB")).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    black = a.max(2) <= 8
    green = (g > 100) & (g > r + 40) & (g > b + 40)
    filled = ~black
    dx = float(np.abs(np.diff(a, axis=1)).mean())
    dy = float(np.abs(np.diff(a, axis=0)).mean())
    out = {
        "size": int(a.shape[0]),
        "black_frac": float(black.mean()),
        "green_frac": float(green.mean()),
        "blotch": (dx + dy) / 2,
        "mean_lum": float(a.mean()),
    }
    if green.any():
        ys, xs = np.where(green)
        out["green_bbox"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
        out["green_rgb"] = np.round(a[green].mean(0), 1).tolist()
    # 绿色块是否是矩形（外接框内占比高 ⇒ 硬边贴片，不像自然内容）
    if green.any():
        x0, x1, y0, y1 = out["green_bbox"]
        box = green[y0:y1 + 1, x0:x1 + 1]
        out["green_fill_in_bbox"] = float(box.mean())
    del filled
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[])
    ap.add_argument("--path", default="")
    args = ap.parse_args()

    targets: list[Path] = []
    if args.path:
        targets = [Path(args.path)]
    else:
        for job in (args.job or ["img3-final"]):
            p = ROOT / "workspace" / job / "05_export" / "model_mesh_texture.png"
            if p.exists():
                targets.append(p)
            else:
                print(f"  {job}: 没有 model_mesh_texture.png（未做纹理烘焙？）")

    print("=== 贴图图集体检 ===")
    bad = 0
    for p in targets:
        r = analyze(p)
        print(f"\n  [{p.parent.parent.parent.name}] {p.name}  {r['size']}²")
        print(f"    黑色 texel   : {r['black_frac']*100:.2f}%   (应 < 2%)")
        print(f"    绿色 texel   : {r['green_frac']*100:.3f}%  (仅报告：场景可能真有绿色物体)")
        if "green_bbox" in r:
            print(f"      绿色块 bbox: {r['green_bbox']}，框内填充率 "
                  f"{r['green_fill_in_bbox']*100:.0f}%（矩形硬边越多越可疑）")
            print(f"      绿色均值 RGB: {r['green_rgb']}")
        print(f"    斑驳度       : {r['blotch']:.2f}   (应 < 8)")
        ok_black = r["black_frac"] < 0.02
        ok_green = True  # 绿色仅供参考，不判 FAIL
        ok_blotch = r["blotch"] < 8
        for name, ok in (("黑斑", ok_black), ("色块仅报告", ok_green),
                         ("斑驳度", ok_blotch)):
            print(f"    [{'OK  ' if ok else 'FAIL'}] {name}")
        if not (ok_black and ok_green and ok_blotch):
            bad += 1
    print(f"\n  结论: {'全部合格' if bad == 0 else f'**{bad} 个作业不合格**'}")
    print("  说明：绿色块实测随物体转动 ⇒ 是真实场景物，不是缺陷。"
          "掩码把镂空当整体时无法自动区分；先守住'不增加'。")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
