"""量化诊断纹理质量：区分"UV 布局差"与"烘焙内容差"。

用户反馈纹理是"灾难"。从 2048² 的图集看，症状是：
  * 大量孤立小块、55% 像素为空 ⇒ **UV 布局碎片化**（空间浪费、接缝多）；
  * 块内是模糊的色斑、没有可辨认的结构 ⇒ **烘焙分辨率/采样差**。

本脚本给出**可量化**的指标，避免只凭肉眼判断：
  1. **UV 占用率**：非空像素占比（越高越好；好的 unwrap 通常 50~75%）；
  2. **连通块数量与大小分布**：碎片化程度（块越多、越小 ⇒ 接缝越多）；
  3. **块内细节量**：非空区域的高频能量（拉普拉斯方差）——糊则低；
  4. **接缝密度**：相邻非空块之间的边界长度 / 非空像素数；
  5. （可选）把纹理里最大的几块单独存成 PNG，便于目视。

用法: scripts\\gpu_python.cmd scripts\\diagnose_texture.py --job img3-crop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--tex", default="", help="纹理路径（默认 05_export/model_mesh_texture.png）")
    ap.add_argument("--save-blocks", type=int, default=0, help=">0 时把最大的 N 块存成 PNG")
    args = ap.parse_args()

    from PIL import Image

    job = ROOT / "workspace" / args.job
    tex = Path(args.tex) if args.tex else (job / "05_export" / "model_mesh_texture.png")
    if not tex.exists():
        print(f"找不到纹理 {tex}")
        return 1

    with Image.open(tex) as im:
        a = np.asarray(im.convert("RGB"))
    H, W = a.shape[:2]
    g = a.mean(axis=2)
    nz = g > 8                     # 非空（非纯黑）
    occ = float(nz.mean())
    print(f"纹理 {tex.name}  {W}x{H}")
    print(f"  ① UV 占用率 = {occ*100:.1f}%"
          f"（良好 unwrap 通常 50~75%；< 35% 说明碎片化严重）")

    # ② 连通块（4 邻接）
    try:
        from scipy import ndimage
        lab, n = ndimage.label(nz)
        sizes = np.bincount(lab.ravel())[1:]
        sizes_sorted = np.sort(sizes)[::-1]
        big = int((sizes > 1000).sum())
        print(f"  ② 连通块：{n} 个（>1000 像素的 {big} 个）")
        print(f"     最大块 {sizes_sorted[0]:,} px，前 10 块占比"
              f" {sizes_sorted[:10].sum()/max(sizes.sum(),1)*100:.1f}%")
        if big > 50:
            print(f"     ⇒ **碎片化严重**：{big} 个大块，说明 unwrap 把表面切得很碎")
    except ImportError:
        n, lab, sizes = -1, None, None
        print("  ② 连通块：跳过（缺 scipy）")

    # ③ 细节量：非空区域的高频能量
    gy, gx = np.gradient(g)
    grad = np.hypot(gx, gy)
    detail = float(grad[nz].mean()) if nz.any() else 0.0
    var_lap = float((np.abs(np.gradient(np.gradient(g, axis=0), axis=0)
                            + np.gradient(np.gradient(g, axis=1), axis=1)))[nz].mean()) \
        if nz.any() else 0.0
    print(f"  ③ 细节量：梯度均值 {detail:.2f}，拉普拉斯均值 {var_lap:.2f}")
    print(f"     （数值越低越糊；干净实拍的纹理梯度通常 > 3）")

    # ④ 接缝密度：非空块边界长度
    if lab is not None:
        # 非空像素中，邻居是空或属于不同标签的数量
        b = np.zeros_like(nz)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(lab, (dy, dx), axis=(0, 1))
            b |= nz & ((~nz) | (sh != lab))
        seam = float(b.sum())
        print(f"  ④ 接缝/边缘像素 = {int(seam):,}"
              f"（占非空 {seam/max(nz.sum(),1)*100:.1f}%）")

    # ⑤ 最大几块单独存
    if args.save_blocks > 0 and lab is not None:
        outdir = job / "05_export" / "texture_blocks"
        outdir.mkdir(exist_ok=True)
        for k, idx in enumerate(np.argsort(sizes)[::-1][: args.save_blocks], 1):
            ys, xs = np.nonzero(lab == idx + 1)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = a[y0:y1, x0:x1]
            Image.fromarray(crop).save(outdir / f"block_{k}_{len(ys)}px.png")
        print(f"  ⑤ 已保存最大的 {args.save_blocks} 块到 {outdir}")

    report = {"texture": tex.name, "size": [W, H], "occupancy": round(occ, 4),
              "detail_grad": round(detail, 3), "laplacian": round(var_lap, 3),
              "n_blocks": int(n) if lab is not None else -1,
              "blocks_gt_1000px": int(big) if lab is not None else -1}
    (job / "05_export" / "texture_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {job / '05_export' / 'texture_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
