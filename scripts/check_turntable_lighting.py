"""转台拍法的**光照自洽性检验** —— 这是它比"缺视角"更根本的问题。

## 为什么转台拍法对 3DGS 是"几何上等价、光照上不等价"

转台拍法（物体转、相机与**光源都固定**）时：
  * 相机看到物体转了 30°；
  * 但**光照方向相对物体也转了 30°**（因为光在房间里不动，物体自己转了）。

而"相机绕物体走 30°"时：
  * 相机看到的是同一个面，**光照方向相对物体不变**（相机和光都不动）。

⇒ 两者**几何等价**（相机相对物体的位姿相同），但**光照不等价**。
3DGS 用 SH 建模"视角相关的颜色"，前提是**同一表面点在不同视角下的亮度差
只来自视角，不来自光照变化**。转台拍法违反了这个前提 ⇒
模型被迫用模糊的、半透明的体积去"平均"掉光照差异 ⇒ **细节必然糊**。

## 本脚本怎么检验

挑一个**在所有帧都能看到的表面区域**（前景里最稳定的像素群），
统计它在不同帧的亮度分布：
  * 若亮度**基本恒定** ⇒ 光照相对物体不变 ⇒ 更像"相机绕行"或光源跟着转台转；
  * 若亮度**随转台角度系统性变化** ⇒ 光照相对物体在变 ⇒ 转台 + 固定光源，
    3DGS 的 SH 前提被破坏，必须做光照补偿或改用颜色一致的方案。

方法：用掩码取前景，按帧统计**前景的亮度直方图均值/分位数**，
并检查它与"转台转角（由相机方位角近似）"的相关性。

用法: python scripts/check_turntable_lighting.py --job img3-tt
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
    ap.add_argument("--max-frames", type=int, default=120)
    args = ap.parse_args()

    from PIL import Image

    job = ROOT / "workspace" / args.job
    img_dir = job / "01_images_raw"
    mask_dir = job / "02_masks"
    if not img_dir.exists():
        print("没有 01_images_raw")
        return 1

    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    step = max(1, len(files) // args.max_frames)
    files = files[::step]
    print(f"采样 {len(files)} 帧")

    # 收集"前景亮度"的时间序列
    fg_means, fg_p10, fg_p90, angles = [], [], [], []
    for k, p in enumerate(files):
        m = None
        if mask_dir.exists():
            for c in (mask_dir / (p.stem + ".png"), mask_dir / (p.name + ".png")):
                if c.exists():
                    with Image.open(c) as im:
                        m = np.asarray(im.convert("L")) > 127
                    break
        with Image.open(p) as im:
            g = np.asarray(im.convert("L"), dtype=np.float32)
        if m is not None and m.shape != g.shape:
            from PIL import Image as PILImage
            m = np.asarray(PILImage.fromarray((m.astype(np.uint8) * 255)).resize(
                (g.shape[1], g.shape[0]), PILImage.NEAREST)) > 127
        if m is None:
            m = g > 12          # 兜底：非黑即前景
        if m.sum() < 100:
            continue
        vals = g[m]
        fg_means.append(float(vals.mean()))
        fg_p10.append(float(np.quantile(vals, 0.10)))
        fg_p90.append(float(np.quantile(vals, 0.90)))
        angles.append(k / max(len(files) - 1, 1) * 360.0)   # 用帧序近似转台角度

    if len(fg_means) < 10:
        print("有效帧太少")
        return 1

    fm = np.asarray(fg_means)
    print(f"\n=== 前景亮度随帧序号的变化 ===")
    print(f"  均值范围 {fm.min():.1f} ~ {fm.max():.1f}（跨度 {fm.max()-fm.min():.1f}）")
    print(f"  变异系数 CV = {fm.std()/max(fm.mean(),1e-9):.3f}")
    print(f"  p10 范围 {min(fg_p10):.1f} ~ {max(fg_p10):.1f}")
    print(f"  p90 范围 {min(fg_p90):.1f} ~ {max(fg_p90):.1f}")

    # 检查随角度的系统性变化（用一阶谐波拟合：a*cos+b*sin+c）
    a = np.asarray(angles) * np.pi / 180.0
    A = np.stack([np.cos(a), np.sin(a), np.ones_like(a)], axis=1)
    coef, *_ = np.linalg.lstsq(A, fm, rcond=None)
    amp = float(np.hypot(coef[0], coef[1]))
    resid = fm - A @ coef
    print(f"\n  一阶谐波振幅 = {amp:.2f}（相对均值 {amp/max(fm.mean(),1e-9)*100:.1f}%）")
    print(f"  拟合残差 std = {resid.std():.2f}")

    print("\n=== 判读 ===")
    cv = fm.std() / max(fm.mean(), 1e-9)
    if cv < 0.05:
        print(f"  [判断] 前景亮度几乎恒定（CV={cv:.3f} < 0.05）")
        print("         ⇒ 光照相对物体基本不变 ⇒ 可能是**光源随转台一起转**，")
        print("           或相机实际在绕行。此时 3DGS 的 SH 前提成立，保留背景即可。")
    else:
        print(f"  [判断] 前景亮度随帧明显变化（CV={cv:.3f} ≥ 0.05，"
              f"一阶振幅占 {amp/max(fm.mean(),1e-9)*100:.1f}%）")
        print("         ⇒ 光照相对物体在变 ⇒ **转台 + 固定光源**，")
        print("           3DGS 的'视角相关颜色'前提被破坏。")
        print("           可行对策（按代价从低到高）：")
        print("             1) 只用 SH 0 阶（关闭视角相关颜色），让模型学'平均外观'；")
        print("             2) 训练前做**亮度归一化**（按帧把前景均值/直方图对齐）；")
        print("             3) 真正做逆渲染（估光照 + 反照率），代价最大。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
