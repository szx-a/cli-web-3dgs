"""判断视频是"相机绕物体转"还是"物体在转台上转、相机静止" —— 不依赖 COLMAP。

**为什么必须独立判断**：用户说明"我是控制物体转来拍屏的"（转台拍法），
而 COLMAP 报出的却是"相机中心绕物体转了 5.0 半径"。
这两种拍法在**几何上是等价的**（物体转 ≡ 相机反向绕），
但**背景**完全不同：
  * 相机绕着走：背景**相对相机会变**（视差 + 遮挡关系变化）；
  * 转台 + 相机静止：**背景几乎不动**，只有中间的物体在转。

若能判定"背景静止、物体在转"，那么：
  1. COLMAP 解出的"相机绕行"是**把物体的旋转错误地归因给了相机**；
  2. 用这套位姿训练 3DGS，模型会把**背景当作空**（因为我们做了前景抠图），
     而物体各面之间的几何关系是**转出来的**，不是"看出来的" ⇒ 视角约束是假的；
  3. **这直接解释了"点云糊、网格/纹理灾难"** —— 不是优化不够，是几何假设错了。

判据（三个独立的量）：
  A. **背景区域**（掩码之外）的帧间差异：转台拍法下应**极小**。
  B. **前景区域**的帧间差异：转台拍法下是"物体整体平移/旋转"，差异较大。
  C. **背景/前景差异比**：转台 ⇒ 比值很小；相机绕行 ⇒ 两者都大。

用法: python scripts/detect_turntable_capture.py --job img3-01
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
    ap.add_argument("--pairs", type=int, default=40, help="检查多少对相邻帧")
    args = ap.parse_args()

    from PIL import Image

    job = ROOT / "workspace" / args.job
    img_dir = job / "01_images_raw"
    mask_dir = job / "02_masks"

    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if len(files) < 3:
        print("图像太少")
        return 1
    print(f"{len(files)} 张图像；掩码目录存在={mask_dir.exists()}")

    def load_mask(p: Path) -> np.ndarray | None:
        if not mask_dir.exists():
            return None
        cands = [mask_dir / (p.stem + ".png"), mask_dir / (p.name + ".png")]
        for c in cands:
            if c.exists():
                with Image.open(c) as im:
                    return np.asarray(im.convert("L")) > 127
        return None

    step = max(1, len(files) // args.pairs)
    pairs = [(files[i], files[i + step]) for i in range(0, len(files) - step, step)]

    bg_diffs, fg_diffs, ratios = [], [], []
    for a_p, b_p in pairs[: args.pairs]:
        with Image.open(a_p) as im1, Image.open(b_p) as im2:
            a = np.asarray(im1.convert("L"), dtype=np.float32)
            b = np.asarray(im2.convert("L"), dtype=np.float32)
        if a.shape != b.shape:
            # 裁到共同尺寸
            h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
        m = load_mask(a_p)
        if m is not None and m.shape != a.shape:
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
                (a.shape[1], a.shape[0]), Image.NEAREST)) > 127
        if m is None:
            # 无掩码：用"两帧都接近黑"当背景
            bg = (a < 12) & (b < 12)
            fg = ~bg
        else:
            bg = ~m
            fg = m
        d = np.abs(a - b)
        if bg.sum() > 100:
            bg_diffs.append(float(d[bg].mean()))
        if fg.sum() > 100:
            fg_diffs.append(float(d[fg].mean()))

    if not bg_diffs or not fg_diffs:
        print("无法区分前景/背景（缺掩码且画面全亮？）")
        return 1

    bg_m = float(np.mean(bg_diffs))
    fg_m = float(np.mean(fg_diffs))
    ratio = bg_m / max(fg_m, 1e-9)
    print(f"\n=== 帧间差异（{len(bg_diffs)} 对相邻帧）===")
    print(f"  背景区域（掩码外）平均差异 = {bg_m:6.2f}")
    print(f"  前景区域（掩码内）平均差异 = {fg_m:6.2f}")
    print(f"  背景/前景 比 = {ratio:.3f}")
    print(f"  背景差异 p10/p50/p90 = "
          f"{np.quantile(bg_diffs,0.1):.2f} / {np.median(bg_diffs):.2f} / {np.quantile(bg_diffs,0.9):.2f}")

    print("\n=== 判据 ===")
    if ratio < 0.15:
        print(f"  ⇒ **背景几乎静止（比值 {ratio:.3f} < 0.15）**")
        print("     强烈支持：**转台拍法（物体转、相机基本静止）**")
        print("     ⚠️ 此时 COLMAP 解出的'相机绕行'是把**物体的旋转归因给了相机**。")
        print("        几何约束是假的 ⇒ 3DGS 会把物体各面当成'不同视角看到的不同面'，")
        print("        但它们其实来自同一台设备的旋转 ⇒ 光照/遮挡不自洽 ⇒ 细节必然糊。")
    elif ratio < 0.5:
        print(f"  ⇒ 背景部分静止（比值 {ratio:.3f}）—— 可能是**混合**拍法或背景较远")
    else:
        print(f"  ⇒ 背景随视角明显变化（比值 {ratio:.3f} ≥ 0.5）")
        print("     支持：**相机绕物体行走**（背景应有视差）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
