"""检查掩码**极性**与覆盖区域是否正确（前景=物体 还是 前景=背景）。

## 为什么要查

`img3-final` 的现象很不合理：

* 训练日志说"已同步 406 个掩码到训练数据（掩码内损失）"；
* 但实测渲染的 **alpha 全为 1.0**（画面 100% 被高斯覆盖），
  且 60% 的像素亮度 > 0.8 —— 说明整幅画面（含背景）都被拟合了；
* 把损失改成 `mask_loss_mode=inside` 几乎没变化 ⇒ 掩码很可能**没起作用或方向反了**。

判据：物体是深色瓦顶 + 白色灰泥，背景是均匀浅灰。

* 若掩码是**物体**：掩码内亮度分布应更"双峰"（有深有浅），
  掩码外应接近单一亮色（背景）；
* 若掩码**反相**：情况正好相反。

用法: python scripts/diag_mask_polarity.py --job img3-final
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
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    from PIL import Image

    data = ROOT / "workspace" / args.job / "04_train" / "data"
    imgs = sorted((data / "images").glob("*"))
    masks = data / "masks"
    if not imgs or not masks.exists():
        print("没有图像或掩码")
        return 1

    print(f"=== 掩码极性检查：{args.job} ===")
    print(f"  {'图像':<14}{'前景%':>8}{'内亮度':>9}{'外亮度':>9}{'内标准差':>10}  判读")
    votes = []
    for p in imgs[: args.n]:
        m_candidates = [masks / (p.name + ".png"), masks / (p.stem + ".png")]
        mp = next((m for m in m_candidates if m.exists()), None)
        if mp is None:
            print(f"  {p.name:<14} **找不到掩码**")
            continue
        with Image.open(p) as im:
            g = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
        with Image.open(mp) as im:
            m = np.asarray(im.convert("L")) > 127
        if m.shape != g.shape:
            # 尺寸不一致时按最近邻缩放掩码（只用于本检查）
            with Image.open(mp) as im:
                m = np.asarray(im.convert("L").resize(
                    (g.shape[1], g.shape[0]), Image.NEAREST)) > 127
        fg = float(m.mean())
        inside = float(g[m].mean()) if m.any() else float("nan")
        outside = float(g[~m].mean()) if (~m).any() else float("nan")
        sd_in = float(g[m].std()) if m.any() else float("nan")
        sd_out = float(g[~m].std()) if (~m).any() else float("nan")
        # 物体区域纹理更复杂（标准差更大）
        verdict = ""
        if sd_in > sd_out and fg < 0.9:
            verdict = "前景=物体（合理）"
            votes.append(1)
        elif sd_out > sd_in:
            verdict = "**可疑：前景更像背景（可能反相）**"
            votes.append(-1)
        else:
            verdict = "无法判断"
        print(f"  {p.name:<14}{fg*100:7.1f}%{inside:9.3f}{outside:9.3f}"
              f"{sd_in:10.3f}  {verdict}")

    if votes:
        print()
        if sum(votes) < 0:
            print("  ⇒ 多数样本显示**掩码方向可能反了**：需要检查分割后端的 alpha 解释")
        else:
            print("  ⇒ 掩码方向看起来正确（前景=物体）")
        print("    补充判据：物体区域标准差应明显大于背景区域。"
              f"（本作业 sd_in/sd_out 见上表）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
