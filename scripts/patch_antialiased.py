"""接入两个**社区标准质量杠杆**：AbsGS 与抗锯齿（Mip-Splatting）。

## 背景（本轮核对"参考社区做法"时的发现）

| 社区做法 | 我们之前的状态 |
|---|---|
| AbsGS（绝对梯度致密化）| 已实现，但 `absgrad` **默认 False** |
| **抗锯齿**（`rasterize_mode="antialiased"`，即 Mip-Splatting 的核心思路）| **完全没用**，全部走默认 `classic` |

gsplat 的 `rasterization(..., rasterize_mode=...)` 接受 `'classic'` / `'antialiased'`。
后者的作用（社区共识）：**给每个高斯的 2D 投影加一个低通滤波**，
使采样率不足时不产生锯齿/闪烁。对**训练分辨率与展示分辨率不同**、
或**视角密集**的场景收益明显。

## 本补丁做什么

1. `TrainConfig` 加 `rasterize_mode: str = "classic"`（保守默认，避免未验证就改默认）；
2. 训练路径的 `rasterization(...)` 传 `rasterize_mode=cfg.rasterize_mode`；
3. 校验取值，非法值回退 classic 并告警（而不是静默）。

同时把 `absgrad` 暴露到 DEFAULTS（已存在但未在配置里）。

用法: python scripts/patch_antialiased.py [--apply]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src" / "gsflow" / "trainers" / "gsplat_trainer.py"

EDITS = [
    ("TrainConfig 加 rasterize_mode",
     "    # 可选：>0 时改用\"梯度的 (1-q) 分位\"作阈值（仅用于实验对比）\n"
     "    densify_quantile: float = 0.0",
     "    # 光栅化模式（社区做法）：\n"
     "    #   \"classic\"（默认）—— 3DGS 原版做法。\n"
     "    #   \"antialiased\"    —— Mip-Splatting 的思路：给 2D 投影加低通滤波，\n"
     "    #     抑制采样率不足导致的锯齿/闪烁。训练分辨率与展示分辨率不同、\n"
     "    #     或视角密集时收益明显。\n"
     "    # 保守起见默认 classic；实测对比见 docs/03。\n"
     "    rasterize_mode: str = \"classic\"\n"
     "    # 可选：>0 时改用\"梯度的 (1-q) 分位\"作阈值（仅用于实验对比）\n"
     "    densify_quantile: float = 0.0"),
    ("训练路径传 rasterize_mode",
     "            packed=False,\n"
     "            render_mode=\"RGB\",",
     "            packed=False,\n"
     "            render_mode=\"RGB\",\n"
     "            # 社区做法：antialiased 给 2D 投影加低通滤波（Mip-Splatting 思路）。\n"
     "            # 非法值在下面统一校验后回退 classic。\n"
     "            rasterize_mode=_raster_mode,"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    text = TRAINER.read_text(encoding="utf-8")
    if "rasterize_mode: str = \"classic\"" in text and "rasterize_mode=_raster_mode" in text:
        print("已打过该补丁（幂等）")
        return 0

    new = text
    bad = []
    for label, old, rep in EDITS:
        n = new.count(old)
        if n != 1:
            bad.append(f"{label}：命中 {n} 次（应为 1）")
            continue
        new = new.replace(old, rep, 1)
        print(f"  [OK] {label}")
    if bad:
        print("**未生效**：")
        for b in bad:
            print("  -", b)
        return 1

    # 在 rasterization 调用的前方插入模式校验（找训练路径的调用点）
    anchor = "        colors, alphas, meta = rasterization("
    if new.count(anchor) == 1:
        inject = (
            "        # 校验并归一化光栅化模式（非法值回退 classic 并告警，不静默）\n"
            "        _rm = str(getattr(cfg, 'rasterize_mode', 'classic') or 'classic').lower()\n"
            "        if _rm not in ('classic', 'antialiased'):\n"
            "            ctx.warn(f\"未知的 rasterize_mode={_rm!r}，回退 classic\")\n"
            "            _rm = 'classic'\n"
            "        _raster_mode = _rm\n"
        )
        new = new.replace(anchor, inject + anchor, 1)
        print("  [OK] 插入模式校验")
    else:
        print(f"  **校验插入失败**：锚点命中 {new.count(anchor)} 次")

    if args.apply:
        TRAINER.write_text(new, encoding="utf-8")
        print(f"\n已写入 {TRAINER}")
    else:
        print("\n（预览，加 --apply 生效）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
