"""按**实测噪声底**重新裁决所有对照实验。

为什么需要：同配置重复训练 3 次得到 PSNR 24.318 / 24.227 / 24.268
⇒ 标准差 0.046 dB、极差 **0.091 dB**。
我此前用 0.01~0.05 dB 的差异宣布"有效/无效"，**全部不成立**。
本脚本把所有对照按批次列出，并标注是否超过噪声底。

注意：只有**同一批次**（同一份数据、同一协议、同一代码版本）内的数字才可互比。
跨批次混比正是我之前犯的错（G74）。

用法: python scripts/adjudicate_experiments.py [--noise 0.091]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "workspace"

# 批次定义：(批次名, [(作业, 说明)], 基线说明)
BATCHES: list[tuple[str, list[tuple[str, str]]]] = [
    ("噪声底（同配置重复 3 次）", [("rep-1", "重复 1"), ("rep-2", "重复 2"),
                                   ("rep-3", "重复 3")]),
    ("深度失真正则 λd（同批，SSIM 未修正）", [("dd-0", "λd=0（基线）"),
                                              ("dd-075", "λd=0.75"),
                                              ("dd-15", "λd=1.5")]),
    ("SSIM eps 写法（同批，均开 λd=1.5）", [("ssim-old", "原始 3DGS 单边 eps"),
                                            ("ssim-new", "修正（对称）写法")]),
    ("学习率调度", [("sch-exp", "纯指数（基线）"), ("sch-cos", "预热 1500 + 余弦")]),
    ("四元数学习率", [("q-1e3", "1e-3（基线）"), ("q-5e4", "5e-4")]),
    ("LPIPS 损失（单次，噪声底未知时做的）",
     [("lp-off", "λ=0（基线）"), ("lp-002", "λ=0.02"),
      ("lp-on", "λ=0.05"), ("lp-01", "λ=0.1")]),
    ("mask_loss_mode", [("ab-inside", "inside（旧）"), ("ab-composite", "composite（新）")]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.091,
                    help="噪声底（同配置重复的 PSNR 极差，dB）")
    args = ap.parse_args()

    for name, items in BATCHES:
        print(f"\n=== {name} ===")
        rows = []
        for job, label in items:
            p = W / job / "04_train" / "metrics.json"
            if not p.exists():
                rows.append((label, None, None, None))
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            rows.append((label, d.get("psnr"), d.get("ssim"), d.get("lpips")))
        vals = [r[1] for r in rows if r[1] is not None]
        if not vals:
            print("  （无数据）")
            continue
        base = vals[0]
        for label, psnr, ssim, lp in rows:
            if psnr is None:
                print(f"  {label:<24} （无数据）")
                continue
            delta = psnr - base
            if abs(delta) < 1e-9:
                verdict = "基线"
            elif abs(delta) > args.noise:
                verdict = f"**超噪声底 ({args.noise})**"
            else:
                verdict = f"噪声内 (<{args.noise})"
            s = f"{ssim:.4f}" if isinstance(ssim, (int, float)) else "-"
            l = f"{lp:.4f}" if isinstance(lp, (int, float)) else "-"
            print(f"  {label:<24} PSNR={psnr:7.3f}  Δ={delta:+7.3f}  "
                  f"SSIM={s:>7}  LPIPS={l:>7}  {verdict}")

    print(f"\n判据：同配置重复的 PSNR 极差 = {args.noise} dB；"
          f"小于它的差异不足以支撑结论。")
    print("提醒：跨批次数字不可互比（不同数据/协议/代码版本 —— 见 docs/03 G74）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
