"""两组多次重复实验的**统计判定**：均值、标准差、差异的 95% 置信区间。

为什么需要：本项目此前用**单次**运行的 0.01~0.05 dB 差异宣布"有效/无效"，
而实测同配置重复的 PSNR 标准差约 0.03~0.05 dB ⇒ 那些结论全部不成立。
判断一项改动是否有效，必须先知道**这个配置自己的**重复波动，再做两组比较。

判据：差异的 95% 置信区间**不含 0** ⇒ 该差异在 5% 水平上显著。
（样本很小时用 t 分布；这里给的是 Welch 版自由度的近似。）

用法:
  python scripts/ab_stats.py --group "λ=0:L0a,L0b,L0c" --group "λ=0.05:L5a,L5b,L5c"
  python scripts/ab_stats.py --group "..." --group "..." --metric ssim
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "workspace"

# t(0.975, df) 的小样本临界值（双尾 95%）
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def load(jobs: list[str], metric: str) -> list[float]:
    vals = []
    for j in jobs:
        p = W / j / "04_train" / "metrics.json"
        if not p.exists():
            print(f"  [跳过] {j}：缺 metrics.json")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        v = d.get(metric)
        if v is None:
            print(f"  [跳过] {j}：metrics.json 里没有 {metric}")
            continue
        vals.append(float(v))
    return vals


def mean(v: list[float]) -> float:
    return sum(v) / len(v)


def sd(v: list[float]) -> float:
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True,
                    help="形如 标签:job1,job2,job3")
    ap.add_argument("--metric", default="psnr")
    args = ap.parse_args()

    groups = []
    for g in args.group:
        label, jobs = g.split(":", 1)
        vals = load([j.strip() for j in jobs.split(",") if j.strip()], args.metric)
        groups.append((label.strip(), vals))

    print(f"\n指标 = {args.metric}")
    for label, v in groups:
        if not v:
            print(f"  {label:<14} （无数据）")
            continue
        print(f"  {label:<14} n={len(v)}  均值={mean(v):.4f}  "
              f"标准差={sd(v):.4f}  极差={max(v) - min(v):.4f}")

    if len(groups) == 2 and all(len(v) >= 2 for _, v in groups):
        (la, a), (lb, b) = groups
        ma, mb = mean(a), mean(b)
        sa, sb = sd(a), sd(b)
        na, nb = len(a), len(b)
        # Welch
        se = math.sqrt(sa * sa / na + sb * sb / nb)
        diff = mb - ma
        if se > 0:
            df_num = (sa * sa / na + sb * sb / nb) ** 2
            df_den = ((sa * sa / na) ** 2 / (na - 1)
                      + (sb * sb / nb) ** 2 / (nb - 1))
            df = df_num / df_den if df_den > 0 else 1.0
            t = T95.get(int(round(df)), 1.96)
            ci = t * se
            lo, hi = diff - ci, diff + ci
            print(f"\n两组比较：{lb} − {la} = {diff:+.4f}")
            print(f"  标准误={se:.4f}  自由度≈{df:.1f}  t95={t}  95% 置信区间=[{lo:+.4f}, {hi:+.4f}]")
            if lo > 0 or hi < 0:
                print(f"  ⇒ **显著**（区间不含 0），方向：{lb if diff > 0 else la} 更好")
            else:
                print("  ⇒ **不显著**（区间含 0）—— 这个量级的差异不足以支撑结论")
        else:
            print("\n  两组方差都为 0，无法估计标准误")
    return 0


if __name__ == "__main__":
    sys.exit(main())
