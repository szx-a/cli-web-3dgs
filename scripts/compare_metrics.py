"""并排比较若干作业的**多指标**评估结果（PSNR / SSIM / LPIPS + 不透明度分布）。

为什么需要：用户方案第 6 项的核心论点是"PSNR 高不等于视觉好"。
`compare_jobs.py` 只列 PSNR/高斯数/不透明度；这个脚本把 SSIM 与 LPIPS 也拉进来，
并把"最佳检查点由哪个指标选出"一并显示 —— 判断"该用哪个模型导出"时缺一不可。

设计要点：**缺失的指标显示为 "-" 而不是 0**。
把"没测"和"测出来是 0"混在一起，正是本项目多次踩过的静默退化
（例如法线正则在大部分高斯上取错轴，数值仍在但不表示任何东西）。

用法: python scripts/compare_metrics.py job1 job2 job3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fmt(v, nd=4, suffix="") -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--sort", default="psnr", choices=["psnr", "ssim", "lpips"],
                    help="按哪个指标排序（lpips 越小越好，会自动反向）")
    args = ap.parse_args()

    rows = []
    for job in args.jobs:
        p = ROOT / "workspace" / job / "04_train" / "metrics.json"
        if not p.exists():
            print(f"  [跳过] {job}: 缺 metrics.json")
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        bm = m.get("best_metrics") or {}
        rows.append({
            "job": job,
            "gaussians": m.get("gaussians"),
            "iters": m.get("iterations"),
            "psnr": m.get("psnr"),
            "ssim": m.get("ssim"),
            "lpips": m.get("lpips"),
            "best_step": m.get("best_step"),
            "best_metric_used": m.get("best_metric_used"),
            "best_psnr": bm.get("psnr"),
            "best_ssim": bm.get("ssim"),
            "best_lpips": bm.get("lpips"),
        })

    if not rows:
        print("没有可比较的作业")
        return 1

    rev = args.sort != "lpips"
    key = args.sort
    rows.sort(key=lambda r: (r.get(key) is None, r.get(key) or 0), reverse=rev)

    hdr = f"{'作业':<22}{'迭代':>7}{'高斯':>9}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}" \
          f"{'最佳步':>9}{'按指标':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['job']:<22}{str(r['iters']):>7}{str(r['gaussians']):>9}"
              f"{fmt(r['psnr'], 3):>9}{fmt(r['ssim']):>9}{fmt(r['lpips']):>9}"
              f"{str(r['best_step'] or '-'):>9}{str(r['best_metric_used'] or '-'):>9}")

    print("\n最佳检查点的完整指标（导出阶段会优先用它）：")
    print(f"{'作业':<22}{'best PSNR':>11}{'best SSIM':>11}{'best LPIPS':>12}")
    for r in rows:
        print(f"{r['job']:<22}{fmt(r['best_psnr'], 3):>11}"
              f"{fmt(r['best_ssim']):>11}{fmt(r['best_lpips']):>12}")

    missing = [r["job"] for r in rows if r.get("ssim") is None]
    if missing:
        print(f"\n注意：{', '.join(missing)} 没有 SSIM 记录 —— "
              f"可能是训练早于多指标评估的加入，或 eval_metrics 被关闭。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
