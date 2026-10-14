"""对比多个训练作业的质量指标，用于参数实验（如 densify_until 扫描）。

输出列：高斯数 / 全留出集 PSNR / 最佳检查点 / 不透明度分位数。
**不透明度分位数很关键**：只看 PSNR 会漏掉"薄雾式退化"（见 docs/03 G43）。

用法：
  env\\venv\\Scripts\\python.exe scripts\\compare_jobs.py img2-opfix img2-den15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def opacity_stats(job: Path) -> tuple[float, float, float] | None:
    try:
        from plyfile import PlyData
    except ImportError:
        return None
    cands = sorted((job / "04_train" / "point_cloud").glob("iteration_*/point_cloud.ply"),
                   key=lambda p: int(p.parent.name.split("_")[1]))
    if not cands:
        return None
    v = np.asarray(PlyData.read(str(cands[-1])).elements[0].data["opacity"], dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-v))
    return float(np.median(p)), float(p.max()), float((p > 0.02).mean())


def ckpt_info(job: Path) -> str:
    """导出实际会用哪个检查点：优先 best_ply（最佳 PSNR 那一步），否则最后一个。"""
    tr = job / "04_train"
    mp = tr / "metrics.json"
    if not mp.exists():
        return "-"
    d = json.loads(mp.read_text(encoding="utf-8"))
    best = d.get("best_ply") or d.get("ply") or ""
    return Path(best).parent.name if best else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    args = ap.parse_args()

    rows = []
    for j in args.jobs:
        job = ROOT / "workspace" / j
        mp = job / "04_train" / "metrics.json"
        if not mp.exists():
            print(f"{j}: 无 metrics.json")
            continue
        d = json.loads(mp.read_text(encoding="utf-8"))
        op = opacity_stats(job)
        rows.append((j, d.get("gaussians"), d.get("psnr"), d.get("best_psnr"),
                     d.get("best_step"), d.get("eval_views"), op, ckpt_info(job)))

    print(f"{'作业':<14}{'高斯数':>10}{'PSNR':>9}{'最佳':>9}{'步':>8}"
          f"{'留出':>6}{'不透明度p50':>13}{'p_max':>9}{'>0.02':>8}{'导出用检查点':>16}")
    for j, g, p, bp, bs, ev, op, ck in rows:
        ps = f"{p:.2f}" if isinstance(p, (int, float)) else "-"
        bs_ = f"{bp:.2f}" if isinstance(bp, (int, float)) else "-"
        if op:
            o = f"{op[0]:.4f}"
            om = f"{op[1]:.4f}"
            of = f"{op[2]*100:.1f}%"
        else:
            o = om = of = "-"
        print(f"{j:<14}{g if g is not None else '-':>10}{ps:>9}{bs_:>9}"
              f"{str(bs or '-'):>8}{str(ev if ev is not None else '-'):>6}"
              f"{o:>13}{om:>9}{of:>8}{ck:>16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
