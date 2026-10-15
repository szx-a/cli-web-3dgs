"""扫描网格后处理参数，用**同一套指标**记录质量（翻转率 / IoU / 面数）。

为什么需要：网格质量的决定因素已经收敛到"一致性过滤 + 法线平均"（docs/03 G69），
但两个关键参数**从未调过**：
  * `mesh_consistency_min_views`（默认 2 —— 要求 ≥2/80 个视角支持；
    仅 2 个视角支持的法线极不可靠，这很可能是翻转率的主要来源）
  * `mesh_consistency_normal_dot`（默认 0.0 —— 只剔除"完全相反"的法线）

做法：对每个参数组合跑一次导出（复用已训练模型，不重训），
再用 `diag_normal_flips.py` 测量**最终网格**的相邻面法线一致度与翻转率。
输出一张表，便于选默认值。

用法：
  python scripts/sweep_mesh_params.py --job flat-ab-base \
      --sweep mesh_consistency_min_views=2,4,8,16
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], timeout: int = 2400) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT), timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def measure(job: str) -> dict:
    """测最终网格：相邻面法线一致度 + 翻转率 + 面数。"""
    _, out = run([sys.executable, str(ROOT / "scripts" / "diag_normal_flips.py"),
                  "--job", job, "--views", "3"], timeout=1800)
    res: dict = {}
    m = re.search(r"导出网格:\s*(\d+)\s*面", out)
    if m:
        res["triangles"] = int(m.group(1))
    m = re.search(r"相邻面法线点积:\s*mean=([\d.]+)\s*median=([\d.]+)\s*翻转\(<\)0\)=([\d.]+)", out)
    if m:
        res["mean_dot"] = float(m.group(1))
        res["median_dot"] = float(m.group(2))
        res["flip_ratio"] = float(m.group(3))
    m = re.search(r"二面角\s*>90°\s*的边占比\s*=\s*([\d.]+)%", out)
    if m:
        res["dihedral_gt90"] = float(m.group(1)) / 100.0
    # 投票统计（若日志里有）
    m = re.search(r"->\s*(\d+)\s*点（保留\s*([\d.]+)%", out)
    if m:
        res["points_kept_ratio"] = float(m.group(2))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--sweep", required=True,
                    help="形如 key=v1,v2,v3（可重复用逗号分隔的多组见 --sweep2）")
    ap.add_argument("--sweep2", default=None, help="第二个参数（与 --sweep 做网格扫描）")
    ap.add_argument("--fixed", default="", help="固定参数，形如 k=v,k=v")
    ap.add_argument("--triangles", type=int, default=80000)
    ap.add_argument("--method", default="2dgs")
    args = ap.parse_args()

    key, vals = args.sweep.split("=", 1)
    vals1 = [v for v in vals.split(",") if v]
    fixed = dict(kv.split("=", 1) for kv in args.fixed.split(",") if kv)

    combos: list[dict] = []
    if args.sweep2:
        key2, vals2 = args.sweep2.split("=", 1)
        for a in vals1:
            for b in [v for v in vals2.split(",") if v]:
                combos.append({key: a, key2: b})
    else:
        combos = [{key: v} for v in vals1]

    print(f"作业 {args.job}  方法 {args.method}  目标面数 {args.triangles}")
    print(f"扫描 {key}" + (f" × {args.sweep2.split('=')[0]}" if args.sweep2 else "")
          + f"  固定 {fixed}\n")

    rows = []
    for i, combo in enumerate(combos, 1):
        overrides = dict(fixed)
        overrides.update(combo)
        label = " ".join(f"{k}={v}" for k, v in combo.items())
        print(f"[{i}/{len(combos)}] {label} ...", flush=True)
        cmd = ["cmd.exe", "/c", str(ROOT / "scripts" / "gsflow_gpu.cmd"), "run",
               "--job-id", args.job, "--stages", "export", "--force", "export",
               "--set", "export.formats=mesh",
               "--set", f"export.mesh_method={args.method}",
               "--set", f"export.mesh_target_triangles={args.triangles}"]
        for k, v in overrides.items():
            cmd += ["--set", f"export.{k}={v}"]
        rc, out = run(cmd)
        if rc not in (0, 1):
            print(f"    导出返回码 {rc}")
        ok = "已导出 model_mesh" in out
        m = measure(args.job)
        m.update(combo)
        m["export_ok"] = ok
        rows.append(m)
        if m.get("flip_ratio") is not None:
            print(f"    面数 {m.get('triangles')}  翻转率 {m['flip_ratio'] * 100:.1f}%  "
                  f"mean_dot {m.get('mean_dot')}  "
                  f">90°边 {m.get('dihedral_gt90', 0) * 100:.1f}%")
        else:
            print("    **测量失败**")

    # 汇总
    print("\n=== 汇总（翻转率越低越好）===")
    hdr = list(combos[0].keys())
    print("  " + "  ".join(f"{h:>28s}" for h in hdr) +
          f"  {'面数':>8s}  {'翻转率':>8s}  {'mean_dot':>9s}")
    best = None
    for m in rows:
        line = "  " + "  ".join(f"{str(m.get(h)):>28s}" for h in hdr)
        line += f"  {str(m.get('triangles', '-')):>8s}"
        fr = m.get("flip_ratio")
        line += f"  {(f'{fr * 100:.1f}%' if fr is not None else '-'):>8s}"
        line += f"  {str(m.get('mean_dot', '-')):>9s}"
        print(line)
        if fr is not None and (best is None or fr < best[0]):
            best = (fr, m)
    if best:
        print(f"\n最佳: 翻转率 {best[0] * 100:.1f}%  at "
              + " ".join(f"{k}={best[1].get(k)}" for k in hdr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
