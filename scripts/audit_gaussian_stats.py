"""审计训练出的高斯参数分布：找出"灰蒙/光晕/细节糊"的直接原因。

用户反馈："点云质量依旧不高，网络和纹理简直是灾难"。
从渲染对比看，症状是：**整体灰蒙、细节糊、边界有光晕** ——
这三者通常对应高斯参数分布里的具体病态，可以量化：

  * **尺度偏大** ⇒ 每个高斯覆盖很多像素 ⇒ 细节被平均掉 ⇒ 糊。
    判据：尺度中位数相对场景尺度的比例；以及"最大尺度/中位尺度"的离散度。
  * **半透明浮点** ⇒ 大量低不透明度高斯叠加成"雾" ⇒ 灰蒙。
    判据：不透明度分位数分布，特别是 p50 与"不透明度 < 0.05 的占比"。
  * **数量不足** ⇒ 细节无法表达 ⇒ 糊。
    判据：每视图高斯数（社区通常 1000~5000/视图）。

用法: scripts\\gpu_python.cmd scripts\\audit_gaussian_stats.py --job final-v1
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
    ap.add_argument("--job", default="final-v1")
    ap.add_argument("--compare", default="", help="逗号分隔的对照作业")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    from gsflow.gsdata.shio import read_gaussians_full
    from gsflow.stages.export import ExportStage

    jobs = [args.job] + [j.strip() for j in args.compare.split(",") if j.strip()]
    rows = []
    for job in jobs:
        job_dir = ROOT / "workspace" / job
        try:
            ply = ExportStage._find_ply(job_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  [跳过] {job}: 找不到 ply（{e}）")
            continue
        g = read_gaussians_full(ply)
        sc = np.asarray(g["scales"], dtype=np.float64)          # 真实尺度
        op = np.asarray(g["opacity"], dtype=np.float64)
        xyz = np.asarray(g["xyz"], dtype=np.float64)

        # 场景尺度：包围盒对角线
        span = xyz.max(0) - xyz.min(0)
        scene = float(np.linalg.norm(span))
        smax = sc.max(axis=1)
        rows.append({
            "job": job,
            "n": g["n"],
            "scene": scene,
            "s_p50": float(np.median(smax)),
            "s_p90": float(np.quantile(smax, 0.90)),
            "s_p99": float(np.quantile(smax, 0.99)),
            "s_max": float(smax.max()),
            "s_p50_rel": float(np.median(smax) / scene) * 100,
            "op_p10": float(np.quantile(op, 0.10)),
            "op_p50": float(np.median(op)),
            "op_p90": float(np.quantile(op, 0.90)),
            "op_lt005": float((op < 0.05).mean()) * 100,
            "op_lt02": float((op < 0.2).mean()) * 100,
        })

    if not rows:
        print("没有可审计的作业")
        return 1

    print("=== 高斯参数分布审计 ===\n")
    hdr = (f"{'作业':<14}{'高斯数':>9}{'场景尺度':>10}{'s_p50':>10}{'s_p90':>10}"
           f"{'p50/场景':>10}{'op_p50':>9}{'op<0.05':>9}{'op<0.2':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['job']:<14}{r['n']:>9}{r['scene']:>10.3f}{r['s_p50']:>10.5f}"
              f"{r['s_p90']:>10.5f}{r['s_p50_rel']:>9.3f}%{r['op_p50']:>9.4f}"
              f"{r['op_lt005']:>8.1f}%{r['op_lt02']:>7.1f}%")

    r = rows[0]
    print(f"\n=== 诊断（以 {r['job']} 为准）===")
    issues = []

    # 1) 细节能力：p50 尺度相对场景
    if r["s_p50_rel"] > 0.5:
        issues.append(f"尺度中位数占场景 {r['s_p50_rel']:.2f}%（偏大）"
                      f"⇒ 每个高斯覆盖很多像素，细节被平均掉")
    else:
        print(f"  [OK] 尺度中位数占场景 {r['s_p50_rel']:.3f}%（较小，细节能力尚可）")

    # 2) 半透明浮点
    if r["op_lt005"] > 20:
        issues.append(f"不透明度 < 0.05 的占 {r['op_lt005']:.1f}%"
                      f" ⇒ 大量近透明高斯叠加成'雾'，造成灰蒙")
    else:
        print(f"  [OK] 近透明高斯（<0.05）占 {r['op_lt005']:.1f}%（不多）")

    # 3) 数量
    per_view = r["n"] / 182.0
    print(f"  [信息] 每视图高斯 {per_view:.0f} 个"
          f"（社区 3DGS 同规模场景通常 1000~5000）")
    if per_view < 1000:
        issues.append(f"每视图仅 {per_view:.0f} 个高斯（社区 1000~5000）⇒ 细节表达不足")

    # 4) 尺度离散度：p99/p50 过大说明有少数超大高斯（光晕来源）
    ratio = r["s_p99"] / max(r["s_p50"], 1e-12)
    if ratio > 50:
        issues.append(f"尺度 p99/p50 = {ratio:.0f}（远大于 50）"
                      f" ⇒ 存在少数超大高斯，会在边界产生光晕")
    else:
        print(f"  [OK] 尺度 p99/p50 = {ratio:.1f}（离散度正常）")

    if issues:
        print(f"\n  发现 {len(issues)} 个问题：")
        for i, s in enumerate(issues, 1):
            print(f"    {i}. {s}")
    else:
        print("\n  未发现参数分布层面的明显病态。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
