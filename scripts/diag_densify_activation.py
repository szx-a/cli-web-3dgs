"""从训练日志还原**致密化激活率**曲线，判断阈值是否标定得合适。

## 为什么要看激活率

`img3-final`（406 视图、30k 迭代、grad2d）只得到 **94,598** 个高斯。
社区同规模素材通常在几十万到百万级。低一个数量级的常见原因只有一个：
**致密化阈值偏高** —— 每步只有极少数高斯越过了 `grad > threshold`。

而"阈值合不合适"不该靠猜：日志里每一行密度控制都记了
`克隆 C / 分裂 S / 裁剪 P -> G 个高斯`，于是可以还原

    激活率 = (C + S) / 上一步的高斯数

* 官方 3DGS 的典型形态是：早期激活率高（**20%~60%**，快速长出细节），
  随迭代衰减到 1%~5%（收敛）。整段曲线几乎贴地（<2%）⇒ 阈值太高；
  长期高居不下（>80%）⇒ 阈值太低、且在原地反复克隆。
* 这个指标**只需要日志**，不用重训，所以可以作为"要不要调阈值"的第一道判据。

## 实测用途

对比 `img3-final`（grad2d 已真正生效）与 `img3-crop`（grad2d 被 G109 静默退回成 grad3d）
的激活率曲线，就能看出两种判据在**同样数据、同样迭代**下选中的高斯比例差多少 ——
这是"判据差别"的直接证据，而不用去比 PSNR（那样还会被裁剪框差异混淆）。

用法:
  python scripts/diag_densify_activation.py --job img3-final --job img3-crop
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DENS_RE = re.compile(
    r"step\s+(\d+)\s+密度：阈值=([\d.eE+-]+)\s+"
    r"克隆\s+(\d+)\s+分裂\s+(\d+)\s+裁剪\s+(\d+)\s*->\s*(\d+)\s+个高斯")


def parse_job(job: str) -> dict:
    f = ROOT / "workspace" / job / "logs" / "events.jsonl"
    out = {"job": job, "rows": [], "exists": f.exists()}
    if not f.exists():
        return out
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if "密度" not in line:
            continue
        try:
            msg = json.loads(line).get("message", "")
        except Exception:  # noqa: BLE001
            msg = line
        m = DENS_RE.search(msg)
        if not m:
            continue
        step, thr, cl, sp, pr, tot = m.groups()
        out["rows"].append({"step": int(step), "threshold": float(thr),
                            "clone": int(cl), "split": int(sp),
                            "prune": int(pr), "total": int(tot)})
    return out


def summarize(rep: dict) -> dict:
    rows = rep["rows"]
    if len(rows) < 2:
        return {**rep, "ok": False, "note": "密度日志不足 2 条"}
    # 激活率需要"上一步的总数"；先用前一条的 total 近似（同量级的估计足够做趋势判断）
    acts = []
    for prev, cur in zip(rows, rows[1:]):
        base = max(prev["total"], 1)
        acts.append((cur["step"], (cur["clone"] + cur["split"]) / base))
    peak = max(a for _s, a in acts)
    early = [a for s, a in acts if s <= 3000]
    late = [a for s, a in acts if s >= 25000]
    return {
        "job": rep["job"], "steps": len(rows),
        "thr": sorted({r["threshold"] for r in rows}),
        "gauss_first": rows[0]["total"], "gauss_last": rows[-1]["total"],
        "peak_rate": peak,
        "early_mean": sum(early) / len(early) if early else None,
        "late_mean": sum(late) / len(late) if late else None,
        "clone_total": sum(r["clone"] for r in rows),
        "split_total": sum(r["split"] for r in rows),
        "prune_total": sum(r["prune"] for r in rows),
        "ok": True, "acts": acts,
        "rows": rows,
    }


def verdict(s: dict) -> str:
    """给出可执行的判断，而不是只报数字。"""
    if not s.get("ok"):
        return "无法判断（日志不足）"
    em, lm, pk = s["early_mean"], s["late_mean"], s["peak_rate"]
    if pk is None:
        return "无法判断"
    if pk < 0.05:
        return ("**阈值偏高**：峰值激活率 <5%，几乎没有高斯越过阈值 ⇒ "
                "考虑降低 `train.densify_grad2d_threshold`（如 1e-4）")
    if em is not None and em < 0.10:
        return ("**早期激活率偏低**（<10%）：细节可能没长出来 ⇒ "
                "可试更低的阈值，或延长 `densify_until`")
    if lm is not None and lm > 0.60:
        return ("**阈值偏低**：收敛期仍在大量克隆（>60%）⇒ "
                "加大阈值或提前结束致密化")
    return "激活率形态正常（早期高、后期衰减）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[])
    args = ap.parse_args()
    jobs = args.job or ["img3-final", "img3-crop"]

    print("=== 致密化激活率（从日志还原，无需重训）===")
    sums = []
    for job in jobs:
        s = summarize(parse_job(job))
        sums.append(s)
        print(f"\n[{job}]")
        if not s.get("ok"):
            print(f"  {s.get('note')}")
            continue
        print(f"  阈值        : {s['thr']}")
        print(f"  高斯        : {s['gauss_first']:,} → {s['gauss_last']:,}"
              f"（{s['gauss_last']/max(s['gauss_first'],1):.1f}×）")
        print(f"  克隆/分裂/裁剪 合计: {s['clone_total']:,} / "
              f"{s['split_total']:,} / {s['prune_total']:,}")
        print(f"  激活率 峰值 : {s['peak_rate']*100:.1f}%")
        if s["early_mean"] is not None:
            print(f"  激活率 早期 : {s['early_mean']*100:.1f}%（step<=3000）")
        if s["late_mean"] is not None:
            print(f"  激活率 收敛 : {s['late_mean']*100:.1f}%（step>=25000）")
        print(f"  判断        : {verdict(s)}")
        # 抽几条曲线点，便于看出形态
        pts = s["acts"]
        pick = [pts[i] for i in range(0, len(pts), max(1, len(pts) // 6))][:7]
        print("  曲线        : " + "  ".join(
            f"{st}:{a*100:.0f}%" for st, a in pick))

    # 判据对比（若同时给了受影响的作业）
    ok = [s for s in sums if s.get("ok")]
    if len(ok) >= 2:
        a, b = ok[0], ok[1]
        if a["gauss_last"] and b["gauss_last"]:
            print(f"\n=== 对比 {a['job']} vs {b['job']} ===")
            print(f"  最终高斯数之比 : {a['gauss_last']/b['gauss_last']:.2f}×")
            if a["early_mean"] and b["early_mean"]:
                print(f"  早期激活率之比 : "
                      f"{a['early_mean']/b['early_mean']:.2f}×")
            print("  注意：两个作业的**裁剪框可能不同**，只能看致密化行为的差别，"
                  "不能直接当质量结论（见 docs/03 G113）")

    out = ROOT / "workspace" / "densify_activation.json"
    out.write_text(json.dumps([{k: v for k, v in s.items() if k != "acts"}
                               for s in sums], ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
