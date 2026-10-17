"""校验一个已完成训练的作业**真的用了它配置的**训练策略。

## 为什么需要（G109 的教训）

`docs/03` G109 记的那个 bug：一个孤儿 `elif` 让 `grad2d` 每步算完就被 3D 梯度覆盖。
结果是"配置里写着 grad2d、日志里每步都在说拿不到 means2d、实际跑的是 grad3d"，
而 **PSNR 一样会照常输出**、验收也一样会通过 —— 这类"配置生效了但功能没执行"
是最难发现的一类问题，因为没有任何东西失败。

所以要有工具能回答：**这个作业，实际用的是哪种致密化判据？**

判据（都从产物里读，不靠日志里的自述）：
  1. `job.json` / `settings_snapshot` 里的 `train.densify_criterion`（用户**请求**的值）；
  2. `logs/events.jsonl` 里有没有"未取到 means2d 梯度"——有 ⇒ **实际退回了 grad3d**；
  3. 高斯数增长曲线：致密化若真的执行，末期高斯数应远大于初始化点数。

用法:
  python scripts/check_train_used_config.py --job img3-final
  python scripts/check_train_used_config.py --job img3-crop --job img3-final
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_events(job_dir: Path) -> list[str]:
    p = job_dir / "logs" / "events.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(str(json.loads(line).get("message", "")))
        except Exception:  # noqa: BLE001
            out.append(line)
    return out


def requested_criterion(job_dir: Path) -> str | None:
    """从 job.json 的 `settings_snapshot` / `spec.params` 里读**请求**的判据。

    ⚠️ 实际结构是 `job.json → settings_snapshot.{train.{...}}` 与
    `job.json → spec.params`（拍平的 `train.x` 键）。我第一版只找了
    `job.json/settings.json/params` 这些不存在的名字，于是永远读到 None
    —— 检查器看起来在跑，其实拿不到"配置请求"这一半信息。
    """
    p = job_dir / "job.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None

    cands: list[dict] = []
    snap = d.get("settings_snapshot")
    if isinstance(snap, dict):
        cands.append(snap)
        if isinstance(snap.get("train"), dict):
            cands.append(snap["train"])
    spec = d.get("spec")
    if isinstance(spec, dict) and isinstance(spec.get("params"), dict):
        cands.append(spec["params"])
    for k in ("params", "settings"):
        if isinstance(d.get(k), dict):
            cands.append(d[k])

    for holder in cands:
        for k in ("densify_criterion", "train.densify_criterion"):
            if k in holder:
                v = holder[k]
                # 拍平键的值可能是字符串也可能被包一层
                return str(v).strip().strip("'\"") or None
    return None


def analyze(job: str) -> dict:
    jd = ROOT / "workspace" / job
    rep: dict = {"job": job, "exists": jd.is_dir()}
    if not rep["exists"]:
        return rep
    msgs = load_events(jd)
    rep["events"] = len(msgs)
    rep["requested"] = requested_criterion(jd)

    # ---- 实际判据 ----
    fallback_warn = sum(1 for m in msgs if "未取到 means2d 梯度" in m)
    rep["fallback_warn"] = fallback_warn
    if fallback_warn > 0:
        rep["effective"] = "grad3d（**已退回**：拿不到 means2d 梯度）"
    elif rep["requested"] == "grad2d":
        rep["effective"] = "grad2d"
    elif rep["requested"] == "grad3d":
        rep["effective"] = "grad3d（配置要求）"
    else:
        rep["effective"] = "未知（日志里既没告警、也没读到配置）"

    # ---- 致密化是否真的执行 ----
    clones = [int(m.group(1)) for m in
              (re.search(r"克隆\s*(\d+)", x) for x in msgs) if m]
    splits = [int(m.group(1)) for m in
              (re.search(r"分裂\s*(\d+)", x) for x in msgs) if m]
    counts = [int(m.group(1)) for m in
              (re.search(r"->\s*(\d+)\s*个高斯", x) for x in msgs) if m]
    rep["densify_steps"] = len(clones)
    rep["clones_total"] = sum(clones)
    rep["splits_total"] = sum(splits)
    rep["gauss_first"] = counts[0] if counts else None
    rep["gauss_last"] = counts[-1] if counts else None
    metrics = jd / "04_train" / "metrics.json"
    rep["psnr"] = None
    rep["gaussians"] = None
    rep["iterations"] = None
    if metrics.exists():
        try:
            d = json.loads(metrics.read_text(encoding="utf-8"))
            rep["psnr"] = d.get("psnr")
            rep["gaussians"] = d.get("gaussians")
            rep["iterations"] = d.get("iterations")
        except Exception:  # noqa: BLE001
            pass

    problems = []
    if rep["events"] == 0:
        problems.append("没有 events.jsonl（训练未跑或日志缺失）")
    if fallback_warn > 0:
        problems.append(f"打出了 {fallback_warn} 次"
                        f"「未取到 means2d 梯度」⇒ 实际判据不是配置的 grad2d"
                        f"（这正是 G109 的症状）")
    if rep["densify_steps"] and rep["gauss_first"] and rep["gauss_last"]:
        growth = rep["gauss_last"] / max(rep["gauss_first"], 1)
        rep["growth"] = round(growth, 2)
        # ⚠️ 只在**训练足够长**时才判"致密化没生效"。
        # 我第一版无条件判，于是几十个 600~3000 迭代的**诊断作业**全被判失败
        # —— 那些作业本来就没跑到 densify_until，高斯少是正常的。
        # 判据用 metrics 里记录的 `iterations`（真实值），取不到就按 densify 步数估。
        iters = rep.get("iterations") or (rep["densify_steps"] * 1000)
        rep["long_enough"] = iters >= 8000
        if rep["long_enough"] and growth < 3.0 and rep["gauss_last"] < 30000:
            problems.append(f"训练 {iters} 迭代，高斯数只增长到 {rep['gauss_last']:,}"
                            f"（{growth:.1f}× 初始化）⇒ 致密化几乎没生效")
    rep["problems"] = problems
    rep["ok"] = not problems
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[])
    args = ap.parse_args()
    jobs = args.job or ["img3-final"]

    print("=== 训练策略是否**真的**生效 ===")
    bad = 0
    for job in jobs:
        r = analyze(job)
        print(f"\n[{job}]")
        if not r["exists"]:
            print("  **作业不存在**")
            bad += 1
            continue
        print(f"  配置请求 : {r['requested']}")
        print(f"  实际判据 : {r['effective']}")
        print(f"  退回告警 : {r['fallback_warn']} 次"
              + ("  **应为 0**" if r["fallback_warn"] else "  （正常）"))
        print(f"  致密化   : {r['densify_steps']} 次，克隆 {r['clones_total']:,}，"
              f"分裂 {r['splits_total']:,}")
        if r.get("gauss_last"):
            print(f"  高斯     : {r['gauss_first']:,} → {r['gauss_last']:,}"
                  f"（{r.get('growth')}×）")
        print(f"  PSNR     : {r['psnr']} dB，高斯 {r['gaussians']:,}"
              if r.get("gaussians") else f"  PSNR     : {r['psnr']}")
        if r["problems"]:
            bad += 1
            for p in r["problems"]:
                print(f"  [FAIL] {p}")
        else:
            print("  [OK  ] 配置与实际行为一致，致密化正常")
    print(f"\n结论: {'全部作业训练策略生效正确' if bad == 0 else f'**{bad} 个作业有问题**'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
