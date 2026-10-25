"""一条命令跑完**全部质量体检**，输出汇总报告（文本 + JSON）。

## 为什么需要

质量这件事被反复拆成一个个小工具（掩码松紧、白边、彩度、实心度、网格封闭、LPIPS…），
排查时要在多个脚本间来回切、手动对数字 —— 这一步很容易漏看某一项，
而"漏看的那项"往往正是问题所在（本项目多次如此：白边 0.28、指标全绿的网格却是空壳）。

本脚本把已有检查器**按固定顺序**跑一遍，汇总成一个报告：

| 检查 | 回答的问题 | 依赖 |
|---|---|---|
| `check_gaussian_solidity` | 模型是否"朦胧/透明" | CPU |
| `check_mesh_quality` | 网格是否封闭、有没有飞边/非流形 | CPU |
| `check_rim_brightness` | 轮廓有没有白色光晕 | GPU（渲染）|
| `render_orbit_sheet` | 有没有彩虹斑块、观感如何 | GPU（渲染）|
| `probe_lpips_new` | 感知指标（同一把尺子横比）| GPU |

用法:
    python scripts/check_quality_suite.py --job img5-v1
    python scripts/check_quality_suite.py --job img5-v1 --out env/tmp/q.json --no-gpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(script: str, args: list[str], timeout: int = 1800) -> dict:
    path = ROOT / "scripts" / script
    if not path.exists():
        return {"ok": False, "skipped": True, "reason": f"缺少脚本 {script}"}
    cmd = [PY, str(path)] + args
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        tail = "\n".join((r.stdout or "").splitlines()[-25:])
        return {"ok": r.returncode == 0, "returncode": r.returncode, "tail": tail,
                "stderr": (r.stderr or "")[-800:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"超时（{timeout}s）"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-gpu", action="store_true", help="跳过需要渲染的检查（GPU 忙时用）")
    ap.add_argument("--views", type=int, default=6)
    args = ap.parse_args()

    jd = ROOT / "workspace" / args.job
    if not jd.exists():
        print(f"作业不存在：{jd}")
        return 1

    report: dict = {"job": args.job, "checks": {}}

    # 1) 高斯实心度（CPU）
    report["checks"]["solidity"] = _run(
        "check_gaussian_solidity.py", ["--jobs", args.job])

    # 2) 网格体检（CPU）—— 优先用 05_export 里的产物
    ex = jd / "05_export"
    meshes = sorted([p for p in ex.glob("model_mesh*.ply")] + [p for p in ex.glob("model_mesh*.obj")])
    if meshes:
        # 只取最大面数的那个（避免 obj/ply 重复统计）
        report["checks"]["mesh"] = _run(
            "check_mesh_quality.py", ["--obj", str(meshes[0])])
    else:
        report["checks"]["mesh"] = {"ok": False, "skipped": True, "reason": "没有网格产物"}

    # 3) 需要渲染的项（GPU）
    if args.no_gpu:
        for name in ("rim", "orbit", "lpips"):
            report["checks"][name] = {"ok": False, "skipped": True, "reason": "--no-gpu"}
    else:
        report["checks"]["rim"] = _run(
            "check_rim_brightness.py", ["--jobs", args.job, "--views", str(args.views)])
        report["checks"]["orbit"] = _run(
            "render_orbit_sheet.py", ["--job", args.job],
            timeout=3600)
        report["checks"]["lpips"] = _run(
            "probe_lpips_new.py", ["--jobs", args.job, "--views", str(args.views)],
            timeout=3600)

    # ---- 汇总 ----
    print(f"=== 质量体检汇总：{args.job} ===")
    for name, res in report["checks"].items():
        if res.get("skipped"):
            print(f"  [跳过] {name}: {res.get('reason', '')}")
            continue
        print(f"  [{'OK ' if res.get('ok') else 'FAIL'}] {name}")
        for line in (res.get("tail") or "").splitlines():
            if line.strip():
                print("      " + line.strip())

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出 {out}")

    bad = [k for k, v in report["checks"].items() if not v.get("ok") and not v.get("skipped")]
    print(f"\n结论：{len(bad)} 项失败" + (f"（{', '.join(bad)}）" if bad else "，全部通过"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
