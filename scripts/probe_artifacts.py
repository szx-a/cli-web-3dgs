"""对每个作业，把 `/api/jobs/{job}` 返回的**全部产物路径**逐个请求一遍。

这是用户看到 404 的第三类可能来源（前两类已排除）：
  * 静态资源 app.js/viewer.js/style.css —— 实测全 200；
  * `/api/models/{job}` 声明的下载 URL —— 实测 60 个全 200；
  * **作业详情里的产物列表**（app.js 行 225 用它渲染成 <a href>，点击即请求）
    —— 本脚本查这一类。

用法: python scripts/probe_artifacts.py [--job JOB] [--all]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jobs_of(base: str) -> list[str]:
    d = json.loads(urllib.request.urlopen(f"{base}/api/jobs", timeout=30).read())
    if isinstance(d, dict):
        items = list(d.get("history") or []) + list(d.get("running") or [])
        if not items:
            items = list(d.get("jobs") or [])
    else:
        items = d
    return [j["job_id"] if isinstance(j, dict) else str(j) for j in items]


def check_job(base: str, job: str, verbose: bool) -> tuple[int, int, list[str]]:
    try:
        d = json.loads(urllib.request.urlopen(
            f"{base}/api/jobs/{urllib.parse.quote(job)}", timeout=30).read())
    except Exception as e:  # noqa: BLE001
        return 0, 1, [f"{job}: 取详情失败 {type(e).__name__}"]
    arts = d.get("artifacts") or d.get("outputs") or []
    ok = bad = 0
    problems: list[str] = []
    for a in arts:
        rel = str(a).replace("\\\\", "/").replace("\\", "/")
        url = f"{base}/api/artifacts/{job}/{urllib.parse.quote(rel)}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                r.read(64)
                ok += 1
                if verbose:
                    print(f"  [OK]  {r.status}  {rel}")
        except urllib.error.HTTPError as e:
            bad += 1
            problems.append(f"{job}  HTTP {e.code}  {rel}")
            print(f"  [404] HTTP {e.code}  {job}  {rel}   <-- 浏览器点击即 404")
        except Exception as e:  # noqa: BLE001
            bad += 1
            problems.append(f"{job}  {type(e).__name__}  {rel}")
    return ok, bad, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--job", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    jobs = [args.job] if args.job else jobs_of(base)
    print(f"检查 {len(jobs)} 个作业的产物链接\n")
    tot_ok = tot_bad = 0
    allp: list[str] = []
    for j in jobs:
        o, b, p = check_job(base, j, args.verbose)
        tot_ok += o
        tot_bad += b
        allp += p
        if b and not args.verbose:
            print(f"  {j}: {o} 可访问 / **{b} 个 404**")
    print(f"\n合计: {tot_ok} 个可访问, {tot_bad} 个 404")
    for p in allp[:20]:
        print(f"  {p}")
    return 0 if tot_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
