"""校验 `/api/models/{job}` 声明的**每一个** URL 是否真的可下载。

为什么需要（这就是用户看到的 404 的真凶类别）：
查看器不是自己拼路径，而是**按 `/api/models` 返回的 url 去取**。
只要那个列表里有一个 URL 指向不存在的文件，浏览器控制台就会出现
`Failed to load resource: 404`，而用户看到的现象就是"查看器又黑了"。
静态资源（app.js/viewer.js/style.css）全是 200，与这个 404 无关。

用法: python scripts/probe_model_urls.py --job img2-2dgs-base [--all]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(base: str, job: str) -> tuple[int, int]:
    url = f"{base}/api/models/{job}"
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=30).read())
    except Exception as e:  # noqa: BLE001
        print(f"  **取 {url} 失败** {type(e).__name__}: {e}")
        return 0, 1
    urls: list[tuple[str, str]] = []
    for item in data.get("downloadable", []) or []:
        urls.append(("downloadable", item.get("url", "")))
    for mode, items in (data.get("modes") or {}).items():
        for item in items or []:
            urls.append((mode, item.get("url", "")))
    ok = bad = 0
    print(f"{job}: 声明 {len(urls)} 个 URL  complete={data.get('complete')} "
          f"gaussians={data.get('gaussians')}")
    for kind, u in urls:
        if not u:
            print(f"  [空 URL] {kind}")
            bad += 1
            continue
        full = base + u
        try:
            req = urllib.request.Request(full, method="GET")
            with urllib.request.urlopen(req, timeout=60) as r:
                n = len(r.read())
                print(f"  [OK]  {r.status}  {n:>10} 字节  {kind:10s} {u}")
                ok += 1
        except urllib.error.HTTPError as e:
            print(f"  [404] HTTP {e.code}  {kind:10s} {u}   <-- 浏览器就在这里报 404")
            bad += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] {type(e).__name__}  {kind:10s} {u}")
            bad += 1
    return ok, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--job", default=None)
    ap.add_argument("--all", action="store_true", help="检查所有作业")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    jobs = [args.job] if args.job else []
    if args.all or not jobs:
        try:
            listing = json.loads(urllib.request.urlopen(f"{base}/api/jobs", timeout=30).read())
            # ⚠️ `/api/jobs` 返回的是 {"running": [...], "history": [...]}。
            # 我第一版按 `jobs` 键取，静默拿到空列表 ⇒ 扫描"成功"但什么都没查
            # （典型的假阴性）。这里两种结构都认。
            items = []
            if isinstance(listing, dict):
                items = list(listing.get("history") or []) + list(listing.get("running") or [])
                if not items:
                    items = list(listing.get("jobs") or [])
            elif isinstance(listing, list):
                items = listing
            jobs = [j["job_id"] if isinstance(j, dict) else str(j) for j in items]
            print(f"（从 /api/jobs 取到 {len(jobs)} 个作业）\n")
        except Exception as e:  # noqa: BLE001
            print(f"取作业列表失败: {type(e).__name__}: {e}")
            return 1

    total_ok = total_bad = 0
    for job in jobs:
        o, b = check(base, job)
        total_ok += o
        total_bad += b
        print()
    print(f"合计: {total_ok} 个可下载, {total_bad} 个失败")
    if total_bad:
        print("**存在 404 -> 查看器会黑 / 控制台报 Failed to load resource**")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
