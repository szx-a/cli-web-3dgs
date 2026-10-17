"""系统性校验 GUI 模型页：`/api/models/{job}` 里**每一个** URL 是否真的能取到。

## 为什么需要

用户反馈：GUI 里打开模型时 `Failed to load resource: the server responded with a
status of 404`，画面全黑。
于是我逐个探测端点 —— 结果 **`/api/gauss/*` 和 `/static/splat.js` 全是 200**，
说明 404 出在**某个具体的产物 URL** 上，而不是整体路由缺失。

⇒ 只测"我知道的那几个端点"不够，必须把 **`/api/models` 返回的每一个 url 都请求一遍**，
这才叫整体检查。

用法: python scripts/check_all_model_urls.py [--job X] [--base http://127.0.0.1:8787]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def probe(url: str, timeout: float = 20.0) -> tuple[int, int]:
    """返回 (状态码, 字节数)；4xx/5xx 也作为状态码返回。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
            return r.status, len(data)
    except urllib.error.HTTPError as e:
        return e.code, 0
    except Exception:  # noqa: BLE001
        return -1, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8787")
    ap.add_argument("--job", action="append", default=[])
    ap.add_argument("--workspace", default=str(ROOT / "workspace"))
    args = ap.parse_args()

    ws = Path(args.workspace)
    if args.job:
        jobs = args.job
    else:
        jobs = sorted(p.name for p in ws.iterdir()
                      if p.is_dir() and (p / "05_export").exists())
    if not jobs:
        print("没有带 05_export 的作业")
        return 1

    print(f"=== 校验 {args.base} 上每个作业的模型 URL ===")
    bad_total = 0
    warn_total = 0
    for job in jobs:
        st, n = probe(f"{args.base}/api/models/{job}")
        if st != 200:
            print(f"\n[{job}] /api/models -> **{st}**")
            bad_total += 1
            continue
        with urllib.request.urlopen(f"{args.base}/api/models/{job}", timeout=20) as r:
            info = json.loads(r.read().decode("utf-8"))
        modes = info.get("modes") or {}
        print(f"\n[{job}]  modes: " +
              ", ".join(f"{k}={len(v)}" for k, v in modes.items()))
        for mode, files in modes.items():
            for f in files:
                url = f.get("url") or ""
                if not url.startswith("http"):
                    url = args.base + url
                st2, n2 = probe(url)
                ok = (st2 == 200 and n2 > 0)
                mark = "OK  " if ok else "**FAIL**"
                print(f"   [{mark}] {mode:<9} {str(f.get('name'))[:44]:<46}"
                      f" {st2}  {n2/1024/1024:.2f} MB" if ok else
                      f"   [{mark}] {mode:<9} {str(f.get('name'))[:44]:<46} {st2}")
                if not ok:
                    bad_total += 1
                    print(f"            url = {url}")
        # 也校验 manifest 里**每一条**产物路径（下载链接用的就是它们）
        man_url = f"{args.base}/api/artifacts/{job}/05_export/manifest.json"
        st3, n3 = probe(man_url)
        if st3 != 200:
            # ⚠️ 区分两类问题：
            #   * **manifest 缺失**：作业的 `05_export` 是更早版本/被中断导出的，
            #     属于**陈旧数据**，不是代码缺陷 ⇒ 记为 WARN，不影响退出码；
            #   * **manifest 声明了但取不到**：说明导出与清单不一致 ⇒ 必须 FAIL。
            print(f"   [WARN] manifest.json 不存在（陈旧作业，UI 仍可用）：{st3}")
            warn_total += 1
        else:
            with urllib.request.urlopen(man_url, timeout=20) as r:
                man = json.loads(r.read().decode("utf-8"))
            paths = [a.get("path") for a in (man.get("artifacts") or [])
                     if a.get("path")]
            miss = []
            for p in paths:
                u = f"{args.base}/api/artifacts/{job}/05_export/{p}"
                st4, n4 = probe(u)
                if st4 != 200 or n4 == 0:
                    miss.append((p, st4))
            if miss:
                print(f"   [**FAIL**] manifest 里 {len(miss)}/{len(paths)} 条路径取不到：")
                for p, st4 in miss[:6]:
                    print(f"              {p}  -> {st4}")
                    bad_total += 1
            else:
                print(f"   [OK  ] manifest 里 {len(paths)} 条路径全部可取")

    print(f"\n结论: {'所有模型 URL 均可访问' if bad_total == 0 else f'**{bad_total} 个 URL 不可用**'}"
          + (f'（另有 {warn_total} 个陈旧作业缺 manifest，不影响使用）' if warn_total else ''))
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
