"""逐个直接请求 `/api/opacity_cleanup`，排除检查器自己骗自己。

背景：`check_server_code_sync.py` 先报"缺 /api/doctor"（假警报），
修掉后报"全部可用"，但我此前直接请求 `/api/opacity_cleanup` 得到过 **404**。
两个结论矛盾 ⇒ 必须用**最原始**的方式复核，而不是依赖任何封装好的检查器。

本脚本只做一件事：对每个相关 URL 打印状态码与响应体前 120 字节，
不做任何"应该是 200"的假设。

用法: python scripts/probe_route_status.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8787"

PATHS = [
    "/api/health",
    "/api/stages",
    "/api/opacity_cleanup",
    "/api/jobs",
    "/api/config",
    "/static/app.js",
]


def main() -> int:
    print(f"目标 {BASE}\n")
    bad = 0
    for p in PATHS:
        try:
            with urllib.request.urlopen(BASE + p, timeout=10) as r:
                code, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, body = e.code, b""
        except Exception as e:  # noqa: BLE001
            code, body = None, str(e).encode()

        snippet = body[:120].decode("utf-8", "replace").replace("\n", " ")
        extra = ""
        if p == "/api/opacity_cleanup" and code == 200:
            try:
                d = json.loads(body)
                extra = f"  方案={[c['value'] for c in d.get('choices', [])]}"
            except Exception:  # noqa: BLE001
                extra = "  （响应不是预期 JSON）"
        flag = "OK  " if code == 200 else "**FAIL**"
        print(f"  [{flag}] {p:<26} {code}  {snippet}{extra}")
        if code != 200:
            bad += 1
    print(f"\n结论: {len(PATHS) - bad}/{len(PATHS)} 个 URL 返回 200")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
