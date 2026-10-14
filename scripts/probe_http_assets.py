"""逐个请求 GUI 的每个前端资源，打印**真实状态码**，定位 404 到底出在哪个 URL。

为什么需要：用户看到 `Failed to load resource: 404`，但浏览器不显示是哪个 URL。
而"服务没在跑"表现为 ERR_CONNECTION_REFUSED（控制台里通常是 net::ERR_ 前缀），
与 404 是**两种不同的故障**，不能混为一谈。

用法: python scripts/probe_http_assets.py [--port 8787]
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    print("=== 1. 首页 ===")
    try:
        with urllib.request.urlopen(base + "/", timeout=15) as r:
            html = r.read().decode("utf-8", "replace")
            print(f"  {r.status}  {len(html)} 字节   Cache-Control={r.headers.get('Cache-Control')}")
    except urllib.error.HTTPError as e:
        print(f"  **HTTP {e.code}**  {e.reason}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"  **连接失败** {type(e).__name__}: {e}")
        return 1

    # 页面里引用的所有资源（含内联脚本里的动态 URL 片段）
    refs = set(re.findall(r'(?:src|href)="([^"]+)"', html))
    print(f"\n=== 2. 页面引用的资源（{len(refs)} 个）===")
    bad = 0
    for ref in sorted(refs):
        if ref.startswith(("http://", "https://", "data:", "#")):
            print(f"  [跳过] {ref}")
            continue
        url = base + (ref if ref.startswith("/") else "/" + ref)
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                body = r.read()
                print(f"  [OK]   {r.status}  {len(body):>8} 字节  {ref}")
        except urllib.error.HTTPError as e:
            print(f"  [404]  HTTP {e.code}  {ref}   <-- 这就是浏览器报的 404")
            bad += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR]  {type(e).__name__}: {e}  {ref}")
            bad += 1

    # 页面脚本里**动态拼接**的 URL（最容易 404 的一类）
    print("\n=== 3. 脚本里动态拼接的 URL 模板 ===")
    tmpl = set()
    for name in ("app.js", "viewer.js", "index.html"):
        p = ROOT / "src" / "gsflow" / "web" / "static" / name
        if not p.exists():
            print(f"  **磁盘缺少 {name}**")
            bad += 1
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        # 抓形如 `/api/xxx` 与 `${...}` 拼接的路径
        tmpl |= set(re.findall(r'["\'`](/api/[a-zA-Z0-9_\-/{}$.]*)', txt))
        tmpl |= set(re.findall(r'["\'`](/static/[a-zA-Z0-9_\-/{}$.]*)', txt))
    for t in sorted(tmpl):
        print(f"  {t}")

    # 用一个真实作业把动态 URL 展开后实测
    print("\n=== 4. 动态 URL 实测（用真实作业展开）===")
    jobs = sorted([p.name for p in (ROOT / "workspace").iterdir()
                   if p.is_dir() and (p / "05_export").exists()]) if (ROOT / "workspace").exists() else []
    if not jobs:
        print("  （workspace 里没有含 05_export 的作业）")
    else:
        job = jobs[-1]
        for path in (f"/api/models/{job}", f"/api/artifacts/{job}/05_export/model_mesh.glb",
                     f"/api/jobs", f"/api/config", f"/api/stages", f"/api/health",
                     f"/api/run/{job}/events"):
            url = base + path
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    body = r.read()
                    print(f"  [OK]   {r.status}  {len(body):>8} 字节  {path}")
            except urllib.error.HTTPError as e:
                print(f"  [404]  HTTP {e.code}  {path}   <-- 404")
                bad += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [ERR]  {type(e).__name__}  {path}")
                bad += 1

    print(f"\n结论: {'全部资源可访问' if bad == 0 else f'**{bad} 个资源 404/失败**'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
