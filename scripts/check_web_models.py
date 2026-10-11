"""通过 HTTP 验证 Web GUI 的新端点与页面资源。

检查项：
  1. /api/health 正常
  2. /api/models/{job} 能列出三种模式的产物（这是"模型查看"页的数据来源）
  3. 新前端资源可访问：/static/viewer.js、/static/app.js、/static/style.css
  4. 首页 HTML 里包含新页签（模型查看）且**不再包含说明页**

用法：env\\venv\\Scripts\\python.exe scripts\\check_web_models.py --port 8787
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(base: str, path: str, timeout: int = 30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.status, r.read(), r.headers.get("content-type", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--jobs", default="", help="只检查这些作业（逗号分隔）；默认取前几个")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    ok = True

    st, body, _ = get(base, "/api/health")
    h = json.loads(body.decode())
    print(f"[{'OK' if st == 200 and h.get('ok') else 'FAIL'}] /api/health -> {h.get('workspace')}")
    ok &= st == 200 and bool(h.get("ok"))

    # 前端资源
    for name, want in (("viewer.js", "application/javascript"),
                       ("app.js", "application/javascript"),
                       ("style.css", "text/css")):
        st, body, ct = get(base, f"/static/{name}")
        good = st == 200 and len(body) > 100
        print(f"[{'OK' if good else 'FAIL'}] /static/{name}  {len(body)} 字节  {ct}")
        ok &= good

    # 首页：新页签在、说明页不在
    st, body, _ = get(base, "/")
    html = body.decode("utf-8")
    has_models = 'data-tab="models"' in html
    has_about = 'data-tab="about"' in html
    has_panel_about = 'id="panel-about"' in html
    has_viewer_js = "/static/viewer.js" in html
    print(f"[{'OK' if has_models else 'FAIL'}] 首页含「模型查看」页签")
    print(f"[{'OK' if not has_about and not has_panel_about else 'FAIL'}] "
          f"首页已移除「说明」页（tab={has_about} panel={has_panel_about}）")
    print(f"[{'OK' if has_viewer_js else 'FAIL'}] 首页已引入 viewer.js")
    ok &= has_models and not has_about and not has_panel_about and has_viewer_js

    # 模型清单端点
    ws = Path(h.get("workspace") or (ROOT / "workspace"))
    if args.jobs:
        jobs = [s.strip() for s in args.jobs.split(",") if s.strip()]
    else:
        jobs = [p.name for p in sorted(ws.iterdir())
                if p.is_dir() and (p / "05_export").exists()]
    print(f"\n有 05_export 的作业 {len(jobs)} 个：{jobs[:6]}")
    if not jobs:
        print("（没有可验证的作业，跳过 /api/models 检查）")
        return 0 if ok else 1
    for job in jobs[:4]:
        try:
            st, body, _ = get(base, f"/api/models/{job}")
            d = json.loads(body.decode())
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] /api/models/{job} -> {type(exc).__name__}: {exc}")
            ok = False
            continue
        cnt = d.get("counts", {})
        print(f"[{'OK' if st == 200 else 'FAIL'}] /api/models/{job} "
              f"manifest={d.get('manifest')} complete={d.get('complete')} "
              f"counts={cnt} note={str(d.get('note',''))[:44]}")
        for mode, items in (d.get("modes") or {}).items():
            for it in items:
                print(f"        [{mode:8s}] {it['name']:32s} {it['size_mb']:>7} MB  "
                      f"{it.get('desc','')[:34]}")
        ok &= st == 200 and isinstance(cnt, dict)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
