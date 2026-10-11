"""检查 Web 服务当前服务的前端资源是否是新代码（而不是被缓存/未重启的旧代码）。

动机（这次排查的起点）：GUI 里打开模型是**黑的**。可能原因之一是
Python 服务进程是**在改代码之前**启动的 —— 静态文件虽是从磁盘读，但：
  * 浏览器会缓存 `/static/*.js`
  * 若服务是旧进程，它读到的仍是磁盘上的新文件，但浏览器可能还用缓存
所以要先分清"是代码逻辑问题"还是"没生效"。

用法：env\\venv\\Scripts\\python.exe scripts\\check_served_assets.py --port 8787
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 这些标记只有"新代码"才有；缺失说明服务/浏览器拿到的是旧版本
MARKERS = {
    "viewer.js": ["uUseNrm", "aNrm", "nbuf"],
    "app.js": ["initModelsPage", "m-canvas", "GsViewer"],
    "index.html": ["data-tab=\"models\"", "m-canvas"],
    "style.css": ["m-canvas-wrap"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    ok = True
    for name, marks in MARKERS.items():
        url = "/" if name == "index.html" else f"/static/{name}"
        try:
            with urllib.request.urlopen(base + url, timeout=20) as r:
                body = r.read()
                hdr = dict(r.headers)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            ok = False
            continue
        disk = (ROOT / "src" / "gsflow" / "web" / "static" / name).read_bytes()
        same = hashlib.sha256(body).hexdigest() == hashlib.sha256(disk).hexdigest()
        missing = [m for m in marks if m.encode() not in body]
        flag = "OK" if (same and not missing) else "FAIL"
        print(f"[{flag}] {name:12s} 服务 {len(body):>7} B / 磁盘 {len(disk):>7} B  "
              f"内容一致={same}  缺失标记={missing or '无'}")
        print(f"        Cache-Control: {hdr.get('cache-control', '(无)')}  "
              f"ETag: {hdr.get('etag', '(无)')}  Last-Modified: {hdr.get('last-modified','(无)')}")
        ok &= same and not missing
    print("\n结论:", "服务的前端资源就是最新代码" if ok
          else "服务返回的不是最新代码（重启服务 + 浏览器硬刷新 Ctrl+F5）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
