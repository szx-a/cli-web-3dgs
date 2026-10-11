"""验证"模型查看"页真正会请求的每个 URL 都能下载、且格式正确。

为什么单独做这一步：前面只验证了 `/api/models` 返回的**清单**正确，
但查看器真正依赖的是那些 `url` 能被下载、且内容是该格式（不是 404 页/错误 JSON）。
这里逐个 GET 并校验文件签名（glTF magic / PNG 签名 / OBJ、PLY 文本头）。

用法：
  env\\venv\\Scripts\\python.exe scripts\\check_viewer_urls.py --job retest-01 --port 8787
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def sniff(name: str, raw: bytes) -> tuple[bool, str]:
    low = name.lower()
    if low.endswith(".glb"):
        ok = raw[:4] == b"glTF"
        return ok, "glTF" if ok else "bad-magic"
    if low.endswith(".png"):
        ok = raw[:8] == b"\x89PNG\r\n\x1a\n"
        return ok, "PNG" if ok else "bad-magic"
    if low.endswith(".ply"):
        head = raw[:64].lstrip()
        ok = head.startswith(b"ply")
        return ok, "PLY" if ok else "bad-header"
    if low.endswith(".obj"):
        head = raw[:200]
        ok = b"v " in head or b"# " in head
        return ok, "OBJ" if ok else "bad-header"
    if low.endswith(".spz"):
        return len(raw) > 100, "SPZ"
    return len(raw) > 0, "raw"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    with urllib.request.urlopen(f"{base}/api/models/{args.job}", timeout=30) as r:
        d = json.loads(r.read().decode("utf-8"))

    print(f"作业 {args.job}  complete={d.get('complete')}  counts={d.get('counts')}")
    all_ok = True
    for mode in ("points", "mesh", "textured"):
        items = (d.get("modes") or {}).get(mode) or []
        if not items:
            print(f"  [{mode}] （无产物）")
            continue
        for it in items:
            url = base + it["url"]
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    raw = r.read()
                    status = r.status
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {mode:8s} {it['name']:30s} {type(exc).__name__}: {exc}")
                all_ok = False
                continue
            good, tag = sniff(it["name"], raw)
            flag = "OK" if (good and status == 200) else "FAIL"
            all_ok &= good and status == 200
            print(f"  [{flag}] {mode:8s} {it['name']:30s} "
                  f"{len(raw)/1e6:7.2f} MB  {tag}  HTTP {status}")

    if d.get("viewer"):
        try:
            with urllib.request.urlopen(base + d["viewer"], timeout=30) as r:
                html = r.read()
            ok = b"<html" in html.lower() or b"<!doctype" in html.lower()
            print(f"  [{'OK' if ok else 'FAIL'}] viewer.html  {len(html)/1e6:.2f} MB")
            all_ok &= ok
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] viewer.html: {type(exc).__name__}: {exc}")
            all_ok = False

    print("\n结论:", "查看器所需的全部 URL 可下载且格式正确" if all_ok else "存在失败项")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
