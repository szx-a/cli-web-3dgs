"""模型权重下载器（统一走 Python urllib，绕开 PowerShell/ultralytics 的 curl+Schannel 问题）。

为什么需要它：
  * 本机 curl 走 Schannel 取不到证书（SEC_E_NO_CREDENTIALS），ultralytics 内部用 curl 下载权重
    必然失败；
  * 而 Python 的 urllib 走 OpenSSL，实测正常。

用法：
    env\\venv\\Scripts\\python.exe scripts\\download_weights.py --list
    env\\venv\\Scripts\\python.exe scripts\\download_weights.py fastsam
    env\\venv\\Scripts\\python\\python.exe scripts\\download_weights.py all
"""

from __future__ import annotations

import argparse
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

CATALOG: dict[str, dict[str, str]] = {
    "fastsam": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt",
        "file": "FastSAM-s.pt",
        "desc": "FastSAM-s 分割权重（ultralytics 默认会从 GitHub 拉，本机 curl 拉不动）",
    },
    "fastsam-x": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt",
        "file": "FastSAM-x.pt",
        "desc": "FastSAM-x 分割权重（更大更准）",
    },
}


def ctx() -> ssl.SSLContext:
    return ssl.create_default_context()


def download(url: str, out: Path, name: str = "", chunk: int = 1024 * 128, retries: int = 60) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    if out.exists() and out.stat().st_size > 0:
        print(f"已存在，跳过：{out}（{out.stat().st_size/1024/1024:.1f} MB）")
        return out

    total = 0
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gsflow/0.1"})
        with urllib.request.urlopen(req, context=ctx(), timeout=30) as r:
            total = int(r.headers.get("Content-Length") or 0)
    except Exception:
        pass

    attempt = 0
    while attempt < retries:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:
            break
        headers = {"User-Agent": "gsflow/0.1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx(), timeout=90) as resp:
                if have and resp.status != 206:
                    have = 0
                mode = "ab" if have else "wb"
                t0 = last = time.time()
                with part.open(mode) as fh:
                    while True:
                        data = resp.read(chunk)
                        if not data:
                            break
                        fh.write(data)
                        have += len(data)
                        if time.time() - last > 3:
                            last = time.time()
                            speed = have / max(last - t0, 1e-6) / 1024 / 1024
                            pct = f"{have*100/total:5.1f}%" if total else "  ?  "
                            print(f"  {name or out.name}: {pct}  {have/1024/1024:7.1f}/{total/1024/1024:.1f} MB  {speed:5.2f} MB/s", flush=True)
        except Exception as exc:
            print(f"  第 {attempt} 次中断：{type(exc).__name__}: {exc}；3 秒后续传（已有 {have/1024/1024:.1f} MB）", flush=True)
            time.sleep(3)
            continue
        if not total or have >= total:
            break

    if total and part.stat().st_size < total:
        raise RuntimeError(f"下载未完成：{part.stat().st_size}/{total}")
    part.replace(out)
    print(f"完成：{out}（{out.stat().st_size/1024/1024:.1f} MB）")
    return out


def install(key: str) -> None:
    item = CATALOG[key]
    download(item["url"], MODELS / item["file"], item["desc"])


def main() -> int:
    ap = argparse.ArgumentParser(description="gsflow 模型权重下载器（Python urllib 版）")
    ap.add_argument("name", nargs="?", help="权重名：fastsam / fastsam-x / all")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.name:
        for k, v in CATALOG.items():
            print(f"{k:<12} {v['file']:<16} {v['desc']}")
        return 0

    if args.name == "all":
        for k in CATALOG:
            install(k)
        return 0
    if args.name not in CATALOG:
        print(f"未知权重 '{args.name}'，可用：{', '.join(CATALOG)}")
        return 2
    install(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
