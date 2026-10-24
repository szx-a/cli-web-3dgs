"""带**断点续传**的 COLMAP 下载器（本机 HTTPS 不稳定，必须能续）。

## 为什么要续传

本机到 GitHub 的 TLS 握手/长连接经常超时（AGENTS.md 也记录过 curl/Schannel 不可用）。
一次 400 MB 的下载中途断掉就前功尽弃，所以：

* 先写 `<file>.part`，**断点用 HTTP Range 续传**；
* 每 1 MB flush 一次，网络抖动时最多丢 1 MB；
* 最多重试 `--retries` 次（默认 8），每次重连都从 `.part` 大小继续；
* 支持 `--proxy`（如本机代理）与 `--insecure`。

用法:
    python scripts/download_colmap.py --tag 4.2.1 --asset colmap-x64-windows-cuda.zip
    python scripts/download_colmap.py --tag 4.2.1 --asset colmap-x64-windows-nocuda.zip
"""
from __future__ import annotations

import argparse
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/colmap/colmap/releases/download"


def _size(url: str, ctx: ssl.SSLContext, timeout: int) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gsflow"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return int(r.headers.get("Content-Length") or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="4.2.1")
    ap.add_argument("--asset", default="colmap-x64-windows-nocuda.zip")
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--proxy", default="")
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--mirrors", default="",
                    help="逗号分隔的前缀镜像（例 * → https://ghfast.top/*），失败时轮换")
    args = ap.parse_args()

    base_url = f"{BASE}/{args.tag}/{args.asset}"
    prefixes = [x.strip().rstrip("/") + "/" for x in args.mirrors.split(",") if x.strip()]
    prefixes = [""] + prefixes  # 第一个是直连
    url = base_url
    out = ROOT / "downloads" / f"colmap-{args.tag}-{args.asset}"
    part = out.with_suffix(out.suffix + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 10_000_000:
        print(f"  已存在：{out.name}（{out.stat().st_size/1e6:.1f} MB），不再下载")
        return 0

    ctx = ssl.create_default_context()
    if args.insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    if args.proxy:
        urllib.request.install_opener(
            urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})))

    total = 0
    for attempt in range(1, args.retries + 1):
        have = part.stat().st_size if part.exists() else 0
        url = prefixes[(attempt - 1) % len(prefixes)] + base_url
        try:
            if not total:
                # HEAD 在本机经常卡住 ⇒ 容许失败，改由 GET 的
                # Content-Range/Content-Length 学习总大小。
                try:
                    total = _size(url, ctx, min(args.timeout, 20))
                except Exception as exc:  # noqa: BLE001
                    print(f"  HEAD 失败（{type(exc).__name__}），改用 GET 探总大小")
                    total = 0
                print(f"  目标 {url}\n  大小 {total/1e6:.1f} MB；已有 {have/1e6:.1f} MB")
            if total and have >= total:
                break
            headers = {"User-Agent": "gsflow"}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=args.timeout, context=ctx) as r, \
                    open(part, "ab") as fh:
                if not total:
                    cr = r.headers.get("Content-Range") or ""
                    if "/" in cr:
                        try:
                            total = int(cr.split("/")[-1])
                        except ValueError:
                            total = 0
                    if not total and r.headers.get("Content-Length"):
                        total = have + int(r.headers["Content-Length"])
                if r.status == 200 and have:
                    # 服务器不支持 Range：从头写，避免拼接出脏文件
                    fh.close()
                    part.unlink(missing_ok=True)
                    raise RuntimeError("服务器忽略了 Range（将从头重下）")
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    have += len(chunk)
                    dt = max(time.time() - t0, 1e-6)
                    print(f"\r    {have/1e6:7.1f}/{total/1e6:7.1f} MB "
                          f"({have/max(total,1)*100:5.1f}%) {have/1e6/dt:5.1f} MB/s",
                          end="", flush=True)
            print()
            if total and have >= total:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"\n  第 {attempt}/{args.retries} 次中断：{type(exc).__name__}: {str(exc)[:100]}")
            time.sleep(min(2 * attempt, 10))

    have = part.stat().st_size if part.exists() else 0
    if not total or have < total:
        print(f"  下载未完成：{have/1e6:.1f}/{total/1e6:.1f} MB（保留 .part，可再次运行续传）")
        return 1
    part.replace(out)
    print(f"  完成：{out.name}（{out.stat().st_size/1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
