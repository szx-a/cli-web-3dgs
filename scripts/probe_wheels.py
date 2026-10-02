"""一次性诊断脚本：查 gsplat 预编译轮子索引与 CUDA 下载可用性。

用法：env\\venv\\Scripts\\python.exe scripts\\probe_wheels.py
"""

from __future__ import annotations

import re
import ssl
import urllib.request

CTX = ssl.create_default_context()
UA = {"User-Agent": "gsflow/0.1"}


def get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def list_links(url: str, limit: int = 30) -> None:
    try:
        html = get(url)
    except Exception as exc:
        print(f"FAIL {url} -> {type(exc).__name__}: {exc}")
        return
    hrefs = re.findall(r'href="([^"]+)"', html)
    print(f"== {url}  ({len(html)} bytes, {len(hrefs)} links)")
    for h in hrefs[:limit]:
        print("   ", h)


if __name__ == "__main__":
    for url in (
        "https://docs.gsplat.studio/whl/",
        "https://docs.gsplat.studio/whl/torch2.11.0/cu128/gsplat/",
        "https://docs.gsplat.studio/whl/torch2.9.0/cu128/gsplat/",
    ):
        list_links(url)
    for url in (
        "https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_576.57_windows.exe",
        "https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe",
        "https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_581.15_windows.exe",
    ):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=UA)
            with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
                print("OK  ", r.status, r.headers.get("Content-Length"), url)
        except Exception as exc:
            print("FAIL", type(exc).__name__, exc, url)
