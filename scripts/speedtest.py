"""测速：对候选下载源做 8 秒采样，估算 MB/s，用于挑选最快的镜像。

用法：env\\venv\\Scripts\\python.exe scripts\\speedtest.py
"""

from __future__ import annotations

import ssl
import time
import urllib.request

CTX = ssl.create_default_context()
UA = {"User-Agent": "gsflow/0.1"}

TARGETS = [
    ("NVIDIA 官方 12.9.1", "https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_576.57_windows.exe"),
    ("清华 pypi", "https://pypi.tuna.tsinghua.edu.cn/simple/"),
    ("清华 anaconda", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/win-64/repodata.json"),
    ("中科大 anaconda", "https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/win-64/repodata.json"),
    ("腾讯 pypi", "https://mirrors.cloud.tencent.com/pypi/simple/"),
    ("华为 pypi", "https://repo.huaweicloud.com/repository/pypi/simple/"),
    ("github release(COLMAP)", "https://github.com/colmap/colmap/releases/download/3.12.0/colmap-x64-windows-cuda.zip"),
]


def sample(url: str, seconds: float = 8.0) -> tuple[float, int]:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
        start = time.time()
        total = 0
        while time.time() - start < seconds:
            chunk = r.read(1024 * 128)
            if not chunk:
                break
            total += len(chunk)
        elapsed = max(time.time() - start, 1e-6)
    return total / elapsed / 1024 / 1024, total


if __name__ == "__main__":
    for name, url in TARGETS:
        try:
            speed, got = sample(url)
            print(f"{name:<20} {speed:7.2f} MB/s   (采样 {got/1024/1024:.1f} MB)")
        except Exception as exc:
            print(f"{name:<20} FAIL {type(exc).__name__}: {exc}")
