"""探测可用的第三方镜像/预编译源（为绕开编译或加速下载找备选）。

用法：env\\venv\\Scripts\\python.exe scripts\\probe_mirrors.py
"""

from __future__ import annotations

import re
import ssl
import urllib.request

CTX = ssl.create_default_context()
UA = {"User-Agent": "gsflow/0.1"}

URLS = [
    "https://docs.gsplat.studio/whl/gsplat/",
    "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/",
    "https://mirrors.aliyun.com/pytorch-wheels/",
    "https://mirror.sjtu.edu.cn/pytorch-wheels/",
    "https://mirrors.bfsu.edu.cn/pytorch-wheels/",
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/win-64/",
]


def get(url: str, timeout: int = 25) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "ignore")


def main() -> int:
    for url in URLS:
        try:
            status, html = get(url)
        except Exception as exc:
            print(f"FAIL {url}\n     {type(exc).__name__}: {exc}")
            continue
        links = re.findall(r'href="([^"]+)"', html)
        interesting = [l for l in links if re.search(r"cu12|cp31|gsplat|torch-", l)][:8]
        print(f"OK   {url}  ({len(links)} 链接)")
        for l in interesting:
            print("      ", l)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
