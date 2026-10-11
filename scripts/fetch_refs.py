"""抓取"Faster-GS / 3DGS 加速优化"相关材料，提取可操作的技术点。

为什么用脚本而不是直接抓：本机 **github.com 直连失败**（`web_fetch failed: TypeError`），
而项目一贯的做法是用 Python `urllib` 走可用的通道（见 AGENTS.md 的网络踩坑记录）。
PDF 也不行（`unsupported content type application/pdf`），所以抓 HTML/raw 文本。

用法：
  env\\venv\\Scripts\\python.exe scripts\\fetch_refs.py --out env\\tmp\\refs
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# 优先 raw（纯文本、无 JS），其次普通页面
URLS = {
    "faster-gs-notes.md":
        "https://raw.githubusercontent.com/zhaoyang97/Paper-Notes-en/refs/heads/main/"
        "docs/CVPR2026/3d_vision/faster-gs_analyzing_and_improving_gaussian_splatting_optimization.md",
    "faster-gs-notes-jsdelivr.md":
        "https://cdn.jsdelivr.net/gh/zhaoyang97/Paper-Notes-en@main/"
        "docs/CVPR2026/3d_vision/faster-gs_analyzing_and_improving_gaussian_splatting_optimization.md",
    "faster-gs-abs.html":
        "https://openaccess.thecvf.com/content/CVPR2026/html/"
        "Hahlbohm_Faster-GS_Analyzing_and_Improving_Gaussian_Splatting_Optimization_CVPR_2026_paper.html",
    "sad-gs.html": "https://vcai.mpi-inf.mpg.de/projects/SAD-GS/",
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def get(url: str, timeout: int = 45) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            ct = r.headers.get("content-type", "")
        if "pdf" in ct.lower():
            return None
        return raw.decode("utf-8", errors="ignore")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = re.sub(r"&nbsp;?", " ", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="env/tmp/refs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    got = 0
    for name, url in URLS.items():
        txt = get(url)
        if not txt:
            print(f"[MISS] {name}  <- {url[:78]}")
            continue
        if name.endswith(".html"):
            txt = strip_html(txt)
        p = out / name
        p.write_text(txt, encoding="utf-8")
        got += 1
        print(f"[OK]   {name}  {len(txt):,} 字符  <- {url[:70]}")
        # 顺手打印含关键词的行，便于立刻抓到要点
        keys = ("densif", "ssim", "opacity", "resolution", "learning rate", "lr ",
                "batch", "sparse", "compute", "memory", "speedup", "improve",
                "progressive", "sh_degree", "optimizer", "adam", "scale")
        hits = [ln.strip() for ln in txt.splitlines()
                if any(k in ln.lower() for k in keys) and len(ln.strip()) > 40]
        for ln in hits[:12]:
            print(f"        | {ln[:150]}")
    print(f"\n共抓到 {got}/{len(URLS)} 份；输出目录 {out}")
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
