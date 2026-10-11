"""抓取 GitHub 上的参考实现（本机 github.com 可达、raw.githubusercontent 不可达）。

网络现状（实测）：
  * `github.com/...`            -> OK（200）
  * `api.github.com/...`        -> OK（200）
  * `raw.githubusercontent.com` -> SSL UNEXPECTED_EOF（不通）
所以一律走 **GitHub API / 网页**，不要用 raw 域名。

用法：
  env\\venv\\Scripts\\python.exe scripts\\fetch_gh.py --repo nerficg-project/faster-gaussian-splatting
  env\\venv\\Scripts\\python.exe scripts\\fetch_gh.py --repo X --file path/in/repo.py
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}


def api(url: str, timeout: int = 40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="ignore"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--file", default="", help="抓单个文件（留空则先列树）")
    ap.add_argument("--out", default="env/tmp/gh")
    ap.add_argument("--max-files", type=int, default=0, help=">0 时批量抓取匹配的文件")
    ap.add_argument("--match", default="", help="批量抓取时的文件名子串")
    args = ap.parse_args()
    out = Path(args.out) / args.repo.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)

    info = api(f"https://api.github.com/repos/{args.repo}")
    print(f"{args.repo}: {info.get('description','')}")
    print(f"  默认分支 {info.get('default_branch')}  stars {info.get('stargazers_count')}")

    tree = api(f"https://api.github.com/repos/{args.repo}/git/trees/"
               f"{info.get('default_branch')}?recursive=1")
    files = [t for t in tree.get("tree", []) if t.get("type") == "blob"]
    print(f"  文件数 {len(files)}")
    py = [f["path"] for f in files if f["path"].endswith((".py", ".md", ".cu", ".cuh"))]
    print("  代码/文档文件（前 40）：")
    for p in py[:40]:
        print("    ", p)

    targets: list[str] = []
    if args.file:
        targets = [args.file]
    elif args.max_files > 0 and args.match:
        targets = [p for p in py if args.match in p][: args.max_files]

    for path in targets:
        try:
            blob = api(f"https://api.github.com/repos/{args.repo}/contents/{path}")
            raw = base64.b64decode(blob["content"])
            dst = out / Path(path).name
            dst.write_bytes(raw)
            print(f"[OK] {path}  {len(raw)} 字节 -> {dst}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {path}: {type(exc).__name__}: {str(exc)[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
