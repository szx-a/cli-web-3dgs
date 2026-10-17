"""在 `docs/03-踩坑记录.md` **末尾**追加一段（幂等：已存在同样的标题就跳过）。

为什么写脚本而不是手工 edit：
文档已 2600+ 行，`edit` 的锚点在长文档里容易**匹配到多处**（刚刚就被拒了一次），
而"手填行号插入"又依赖当时的行数。追加到末尾是最稳的做法，且可重复执行。

用法: python scripts/append_doc_section.py --file docs/03-踩坑记录.md --marker "### G84." --content-file env/tmp/g84.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--marker", required=True, help="用于判断是否已追加过（如 '### G84.'）")
    ap.add_argument("--content-file", required=True)
    args = ap.parse_args()

    target = ROOT / args.file
    body = (ROOT / args.content_file).read_text(encoding="utf-8")
    text = target.read_text(encoding="utf-8")

    if args.marker in text:
        print(f"已存在 {args.marker!r}，跳过（幂等）")
        return 0

    # 去掉末尾多余空行，保证只留一个分隔
    text = text.rstrip("\n") + "\n\n" + body.strip("\n") + "\n"
    target.write_text(text, encoding="utf-8")
    print(f"已追加 {len(body.splitlines())} 行到 {target}（现有 {text.count(chr(10)) + 1} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
