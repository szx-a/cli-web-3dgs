"""把 `docs/03-踩坑记录.md` 里的 `### Gxxx.` 小节按编号重新排序。

## 为什么需要

我分两次追加小节（先写 G111/G112，再写 G109/G110），结果文件里的顺序是
G108 → G111 → G112 → G109 → G110，读起来跳跃。文档是给人看的，
**顺序本身就是信息**（G109 是 G111 的前置发现）。

本脚本只重排 `### G` 开头的同级小节，其它内容（前言、其它标题）保持不动；
并且**幂等**（已排好序时不做任何改动）。

用法: python scripts/resort_pitfalls_doc.py [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "03-踩坑记录.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DOC))
    ap.add_argument("--check", action="store_true", help="只检查是否有序，不写回")
    args = ap.parse_args()

    p = Path(args.file)
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # 找出所有 `### G<数字>.` 小节的起始行
    starts = [(i, m.group(1)) for i, line in enumerate(lines)
              if (m := re.match(r"### G(\d+)\.", line))]
    if len(starts) < 2:
        print("少于 2 个小节，无需排序")
        return 0

    first = starts[0][0]
    head = "".join(lines[:first])
    # 每个小节到"下一个 ### G 开头"或文件末尾
    blocks = []
    for k, (i, num) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        blocks.append((int(num), "".join(lines[i:end])))

    nums = [n for n, _ in blocks]
    ordered = sorted(blocks, key=lambda t: t[0])
    already = nums == [n for n, _ in ordered]
    print(f"当前顺序: {nums}")
    print(f"应有顺序: {[n for n, _ in ordered]}")
    if already:
        print("已有序，无需改动")
        return 0
    if args.check:
        print("**顺序不对**")
        return 1

    new_text = head + "".join(b for _, b in ordered)
    if not new_text.endswith("\n"):
        new_text += "\n"
    p.write_text(new_text, encoding="utf-8")
    print(f"已重排并写回 {p}（{len(blocks)} 个小节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
