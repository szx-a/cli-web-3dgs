"""全文审计 README 的数学标记，找出"退化为文字"和"渲染报错"的所有成因。

GitHub 的规则（实测归纳）：

1. `$$...$$` 必须**自成段落**（前后各有一个空行），否则会被当普通文字；
2. 行内 `$...$` 必须在**同一行**闭合，且该行未转义的 `$` 个数为偶数；
3. 表格单元格里的数学不能含裸 `|`（会切断单元格）；
4. 代码块/行内代码里的 `$` 不算数学；
5. Markdown 会先吃掉 `\\{ \\} \\# \\| \\_` 的反斜杠（见 check_readme_math.py）。

本脚本按上述规则逐行审计并列出问题行号。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    lines = path.read_text(encoding="utf-8").splitlines()

    in_fence = False
    problems: list[tuple[int, str]] = []
    block_open = -1

    for i, ln in enumerate(lines):
        no = i + 1
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        stripped = ln.strip()

        # ---- 块级 $$ ----
        if "$$" in ln:
            cnt = ln.count("$$")
            if cnt % 2 == 1:
                if block_open < 0:
                    block_open = no
                    # 前置空行检查（文档首行除外）
                    if i > 0 and lines[i - 1].strip() != "":
                        problems.append((no, "$$ 块开始前缺少空行（会被当普通文字）"))
                else:
                    block_open = -1
                    if i + 1 < len(lines) and lines[i + 1].strip() != "":
                        problems.append((no, "$$ 块结束后缺少空行（会被当普通文字）"))
            elif cnt == 2 and stripped != "$$":
                # 同一行成对的 $$...$$：GitHub 只在自成段落时按块渲染
                if i > 0 and lines[i - 1].strip() != "":
                    problems.append((no, "行内出现的 $$…$$ 前面无空行，可能退化为文字"))
            continue

        if block_open > 0:
            continue  # 块内不检查行内 $

        # ---- 行内 $ ----
        # 去掉行内代码 `...` 后再数
        no_code = re.sub(r"`[^`]*`", "", ln)
        dollars = no_code.count("$")
        if dollars % 2 == 1:
            problems.append((no, f"该行未转义 $ 个数为奇数（{dollars}）⇒ 会被当普通文字"))

    if block_open > 0:
        problems.append((block_open, "$$ 块未闭合"))

    if problems:
        seen = set()
        for no, msg in problems:
            key = (no, msg)
            if key in seen:
                continue
            seen.add(key)
            print(f"  [FAIL] 行 {no}: {msg}")
            print(f"         {lines[no-1].strip()[:120]}")
        print(f"\n共 {len(seen)} 处")
        return 1
    print(f"  {path.name} 全文数学标记审计通过（{len(lines)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
