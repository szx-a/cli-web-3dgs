"""检查文档里 `$` / `$$` 是否配对（跨行状态机，先剔除代码）。

之前版本按行数 `$$` 个数，把**跨行块级公式**误判成不配对；也没剔除行内代码里
的 `$`（例如 `` `$T` ``）。这里按顺序处理：

1. 围栏代码块整段跳过；
2. 行内代码 `` `...` `` 用空格遮住；
3. 用状态机跟踪 `$$`（块级）——只有在块外的行才检查行内 `$` 的奇偶。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(r"F:\3dgs-and-ai")


def scan(path: Path) -> list[tuple[int, str, str]]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    problems: list[tuple[int, str, str]] = []
    fence = None
    in_block = False
    block_start = 0

    for i, line in enumerate(raw_lines, start=1):
        clean = line.rstrip("\r")
        fm = re.match(r"^\s*(`{3,}|~{3,})([^`]*)$", clean)
        if fence:
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= fence[1] and not fm.group(2).strip():
                fence = None
            continue
        if fm:
            fence = (fm.group(1)[0], len(fm.group(1)))
            continue
        if re.match(r"^\s{4,}", clean):
            continue

        def _mask_code(s: str) -> str:
            # 先遮双反引号片段（``...``），再遮单反引号（`...`）
            s = re.sub(r"``[^`\n]+``", lambda m: " " * len(m.group(0)), s)
            return re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), s)

        masked = _mask_code(clean)
        # 块级状态：一行里 $$ 出现奇数次 ⇒ 切换
        n = masked.count("$$")
        if n % 2 == 1:
            if not in_block:
                in_block = True
                block_start = i
            else:
                in_block = False
            # 去掉 $$ 后，块外/块内都再检查剩余的行内 $
        if in_block:
            continue  # 块内不检查行内
        rest = masked.replace("$$", "")
        if rest.count("$") % 2 == 1:
            problems.append((i, "行内 $ 不配对", clean.strip()[:100]))

    if in_block:
        problems.append((block_start, "块级 $$ 未闭合", raw_lines[block_start - 1].strip()[:100]))
    return problems


def main() -> int:
    targets = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
    total = 0
    for p in targets:
        probs = scan(p)
        if probs:
            print(f"=== {p.relative_to(ROOT)} ===")
            for no, kind, txt in probs[:15]:
                print(f"  行 {no}: {kind}   {txt}")
            total += len(probs)
    print(f"\n共 {total} 处" if total else "\n所有文档 $ / $$ 都配对 ✔")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
