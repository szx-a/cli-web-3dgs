"""补充审计：GitHub 上另外两类容易"退化为文字"的写法。

1. **引用块（>`）里的 `$$` 块**：块内每一行都必须带 `>`，否则块被截断；
2. **加粗/斜体把公式包住**（`**$x$**`）：Markdown 的强调解析会先切分文本，跨标记的 `$` 容易失效；
3. `$` 与 `**` 交错（同一行里 `**` 出现奇数次且含 `$`）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    bad = 0
    in_fence = False
    i = 0
    while i < len(lines):
        ln = lines[i]
        no = i + 1
        if ln.strip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            # 1) 引用块里的 $$ 块：检查块内后续行是否都带 >
            if ln.lstrip().startswith(">") and "$$" in ln:
                depth = 0
                j = i
                while j < len(lines):
                    cur = lines[j]
                    depth += cur.count("$$")
                    if depth % 2 == 0 and j > i:
                        break
                    if j > i and not cur.lstrip().startswith(">"):
                        bad += 1
                        print(f"  [FAIL] 行 {j+1}: 引用块内的 $$ 块缺少 '>' 前缀（块会被截断）")
                        print(f"         {cur.strip()[:110]}")
                        break
                    j += 1
            # 2) 加粗/斜体包住公式
            if re.search(r"\*\*[^*]*\$[^*]*\*\*", ln) or re.search(r"(?<!\*)\*[^*]*\$[^*]*\*(?!\*)", ln):
                bad += 1
                print(f"  [FAIL] 行 {no}: 强调标记与公式交错（$ 可能失效）")
                print(f"         {ln.strip()[:110]}")
        i += 1

    if bad:
        print(f"\n共 {bad} 处可疑")
        return 1
    print(f"  {path.name} 引用块/强调与公式的交互检查通过（{len(lines)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
