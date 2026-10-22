"""清理 `fix_ply_sorting.py` 替换模板留下的 `\\3` 字面量（幂等）。

替换模板末尾多了一个反向引用 `\\3`（正则里并不存在第三组），
于是代码变成 `... or 0))\\3`，语法错误 `unexpected character after line continuation`。

这里把「`or 0))` + 反斜杠 + `3`」这 7 个字符替换为「`or 0))`」。

用法: python scripts/fix_ply_sorting_cleanup.py [--check]
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDLE = "or 0))" + "\\" + "3"
REPL = "or 0))"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    fixed = []
    for d in ("scripts", "src", "tests"):
        for p in (ROOT / d).rglob("*.py"):
            try:
                t = p.read_text(encoding="utf-8")
            except OSError:
                continue
            if NEEDLE not in t:
                continue
            t2 = t.replace(NEEDLE, REPL)
            fixed.append(p.relative_to(ROOT))
            if not args.check:
                p.write_text(t2, encoding="utf-8")
    print(f"  清理 {len(fixed)} 个文件")
    for f in fixed:
        print(f"    {f}")

    bad = []
    for d in ("scripts", "src", "tests"):
        for p in (ROOT / d).rglob("*.py"):
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as e:
                bad.append((str(p.relative_to(ROOT)), e.lineno, e.msg))
    print(f"  语法错误: {bad if bad else '无'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
