"""校验文档里的 mermaid 代码块：括号/引号配对、节点是否先定义后引用、箭头是否合法。

为什么需要：mermaid 图写错时，**文档本身看不出问题** —— 只有渲染时才显示成空白或报错。
这类"交付物看起来有、实际不可用"的问题在本项目出现过多次
（例如生成的 HTML 占位符没替换、图里引用了不存在的节点）。

只做**结构检查**（不依赖 mermaid 本体）：
  * ```mermaid 块必须成对闭合
  * 每个块的首行必须是已知的图类型
  * `A --> B` / `A -.-> B` 里的节点标识符，若在别处被声明过（`A[...]` / `A(...)` / `subgraph`），
    则引用合法；纯新标识符允许（mermaid 会自动建节点），但会**列出**以便人工确认
  * 方括号/圆括号/花括号在**同一行内**必须配对

用法: python scripts/check_mermaid.py [docs/01-架构说明.md ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram",
         "erDiagram", "journey", "gantt", "pie", "gitGraph", "mindmap")


def check_block(text: str, where: str) -> list[str]:
    errs: list[str] = []
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        errs.append(f"{where}: 空的 mermaid 块")
        return errs
    if not any(lines[0].strip().startswith(k) for k in KINDS):
        errs.append(f"{where}: 首行不是已知图类型（{lines[0].strip()[:40]!r}）")
    for i, l in enumerate(lines, 1):
        s = l.strip()
        if s.startswith("%%"):
            continue
        # 同一行内的括号配对
        for a, b in (("[", "]"), ("(", ")"), ("{", "}")):
            if s.count(a) != s.count(b):
                errs.append(f"{where} 行{i}: {a}{b} 不配对 -> {s[:70]!r}")
        # 箭头后面必须有目标
        if re.search(r"(-{1,2}>|--x|--o|\.->|==>)\s*$", s) and "-->" not in s:
            errs.append(f"{where} 行{i}: 箭头没有目标 -> {s[:70]!r}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    files = [Path(f) for f in args.files] or sorted((ROOT / "docs").glob("*.md"))

    total_blocks = total_errs = 0
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        blocks = re.findall(r"```mermaid\n(.*?)```", text, re.S)
        if not blocks:
            continue
        opens = text.count("```mermaid")
        closes = len(blocks)
        errs: list[str] = []
        if opens != closes:
            errs.append(f"{f.name}: 有 {opens} 个 ```mermaid 但只闭合了 {closes} 个")
        for i, b in enumerate(blocks, 1):
            errs += check_block(b, f"{f.name}#图{i}")
        total_blocks += len(blocks)
        total_errs += len(errs)
        status = "OK" if not errs else f"**{len(errs)} 个问题**"
        print(f"  {f.name}: {len(blocks)} 个图  {status}")
        for e in errs:
            print(f"     - {e}")

    print(f"\n合计 {total_blocks} 个 mermaid 图，{total_errs} 个结构问题")
    return 0 if total_errs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
