r"""检查 README 里的数学能否被 **GitHub 的 MathJax** 渲染。

## 为什么需要（2026-10-26，用户两次报错后定位）

第一次报：

    Extra open brace or missing close brace
    You can't use 'macro parameter character #' in math mode

根因**不是 MathJax 宏支持**，而是 **Markdown 会先吃掉这些转义**：

| 写法 | Markdown 处理后 | TeX 收到 | 结果 |
|---|---|---|---|
| `\{` `\}` | 裸 `{` `}` | 未配对括号 | "Extra open brace..." |
| `\#` | 裸 `#` | `#` 是宏参数符 | "macro parameter character #" |
| `\|` | 裸 `|` | 表格里切断单元格；块级里退化成绝对值 | 公式截断 |
| `\_` | 裸 `_` | 变成下标 | 语法错误 |

安全写法：`\lbrace` / `\rbrace`、`\mathrm{card}`、`\Vert`、`\vert`、文本里用 `-` 代替 `_`。

另外 GitHub 上不稳定的宏：`\operatorname`、`\varnothing`、`\mathbb{1}`、`\substack`。

## 检查范围

* 块级 `$$...$$`：逐段检查（**不**要求裸 `|` 替换，块级里 `|` 是合法定界符）；
* 行内 `$...$`：**按行**切分（GitHub 的行内公式不能跨行）；
* 位于表格行（以 `|` 开头）里的数学：额外要求没有裸 `|`，否则会切断单元格。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BAD_MACROS = {
    r"\operatorname": r"\mathrm",
    r"\varnothing": r"\emptyset",
    r"\mathbb{1}": r"\mathbf{1}",
    r"\substack": "（避免）",
}

MARKDOWN_ESCAPED = {
    r"\{": r"\lbrace",
    r"\}": r"\rbrace",
    r"\#": r"\mathrm{card}",
    r"\|": r"\Vert",
    r"\_": "-（文本里改用连字符）",
}

BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")


def _brace_balance(s: str) -> int:
    depth = 0
    i = 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    return depth


def _check(seg: str) -> list[str]:
    problems = []
    for k, good in MARKDOWN_ESCAPED.items():
        if k in seg:
            problems.append(f"含 {k}（Markdown 会吃掉它 → 改用 {good}）")
    d = _brace_balance(seg)
    if d != 0:
        problems.append(f"花括号不配对（净差 {d:+d}）")
    # \lbrace/\rbrace \u540e\u76f4\u63a5\u8ddf\u5b57\u6bcd\u4f1a\u88ab TeX \u5f53\u6210\u672a\u77e5\u5b8f
    import re as _re
    if _re.search(r"\\\\[lr]brace[A-Za-z0-9]", seg):
        problems.append("\\lbrace/\\rbrace \u540e\u7f3a\u7a7a\u683c\uff08\u4f1a\u88ab\u5f53\u6210\u672a\u77e5\u5b8f\uff09")
    for macro, alt in BAD_MACROS.items():
        if macro in seg:
            problems.append(f"含不可靠宏 {macro}（改用 {alt}）")
    return problems


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    bad = 0
    nseg = 0

    # ---- 行内公式：逐行 ----
    in_block = False
    for no, ln in enumerate(lines, start=1):
        if ln.count("$$") % 2 == 1:
            in_block = not in_block
            continue
        if in_block:
            continue
        is_table = ln.lstrip().startswith("|")
        for m in INLINE.finditer(ln):
            nseg += 1
            seg = m.group(1)
            problems = _check(seg)
            if is_table and "|" in seg:
                problems.append("表格单元格里的数学含裸 | ⇒ 会切断单元格（改用 \\vert）")
            if problems:
                bad += 1
                print(f"  [FAIL] 行 {no}（行内/表格={is_table}）：" + "；".join(problems))
                print(f"         {seg[:140]}")

    # ---- 块级公式 ----
    for m in BLOCK.finditer(text):
        nseg += 1
        no = text[: m.start()].count("\n") + 1
        problems = _check(m.group(1))
        if problems:
            bad += 1
            print(f"  [FAIL] 行 {no}（块级）：" + "；".join(problems))
            print(f"         {m.group(1).strip()[:140]}")

    if bad:
        print(f"\n共 {bad} 处需要修")
        return 1
    print(f"  {path.name} 数学片段检查通过（{nseg} 段）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
