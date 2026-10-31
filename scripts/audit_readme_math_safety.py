"""README 数学安全的综合审计（第四轮，2026-10-26）。

覆盖本项目实际踩过的全部坑，一次跑完：

| # | 检查项 | 事故 |
|---|---|---|
| 1 | 控制字符（U+0008 退格、U+0009 制表） | `\\b` `\\t` 被 shell 解释，吃掉字符 |
| 2 | 数学里的中文 | MathJax 字体回退不稳，可能整段退化为文字 |
| 3 | 数学里的 `\\texttt{...}` | 里面常带 `_`/`"`，易与 Markdown 冲突 |
| 4 | 正文里的裸 `<` `>`（非代码、非数学、非引用） | 可能被当 HTML 标签 |
| 5 | 高风险宏 `\\big` `\\Big` `\\underbrace` `\\overline` | MathJax 偶发不兼容 |
| 6 | 数学里 `\\Vert`（改用 `| |`） | GitHub 上曾整段不渲染 |

用法：`python scripts/audit_readme_math_safety.py [README.md]`
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CN = re.compile(r"[\u4e00-\u9fff]")
RISKY = [r"\big", r"\Big", r"\bigg", r"\underbrace", r"\overbrace", r"\overline", r"\Vert"]


def spans(text: str):
    out = [(m.start(), m.end()) for m in re.finditer(r"\$\$.+?\$\$", text, re.S)]
    stripped = re.sub(r"\$\$.+?\$\$", lambda m: " " * len(m.group(0)), text, flags=re.S)
    for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", stripped):
        out.append((m.start(), m.end()))
    return out


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    bad = 0

    # 1) 控制字符
    for no, ln in enumerate(text.splitlines(), start=1):
        for ch in ln:
            if unicodedata.category(ch) == "Cc":
                bad += 1
                print(f"  [FAIL] 行 {no}: 控制字符 U+{ord(ch):04X}（shell 转义残留）")
                print(f"         {ln[:100]}")

    # 2) 数学里的中文 / \texttt / 高风险宏
    for s, e in spans(text):
        seg = text[s:e]
        no = text[:s].count("\n") + 1
        if CN.search(seg):
            bad += 1
            print(f"  [FAIL] 行 {no}: 数学里有中文 → 移到公式外")
            print(f"         {seg.strip()[:110]}")
        if r"\texttt" in seg:
            bad += 1
            print(f"  [FAIL] 行 {no}: 数学里有 \\texttt{{}} → 改用正文反引号")
            print(f"         {seg.strip()[:110]}")
        for mac in RISKY:
            if mac in seg:
                bad += 1
                print(f"  [FAIL] 行 {no}: 数学里有 {mac} → 换成稳妥写法")
                print(f"         {seg.strip()[:110]}")

    # 3) 正文裸 < >（跳过跨行 $$ 块、代码围栏、引用块）
    in_fence = False
    in_block = False
    for no, ln in enumerate(text.splitlines(), start=1):
        s = ln.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if ln.count("$$") % 2 == 1:
            in_block = not in_block
            continue
        if in_block:
            continue
        if s.startswith(">"):
            continue
        no_math = re.sub(r"\$[^$\n]+?\$", "", ln)
        no_code = re.sub(r"`[^`]*`", "", no_math)
        for ch in "<>":
            for m in re.finditer(re.escape(ch), no_code):
                if m.start() and no_code[m.start() - 1] == "\\":
                    continue
                bad += 1
                print(f"  [FAIL] 行 {no}: 正文裸 `{ch}` → 用 &lt;/&gt; 或中文")
                print(f"         {ln.strip()[:110]}")

    if bad:
        print(f"\n共 {bad} 处")
        return 1
    print(f"  {path.name} 数学安全审计通过（控制字符/中文/texttt/裸尖括号/高风险宏）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
