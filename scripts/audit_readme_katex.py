"""找出 README 里会被 Markdown 吃掉、或 GitHub 不认的行内数学写法。

## 用户报的两个现象

1. **"点乘号变成逗号"**：`\\,`（thin space）在 Markdown 里是**可转义标点**，
   cmark 会把它还原成裸 `,` ⇒ 公式里显示成逗号。同理 `\\;` → `;`、`\\:` → `:`、`\\!` → `!`。
   安全写法：`\\ `（反斜杠+空格，TeX 的控制空格）或 `\\quad` / `\\qquad`。
2. **"部分数学退化成文字"**：GitHub 的行内数学要求 `$` 紧贴内容 ——
   开 `$` 后不能有空格、闭 `$` 前不能有空格、闭 `$` 后不能紧跟数字；否则整段按纯文本渲染。

本脚本同时检查：间距命令、行内 `$` 的空格规则、以及 `$$` 块内的间距命令。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACING = {r"\, ", r"\,", r"\;", r"\:", r"\!", r"\/"}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    bad = 0

    print("=== 1) 会被 Markdown 吃掉的间距命令 ===")
    for i, ln in enumerate(lines, start=1):
        for pat in (r"\,", r"\;", r"\:", r"\!"):
            c = ln.count(pat)
            if c:
                bad += c
                print(f"  行 {i}: `{pat}` ×{c}   {ln.strip()[:90]}")

    print("=== 2) 行内 `$` 的空格规则（GitHub 要求紧贴）===")
    for i, ln in enumerate(lines, start=1):
        if "$$" in ln:
            continue
        for m in re.finditer(r"(?<!\$)\$(?!\$)([^$\n]*?)(?<!\$)\$(?!\$)", ln):
            seg = m.group(1)
            probs = []
            if seg[:1].isspace():
                probs.append("开 $ 后有空格")
            if seg[-1:].isspace():
                probs.append("闭 $ 前有空格")
            after = ln[m.end():m.end() + 1]
            if after.isdigit():
                probs.append("闭 $ 后紧跟数字")
            if probs:
                bad += 1
                print(f"  行 {i}: {'、'.join(probs)}   {ln.strip()[:90]}")

    print("=== 2b) 表格行里不允许行内数学（GitHub 单元格内 $...$ 不稳定）===")
    for i, ln in enumerate(lines, start=1):
        if not ln.lstrip().startswith("|"):
            continue
        if "$$" in ln:
            continue
        # 去掉反引号代码后再看
        masked = re.sub(r"`[^`\n]*`", "", ln)
        if re.search(r"(?<!\$)\$(?!\$)[^$\n]*?(?<!\$)\$(?!\$)", masked):
            bad += 1
            print(f"  行 {i}: 表格单元格里有行内数学 → 改纯文本（如 λ_lpips）或反引号参数名")
            print(f"         {ln.strip()[:100]}")

    print("=== 3) 其它高风险（KaTeX 不支持/不稳定）===")
    for name in (r"\operatorname", r"\varnothing", r"\mathbb{1}", r"\substack", r"\begin{aligned}"):
        n = text.count(name)
        if n:
            bad += n
            print(f"  `{name}` ×{n}")

    print(f"\n共 {bad} 处" if bad else "\n未发现问题")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
