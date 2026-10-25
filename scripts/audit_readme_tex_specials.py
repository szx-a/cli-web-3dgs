"""扫描数学片段里的 TeX 特殊字符：`%`（注释）、`&`（对齐）、`~`、未转义的 `#`。

`%` 一旦出现在数学里，TeX 会把**其后整行**当注释吞掉 ⇒ 括号看起来"少了一个"
⇒ GitHub 报 `Extra open brace or missing close brace`，或整段退化为文字。
`&` 在非对齐环境里同样报错。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPECIALS = {
    "%": "TeX 注释符，会把后面整行吞掉",
    "&": "TeX 对齐符，非对齐环境里报错",
    "~": "TeX 不换行空格，MathJax 里常被当普通字符但与文本冲突",
}


def segments(text: str):
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S):
        yield text[: m.start()].count("\n") + 1, "block", m.group(1)
    stripped = re.sub(r"\$\$(.+?)\$\$", lambda m: " " * len(m.group(0)), text, flags=re.S)
    for no, ln in enumerate(stripped.splitlines(), start=1):
        for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", ln):
            yield no, "inline", m.group(1)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    bad = 0
    for no, kind, seg in segments(text):
        for ch, why in SPECIALS.items():
            # 只看未转义的
            for m in re.finditer(re.escape(ch), seg):
                i = m.start()
                if i > 0 and seg[i - 1] == "\\":
                    continue
                bad += 1
                print(f"  [FAIL] 行 {no}（{kind}）：数学里有未转义 `{ch}` —— {why}")
                print(f"         {seg[max(0, i-40):i+40].strip()}")
    if bad:
        print(f"\n共 {bad} 处")
        return 1
    print(f"  {path.name} 数学片段里没有 `%`/`&`/`~` 等 TeX 特殊字符")
    return 0


if __name__ == "__main__":
    sys.exit(main())
