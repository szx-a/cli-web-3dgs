"""用 matplotlib 的 mathtext 作为**第二解析器**交叉验证 README 数学语法。

MathJax 与 mathtext 支持的宏集合不同（mathtext 更小），所以：
* **两边都拒绝** ⇒ 基本可以确定是语法错误（括号、未知宏、`_`/`^` 位置）；
* 只有 mathtext 拒绝 ⇒ 多半是 mathtext 不支持（如 `\\underbrace`、`\\Vert`），仅提示。

同时检查 `$$` 块是否被 4 空格以上缩进（GitHub 会当代码块 ⇒ 公式退化为文字）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import mathtext  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def segments(text: str):
    out = []
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S):
        out.append((text[: m.start()].count("\n") + 1, "block", m.group(1).strip()))
    stripped = re.sub(r"\$\$(.+?)\$\$", lambda m: " " * len(m.group(0)), text, flags=re.S)
    for no, ln in enumerate(stripped.splitlines(), start=1):
        for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", ln):
            out.append((no, "inline", m.group(1).strip()))
    return out


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    text = path.read_text(encoding="utf-8")

    # 1) 缩进导致的代码块化
    bad = 0
    for no, ln in enumerate(text.splitlines(), start=1):
        if ln.startswith("    ") and "$$" in ln:
            bad += 1
            print(f"  [FAIL] 行 {no}: `$$` 被缩进 4 空格 ⇒ GitHub 会当代码块（公式退化为文字）")

    # 2) 语法交叉验证
    segs = segments(text)
    fail = 0
    for no, kind, seg in segs:
        try:
            mathtext.MathTextParser("path").parse(f"${seg}$", dpi=100)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            msg = str(exc).splitlines()[0][:120]
            print(f"  [WARN] 行 {no}（{kind}）mathtext 拒绝：{msg}")
            print(f"         {seg[:120]}")
    print(f"\n  共 {len(segs)} 段；缩进问题 {bad} 处；mathtext 拒绝 {fail} 段")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
