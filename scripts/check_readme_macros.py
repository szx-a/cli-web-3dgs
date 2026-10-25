"""只提取**数学片段内**的宏，逐个对照白名单，找出 MathJax 不认识的命令。

（之前清点的是全文反斜杠，会把 `scripts\\web.cmd` 这类路径也算进去，噪声太大。）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# MathJax 3（TeX 输入，AMS）里可靠支持的命令 —— 本项目实际用到的都在这里
SAFE = set("""
frac dfrac tfrac sqrt root over choose binom
sum prod int oint lim inf sup max min arg det exp log ln sin cos tan
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa
lambda mu nu xi omicron pi varpi rho varrho sigma varsigma tau upsilon phi
varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
mathbf mathrm mathcal mathbb mathit mathsf mathtt mathbfit
text textbf textit texttt textrm
left right big Big bigg Bigg bigl bigr Bigl Bigr middle
lbrace rbrace langle rangle lfloor rfloor lceil rceil vert Vert
cdot cdots ldots dots vdots ddots times div pm mp ast star circ bullet
le ge ne approx sim simeq equiv propto ll gg
in notin ni subset supset subseteq supseteq cup cap setminus emptyset
forall exists nexists neg land lor wedge vee oplus otimes
to rightarrow leftarrow leftrightarrow Rightarrow Leftarrow uparrow downarrow
mapsto longrightarrow iff implies deg ominus top
partial nabla prime infty aleph hbar ell Re Im wp
quad qquad space enspace thinspace
overline underline widehat widetilde vec dot ddot hat bar tilde
overbrace underbrace stackrel overset underset
bar overrightarrow
nabla varnothing
rangle lbrace
not
tag label
bmod pmod
dots
""".split())

# 文本类环境/装饰里允许出现的（不算宏错误）
CONTEXT_SAFE = {"begin", "end", "text", "texttt", "mathrm", "mathbf", "left", "right"}


def math_segments(text: str):
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S):
        yield text[: m.start()].count("\n") + 1, m.group(1)
    stripped = re.sub(r"\$\$(.+?)\$\$", lambda m: " " * len(m.group(0)), text, flags=re.S)
    for no, ln in enumerate(stripped.splitlines(), start=1):
        for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", ln):
            yield no, m.group(1)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    unknown: dict[str, list[int]] = {}
    for no, seg in math_segments(text):
        for m in re.finditer(r"\\([A-Za-z]+)", seg):
            name = m.group(1)
            if name not in SAFE and name not in CONTEXT_SAFE:
                unknown.setdefault(name, []).append(no)

    if unknown:
        for name, nos in sorted(unknown.items()):
            print(f"  [WARN] 宏 \\{name} 不在白名单（行 {sorted(set(nos))[:8]}）")
        print(f"\n共 {len(unknown)} 个未知宏")
        return 1
    print(f"  {path.name} 数学片段内的宏全部在白名单内")
    return 0


if __name__ == "__main__":
    sys.exit(main())
