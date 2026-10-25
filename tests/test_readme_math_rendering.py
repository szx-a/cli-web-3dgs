"""把 README 数学的六道审计都接进测试闸门。

由来（2026-10-26，用户四轮反馈"公式不渲染/退化为文字"）：

1. **Markdown 先吃掉反斜杠转义**：`\\{` `\\}` `\\#` `\\|` `\\_` → TeX 收到裸字符；
2. **`$$` 块前后必须空行**，否则整块退化为普通文字；
3. **宏后紧跟字母被当未知宏**（`\\lbracee:`、`\\VertJ_t`）；
4. **shell 双引号字符串编辑**：`$T` 被当变量删掉、`$$` 被展开、`\\b` `\\t` 变成控制字符；
5. **数学里的中文与 `\\texttt{}`**、**高风险宏**（`\\big` `\\underbrace` `\\Vert`）；
6. **正文裸 `<job>`** 会被 GitHub 当未知 HTML 标签整段吞掉。

六道审计全部跑一遍，任何一道失败则测试失败。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    ("片段规则（转义/括号/宏/表格）", "check_readme_math.py"),
    ("块级与行内标记", "audit_readme_math_blocks.py"),
    ("shell 展开痕迹", "audit_readme_shell_damage.py"),
    ("宏白名单", "check_readme_macros.py"),
    ("TeX 特殊字符", "audit_readme_tex_specials.py"),
    ("数学安全（控制字符/中文/尖括号）", "audit_readme_math_safety.py"),
]


@pytest.mark.parametrize("label,script", SCRIPTS, ids=[s for _, s in SCRIPTS])
def test_readme_math_audits_pass(label: str, script: str) -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert r.returncode == 0, f"{label} 未通过：\n{(r.stdout or '')[-2500:]}\n{(r.stderr or '')[-800:]}"
