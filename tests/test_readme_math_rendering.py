"""把 README 数学的四道审计都接进测试闸门。

由来（2026-10-26，用户三次反馈"公式不渲染/退化为文字"）：

1. **Markdown 先吃掉反斜杠转义**：`\\{` `\\}` `\\#` `\\|` `\\_` 会被还原成裸字符，
   TeX 于是报 "Extra open brace" / "macro parameter character #"；
2. **`$$` 块前后必须空行**，否则整块退化为普通文字；
3. **宏后紧跟字母会被当成未知宏**（`\\lbracee:`、`\\VertJ_t`）；
4. **我自己的编辑事故**：用 PowerShell 双引号字符串改文件，`$T`/`$$`/`\\t` 被 shell 展开，
   把数学标记和路径吃掉了 —— 教训是**改 Markdown 只能用编辑工具，不能用 shell 字符串**。

这些测试保证四类问题都不会再回来。
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
]


@pytest.mark.parametrize("label,script", SCRIPTS, ids=[s for _, s in SCRIPTS])
def test_readme_math_audits_pass(label: str, script: str) -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert r.returncode == 0, f"{label} 未通过：\n{(r.stdout or '')[-2500:]}\n{(r.stderr or '')[-800:]}"
