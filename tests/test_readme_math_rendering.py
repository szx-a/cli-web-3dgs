r"""把 README 数学检查接进文档闸门（tests），防止以后再加回不可渲染的写法。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_math_renders_on_github() -> None:
    """README 的数学必须通过 `scripts/check_readme_math.py`。

    这条测试的由来（2026-10-26）：用户在 GitHub 上两次看到 TeX 报错

        Extra open brace or missing close brace
        You can't use 'macro parameter character #' in math mode

    根因是 **Markdown 先吃掉 `\{` `\}` `\#` `\|` `\_` 的反斜杠**，TeX 收到裸字符；
    另外矩阵环境里的 `\\\\` 也会被吃成单个 `\`，矩阵塌成一行。
    检查器把这些规则固化下来，这里只要保证它在 CI/本地闸门里真的跑。
    """
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_readme_math.py")],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert r.returncode == 0, (
        "README 数学检查未通过：\n" + (r.stdout or "")[-3000:] + "\n" + (r.stderr or "")[-1000:]
    )
    assert "检查通过" in (r.stdout or "")
