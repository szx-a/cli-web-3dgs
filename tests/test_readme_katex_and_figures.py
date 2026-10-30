"""把 KaTeX/GitHub 渲染审计接进测试闸门，并校验 README 里引用的图片都存在。

由来（2026-10-30）：用户报"公式里的点乘号变成逗号"——实际是 `\\,`（thin space）被 Markdown
当成转义标点、还原成裸逗号；另有行内 `$` 内侧带空格导致整段退化成文字。
这两个坑都不能再靠肉眼发现，必须有自动检查。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_katex_audit_passes() -> None:
    script = ROOT / "scripts" / "audit_readme_katex.py"
    if not script.exists():
        pytest.skip("没有 katex 审计脚本")
    r = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, "KaTeX/GitHub 渲染审计未通过：\n" + (r.stdout or "")[-2500:]
    assert "未发现问题" in (r.stdout or "")


def test_readme_figure_links_exist() -> None:
    """README 里引用的每个图片路径都必须真实存在（否则 GitHub 上是裂图）。"""
    text = README.read_text(encoding="utf-8")
    missing = []
    for m in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)\)", text):
        rel = m.group(1)
        if rel.startswith(("http://", "https://")):
            continue
        if not (ROOT / rel).exists():
            missing.append(rel)
    assert not missing, f"README 引用了不存在的图片：{missing}"


def test_figures_are_generated_by_script() -> None:
    """图必须能由脚本复现（不是手工截图）。"""
    script = ROOT / "scripts" / "make_readme_figures.py"
    assert script.exists(), "缺少生成脚本"
    src = script.read_text(encoding="utf-8")
    for name in ("fig_grad_dose.png", "fig_depthvar_dose.png", "fig_60k_compare.png", "fig_lpips_cap.png"):
        assert name in src, f"脚本里没有生成 {name}"
        assert (ROOT / "docs" / "figs" / name).exists(), f"缺少图片 {name}"
