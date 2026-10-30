"""用真实 MathJax 校验文档公式，并检查 `$` 配对（2026-10-30）。

背景：用户三次反馈"公式点乘号变成逗号 / 部分表达式退化成文字"。逐层排查后确认：

* `\\,` `\\;` 等间距命令会被 Markdown 当转义吃掉（显示成逗号）——已改 `\\ `；
* `\\tfrac` 在 **MathJax** 里是未定义宏（KaTeX 支持，所以 KaTeX 校验发现不了）；
* 正文里裸 `$`（例如把故障现象写成 `$T` / `$$`）会吞掉后面的正文。

这个测试用 **mathjax-full**（GitHub 的引擎）逐段编译 README 与 docs。
依赖是本地开发依赖（`env/tmp/node_modules/mathjax-full`），缺失时跳过而不是失败：

    cd env/tmp && npm install mathjax-full --cache env/tmp/npm-cache
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_readme_mathjax.js"
MATHJAX = ROOT / "env" / "tmp" / "node_modules" / "mathjax-full"
NODE = shutil.which("node")

DOCS = ["README.md", "docs/03-踩坑记录.md", "docs/05-训练框架.md", "docs/08-训练框架现状.md"]


def _skip_if_unavailable() -> None:
    if not VALIDATOR.exists():
        pytest.skip("没有 MathJax 校验脚本")
    if not NODE:
        pytest.skip("环境里没有 node")
    if not MATHJAX.exists():
        pytest.skip("没有 mathjax-full（cd env/tmp && npm install mathjax-full）")


@pytest.mark.parametrize("rel", DOCS, ids=[Path(d).stem for d in DOCS])
def test_docs_compile_with_mathjax(rel: str) -> None:
    _skip_if_unavailable()
    target = ROOT / rel
    if not target.exists():
        pytest.skip(f"没有 {rel}")
    r = subprocess.run([NODE, str(VALIDATOR), rel], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, f"{rel} 有公式过不了 MathJax：\n" + (r.stdout or "")[-2500:]
    assert "全部通过 MathJax 编译" in (r.stdout or "")


def test_dollar_pairs_are_balanced() -> None:
    script = ROOT / "scripts" / "audit_docs_dollar_pairs.py"
    if not script.exists():
        pytest.skip("没有配对检查脚本")
    r = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, "`$` / `$$` 不配对：\n" + (r.stdout or "")[-2500:]
