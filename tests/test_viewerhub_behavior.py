"""用 node 跑 ViewerHub 的行为测试（有 node 才跑，没有就跳过）。

对应事故：服务端渲染 ↔ WebGL 来回切之后拖拽失效。
静态契约测试（`test_viewer_mode_switch.py`）只能证明"代码里调了 resume"，
这个测试真的把 `ViewerHub` 抽出来在桩 DOM 上跑一遍切换序列，断言 pointer-events 恢复。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_JS = ROOT / "tests" / "js" / "viewerhub_test.js"
NODE = shutil.which("node")


def test_viewerhub_switch_behavior() -> None:
    if not TEST_JS.exists():
        pytest.skip("没有 node 行为测试脚本")
    if not NODE:
        pytest.skip("环境里没有 node")
    r = subprocess.run([NODE, str(TEST_JS)], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, "ViewerHub 行为测试失败：\n" + (r.stdout or "")[-2500:] + (r.stderr or "")[-800:]
    assert "全部通过" in (r.stdout or "")
