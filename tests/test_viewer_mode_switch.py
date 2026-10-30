"""「服务端渲染 ↔ WebGL 来回切」后交互不能被锁死（2026-10-30 修）。

## 事故

用户报告：先在「模型查看」里用 WebGL 拖拽，点「服务端渲染当前视角」看 gsplat 出图，
再点「返回 WebGL 交互」——**鼠标拖不动了**。

## 根因

临时切到服务端图/图片时调用了 `ViewerHub.set(null)`，它会把 active 置空并把
两层渲染器的 pointer-events 都关掉；而「返回」按钮只切了 display，**没有恢复 ViewerHub**。
于是 active 恒为 null、pointer-events 停在 `none`，交互永久失效。

## 这些测试锁住的行为

1. `ViewerHub` 必须提供 `suspend()` / `resume()` 一对；
2. 临时切走（服务端渲染、图片模式）只能调 `suspend()`，不得直接 `set(null)`；
3. 「返回 WebGL」必须调 `resume()`；
4. `resume()` 必须同时处理**显示哪一层**（高斯在 `.gs-ref-stage`、glb/网格在 `#m-canvas`）
   与 pointer-events 恢复 —— 否则会出现"状态对了但屏幕上是空的"。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "gsflow" / "web" / "static" / "app.js"


@pytest.fixture(scope="module")
def src() -> str:
    if not APP.exists():
        pytest.skip("没有 app.js")
    return APP.read_text(encoding="utf-8")


def test_viewerhub_exposes_suspend_and_resume(src: str) -> None:
    assert "function suspend(" in src, "ViewerHub 缺少 suspend()"
    assert "function resume()" in src, "ViewerHub 缺少 resume()"
    # 导出（对象字面量里要有）
    assert re.search(r"set,\s*supports,.*suspend,\s*resume", src), "suspend/resume 未导出"


def test_temporary_switch_uses_suspend_not_set_null(src: str) -> None:
    """服务端渲染与图片模式都必须用 suspend()，不能再直接 set(null)。"""
    assert "ViewerHub.suspend('server')" in src, "服务端渲染未使用 suspend"
    assert "ViewerHub.suspend('image')" in src, "图片模式未使用 suspend"
    # set(null) 只允许出现在 ViewerHub 内部（suspend 里）与 resume 的兜底
    calls = re.findall(r"ViewerHub\.set\(null\)", src)
    assert not calls, f"仍有 {len(calls)} 处直接 ViewerHub.set(null)（应改为 suspend）"


def test_back_button_calls_resume(src: str) -> None:
    i = src.find("m-back-webgl")
    assert i > 0, "找不到返回按钮绑定"
    window = src[i:i + 2600]
    assert "ViewerHub.resume()" in window, "返回按钮没有调用 resume()"


def test_resume_restores_the_visible_layer(src: str) -> None:
    """resume 里必须显式切换可见层：高斯 → .gs-ref-stage，glb/网格 → #m-canvas。"""
    i = src.find("function resume()")
    assert i > 0
    body = src[i:i + 2200]
    assert "gs-ref-stage" in body, "resume 没处理高斯层（.gs-ref-stage）"
    assert "m-canvas" in body, "resume 没处理 canvas 层"
    assert body.count("style.display") >= 4, "resume 应同时控制两层的 display"
