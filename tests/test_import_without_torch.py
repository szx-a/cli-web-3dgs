"""无 torch 环境下也必须能 import（安装器的 core-only 安装 + doctor 自检都依赖）。

## 背景（2026-10-25，安装器实测抓到）

`gsplat_trainer` 里原来有两个**模块级** `@torch.no_grad()` 装饰器。在没有 torch 的环境里
（`requirements.txt` 的 core-only 安装、`gsflow doctor` 的环境自检），模块里的
`torch = None` ⇒ 装饰器在**定义时**求值 ⇒
`AttributeError: 'NoneType' object has no attribute 'no_grad'` ⇒
连 `python -m gsflow doctor` 都起不来，安装器的最后一步验证直接失败。

修法：把 `no_grad` 移进函数体（`with torch.no_grad():`），模块顶层不再触碰 torch 属性。
"""
from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_trainer_module_imports_without_torch(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked for this test")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in list(sys.modules) if m.startswith("gsflow.trainers")]:
        sys.modules.pop(mod, None)
    monkeypatch.setattr(builtins, "__import__", fake)

    m = importlib.import_module("gsflow.trainers.gsplat_trainer")
    assert m.torch is None
    # 关键：模块级的两个辅助函数可以被引用（定义时不求值 torch 属性）
    assert hasattr(m, "_apply_keep") and hasattr(m, "_grow")


def test_no_module_level_torch_attribute_use() -> None:
    """静态护栏：模块级不允许出现 `@torch.` 装饰器（会被 import 时求值）。"""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "gsflow" / "trainers"
           / "gsplat_trainer.py").read_text(encoding="utf-8")
    for i, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("@torch."):
            pytest.fail(f"第 {i} 行是模块级 @torch 装饰器：{stripped}（无 torch 环境会崩）")
