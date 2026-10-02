"""阶段注册表：名字 -> 阶段实例。

新增阶段只需要在 stages/__init__.py 里导入并注册，CLI 与 GUI 会自动多出
这个阶段（两端的阶段列表都从这里取，保证同构）。
"""

from __future__ import annotations

from .base import Stage

_REGISTRY: dict[str, Stage] = {}
_ORDER: list[str] = []


def register(stage: Stage, *, first: bool = False) -> Stage:
    if not stage.name:
        raise ValueError("阶段必须有 name")
    if stage.name in _REGISTRY:
        raise ValueError(f"阶段名重复：{stage.name}")
    _REGISTRY[stage.name] = stage
    if first:
        _ORDER.insert(0, stage.name)
    else:
        _ORDER.append(stage.name)
    return stage


def get(name: str) -> Stage:
    if name not in _REGISTRY:
        raise KeyError(f"未知阶段 '{name}'，可用：{', '.join(_ORDER)}")
    return _REGISTRY[name]


def all_stages() -> list[Stage]:
    return [_REGISTRY[n] for n in _ORDER]


def names() -> list[str]:
    return list(_ORDER)


def resolve(selected: list[str] | None) -> list[str]:
    """把用户选择展开成实际执行序列（含依赖闭包，保持注册顺序）。"""
    if not selected:
        return list(_ORDER)
    chosen: set[str] = set()
    stack = list(selected)
    while stack:
        name = stack.pop()
        if name in chosen:
            continue
        stage = get(name)
        chosen.add(name)
        stack.extend(stage.requires)
    return [n for n in _ORDER if n in chosen]


def describe() -> list[dict]:
    out = []
    for s in all_stages():
        out.append(
            {
                "name": s.name,
                "title": s.title,
                "description": s.description,
                "requires": list(s.requires),
                "optional": s.optional,
                "rerunnable": s.rerunnable,
                "inputs": [{"path": a.path, "desc": a.desc} for a in s.inputs],
                "outputs": [{"path": a.path, "desc": a.desc} for a in s.outputs],
            }
        )
    return out
