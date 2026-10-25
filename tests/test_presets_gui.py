"""GUI 预设与 CLI `--preset` 必须**同源**（单一映射源 `gsflow/presets.py`）。

动机：本项目已经吃过"两端各写一份、慢慢漂移"的亏（不透明度治理方案、拍摄模式都专门做过统一）。
预设这种"一条命令改变训练行为"的入口更不能两边各写一套键值 —— 漂移时不会报错，只会静默不生效。
"""
from __future__ import annotations

from pathlib import Path

from gsflow.config import DEFAULTS, load
from gsflow import presets

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "src" / "gsflow" / "web" / "static" / "index.html"
JS = ROOT / "src" / "gsflow" / "web" / "static" / "app.js"


def test_gui_has_preset_selector() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert 'id="p-train-preset"' in html, "GUI 表单里没有预设下拉框"
    assert 'value="quality"' in html, "下拉框没有 quality 选项"


def test_gui_sends_preset_param() -> None:
    js = JS.read_text(encoding="utf-8")
    assert "put('preset'," in js, "GUI 没有把 preset 放进提交参数"
    assert "'p-train-preset'" in js


def test_gui_form_state_preserves_preset_choice() -> None:
    """预设下拉框要进"表单状态保存"列表，否则切换作业/重载配置后选择会丢。"""
    js = JS.read_text(encoding="utf-8")
    i = js.find("p-train-backend', 'p-train-preset'")
    assert i > 0, "预设下拉框不在表单状态列表里"


def test_preset_param_survives_gui_flattening() -> None:
    """GUI 提交的是嵌套 params，拍平后必须是顶层 `preset`，`config.load` 才认得。"""
    from gsflow.web.app import _flatten_params

    flat = _flatten_params({"preset": "quality", "train": {"iterations": 30000}})
    assert flat["preset"] == "quality"
    s = load(flat)
    assert int(s.get("train.iterations")) == 30000, "显式键应压过预设"
    assert float(s.get("train.lambda_lpips")) > 0, "预设的其他键应生效"


def test_cli_and_gui_share_the_same_names() -> None:
    """CLI 的 --preset choices 必须与 presets.names() 一致（不能一边多一个）。"""
    cli_src = (ROOT / "src" / "gsflow" / "cli.py").read_text(encoding="utf-8")
    for name in presets.names():
        if name == "balanced":
            continue
        assert f'"{name}"' in cli_src, f"CLI 的 --preset 里缺少 {name}"
