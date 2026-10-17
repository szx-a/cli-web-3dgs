"""CLI 与 GUI 的"不透明度治理方案"必须产出**完全相同**的配置参数。

为什么必须锁住：CLI 的 `--opacity-cleanup` 与 GUI 下拉框若各写一份映射，
两端就会漂移。本会话已多次因"同一个决定写两遍"而出错
（最严重的一次是 22 个脚本各自用 `sh_degree=0` 渲染，白丢 1.91 dB）。

设计：映射只在 `trainers/opacity_cleanup.py::opacity_cleanup_params` 一处定义；
CLI 调它，GUI 通过 `/api/opacity_cleanup` 取同一份。
本测试断言"后端接口给出的参数"与"映射函数给出的参数"逐项相同。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_choices_are_complete() -> None:
    from gsflow.trainers.opacity_cleanup import CHOICES

    assert set(CHOICES) == {"none", "reset", "decay", "utilization", "recovery"}, \
        "方案集合变化时必须同步更新 GUI 下拉框与文档"


@pytest.mark.parametrize("choice", ["none", "reset", "decay", "utilization", "recovery"])
def test_each_choice_returns_its_own_keys_only(choice: str) -> None:
    """选了某个方案时，**不能**顺带打开别的方案（不做隐式联动）。"""
    from gsflow.trainers.opacity_cleanup import opacity_cleanup_params

    p = opacity_cleanup_params(choice)
    assert p["train.opacity_cleanup"] == choice
    others = {
        "reset": "train.opacity_reset_interval",
        "decay": "train.opacity_decay",
        "utilization": "train.utilization_prune_interval",
        "recovery": "train.recovery_prune_interval",
    }
    for name, key in others.items():
        if name != choice and not (choice == "recovery" and name == "decay"):
            assert key not in p, f"选 {choice} 时不该带上 {name} 的键 {key}"
    if choice == "none":
        assert len(p) == 1, "none 只应标记方案名，不应打开任何开关"


def test_param_is_honoured_where_it_matters() -> None:
    from gsflow.trainers.opacity_cleanup import opacity_cleanup_params

    assert opacity_cleanup_params("decay", 0.995)["train.opacity_decay"] == "0.995"
    assert opacity_cleanup_params("utilization", 0.35)[
        "train.utilization_prune_quantile"] == "0.35"
    assert opacity_cleanup_params("recovery", 0.1)[
        "train.recovery_prune_quantile"] == "0.1"


def test_unknown_choice_raises() -> None:
    from gsflow.trainers.opacity_cleanup import opacity_cleanup_params

    with pytest.raises(ValueError):
        opacity_cleanup_params("make_it_fast")


def test_web_api_returns_same_mapping_as_python() -> None:
    """**两端同构的核心断言**：GUI 取到的参数必须与 CLI 用的完全相同。"""
    from fastapi.testclient import TestClient

    from gsflow.trainers.opacity_cleanup import CHOICES, opacity_cleanup_params
    from gsflow.web.app import create_app

    client = TestClient(create_app())
    r = client.get("/api/opacity_cleanup")
    assert r.status_code == 200, f"接口不可用：{r.status_code}"
    got = {x["value"]: x["params"] for x in r.json()["choices"]}
    assert set(got) == set(CHOICES)
    for c in CHOICES:
        assert got[c] == opacity_cleanup_params(c), \
            f"方案 {c} 的 GUI 参数与 CLI 不一致：{got[c]} vs {opacity_cleanup_params(c)}"


def test_gui_select_options_match_choices() -> None:
    """HTML 下拉框的 option 值必须与 `CHOICES` 一致（否则界面能选到后端不认识的方案）。"""
    from gsflow.trainers.opacity_cleanup import CHOICES

    html = (ROOT / "src" / "gsflow" / "web" / "static" / "index.html").read_text(
        encoding="utf-8")
    import re
    m = re.search(r'id="p-train-opacityclean"(.*?)</select>', html, re.S)
    assert m, "index.html 里找不到 opacity_cleanup 下拉框"
    vals = set(re.findall(r'<option value="([^"]+)"', m.group(1)))
    assert vals == set(CHOICES), f"下拉框选项 {vals} 与 CHOICES {set(CHOICES)} 不一致"
