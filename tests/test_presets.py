"""预设模块的契约测试（纯 CPU）。

动机：预设是"一条命令改变训练行为"的入口，键名写错/类型转错**不会报错**，只会静默不生效。
本项目的经验是——静默失效最终都会变成"为什么优化没效果"的追问。
"""

from __future__ import annotations

import pytest

from gsflow import presets
from gsflow.config import DEFAULTS, load


def test_presets_are_listed() -> None:
    assert "quality" in presets.names()
    assert "balanced" in presets.names()


def test_balanced_is_a_no_op() -> None:
    assert presets.resolve("balanced", DEFAULTS) == {}


def test_unknown_preset_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError) as ei:
        presets.resolve("ultra", DEFAULTS)
    assert "quality" in str(ei.value)


def test_every_preset_key_exists_in_defaults() -> None:
    """键名必须存在于 DEFAULTS —— 写错立刻暴露（这条抓过 train.eval_interval 的错）。"""
    for name in presets.names():
        presets.resolve(name, DEFAULTS)      # 不抛异常即通过


def test_quality_preset_is_actually_applied() -> None:
    s = load({"preset": "quality"})
    assert int(s.get("train.iterations")) == 60000
    assert int(s.get("train.densify_until")) == 20000
    assert float(s.get("train.lambda_lpips")) > 0
    assert int(s.get("train.lpips_every")) == 1, "LPIPS 必须每步都算（G152：稀疏施加会让梯度打架）"
    assert str(s.get("train.depth_mode")) == "expected", "训练侧深度必须用 ED（与导出统一）"
    assert float(s.get("train.lambda_normal")) > 0
    assert float(s.get("train.lambda_depth_distortion")) > 0


def test_explicit_override_beats_preset() -> None:
    s = load({"preset": "quality", "train.iterations": "12000"})
    assert int(s.get("train.iterations")) == 12000


def test_quality_preset_does_not_enable_uncapped_official_strategy() -> None:
    """实测教训：`official` + 2e-4 会致密化失控（step 10000 → 251 万高斯）。

    预设必须用已校准的内置策略；想试官方策略要显式覆盖，并同时把 cap_max 设好。
    """
    s = load({"preset": "quality"})
    assert str(s.get("train.densify_strategy")) == "builtin"
    assert int(s.get("train.cap_max")) > 0


def test_bool_strings_are_coerced() -> None:
    """`bool("false")` 是 True —— 这里必须按语义转成 False。"""
    assert presets._coerce_like("false", True) is False
    assert presets._coerce_like("1", False) is True
    assert presets._coerce_like("0.5", 0.0) == 0.5
    assert presets._coerce_like("700", 1) == 700
