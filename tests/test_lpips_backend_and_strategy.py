"""锁定 2026-11-01 的三项能力：LPIPS 算法可选、密度控制算法可选、官方机制成为默认。

为什么要测：
1. `lpips_backend` / `densify_strategy` 这类"算法选择"一旦在 GUI 里接好、后端却没接，
   表现是**静默不生效**（本项目已因"同一决定写两处"踩过多次）。所以既测后端行为，
   也测前端控件与参数名同时存在。
2. `config.py` 里 `absgrad` 曾被写了两遍（后者覆盖前者）⇒ DEFAULTS 改了却不生效。
   这里断言"只有一个键"且值为 True，防止回归。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from gsflow.config import DEFAULTS, load
from gsflow.trainers import perceptual

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "src" / "gsflow" / "web" / "static" / "index.html"
JS = ROOT / "src" / "gsflow" / "web" / "static" / "app.js"
CONFIG_PY = ROOT / "src" / "gsflow" / "config.py"


# ---------------------------------------------------------------- LPIPS 算法
def test_available_backends_reports_availability() -> None:
    items = perceptual.available_backends()
    keys = [b["key"] for b in items]
    assert keys == ["vgg16", "vgg16_uniform", "off"]
    for b in items:
        assert set(b) >= {"key", "label", "hint", "available", "reason"}
    # `off` 永远可用
    assert [b for b in items if b["key"] == "off"][0]["available"] is True


def test_uniform_backend_uses_equal_layer_weights() -> None:
    pytest.importorskip("torchvision")
    if not perceptual.vgg_weights_present():
        pytest.skip("缺 VGG16 权重")
    lp_u = perceptual.LPIPS(device="cpu", input_size=64, backend="vgg16_uniform")
    assert all(abs(w - 1.0) < 1e-9 for w in lp_u.layer_w)
    lp_o = perceptual.LPIPS(device="cpu", input_size=64, backend="vgg16")
    assert lp_u.layer_w != lp_o.layer_w, "等权应与论文线性权重不同"


def test_unknown_backend_raises() -> None:
    pytest.importorskip("torchvision")
    if not perceptual.vgg_weights_present():
        pytest.skip("缺 VGG16 权重")
    with pytest.raises(ValueError, match="backend"):
        perceptual.LPIPS(device="cpu", input_size=64, backend="not-a-backend")


# ---------------------------------------------------------------- GUI 接线
_ID_HTML = ["p-lpips-backend", "p-lpips-size", "p-train-lpips",
            "p-densify-strategy", "p-mcmc-noise"]
_KEY_JS = ["train.lpips_backend", "train.lpips_size", "train.lambda_lpips",
           "train.densify_strategy", "train.mcmc_noise_lr"]


def test_gui_has_the_controls() -> None:
    html = HTML.read_text(encoding="utf-8")
    for i in _ID_HTML:
        assert f'id="{i}"' in html, f"GUI 缺少控件 {i}"
    # 预设下拉框必须包含 official（否则 GUI 用不到实测最优预设）
    assert 'value="official"' in html


def test_gui_sends_the_params() -> None:
    js = JS.read_text(encoding="utf-8")
    for k in _KEY_JS:
        assert f"put('{k}'" in js, f"app.js 没有提交 {k}"
    # 三个控件必须进入"表单状态保存"列表，否则切换作业后选择会丢
    for i in _ID_HTML:
        assert f"'{i}'" in js, f"{i} 不在表单状态列表里"


def test_densify_strategy_options_cover_official_and_mcmc() -> None:
    html = HTML.read_text(encoding="utf-8")
    for v in ('value="builtin"', 'value="official"', 'value="mcmc"'):
        assert v in html, f"密度控制算法缺少选项 {v}"


# ---------------------------------------------------------------- 默认值
def test_official_mechanisms_are_defaults() -> None:
    tr = DEFAULTS["train"]
    assert int(tr["opacity_reset_interval"]) == 3000, "不透明度重置应为默认开启（官方机制）"
    # ⚠ absgrad 默认**仍为 False**：历史对照 +0.098 dB、95% CI [-0.114,+0.310] 不显著
    # （见 tests/test_texture_mask_sampling.py）。"补上 AbsGS" 的含义是**可用且真正生效**：
    # GUI 可选 official 策略，`official` 预设已同时开 strategy=official + absgrad=True。
    assert bool(tr["absgrad"]) is False, "absgrad 默认保持关闭（除非有新实测推翻）"
    assert int(tr["sh_degree_interval"]) == 1000, "SH 阶数递增应为默认开启"
    assert float(tr["opacity_floor"]) < 0.05, "opacity_floor 必须已从 0.10 降下来"
    assert int(tr["densify_until"]) >= 12000, "致密化窗口应接近官方量级"
    assert tr["lpips_backend"] == "vgg16"


def test_official_preset_enables_absgs_properly() -> None:
    """AbsGS 只在 official / mcmc 策略下生效。

    历史坑：`official` 预设曾同时写着 `densify_strategy=builtin` 与 `absgrad=True`，
    而 builtin 会**忽略** absgrad ⇒ 看似开了、实际没生效（config 注释里已写明这一点）。
    """
    from gsflow.presets import PRESETS

    o = PRESETS["official"]
    assert o["train.densify_strategy"] == "official", "official 预设必须走 official 策略"
    assert o["train.absgrad"] is True, "official 预设应启用 AbsGS"


def test_config_has_no_duplicate_absgrad_key() -> None:
    src = CONFIG_PY.read_text(encoding="utf-8")
    assert len(re.findall(r'"absgrad"\s*:', src)) == 1, "config.py 里 absgrad 只能出现一次"


def test_preset_official_resolves() -> None:
    s = load({"preset": "official"})
    assert float(s.get("train.lambda_lpips")) > 0
    assert float(s.get("train.opacity_floor")) <= 0.05
    assert int(s.get("train.cap_max")) >= 600000
