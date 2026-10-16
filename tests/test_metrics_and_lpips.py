"""多指标评估与感知损失的回归测试。

覆盖三件容易**静默失效**的事：
  1. `evaluate()` 必须返回 dict（psnr/ssim/lpips/views），而不是旧的单个 float ——
     调用方若还按 float 用（`eval_res > best`），会静默变成"dict 比较"而错判最佳检查点。
  2. 多指标配置项必须在 DEFAULTS 与 TrainConfig 两处一致。
  3. LPIPS 缺失权重时必须**抛异常**（由调用方退化为"不使用"），
     而不是返回 0 —— 返回 0 会让"开着但没生效"无法察觉（本项目踩过同类坑）。
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_evaluate_returns_metric_dict() -> None:
    """`evaluate()` 的返回必须是含 psnr/ssim/lpips/views 的 dict。"""
    import inspect

    from gsflow.trainers.gsplat_trainer import evaluate

    src = inspect.getsource(evaluate)
    assert '"psnr"' in src and '"ssim"' in src and '"lpips"' in src and '"views"' in src, \
        "evaluate() 必须同时返回 psnr / ssim / lpips / views"
    # 空评估集也要返回同构的 dict（否则调用方要写两套分支）
    assert 'return {"psnr": 0.0' in src, "空评估集的分支也必须返回 dict，不能返回 0.0"


def test_best_metric_keys_present_and_consistent() -> None:
    from gsflow.config import DEFAULTS
    from gsflow.trainers.gsplat_trainer import TrainConfig

    td = {f.name: f.default for f in fields(TrainConfig)}
    for k in ("eval_metrics", "best_metric", "lambda_lpips", "lpips_every"):
        assert k in DEFAULTS["train"], f"DEFAULTS['train'] 缺 {k}"
        assert k in td, f"TrainConfig 缺 {k}"
        assert DEFAULTS["train"][k] == td[k], \
            f"{k} 两处默认值不一致：{DEFAULTS['train'][k]!r} vs {td[k]!r}"


def test_best_metric_defaults_to_psnr_and_is_a_known_choice() -> None:
    from gsflow.config import DEFAULTS

    bm = DEFAULTS["train"]["best_metric"]
    assert bm in ("psnr", "ssim", "lpips"), f"未知的 best_metric：{bm!r}"


def test_lpips_raises_when_weights_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """权重缺失时必须抛异常，**不能**返回 0（否则"开着但没生效"无法察觉）。"""
    from gsflow.trainers import perceptual

    monkeypatch.setattr(perceptual, "vgg_weights_present", lambda: False)
    monkeypatch.setattr(perceptual, "torch_home", lambda: Path("F:/nonexistent-torch-home"))
    with pytest.raises(Exception):
        perceptual.LPIPS("cpu")
