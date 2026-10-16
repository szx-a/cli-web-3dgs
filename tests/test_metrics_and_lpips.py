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


def test_lpips_loss_exists_but_defaults_off() -> None:
    """感知损失必须**存在**（否则无法做同数据对照），但默认关闭。

    为什么不允许把它删掉（这条测试的由来）：
    我曾据一轮对照把 LPIPS 判为"有害"并删掉代码。后来发现那轮测量用的是
    `sh_degree=0` 渲染 —— 丢掉了 45 个高阶 SH 分量，实测低估 1.91 dB，
    而且恰好抹掉了**视角相关**外观，也就是感知损失要改善的那部分。
    ⇒ 用看不见优化目标的尺子去判断优化有没有用，必然得到假阴性。

    所以：代码留着（可随时重做对照），默认值不许悄悄打开。
    """
    from gsflow.config import DEFAULTS
    from gsflow.trainers.gsplat_trainer import TrainConfig

    td = {f.name: f.default for f in fields(TrainConfig)}
    assert "lambda_lpips" in td, (
        "感知损失开关被删掉了。删除会让这个方法无法再被验证；"
        "如果确实要删，请在 docs/03 说明为什么不再需要它，并更新这条测试"
    )
    assert float(td["lambda_lpips"]) == 0.0, \
        "lambda_lpips 默认必须是 0：开启前需要新的同数据对照证据（噪声底约 0.09 dB）"
    assert float(DEFAULTS["train"]["lambda_lpips"]) == 0.0


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
