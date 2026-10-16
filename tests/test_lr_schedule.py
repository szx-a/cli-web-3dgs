"""学习率调度的回归测试：预热 + 余弦退火的取值必须与**解析端点**一致。

为什么用解析断言而不是"跑一遍看没报错"：调度曲线写错的后果是**静默**的 ——
形状不对只会让训练"差一点"，不抛任何异常。本项目已多次踩"没有已知答案就发现不了错误"
（法线公式错 26.6° 不报错；诊断脚本量出无意义的 2.31 倍）。
所以这里断言的是**可以手算出来的点**。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_lr_at_matches_analytic_endpoints() -> None:
    from gsflow.trainers.gsplat_trainer import lr_at

    P, T, W, r = 1e-3, 1000, 100, 0.1
    assert lr_at(0, P, r, W, T) == pytest.approx(P / W, rel=1e-12), \
        "预热第一步应为 peak/warmup（非 0）"
    assert lr_at(W - 1, P, r, W, T) == pytest.approx(P, rel=1e-12), "预热末步应到峰值"
    assert lr_at(W, P, r, W, T) == pytest.approx(P, rel=1e-12), "预热结束仍应为峰值"
    assert lr_at((W + T) // 2, P, r, W, T) == pytest.approx((P + P * r) / 2, rel=1e-3), \
        "余弦中点应为 (peak+min)/2"
    assert lr_at(T - 1, P, r, W, T) == pytest.approx(P * r, rel=1e-2), "末尾应退火到下限"


def test_lr_at_is_monotonic_in_both_phases() -> None:
    from gsflow.trainers.gsplat_trainer import lr_at

    P, T, W, r = 1e-3, 1000, 100, 0.1
    warm = [lr_at(s, P, r, W, T) for s in range(W)]
    assert all(warm[i] < warm[i + 1] for i in range(len(warm) - 1)), "预热段必须严格递增"
    cool = [lr_at(s, P, r, W, T) for s in range(W, T)]
    assert all(cool[i] >= cool[i + 1] - 1e-15 for i in range(len(cool) - 1)), \
        "退火段必须单调不增"
    assert cool[0] > cool[-1], "退火段整体应下降"


def test_lr_at_with_warmup_disabled_starts_at_peak() -> None:
    """warmup=0（默认，保持与历史结果可比）时，首步即为峰值。"""
    from gsflow.trainers.gsplat_trainer import lr_at

    P, T = 1e-3, 1000
    assert lr_at(0, P, 0.1, 0, T) == pytest.approx(P, rel=1e-12)


def test_lr_schedule_config_keys_are_consistent() -> None:
    """新配置项必须在 DEFAULTS 与 TrainConfig 两处都存在且一致（曾有"三处来源"的坑）。"""
    from dataclasses import fields

    from gsflow.config import DEFAULTS
    from gsflow.trainers.gsplat_trainer import TrainConfig

    td = {f.name: f.default for f in fields(TrainConfig)}
    for k in ("lr_schedule", "warmup_iters", "lr_min_ratio", "lr_quats"):
        assert k in DEFAULTS["train"], f"DEFAULTS['train'] 缺 {k}（界面/文件里会看不到）"
        assert k in td, f"TrainConfig 缺 {k}"
        assert DEFAULTS["train"][k] == td[k], \
            f"{k} 两处默认值不一致：DEFAULTS={DEFAULTS['train'][k]!r} TrainConfig={td[k]!r}"


def test_cosine_and_exp_schedules_differ_as_expected() -> None:
    """旧策略 exp 与新策略 cosine 在**同一迭代数**下应给出不同的曲线（确认开关真的生效）。"""
    from gsflow.trainers.gsplat_trainer import lr_at

    P, T, W, r = 1e-3, 1000, 0, 0.1
    cosine_mid = lr_at(T // 2, P, r, W, T)
    gamma = 0.01 ** (1 / T)
    exp_mid = P * (gamma ** (T // 2))
    assert cosine_mid != pytest.approx(exp_mid, rel=1e-6), \
        "cosine 与 exp 在中点应显著不同，否则说明开关没生效"
    assert cosine_mid == pytest.approx((P + P * r) / 2, rel=1e-3)
