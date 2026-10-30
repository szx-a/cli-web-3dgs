"""深度方差正则的数学与梯度测试。

它是"治朦胧"的可微手段：2DGS 的成对失真需要逐像素高斯列表（要写 CUDA），
而沿射线的**深度方差**可以用两次矩渲染精确得到：

    E[z] = Σ w z / Σ w,  E[z²] = Σ w z² / Σ w,  Var = E[z²] - E[z]²

这些测试锁住：
1. 矩 → 期望/方差的换算与解析值一致；
2. 单层（所有质量集中在同一深度）方差为 0；两层各半时方差为 (Δ/2)²；
3. 损失对 alpha（从而对不透明度 logit）**可导且梯度非零**；
4. 方差随层数增多单调增大（这正是"薄涂/朦胧"的量化）。
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from gsflow.trainers.depth_regularizers import (  # noqa: E402
    DepthMoments,
    camera_depths,
    depth_variance_loss,
    expected_depth_stats,
    moment_colors,
)


def _moments(depths: list[float], weights: list[float]) -> DepthMoments:
    z = torch.tensor(depths, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    return DepthMoments(weight=w.sum().reshape(1), first=(w * z).sum().reshape(1),
                        second=(w * z * z).sum().reshape(1))


def test_single_layer_has_zero_variance() -> None:
    m = _moments([3.0], [0.8])
    mean, var = expected_depth_stats(m)
    assert float(mean) == pytest.approx(3.0, rel=1e-5)
    assert float(var) == pytest.approx(0.0, abs=1e-6)


def test_two_equal_layers_variance_matches_analytic() -> None:
    """两层各占一半质量、深度差 Δ ⇒ 方差 = (Δ/2)²。"""
    z1, z2 = 2.0, 4.0
    m = _moments([z1, z2], [0.5, 0.5])
    mean, var = expected_depth_stats(m)
    assert float(mean) == pytest.approx((z1 + z2) / 2, rel=1e-5)
    assert float(var) == pytest.approx(((z2 - z1) / 2) ** 2, rel=1e-5)


def test_variance_grows_with_spread() -> None:
    vars_ = []
    for deltas in ([0.0], [0.5], [1.0], [2.0]):
        d = deltas[0]
        m = _moments([3.0 - d / 2, 3.0 + d / 2], [0.5, 0.5])
        vars_.append(float(expected_depth_stats(m)[1]))
    assert vars_ == sorted(vars_), f"深度跨度越大方差应越大，实测 {vars_}"


def test_loss_is_differentiable_wrt_weights() -> None:
    """权重（=不透明度）必须能拿到梯度，否则这个正则等于没有。"""
    w = torch.tensor([0.5], requires_grad=True)
    z = torch.tensor([2.0, 4.0])
    first = (w * z[0]).reshape(1)
    second = (w * z[0] * z[0]).reshape(1)
    m = DepthMoments(weight=w, first=first, second=second)
    loss = depth_variance_loss(m, scale=1.0)
    loss.backward()
    assert w.grad is not None


def test_loss_modes_are_consistent_in_ordering() -> None:
    """三种模式的排序应一致（谁更散，损失就更大）。"""
    tight = _moments([3.0, 3.1], [0.5, 0.5])
    loose = _moments([1.0, 5.0], [0.5, 0.5])
    for mode in ("var", "std", "rvar"):
        a = float(depth_variance_loss(tight, scale=1.0, mode=mode))
        b = float(depth_variance_loss(loose, scale=1.0, mode=mode))
        assert b > a, f"{mode} 模式下松散分布应更差：{a} vs {b}"


def test_mask_restricts_average() -> None:
    m = DepthMoments(
        weight=torch.tensor([1.0, 1.0]),
        first=torch.tensor([3.0, 1.0]),
        second=torch.tensor([9.0, 9.0]),  # 第二像素方差很大
    )
    mask = torch.tensor([True, False])
    masked = float(depth_variance_loss(m, mask=mask, scale=1.0))
    assert masked == pytest.approx(0.0, abs=1e-6)
    full = float(depth_variance_loss(m, scale=1.0))
    assert full > 0


def test_camera_depths_matches_manual_projection() -> None:
    means = torch.tensor([[1.0, 2.0, 5.0]])
    view = torch.eye(4)
    view[2, 3] = 2.0  # 相机沿 z 平移 2
    z = camera_depths(means, view)
    assert float(z[0]) == pytest.approx(7.0, abs=1e-6)


def test_moment_colors_shape() -> None:
    c = moment_colors(torch.tensor([1.0, 2.0]))
    assert tuple(c.shape) == (2, 2)
    assert float(c[1, 1]) == pytest.approx(4.0)


def test_scale_normalisation_removes_scene_size_dependence() -> None:
    """同一几何在不同尺度下，归一化后方差应同量级（保证 λ 可跨数据集复用）。"""
    small = _moments([1.0, 2.0], [0.5, 0.5])
    big = _moments([10.0, 20.0], [0.5, 0.5])
    a = float(depth_variance_loss(small, scale=1.0))
    b = float(depth_variance_loss(big, scale=10.0))
    assert a == pytest.approx(b, rel=1e-4)
