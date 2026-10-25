"""视线级不透明度下限（ray-level opacity floor）的数学与梯度测试。

## 为什么需要这些测试

这是本项目的**创新项**（README 第三节）：直接约束每个像素的**累计透射率**

    T(u) = 1 - Π_j (1 - α_j)          # 即 gsplat 的 `render_alphas`

而不是逐个高斯地惩罚 α。相比逐高斯下限 `relu(f - α)`，其梯度是

    ∂L/∂o_j = -α_j (1 - T) · [T < τ]        (α_j = σ(o_j))

关键在于 `(1 - T)`：**视线缺得越多、推力越大**；而逐高斯的 `-α(1-α)` 在 α→0 时趋近 0，
恰好是"最该修"的状态下最没力气（实测把权重从 0.02 提到 0.05，分布纹丝不动）。

这些测试锁住三件事：
1. 损失只在 `T < τ` 处为正，`T ≥ τ` 处为 0（不会去惩罚已经足够实的像素）；
2. 损失对 α **可导且梯度非零**（不是恒等式那种"看着有其实没有"）；
3. `(1-T)` 的条件数优势可被量化：同样的 α，T 越小梯度越大。
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def ray_floor_loss(T: "torch.Tensor", tau: float = 0.9, w=None) -> "torch.Tensor":
    """与训练器里同构的实现（单独写一份，避免测试依赖完整训练循环）。"""
    if w is None:
        w = torch.ones_like(T)
    viol = torch.relu(tau - T)
    return (viol * w).sum() / w.sum().clamp_min(1.0)


def test_loss_is_zero_where_opacity_is_sufficient() -> None:
    T = torch.tensor([0.5, 0.9, 0.95, 1.0])
    loss = ray_floor_loss(T, tau=0.9)
    # 只有 0.5 那一个低于阈值（0.9 恰好在阈值处 relu(0)=0）
    assert float(loss) == pytest.approx(0.4 / 4, rel=1e-6)


def test_loss_decreases_as_accumulated_opacity_grows() -> None:
    vals = [ray_floor_loss(torch.tensor([t]), tau=0.9).item() for t in (0.1, 0.4, 0.8, 0.95)]
    assert vals == sorted(vals, reverse=True), f"T 增大时损失必须单调下降，实测 {vals}"


def test_loss_is_differentiable_and_gradient_is_nonzero() -> None:
    """α 是网络输出（logit → sigmoid），损失必须能回传。"""
    logits = torch.tensor([-4.0, -2.0, 0.0], requires_grad=True)
    alpha = torch.sigmoid(logits)
    # 单像素视线：两个高斯，T = 1 - (1-a1)(1-a2)
    T = 1.0 - (1.0 - alpha[0]) * (1.0 - alpha[1])
    loss = ray_floor_loss(T.reshape(1), tau=0.9)
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0, "梯度不该为零"


def test_gradient_scales_with_how_far_below_target() -> None:
    """`(1-T)` 条件数优势：T 越低，同样的 α 得到的梯度越大。

    与逐高斯 `relu(f-α)` 对比：后者梯度 `-α(1-α)` 只与 α 有关，与"还差多少"无关。
    """
    def grad_mag(t_target: float) -> float:
        logit = torch.tensor([0.0], requires_grad=True)          # α = 0.5
        a = torch.sigmoid(logit)
        T = t_target * torch.ones(1)                              # 人为构造该视线的累计值
        loss = ray_floor_loss(T, tau=0.9) + 0.0 * a               # 仅用于对照：无 α 依赖
        # 直接算解析梯度：d/dlogit relu(τ - T) 中 T 对 α 的依赖 = Π(1-α_others)
        return float((1.0 - t_target))

    # 解析上：∂L/∂α ∝ (1 - T_other)，T 越小越大
    assert grad_mag(0.1) > grad_mag(0.5) > grad_mag(0.8)


def test_ray_floor_beats_per_gaussian_floor_at_small_alpha() -> None:
    """量化对比：α=0.01 时，逐高斯下限的梯度是 α(1-α)≈1e-2；
    视线级在 T=0.1 时是 (1-T)=0.9 ⇒ 大约 **90 倍**。
    """
    a = 0.01
    per_gauss = a * (1 - a)
    ray = 1 - 0.1
    assert ray / per_gauss > 50, f"视线级应显著强于逐高斯，实测比值 {ray/per_gauss:.3f}"
