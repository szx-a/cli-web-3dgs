"""梯度（细节）损失的数学与边界行为测试。

`L_grad = λ · mean_M |∇C − ∇Ĩ|₁`，用来治"形状对、表面发糊"。

这些测试锁住：
1. 完全相同图像 ⇒ 损失为 0；
2. **只把纹理抹平**（低频化）会让损失变大——这正是它相对 L1/SSIM/LPIPS 的价值；
3. 只在掩码内统计（掩码外不参与）；
4. 可微（能反传到渲染图）；
5. 与"纯亮度偏差"区分：整体加常数不影响梯度损失（梯度为 0）。
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gsflow.trainers.gsplat_trainer import _grad_loss  # noqa: E402


def _img(h: int = 32, w: int = 32, checker: bool = False) -> "torch.Tensor":
    if checker:
        x = torch.zeros(h, w, 3)
        x[::2, ::2] = 1.0
        x[1::2, 1::2] = 1.0
        return x
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = (0.3 + 0.4 * (xx / w) + 0.2 * (yy / h)).unsqueeze(-1)
    return base.repeat(1, 1, 3)


def test_identical_images_give_zero() -> None:
    a = _img()
    assert float(_grad_loss(a, a)) == pytest.approx(0.0, abs=1e-7)


def test_flat_texture_costs_more_than_sharp_texture() -> None:
    gt = _img(checker=True)
    flat = torch.full_like(gt, float(gt.mean()))
    sharp = gt + 0.05 * torch.randn_like(gt)
    l_flat = float(_grad_loss(flat, gt))
    l_sharp = float(_grad_loss(sharp, gt))
    assert l_flat > l_sharp, f"抹平图像应更差：flat={l_flat:.4f} sharp={l_sharp:.4f}"


def test_constant_brightness_offset_is_free() -> None:
    """整体加常数不改变梯度 —— 与 L1 不同，说明它约束的是结构而不是亮度。"""
    gt = _img()
    shifted = gt + 0.2
    assert float(_grad_loss(shifted, gt)) == pytest.approx(0.0, abs=1e-6)


def test_mask_restricts_statistic() -> None:
    gt = _img()
    render = gt.clone()
    render[0, 0] = 1.0 - render[0, 0]  # 只在左上角制造差异
    mask = torch.zeros(gt.shape[:2], dtype=torch.bool)
    mask[5:25, 5:25] = True  # 掩码避开 (0,0)
    assert float(_grad_loss(render, gt, mask)) == pytest.approx(0.0, abs=1e-6)
    assert float(_grad_loss(render, gt)) > 0


def test_is_differentiable() -> None:
    """注意：必须用**不同**的图求导——完全相同时损失恒为 0、梯度也是 0。"""
    gt = _img()
    r = (_img() + 0.05 * torch.randn_like(gt)).requires_grad_(True)
    loss = _grad_loss(r, gt)
    loss.backward()
    assert r.grad is not None and bool((r.grad != 0).any())


def test_scale_is_comparable_to_l1() -> None:
    """量级检查：随机噪声下它应与 L1 同阶（避免权重需要跨数量级调整）。"""
    gt = _img()
    noisy = (gt + 0.05 * torch.randn_like(gt)).clamp(0, 1)
    g = float(_grad_loss(noisy, gt))
    l1 = float((noisy - gt).abs().mean())
    assert 0.1 * l1 < g < 10 * l1, f"梯度损失（{g:.4f}）与 L1（{l1:.4f}）量级差太多"
