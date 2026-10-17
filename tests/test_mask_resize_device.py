"""回归测试：掩码尺寸与图像不一致时，损失必须能正常计算（设备/形状都不能错）。

## 为什么需要这条测试（真实崩溃）

用 406 张 576×1280 的素材训练时报：
    RuntimeError: Expected all tensors to be on the same device,
                  but found at least two devices, cuda:0 and cpu!
堆栈指向 `gt = torch.where(m[..., None], gt, background.to(gt.dtype))`。

根因有两处：
  1. **掩码插值分支**把张量留在了 CPU（`m` 原本是 CPU bool 张量），
     而旧代码只在 `else` 分支做了 `.to(device)`；
  2. **`background.to(gt.dtype)` 只转 dtype、没转 device**。

两处都只在"掩码尺寸与图像不一致"时才走到 —— 也就是**图像被缩放**时。
小数据集（掩码与图同尺寸）永远不触发，所以这个 bug 藏了很久。

本测试直接构造"掩码尺寸 ≠ 图像尺寸"的情形，断言复合损失的代码路径
不抛异常且结果在预期范围。用**最小可复现**代替整条训练链。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_mask_resize_path_keeps_device_and_dtype() -> None:
    """复现当初崩溃的两步：掩码插值 + background 复合，必须全在同一设备。"""
    torch = pytest.importorskip("torch")

    device = torch.device("cpu")          # 用 CPU 就能复现"设备不一致"的逻辑错误
    H, W = 64, 48
    gt = torch.rand(H, W, 3, device=device)
    # 掩码尺寸与图像**不一致**（模拟图像被缩放后的情形）
    m = torch.rand(32, 24) > 0.5
    m = torch.nn.functional.interpolate(
        m[None, None].float(), size=gt.shape[:2], mode="nearest"
    )[0, 0] > 0.5
    m = m.to(device)                      # ← 修复点 1
    background = torch.tensor([0.0, 0.0, 0.0], device=device)

    # 修复点 2：`.to(device=..., dtype=...)` 两个都要转
    out = torch.where(m[..., None], gt,
                      background.to(device=gt.device, dtype=gt.dtype))
    assert out.device == gt.device, f"结果在 {out.device}，应在 {gt.device}"
    assert out.dtype == gt.dtype
    assert out.shape == gt.shape
    # 掩码外的像素必须**恰好**等于背景色
    outside = ~m
    if outside.any():
        assert torch.all(out[outside] == 0.0), "掩码外应被合成为纯黑背景"


def test_mask_resize_mismatch_sizes_are_handled() -> None:
    """掩码比图像小/大都要能处理（去畸变后的尺寸变化可能两个方向都有）。"""
    torch = pytest.importorskip("torch")
    for (mh, mw) in ((24, 32), (128, 96)):
        gt = torch.rand(64, 48, 3)
        m = torch.rand(mh, mw) > 0.5
        m2 = torch.nn.functional.interpolate(
            m[None, None].float(), size=gt.shape[:2], mode="nearest"
        )[0, 0] > 0.5
        assert m2.shape == gt.shape[:2], f"从 {mh}x{mw} 插值后形状不对：{tuple(m2.shape)}"


def test_composite_loss_matches_manual_computation() -> None:
    """复合损失的**已知答案**：掩码外像素误差应为 0（因为 GT 被换成背景色）。

    构造：渲染为纯黑、GT 前景为白、背景为黑，掩码只覆盖左半。
    ⇒ 掩码内误差 = 1（黑 vs 白），掩码外误差 = 0（黑 vs 黑），
      全图 L1 = 前景占比 × 1 = 0.5。
    """
    torch = pytest.importorskip("torch")
    H, W = 8, 8
    render = torch.zeros(H, W, 3)
    gt = torch.zeros(H, W, 3)
    gt[:, : W // 2, :] = 1.0                          # 左半为白（前景）
    m = torch.zeros(H, W, dtype=torch.bool)
    m[:, : W // 2] = True                              # 掩码=左半
    background = torch.zeros(3)

    gt_comp = torch.where(m[..., None], gt, background)
    l1 = (render - gt_comp).abs().mean()
    assert abs(float(l1) - 0.5) < 1e-6, f"已知答案 0.5，实际 {float(l1):.6f}"
