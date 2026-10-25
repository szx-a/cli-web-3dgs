"""深度语义自洽性：训练侧必须用 **ED（期望深度）**，不能用 `Σwᵢzᵢ`。

## 背景（2026-10-25 修复的真实缺陷）

`gsplat.rasterization` 的两种深度：

| 模式 | 返回 | 是不是"深度" |
|---|---|---|
| `render_mode="D"` | `Σ wᵢ zᵢ`（**未除以 alpha**）| 不是：它被不透明度同比压低 |
| `render_mode="RGB+ED"` | `Σ wᵢ zᵢ / Σ wᵢ` | 是：与导出阶段 `mesh_export.py` 同一个量 |

训练里算"法线一致性 / 深度失真"原来用的是 `D`，而导出反投影用的是 `ED`
⇒ 两段优化/使用**不同的量**（相差 1/alpha 倍），属于"各段都能跑、但语义不一致"。

这些测试把差异钉死：
1. 常数深度（正对相机的平面）→ 法线必须 ≈ [0,0,-1]；
2. 若错误地用 `Σwz`（alpha 沿 x 渐变），同一个平面会算出**倾斜**法线（>5°），
   这就是旧实现的症状；
3. 静态契约：训练代码默认走 `RGB+ED`，默认 `depth_mode="expected"`。
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from gsflow.trainers.gsplat_trainer import depth_to_normal

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src" / "gsflow" / "trainers" / "gsplat_trainer.py"


def _K(h: int, w: int) -> torch.Tensor:
    f = 1.2 * max(h, w)
    return torch.tensor([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def _angle_deg(n: torch.Tensor, ref=(0.0, 0.0, -1.0)) -> float:
    r = torch.tensor(ref, dtype=n.dtype)
    c = float((n * r).sum().clamp(-1, 1))
    return math.degrees(math.acos(c))


def test_constant_expected_depth_plane_has_axis_aligned_normal() -> None:
    h, w = 48, 64
    depth = torch.full((h, w), 2.5)
    n, valid = depth_to_normal(depth, _K(h, w))
    # 只取内部（中心差分在边界会退化）
    inner = n[8:-8, 8:-8].reshape(-1, 3)
    angs = [_angle_deg(v) for v in inner]
    assert max(angs) < 1.0, f"常数 ED 深度应得到正对法线，实测最大偏差 {max(angs):.2f}°"
    assert bool(valid[8:-8, 8:-8].all())


def test_accumulated_depth_with_alpha_ramp_tilts_the_normal() -> None:
    """错误用法（`D = Σwz`）：alpha 沿 x 从 0.2 到 1.0，深度恒为 2.5。

    `Σwz = alpha * 2.5` 会随 x 变化 ⇒ 算出的法线**倾斜**。这就是旧实现的症状，
    也是导出与训练语义不一致的证据。
    """
    h, w = 48, 64
    alpha = torch.linspace(0.2, 1.0, w).view(1, -1).expand(h, w)
    depth_accum = alpha * 2.5
    n_bad, _ = depth_to_normal(depth_accum, _K(h, w))
    inner = n_bad[8:-8, 8:-8].reshape(-1, 3)
    angs = [_angle_deg(v) for v in inner]
    assert max(angs) > 5.0, (
        f"`Σwz` 在 alpha 渐变时应产生明显倾斜法线，实测最大偏差仅 {max(angs):.2f}°"
    )


def test_expected_depth_equals_accum_over_alpha() -> None:
    """ED 恒等式：`ED = D / Σw`（浮点容差内）。"""
    alpha = torch.rand(32, 32) * 0.9 + 0.1
    z = torch.rand(32, 32) * 3.0 + 1.0
    d_accum = alpha * z
    ed = d_accum / alpha.clamp_min(1e-6)
    assert torch.allclose(ed, z, atol=1e-5)


def test_trainer_defaults_to_expected_depth() -> None:
    t = TRAINER.read_text(encoding="utf-8")
    assert 'depth_mode = str(getattr(cfg, "depth_mode", "expected")' in t, \
        "训练侧没有按 depth_mode 选择 ED/accum"
    assert 'render_mode="RGB+ED"' in t, "训练侧缺少 ED 渲染路径"
    assert 'depth_mode: str = "expected"' in t, "TrainConfig 默认应为 expected"
