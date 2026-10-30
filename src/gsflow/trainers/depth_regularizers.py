"""深度集中度正则（"朦胧"的正解方向）。

## 为什么要它

实测反复出现"颜色对得上、换个视角就透/薄"的模型：掩码内 PSNR 26~30 dB，
但不透明度中位只有 0.012~0.095。根因不是累计不透明度不足（实测 $T$ 中位 0.99），
而是**同一条视线上颜色的来源分布在很多不同深度**——表面位置不确定，视觉上就是"糊/薄"。

2DGS 用成对失真直接惩罚这种分布：

    L_d = Σ_u Σ_{i,j} w_i w_j |z_i - z_j|

它需要光栅化核里逐像素的高斯列表（要写 CUDA），gsplat 不暴露。但它的**L2 对应量**——
沿射线的深度方差——可以用**两次矩渲染**精确算出来：

    Var_u[z] = E_u[z²] - E_u[z]²
    E_u[z]   = Σ_j w_j z_j / Σ_j w_j
    E_u[z²]  = Σ_j w_j z_j² / Σ_j w_j

技巧：把逐高斯相机深度 $z_j$ 与 $z_j^2$ 塞进光栅化的颜色通道（gsplat 的 `colors` 支持任意通道 D），
于是 $\sum_j w_j z_j$ 与 $\sum_j w_j z_j^2$ 直接以图像形式返回，$\sum_j w_j$ 就是返回的 alpha。
两者相除即得期望，方差就是本模块的损失。

## 与 2DGS 原式的差别（如实记录）

* 2DGS 用的是 $L_1$ 型成对差 $|z_i-z_j|$，本模块用二阶矩（$L_2$ 型）；
* 两者都对"多层薄涂"施加惩罚、都把同一像素的深度挤到一起，但**不相等**；
* 二阶矩版本**不需要自定义 CUDA 核**，可直接在 gsplat 上反向传播。

对"朦胧"的效果需要用同一把尺子实测（见 README 与 docs）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DepthMoments:
    """一次矩渲染的结果（都是逐像素张量）。"""

    weight: torch.Tensor  # Σ_j w_j      即累计 alpha，[H, W]
    first: torch.Tensor   # Σ_j w_j z_j
    second: torch.Tensor  # Σ_j w_j z_j²


def camera_depths(means: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    """世界坐标高斯 → 当前相机的深度 z（沿光轴）。

    means   : [N, 3]
    viewmat : [4, 4]  世界→相机
    返回     : [N]
    """
    r = viewmat[:3, :3]
    t = viewmat[:3, 3]
    return (means @ r.transpose(0, 1) + t)[:, 2]


def moment_colors(depths: torch.Tensor) -> torch.Tensor:
    """构造“矩通道”：[z, z²]，交给光栅化当颜色用。"""
    return torch.stack([depths, depths * depths], dim=-1)


def expected_depth_stats(m: DepthMoments, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """由矩得到 (期望深度, 深度方差)，对权重过小的像素返回 0，避免除零放大噪声。"""
    w = m.weight.clamp_min(eps)
    mean = m.first / w
    mean_sq = m.second / w
    var = (mean_sq - mean * mean).clamp_min(0.0)
    valid = (m.weight > 1e-3).to(var.dtype)
    return mean * valid, var * valid


def depth_variance_loss(
    m: DepthMoments,
    mask: torch.Tensor | None = None,
    scale: float = 1.0,
    mode: str = "var",
    eps: float = 1e-6,
) -> torch.Tensor:
    """深度集中度损失。

    mode="var"  : 归一化方差  Var[z] / scale²      （推荐，量纲无关）
    mode="std"  : 标准差      sqrt(Var[z]) / scale  （更线性，梯度更稳）
    mode="rvar" : 稳健版，对超过 3 倍中位数的像素截断（抑制离群射线）

    mask 为 None 时对全图取均值；否则只在 mask 内取均值。
    """
    _, var = expected_depth_stats(m, eps=eps)
    s = max(float(scale), 1e-8)
    if mode == "var":
        val = var / (s * s)
    elif mode == "std":
        val = torch.sqrt(var) / s
    elif mode == "rvar":
        v = var / (s * s)
        if bool((v > 0).any()):
            med = torch.median(v[v > 0])
            v = torch.clamp(v, max=3.0 * float(med))
        val = v
    else:
        raise ValueError(f"未知 depth_var mode: {mode!r}")

    if mask is not None and bool(mask.any()):
        w = mask.to(val.dtype)
        return (val * w).sum() / w.sum().clamp_min(1.0)
    return val.mean()
