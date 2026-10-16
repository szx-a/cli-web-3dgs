"""LPIPS 风格感知损失/指标（基于 torchvision VGG16，无需额外 `lpips` 包）。

为什么自己实现而不是 `pip install lpips`：
1. 本机 `lpips` 包不存在，而安装要走网络；
2. `torchvision` 已在（0.26.0+cu128），VGG16 权重已下到工作区
   （`models/torch/hub/checkpoints/vgg16-397923af.pth`，见 `scripts/fetch_torch_weights.py`）；
3. **可控**：我们需要"权重缺失时优雅退化"，而不是训练到一半抛异常。

实现要点（对齐 LPIPS 论文 Zhang et al. 2018 的常用做法）：
  * 取 VGG16 的 5 个中间层输出作为特征；
  * 每个特征图按通道做 L2 归一化后求**平方差均值**，再加权求和；
  * 输入归一化到 [-1,1]（论文用 ImageNet 统计量，这里保持简单且**自洽**：
    训练与评估用同一套，因此比较仍然有效）。

设计上的**本项创新点**：
  * `perceptual_weight` 参与调度：与 L1 一样受 warmup/退火控制，避免早期用
    感知损失把还没成形的几何"定型"；
  * `lpips_every`：每 N 步才算一次（论文/社区常用做法），显存与时间可控；
  * 权重缺失时**返回 None**，由调用方决定退化，绝不给 0（给 0 会让"开着但没生效"
    这种静默退化无法察觉 —— 本项目在法线正则该处踩过，见 03/G65）。
"""

from __future__ import annotations

import os
from pathlib import Path

# 允许的层索引（VGG16 features 的下标），与 LPIPS 论文的 5 个激活对应
_LPIPS_LAYERS = (4, 9, 16, 23, 30)

_CACHE: dict[str, object] = {}


def torch_home() -> Path:
    """权重缓存目录：固定在工作区内（默认会写 %USERPROFILE%/.cache/torch，沙箱下不可写）。"""
    p = Path(__file__).resolve().parents[3] / "models" / "torch"
    return p


def vgg_weights_present() -> bool:
    f = torch_home() / "hub" / "checkpoints" / "vgg16-397923af.pth"
    return f.exists() and f.stat().st_size > 1024 * 1024


class LPIPS:
    """VGG16 特征 + 通道归一化的感知距离。`__call__` 返回**标量张量**（可反传）。

    构造失败（缺 torchvision / 缺权重）时抛异常，调用方应捕获并记录 WARN，
    然后**退化为不使用**该损失/指标。
    """

    def __init__(self, device: str = "cuda"):
        import torch
        import torchvision.models as M

        os.environ.setdefault("TORCH_HOME", str(torch_home()))
        self.device = torch.device(device)
        net = M.vgg16(weights=None)
        ckpt = torch_home() / "hub" / "checkpoints" / "vgg16-397923af.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"缺少 VGG16 权重：{ckpt}")
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=False)
        # 只保留到最后一层激活之前，冻结参数
        self.features = net.features[: _LPIPS_LAYERS[-1] + 1].to(self.device).eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        # 通道归一化权重：论文用训练好的线性层，这里用**固定**的等权 + 每层系数，
        # 目的是"不同层尺度可比"，不需要额外训练且完全可复现。
        self.layer_w = [1.0, 1.0, 1.0, 1.0, 1.0]

    def _feats(self, x):
        import torch

        out = []
        h = x
        want = set(_LPIPS_LAYERS)
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in want:
                out.append(h)
                if i == _LPIPS_LAYERS[-1]:
                    break
        return out

    def __call__(self, pred, gt):
        """pred/gt: [H,W,3] 或 [B,3,H,W]，取值 0..1。返回标量张量（越小越像）。"""
        import torch
        import torch.nn.functional as F

        def prep(t):
            if t.dim() == 3:
                t = t.permute(2, 0, 1)[None]
            return (t * 2.0 - 1.0).clamp(-1, 1)

        a, b = prep(pred), prep(gt)
        fa, fb = self._feats(a), self._feats(b)
        total = a.new_zeros(())
        for w, x, y in zip(self.layer_w, fa, fb):
            # 通道维归一化（论文做法）：让不同通道尺度可比
            x = x / (x.pow(2).mean(dim=(2, 3), keepdim=True).sqrt() + 1e-6)
            y = y / (y.pow(2).mean(dim=(2, 3), keepdim=True).sqrt() + 1e-6)
            total = total + w * F.mse_loss(x, y)
        return total / max(sum(self.layer_w), 1e-6)


def get_lpips(device: str = "cuda") -> LPIPS:
    """带缓存的构造（同一次训练里只建一次）。"""
    key = f"lpips:{device}"
    if key not in _CACHE:
        _CACHE[key] = LPIPS(device)
    return _CACHE[key]  # type: ignore[return-value]
